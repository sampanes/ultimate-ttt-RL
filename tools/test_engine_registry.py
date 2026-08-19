"""Tests for the frozen engine registry -- does the anti-drift guard bite?

The registry's whole value is a claim about what CANNOT happen quietly. That
claim is worthless untested: a fingerprint check that never fires looks exactly
like a fingerprint check that cannot fire. Each test below breaks something the
guard is supposed to notice and asserts that it does.

    python -m tools.test_engine_registry
"""
from __future__ import annotations

import unittest
from unittest import mock

from tools import engine_registry as reg
from tools.arena_1s import TimedPlayer

DEV = "cpu"

# CUDA-graph engines cannot be built on the CPU: capture fails, and TimedPlayer
# refuses to hand back an arm that quietly fell back to the eager path. Their
# CONFIGURATION is still checkable everywhere -- that is what a fingerprint is
# -- so device-independent tests use the resolved spec, and the few that need a
# built player skip when there is no GPU.
try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except Exception:                                    # pragma: no cover
    HAVE_CUDA = False


def needs_cuda(name):
    return reg.ENGINES.get(name, {}).get("graph") == "1"


def device_for(name):
    return "cuda" if needs_cuda(name) else DEV


def buildable(name):
    return HAVE_CUDA or not needs_cuda(name)


def build(name):
    return TimedPlayer(f"engine:{name}", device_for(name))


def build_spec(spec):
    return TimedPlayer(spec, DEV)


class TestFrozenSet(unittest.TestCase):

    def test_every_engine_verifies(self):
        for name in reg.ENGINES:
            with self.subTest(engine=name):
                p = build(name)
                self.assertTrue(p.provenance["verified"])
                self.assertEqual(p.provenance["fingerprint"],
                                 reg.FINGERPRINTS[name])

    def test_every_engine_has_a_frozen_fingerprint(self):
        self.assertEqual(set(reg.ENGINES), set(reg.FINGERPRINTS))

    def test_specs_pin_every_option_rather_than_inheriting_defaults(self):
        # The point of the registry. If an option is absent from a frozen spec
        # it comes from a code default, and a later edit moves the engine.
        pinned = {"ckpt", "arch", "ms", "wave", "cpuct", "reuse", "solve",
                  "maxsims", "reserve", "bexp", "name", "graph", "select",
                  "defer", "pfilter"}
        for name, spec in reg.ENGINES.items():
            with self.subTest(engine=name):
                want = reg.RAW_PINNED if reg.is_raw(name) else pinned
                self.assertEqual(set(spec), want)


class TestGraphRefreeze(unittest.TestCase):
    """Every fingerprint has now moved four times, each time because
    `resolved_config` gained keys: `graph_wave` on 2026-08-09, `native_select`
    on 2026-08-12, `defer_release`/`retire_watermark` on 2026-08-15, and
    `probe_filter` on 2026-08-18. Nothing was re-measured on any of them, so
    the claim that no ENGINE moved has to stay checkable at EVERY step --
    strip the keys added since a generation and that generation's hashes must
    come back.
    """

    # The keys `resolved_config` has gained since each frozen generation, and
    # the hashes that generation's results were measured under. Cumulative,
    # oldest last.
    _FILTER = {"probe_filter"}
    _DEFER = _FILTER | {"defer_release", "retire_watermark"}
    GENERATIONS = (
        ("PRE_FILTER_FINGERPRINTS", set(_FILTER)),
        ("PRE_DEFER_FINGERPRINTS", set(_DEFER)),
        ("PRE_SELECT_FINGERPRINTS", _DEFER | {"native_select"}),
        ("PRE_GRAPH_FINGERPRINTS", _DEFER | {"native_select", "graph_wave"}),
    )

    def test_stripping_the_added_keys_reproduces_every_older_fingerprint(self):
        for attr, added in self.GENERATIONS:
            table = getattr(reg, attr)
            # Engines that existed at that freeze. Newer ones have no identity
            # in it, and they are excluded by construction rather than by an
            # exception list that could quietly grow.
            pre = set(table)
            self.assertTrue(pre <= set(reg.ENGINES))
            for name in pre:
                with self.subTest(generation=attr, engine=name):
                    cfg = reg.resolved_config(build(name))
                    stripped = {k: v for k, v in cfg.items()
                                if k not in added}
                    self.assertEqual(reg.fingerprint(stripped), table[name])

    def test_every_newer_engine_is_a_declared_candidate(self):
        """A new engine with no older identity has to be one of the candidates
        the new keys exist for -- otherwise the tables above stopped covering
        the registry and nobody noticed."""
        newest = set(reg.PRE_FILTER_FINGERPRINTS)
        for name in set(reg.ENGINES) - newest:
            spec = reg.ENGINES[name]
            self.assertTrue(spec.get("graph") == "1"
                            or spec.get("select") == "1"
                            or spec.get("defer") == "1"
                            or spec.get("pfilter") == "1",
                            "%s is new since the last re-freeze but enables "
                            "no candidate flag -- it needs an older "
                            "fingerprint or an explicit reason" % name)

    def test_only_the_deployed_engine_and_its_lineage_enable_the_graph(self):
        """It IS promoted now, as part of the deployed stack. What still has to
        hold is that no ruler and no superseded baseline enables it: the ladder
        and the historical comparator have to keep playing as measured. The set
        grows each time a candidate is BUILT ON another one -- `pocket_sel` on
        `pocket_graph`, `pocket_defer` on `pocket_sel` -- which is inheriting
        the flag, not granting it."""
        on = {n for n in reg.ENGINES if reg.ENGINES[n].get("graph") == "1"}
        self.assertEqual(on, {"pocket_graph", "pocket_sel", "pocket_defer",
                              "pocket_filter"})
        self.assertIn(reg.DEPLOYED, on)
        self.assertFalse(on & reg.ANCHOR_ROLES)
        self.assertFalse(on & set(reg.SUPERSEDED))
        self.assertNotIn("final", on)

    def test_only_the_deployed_engine_and_its_successor_defer_retirement(self):
        on = {n for n in reg.ENGINES if reg.ENGINES[n].get("defer") == "1"}
        self.assertEqual(on, {reg.DEPLOYED, "pocket_filter"})
        self.assertFalse(on & reg.ANCHOR_ROLES)
        self.assertFalse(on & set(reg.SUPERSEDED))
        self.assertNotIn("final", on)
        self.assertNotIn("pocket_sel", on)

    def test_only_the_filter_candidate_filters_probes(self):
        """The newest flag, and the only engine allowed to set it is the one
        candidate that exists for it. The deployed baseline must NOT: an
        optimisation that turned itself on inside the B side of its own A/B
        would make the comparison vacuous."""
        on = {n for n in reg.ENGINES if reg.ENGINES[n].get("pfilter") == "1"}
        self.assertEqual(on, {"pocket_filter"})
        self.assertNotIn(reg.DEPLOYED, on)
        self.assertFalse(on & reg.ANCHOR_ROLES)
        self.assertFalse(on & set(reg.SUPERSEDED))

    def test_promotion_did_not_move_the_shared_defaults(self):
        """The promoted engine sets graph/select/defer to "1", but `_FINAL` --
        the base `original`, `final`, the four anchors and the superseded
        baselines are all built from -- must still pin them OFF. Promotion is
        recorded by DEPLOYED; moving the defaults would silently re-play the
        whole ladder."""
        for flag in ("graph", "select", "defer"):
            with self.subTest(flag=flag):
                self.assertEqual(reg._FINAL[flag], "0")
                self.assertEqual(reg.ENGINES[reg.DEPLOYED][flag], "1")
        # Not in the loop above: `pfilter` is a CANDIDATE flag, so the base and
        # the deployed engine must both pin it off. Once it promotes the loop
        # gains it and this line goes.
        self.assertEqual(reg._FINAL["pfilter"], "0")
        self.assertEqual(reg.ENGINES[reg.DEPLOYED]["pfilter"], "0")

    def test_the_filter_candidate_differs_in_exactly_one_key(self):
        """The first candidate in the series that does not move the reserve.
        Every earlier one paid for a bigger tree to walk outside the search's
        deadline; #46 removed that walk, so more search should cost nothing
        here -- and if tools/regress_engine disagrees, THIS test is what fails
        and forces the number to be written down rather than absorbed."""
        a, b = reg.ENGINES[reg.DEPLOYED], reg.ENGINES["pocket_filter"]
        diff = {k for k in a if a[k] != b[k]} - {"name"}
        self.assertEqual(diff, {"pfilter"})
        self.assertEqual(b["reserve"], a["reserve"])

    def test_the_defer_candidate_differs_in_the_flag_and_the_reserve(self):
        """Two keys again, and this time the reserve goes DOWN. Every previous
        candidate raised it to pay for a bigger tree to walk; this one removes
        the walk, so #46a's measurement (caller-side overhead p99 0.06 ms with
        release taken out, worst chunk p99 3.24) is what licenses 20."""
        a, b = reg.ENGINES["pocket_sel"], reg.ENGINES["pocket_defer"]
        diff = {k for k in a if a[k] != b[k]} - {"name"}
        self.assertEqual(diff, {"defer", "reserve"})
        self.assertEqual(b["reserve"], "20")
        self.assertEqual(a["reserve"], "95")

    def test_the_candidate_differs_from_the_incumbent_in_two_declared_keys(self):
        a, b = reg.ENGINES["pocket_r35"], reg.ENGINES["pocket_graph"]
        diff = {k for k in a if a[k] != b[k]} - {"name"}
        self.assertEqual(diff, {"graph", "reserve"},
                         "the candidate must differ from pocket_r35 in the "
                         "graph flag and the reserve it forces, and nothing "
                         "else")

    def test_graph_is_now_a_declarable_override(self):
        spec, diff = reg.derived_spec("pocket_r35", {"graph": "1"})
        self.assertEqual(diff, {"graph"})
        self.assertIn("graph=1", spec)


class TestRawNetworkArms(unittest.TestCase):
    """sims=0 must really be the network, not a degenerate one-sim search."""

    RAW = ["gen22_raw", "pocket_raw", "midsize_raw"]

    def test_raw_players_have_no_search(self):
        for n in self.RAW:
            with self.subTest(engine=n):
                p = build(n)
                self.assertTrue(p.raw)
                self.assertIsNone(p.searcher)
                self.assertIsNone(p.budget_ms)

    def test_raw_move_is_the_masked_policy_argmax(self):
        # The reason this arm exists as its own code path. A 1-simulation
        # search is NOT the raw network: at the root every child has N=0, so
        # sqrt(N_parent)=0 kills the PUCT exploration term for all of them, the
        # scores tie, and the pick falls out of dict order. Measured agreement
        # with the policy argmax was 0.197.
        import numpy as np
        import torch
        from agents.agent_base import board_to_tensor_from_gamestate
        from engine.game import GameState
        from engine.rules import rule_utl_valid_moves

        p = build("pocket_raw")
        state = GameState()
        for _ in range(9):
            if state.is_over():
                break
            valid = rule_utl_valid_moves(state.board, state.last_move,
                                         state.mini_winners)
            x = board_to_tensor_from_gamestate(state, v_computed=valid)
            with torch.no_grad():
                logits, _ = p.model.forward_both(x.unsqueeze(0))
            lg = logits.reshape(-1).clone()
            masked = torch.full((81,), float("-inf"))
            masked[valid] = 0.0
            self.assertEqual(p.move(state, 0), int(torch.argmax(lg + masked)))
            state.make_move(valid[len(valid) // 2])

    def test_raw_policy_is_a_distribution_over_legal_moves_only(self):
        import numpy as np
        from engine.game import GameState
        from engine.rules import rule_utl_valid_moves

        p = build("gen22_raw")
        state = GameState()
        state.make_move(40)
        valid = set(rule_utl_valid_moves(state.board, state.last_move,
                                         state.mini_winners))
        pi = p._raw_policy(state)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
        self.assertEqual({i for i in range(81) if pi[i] > 0}, valid)

    def test_raw_arms_cover_every_searched_model(self):
        # A model-size comparison needs the network arm for each size, or the
        # "was it the net or the search" question cannot be answered.
        searched = {reg.ENGINES[n]["ckpt"]
                    for n in ("final", "pocket", "midsize")}
        raw = {reg.ENGINES[n]["ckpt"] for n in self.RAW}
        self.assertEqual(searched, raw)


class TestGuardBites(unittest.TestCase):

    def test_changed_default_trips_the_fingerprint(self):
        # Simulate someone changing a DEFAULT that no spec string mentions.
        # This is the case a spec-only registry would miss entirely.
        p = build("final")
        with mock.patch.object(type(p.mcts), "_VL", 2.0):
            with self.assertRaises(SystemExit) as cm:
                reg.verify("final", p)
        self.assertIn("DRIFTED", str(cm.exception))

    def test_changed_search_parameter_trips_the_fingerprint(self):
        p = build("final")
        p.mcts.deadline_margin = 1.5
        with self.assertRaises(SystemExit) as cm:
            reg.verify("final", p)
        self.assertIn("DRIFTED", str(cm.exception))

    def test_changed_checkpoint_bytes_trip_the_hash_check(self):
        p = build("final")
        bogus = dict(reg.CHECKPOINTS, **{p.ckpt: "0" * 64})
        with mock.patch.object(reg, "CHECKPOINTS", bogus):
            with self.assertRaises(SystemExit) as cm:
                reg.verify("final", p)
        self.assertIn("has changed", str(cm.exception))

    def test_source_drift_is_fatal_for_an_anchor(self):
        # An anchor must never move: it is the ruler.
        p = build("anchor_C")
        bogus = dict(reg.ENGINE_SOURCES, **{"agents/mcts.py": "0" * 64})
        with mock.patch.object(reg, "ENGINE_SOURCES", bogus):
            with self.assertRaises(SystemExit) as cm:
                reg.verify("anchor_C", p)
        self.assertIn("source drift", str(cm.exception))

    def test_source_drift_only_warns_for_a_candidate(self):
        # A candidate is EXPECTED to change the search. Failing here would make
        # the registry unusable for the thing it exists to support.
        p = build("final")
        bogus = dict(reg.ENGINE_SOURCES, **{"agents/mcts.py": "0" * 64})
        with mock.patch.object(reg, "ENGINE_SOURCES", bogus):
            prov = reg.verify("final", p)
        self.assertEqual(prov["source_drift"], ["agents/mcts.py"])
        self.assertTrue(prov["verified"])

    def test_anchor_drift_override_is_available_but_explicit(self):
        bogus = dict(reg.ENGINE_SOURCES, **{"agents/mcts.py": "0" * 64})
        with mock.patch.object(reg, "ENGINE_SOURCES", bogus):
            p = TimedPlayer("engine:anchor_C", DEV, allow_anchor_drift=True)
        self.assertEqual(p.provenance["source_drift"], ["agents/mcts.py"])

    def test_unknown_engine_is_refused(self):
        with self.assertRaises(SystemExit):
            reg.spec_of("no_such_engine")


class TestTheTwoHeadlineConfigurations(unittest.TestCase):

    def test_original_and_final_differ_in_exactly_the_two_shipped_changes(self):
        # The 0.7229 headline is a claim about tree reuse plus batched
        # expansion and NOTHING else. If a third difference ever appears, the
        # attribution is wrong.
        a, b = reg.ENGINES["original"], reg.ENGINES["final"]
        diff = {k for k in a if a[k] != b[k]} - {"name"}
        self.assertEqual(diff, {"reuse", "bexp"})

    def test_original_is_the_rebuild_every_move_per_leaf_engine(self):
        p = build("original")
        self.assertFalse(p.reuse)
        self.assertFalse(p.mcts.batched_expand)
        self.assertEqual(p.budget_ms, 1000.0)

    def test_final_is_reuse_plus_batched_expansion(self):
        p = build("final")
        self.assertTrue(p.reuse)
        self.assertTrue(p.mcts.batched_expand)
        self.assertEqual(p.budget_ms, 1000.0)


class TestLadder(unittest.TestCase):

    RUNGS = ["anchor_A", "anchor_B", "final", "anchor_C", "anchor_D"]

    def test_budgets_are_strictly_increasing(self):
        got = [float(reg.ENGINES[n]["ms"]) for n in self.RUNGS]
        self.assertEqual(got, sorted(got))
        self.assertEqual(got, [250.0, 500.0, 1000.0, 2000.0, 4000.0])

    def test_budget_is_the_only_difference_between_rungs(self):
        # The ladder's meaning depends on this: if rungs differ in anything
        # else, an ordering result is not about time at all.
        varying = {"ms", "reserve", "name"}
        base = {k: v for k, v in reg.ENGINES["final"].items()
                if k not in varying}
        for n in self.RUNGS:
            with self.subTest(rung=n):
                other = {k: v for k, v in reg.ENGINES[n].items()
                         if k not in varying}
                self.assertEqual(other, base)

    def test_reserve_tracks_the_budget(self):
        for n in self.RUNGS:
            with self.subTest(rung=n):
                ms = float(reg.ENGINES[n]["ms"])
                self.assertEqual(float(reg.ENGINES[n]["reserve"]),
                                 max(5.0, 0.02 * ms))

    def test_only_rungs_above_the_deployment_budget_are_latency_exempt(self):
        for n in self.RUNGS:
            with self.subTest(rung=n):
                ms = float(reg.ENGINES[n]["ms"])
                exempt = build(n).latency_exempt
                self.assertEqual(exempt, ms > reg.REQUIREMENT["budget_ms"])

    def test_exempt_rung_reports_latency_but_is_not_judged(self):
        from tools.arena_1s import latency_report
        p = build("anchor_D")
        p.records = [(3000.0, 2990.0, 100, 90, 80, 10, 5, 0, 40, 20, 5, 4.0)]
        rep = latency_report(p)
        self.assertTrue(rep["requirement"]["exempt"])
        self.assertNotIn("PASS", rep["requirement"])
        self.assertGreater(rep["latency_ms"]["p99"], 0)


class TestLatencyCorrectedPocket(unittest.TestCase):
    """`pocket_r35` -- the strength winner with a reserve that covers the work
    the caller waits for but the search does not time."""

    def test_the_measured_arm_is_left_exactly_as_measured(self):
        # `pocket` scored 0.5854 against `final` AND missed p99 by 2.3 ms.
        # Editing it to fix that would detach a published number from the
        # configuration that produced it.
        self.assertEqual(reg.ENGINES["pocket"]["reserve"], "20")
        # The literal is the fingerprint the 0.5854 was measured under. It is
        # no longer the CURRENT one -- 2026-08-09 added `graph_wave` to
        # `resolved_config`, 2026-08-12 added `native_select`, 2026-08-15 added
        # `defer_release` and `retire_watermark`, and 2026-08-18 added
        # `probe_filter`, moving all of them four times -- so the guard checks
        # the configuration minus EVERY key added since. Weakening it to
        # `== FINGERPRINTS["pocket"]` would make it tautological.
        self.assertEqual(reg.PRE_GRAPH_FINGERPRINTS["pocket"],
                         "036f17c9aa644aad")
        added = ("graph_wave", "native_select", "defer_release",
                 "retire_watermark", "probe_filter")
        cfg = reg.resolved_config(build("pocket"))
        self.assertFalse(cfg["graph_wave"])
        self.assertFalse(cfg["native_select"])
        self.assertFalse(cfg["defer_release"])
        self.assertFalse(cfg["probe_filter"])
        self.assertEqual(
            reg.fingerprint({k: v for k, v in cfg.items() if k not in added}),
            "036f17c9aa644aad")

    def test_it_differs_from_pocket_in_exactly_the_reserve(self):
        a, b = reg.ENGINES["pocket"], reg.ENGINES["pocket_r35"]
        diff = {k for k in a if a[k] != b[k]} - {"name"}
        self.assertEqual(diff, {"reserve"})

    def test_the_reserve_reaches_the_built_search(self):
        # The bug this guards: _engine() used to force reserve from the budget
        # table AFTER applying overrides, so a declared reserve was silently
        # discarded and the "corrected" engine would have been the broken one.
        p = build("pocket_r35")
        self.assertEqual(p.mcts.reserve_ms, 35.0)
        self.assertEqual(build("pocket").mcts.reserve_ms, 20.0)

    def test_it_covers_the_measured_overshoot(self):
        # Measured over 5,517 moves: overhead p99 23.78 ms, worst-chunk p99
        # 5.4 ms. A reserve below their sum cannot pass the requirement.
        self.assertGreaterEqual(float(reg.ENGINES["pocket_r35"]["reserve"]),
                                23.78 + 5.4)

    def test_it_is_not_latency_exempt(self):
        # It is a DEPLOYMENT candidate. The whole point is that it is judged.
        self.assertNotIn("pocket_r35", reg.LADDER_EXEMPT)
        self.assertFalse(build("pocket_r35").latency_exempt)

    def test_it_is_not_an_anchor(self):
        self.assertNotIn("pocket_r35", reg.ANCHOR_ROLES)

    def test_every_other_fingerprint_survived_the_precedence_change(self):
        # Overrides now win over the budget-derived reserve. No pre-existing
        # entry overrides reserve, so nothing already published may move.
        for name in ("original", "final", "anchor_A", "anchor_B", "anchor_C",
                     "anchor_D", "pocket", "midsize"):
            with self.subTest(engine=name):
                ms = int(float(reg.ENGINES[name]["ms"]))
                self.assertEqual(reg.ENGINES[name]["reserve"],
                                 reg._RESERVE[ms])


class TestTheDeploymentBaseline(unittest.TestCase):
    """`DEPLOYED` is a promise about which engine ships. Untested, it is a
    string that drifts away from the registry the first time one is added."""

    def test_the_deployed_engine_exists_and_is_not_raw(self):
        self.assertIn(reg.DEPLOYED, reg.ENGINES)
        self.assertFalse(reg.is_raw(reg.DEPLOYED))

    def test_promotion_grants_no_exemption(self):
        # Being the baseline is not a way to stop being judged. It is still
        # latency-gated at the frozen requirement like any other 1-second arm.
        self.assertNotIn(reg.DEPLOYED, reg.LADDER_EXEMPT)
        if buildable(reg.DEPLOYED):
            p = build(reg.DEPLOYED)
            self.assertFalse(p.latency_exempt)
            self.assertEqual(p.mcts.time_budget_ms,
                             reg.REQUIREMENT["budget_ms"])

    def test_the_baseline_is_not_a_ruler(self):
        # An anchor may never be overridden, so if the deployed engine were one
        # the next ablation (`engine:pocket_defer+solve=0`) would be impossible.
        self.assertNotIn(reg.DEPLOYED, reg.ANCHOR_ROLES)
        spec, diff = reg.derived_spec(reg.DEPLOYED, {"solve": "0"})
        self.assertEqual(diff, {"solve"})
        self.assertIn("solve=0", spec)

    def test_the_baseline_gets_the_strict_source_check(self):
        # The B side of every A/B. If a source edit changes what it does, the
        # candidate's edge is measured against something nobody agreed to.
        self.assertIn(reg.DEPLOYED, reg.STRICT_SOURCE_ROLES)
        self.assertTrue(reg.ANCHOR_ROLES <= reg.STRICT_SOURCE_ROLES)

    def test_source_drift_under_the_baseline_is_a_hard_failure(self):
        stub = mock.Mock()
        stub.ckpt = reg.ENGINES[reg.DEPLOYED]["ckpt"]
        with mock.patch.object(reg, "resolved_config",
                               return_value={"ckpt": stub.ckpt,
                                             "ckpt_sha256": None}), \
             mock.patch.object(reg, "CHECKPOINTS", {}), \
             mock.patch.object(reg, "FINGERPRINTS",
                               {reg.DEPLOYED: "x", "pocket_graph": "x"}), \
             mock.patch.object(reg, "fingerprint", return_value="x"), \
             mock.patch.object(reg, "ENGINE_SOURCES",
                               {"agents/mcts.py": "0" * 64}):
            with self.assertRaises(SystemExit) as cm:
                reg.verify(reg.DEPLOYED, stub)
            self.assertIn("engine source drift", str(cm.exception))
            # A plain candidate warns instead -- it is EXPECTED to change the
            # search. That difference is the whole point of the new role.
            prov = reg.verify("pocket_graph", stub)
            self.assertEqual(prov["source_drift"], ["agents/mcts.py"])

    def test_superseded_baselines_stay_buildable(self):
        # A promotion whose predecessor stops building is a published result
        # that stops being checkable.
        self.assertTrue(set(reg.SUPERSEDED) <= set(reg.ENGINES))
        self.assertNotIn(reg.DEPLOYED, reg.SUPERSEDED)
        for name in reg.SUPERSEDED:
            with self.subTest(engine=name):
                self.assertIn(name, reg.FINGERPRINTS)

    def test_the_promotion_log_agrees_with_the_registry(self):
        self.assertEqual(reg.PROMOTIONS[0]["engine"], reg.DEPLOYED)
        seen = [row["engine"] for row in reg.PROMOTIONS]
        self.assertEqual(seen[1:], list(reg.SUPERSEDED),
                         "every engine that was once deployed and no longer is "
                         "has to appear in SUPERSEDED, newest first")
        dates = [row["date"] for row in reg.PROMOTIONS]
        self.assertEqual(dates, sorted(dates, reverse=True))
        for row in reg.PROMOTIONS:
            with self.subTest(engine=row["engine"]):
                self.assertIn(row["engine"], reg.ENGINES)
                self.assertIn(row["replaced"], reg.ENGINES)
                lo, hi = row["ci"]
                self.assertGreater(lo, 0.5, "a promotion whose interval does "
                                            "not exclude parity is not one")
                self.assertLess(lo, row["score"])
                self.assertLess(row["score"], hi)
                self.assertIn(row["seed"], reg.SEEDS.values())

    def test_the_pre_registered_band_is_recorded_where_there_was_one(self):
        # #46d is the first promotion on this branch whose SIZE and expected
        # effect were set before the match. Recording the band next to the
        # result is what makes that checkable a year from now.
        row = reg.PROMOTIONS[0]
        lo, hi = row["predicted"]
        self.assertLessEqual(lo, row["score"])
        self.assertLessEqual(row["score"], hi)


class TestDerivedEngines(unittest.TestCase):
    """`engine:final+cpuct=2.0` -- a sweep candidate, traceable to a frozen base."""

    def test_override_changes_only_what_was_declared(self):
        p = build_spec("engine:final+cpuct=2.0")
        base = build("final")
        self.assertEqual(p.mcts.c_puct, 2.0)
        self.assertEqual(base.mcts.c_puct, 1.5)
        for field in ("time_budget_ms", "wave_size", "solve", "max_sims",
                      "min_sims", "reserve_ms", "batched_expand",
                      "deadline_margin"):
            with self.subTest(field=field):
                self.assertEqual(getattr(p.mcts, field),
                                 getattr(base.mcts, field))
        self.assertEqual(p.reuse, base.reuse)
        self.assertEqual(p.ckpt, base.ckpt)

    def test_derived_records_its_frozen_base(self):
        p = build_spec("engine:final+cpuct=3.0")
        self.assertEqual(p.provenance["derived_from"], "final")
        self.assertEqual(p.provenance["base_fingerprint"],
                         reg.FINGERPRINTS["final"])
        self.assertEqual(p.provenance["overrides"], {"cpuct": "3.0"})
        # Its own fingerprint MUST differ -- that is what makes it a candidate.
        self.assertNotEqual(p.provenance["fingerprint"],
                            reg.FINGERPRINTS["final"])

    def test_an_anchor_can_never_be_overridden(self):
        # The single most important guard here: a candidate must not be able to
        # move the ruler it is being measured against.
        for a in reg.ANCHOR_ROLES:
            with self.subTest(anchor=a):
                with self.assertRaises(SystemExit) as cm:
                    build_spec(f"engine:{a}+cpuct=2.0")
                self.assertIn("must not be overridden", str(cm.exception))

    def test_override_must_replace_a_key_the_base_pins(self):
        # Otherwise the base was not fully specified and the registry's whole
        # guarantee is void.
        with self.assertRaises(SystemExit) as cm:
            reg.derived_spec("pocket_raw", {"cpuct": "2.0"})
        self.assertIn("does not pin", str(cm.exception))

    def test_plain_registry_engine_has_no_overrides(self):
        p = build("final")
        self.assertEqual(p.overrides, {})
        self.assertNotIn("derived_from", p.provenance)

    def test_restating_the_frozen_value_is_not_a_derived_engine(self):
        """A sweep's incumbent arm is `engine:final`, never
        `engine:final+cpuct=1.5`.

        The two would play identically, but only the first is checked against a
        frozen fingerprint -- the second is a derived engine that happens to
        match, and derived engines are exempt from that check by design. The
        control arm of a sweep is precisely where that exemption must not
        apply, so this is refused rather than quietly accepted.
        """
        _spec, diff = reg.derived_spec("final", {"cpuct": "1.5"})
        self.assertEqual(diff, set(), "1.5 is no longer the frozen c_puct; "
                                      "the sweep's incumbent arm must be "
                                      "re-derived")
        with self.assertRaises(SystemExit) as cm:
            build_spec("engine:final+cpuct=1.5")
        self.assertIn("changes nothing", str(cm.exception))


class TestRegressionGateOpponent(unittest.TestCase):
    """The gate shipped as a mirror and PASSED an engine that fails against a
    real opponent. These lock the fix."""

    def test_a_mirror_is_refused(self):
        from tools import regress_engine as rg
        for name in ("final", "pocket", "pocket_r35"):
            with self.subTest(engine=name):
                with self.assertRaises(SystemExit) as cm:
                    rg.choose_opponent(name, requested=name)
                self.assertIn("mirror", str(cm.exception))

    def test_a_mirror_is_still_possible_when_asked_for_explicitly(self):
        from tools import regress_engine as rg
        self.assertEqual(
            rg.choose_opponent("final", requested="final", allow_mirror=True),
            "final")

    def test_defaults_never_mirror_and_always_exist(self):
        from tools import regress_engine as rg
        for name in reg.ENGINES:
            if reg.is_raw(name):
                continue
            with self.subTest(engine=name):
                opp = rg.choose_opponent(name)
                self.assertNotEqual(opp, name)
                self.assertIn(opp, reg.ENGINES)

    def test_the_default_opponent_uses_a_different_network(self):
        # Shared WEIGHTS are what inflate inheritance, not a shared name. Every
        # non-raw engine -- including `original` and every anchor rung -- must
        # default to an opponent that does not use its checkpoint.
        from tools import regress_engine as rg
        for name in reg.ENGINES:
            if reg.is_raw(name):
                continue
            with self.subTest(engine=name):
                opp = rg.choose_opponent(name)
                self.assertNotEqual(reg.ENGINES[name]["ckpt"],
                                    reg.ENGINES[opp]["ckpt"])

    def test_the_inherit_floor_clears_the_measured_cross_network_value(self):
        # `final` inherits 1551.9 / 3242.4 = 0.479 against `pocket`. The old
        # 0.50 floor was calibrated on the same-network case (0.81) and would
        # fail an engine that is behaving correctly.
        from tools import regress_engine as rg
        self.assertLess(rg.INHERIT_FLOOR, 1551.9 / 3242.4)


class TestSeedPlan(unittest.TestCase):

    def test_seed_namespaces_are_distinct(self):
        self.assertEqual(len(set(reg.SEEDS.values())), len(reg.SEEDS))

    def test_tuning_and_confirmation_use_different_openings(self):
        # Eliminating and confirming on the same fixed positions overfits them.
        self.assertNotEqual(reg.SEEDS["tune"], reg.SEEDS["confirm"])


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        __import__(__name__, fromlist=["*"]))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if res.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
