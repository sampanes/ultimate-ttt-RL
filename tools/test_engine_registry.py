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


def build(name):
    return TimedPlayer(f"engine:{name}", DEV)


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
                  "maxsims", "reserve", "bexp", "name"}
        for name, spec in reg.ENGINES.items():
            with self.subTest(engine=name):
                self.assertEqual(set(spec), pinned)


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
