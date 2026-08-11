"""Tests for the #44 re-profile.

The instrument's failure modes are all SILENT -- a gate that never closes, a
pybind attribute that refuses to be patched, a scale factor that inverts -- and
each of them produces a plausible-looking table rather than an error. So each
one gets a test.

    python -m tools.test_profile_selection
"""

import unittest

import numpy as np

from agents.graph_wave import GraphedWave
from agents.mcts import MCTS, MCTSNode, TreeReuseSearcher
from engine.constants import X, O
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from tools import engine_registry
from tools import profile_selection as ps


class TestTimer(unittest.TestCase):

    def test_exclusive_subtracts_the_nested_call(self):
        t = ps.AttributedTimer()

        def inner():
            end = ps.time.perf_counter() + 0.02
            while ps.time.perf_counter() < end:
                pass
        wi = t.wrap("inner", inner)

        def outer():
            end = ps.time.perf_counter() + 0.01
            while ps.time.perf_counter() < end:
                pass
            wi()
        t.wrap("outer", outer)()

        self.assertGreater(t.total["inner"], 0.015)
        # outer ran ~30 ms of wall but only ~10 ms of its own
        self.assertLess(t.total["outer"], 0.018)
        self.assertGreater(t.inclusive["outer"], 0.025)

    def test_caller_attribution_separates_two_call_sites(self):
        t = ps.AttributedTimer()
        leaf = t.wrap("leaf", lambda: None)
        t.wrap("a", lambda: [leaf() for _ in range(3)])()
        t.wrap("b", lambda: [leaf() for _ in range(5)])()
        leaf()
        self.assertEqual(t.calls_from[("a", "leaf")][0], 3)
        self.assertEqual(t.calls_from[("b", "leaf")][0], 5)
        self.assertEqual(t.calls_from[("-", "leaf")][0], 1)
        self.assertEqual(t.calls["leaf"], 9)

    def test_gate_closed_means_nothing_is_recorded(self):
        ctx = {"on": False}
        t = ps.AttributedTimer(ctx)
        f = t.wrap("x", lambda: None)
        f()
        self.assertEqual(t.calls["x"], 0)
        ctx["on"] = True
        f()
        self.assertEqual(t.calls["x"], 1)

    def test_calibration_leaves_no_residue(self):
        t = ps.AttributedTimer()
        us = t.calibrate(n=2000)
        self.assertGreaterEqual(us, 0.0)
        self.assertNotIn("_calibration", t.total)
        self.assertNotIn("_calibration", t.inclusive)
        self.assertNotIn("_calibration", t.calls)
        self.assertFalse([k for k in t.calls_from if k[1] == "_calibration"])


class TestPatching(unittest.TestCase):
    """Every target has to actually be reachable through its holder."""

    def test_every_target_resolves(self):
        for name, key, attr in ps.WRAP_TARGETS:
            holder = ps.HOLDERS[key]
            self.assertTrue(hasattr(holder, attr),
                            "%s: %s has no %s" % (name, key, attr))

    def test_graph_replay_is_wrapped_separately_from_its_glue(self):
        """One row for the device round trip, one for the Python around it.

        Merged, `_expand_wave_graphed` reports the replay and its own glue as a
        single number and the report cannot say which of the two a further
        optimisation would target.
        """
        names = {n for n, _k, _a in ps.WRAP_TARGETS}
        self.assertIn("device: graph replay", names)
        self.assertIn("device: graphed wave", names)
        self.assertIs(ps.HOLDERS["GRAPHW"], GraphedWave)

    def test_gamestate_is_patchable(self):
        """A pybind11 type that refused setattr would report make_move at 0 ms.

        Silently. That is the whole reason this test exists rather than a
        comment saying it works.
        """
        raw = GameState.__dict__["make_move"]
        seen = []
        try:
            GameState.make_move = lambda self, mv: (seen.append(mv),
                                                    raw(self, mv))[1]
            s = GameState()
            s.make_move(40)
        finally:
            GameState.make_move = raw
        self.assertEqual(seen, [40])
        # and restoring really restored
        s2 = GameState()
        s2.make_move(0)
        self.assertEqual(seen, [40])

    def test_patch_and_restore_round_trips(self):
        before = {}
        for name, key, attr in ps.WRAP_TARGETS:
            before[(key, attr)] = ps.HOLDERS[key].__dict__.get(
                attr, getattr(ps.HOLDERS[key], attr))

        class FakeModel:
            def forward_both(self, x):
                return x, x

        class FakePlayer:
            model = FakeModel()

        t = ps.AttributedTimer()
        p = FakePlayer()
        restore = ps.patch(t, p)
        # something actually changed
        self.assertIsNot(MCTS._best_child, before[("MCTS", "_best_child")])
        self.assertIn("forward_both", vars(p.model))
        restore()
        for (key, attr), raw in before.items():
            holder = ps.HOLDERS[key]
            now = holder.__dict__.get(attr, getattr(holder, attr))
            self.assertIs(now, raw, "%s.%s not restored" % (key, attr))
        # The instance shadow is DELETED, not written back: a module holding a
        # bound method of itself is a cycle, and this profile disables gc.
        self.assertNotIn("forward_both", vars(p.model))
        self.assertEqual(p.model.forward_both(7), (7, 7))

    def test_wrapped_best_child_still_picks_the_same_child(self):
        """The instrument must not change what the search decides."""
        m = MCTS(model=None, device="cpu", n_sims=1, c_puct=1.5, wave_size=8,
                 solve=True)
        parent = MCTSNode(to_play=X)
        parent.N = 40
        for i in range(9):
            c = MCTSNode(parent=parent, prior=0.1 + 0.01 * i, move=i, to_play=O)
            c.N = i
            c.W = 0.05 * i
            parent.children[i] = c
        want = m._best_child(parent).move

        t = ps.AttributedTimer()
        raw = MCTS._best_child
        try:
            MCTS._best_child = t.wrap("_best_child", raw)
            got = m._best_child(parent).move
        finally:
            MCTS._best_child = raw
        self.assertEqual(want, got)
        self.assertEqual(t.calls["_best_child"], 1)


class TestGate(unittest.TestCase):

    def test_gate_closes_after_the_move(self):
        """The opponent's search must not be charged to our arm.

        `tools/profile_tree` never had to close its gate because both of its
        players were instrumented and each move reopened it. Here the opponent
        is a different engine, so a gate left open would silently double the
        measured cost of every operation.
        """
        ctx = {"on": False}

        class P:
            recording = True

            def move(self, state, move_num):
                assert ctx["on"], "gate should be open inside our own move"
                return 0

        p = P()
        ps.gate(p, ctx, True)
        p.move(None, 0)
        self.assertFalse(ctx["on"])

    def test_gate_stays_shut_for_the_opponent(self):
        ctx = {"on": False}
        seen = []

        class P:
            recording = True

            def move(self, state, move_num):
                seen.append(ctx["on"])
                return 0

        p = P()
        ps.gate(p, ctx, False)
        p.move(None, 0)
        self.assertEqual(seen, [False])

    def test_warmup_moves_are_not_counted(self):
        ctx = {"on": False}
        seen = []

        class P:
            recording = False

            def move(self, state, move_num):
                seen.append(ctx["on"])
                return 0

        p = P()
        ps.gate(p, ctx, True)
        p.move(None, 0)
        p.recording = True
        p.move(None, 1)
        self.assertEqual(seen, [False, True])


class TestScaling(unittest.TestCase):

    def test_scale_is_above_one_when_the_instrument_slows_the_search(self):
        clean = {"nn_per_move": 5000.0}
        dirty = {"nn_per_move": 4000.0}
        self.assertAlmostEqual(ps.scale_of(clean, dirty), 1.25)

    def test_scale_direction_cannot_silently_invert(self):
        """An inverted scale would SHRINK every cost and look like good news."""
        clean = {"nn_per_move": 5000.0}
        dirty = {"nn_per_move": 4000.0}
        self.assertGreater(ps.scale_of(clean, dirty), 1.0)

    def test_the_default_unit_is_network_evaluations(self):
        """Not simulations. A descent into a proven subtree is counted as a
        simulation and costs almost nothing, so sims/move tracks proof
        structure as well as throughput."""
        arm = {"nn_per_move": 4000.0, "sims_per_move": 9000.0}
        clean = {"nn_per_move": 5000.0, "sims_per_move": 9500.0}
        self.assertAlmostEqual(ps.scale_of(clean, arm), 1.25)
        self.assertAlmostEqual(ps.scale_of(clean, arm, "sims_per_move"),
                               9500.0 / 9000.0)

    def test_scale_survives_a_zero(self):
        self.assertTrue(np.isnan(ps.scale_of({"nn_per_move": 1.0},
                                             {"nn_per_move": 0.0})))


class TestSummarize(unittest.TestCase):

    def _arm(self):
        class FakeMCTS:
            graph_wave = None
            graph_wave_requested = False
            reserve_ms = 50.0

        class FakePlayer:
            engine_name = "fake"
            provenance = {"fingerprint": "abc"}
            net_info = {"params": 172389}
            budget_ms = 1000.0
            wave = 8
            mcts = FakeMCTS()

        ctx = {"on": True}
        t = ps.AttributedTimer(ctx)
        f = t.wrap("_best_child", lambda: None)
        for _ in range(100):
            f()
        rows = [{"sims": 5000, "nn": 3900, "expansions": 2700,
                 "search_ms": 990.0}]
        return ps.summarize(FakePlayer(), rows, t, 10, 990.0)

    def test_rows_and_totals_are_present(self):
        out = self._arm()
        self.assertEqual(out["ops"]["_best_child"]["calls"], 100)
        self.assertAlmostEqual(out["ops"]["_best_child"]["calls_per_move"],
                               10.0)
        self.assertTrue(out["wrapped"])

    def test_summarize_leaves_the_rows_UNPRICED(self):
        """The price depends on the clean arm, which `summarize` cannot see.

        A row that arrived already priced would be priced twice as soon as
        `price_arm` ran, and the double subtraction would be invisible.
        """
        out = self._arm()
        self.assertIn("raw_exclusive_ms_per_move", out["ops"]["_best_child"])
        self.assertNotIn("exclusive_ms_per_move", out["ops"]["_best_child"])
        self.assertNotIn("measured_ms_per_move", out)

    def test_overhead_is_subtracted_from_the_total_once_priced(self):
        out = self._arm()
        ps.price_arm(out, {"nn_per_move": out["nn_per_move"] * 1.2,
                           "sims_per_move": out["sims_per_move"] * 1.2})
        self.assertLess(out["measured_ms_per_move"], out["wall_ms_per_move"])

    def test_clean_arm_reports_no_ops(self):
        class FakeMCTS:
            graph_wave = None
            graph_wave_requested = True
            reserve_ms = 50.0

        class FakePlayer:
            engine_name = "fake"
            provenance = None
            net_info = {"params": 1}
            budget_ms = 1000.0
            wave = 8
            mcts = FakeMCTS()

        out = ps.summarize(FakePlayer(), [{"sims": 1, "nn": 1, "expansions": 1,
                                           "search_ms": 1.0}], None, 1, 1.0)
        self.assertNotIn("ops", out)
        self.assertFalse(out["wrapped"])


def payload_fixture(moves=10, wall=995.0, wrapped_sims=4000.0,
                    clean_sims=5200.0, wrapped_nn=3000.0, clean_nn=3900.0):
    """A synthetic payload in the SCHEMA `summarize` actually produces.

    Raw sums only, exactly as an arm comes off the timer -- the price is applied
    downstream by `price_arm`, and a fixture that pre-applied it would let the
    pricing break without any test noticing.
    """
    # Shaped like the real thing: a few hot sub-microsecond leaf ops that a
    # uniform per-call charge would drive negative, and some expensive
    # structural ones that it would not.
    shape = {"_best_child": (3000.0, 3.5), "state.make_move": (6000.0, 0.8),
             "state clone": (4000.0, 0.9), "legal moves": (400.0, 1.8),
             "terminal probes": (400.0, 20.0), "backup": (400.0, 1.1),
             "node creation": (50.0, 60.0), "wave loop": (50.0, 170.0),
             "tree reuse: adopt": (1.0, 300.0), "tree release": (1.0, 900.0),
             "device: graphed wave": (50.0, 27.0),
             "device: graph replay": (50.0, 700.0), ps.FORWARD: (2.0, 850.0)}

    def arm(engine, workload, graph):
        ops = {}
        per_move = 0.0
        for name, (n, us) in shape.items():
            per_move += n
            ops[name] = {"calls": int(n * moves), "calls_per_move": n,
                         "nested_calls": int(n * moves // 4),
                         "raw_exclusive_ms_per_move": n * us / 1000.0,
                         "raw_inclusive_ms_per_move": n * us / 1000.0 * 1.4}
        return {"engine": engine, "workload": workload, "params": 172389,
                "graph": graph, "graph_requested": graph,
                "budget_ms": 1000.0, "reserve_ms": 50.0, "wave": 8,
                "moves": moves, "wall_ms_per_move": wall,
                "sims_per_move": wrapped_sims, "nn_per_move": wrapped_nn,
                "expansions_per_move": 2000.0, "wrapped": True,
                "ops": ops, "calls_per_move": per_move,
                "calibrated_us_per_call": 0.9,
                "inside_us_per_call": 0.3,
                "top_level_calls_per_move": 60.0,
                "callers": {"wave loop -> state.make_move":
                            {"calls": 100, "raw_ms_per_move": 20.0},
                            "terminal probes -> state.make_move":
                            {"calls": 50, "raw_ms_per_move": 8.0},
                            "terminal probes -> state clone":
                            {"calls": 50, "raw_ms_per_move": 7.0}}}

    def clean(engine, workload):
        return {"engine": engine, "workload": workload,
                "wall_ms_per_move": wall, "sims_per_move": clean_sims,
                "nn_per_move": clean_nn, "params": 172389, "moves": moves,
                "expansions_per_move": 2600.0, "wrapped": False}

    arms, cleans, solos, counts = [], [], [], []
    for engine, graph in (("pocket_r35", False), ("pocket_graph", True)):
        for workload in ("fixed", "game"):
            arms.append(arm(engine, workload, graph))
            cleans.append(clean(engine, workload))
            # The counting arm is barely perturbed, so it sees MORE calls than
            # the timed arm -- that gap is the whole reason it exists.
            c = arm(engine, workload, graph)
            c["counting"] = True
            c["sims_per_move"] = clean_sims * 0.99
            c["nn_per_move"] = clean_nn * 0.99
            for o in c["ops"].values():
                o["calls_per_move"] *= 1.3
            counts.append(c)
        solo = arm(engine, "fixed", graph)
        solo["only"] = ["_best_child"]
        solo["ops"] = {"_best_child": solo["ops"]["_best_child"]}
        solo["sims_per_move"] = clean_sims * 0.92
        solo["nn_per_move"] = clean_nn * 0.92
        solos.append(solo)
    return {"arms": arms, "clean": cleans, "solo": solos, "count": counts}


class TestPricing(unittest.TestCase):
    """The instrument is charged at what it cost, not what it benchmarks."""

    def test_direct_plus_diffuse_equals_the_measured_instrument_cost(self):
        p = payload_fixture()
        arm, clean = p["arms"][0], p["clean"][0]
        ps.price_arm(arm, clean)
        self.assertAlmostEqual(arm["direct_ms_per_move"]
                               + arm["diffuse_ms_per_move"],
                               arm["instrument_ms_per_move"], places=9)

    def test_scaled_total_reconciles_to_the_clean_arm(self):
        """The whole point of pricing this way: the table has to add up."""
        p = payload_fixture()
        arm, clean = p["arms"][0], p["clean"][0]
        ps.price_arm(arm, clean)
        self.assertAlmostEqual(arm["measured_ms_per_move"] * arm["scale"],
                               clean["wall_ms_per_move"], places=6)

    def test_a_uniform_per_call_charge_would_go_negative(self):
        """Why the diffuse half is a deflation and not a per-call charge.

        Charging the whole measured instrument cost per wrapper entry drives
        the cheapest rows below zero -- which is how that model was found to be
        wrong. The deflation keeps every row on the right side of zero.
        """
        p = payload_fixture()
        arm, clean = p["arms"][0], p["clean"][0]
        uniform = (arm["wall_ms_per_move"]
                   * (1.0 - arm["nn_per_move"] / clean["nn_per_move"])
                   * 1000.0 / arm["calls_per_move"])
        cheapest = min(o["raw_exclusive_ms_per_move"] / o["calls_per_move"]
                       * 1000.0 for o in arm["ops"].values())
        self.assertLess(cheapest, uniform)
        ps.price_arm(arm, clean)
        for name, o in arm["ops"].items():
            self.assertGreater(o["exclusive_ms_per_move"], 0.0, name)

    def test_pricing_lowers_every_row(self):
        p = payload_fixture()
        arm, clean = p["arms"][0], p["clean"][0]
        ps.price_arm(arm, clean)
        for name, o in arm["ops"].items():
            self.assertLess(o["exclusive_ms_per_move"],
                            o["raw_exclusive_ms_per_move"], name)

    def test_the_row_is_charged_the_INSIDE_half_only(self):
        """A row cannot be charged the whole wrapper price.

        Most of a wrapper's cost is spent before its clock starts and after it
        stops, so it was never in that row's recorded time. Charging the whole
        price is a double subtraction, and it is what drove `_best_child` to
        -1.113 us/call on the first full run.
        """
        p = payload_fixture()
        arm, clean = p["arms"][0], p["clean"][0]
        ps.price_arm(arm, clean)
        self.assertLess(arm["inside_us"], arm["calibrated_us_per_call"])
        for o in arm["ops"].values():
            self.assertAlmostEqual(o["_price_us"], arm["inside_us"])
        bc = arm["ops"]["_best_child"]
        self.assertAlmostEqual(
            bc["_direct_ms"],
            (bc["calls_per_move"] * arm["inside_us"]
             + bc["nested_calls"] / arm["moves"] * arm["outside_us"]) / 1000.0)

    def test_the_outside_half_is_charged_to_the_caller(self):
        """A nested wrapper's out-of-interval cost lands in its PARENT's
        recorded time, which is the only row it can honestly come off."""
        p = payload_fixture()
        arm, clean = p["arms"][0], p["clean"][0]
        arm["ops"]["wave loop"]["nested_calls"] = 500000
        ps.price_arm(arm, clean)
        heavy = arm["ops"]["wave loop"]["_direct_ms"]
        arm2, clean2 = payload_fixture()["arms"][0], payload_fixture()["clean"][0]
        arm2["ops"]["wave loop"]["nested_calls"] = 0
        ps.price_arm(arm2, clean2)
        self.assertGreater(heavy, arm2["ops"]["wave loop"]["_direct_ms"])

    def test_callers_are_priced_too(self):
        """An unpriced caller row would report the descent's make_move as
        costing more than the make_move row it is a subset of."""
        p = payload_fixture()
        arm, clean = p["arms"][0], p["clean"][0]
        ps.price_arm(arm, clean)
        for c in arm["callers"].values():
            self.assertLess(c["ms_per_move"], c["raw_ms_per_move"])


class TestCountingArm(unittest.TestCase):
    """calls/move comes from the arm that barely perturbs it."""

    def test_the_counting_timer_only_counts(self):
        t = ps.CountingTimer()
        f = t.wrap("x", lambda a: a * 2)
        self.assertEqual(f(21), 42)
        self.assertEqual(t.calls["x"], 1)
        self.assertEqual(t.total["x"], 0.0)

    def test_summarize_keeps_rows_for_a_timer_with_no_durations(self):
        """`summarize` used to iterate `total`, which a counting timer never
        writes -- so the counting arm came back with an EMPTY ops dict and
        `per_move` fell straight back to the timed arm's call counts. Silent,
        and it reinstated the extrapolation the counting arm removes."""
        class FakeMCTS:
            graph_wave = None
            graph_wave_requested = False
            reserve_ms = 50.0

        class FakePlayer:
            engine_name = "fake"
            provenance = None
            net_info = {"params": 1}
            budget_ms = 1000.0
            wave = 8
            mcts = FakeMCTS()

        t = ps.CountingTimer({"on": True})
        f = t.wrap("_best_child", lambda: None)
        for _ in range(50):
            f()
        out = ps.summarize(FakePlayer(),
                           [{"sims": 10, "nn": 8, "expansions": 4,
                             "search_ms": 900.0}], t, 5, 900.0)
        self.assertIn("_best_child", out["ops"])
        self.assertAlmostEqual(out["ops"]["_best_child"]["calls_per_move"], 10.0)
        self.assertAlmostEqual(
            out["ops"]["_best_child"]["raw_exclusive_ms_per_move"], 0.0)

    def test_counting_is_far_cheaper_than_timing(self):
        """If it were not, it would perturb what it is there to measure."""
        import timeit

        def noop():
            return None
        a = timeit.timeit(ps.CountingTimer().wrap("x", noop), number=200000)
        b = timeit.timeit(ps.AttributedTimer().wrap("x", noop), number=200000)
        self.assertLess(a, b * 0.75)

    def test_per_move_prefers_the_counting_arms_call_rate(self):
        p = payload_fixture()
        arm, clean = p["arms"][0], p["clean"][0]
        count = p["count"][0]
        ps.price_arm(arm, clean)
        calls, ms = ps.per_move(arm, count, "_best_child")
        self.assertAlmostEqual(
            calls, count["ops"]["_best_child"]["calls_per_move"])
        self.assertGreater(calls, arm["ops"]["_best_child"]["calls_per_move"])
        self.assertAlmostEqual(
            ms, calls * arm["ops"]["_best_child"]["us_per_call"] / 1000.0)

    def test_the_counting_arms_own_perturbation_is_corrected(self):
        """It is cheap, not free. Its call rate is still a few percent low."""
        p = payload_fixture()
        arm, clean, count = p["arms"][0], p["clean"][0], p["count"][0]
        ps.price_arm(arm, clean)
        count["scale"] = ps.scale_of(clean, count)
        self.assertGreater(count["scale"], 1.0)
        calls, _ms = ps.per_move(arm, count, "_best_child")
        self.assertAlmostEqual(
            calls,
            count["ops"]["_best_child"]["calls_per_move"] * count["scale"])

    def test_per_move_falls_back_to_the_timed_arm(self):
        p = payload_fixture()
        arm, clean = p["arms"][0], p["clean"][0]
        ps.price_arm(arm, clean)
        calls, _ms = ps.per_move(arm, None, "_best_child")
        self.assertAlmostEqual(calls,
                               arm["ops"]["_best_child"]["calls_per_move"])

    def test_a_missing_op_in_the_counting_arm_is_not_fatal(self):
        p = payload_fixture()
        arm, clean, count = p["arms"][0], p["clean"][0], p["count"][0]
        ps.price_arm(arm, clean)
        del count["ops"]["_best_child"]
        calls, _ms = ps.per_move(arm, count, "_best_child")
        self.assertAlmostEqual(calls,
                               arm["ops"]["_best_child"]["calls_per_move"])


class TestInSitu(unittest.TestCase):

    def _solo(self, nn):
        p = payload_fixture(wrapped_nn=nn)
        solo = p["arms"][0]
        solo["only"] = ["_best_child"]
        return solo, p["clean"][0]

    def test_price_comes_from_the_search_the_wrapper_cost(self):
        solo, clean = self._solo(3510.0)     # 10% of 3900 nn-evals lost
        pr = ps.insitu_price(solo, clean, "_best_child")
        self.assertAlmostEqual(pr["nn_lost_pct"], 10.0)
        self.assertAlmostEqual(pr["cost_ms_per_move"], 99.5)
        self.assertAlmostEqual(pr["us_per_call"], 99.5 * 1000.0 / 3000.0)

    def test_missing_op_is_not_an_error(self):
        solo, clean = self._solo(3510.0)
        self.assertIsNone(ps.insitu_price(solo, clean, "no such op"))

    def test_elasticity_is_a_measured_slope(self):
        solo, clean = self._solo(3510.0)
        pr = ps.insitu_price(solo, clean, "_best_child")
        el = ps.elasticity(pr, 80.0, clean)
        # 390 network evaluations lost for 99.5 ms of added cost
        self.assertAlmostEqual(el["nn_per_ms_of_selection"], 390.0 / 99.5)
        self.assertAlmostEqual(
            el["nn_if_selection_were_free"], 3900.0 + 80.0 * 390.0 / 99.5)
        self.assertGreater(el["pct_more_search"], 0.0)
        # simulations are carried alongside but are NOT the headline
        self.assertIn("pct_more_sims", el)

    def test_elasticity_on_a_free_wrapper_is_none(self):
        solo, clean = self._solo(3900.0)
        pr = ps.insitu_price(solo, clean, "_best_child")
        self.assertIsNone(ps.elasticity(pr, 80.0, clean))


class TestReportSmoke(unittest.TestCase):
    """The report must not be able to destroy a completed measurement.

    A nine-minute run was once thrown away by a NameError in the line that
    printed its verdict, so rendering is exercised on a synthetic payload.
    """

    def _payload(self):
        return payload_fixture()

    def test_render_runs_and_returns_a_verdict(self):
        import io
        import contextlib
        payload = self._payload()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = ps.render(payload)
        self.assertIn("verdict", out)
        self.assertIn("dominant_host_op", out["verdict"])
        text = buf.getvalue()
        self.assertIn("_BEST_CHILD", text)
        self.assertIn("#44 VERDICT", text)
        # the in-situ price and the slope it hands #45 both rendered
        self.assertIn("IN-SITU PRICE", text)
        self.assertIn("Measured slope", text)
        self.assertTrue(out["insitu"])
        self.assertTrue(out["elasticity"])

    def test_report_is_ascii(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ps.render(self._payload())
        buf.getvalue().encode("ascii")


class TestWiring(unittest.TestCase):

    def test_device_ops_are_excluded_from_the_host_ranking(self):
        """A device row winning the host ranking would misdirect #36."""
        for name in ps.DEVICE_OPS:
            self.assertIn(name, ps.ORDER + [ps.FORWARD])

    def test_every_wrap_target_has_a_place_in_the_report(self):
        for name, _key, _attr in ps.WRAP_TARGETS:
            self.assertIn(name, ps.ORDER)

    def test_seed_namespace_is_distinct(self):
        self.assertEqual(ps.SELECT_SEED, engine_registry.SEEDS["select"])
        self.assertEqual(len(set(engine_registry.SEEDS.values())),
                         len(engine_registry.SEEDS))
        for other in ("confirm", "tune", "headline"):
            self.assertNotEqual(ps.SELECT_SEED,
                                engine_registry.SEEDS[other])

    def test_reuse_ops_are_not_in_the_device_bucket(self):
        self.assertFalse(set(ps.REUSE_OPS) & set(ps.DEVICE_OPS))


class TestFixedDriverShape(unittest.TestCase):
    """The fixed driver on a CPU toy model -- structure, not timing."""

    def test_bare_root_never_adopts_or_releases_inside_the_gate(self):
        """`_adopt` and `release` CANNOT fire from a bare root.

        If they ever show a non-zero count in fixed mode, the driver has
        started going through TreeReuseSearcher and the workload is no longer
        the one the report claims.
        """
        import torch

        class Toy:
            def forward_both(self, x):
                # Real `forward_both` squeezes batch=1 to (81,)/scalar, and
                # `_expand` (the root expansion) depends on that. A toy that
                # kept the batch dim would fail before any wave ran.
                b = x.shape[0]
                lg, v = torch.zeros((b, 81)), torch.zeros((b,))
                return (lg[0], v[0]) if b == 1 else (lg, v)

        m = MCTS(model=Toy(), device="cpu", n_sims=24, c_puct=1.5,
                 wave_size=8, solve=True)
        ctx = {"on": True}
        t = ps.AttributedTimer(ctx)
        raws = []
        for name, attr in (("tree reuse: adopt", "_adopt"),):
            raws.append((TreeReuseSearcher, attr,
                         TreeReuseSearcher.__dict__[attr]))
        raws.append((TreeReuseSearcher, "release",
                     TreeReuseSearcher.__dict__["release"]))
        try:
            for holder, attr, raw in raws:
                fn = raw.__func__ if isinstance(raw, staticmethod) else raw
                w = t.wrap("tree reuse: adopt" if attr == "_adopt"
                           else "tree release", fn)
                setattr(holder, attr,
                        staticmethod(w) if isinstance(raw, staticmethod) else w)
            s = GameState()
            for mv in (40, 4, 36):
                s.make_move(mv)
            m.search(s)
        finally:
            for holder, attr, raw in raws:
                setattr(holder, attr, raw)
        self.assertEqual(t.calls["tree reuse: adopt"], 0)
        self.assertEqual(t.calls["tree release"], 0)

    def test_make_move_is_called_from_both_the_descent_and_the_probes(self):
        """The split the report turns on has to exist in the real search."""
        import torch

        class Toy:
            def forward_both(self, x):
                # Real `forward_both` squeezes batch=1 to (81,)/scalar, and
                # `_expand` (the root expansion) depends on that. A toy that
                # kept the batch dim would fail before any wave ran.
                b = x.shape[0]
                lg, v = torch.zeros((b, 81)), torch.zeros((b,))
                return (lg[0], v[0]) if b == 1 else (lg, v)

        m = MCTS(model=Toy(), device="cpu", n_sims=32, c_puct=1.5,
                 wave_size=8, solve=True)
        ctx = {"on": True}
        t = ps.AttributedTimer(ctx)
        saved = [(MCTS, "_run_wave", MCTS.__dict__["_run_wave"]),
                 (MCTS, "_mark_terminal_children",
                  MCTS.__dict__["_mark_terminal_children"]),
                 (GameState, "make_move", GameState.__dict__["make_move"])]
        try:
            MCTS._run_wave = t.wrap("wave loop", saved[0][2])
            MCTS._mark_terminal_children = t.wrap("terminal probes",
                                                  saved[1][2])
            GameState.make_move = t.wrap("state.make_move", saved[2][2])
            s = GameState()
            for mv in (40, 4, 36):
                s.make_move(mv)
            m.search(s)
        finally:
            for holder, attr, raw in saved:
                setattr(holder, attr, raw)
        self.assertGreater(
            t.calls_from[("wave loop", "state.make_move")][0], 0)
        self.assertGreater(
            t.calls_from[("terminal probes", "state.make_move")][0], 0)

    def test_valid_moves_helper_is_the_one_mcts_actually_calls(self):
        self.assertIs(ps.mcts_mod.rule_utl_valid_moves, rule_utl_valid_moves)


if __name__ == "__main__":
    unittest.main(verbosity=2)
