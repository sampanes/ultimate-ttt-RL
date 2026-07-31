"""Tests for the tree profiler -- including whether the instrument itself is
honest.

The load-bearing test is `TestSamplerAccuracy`: a workload whose split is known
in advance by construction, checked against what the sampler reports. A
profiler that is merely plausible is worse than none, because its output gets
spent on a port that takes weeks.

    python -m tools.test_profile_tree
"""

import os
import threading
import time
import unittest

from agents.mcts import MCTS, MCTSNode
from engine.constants import X, O
from engine.rules import rule_utl_valid_moves
from tools import engine_registry
from tools import profile_tree as pt


def _spin_alpha(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def _spin_beta(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


class TestSamplerAccuracy(unittest.TestCase):
    """Ground truth is set by construction, then read back off the sampler."""

    def _profile(self, alpha_s, beta_s, hz=500):
        ctx = {"phase": "mid"}
        s = pt.StackSampler(threading.get_ident(), ctx, hz=hz)
        s.start()
        try:
            _spin_alpha(alpha_s)
            _spin_beta(beta_s)
        finally:
            s.stop()
        by_func = {}
        for (_ph, _f, func, _l), n in s.leaf.items():
            by_func[func] = by_func.get(func, 0) + n
        total = sum(by_func.values())
        return {k: v / total for k, v in by_func.items()}, total, s

    def test_split_is_recovered(self):
        # 2.4 s of work: Windows delivers ~130 Hz however fast we ask, so this
        # is what it takes to get a few hundred samples and a +/-0.08 claim.
        shares, total, _s = self._profile(1.80, 0.60)
        self.assertGreater(total, 150, "sampler collected too few samples")
        self.assertAlmostEqual(shares.get("_spin_alpha", 0.0), 0.75, delta=0.08)
        self.assertAlmostEqual(shares.get("_spin_beta", 0.0), 0.25, delta=0.08)

    def test_jitter_does_not_touch_the_shared_rng(self):
        """The harness reseeds `random` per game. A sampler drawing from it
        would change the openings -- a profiler altering its own workload."""
        import random as pyrandom
        pyrandom.seed(12345)
        want = [pyrandom.random() for _ in range(5)]

        pyrandom.seed(12345)
        ctx = {"phase": "mid"}
        s = pt.StackSampler(threading.get_ident(), ctx, hz=500)
        s.start()
        try:
            _spin_alpha(0.30)
        finally:
            s.stop()
        got = [pyrandom.random() for _ in range(5)]
        self.assertEqual(want, got)
        self.assertGreater(s.samples, 10, "sampler never ran")

    def test_sampler_overhead_is_small(self):
        """The tool's own cost must not be the thing it measures."""
        t0 = time.perf_counter()
        _spin_alpha(0.30)
        clean = time.perf_counter() - t0

        ctx = {"phase": "mid"}
        s = pt.StackSampler(threading.get_ident(), ctx, hz=1000)
        s.start()
        t0 = time.perf_counter()
        try:
            _spin_alpha(0.30)
        finally:
            dirty = time.perf_counter() - t0
            s.stop()
        # _spin is wall-clock bounded, so overhead shows up as overshoot.
        self.assertLess(dirty - clean, 0.05,
                        "sampler cost %.1f ms over 300 ms"
                        % ((dirty - clean) * 1000.0))

    def test_per_sample_ms_self_calibrates(self):
        """A rate the OS cannot deliver must not corrupt the ms/move figure."""
        shares, total, s = self._profile(0.80, 0.20, hz=4000)
        per_sample_ms = s.wall * 1000.0 / s.samples
        # Whatever rate was achieved, samples x per-sample must recover the
        # wall clock -- that identity is what makes the nominal Hz irrelevant.
        self.assertAlmostEqual(s.samples * per_sample_ms / 1000.0, s.wall,
                               places=6)
        self.assertAlmostEqual(shares.get("_spin_alpha", 0.0), 0.8, delta=0.10)


class TestSamplerRobustness(unittest.TestCase):
    """A long measurement must not be destroyed by the code that saves it."""

    def test_a_none_line_number_is_recorded_not_crashed_on(self):
        # f_lineno is None for a frame caught mid-teardown. It threw away a
        # nine-minute run once, at the serialization line, after all the work
        # was already done.
        s = pt.StackSampler(threading.get_ident(), {"phase": "mid"})
        s.leaf[("mid", "x.py", "f", None)] = 3
        s.leaf[("mid", "x.py", "f", 12)] = 5
        keys = ["%s|%s|%s|%s" % k for k in s.leaf]
        self.assertEqual(len(keys), 2)
        for k in keys:
            phase, filename, func, line = k.rsplit("|", 3)
            self.assertIn(line, ("None", "12"))

    def test_report_survives_an_unparseable_line_number(self):
        res = {
            "in_search_samples": 8, "ms_per_move": 100.0,
            "leaf": {"mid|/x/agents/mcts.py|_best_child|-1": 8},
        }
        shares = pt.report_sample(res)
        self.assertAlmostEqual(shares["child scoring / best-child"], 1.0)


class TestSamplerGilBias(unittest.TestCase):
    """The defect that invalidated the first profile, kept as a regression.

    The original ground-truth test compared two PURE-PYTHON functions and so
    could never have caught this. A calibration workload has to span the axis
    the instrument is used across.
    """

    def test_the_gil_bias_is_real_and_large(self):
        def py_spin(sec):
            end = time.perf_counter() + sec
            while time.perf_counter() < end:
                pass

        def c_block(sec):
            time.sleep(sec)     # GIL released: what a CUDA sync looks like

        ctx = {"phase": "mid"}
        s = pt.StackSampler(threading.get_ident(), ctx, hz=500)
        s.start()
        try:
            for _ in range(2):
                py_spin(0.25)
                c_block(0.25)
        finally:
            s.stop()
        by = {}
        for (_ph, _f, fn, _l), n in s.leaf.items():
            by[fn] = by.get(fn, 0) + n
        total = sum(by.values())
        self.assertGreater(total, 50)
        c_share = by.get("c_block", 0) / total
        # Ground truth is 0.5. If this ever drops near 0.5 the bias has been
        # fixed and `--mode wrap` could be reconsidered as the primary.
        self.assertGreater(c_share, 0.70,
                           "GIL bias measured at %.3f -- was 0.864" % c_share)


class TestExclusiveTimer(unittest.TestCase):

    def test_nested_time_is_charged_to_the_inner_operation(self):
        t = pt.ExclusiveTimer()

        def inner():
            end = time.perf_counter() + 0.05
            while time.perf_counter() < end:
                pass
        wrapped_inner = t.wrap("inner", inner)

        def outer():
            wrapped_inner()
            end = time.perf_counter() + 0.05
            while time.perf_counter() < end:
                pass
        wrapped_outer = t.wrap("outer", outer)

        wrapped_outer()
        self.assertAlmostEqual(t.total["inner"], 0.05, delta=0.02)
        self.assertAlmostEqual(t.total["outer"], 0.05, delta=0.02)
        self.assertEqual(t.calls["outer"], 1)
        self.assertEqual(t.calls["inner"], 1)

    def test_calibration_prices_a_wrapped_call(self):
        t = pt.ExclusiveTimer()
        us = t.calibrate(n=20000)
        self.assertGreater(us, 0.0)
        self.assertLess(us, 5.0, "wrapper cost %.2f us is too high to "
                                 "subtract credibly" % us)
        self.assertNotIn("_calibration", t.total)

    def test_every_wrap_target_resolves(self):
        holders = {"MCTS": MCTS, "MOD": pt.mcts_mod,
                   "REUSE": pt.TreeReuseSearcher}
        for name, key, attr in pt.WRAP_TARGETS:
            with self.subTest(op=name):
                self.assertTrue(hasattr(holders[key], attr),
                                "%s has no %s" % (key, attr))

    def test_staticmethods_survive_being_wrapped(self):
        # `release` and `_solve_from_children` are staticmethods; wrapping them
        # as plain functions would silently pass `self` as the first argument.
        for holder, attr in ((pt.TreeReuseSearcher, "release"),
                             (MCTS, "_solve_from_children")):
            with self.subTest(attr=attr):
                self.assertIsInstance(holder.__dict__[attr], staticmethod)


class TestClassification(unittest.TestCase):
    def test_every_named_operation_is_reachable(self):
        """The seven operations the owner named must each have a mapping."""
        named = {
            "selection traversal", "child scoring / best-child",
            "node allocation", "backup traversal", "legal-child iteration",
            "proof propagation", "tree release",
        }
        mapped = set(pt.OPERATION.values())
        self.assertEqual(named - mapped, set(),
                         "unmapped operations: %s" % (named - mapped))

    def test_mapped_functions_exist_in_the_frozen_sources(self):
        """A mapping for a function that no longer exists is a silent hole:
        its time would land in `other:` and read as unexplained."""
        import inspect

        from agents import agent_base, mcts as mcts_mod
        from engine import rules
        mods = {"mcts.py": mcts_mod, "agent_base.py": agent_base,
                "rules.py": rules}
        missing = []
        for (base, func) in pt.OPERATION:
            mod = mods.get(base)
            if mod is None:
                continue          # neural_net_agent_3 methods; checked by use
            src = inspect.getsource(mod)
            if func in ("__init__", "Q", "U", "<lambda>", "release", "_count",
                        "_adopt", "search", "forward", "forward_both"):
                continue          # methods / lambdas, not module-level names
            if ("def %s(" % func) not in src:
                missing.append("%s:%s" % (base, func))
        self.assertEqual(missing, [], "mapped but absent: %s" % missing)

    def test_unknown_code_is_named_not_swallowed(self):
        got = pt.classify("/x/y/mystery_module.py", "do_thing")
        self.assertEqual(got, "other: mystery_module.py:do_thing")

    def test_harness_is_separated_from_the_engine(self):
        self.assertEqual(pt.classify("/x/tools/arena_1s.py", "play_match"),
                         "harness")

    def test_line_split_targets_are_real_methods(self):
        for name in pt.LINE_SPLIT:
            self.assertTrue(hasattr(MCTS, name) or name == "release",
                            "%s is not an MCTS method" % name)


class TestSourceGuard(unittest.TestCase):
    def test_frozen_sources_match_right_now(self):
        """If this fails the tree has drifted from arena-1s-baseline, and no
        line number in any profile written since means what it says."""
        pt.assert_frozen_sources()

    def test_guard_is_fatal_not_advisory(self):
        real = dict(engine_registry.ENGINE_SOURCES)
        try:
            engine_registry.ENGINE_SOURCES["agents/mcts.py"] = "0" * 64
            with self.assertRaises(SystemExit):
                pt.assert_frozen_sources()
        finally:
            engine_registry.ENGINE_SOURCES.clear()
            engine_registry.ENGINE_SOURCES.update(real)

    def test_profile_seed_is_its_own_namespace(self):
        seeds = engine_registry.SEEDS
        self.assertEqual(pt.PROFILE_SEED, seeds["profile"])
        self.assertEqual(len(set(seeds.values())), len(seeds),
                         "two experiment namespaces share an opening set")


class TestMicrobench(unittest.TestCase):
    def test_sampled_states_are_playable(self):
        states = pt._sample_states(20, seed=3)
        self.assertEqual(len(states), 20)
        for s in states:
            self.assertFalse(s.is_over())
            self.assertTrue(rule_utl_valid_moves(s.board, s.last_move,
                                                 s.mini_winners))

    def test_microbench_returns_positive_costs(self):
        out = pt.run_microbench("cpu")
        for key in ("backup_us_per_step", "node_alloc_us", "valid_moves_us",
                    "clone_us", "clone_move_us", "release_us_per_node"):
            self.assertGreater(out[key], 0.0, key)
        ks = out["best_child_us_by_k"]
        self.assertGreater(ks[20], ks[2],
                           "best_child must cost more with more children")

    def test_backup_cost_scales_with_depth(self):
        m = MCTS(model=None, device="cpu", n_sims=1)
        def chain(d):
            node = MCTSNode(to_play=X)
            for i in range(d):
                nxt = MCTSNode(parent=node, prior=0.1, move=i, to_play=O)
                node.children[i] = nxt
                node = nxt
            return node
        shallow, deep = chain(2), chain(40)
        t_s = pt._timeit(lambda: m._backup(shallow, 0.1), 5000)
        t_d = pt._timeit(lambda: m._backup(deep, 0.1), 5000)
        self.assertGreater(t_d, t_s * 2.0)


class TestReconciliation(unittest.TestCase):
    def test_interpolates_best_child_cost_at_the_observed_branching(self):
        bench = {"best_child_us_by_k": {2: 2.0, 4: 4.0, 20: 20.0},
                 "backup_us_per_step": 0.1, "node_alloc_us": 0.2,
                 "valid_moves_us": 1.0, "clone_move_us": 2.0,
                 "release_us_per_node": 0.05}
        counts = {"mean_children_scanned": 3.0, "best_child_per_move": 1000.0,
                  "backup_steps_per_move": 100.0,
                  "nodes_created_per_move": 100.0,
                  "valid_calls_per_move": 10.0, "probes_per_move": 10.0,
                  "nodes_released_per_move": 100.0}
        shares = {"child scoring / best-child": 0.5}
        model = pt.reconcile(shares, counts, bench, ms_per_move=6.0)
        # k=3 sits halfway between the 2 and 4 rows: 3.0 us x 1000 = 3.0 ms
        self.assertAlmostEqual(model["child scoring / best-child"], 3.0,
                               places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
