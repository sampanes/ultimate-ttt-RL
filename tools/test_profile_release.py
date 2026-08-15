"""Tests for the #46a release characterisation.

THE RISK THIS FILE EXISTS FOR is `counting_release`. It is a hand copy of
`TreeReuseSearcher.release` carrying one extra tally, taken because the count
has to come from inside the walk -- and a copy of production code drifts from
production code silently. Every number #46a reports about node counts and about
the ms-vs-nodes slope comes out of that copy, so it is checked against the real
function on real mirrored trees, field by field, and the check is itself checked
by planting defects that it must catch.

The second risk is the patch: `release` is a staticmethod and `_adopt` is a
bound method, patched onto an INSTANCE so the opponent's searcher is untouched.
A patch that silently failed to land would report a release cost of zero, which
reads like a spectacular result rather than like a bug.

    python -m tools.test_profile_release
"""

import gc
import unittest

import numpy as np

from agents.mcts import MCTS, MCTSNode, TreeReuseSearcher
from tools import engine_registry
from tools.profile_release import (counting_release, count_subtree, depth_for,
                                   fit_slope, instrument, measure_bytes_per_node,
                                   memory_now, structural_bytes,
                                   _synthetic_tree, Probe)

BRANCHING = 5
DEPTH = 4


def build(mirrored=True):
    """A tree in production's own shape, with or without the native mirror."""
    m = MCTS(model=None, device="cpu", n_sims=1, c_puct=1.5, wave_size=8,
             solve=True, batched_expand=True, native_select=mirrored)
    root, _leaves = _synthetic_tree(m, DEPTH, BRANCHING)
    return root


def snapshot(root):
    """Every field `release` is responsible for, in a stable order.

    Taken BEFORE the walk, because the point of the walk is that afterwards
    there is no tree left to traverse. The node objects themselves are held so
    they can be re-read after.
    """
    order, stack = [], [root]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(n.children.values())
    return order


def fields(n):
    return (len(n.children), n.parent is None, n.sel is None,
            getattr(n, "kids", "unset") is None,
            getattr(n, "selN", "unset") is None,
            getattr(n, "selW", "unset") is None,
            getattr(n, "selS", "unset") is None,
            n.N, n.W, n.move, n.to_play)


class TestTheCopyMatchesProduction(unittest.TestCase):
    """`counting_release` against `TreeReuseSearcher.release`, on real trees."""

    def test_identical_object_graph_after_the_walk(self):
        a, b = build(), build()
        na, nb = snapshot(a), snapshot(b)
        self.assertEqual(len(na), len(nb))
        self.assertGreater(len(na), 500, "too small a tree to prove anything")

        TreeReuseSearcher.release(a)
        counting_release(b)

        for i, (x, y) in enumerate(zip(na, nb)):
            self.assertEqual(fields(x), fields(y),
                             "node %d differs after release" % i)

    def test_the_count_equals_an_independent_walk(self):
        t = build()
        expected = count_subtree(t)
        out = [0, 0]
        got = counting_release(t, None, out)
        self.assertEqual(got, expected)
        self.assertEqual(out[0], expected)

    def test_the_expanded_tally_is_the_mirrored_subset(self):
        """`expanded` must be exactly the nodes carrying a ChildArray: the
        memory projection prices those an order of magnitude above a leaf."""
        t = build()
        want = sum(1 for n in snapshot(t) if n.children)
        want_mirrored = sum(1 for n in snapshot(t) if n.sel is not None)
        self.assertEqual(want, want_mirrored)
        out = [0, 0]
        counting_release(t, None, out)
        self.assertEqual(out[1], want)

    def test_keep_is_spared_identically(self):
        a, b = build(), build()
        ka = list(a.children.values())[2]
        kb = list(b.children.values())[2]
        na, nb = snapshot(a), snapshot(b)
        kept_a, kept_b = snapshot(ka), snapshot(kb)

        TreeReuseSearcher.release(a, keep=ka)
        out = [0, 0]
        counting_release(b, kb, out)

        for x, y in zip(na, nb):
            self.assertEqual(fields(x), fields(y))
        # The retained subtree is wholly untouched by both.
        for x in kept_a + kept_b:
            self.assertIsNotNone(x.sel if x.children else None) \
                if x.children else None
            if x.children:
                self.assertGreater(len(x.children), 0)
        self.assertEqual(out[0], len(na) - len(kept_a))

    def test_unmirrored_trees_too(self):
        """The engine being characterised has the mirror on, but `release`
        serves both and a copy that only worked for one would be a live trap
        the day someone profiles `pocket_graph`."""
        a = build(mirrored=False)
        b = build(mirrored=False)
        na, nb = snapshot(a), snapshot(b)
        TreeReuseSearcher.release(a)
        counting_release(b)
        for x, y in zip(na, nb):
            self.assertEqual(fields(x), fields(y))


class TestTheComparisonCanFail(unittest.TestCase):
    """Plant defects. A comparison that cannot fail proves nothing."""

    @staticmethod
    def _drifted_leaves_kids(root, keep=None, out=None):
        """The #45a bug, re-introduced: `kids` left in place. It keeps the
        parent<->child cycle alive and is invisible to any functional check."""
        n_nodes = 0
        stack = [root]
        while stack:
            n = stack.pop()
            if n is keep:
                continue
            n_nodes += 1
            kids = n.children
            n.children = {}
            n.parent = None
            if n.sel is not None:
                n.sel = None
                n.selN = None
                n.selW = None
                n.selS = None
            stack.extend(kids.values())
        if out is not None:
            out[0] = n_nodes
        return n_nodes

    @staticmethod
    def _drifted_skips_keep_check(root, keep=None, out=None):
        n_nodes = 0
        stack = [root]
        while stack:
            n = stack.pop()
            n_nodes += 1
            kids = n.children
            n.children = {}
            n.parent = None
            stack.extend(kids.values())
        if out is not None:
            out[0] = n_nodes
        return n_nodes

    def test_a_copy_that_forgets_kids_is_caught(self):
        a, b = build(), build()
        na, nb = snapshot(a), snapshot(b)
        TreeReuseSearcher.release(a)
        self._drifted_leaves_kids(b)
        self.assertTrue(any(fields(x) != fields(y) for x, y in zip(na, nb)),
                        "the field comparison did not notice `kids` surviving")

    def test_a_copy_that_ignores_keep_is_caught(self):
        a, b = build(), build()
        ka = list(a.children.values())[0]
        kb = list(b.children.values())[0]
        kept = snapshot(ka)
        na, nb = snapshot(a), snapshot(b)
        TreeReuseSearcher.release(a, keep=ka)
        self._drifted_skips_keep_check(b, kb)
        self.assertTrue(any(fields(x) != fields(y) for x, y in zip(na, nb)),
                        "a walk that destroyed the retained subtree looked "
                        "identical to one that spared it")
        self.assertGreater(len(kept), 1)

    def test_an_off_by_one_count_is_caught(self):
        t = build()
        expected = count_subtree(t)
        out = [0, 0]
        counting_release(t, None, out)
        self.assertNotEqual(out[0], expected - 1)
        self.assertNotEqual(out[0], expected + 1)


class TestTheWalkReallyKillsTheTree(unittest.TestCase):
    """The whole design question is whether this walk can be deferred. If the
    copy left the tree cyclic, the memory projection would be measuring
    something that never actually dies."""

    def test_the_tree_dies_by_refcount_with_gc_off(self):
        dead = []

        class Tracked(MCTSNode):
            __slots__ = ("__weakref__",)

            def __del__(self):
                dead.append(1)

        m = MCTS(model=None, device="cpu", n_sims=1, c_puct=1.5, wave_size=8,
                 solve=True, batched_expand=True, native_select=True)
        row = np.full(81, 0.2, dtype=np.float32)
        valid = list(range(4))
        was = gc.isenabled()
        gc.disable()
        try:
            gc.collect()
            dead.clear()
            root = Tracked(to_play=1)
            m._build_children_mirrored(root, valid, row, 2)
            kids = list(root.children.values())
            for k in kids:
                m._build_children_mirrored(k, valid, row, 1)
            n_total = count_subtree(root)
            counting_release(root)
            del root, kids
            self.assertEqual(sum(dead), 1,
                             "only the Tracked root is instrumented; the rest "
                             "are plain MCTSNodes")
            self.assertGreater(n_total, 4)
        finally:
            if was:
                gc.enable()


class TestThePatchLands(unittest.TestCase):
    """A probe that never fires reports a cost of zero, which reads as a win."""

    class _Searcher:
        """The two attribute shapes `instrument` has to handle: a staticmethod
        on the class and a bound method on the instance."""

        def __init__(self):
            self.calls = []
            self._root = None

        release = staticmethod(TreeReuseSearcher.release)

        def _adopt(self, state):
            self.calls.append(state)
            return None, "no_tree"

    class _Player:
        def __init__(self, searcher):
            self.searcher = searcher
            self.recording = True

        def new_game(self):
            return None

        def move(self, state, move_num):
            return 0

    def _rig(self, counting=True):
        s = self._Searcher()
        p = self._Player(s)
        probe = Probe()
        return s, p, probe, instrument(p, probe, counting)

    def test_release_is_shadowed_and_recorded(self):
        s, p, probe, restore = self._rig()
        tree = build()
        n = count_subtree(tree)
        s.release(tree)
        restore()
        self.assertEqual(probe.release_nodes, [n])
        self.assertEqual(len(probe.release_ms), 1)
        self.assertGreater(probe.release_ms[0], 0.0)

    def test_adopt_is_wrapped_and_still_returns_its_pair(self):
        s, p, probe, restore = self._rig()
        node, reason = s._adopt("STATE")
        restore()
        self.assertIsNone(node)
        self.assertEqual(reason, "no_tree")
        self.assertEqual(s.calls, ["STATE"])
        self.assertEqual(len(probe.adopt_ms), 1)

    def test_warmup_moves_are_not_recorded(self):
        """`play_match` plays warmup games for real. A warmup move in the p99
        is the same defect as counting warmup in a denominator."""
        s, p, probe, restore = self._rig()
        p.recording = False
        s.release(build())
        s._adopt("STATE")
        p.recording = True
        s.release(build())
        restore()
        self.assertEqual(len(probe.release_ms), 1)
        self.assertEqual(len(probe.adopt_ms), 0)

    def test_restore_leaves_no_instance_attributes(self):
        """A bound method written back into the instance's own __dict__ is a
        reference cycle, and this tool runs with the collector off."""
        s, p, probe, restore = self._rig()
        self.assertIn("release", vars(s))
        self.assertIn("_adopt", vars(s))
        restore()
        self.assertNotIn("release", vars(s))
        self.assertNotIn("_adopt", vars(s))

    def test_the_uncounted_arm_still_calls_the_real_release(self):
        """mode=timed must run PRODUCTION's walk, not the copy -- those are the
        percentiles that get quoted."""
        s, p, probe, restore = self._rig(counting=False)
        tree = build()
        nodes = snapshot(tree)
        s.release(tree)
        restore()
        self.assertTrue(all(n.parent is None and not n.children
                            for n in nodes))
        # No count is available from the real function, and the report must not
        # silently read this zero as "nothing was retired".
        self.assertEqual(probe.release_nodes, [0])


class TestPerGameAccumulator(unittest.TestCase):
    """`game_peak` is the projected peak of a whole-game deferred queue, which
    is the number the design decision turns on."""

    def test_peak_is_the_worst_game_not_the_whole_match(self):
        p = Probe()
        for n in (10, 20, 30):
            p.game_cum += n
            p.game_peak = max(p.game_peak, p.game_cum)
        p.new_game()
        for n in (5, 5):
            p.game_cum += n
            p.game_peak = max(p.game_peak, p.game_cum)
        p.finish()
        self.assertEqual(p.game_peak, 60)
        self.assertEqual(p.per_game_cum, [60, 10])


class TestMemoryMeasurement(unittest.TestCase):

    def test_process_memory_is_readable(self):
        priv, peak = memory_now()
        self.assertGreater(priv, 8 * 1024 * 1024,
                           "a Python process with torch loaded holds more "
                           "than 8 MB; a zero here means the probe failed")
        self.assertGreaterEqual(peak, 0)

    def test_structural_accounting_covers_the_mirror(self):
        """A mirrored tree must account for strictly more than a bare one."""
        bare = build(mirrored=False)
        mirrored = build(mirrored=True)
        b_bytes, b_n = structural_bytes(bare)
        m_bytes, m_n = structural_bytes(mirrored)
        self.assertEqual(b_n, m_n)
        self.assertGreater(m_bytes, b_bytes)
        TreeReuseSearcher.release(bare)
        TreeReuseSearcher.release(mirrored)

    def test_bytes_per_node_is_in_a_plausible_band(self):
        """An MCTSNode is 16 slots plus an empty dict plus a GC header, so it
        cannot be under ~150 bytes; a mirrored tree averaged over its leaves
        cannot plausibly be over a few kilobytes. This is a smoke bound on the
        instrument, not a claim about the engine."""
        got = measure_bytes_per_node(BRANCHING, reps=2)
        self.assertGreater(got["bytes_per_node"], 120.0)
        self.assertLess(got["bytes_per_node"], 4096.0)
        self.assertGreater(got["structural_bytes_per_node"], 120.0)

    def test_depth_stays_in_a_sane_size_range(self):
        for b in (2, 4, 6, 8, 12, 30):
            d = depth_for(b, want_nodes=60000)
            total = sum(b ** i for i in range(d + 1))
            self.assertGreaterEqual(d, 1)
            self.assertLessEqual(total, 60000 * b,
                                 "branching %d overshot badly" % b)


class TestSlopeFit(unittest.TestCase):

    def test_recovers_a_known_slope(self):
        nodes = list(range(1000, 40000, 500))
        ms = [0.4 + 0.00018 * n for n in nodes]
        got = fit_slope(nodes, ms)
        self.assertAlmostEqual(got["us_per_node"], 0.18, places=3)
        self.assertAlmostEqual(got["intercept_ms"], 0.4, places=3)
        self.assertGreater(got["r2"], 0.999)

    def test_refuses_to_fit_nothing(self):
        self.assertIsNone(fit_slope([0, 0, 0], [1.0, 2.0, 3.0]))
        self.assertIsNone(fit_slope([5], [1.0]))


class TestSeedNamespace(unittest.TestCase):

    def test_release_seed_exists_and_is_unique(self):
        seeds = engine_registry.SEEDS
        self.assertIn("release", seeds)
        self.assertEqual(len(set(seeds.values())), len(seeds),
                         "two namespaces share a seed, so an instrumented run "
                         "and a scored one would play the same openings")


if __name__ == "__main__":
    unittest.main(verbosity=2)
