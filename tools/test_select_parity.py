"""Tests for native PUCT selection and its parity oracle (#45a).

    python -m tools.test_select_parity

THE MOST IMPORTANT TEST IN THIS FILE is `TestTheOracleCanFail`. A parity oracle
that cannot report a disagreement is not evidence of anything, and every other
green result here is conditional on it. It plants a deliberately wrong native
selector and requires the oracle to catch it, to keep playing the Python
answer, and to say which child each side wanted.

The second most important is `TestReleaseStillBreaksTheCycle`. `node.kids` is a
SECOND list of strong references to the children that `release()` drops from
`node.children`. Forgetting it would leave the parent<->child cycle intact,
which does not break anything a functional test would notice -- it silently
hands the tree back to the cyclic collector and returns the p99 latency to
1037 ms, which is a promotion-gate failure discovered a day later on a match.
"""
import gc
import math
import sys
import unittest

import numpy as np

from agents import native_select as ns
from agents.mcts import MCTS, MCTSNode, TreeReuseSearcher
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from tools import engine_registry as reg
from tools import select_parity as sp


C_PUCT = 1.5


def _mcts(solve=True, mirror=True):
    m = MCTS(None, "cpu", n_sims=1, c_puct=C_PUCT, solve=solve)
    m._mirror = mirror
    m.native_select = mirror
    return m


def _expand(node, moves, priors, solve=True, mirror=True):
    """Build children through the ENGINE's own builder, not by hand.

    A test that assembles the mirror itself is testing its own assembly.
    """
    m = _mcts(solve=solve, mirror=mirror)
    row = np.zeros(81, dtype=np.float32)
    for mv, p in zip(moves, priors):
        row[mv] = p
    m._build_children_mirrored(node, list(moves), row, to_play_of(node))
    return m


def to_play_of(node):
    return 2 if node.to_play == 1 else 1


class TestChildArray(unittest.TestCase):

    def test_columns_start_at_the_python_defaults(self):
        ca = ns.ChildArray([3, 4], [0.5, 0.5], C_PUCT, True)
        self.assertEqual(list(ca.N), [0, 0])
        self.assertEqual(list(ca.W), [0.0, 0.0])
        # SOLVED_NONE encodes Python's None, and it must be outside {-1,0,1}
        # or a real proof would be indistinguishable from an absent one.
        self.assertNotIn(ns.SOLVED_NONE, (-1, 0, 1))
        self.assertEqual(list(ca.S), [ns.SOLVED_NONE, ns.SOLVED_NONE])

    def test_the_columns_are_views_not_copies(self):
        """Mirror maintenance writes through these; a copy would make every
        write a no-op and every parity check pass on stale-but-consistent
        data."""
        ca = ns.ChildArray([3, 4], [0.5, 0.5], C_PUCT, True)
        n, w = ca.N, ca.W
        n[1] = 17
        w[1] = -2.5
        self.assertEqual(list(ca.N), [0, 17])
        self.assertEqual(list(ca.W), [0.0, -2.5])
        # And a second fetch sees the same buffer.
        self.assertEqual(int(ca.N[1]), 17)

    def test_dtypes_are_pinned(self):
        ca = ns.ChildArray([3], [0.5], C_PUCT, True)
        self.assertEqual(ca.N.dtype, np.int32)
        self.assertEqual(ca.W.dtype, np.float64)
        self.assertEqual(ca.S.dtype, np.int8)

    def test_prior_is_held_as_a_double_bit_exactly(self):
        """Python stores `float(row[mv])` -- a float32 value in a double. The
        mirror must hold that double, not a second float32 round trip."""
        p = float(np.float32(0.1))
        ca = ns.ChildArray([3], [p], C_PUCT, True)
        self.assertEqual(float(ca.prior[0]), p)
        # A denormal survives too.
        ca2 = ns.ChildArray([3], [5e-324], C_PUCT, True)
        self.assertEqual(float(ca2.prior[0]), 5e-324)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(Exception):
            ns.ChildArray([3, 4], [0.5], C_PUCT, True)

    def test_empty_best_raises_rather_than_returning_minus_one(self):
        ca = ns.ChildArray([], [], C_PUCT, True)
        with self.assertRaises(Exception):
            ca.best(0)

    def test_scores_match_the_python_expression_bit_for_bit(self):
        moves = [3, 40, 76]
        priors = [0.1, 0.65, 0.25]
        Ns, Ws = [0, 7, 3], [0.0, -1.25, 2.5]
        ca = ns.ChildArray(moves, priors, C_PUCT, True)
        ca.load(Ns, Ws, [ns.SOLVED_NONE] * 3)
        parent_N = 10
        got = ca.scores(parent_N)
        for j in range(3):
            q = Ws[j] / Ns[j] if Ns[j] > 0 else 0.0
            u = C_PUCT * priors[j] * math.sqrt(parent_N) / (1 + Ns[j])
            self.assertEqual(float(got[j]), -q + u,
                             f"child {j} score differs in the last bits")


class TestTheOracleCanFail(unittest.TestCase):
    """Every other result in this file is conditional on these."""

    def _tree(self):
        root = MCTSNode(to_play=1)
        m = _expand(root, [3, 4, 5], [0.2, 0.5, 0.3])
        for j, c in enumerate(root.kids):
            c.N = j + 1
            c.W = 0.25 * j
            root.selN[j] = c.N
            root.selW[j] = c.W
        root.N = 6
        return m, root

    def test_a_wrong_native_selector_is_caught(self):
        m, root = self._tree()
        truth = m._best_child(root)          # python path via a real search obj

        class Wrong:
            """Always answers with the LAST child."""
            def __init__(self, real):
                self._real = real
                self.move = real.move
                self.prior = real.prior

            def best(self, parent_N):
                return len(self._real.N) - 1

            def scores(self, parent_N):
                return self._real.scores(parent_N)

        # Make sure the fixture is one where "last" is actually wrong.
        self.assertIsNot(truth, root.kids[-1])
        root.sel = Wrong(root.sel)

        sh = sp.Shadow(m, deep=False)
        with sh:
            got = m._best_child(root)
        self.assertEqual(len(sh.mismatches), 1)
        self.assertIs(got, truth,
                      "shadow must return the PYTHON answer so the trajectory "
                      "stays canonical after a miss")
        rec = sh.mismatches[0]
        self.assertEqual(rec["python_move"], truth.move)
        self.assertEqual(rec["native_move"], root.kids[-1].move)
        self.assertEqual(len(rec["children"]), 3)

    def test_a_drifted_W_column_is_caught_even_when_the_index_agrees(self):
        m, root = self._tree()
        before = m._best_child(root)
        # One ULP on a child that is not going to be chosen anyway.
        loser = 0 if root.kids.index(before) != 0 else 1
        root.selW[loser] = math.nextafter(float(root.selW[loser]), 1e9)
        sh = sp.Shadow(m, deep=True, deep_every=1)
        with sh:
            got = m._best_child(root)
        self.assertIs(got, before, "the fixture must not change the index")
        self.assertEqual(len(sh.mismatches), 0)
        self.assertEqual(len(sh.column_faults), 1,
                         "a one-ULP W drift that does not move the argmax is "
                         "exactly what index parity alone cannot see")
        self.assertTrue(any("W mirror" in f
                            for f in sh.column_faults[0]["faults"]))

    def test_a_reordered_kids_list_is_caught(self):
        _m, root = self._tree()
        root.kids = list(reversed(root.kids))
        _n, faults = sp.verify_subtree(root)
        self.assertTrue(faults)
        joined = "; ".join(faults[0]["faults"])
        self.assertIn("ORDER differs", joined)

    def test_a_stale_solved_column_is_caught(self):
        _m, root = self._tree()
        root.kids[1].solved = -1          # node updated, mirror not
        _n, faults = sp.verify_subtree(root)
        self.assertTrue(any("S mirror" in f
                            for rec in faults for f in rec["faults"]))

    def test_a_clean_tree_reports_no_faults(self):
        _m, root = self._tree()
        n, faults = sp.verify_subtree(root)
        self.assertEqual(faults, [])
        self.assertEqual(n, 1)


class TestFixtureCoverage(unittest.TestCase):

    REQUIRED = {
        "proven winning child": "proven_win_beats_everything",
        "all but one child refuted": "all_but_one_refuted",
        "unvisited children": "all_unvisited_equal_priors_first_wins",
        "N=0 alongside visited Q=0": "zero_visits_beside_visited_zero_q",
        "very large N": "very_large_n",
        "very small prior": "very_small_prior",
        "equal PUCT scores": "exact_tie_two_identical_children",
        "near-equal floating scores": "one_ulp_higher_second",
        "every child refuted": "every_child_refuted_falls_through",
    }

    def test_every_named_case_from_the_brief_exists(self):
        have = {c.name for c in sp.fixtures()}
        for label, name in self.REQUIRED.items():
            self.assertIn(name, have, f"missing fixture for '{label}'")

    def test_all_fixtures_agree(self):
        for c in sp.fixtures():
            with self.subTest(case=c.name):
                ok, p, n = sp.check(c)
                self.assertTrue(ok, f"{c.name}: python {p} native {n}\n"
                                    f"{c.describe()}")

    def test_the_tie_fixture_would_distinguish_the_two_tie_rules(self):
        """A tie case only proves something if first-in-order and
        lowest-move-index give DIFFERENT answers on it."""
        c = [x for x in sp.fixtures()
             if x.name == "tie_with_descending_move_order"][0]
        self.assertNotEqual(c.moves, sorted(c.moves))
        self.assertEqual(c.moves[0], max(c.moves))


class TestOrdering(unittest.TestCase):

    def test_send_anywhere_order_is_not_ascending_and_is_preserved(self):
        """The mirror must store rule_utl_valid_moves order, which is
        mini-major. A native side that sorted, or that scanned an 81-cell
        mask, would break ties differently several times a game."""
        rng = np.random.default_rng(45)
        found = 0
        for _ in range(4000):
            s = GameState()
            for _ply in range(int(rng.integers(4, 30))):
                v = rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
                if not v or s.winner is not None:
                    break
                s.make_move(int(v[int(rng.integers(len(v)))]))
            v = rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
            if s.winner is not None or len(v) < 2 or v == sorted(v):
                continue
            found += 1
            node = MCTSNode(to_play=s.player)
            _expand(node, v, [1.0 / len(v)] * len(v))
            self.assertEqual([int(x) for x in node.sel.move], list(v))
            self.assertNotEqual([int(x) for x in node.sel.move], sorted(v))
            self.assertEqual([c.move for c in node.kids], list(v))
            self.assertEqual(list(node.children.keys()), list(v))
            if found >= 12:
                break
        self.assertGreaterEqual(found, 5,
                                "no unsorted send-anywhere position was "
                                "generated, so this test proved nothing")

    def test_cpp_valid_moves_agrees_with_the_python_generator_order(self):
        """The mirror inherits the Python order today. If the C++ engine's own
        ordering ever diverges, a future native expansion path would silently
        adopt a different tie-break."""
        rng = np.random.default_rng(451)
        checked = 0
        for _ in range(600):
            s = GameState()
            for _ply in range(int(rng.integers(2, 40))):
                v = rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
                if not v or s.winner is not None:
                    break
                s.make_move(int(v[int(rng.integers(len(v)))]))
            if s.winner is not None:
                continue
            py = rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
            if hasattr(s, "valid_moves"):
                self.assertEqual(list(s.valid_moves()), list(py))
                checked += 1
        self.assertGreater(checked, 100)


class TestMirrorMaintenance(unittest.TestCase):

    def _small_tree(self, solve=True):
        root = MCTSNode(to_play=1)
        m = _expand(root, [3, 4, 5], [0.2, 0.5, 0.3], solve=solve)
        mid = root.kids[1]
        _expand(mid, [40, 41], [0.6, 0.4], solve=solve)
        return m, root, mid

    def test_backup_writes_through_every_level(self):
        m, root, mid = self._small_tree()
        leaf = mid.kids[0]
        m._backup(leaf, 0.75)
        self.assertEqual(sp.verify_subtree(root)[1], [])
        self.assertEqual(int(mid.selN[0]), 1)
        self.assertEqual(float(mid.selW[0]), 0.75)
        self.assertEqual(int(root.selN[1]), 1)
        self.assertEqual(float(root.selW[1]), -0.75)
        self.assertEqual(root.N, 1)

    def test_backup_repairs_a_corrupted_mirror(self):
        """Write-through, not accumulate. A native `+= v` running beside the
        Python one would be a second accumulator with a second answer; copying
        the post-update value means the mirror can be stale but never wrong
        for long."""
        m, root, mid = self._small_tree()
        root.selW[1] = 12345.0
        root.selN[1] = 999
        m._backup(mid, 0.5)
        self.assertEqual(int(root.selN[1]), mid.N)
        self.assertEqual(float(root.selW[1]), mid.W)

    def test_virtual_loss_apply_undo_is_not_the_identity(self):
        """Why the mirror must receive the same UPDATE SEQUENCE rather than be
        reconstructed from a running total.

        Virtual loss adds 1.0 and later subtracts it, and that is NOT the
        identity on a float -- not at exotic magnitudes but at the ordinary
        ones this tree lives in. A child with W = 0.03, which is any lightly
        visited node, comes back as 0.030000000000000027. Both the node and
        the mirror must land on that same wrong value, which they do because
        both receive the same two operations in the same order.
        """
        for w in (0.03, 0.1, 1e-20):
            self.assertNotEqual((w + 1.0) - 1.0, w, f"w={w!r}")

    def test_the_mirror_survives_a_virtual_loss_round_trip(self):
        m, root, mid = self._small_tree()
        c = root.kids[0]
        c.W = 0.03
        root.selW[0] = c.W
        # Exactly what _run_wave does, apply then undo.
        c.N += 1
        c.W += m._VL
        root.selN[0], root.selW[0] = c.N, c.W
        c.N -= 1
        c.W -= m._VL
        root.selN[0], root.selW[0] = c.N, c.W
        self.assertNotEqual(c.W, 0.03, "the fixture must actually drift")
        self.assertEqual(float(root.selW[0]), c.W)
        self.assertEqual(sp.verify_subtree(root)[1], [])

    def test_mark_solved_writes_the_parent_column(self):
        m, root, mid = self._small_tree()
        m._mark_solved(mid, -1)
        self.assertEqual(int(root.selS[1]), -1)
        self.assertEqual(sp.verify_subtree(root)[1], [])

    def test_proof_propagation_writes_every_column_it_touches(self):
        m, root, mid = self._small_tree()
        # Every reply to `mid` refuted -> mid is proven lost -> that is a win
        # for root's mover, which propagate must record on root's column. The
        # LAST child goes through _mark_solved rather than being set directly:
        # _mark_solved returns early on an already-solved node, so pre-setting
        # all of them would make this test pass without propagating anything.
        for j, c in enumerate(mid.kids[:-1]):
            c.solved = 1
            mid.selS[j] = 1
        m._mark_solved(mid.kids[-1], 1)
        self.assertEqual(mid.solved, -1)
        self.assertEqual(int(root.selS[1]), -1)
        self.assertEqual(sp.verify_subtree(root)[1], [])

    def test_mark_terminal_children_writes_the_solved_column(self):
        s = GameState()
        for mv in (40, 4, 36, 0, 41, 13, 38):
            s.make_move(mv)
        v = rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
        node = MCTSNode(to_play=s.player)
        m = _expand(node, v, [1.0 / len(v)] * len(v))
        m._mark_terminal_children(node, s)
        self.assertEqual(sp.verify_subtree(node)[1], [])

    def test_root_without_a_parent_is_not_written_through(self):
        m, root, _mid = self._small_tree()
        m._backup(root, 0.5)             # root.parent is None
        self.assertEqual(root.N, 1)
        self.assertEqual(sp.verify_subtree(root)[1], [])

    def test_an_adopted_root_keeps_its_mirror(self):
        _m, root, mid = self._small_tree()
        sel_before = mid.sel
        mid.parent = None                # what _adopt does
        self.assertIs(mid.sel, sel_before)
        self.assertEqual(sp.verify_subtree(mid)[1], [])


class TestReleaseStillBreaksTheCycle(unittest.TestCase):
    """`node.kids` holds the same strong references `node.children` does."""

    class Tracked(MCTSNode):
        __slots__ = ()
        DEAD = []

        def __del__(self):
            TestReleaseStillBreaksTheCycle.Tracked.DEAD.append(1)

    def _build(self, mirror):
        """A two-level tree of Tracked nodes, mirrored the way the engine
        builds one."""
        m = _mcts(mirror=mirror)
        root = self.Tracked(to_play=1)
        row = np.zeros(81, dtype=np.float32)
        for mv in (3, 4, 5):
            row[mv] = 1 / 3

        def expand(parent, moves):
            if mirror:
                kids, priors = [], []
                for j, mv in enumerate(moves):
                    c = self.Tracked(parent=parent, prior=float(row[mv]),
                                     move=mv, to_play=2)
                    c.cidx = j
                    parent.children[mv] = c
                    kids.append(c)
                    priors.append(float(row[mv]))
                sel = ns.ChildArray(list(moves), priors, C_PUCT, True)
                parent.kids = kids
                parent.selN, parent.selW, parent.selS = sel.N, sel.W, sel.S
                parent.sel = sel
            else:
                for mv in moves:
                    parent.children[mv] = self.Tracked(
                        parent=parent, prior=float(row[mv]), move=mv,
                        to_play=2)
            return list(parent.children.values())

        first = expand(root, (3, 4, 5))
        expand(first[1], (3, 4, 5))
        return root

    def _count_released(self, mirror):
        # Collect FIRST, then clear. The other tests in this class leave
        # Tracked nodes for the collector, and a collect after the clear would
        # count them here -- which is how this read 12 for a 7-node tree.
        gc.collect()
        self.Tracked.DEAD.clear()
        was = gc.isenabled()
        gc.disable()
        try:
            root = self._build(mirror)
            TreeReuseSearcher.release(root)
            del root
            # NO gc.collect() here on purpose: if release() did its job the
            # tree dies by refcount alone, which is the entire point of it.
            return len(self.Tracked.DEAD)
        finally:
            if was:
                gc.enable()
            gc.collect()

    def test_mirrored_tree_dies_by_refcount_with_the_collector_off(self):
        self.assertEqual(self._count_released(mirror=True), 7)

    def test_unmirrored_tree_still_dies_by_refcount(self):
        self.assertEqual(self._count_released(mirror=False), 7)

    def test_release_clears_the_mirror_slots(self):
        root = self._build(mirror=True)
        nodes, mirrored = [root], []
        stack = [root]
        while stack:
            n = stack.pop()
            if n.sel is not None:
                mirrored.append(n)
            stack.extend(n.children.values())
            nodes.extend(n.children.values())
        # An unexpanded leaf never had `kids` set at all, so the release guard
        # is `sel is not None` and only expanded nodes are checked for it.
        self.assertEqual(len(mirrored), 2)
        TreeReuseSearcher.release(root)
        for n in nodes:
            self.assertIsNone(n.sel)
            self.assertEqual(n.children, {})
            self.assertIsNone(n.parent)
        for n in mirrored:
            self.assertIsNone(n.kids)
            self.assertIsNone(n.selN)
            self.assertIsNone(n.selW)
            self.assertIsNone(n.selS)

    def test_release_keeps_the_kept_subtree_and_its_mirror(self):
        root = self._build(mirror=True)
        keep = list(root.children.values())[1]
        keep_kids = keep.kids
        TreeReuseSearcher.release(root, keep=keep)
        self.assertIsNotNone(keep.sel)
        self.assertIs(keep.kids, keep_kids)
        self.assertEqual(sp.verify_subtree(keep)[1], [])


class TestIncumbentPathUntouched(unittest.TestCase):

    def test_nodes_have_no_mirror_when_native_select_is_off(self):
        m = _mcts(mirror=False)
        node = MCTSNode(to_play=1)
        row = np.zeros(81, dtype=np.float32)
        row[3] = row[4] = 0.5
        # The non-mirrored builder is the inline loop in _expand_children;
        # exercise it through the same shape of call.
        for mv in (3, 4):
            node.children[mv] = MCTSNode(parent=node, prior=float(row[mv]),
                                         move=mv, to_play=2)
        self.assertIsNone(node.sel)
        self.assertIsNotNone(m._best_child(node))

    def test_a_fresh_node_reports_no_mirror(self):
        self.assertIsNone(MCTSNode().sel)

    def test_best_child_falls_to_python_when_sel_is_none(self):
        m = _mcts(mirror=True)
        root = MCTSNode(to_play=1)
        _expand(root, [3, 4], [0.4, 0.6])
        native = m._best_child(root)
        root.sel = None
        python = m._best_child(root)
        self.assertIs(native, python)


class TestRegistryWiring(unittest.TestCase):

    def test_select_is_pinned_in_every_engine(self):
        for name, spec in reg.ENGINES.items():
            if reg.is_raw(name):
                continue
            with self.subTest(engine=name):
                self.assertIn("select", spec)

    def test_only_the_declared_candidate_enables_it(self):
        on = {n for n in reg.ENGINES if reg.ENGINES[n].get("select") == "1"}
        self.assertEqual(on, {"pocket_sel"})
        self.assertFalse(on & reg.ANCHOR_ROLES)
        self.assertNotIn("final", on)
        self.assertNotIn("pocket_graph", on)

    def test_the_candidate_differs_in_the_flag_and_the_reserve_it_forces(self):
        """Two keys, for the same reason `pocket_r35` -> `pocket_graph` needed
        two. The reserve is not a free parameter of the comparison: a faster
        search builds a bigger tree, `release()` walks it outside the search's
        own deadline, and the engine misses the frozen p99 unless the reserve
        grows with it. Measured, not assumed -- at reserve 50 the gate failed
        at p99 1027.9 with caller-side overhead p99 78.99."""
        a, b = reg.ENGINES["pocket_graph"], reg.ENGINES["pocket_sel"]
        diff = {k for k in a if a[k] != b[k]} - {"name"}
        self.assertEqual(diff, {"select", "reserve"})
        self.assertEqual(b["reserve"], "95")

    def test_the_reserve_reaches_the_built_search(self):
        """The bug this guards is real and already happened once: `_engine()`
        used to force the reserve from the budget table AFTER applying
        overrides, so a declared reserve was silently discarded."""
        from tools.arena_1s import parse_spec
        self.assertEqual(parse_spec(reg.spec_of("pocket_sel"))["reserve"],
                         "95")

    def test_resolved_config_carries_native_select(self):
        """A flag that changes how much search fits in the clock is
        configuration, and configuration that is not fingerprinted can move
        under a published number without tripping the guard."""
        class _Stub:
            ckpt = reg.POCKET
            arch = "squeeze"
            net_info = {"params": 172389, "value_tanh": True}
            reuse = True

        stub = _Stub()
        stub.mcts = MCTS(None, "cpu", n_sims=1, time_budget_ms=1000)
        off = reg.resolved_config(stub)
        self.assertIn("native_select", off)
        self.assertFalse(off["native_select"])
        stub.mcts.native_select = True
        on = reg.resolved_config(stub)
        self.assertTrue(on["native_select"])
        self.assertNotEqual(reg.fingerprint(off), reg.fingerprint(on))


class TestSweepAndFuzzAreRealCoverage(unittest.TestCase):
    """A sweep that never produces a tie, or a fuzz whose cases are all
    trivial, would pass forever without testing the tie-break."""

    def test_the_sweep_produces_exact_ties(self):
        ties = 0
        for i, c in enumerate(sp.sweep_cases(2)):
            ca = ns.ChildArray(c.moves, c.priors, c.c_puct, c.solve)
            ca.load(c.N, c.W, [sp.encode_solved(s) for s in c.S])
            sc = list(ca.scores(c.parent_N))
            if len(set(sc)) < len(sc):
                ties += 1
            if i > 4000:
                break
        self.assertGreater(ties, 200, "the sweep grids are not colliding")

    def test_the_fuzz_produces_exact_ties_and_solved_children(self):
        ties = solved = 0
        for c in sp.fuzz_cases(3000, seed=1):
            ca = ns.ChildArray(c.moves, c.priors, c.c_puct, c.solve)
            ca.load(c.N, c.W, [sp.encode_solved(s) for s in c.S])
            sc = list(ca.scores(c.parent_N))
            if len(set(sc)) < len(sc):
                ties += 1
            if any(s is not None for s in c.S):
                solved += 1
        self.assertGreater(ties, 100)
        self.assertGreater(solved, 1000)

    def test_a_short_sweep_and_fuzz_agree(self):
        for c in sp.fuzz_cases(20000, seed=7):
            ok, p, n = sp.check(c)
            if not ok:
                self.fail(f"{c.name}: python {p} native {n}\n{c.describe()}")


class TestSearchLevelParity(unittest.TestCase):
    """Whole searches, at a fixed simulation count, on CPU.

    THIS IS THE GATE SHADOW MODE CANNOT BE. Shadow returns the PYTHON answer on
    every call so that a disagreement cannot fork the tree -- which means the
    search never actually walks the trajectory the native selector chose. Seven
    million agreeing comparisons still leave one thing unobserved: what the
    engine does when the native answer is the one it follows.

    So this runs the search for real, both ways, and requires the visit-count
    policy to come back bit-identical. CPU and a fixed simulation count are
    both load-bearing: CUDA reductions are not bit-reproducible run to run, and
    under a clock the two arms complete different amounts of work by design.
    """

    N_SIMS = 240

    @classmethod
    def setUpClass(cls):
        from tools.arena_1s import load_net
        cls.model, _info = load_net(reg.POCKET, "squeeze", "cpu")

    def _mcts(self, native):
        return MCTS(self.model, "cpu", n_sims=self.N_SIMS, c_puct=C_PUCT,
                    wave_size=8, solve=True, native_select=native)

    @staticmethod
    def _states():
        """A bare opening, a mid-game position, and a send-anywhere position --
        the last because that is where child order is not ascending."""
        out = [GameState()]
        for line in ((40, 4, 36, 0, 41, 13, 38),
                     (40, 36, 0, 4, 38, 20, 22, 14, 45, 8, 76, 40, 39)):
            s = GameState()
            for mv in line:
                if s.winner is not None:
                    break
                s.make_move(mv)
            out.append(s)
        return out

    def test_fixed_sim_searches_return_identical_policies(self):
        import torch
        for i, state in enumerate(self._states()):
            with self.subTest(position=i):
                pi_a, root_a = None, None
                stats = []
                for native in (False, True):
                    m = self._mcts(native)
                    with torch.no_grad():
                        pi, root = m.search(state.clone())
                    stats.append((m.stat_sims, m.stat_nn_evals,
                                  m.stat_expansions, m.stat_probes, root.N,
                                  {mv: c.N for mv, c in root.children.items()},
                                  {mv: c.solved
                                   for mv, c in root.children.items()}))
                    if native:
                        # Bit-identical, not close: the policy is built from
                        # integer visit counts, so any difference at all means
                        # the two searches went somewhere different.
                        self.assertTrue(np.array_equal(pi_a, pi),
                                        "visit policy differs")
                        self.assertEqual(stats[0], stats[1])
                        self.assertEqual(sp.verify_subtree(root)[1], [])
                    else:
                        pi_a, root_a = pi, root
                        self.assertIsNone(root.sel)
                    TreeReuseSearcher.release(root)

    def test_reused_trees_stay_identical_across_several_moves(self):
        """Re-rooting is where the mirror has to survive an operation it does
        not perform: `_adopt` keeps a subtree and `release` destroys the rest.
        A search that is identical move one and drifts by move four would pass
        every other test here."""
        import torch
        seqs = []
        for native in (False, True):
            m = self._mcts(native)
            searcher = TreeReuseSearcher(m, enabled=True)
            state = GameState()
            rec = []
            for ply in range(8):
                if state.winner is not None:
                    break
                with torch.no_grad():
                    pi, root = searcher.search(state.clone())
                rec.append((pi.copy(), root.N, m.stat_nn_evals,
                            tuple(sorted((mv, c.N)
                                         for mv, c in root.children.items()))))
                mv = int(pi.argmax())
                state.make_move(mv)
                # The opponent replies with the lowest legal move: fully
                # deterministic, and it keeps the two-ply shape `_adopt` needs.
                if state.winner is None:
                    state.make_move(min(rule_utl_valid_moves(
                        state.board, state.last_move, state.mini_winners)))
            searcher.reset()
            seqs.append(rec)
        self.assertEqual(len(seqs[0]), len(seqs[1]))
        self.assertGreaterEqual(len(seqs[0]), 4, "too few plies to test reuse")
        for ply, (a, b) in enumerate(zip(*seqs)):
            with self.subTest(ply=ply):
                self.assertTrue(np.array_equal(a[0], b[0]))
                self.assertEqual(a[1:], b[1:])


class TestLoader(unittest.TestCase):

    def test_require_returns_the_class_when_present(self):
        self.assertIs(ns.require(), ns.ChildArray)

    def test_solved_none_is_defined_even_without_the_extension(self):
        """The encoding is part of the format, not of the extension."""
        self.assertEqual(ns.SOLVED_NONE, 2)

    def test_mcts_refuses_rather_than_degrading(self):
        real = ns.HAVE_NATIVE_SELECT
        try:
            ns.HAVE_NATIVE_SELECT = False
            with self.assertRaises(RuntimeError):
                MCTS(None, "cpu", native_select=True)
        finally:
            ns.HAVE_NATIVE_SELECT = real


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
