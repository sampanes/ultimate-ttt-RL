"""Tests for the #49 residual decomposition -- is the ladder a ladder?

The whole tool rests on one claim: the difference between two rungs is the cost
of the single operation that separates them. That claim fails silently in three
ways, and there is a test here for each.

  1. A rung stops being a superset of the one below it. Then a "step" is a
     difference between two unrelated bodies and the sign is not even
     guaranteed. `TestTheRungsAreCumulative` counts the actual primitive calls
     each rung makes rather than reading the source and believing it.
  2. The top rung stops matching production. The tool already checks this at
     run time as a ratio, but a ratio only catches a SPEED difference -- two
     loops can cost the same and mark different children. Here the marks
     themselves are compared, on a corpus that really contains terminal
     children.
  3. `reset` stops resetting. Then every rung after the first sees nodes that
     are already solved, `_solve_from_children` short-circuits, and the
     expensive rungs look cheap in exactly the direction that would sell a
     change.

The fourth thing worth breaking is the candidate path itself: `r3_raw_winner`
is what a `probe_make_move` would do, and it has to mark the SAME children as
the shipped body or the saving is being bought with a behaviour change.

    python -m tools.test_probe_cost
"""
from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from agents import native_select as _ns
from agents.mcts import MCTS, MCTSNode
from engine.constants import X, O
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from tools import probe_cost as pc


class Toy:
    def forward_both(self, x):
        b = x.shape[0]
        lg, v = torch.zeros((b, 81)), torch.zeros((b,))
        return (lg[0], v[0]) if b == 1 else (lg, v)


def an_mcts(**kw):
    kw.setdefault("native_select", _ns.HAVE_NATIVE_SELECT)
    return MCTS(model=Toy(), device="cpu", n_sims=8, c_puct=1.5,
                wave_size=8, solve=True, **kw)


def near_terminal_states(n=6, seed=17):
    """Real positions that have at least one game-ending legal move.

    The corpus MUST contain terminal children or every mark-comparison test
    below passes by agreeing that nothing was marked -- the vacuity failure
    that `tools/probe_ablation`'s own tests were built around.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(600):
        s = GameState()
        while s.winner is None:
            moves = rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
            if not moves:
                break
            for mv in moves:
                probe = s.clone()
                probe.make_move(mv)
                if probe.winner is not None:
                    out.append((s.clone(), tuple(moves),
                                tuple(1.0 / len(moves) for _ in moves)))
                    break
            if len(out) >= n:
                return out
            s.make_move(rng.choice(moves))
    raise AssertionError("could not build a near-terminal corpus")


def marks(pairs):
    """(root index, move) -> solved, for everything a rung marked."""
    out = {}
    for i, (node, _state) in enumerate(pairs):
        for mv, c in node.children.items():
            if c.solved is not None:
                out[(i, mv)] = (c.solved, c.is_terminal, c.terminal_value)
    return out


class TestTheRungsAreCumulative(unittest.TestCase):
    """Each rung must call a SUPERSET of what the rung below it calls.

    Counted, not read. A rung that quietly dropped its clone would report the
    clone as free and the step above it as enormous, and the source would still
    look right.
    """

    @classmethod
    def setUpClass(cls):
        cls.m = an_mcts()
        cls.pairs = pc.rebuild(cls.m, near_terminal_states(4))

    def counted(self, fn):
        n = {"clone": 0, "make": 0, "winner": 0, "raw": 0}
        raw_clone = GameState.__dict__["clone"]
        raw_make = GameState.__dict__["make_move"]
        raw_win = GameState.__dict__["winner"].fget
        raw_rawwin = GameState.__dict__.get("_raw_winner")
        cpp = GameState.__mro__[1]
        raw_cpp_make = cpp.__dict__["make_move"]

        def c_clone(self):
            n["clone"] += 1
            return raw_clone(self)

        def c_make(self, i):
            n["make"] += 1
            return raw_make(self, i)

        def c_win(self):
            n["winner"] += 1
            return raw_win(self)

        GameState.clone = c_clone
        GameState.make_move = c_make
        GameState.winner = property(c_win)
        try:
            pc.reset(self.pairs)
            fn(self.pairs, self.m)
        finally:
            GameState.clone = raw_clone
            GameState.make_move = raw_make
            GameState.winner = property(raw_win)
        del raw_rawwin, raw_cpp_make, cpp
        return n

    def test_the_first_rung_touches_no_state_at_all(self):
        n = self.counted(pc.r0_iterate)
        self.assertEqual((n["clone"], n["make"], n["winner"]), (0, 0, 0))

    def test_the_clone_rung_clones_once_per_child_and_nothing_else(self):
        kids = sum(len(node.children) for node, _ in self.pairs)
        n = self.counted(pc.r1_clone)
        self.assertEqual(n["clone"], kids)
        self.assertEqual(n["make"], 0)

    def test_the_native_rung_does_not_go_through_the_python_make_move(self):
        """It calls the BASE class method directly. If it went through
        `GameState.make_move` it would already be paying the tuple and the
        winner read, and the step it defines would be zero."""
        kids = sum(len(node.children) for node, _ in self.pairs)
        n = self.counted(pc.r2_make_native)
        self.assertEqual(n["clone"], kids)
        self.assertEqual(n["make"], 0)
        self.assertEqual(n["winner"], 0)

    def test_the_candidate_rung_reads_the_property_only_on_a_hit(self):
        """`r3` swaps the per-CHILD crossing for `_raw_winner` and keeps the
        `_terminal_value` call on a hit, deliberately -- see its docstring.
        So the property must vanish from the per-child path (the thing #49b
        prices) while surviving on hits (the thing #49b does not).

        `r3b` is the rung that removes it there too, and it must read the
        property exactly zero times."""
        n = self.counted(pc.r3_raw_winner)
        hits = len(marks(self.pairs))
        self.assertGreater(hits, 0, "vacuous: nothing was marked")
        self.assertEqual(n["winner"], 2 * hits)
        kids = sum(len(node.children) for node, _ in self.pairs)
        self.assertLess(n["winner"], kids)

    def test_the_inline_rung_reads_no_winner_property_at_all(self):
        n = self.counted(pc.r3b_inline_value)
        self.assertEqual(n["winner"], 0)
        self.assertGreater(len(marks(self.pairs)), 0)

    def test_the_shipped_rung_reads_the_property_twice_per_child(self):
        """Once inside `make_move` for the discarded tuple, once for the test,
        plus two more per hit from `_terminal_value`. This is the redundancy
        #49b prices; if it stops holding, the price has already moved."""
        kids = sum(len(node.children) for node, _ in self.pairs)
        n = self.counted(pc.r4_current)
        self.assertEqual(n["clone"], kids)
        self.assertEqual(n["make"], kids)
        self.assertGreaterEqual(n["winner"], 2 * kids)


class TestTheReplicaIsProduction(unittest.TestCase):
    """The top rung has to BE the shipped loop, not merely cost like it."""

    @classmethod
    def setUpClass(cls):
        cls.m = an_mcts()
        cls.corpus = near_terminal_states(6)

    def marked_by(self, fn):
        pairs = pc.rebuild(self.m, self.corpus)
        fn(pairs, self.m)
        return marks(pairs), pairs

    def test_the_corpus_is_not_vacuous(self):
        got, _ = self.marked_by(pc.r5_solve)
        self.assertGreater(len(got), 0,
                           "no child was marked: every comparison below "
                           "would pass by agreeing about nothing")

    def test_the_replica_marks_exactly_what_production_marks(self):
        mine, _ = self.marked_by(pc.r5_solve)
        theirs, _ = self.marked_by(pc.r_production)
        self.assertEqual(mine, theirs)

    def test_the_replica_solves_the_same_roots(self):
        _mine, pa_ = self.marked_by(pc.r5_solve)
        _theirs, pb_ = self.marked_by(pc.r_production)
        self.assertEqual([n.solved for n, _ in pa_],
                         [n.solved for n, _ in pb_])

    def test_the_candidate_path_marks_the_same_children(self):
        """`r3` is what a `probe_make_move` would do. If it disagreed with the
        shipped body, the crossing saving would be buying a behaviour change,
        which is the one thing #48's promotion argument does not cover."""
        cand, _ = self.marked_by(pc.r3_raw_winner)
        ship, _ = self.marked_by(pc.r4_current)
        self.assertEqual(cand, ship)

    def test_the_solve_rung_is_the_only_one_that_solves_the_root(self):
        _m4, p4 = self.marked_by(pc.r4_current)
        _m5, p5 = self.marked_by(pc.r5_solve)
        self.assertTrue(all(n.solved is None for n, _ in p4))
        self.assertTrue(any(n.solved is not None for n, _ in p5))


class TestResetReallyResets(unittest.TestCase):

    def setUp(self):
        self.m = an_mcts()
        self.pairs = pc.rebuild(self.m, near_terminal_states(4))

    def test_a_second_pass_sees_fresh_nodes(self):
        pc.r5_solve(self.pairs, self.m)
        first = marks(self.pairs)
        self.assertGreater(len(first), 0)
        pc.reset(self.pairs)
        self.assertEqual(marks(self.pairs), {})
        pc.r5_solve(self.pairs, self.m)
        self.assertEqual(marks(self.pairs), first)

    def test_the_native_mirror_is_reset_too(self):
        """The Python child and its mirror column are two copies of `solved`.
        Clearing only the Python one would leave the native selector seeing
        proofs from a previous rung."""
        if not _ns.HAVE_NATIVE_SELECT:
            self.skipTest("native selection not built")
        pc.r5_solve(self.pairs, self.m)
        dirty = any((node.selS != _ns.SOLVED_NONE).any()
                    for node, _ in self.pairs)
        self.assertTrue(dirty, "no mirror column was written: vacuous")
        pc.reset(self.pairs)
        for node, _ in self.pairs:
            self.assertTrue((node.selS == _ns.SOLVED_NONE).all())


class TestRebuiltNodesLookLikeTheEngineBuilds(unittest.TestCase):

    def setUp(self):
        self.m = an_mcts()
        self.corpus = near_terminal_states(3)
        self.pairs = pc.rebuild(self.m, self.corpus)

    def test_children_are_the_legal_moves_in_engine_order(self):
        for (node, state), (_s, moves, _p) in zip(self.pairs, self.corpus):
            legal = rule_utl_valid_moves(state.board, state.last_move,
                                         state.mini_winners)
            self.assertEqual(list(node.children), list(moves))
            self.assertEqual(list(node.children), list(legal))

    def test_the_child_to_play_is_the_opponent_of_the_state(self):
        """`_terminal_value` compares the winner against `child.to_play`, so a
        flipped perspective here would invert every terminal value and the
        replica would still run."""
        for node, state in self.pairs:
            want = O if state.player == X else X
            for c in node.children.values():
                self.assertEqual(c.to_play, want)

    def test_the_mirror_is_attached_because_the_deployed_engine_has_one(self):
        if not _ns.HAVE_NATIVE_SELECT:
            self.skipTest("native selection not built")
        for node, _ in self.pairs:
            self.assertIsNotNone(node.sel)
            self.assertEqual(len(node.selS), len(node.children))


class TestTheCorpusCapture(unittest.TestCase):

    def test_the_reservoir_never_exceeds_its_size(self):
        cap = pc.RootCapture(want=3, seed=1)
        s = GameState()
        s.make_move(40)
        kids = {mv: MCTSNode(prior=0.0, move=mv) for mv in (0, 1, 2)}
        for i in range(50):
            cap.n_admitted += 1
            cap.offer(cap.admitted, cap.n_admitted, s, kids)
        self.assertEqual(len(cap.admitted), 3)
        self.assertTrue(all(x is not None for x in cap.admitted))

    def test_a_rejected_offer_does_not_clone(self):
        """The reservoir sees ~90,000 roots a search. Cloning on every offer
        would add an allocation to each one and perturb the allocator the heap
        arm exists to measure."""
        cap = pc.RootCapture(want=1, seed=1)
        s = GameState()
        s.make_move(40)
        kids = {0: MCTSNode(prior=0.0, move=0)}
        cap.n_admitted = 1
        cap.offer(cap.admitted, 1, s, kids)
        n = [0]
        raw = GameState.__dict__["clone"]

        def counting(self):
            n[0] += 1
            return raw(self)

        GameState.clone = counting
        try:
            for _ in range(200):
                cap.n_admitted += 1
                cap.offer(cap.admitted, cap.n_admitted, s, kids)
        finally:
            GameState.clone = raw
        self.assertLess(n[0], 20, "cloned on nearly every rejected offer")

    def test_the_hook_is_restored(self):
        before = MCTS.__dict__["_mark_terminal_children"]
        cap = pc.RootCapture(want=2, seed=1)
        restore = cap.hook(None)
        self.assertIsNot(MCTS.__dict__["_mark_terminal_children"], before)
        restore()
        self.assertIs(MCTS.__dict__["_mark_terminal_children"], before)

    def test_the_hook_splits_roots_by_the_shipped_predicate(self):
        """Both buckets have to fill, or the ladder runs on one population and
        the admitted-vs-all comparison is between a corpus and itself."""
        m = an_mcts(probe_filter=True)
        cap = pc.RootCapture(want=40, seed=3)
        restore = cap.hook(m)
        try:
            s = GameState()
            for mv in (40, 4, 36):
                s.make_move(mv)
            m.search(s)
        finally:
            restore()
        self.assertGreater(cap.n_skipped, 0)
        self.assertGreater(cap.n_admitted + cap.n_skipped, 0)


class TestPrimitivePricing(unittest.TestCase):

    def test_the_empty_loop_is_subtracted(self):
        """A no-op function must price well under the loop that calls it a
        million times. If the baseline stopped being subtracted every crossing
        below would carry the interpreter's loop overhead."""
        ns = pc.primitive(lambda: None, 20000, reps=3)
        self.assertLess(ns, 400.0)

    def test_a_heavier_call_prices_higher(self):
        light = pc.primitive(lambda: None, 20000, reps=3)
        s = GameState()
        heavy = pc.primitive(lambda: s.clone(), 20000, reps=3)
        self.assertGreater(heavy, light)

    def test_bench_returns_the_minimum_not_the_mean(self):
        mn, med, allv = pc.bench(lambda: sum(range(2000)), 5, warmup=1)
        self.assertEqual(mn, min(allv))
        self.assertLessEqual(mn, med)


class TestTheDecision(unittest.TestCase):
    """#49b's rule has to be applied by code, not by whoever reads the table."""

    VOLUME = 5806.1667

    def prices(self):
        return {"pybind bound method, 0 args": 250.0,
                "pybind bound method, 1 arg": 310.0,
                "python function": 26.0,
                "python bound method": 30.0,
                "python property get": 60.0,
                "2-tuple build": 34.0,
                "GameState._raw_winner": 252.0,
                "GameState.winner (property)": 379.0}

    def lad(self, saved_us):
        return {"steps": {"tuple construction + redundant crossing": saved_us},
                "rungs": {"+ _raw_winner + mark": {"us_per_child": 2.0}},
                "extra": {"r3 with the terminal value inlined": 1.98}}

    def test_the_bands_are_the_owners_bands(self):
        for saved_ms, want in ((1.0, "archive"),
                               (3.0, "probably archive unless trivial"),
                               (7.0, "implement + parity/throughput"),
                               (12.0, "real optimisation candidate")):
            us = saved_ms * 1000.0 / self.VOLUME
            got = pc.crossing(self.lad(us), self.prices(), self.VOLUME)
            with self.subTest(ms=saved_ms):
                self.assertEqual(got["band"], want)
                self.assertAlmostEqual(got["projected_ms_per_move"], saved_ms,
                                       places=6)

    def test_the_projection_takes_the_call_volume_from_the_caller(self):
        """Not a module constant, and not the legacy 39,169. The volume has to
        come from the same run as the unit cost -- two runs of a wall-clock
        search on different positions probe different amounts, and the first
        version of this tool mixed a gate's count with its own us/child."""
        a = pc.crossing(self.lad(1.0), self.prices(), 5806.1667)
        b = pc.crossing(self.lad(1.0), self.prices(), 4284.3)
        self.assertAlmostEqual(a["projected_ms_per_move"], 5.8062, places=3)
        self.assertAlmostEqual(b["projected_ms_per_move"], 4.2843, places=3)

    def test_the_structural_estimate_is_independent_of_the_measurement(self):
        """It is built from primitives only. If it started reading the ladder
        the two readings would agree by construction and their agreement would
        prove nothing."""
        a = pc.crossing(self.lad(0.5), self.prices(), self.VOLUME)
        b = pc.crossing(self.lad(9.9), self.prices(), self.VOLUME)
        self.assertEqual(a["structural_us_per_child"],
                         b["structural_us_per_child"])

    def test_the_structural_model_prices_the_whole_property_not_a_bare_one(
            self):
        """`GameState.winner` runs `_raw_winner()` and translates the
        sentinel: 379 ns, not the 61 ns a bare descriptor costs. The first
        version summed bare property gets and landed 54% under the
        measurement, which would have been read as the ladder being wrong."""
        got = pc.crossing(self.lad(0.6), self.prices(), self.VOLUME)
        p = self.prices()
        self.assertAlmostEqual(
            got["structural_us_per_child"],
            (p["python bound method"] + 2 * p["GameState.winner (property)"]
             + p["2-tuple build"] - p["GameState._raw_winner"]) / 1000.0,
            places=9)
        self.assertGreater(got["structural_us_per_child"], 0.5)


class TestTheSplitReconciles(unittest.TestCase):

    ENVELOPE = 46.5618
    ROOTS = 5868.15

    def payload(self):
        return {
            "boundary": {"pybind bound method, 0 args": 250.0,
                         "python bound method": 31.0},
            "insitu": {"children_per_move": 5806.1667,
                       "roots_per_move": self.ROOTS,
                       "scanned_per_move": 788.2833,
                       "probe_ms_per_move": self.ENVELOPE},
            "ladder_admitted": {
                "predicate_us_per_root": 1.7,
                "steps": {"Python child iteration": 0.3,
                          "clone (alloc + copy ctor)": 1.0,
                          "make_move native execution": 1.5,
                          "probe.winner readback": 0.5,
                          "tuple construction + redundant crossing": 0.6,
                          "solved-status writes / propagation": 0.2}},
        }

    def test_the_rows_and_the_spillover_sum_to_the_insitu_total(self):
        s = pc.build_split(self.payload())
        total = sum(ms for _n, ms in s["rows"]) + s["spillover_ms"]
        self.assertAlmostEqual(total, self.ENVELOPE, places=6)

    def test_the_envelope_and_the_volume_come_from_the_payload(self):
        """Not from a module constant holding another run's numbers. Halving
        the measured call volume must halve every per-child row."""
        a = pc.build_split(self.payload())
        p = self.payload()
        p["insitu"]["children_per_move"] /= 2.0
        b = pc.build_split(p)
        ra, rb = dict(a["rows"]), dict(b["rows"])
        self.assertAlmostEqual(rb["clone (alloc + copy ctor)"],
                               ra["clone (alloc + copy ctor)"] / 2.0,
                               places=9)
        # ... but a per-ROOT row must not move.
        self.assertAlmostEqual(rb["predicate (could_end, every root)"],
                               ra["predicate (could_end, every root)"],
                               places=9)

    def test_the_spillover_is_a_residual_and_may_be_negative(self):
        """It is `in-situ minus ladder`, and it is allowed to come out
        negative -- that would mean the wrapper-priced envelope is BELOW an
        isolated replay of the same loop, which is a finding about the
        instrument and must not be clamped away into a comfortable zero."""
        p = self.payload()
        for k in p["ladder_admitted"]["steps"]:
            p["ladder_admitted"]["steps"][k] = 99.0
        s = pc.build_split(p)
        self.assertLess(s["spillover_ms"], 0.0)

    def test_the_predicate_is_charged_per_root_not_per_child(self):
        """It fires on every root, including the ~87% that skip. Charging it
        per probed child would divide it by the wrong number and shrink it
        7.4x."""
        s = pc.build_split(self.payload())
        pred = dict(s["rows"])["predicate (could_end, every root)"]
        self.assertAlmostEqual(pred, 1.7 * self.ROOTS / 1000.0, places=6)

    def test_no_split_without_an_insitu_arm(self):
        """A split against a hard-coded envelope is the error this replaced."""
        p = self.payload()
        del p["insitu"]
        self.assertIsNone(pc.build_split(p))


class TestTheProductionComparison(unittest.TestCase):
    """The reconciliation ratio has to compare like with like.

    Both signs of getting this wrong have now been produced by a real run:
    omitting the predicate when production runs it read as the replica being
    13.5% too CHEAP, and adding it when production does NOT run it read as the
    replica being 7.7% too EXPENSIVE. Neither was a fault in the ladder, and
    either would have been reported as one.
    """

    def setUp(self):
        self.m = an_mcts(probe_filter=True)
        self.pairs = pc.rebuild(self.m, near_terminal_states(4))

    def test_the_predicate_is_added_only_when_the_filter_runs(self):
        on = pc.run_ladder(self.pairs, self.m, 2, "on", filtered=True)
        off = pc.run_ladder(self.pairs, self.m, 2, "off", filtered=False)
        top = on["rungs"]["+ solve/propagate"]["us_per_child"]
        self.assertAlmostEqual(
            off["replica_us_per_child"],
            off["rungs"]["+ solve/propagate"]["us_per_child"], places=9)
        self.assertGreater(on["replica_us_per_child"], top)

    def test_the_flag_is_restored_after_the_production_arm(self):
        """It is toggled to time the right function. Leaving it flipped would
        silently change the engine for every arm that follows."""
        self.m.probe_filter = True
        pc.run_ladder(self.pairs, self.m, 2, "x", filtered=False)
        self.assertTrue(self.m.probe_filter)

    def test_the_ladder_records_which_arm_it_timed(self):
        out = pc.run_ladder(self.pairs, self.m, 2, "x", filtered=False)
        self.assertFalse(out["production_filtered"])


class TestTheOtherCallSite(unittest.TestCase):
    """#49b priced the probe. The counting arm says the probe is the SMALL
    caller, and that has to be arithmetic in the tool rather than a remark."""

    def insitu(self, probe=5806.2, total=58886.7, winner_other=106533.0,
               make_other=53080.5):
        return {
            "search_ms_per_move": 900.0,
            "units": {"make_move_from_probes_per_move": probe,
                      "make_move_per_move": total,
                      "crossings_elsewhere_per_move": 167597.7},
            "counters": {"winner_other": winner_other,
                         "make_other": make_other},
        }

    def test_the_descent_is_priced_at_its_own_call_count(self):
        e = pc.elsewhere({"measured_us_per_child": 0.54}, self.insitu())
        self.assertAlmostEqual(e["descent_calls_per_move"], 53080.5, places=3)
        self.assertAlmostEqual(e["descent_ms_per_move"],
                               0.54 * 53080.5 / 1000.0, places=6)
        self.assertGreater(e["ratio_descent_to_probe"], 9.0)

    def test_a_descent_that_does_not_re_read_voids_the_estimate(self):
        """The ratio is the structural check. If the descent read the winner
        once per make_move, the redundancy would not be there and the whole
        descent number would be an artefact of reading the source."""
        good = pc.elsewhere({"measured_us_per_child": 0.54}, self.insitu())
        self.assertAlmostEqual(
            good["winner_reads_per_make_move_elsewhere"], 2.0, places=1)
        bad = pc.elsewhere({"measured_us_per_child": 0.54},
                           self.insitu(winner_other=53080.5))
        self.assertAlmostEqual(
            bad["winner_reads_per_make_move_elsewhere"], 1.0, places=1)

    def test_the_bands_apply_to_the_bigger_site_too(self):
        e = pc.elsewhere({"measured_us_per_child": 0.54}, self.insitu())
        self.assertEqual(e["descent_band"], "real optimisation candidate")
        self.assertEqual(e["combined_band"], "real optimisation candidate")

    def test_band_boundaries(self):
        self.assertEqual(pc.band_for(1.99), "archive")
        self.assertEqual(pc.band_for(2.0), "probably archive unless trivial")
        self.assertEqual(pc.band_for(5.0), "implement + parity/throughput")
        self.assertEqual(pc.band_for(10.0), "real optimisation candidate")


class TestSourcesAreAscii(unittest.TestCase):

    def test_no_decorative_non_ascii_in_the_tool(self):
        for path in ("tools/probe_cost.py", "tools/test_probe_cost.py"):
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            bad = [(i + 1, line) for i, line in enumerate(src.splitlines())
                   if any(ord(ch) > 127 for ch in line)]
            self.assertEqual(bad, [], "%s has non-ASCII" % path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
