"""Tests for the probe instrument -- does it count the right things?

The #47 decision turns on two numbers that nothing else in this repo produces:
how many pybind crossings a probed child costs, and how often a probe changes
anything. Both are counted by wrappers that could be silently wrong -- a
counter that never fires reads exactly like a feature that never does. Each
test below breaks something the instrument is supposed to notice.

    python -m tools.test_probe_ablation
"""
from __future__ import annotations

import random
import unittest

import torch

from agents.mcts import MCTS
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from tools import engine_registry as reg
from tools import probe_ablation as pa


class Toy:
    """Real `forward_both` squeezes batch=1 to (81,)/scalar, and the root
    expansion depends on that. A toy that kept the batch dim fails before any
    wave runs."""

    def forward_both(self, x):
        b = x.shape[0]
        lg, v = torch.zeros((b, 81)), torch.zeros((b,))
        return (lg[0], v[0]) if b == 1 else (lg, v)


def a_state(moves=(40, 4, 36)):
    s = GameState()
    for mv in moves:
        s.make_move(mv)
    return s


def near_terminal(seed=17):
    """A real position with at least one game-ending legal move.

    Hand-picked move sequences do NOT produce one. An ultimate win needs three
    mini-boards in a line, which is 20+ plies away from anything writable as a
    literal -- and a fixture with no terminal children makes the hit tests pass
    vacuously, which is worse than failing. So: random-walk real games with a
    fixed seed until a position with a terminal reply turns up.
    """
    rng = random.Random(seed)
    for _ in range(400):
        s = GameState()
        while s.winner is None:
            moves = rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
            if not moves:
                break
            for mv in moves:
                probe = s.clone()
                probe.make_move(mv)
                if probe.winner is not None:
                    return s
            s.make_move(rng.choice(moves))
    raise AssertionError("no near-terminal position in 400 games")


def searched(counters=None, ctx=None, sims=48, solve=True, moves=(40, 4, 36),
             pfilter=False):
    """Run a real search with the counters installed. Returns the MCTS."""
    m = MCTS(model=Toy(), device="cpu", n_sims=sims, c_puct=1.5,
             wave_size=8, solve=solve, probe_filter=pfilter)
    if counters is None:
        m.search(a_state(moves))
        return m
    restore = pa.instrument(counters, ctx)
    try:
        ctx["on"] = True
        m.search(a_state(moves))
        ctx["on"] = False
    finally:
        restore()
    return m


def terminal_children(state, node):
    """The oracle: a child is terminal iff playing its move ends the game.

    This is the DEFINITION the probe implements, written independently. If the
    two ever disagree the instrument is measuring something else.
    """
    out = set()
    for mv in node.children:
        probe = state.clone()
        probe.make_move(mv)
        if probe.winner is not None:
            out.add(mv)
    return out


class TestTheCountersAgreeWithProduction(unittest.TestCase):
    """The strongest available cross-check: MCTS keeps its own probe tally, and
    it was written for a different purpose by different code."""

    def test_children_counted_equals_stat_probes(self):
        c = pa.ProbeCounters()
        m = searched(c, {"on": False})
        self.assertGreater(c.children, 0)
        self.assertEqual(c.children, m.stat_probes)

    def test_every_probe_root_is_an_expansion(self):
        c = pa.ProbeCounters()
        m = searched(c, {"on": False})
        self.assertGreater(c.roots, 0)
        self.assertLessEqual(c.roots, m.last["nodes_expanded"])

    def test_children_per_root_is_a_legal_move_count(self):
        c = pa.ProbeCounters()
        searched(c, {"on": False})
        per_root = c.children / c.roots
        self.assertGreater(per_root, 1.0)
        self.assertLessEqual(per_root, 81.0)


class TestTheHitCountIsTheDefinition(unittest.TestCase):

    def test_hits_match_an_independent_walk(self):
        m = MCTS(model=Toy(), device="cpu", n_sims=1, c_puct=1.5,
                 wave_size=8, solve=True)
        s = near_terminal()
        c = pa.ProbeCounters()
        ctx = {"on": True}
        restore = pa.instrument(c, ctx)
        try:
            _pi, root = m.search(s.clone())
        finally:
            ctx["on"] = False
            restore()
        want = terminal_children(s, root)
        self.assertTrue(want, "vacuous fixture: nothing to find")
        got = {mv for mv, ch in root.children.items()
               if ch.solved is not None}
        self.assertEqual(got, want)
        self.assertGreaterEqual(c.hits, len(want))

    def test_a_probe_that_marks_nothing_is_caught(self):
        """Plant the defect: a probe loop that walks and writes nothing. The
        hit counter must fall to zero while the oracle still finds terminals."""
        raw = MCTS.__dict__["_mark_terminal_children"]

        def blind(self, node, state):
            for mv in node.children:
                probe = state.clone()
                probe.make_move(mv)
                self.stat_probes += 1

        s = near_terminal()
        m = MCTS(model=Toy(), device="cpu", n_sims=1, c_puct=1.5,
                 wave_size=8, solve=True)
        c = pa.ProbeCounters()
        ctx = {"on": True}
        MCTS._mark_terminal_children = blind
        restore = pa.instrument(c, ctx)
        try:
            _pi, root = m.search(s.clone())
        finally:
            ctx["on"] = False
            restore()
            MCTS._mark_terminal_children = raw
        self.assertTrue(terminal_children(s, root),
                        "the fixture must have terminals or it proves nothing")
        self.assertEqual(c.hits, 0)
        self.assertEqual(c.roots_proved, 0)


class TestTheCrossingCount(unittest.TestCase):
    """The number a native port would be sized against, measured rather than
    read off the source."""

    def setUp(self):
        self.c = pa.ProbeCounters()
        searched(self.c, {"on": False})

    def test_one_clone_and_one_make_move_per_probed_child(self):
        self.assertEqual(self.c.clone_probe, self.c.children)
        self.assertEqual(self.c.make_probe, self.c.children)

    def test_the_winner_property_is_read_twice_per_child(self):
        """Once inside `make_move`, for a return tuple the probe loop DISCARDS,
        and once by the loop's own `probe.winner is not None` test. Plus one
        more per hit, from `_terminal_value`. If this ever stops holding, the
        redundant read has been removed and the ceiling below has moved."""
        self.assertEqual(self.c.winner_probe,
                         2 * self.c.children + self.c.hits)

    def test_four_crossings_per_child_and_five_per_hit(self):
        cross = self.c.crossings()["probe"]
        self.assertEqual(cross, 4 * self.c.children + self.c.hits)
        per_child = cross / self.c.children
        self.assertGreaterEqual(per_child, 4.0)
        self.assertLess(per_child, 5.0)

    def test_probe_and_non_probe_crossings_are_kept_apart(self):
        # The descent clones and makes moves too. If the flag never closed,
        # every crossing in the search would be charged to the probes.
        self.assertGreater(self.c.clone_other + self.c.make_other, 0)


class TestTheGateAndTheRestore(unittest.TestCase):

    def test_nothing_is_counted_while_the_gate_is_shut(self):
        c = pa.ProbeCounters()
        ctx = {"on": False}
        restore = pa.instrument(c, ctx)
        try:
            m = MCTS(model=Toy(), device="cpu", n_sims=32, c_puct=1.5,
                     wave_size=8, solve=True)
            m.search(a_state())
        finally:
            restore()
        self.assertEqual(c.children, 0)
        self.assertEqual(c.clone_probe + c.clone_other, 0)
        self.assertEqual(c.winner_probe + c.winner_other, 0)

    def test_restore_puts_back_the_original_objects(self):
        before = {
            ("MCTS", "_mark_terminal_children"):
                MCTS.__dict__["_mark_terminal_children"],
            ("MCTS", "_propagate_solved"): MCTS.__dict__["_propagate_solved"],
            ("GameState", "clone"): GameState.__dict__["clone"],
            ("GameState", "make_move"): GameState.__dict__["make_move"],
            ("GameState", "winner"): GameState.__dict__["winner"],
        }
        restore = pa.instrument(pa.ProbeCounters(), {"on": True})
        self.assertIsNot(GameState.__dict__["winner"],
                         before[("GameState", "winner")])
        restore()
        for (holder, attr), obj in before.items():
            with self.subTest(target="%s.%s" % (holder, attr)):
                got = (MCTS if holder == "MCTS" else GameState).__dict__[attr]
                self.assertIs(got, obj)

    def test_the_patched_winner_still_translates_the_sentinel(self):
        """The property is REPLACED, not wrapped. If the replacement dropped
        the -1 -> None translation, every `winner is not None` test in the
        engine would flip and the counters would still look perfectly healthy.

        Both branches are exercised: an unfinished position (sentinel) and a
        finished one (a real result)."""
        s = near_terminal()
        end = None
        for mv in rule_utl_valid_moves(s.board, s.last_move,
                                       s.mini_winners):
            trial = s.clone()
            trial.make_move(mv)
            if trial.winner is not None:
                end = trial
                break
        self.assertIsNotNone(end, "vacuous fixture: no finishing move")
        restore = pa.instrument(pa.ProbeCounters(), {"on": True})
        try:
            self.assertIsNone(GameState().winner)
            self.assertEqual(s.winner, None if s._raw_winner() == -1
                             else s._raw_winner())
            self.assertIsNotNone(end.winner)
            self.assertEqual(end.winner, end._raw_winner())
        finally:
            restore()


class TestPricingTheInstrument(unittest.TestCase):
    """Wrapping `clone` and `make_move` puts tens of thousands of wrappers
    INSIDE the probe's own interval. Not subtracting them prices the profiler
    and calls it the probe."""

    BLOB = {
        "inside_us": 0.30, "total_us": 1.10,
        "calls": {"terminal probes": 1000},
        "calls_from": {"terminal probes>state clone": 30000,
                       "terminal probes>state.make_move": 30000},
        "inclusive_ms": {"terminal probes": 200.0},
        "exclusive_ms": {"terminal probes": 90.0},
    }

    def test_the_correction_is_large_and_in_the_right_direction(self):
        p = pa.price(self.BLOB, moves=10)
        self.assertLess(p["inclusive_ms"], p["raw_inclusive_ms"])
        self.assertLess(p["exclusive_ms"], p["raw_exclusive_ms"])
        # 60,000 nested calls at 1.10 us is 66 ms of pure instrument against a
        # 200 ms raw reading. A tool that skipped this would overstate by 49%.
        self.assertAlmostEqual(p["wrapper_ms"], 0.3 + 66.0, places=6)

    def test_a_nested_call_charges_its_parent_the_whole_price(self):
        p = pa.price(self.BLOB, moves=10)
        self.assertAlmostEqual(p["inclusive_ms"], 200.0 - 66.3, places=6)

    def test_exclusive_only_pays_the_part_that_ran_outside_the_child_clock(self):
        # The nested DT was already subtracted by the timer; what was not is
        # the wrapper work that happened outside the child's own interval.
        p = pa.price(self.BLOB, moves=10)
        self.assertAlmostEqual(p["exclusive_ms"], 90.0 - (0.3 + 48.0),
                               places=6)

    def test_no_nested_calls_means_only_the_function_charges_itself(self):
        blob = dict(self.BLOB, calls_from={})
        p = pa.price(blob, moves=10)
        self.assertAlmostEqual(p["wrapper_ms"], 0.3, places=6)
        self.assertAlmostEqual(p["inclusive_ms"], 199.7, places=6)


class TestTheThroughputEstimator(unittest.TestCase):
    """`uttt-deferred-retirement-wins`: nn/move moves with the early-stop mix,
    and this is the one tool where one arm can stop early and the other cannot
    -- by construction."""

    def arm(self, rows):
        return {"rows": rows, "budget_ms": 1000.0, "reserve_ms": 20.0}

    def full(self, ms, nn, solved=None):
        return {"phase": "mid", "search_ms": ms, "nn": nn, "sims": nn,
                "expansions": nn // 4, "probes": nn, "root_solved": solved,
                "root_n": nn, "move": 0}

    def test_an_early_return_drags_nn_per_move_but_not_the_estimator(self):
        steady = [self.full(980.0, 5000) for _ in range(9)]
        early = self.full(3.0, 15, solved=1)
        a = pa.rate(self.arm(steady + [steady[0]]))
        b = pa.rate(self.arm(steady + [early]))
        self.assertLess(b["nn_per_move"], 0.92 * a["nn_per_move"])
        self.assertAlmostEqual(b["nn_full"], a["nn_full"], delta=0.02
                               * a["nn_full"])

    def test_nn_full_is_the_rate_times_the_deadline(self):
        a = pa.rate(self.arm([self.full(980.0, 4900)]))
        self.assertAlmostEqual(a["nn_per_second"], 5000.0, places=6)
        self.assertAlmostEqual(a["nn_full"], 5000.0 * 0.98, places=6)

    def test_early_returns_need_a_proven_root_and_a_short_search(self):
        rows = [self.full(3.0, 15, solved=1),      # early
                self.full(980.0, 5000, solved=1),  # proven but ran the clock
                self.full(3.0, 15)]                # short but unproven
        r = pa.rate(self.arm(rows))
        self.assertEqual(r["solved_roots"], 2)
        self.assertEqual(r["early_returns"], 1)


class TestDisagreement(unittest.TestCase):

    def rows(self, moves, phases=None):
        phases = phases or ["mid"] * len(moves)
        return {"rows": [{"move": m, "phase": p}
                         for m, p in zip(moves, phases)]}

    def test_identical_arms_disagree_on_nothing(self):
        d = pa.disagreement(self.rows([1, 2, 3]), self.rows([1, 2, 3]))
        self.assertEqual(d["differ"], 0)
        self.assertEqual(d["rate"], 0.0)

    def test_the_rate_is_over_the_paired_positions(self):
        d = pa.disagreement(self.rows([1, 2, 3, 4]), self.rows([1, 9, 3, 9]))
        self.assertEqual((d["differ"], d["n"]), (2, 4))
        self.assertAlmostEqual(d["rate"], 0.5)

    def test_phases_partition_the_positions(self):
        a = self.rows([1, 2, 3, 4], ["early", "early", "late", "late"])
        b = self.rows([1, 9, 3, 9], ["early", "early", "late", "late"])
        d = pa.disagreement(a, b)
        self.assertEqual(sum(v["n"] for v in d["by_phase"].values()), 4)
        self.assertEqual(sum(v["differ"] for v in d["by_phase"].values()), 2)


class TestWiring(unittest.TestCase):

    def test_the_arms_follow_whatever_is_deployed(self):
        self.assertEqual(pa.ON_SPEC, "engine:%s" % reg.DEPLOYED)
        self.assertTrue(pa.OFF_SPEC.startswith(pa.ON_SPEC + "+"))

    def test_the_ablation_changes_exactly_one_declared_key(self):
        _spec, diff = reg.derived_spec(reg.DEPLOYED, {"solve": "0"})
        self.assertEqual(diff, {"solve"})

    def test_the_off_arm_really_stops_probing(self):
        c = pa.ProbeCounters()
        m = searched(c, {"on": False}, solve=False)
        self.assertEqual(c.roots, 0)
        self.assertEqual(c.children, 0)
        self.assertEqual(m.stat_probes, 0)
        # ...and it still searches, so the arm is a player and not a corpse.
        self.assertGreater(m.last["simulations_completed"], 0)

    def test_the_on_arm_does_probe(self):
        c = pa.ProbeCounters()
        searched(c, {"on": False}, solve=True)
        self.assertGreater(c.roots, 0)

    def test_seed_namespace_is_distinct_and_held_out(self):
        self.assertEqual(pa.PROBE_SEED, reg.SEEDS["probe"])
        self.assertEqual(len(set(reg.SEEDS.values())), len(reg.SEEDS))
        for other in ("confirm", "select_ab", "select_confirm", "release_ab",
                      "probe_ab"):
            self.assertNotEqual(pa.PROBE_SEED, reg.SEEDS[other])

    def test_the_timed_arm_wraps_the_probe_its_primitives_and_the_predicate(
            self):
        """Four, not sixteen. The whole-engine breakdown is
        tools/profile_selection's job and every extra wrapper here costs search
        that the probe rows are then scaled against. `could_end` earns its place
        because on the #48 arm it is what runs INSTEAD of the loop -- unwrapped,
        its cost would hide inside "terminal probes" and make the saving look
        bigger than it is."""
        names = {n for n, _h, _a in pa.TIMED_TARGETS}
        self.assertEqual(names, {"terminal probes", "state clone",
                                 "state.make_move", "could_end"})

    def test_the_gate_compares_a_pinned_pair(self):
        """BOTH sides literal. #48c is a published comparison, so it has to
        keep building the same two engines forever -- and this is not
        hypothetical: `pocket_filter` was promoted on the strength of that
        measurement, so a DEPLOYED-derived base would now build the candidate
        against itself and report a dead heat."""
        self.assertEqual(pa.GATE_BASE_SPEC, "engine:pocket_defer")
        self.assertEqual(pa.SEL_SPEC, "engine:pocket_filter")
        for name in ("pocket_defer", "pocket_filter"):
            self.assertIn(name, reg.ENGINES)
        base = reg.ENGINES["pocket_defer"]
        cand = reg.ENGINES["pocket_filter"]
        self.assertEqual({k for k in base if base[k] != cand[k]} - {"name"},
                         {"pfilter"})

    def test_a_skipped_root_contributes_no_children(self):
        """The counter split the gate rests on. If a filtered root's children
        were counted, the selective arm would be credited with per-child work
        it never did and every us/child figure would be wrong in the direction
        that flatters the change."""
        c = pa.ProbeCounters()
        m = searched(c, {"on": True}, solve=True, pfilter=True)
        self.assertEqual(c.roots, c.scanned + c.roots_skipped)
        self.assertGreater(c.roots_skipped, 0, "vacuous: nothing was skipped")
        self.assertEqual(c.children, m.stat_probes)
        self.assertLess(c.children, c.roots * 2,
                        "children were counted for skipped roots")


if __name__ == "__main__":
    unittest.main(verbosity=2)
