"""#48: the selective terminal probe must be the SAME SEARCH, not a similar one.

WHY THE BAR IS ZERO DRIFT. Every optimisation before this one was allowed to
change something and then argued about whether the change was harmless: the
CUDA-graph wave computes the same numbers in a different order, native
selection returns the same index by a different route, deferred retirement
destroys the same nodes at a different moment. This one is allowed to change
NOTHING. `could_end` is offered as a NECESSARY condition -- if it is False, no
legal move from that node can end the game -- so the loop it skips could not
have marked a child, and the search that follows must be bit-identical.

That claim is what buys the promotion exemption: a search proven identical at a
fixed simulation count, which simply runs more of itself under a clock, does not
need a 700-1,200 game strength match to show it is not worse. The exemption is
only as good as this file, so the tests here accept ZERO semantic drift rather
than statistical equivalence, and they check the two halves separately:

  1. THE PREDICATE IS NECESSARY. Exhaustive over all 4^9 macro configurations
     times both movers, against an oracle that re-derives `check_ultimate_win`
     independently -- then again against the real engine on real positions,
     where the oracle is the probe loop itself.
  2. THE SEARCH IS IDENTICAL. Same visit policy bit for bit, same nodes, same
     `solved` on every one of them -- every legacy proof reproduced AND no
     proof that legacy did not make.

Both halves guard against vacuity, because the expensive mistake in this project
has been a fixture that had nothing to find (tools/test_probe_ablation's
`near_terminal` exists for exactly that reason). A corpus in which the filter
never skips, or in which no proof is ever made, would pass both halves while
testing nothing, so each is asserted to have happened.

    python -m agents.test_probe_filter
    pytest agents/test_probe_filter.py
"""

import itertools
import random
import unittest

import numpy as np
import torch

from agents import native_select as _ns
from agents.mcts import MCTS, could_end, _MACRO_TRIPLES
from engine.constants import EMPTY, X, O, DRAW
from engine.game import GameState
from engine.rules import rule_utl_valid_moves


# --------------------------------------------------------------------------
# Oracles. Deliberately re-derived rather than imported: an oracle that calls
# the code under test proves only that the code equals itself.
# --------------------------------------------------------------------------

def ultimate_winner(mini_winners):
    """`GameState.check_ultimate_win`, written out again from the rule.

    Not imported on purpose. The predicate is derived FROM this function, so an
    oracle that called the shipped one would make a shared misreading of the
    rule invisible -- both sides would be wrong together.
    """
    for a, b, c in _MACRO_TRIPLES:
        v = mini_winners[a]
        if v in (X, O) and v == mini_winners[b] == mini_winners[c]:
            return v
    if all(m != EMPTY for m in mini_winners):
        return DRAW
    return None


def some_move_can_end(mini_winners, mover):
    """Ground truth at the macro level: can ONE ply end the game from here?

    A move can only newly decide the mini-board it is played in, and the only
    values that decision can take on `mover`'s ply are `mover` (they completed a
    line in it) or DRAW (they filled its last cell without a line). It can never
    become the opponent's, which is the asymmetry the predicate's condition (a)
    is built on.
    """
    for m in range(9):
        if mini_winners[m] != EMPTY:
            continue
        for v in (mover, DRAW):
            trial = list(mini_winners)
            trial[m] = v
            if ultimate_winner(trial) is not None:
                return True
    return False


def terminal_children(state):
    """Ground truth on a REAL position: which legal moves end the game.

    This is the probe loop's own definition, run independently of the search --
    what `_mark_terminal_children` would have found if it ran.
    """
    out = []
    for mv in rule_utl_valid_moves(state.board, state.last_move,
                                   state.mini_winners):
        probe = state.clone()
        probe.make_move(mv)
        if probe.winner is not None:
            out.append(mv)
    return out


# --------------------------------------------------------------------------
# Position corpora
# --------------------------------------------------------------------------

def random_states(n, seed, min_plies=0):
    """Reachable positions from real random play, one per game.

    Random play rather than search play on purpose: this half of the file is
    about the PREDICATE, and a corpus sampled by the engine would only cover the
    positions that engine happens to reach. The search-parity tests below use
    the engine's own distribution instead.
    """
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = GameState()
        plies = 0
        keep = None
        while s.winner is None:
            valid = rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
            if not valid:
                break
            s.make_move(rng.choice(valid))
            plies += 1
            if plies >= min_plies and keep is None and rng.random() < 0.08:
                keep = s.clone()
        out.append(keep if keep is not None else s)
    return out


def endgame_closure(seed, max_states=6000, walk=52):
    """EXHAUSTIVE over the tail of a game, not sampled.

    Random-walks `walk` plies in, then enumerates every distinct position
    reachable from there by breadth-first expansion until the budget runs out.
    Late positions are where the predicate has to be exactly right: they are the
    only ones where a proof is available at all, so a corpus that thins out
    before the end would be testing the easy half.
    """
    rng = random.Random(seed)
    root = GameState()
    for _ in range(walk):
        if root.winner is not None:
            break
        valid = rule_utl_valid_moves(root.board, root.last_move,
                                     root.mini_winners)
        if not valid:
            break
        root.make_move(rng.choice(valid))
    if root.winner is not None:
        return []

    seen = set()
    frontier = [root]
    out = []
    while frontier and len(out) < max_states:
        nxt = []
        for s in frontier:
            key = (tuple(s.board), s.player, s.last_move)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= max_states:
                break
            for mv in rule_utl_valid_moves(s.board, s.last_move,
                                           s.mini_winners):
                child = s.clone()
                child.make_move(mv)
                if child.winner is None:
                    nxt.append(child)
        frontier = nxt
    return out


def is_forced(state):
    """True when the last move sent the mover into ONE specific mini-board.

    The two move-generation regimes have to both appear in the corpus: a forced
    state offers moves in a single mini, a send-anywhere state offers moves
    across many, and the predicate is a statement about which minis can be
    decided this ply.
    """
    valid = rule_utl_valid_moves(state.board, state.last_move,
                                 state.mini_winners)
    return len({v // 27 * 3 + (v % 9) // 3 for v in valid}) == 1


# --------------------------------------------------------------------------
# 1. The predicate is a NECESSARY condition
# --------------------------------------------------------------------------

class TestThePredicateIsNecessary(unittest.TestCase):

    def test_exhaustive_over_every_macro_configuration(self):
        """All 4^9 macro boards times both movers. 524,288 cases, no sampling.

        This is the whole proof at the macro level, and it is cheap enough to
        run every time precisely because the predicate only reads
        `mini_winners`: the position underneath cannot affect it.
        """
        checked = admitted = truly = 0
        for mw in itertools.product((EMPTY, X, O, DRAW), repeat=9):
            if ultimate_winner(mw) is not None:
                continue          # already over; no ply is played from here
            for mover in (X, O):
                checked += 1
                want = some_move_can_end(mw, mover)
                got = could_end(mw, mover)
                truly += want
                admitted += got
                if want and not got:
                    self.fail("FALSE NEGATIVE, so the predicate is not a "
                              "necessary condition and the skip is not "
                              "equivalent: mini_winners=%r mover=%d"
                              % (list(mw), mover))
        # 4^9 x 2 = 524,288 minus the configurations that are already decided
        # and therefore never have a ply played from them. The literal is what
        # that leaves; it is here so a `continue` that started firing too often
        # shows up as a failure rather than as a quietly smaller sweep.
        self.assertEqual(checked, 391550)
        self.assertGreater(truly, 0, "vacuous: nothing could end anywhere")
        # IT IS EXACT AT THIS LEVEL, not merely necessary, and that was a
        # finding rather than the design: 177,566 configurations admitted and
        # 177,566 that can really end, zero slack. Necessity is all the
        # correctness argument needs -- False must mean impossible, True only
        # means "look" -- but exactness says the two conditions are not a
        # convenient over-approximation of the rule, they ARE the rule
        # projected onto `mini_winners`. Asserted rather than noted, because if
        # a later edit loosens it into a genuine over-approximation that is a
        # real change and should have to be written down.
        self.assertEqual(admitted, truly,
                         "the predicate is no longer an exact reading of "
                         "`check_ultimate_win` at the macro level")
        self.assertEqual(truly, 177566)

    def test_two_owned_and_the_third_undecided_is_admitted(self):
        mw = [X, X, EMPTY, O, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
        self.assertTrue(could_end(mw, X))
        self.assertTrue(some_move_can_end(mw, X))
        # O owns nothing near a line and five minis are open, so O cannot end
        # it this ply -- the predicate is per-MOVER, not a property of the board.
        self.assertFalse(could_end(mw, O))
        self.assertFalse(some_move_can_end(mw, O))

    def test_two_owned_but_the_third_taken_is_rejected(self):
        for third in (O, DRAW):
            with self.subTest(third=third):
                mw = [X, X, third, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
                self.assertFalse(some_move_can_end(mw, X))
                self.assertFalse(could_end(mw, X))

    def test_exactly_one_undecided_is_admitted_whoever_moves(self):
        """Condition (b). Deciding the last mini fills the macro board, so the
        game ends -- as a win if it completes a line, otherwise as a draw."""
        mw = [X, O, X, O, X, O, O, X, EMPTY]
        self.assertEqual(sum(m == EMPTY for m in mw), 1)
        for mover in (X, O):
            with self.subTest(mover=mover):
                self.assertTrue(some_move_can_end(mw, mover))
                self.assertTrue(could_end(mw, mover))

    def test_two_undecided_and_no_threat_is_rejected(self):
        """One short of condition (b) and with no line available: deciding
        either mini leaves the other open, so the board cannot fill.

        The draws are load-bearing and the first draft of this fixture did not
        have them -- with seven minis decided between two players it is hard to
        avoid leaving somebody two-of-a-line, and the oracle caught it.
        """
        mw = [X, O, DRAW, O, DRAW, X, DRAW, EMPTY, EMPTY]
        self.assertEqual(sum(m == EMPTY for m in mw), 2)
        for mover in (X, O):
            with self.subTest(mover=mover):
                self.assertFalse(some_move_can_end(mw, mover))
                self.assertFalse(could_end(mw, mover))

    def test_overlapping_macro_threats(self):
        """Cell 4 completes three lines at once for X. Overlap must not
        double-count into a wrong answer, and it must not let the loop exit
        before finding the live triple."""
        mw = [X, EMPTY, X, EMPTY, EMPTY, EMPTY, X, EMPTY, X]
        self.assertTrue(could_end(mw, X))
        self.assertTrue(some_move_can_end(mw, X))
        # Same board, the other mover: O owns nothing, so nothing can complete.
        self.assertFalse(could_end(mw, O))
        self.assertFalse(some_move_can_end(mw, O))

    def test_a_drawn_mini_can_never_complete_a_line(self):
        """The reason condition (a) tests ownership by `mover` and not merely
        non-emptiness. DRAW is decided, so it blocks a line; treating it as
        "taken by someone" would be right, treating it as "not empty, so maybe
        mine" would admit a triple that can never complete."""
        mw = [X, X, DRAW, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
        self.assertFalse(could_end(mw, X))
        mw2 = [DRAW, DRAW, EMPTY, DRAW, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
        self.assertFalse(could_end(mw2, X))
        self.assertFalse(some_move_can_end(mw2, X))

    def test_a_board_of_draws_with_one_open_still_ends(self):
        """Draws satisfy condition (b) even though they satisfy no line: the
        macro board fills and `check_ultimate_win` returns DRAW."""
        mw = [DRAW] * 8 + [EMPTY]
        for mover in (X, O):
            with self.subTest(mover=mover):
                self.assertTrue(some_move_can_end(mw, mover))
                self.assertTrue(could_end(mw, mover))


class TestThePredicateAgainstTheEngine(unittest.TestCase):
    """The macro sweep proves the rule; this proves the WIRING -- that
    `state.mini_winners` and `state.player` are the arguments the derivation
    assumed, on positions the engine actually produces."""

    def _check(self, states, label):
        skipped = with_terminal = 0
        for s in states:
            truth = terminal_children(s)
            keep = could_end(s.mini_winners, s.player)
            with_terminal += bool(truth)
            skipped += not keep
            if truth and not keep:
                self.fail("FALSE NEGATIVE in %s: %d terminal move(s) %r at a "
                          "position the filter would have skipped. "
                          "mini_winners=%r player=%d"
                          % (label, len(truth), truth, s.mini_winners,
                             s.player))
        return skipped, with_terminal

    def test_random_reachable_states(self):
        states = random_states(1500, seed=48001)
        skipped, found = self._check(states, "random reachable")
        self.assertGreater(skipped, 0, "vacuous: the filter never skipped")
        self.assertGreater(found, 0, "vacuous: no position had a terminal move")

    def test_late_states_where_proofs_actually_live(self):
        states = random_states(600, seed=48002, min_plies=45)
        skipped, found = self._check(states, "late random")
        self.assertGreater(skipped, 0, "vacuous: the filter never skipped")
        self.assertGreater(found, 0, "vacuous: no position had a terminal move")

    def test_exhaustive_endgame_closure(self):
        """Every distinct position in the tail of three games, not a sample."""
        total = skipped = found = 0
        for seed in (48003, 48004, 48005):
            states = endgame_closure(seed)
            if not states:
                continue
            total += len(states)
            s, f = self._check(states, "endgame closure seed %d" % seed)
            skipped += s
            found += f
        self.assertGreater(total, 3000, "the closure was too small to matter")
        self.assertGreater(found, 0, "vacuous: no position had a terminal move")

    def test_both_move_generation_regimes_appear(self):
        """Forced-mini and send-anywhere. Without this the suite could be
        exercising one regime and claiming to cover both -- and the predicate is
        a statement about which minis this ply can decide, which is exactly what
        the regime controls."""
        states = random_states(400, seed=48006, min_plies=20)
        forced = sum(is_forced(s) for s in states)
        self.assertGreater(forced, 0, "no forced-mini state in the corpus")
        self.assertLess(forced, len(states),
                        "no send-anywhere state in the corpus")
        self._check([s for s in states if is_forced(s)], "forced")
        self._check([s for s in states if not is_forced(s)], "send-anywhere")


# --------------------------------------------------------------------------
# 2. The search is identical
# --------------------------------------------------------------------------

class HashStub:
    """Deterministic, position-dependent, and NOT uniform.

    A uniform stub would make every prior equal, which flattens the tree into a
    breadth-first sweep and hides any ordering effect. These logits are a pure
    function of the input planes, so both arms see the same numbers, but they
    vary enough position to position to produce a realistically lopsided tree.
    """

    def forward_both(self, x):
        flat = x.reshape(x.shape[0], -1) if x.dim() == 4 else x.reshape(1, -1)
        # A fixed linear projection of the planes: deterministic, cheap, and
        # dependent on the whole board rather than on a summary statistic.
        idx = torch.arange(flat.shape[1], dtype=torch.float32)
        h = torch.sin(flat * 1.7 + idx * 0.013)
        logits = torch.stack([h[:, i::81].sum(dim=1) for i in range(81)], dim=1)
        value = torch.tanh(h.sum(dim=1) * 0.001)
        if x.dim() == 4 and x.shape[0] > 1:
            return logits, value
        return logits[0], value[0]


def build(n_sims, filt, mirror=False, wave=8):
    return MCTS(model=HashStub(), device="cpu", n_sims=n_sims, c_puct=1.5,
                add_dirichlet_at_root=False, wave_size=wave, solve=True,
                native_select=mirror, probe_filter=filt)


def walk(node, path=()):
    """Every node of a tree, keyed by the move path that reaches it.

    Keying by path rather than by traversal order means a structural difference
    shows up as a missing key instead of silently shifting every later
    comparison by one.
    """
    out = {path: node}
    for mv, child in node.children.items():
        out.update(walk(child, path + (mv,)))
    return out


class SearchParity(unittest.TestCase):
    """Fixed simulations. Same policy, same tree, same proofs -- or it does not
    ship, because the promotion exemption rests entirely on this."""

    MIRROR = False

    @classmethod
    def setUpClass(cls):
        if cls.MIRROR and not _ns.HAVE_NATIVE_SELECT:
            raise unittest.SkipTest("uttt_select extension not built")
        # A mixed corpus: openings the search has to build from nothing, and
        # late positions where proofs are available and the filter is doing
        # real work. Both matter -- parity in the opening alone would say
        # nothing about the case the feature exists for.
        cls.states = (random_states(10, seed=48010, min_plies=6)
                      + random_states(14, seed=48011, min_plies=44))

    def _run(self, n_sims=160):
        totals = {"skips": 0, "roots": 0, "proofs": 0, "probes_a": 0,
                  "probes_b": 0}
        for i, s in enumerate(self.states):
            legacy = build(n_sims, filt=False, mirror=self.MIRROR)
            selective = build(n_sims, filt=True, mirror=self.MIRROR)
            pi_a, root_a = legacy.search(s.clone())
            pi_b, root_b = selective.search(s.clone())

            with self.subTest(position=i):
                self.assertTrue(np.array_equal(pi_a, pi_b),
                                "visit policy differs at position %d: the "
                                "filter skipped a scan that mattered" % i)
                ta, tb = walk(root_a), walk(root_b)
                self.assertEqual(set(ta), set(tb),
                                 "the two trees have different SHAPES at "
                                 "position %d" % i)
                for path in ta:
                    a, b = ta[path], tb[path]
                    self.assertEqual(a.N, b.N, "visits differ at %r" % (path,))
                    self.assertEqual(a.W, b.W, "value differs at %r" % (path,))
                    self.assertEqual(a.solved, b.solved,
                                     "PROOF DIFFERS at %r: legacy %r, "
                                     "selective %r" % (path, a.solved,
                                                       b.solved))
                    self.assertEqual(a.is_terminal, b.is_terminal,
                                     "terminal flag differs at %r" % (path,))
                    self.assertEqual(a.terminal_value, b.terminal_value,
                                     "terminal value differs at %r" % (path,))

            totals["skips"] += selective.stat_probe_skips
            totals["roots"] += selective.stat_probe_roots
            totals["proofs"] += sum(1 for n in walk(root_a).values()
                                    if n.solved is not None)
            totals["probes_a"] += legacy.stat_probes
            totals["probes_b"] += selective.stat_probes
        return totals

    def test_identical_policy_tree_and_proofs(self):
        t = self._run()
        # Vacuity guards. Either of these at zero would make every assertion
        # above pass while proving nothing: no skip means the two arms ran the
        # same code, and no proof means "every legacy proof reproduced" is a
        # statement about the empty set.
        self.assertGreater(t["skips"], 0,
                           "vacuous: the filter never skipped a single probe "
                           "root, so both arms ran the legacy path")
        self.assertGreater(t["proofs"], 0,
                           "vacuous: no node was ever proved, so proof parity "
                           "was never tested")
        self.assertLess(t["probes_b"], t["probes_a"],
                        "the filter did not reduce probed children at all")

    def test_the_filter_is_off_by_default(self):
        m = MCTS(model=HashStub(), device="cpu", n_sims=8, solve=True)
        self.assertFalse(m.probe_filter)
        pi_a, _ = m.search(self.states[0].clone())
        pi_b, _ = build(8, filt=False).search(self.states[0].clone())
        self.assertTrue(np.array_equal(pi_a, pi_b))

    def test_the_serial_path_matches_too(self):
        """wave_size=1 takes `_expand_from_logits`, a different call site for
        the same probe. Parity on the wave alone would leave the serial path
        untested, and expert iteration still uses it.

        Runs the whole corpus rather than one position: a serial search at 40
        sims builds a small tree, and a single late position can easily contain
        no skippable node at all -- which is how the first draft of this test
        failed its own vacuity guard.
        """
        skips = 0
        for i, s in enumerate(self.states):
            a = build(40, filt=False, wave=1)
            b = build(40, filt=True, wave=1)
            pi_a, ra = a.search(s.clone())
            pi_b, rb = b.search(s.clone())
            with self.subTest(position=i):
                self.assertTrue(np.array_equal(pi_a, pi_b))
                ta, tb = walk(ra), walk(rb)
                self.assertEqual(set(ta), set(tb))
                for path in ta:
                    self.assertEqual(ta[path].N, tb[path].N)
                    self.assertEqual(ta[path].solved, tb[path].solved)
            skips += b.stat_probe_skips
        self.assertGreater(skips, 0, "vacuous: no skip on the serial path")

    def test_the_counters_add_up(self):
        m = build(120, filt=True, mirror=self.MIRROR)
        m.search(self.states[-1].clone())
        self.assertEqual(m.stat_probe_roots, m.stat_expansions)
        self.assertLessEqual(m.stat_probe_skips, m.stat_probe_roots)
        self.assertEqual(m.last["probe_roots"], m.stat_probe_roots)
        self.assertEqual(m.last["probe_skips"], m.stat_probe_skips)

    def test_a_skipped_root_really_had_nothing_to_find(self):
        """The end-to-end statement, checked against the engine rather than
        against the predicate: every root the search skipped is a root whose
        children the probe loop would all have left unmarked."""
        checked = 0
        for s in self.states:
            m = build(120, filt=False, mirror=self.MIRROR)
            _, root = m.search(s.clone())
            for path, node in walk(root).items():
                if not node.children:
                    continue
                st = s.clone()
                for mv in path:
                    st.make_move(mv)
                if st.winner is not None:
                    continue
                if could_end(st.mini_winners, st.player):
                    continue
                checked += 1
                self.assertEqual(
                    terminal_children(st), [],
                    "a node the filter would skip has %d terminal child(ren) "
                    "-- path %r" % (len(terminal_children(st)), path))
        self.assertGreater(checked, 0, "vacuous: no skippable node in any tree")


class SearchParityMirrored(SearchParity):
    """The same suite with the native mirror on, because that is what ships.

    The filter's early return leaves `node.selS` untouched, and the legacy loop
    also leaves it untouched when it marks nothing -- so they agree. That is an
    argument; this is the check. A mirror left half-written would show up as a
    selection difference and therefore as a visit-count difference.
    """

    MIRROR = True


if __name__ == "__main__":
    unittest.main(verbosity=2)
