"""Regression tests for the mini-board indexing bug in the stratum layer.

The bug: `move // 9` was used as the mini-board index. It is the board ROW.
The board is row-major 9x9 (index = r*9 + c), so the mini-board is
(r//3)*3 + (c//3). The two agree only for moves in the leftmost mini of each
horizontal band -- 27 of 81 squares -- which is why the mistake survived
spot-checking and silently mislabelled the mini_win_available stratum.

The sibling bug: `last_move % 9` was used as the mini the mover is SENT to.
That is the COLUMN. The forcing rule uses the LOCAL cell of the last move,
(r%3)*3 + (c%3), which is `rule_utl_get_next_mini`.

Both tests below deliberately assert that the OLD formula disagrees on real
positions, so neither can quietly degrade into a tautology if someone
"simplifies" the reference back into the thing under test.

    python -m tools.test_summarize_search_disagreement
"""
from __future__ import annotations

import random
import unittest

from engine.constants import EMPTY
from engine.game import GameState
from engine.rules import _MINI_INDICES, rule_utl_get_next_mini
from tools.summarize_search_disagreement import (
    forced_target_state, mini_of, tactical_flags)


# Independent reference: which mini owns each square, by membership in the
# precomputed index table rather than by arithmetic. If mini_of() and this
# disagree, one of them is wrong and it is not this one.
MINI_LOOKUP = [None] * 81
for _k, _cells in enumerate(_MINI_INDICES):
    for _c in _cells:
        MINI_LOOKUP[_c] = _k
assert all(v is not None for v in MINI_LOOKUP)


def buggy_mini(move):
    """The formula that shipped. Kept ONLY so the tests can prove they fail."""
    return move // 9


def buggy_next_mini(move):
    """Ditto, for the forcing-target sibling bug."""
    return move % 9


def random_position(rng, plies):
    """Play `plies` random legal moves; return the state (None if it ended)."""
    st = GameState()
    for _ in range(plies):
        legal = st.valid_moves()
        if not legal or st.winner is not None:
            return None
        st.make_move(rng.choice(legal))
    return None if st.winner is not None else st


class TestMiniIndexMath(unittest.TestCase):

    def test_mini_of_matches_index_table_for_all_81_squares(self):
        for m in range(81):
            self.assertEqual(mini_of(m), MINI_LOOKUP[m], f"square {m}")

    def test_row_and_mini_index_actually_differ(self):
        """The bug is only meaningful if the two formulas disagree a lot."""
        differ = [m for m in range(81) if buggy_mini(m) != MINI_LOOKUP[m]]
        # 54 of 81: everything outside the leftmost mini of each band.
        self.assertEqual(len(differ), 54)
        # Spot the shape of it: square 5 is board row 0 but mini 1.
        self.assertEqual(buggy_mini(5), 0)
        self.assertEqual(mini_of(5), 1)
        # ...and square 27 is board row 3 but mini 3 -- a coincidental match,
        # which is exactly how this survived review.
        self.assertEqual(buggy_mini(27), 3)
        self.assertEqual(mini_of(27), 3)

    def test_next_mini_is_local_cell_not_column(self):
        for m in range(81):
            r, c = divmod(m, 9)
            self.assertEqual(rule_utl_get_next_mini(m), (r % 3) * 3 + (c % 3))
        differ = [m for m in range(81)
                  if buggy_next_mini(m) != rule_utl_get_next_mini(m)]
        self.assertEqual(len(differ), 54)


class TestTacticalFlagsOnRealPositions(unittest.TestCase):
    """Differential test: fixed code vs a from-scratch reference, on real play."""

    @staticmethod
    def reference_flags(st, legal):
        immediate_win, mini_win = False, False
        mover = st.player
        for m in legal:
            s = st.clone()
            s.make_move(m)
            if s.winner == mover:
                return True, True
            k = MINI_LOOKUP[m]
            if s.mini_winners[k] == mover and st.mini_winners[k] == EMPTY:
                mini_win = True
        return immediate_win, mini_win

    @staticmethod
    def buggy_flags(st, legal):
        immediate_win, mini_win = False, False
        mover = st.player
        for m in legal:
            s = st.clone()
            s.make_move(m)
            if s.winner == mover:
                return True, True
            if (s.mini_winners[buggy_mini(m)] == mover
                    and st.mini_winners[buggy_mini(m)] == EMPTY):
                mini_win = True
        return immediate_win, mini_win

    def test_fixed_matches_reference_and_old_code_did_not(self):
        rng = random.Random(20260727)
        checked = wrong_before = 0
        for _ in range(600):
            st = random_position(rng, rng.randint(8, 55))
            if st is None:
                continue
            legal = st.valid_moves()
            if not legal:
                continue
            checked += 1
            ref = self.reference_flags(st, legal)
            self.assertEqual(tactical_flags(st, legal), ref)
            if self.buggy_flags(st, legal) != ref:
                wrong_before += 1
        # Guard against a vacuous pass: the sample must be real, and the old
        # formula must actually have been wrong on it.
        self.assertGreaterEqual(checked, 300)
        self.assertGreaterEqual(wrong_before, 20)


class TestForcedTargetOnRealPositions(unittest.TestCase):

    @staticmethod
    def buggy_forced_target(st):
        if st.last_move is None:
            return "none"
        target = buggy_next_mini(st.last_move)
        w = st.mini_winners[target]
        if w in (1, 2):          # X, O
            return "won"
        cells = [st.board[i] for i in _MINI_INDICES[target]]
        if w == 3 or all(c != EMPTY for c in cells):   # DRAW
            return "drawn"
        return "open"

    def test_forced_target_uses_local_cell(self):
        """The stratum must agree with the engine's own forcing rule."""
        rng = random.Random(20260727)
        checked = wrong_before = 0
        for _ in range(600):
            st = random_position(rng, rng.randint(8, 55))
            if st is None or st.last_move is None:
                continue
            checked += 1
            target = rule_utl_get_next_mini(st.last_move)
            got = forced_target_state(st)

            # An "open" verdict must mean the engine really does confine the
            # mover to that one mini -- this ties the stratum to real rules,
            # not to a reimplementation of them.
            legal = st.valid_moves()
            if got == "open":
                self.assertTrue(all(MINI_LOOKUP[m] == target for m in legal))
            else:
                self.assertGreater(len({MINI_LOOKUP[m] for m in legal}), 0)

            if self.buggy_forced_target(st) != got:
                wrong_before += 1
        self.assertGreaterEqual(checked, 300)
        self.assertGreaterEqual(wrong_before, 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
