"""Tests for engine/solver.py -- the alpha-beta endgame oracle.

All tests are pure-Python / no-torch. Run with:
    python -m engine.test_solver          # standalone PASS/FAIL
    pytest engine/test_solver.py          # or under pytest
"""

from engine.game import _PyGameState
from engine.constants import X, O, DRAW, EMPTY
from engine.solver import solve, grade_move, find_blunders, clear_tt


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _fresh():
    """Empty starting position -- no immediate wins, large tree."""
    return _PyGameState()


def _won_by(player):
    """Terminal: `player` has won the top row of minis; it is the loser's turn."""
    return _PyGameState(
        board=[EMPTY] * 81,
        player=O if player == X else X,
        last_move=None,
        mini_winners=[player, player, player] + [EMPTY] * 6,
        winner=player,
    )


def _immediate_win_for_x():
    """X to move, free move. Cell 8 completes mini 2's top row; X then owns
    minis {0,1,2} = ultimate win. This is the canonical 1-ply forced win."""
    board = [EMPTY] * 81
    board[6] = X
    board[7] = X
    return _PyGameState(
        board=board, player=X, last_move=None,
        mini_winners=[X, X, EMPTY] + [EMPTY] * 6, winner=None,
    )


def _blunder_fixture():
    """Exactly 2 legal moves for X (cells 8 and 17 in mini 2).

    Construction verified by direct rule-engine probe:
      - last_move=29 -> next_mini=2 -> X is forced to mini 2
      - mini 2 has board[6]=X,board[7]=X (X's two cells), cells 8 and 17 empty
      - mini_winners=[X,X,EMPTY,DRAW,DRAW,DRAW,DRAW,DRAW,DRAW] (mini 2 is the
        ONLY undecided mini; X already has minis 0 and 1)

    Cell 8 (local 2): completes X's top row in mini 2 -> X wins game (minis 0,1,2).
    Cell 17 (local 5): does NOT complete a mini-2 line; O then has exactly 1
    reply (cell 8), which draws mini 2 -> game draws (all minis decided, no line).

    solve(state).value = 1 (best outcome: play 8 and win).
    After playing 17: solve returns 0 (O draws). Blunder confirmed.
    """
    board = [EMPTY] * 81
    board[6] = X;  board[7] = X          # mini 2: two X's (top row, needs cell 8)
    board[15] = O; board[16] = O         # mini 2: O's two cells in middle row
    board[24] = X; board[25] = O; board[26] = X  # mini 2: bottom row (X-O-X)
    board[29] = O                        # O's last move; last_move=29 -> mini 2
    return _PyGameState(
        board=board, player=X, last_move=29,
        mini_winners=[X, X, EMPTY, DRAW, DRAW, DRAW, DRAW, DRAW, DRAW],
        winner=None,
    )


# --------------------------------------------------------------------------- #
# Tests: terminal detection
# --------------------------------------------------------------------------- #

def test_terminal_loss_returns_minus_one():
    """Position already won by X: O to move and lost -> value=-1."""
    state = _won_by(X)
    r = solve(state)
    assert r.exact
    assert r.value == -1
    assert r.best_move is None


def test_terminal_draw_returns_zero():
    """Draw position: value=0."""
    state = _PyGameState(winner=DRAW)
    r = solve(state)
    assert r.exact
    assert r.value == 0
    assert r.best_move is None


# --------------------------------------------------------------------------- #
# Tests: search correctness
# --------------------------------------------------------------------------- #

def test_immediate_win_detected():
    """One move away from X winning: solve returns (value=1, best_move=8)."""
    clear_tt()
    r = solve(_immediate_win_for_x())
    assert r.exact
    assert r.value == 1
    assert r.best_move == 8, f"expected best_move=8, got {r.best_move}"


def test_budget_exceeded_returns_inexact():
    """budget=0 exceeds immediately on any non-terminal state.
    The node counter increments before winning_moves, so even a 1-ply win
    exceeds budget=0 (unless a TT hit returns first). Terminal states are
    free because they return before counting a node."""
    # Terminal: returns before counting a node -> always exact.
    assert solve(_PyGameState(winner=DRAW), budget=0).exact

    # Live state with no TT entry: node increment fires, 1 > 0 -> inexact.
    r = solve(_fresh(), budget=0)
    assert not r.exact
    assert isinstance(r.value, int)   # doesn't crash


# --------------------------------------------------------------------------- #
# Tests: blunder detection
# --------------------------------------------------------------------------- #

def test_grade_move_no_blunder_on_optimal():
    """Playing the only winning move (cell 8) is not a blunder."""
    clear_tt()
    g = grade_move(_immediate_win_for_x(), 8)
    assert g is not None
    assert g.exact
    assert g.played_value == 1
    assert g.best_value == 1
    assert not g.is_blunder


def test_grade_move_finds_blunder():
    """Playing cell 17 instead of the winning cell 8 is a confirmed blunder.

    best_value=1 (win available), played_value=0 (draw after cell 17).
    Both solves are tiny (<5 nodes total) so they always complete."""
    clear_tt()
    g = grade_move(_blunder_fixture(), 17)
    assert g is not None, "grade_move returned None -- budget exceeded unexpectedly"
    assert g.exact
    assert g.best_value == 1
    assert g.played_value == 0
    assert g.is_blunder


def test_find_blunders_flags_suboptimal_history():
    """find_blunders returns the blundered position from a 1-entry history."""
    clear_tt()
    blunders = find_blunders([(_blunder_fixture(), 17)])
    assert len(blunders) == 1
    assert blunders[0].played_move == 17
    assert blunders[0].is_blunder


def test_find_blunders_empty_on_perfect_play():
    """find_blunders returns [] when the played move is optimal."""
    clear_tt()
    blunders = find_blunders([(_blunder_fixture(), 8)])
    assert blunders == []


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
