"""Unit tests for engine.tactics -- pure, no torch, runs on any box.

    python -m engine.test_tactics

Positions are built directly as _PyGameState so the test is backend-independent
(the C++ GameState can't be hand-constructed mid-game; the tactics functions only
read board/last_move/mini_winners/player and clone(), all of which _PyGameState
provides identically).

Cell layout reference (global indices 0-80, row-major 9x9):
  mini 0 = [0,1,2, 9,10,11, 18,19,20]   local (0,0..2) = 0,1,2
  mini 2 = [6,7,8, 15,16,17, 24,25,26]  local (0,0..2) = 6,7,8
"""
from engine.constants import EMPTY, X, O
from engine.game import _PyGameState
from engine.tactics import (
    mini_tactical_filter,
    mini_winning_moves,
    tactical_filter,
    losing_moves,
    winning_moves,
)

_failures = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}: {msg}")
    if not cond:
        _failures.append(msg)


def test_win_in_1():
    """X owns minis 0 and 1; completing mini 2 (cell 8) wins the game."""
    board = [EMPTY] * 81
    board[6] = X
    board[7] = X  # X has local row 0 of mini 2; cell 8 completes it
    mini_winners = [EMPTY] * 9
    mini_winners[0] = X
    mini_winners[1] = X  # ultimate line [0,1,2] needs mini 2
    s = _PyGameState(board=board, player=X, last_move=None,
                     mini_winners=mini_winners, winner=None)

    print("test_win_in_1:")
    wins = winning_moves(s)
    check(wins == [8], f"winning_moves == [8] (got {wins})")
    w, safe = tactical_filter(s)
    check(w == [8] and safe == [8], f"tactical_filter == ([8],[8]) (got ({w},{safe}))")


def test_block_in_1():
    """O owns minis 0 and 1 and threatens mini 2 via cell 8. X to move:
    cell 8 occupies the threat (safe); a move that lets O reach cell 8 loses."""
    board = [EMPTY] * 81
    board[6] = O
    board[7] = O  # O threatens to complete mini 2 (and the game) at cell 8
    mini_winners = [EMPTY] * 9
    mini_winners[0] = O
    mini_winners[1] = O
    s = _PyGameState(board=board, player=X, last_move=None,
                     mini_winners=mini_winners, winner=None)

    print("test_block_in_1:")
    losing = losing_moves(s)
    check(2 in losing, f"move 2 flagged losing -- it forces O into the kill square (got {sorted(losing)[:8]}...)")
    check(8 not in losing, "move 8 NOT losing -- it occupies O's winning square")
    check(winning_moves(s) == [], "X has no immediate win here")
    w, safe = tactical_filter(s)
    check(w == [], "tactical_filter winning is empty")
    check(8 in safe, "safe pool includes the blocking move 8")
    check(2 not in safe, "safe pool excludes the loss-enabling move 2")


def test_neutral_empty_board():
    """Fresh board: no win, no immediate loss, everything is safe."""
    s = _PyGameState()
    print("test_neutral_empty_board:")
    check(winning_moves(s) == [], "no winning move on an empty board")
    check(losing_moves(s) == [], "no loss-enabling move on an empty board")
    w, safe = tactical_filter(s)
    check(w == [] and len(safe) == 81, f"tactical_filter == ([], all 81) (got {w}, |safe|={len(safe)})")


def test_mini_tactical_filter():
    """Local mini-board rules mirror WinBlockAgent but stay separate from proof tactics."""
    board = [EMPTY] * 81
    board[0] = X
    board[1] = X
    board[9] = O
    board[10] = O
    s = _PyGameState(board=board, player=X, last_move=None,
                     mini_winners=[EMPTY] * 9, winner=None)

    print("test_mini_tactical_filter:")
    check(mini_winning_moves(s) == [2], "X can complete mini 0 at cell 2")
    wins, blocks, preferred = mini_tactical_filter(s)
    check(wins == [2], f"mini wins == [2] (got {wins})")
    check(blocks == [11], f"mini blocks == [11] (got {blocks})")
    check(preferred == [2], f"mini preferred takes win before block (got {preferred})")

    board[0] = EMPTY
    board[1] = EMPTY
    s = _PyGameState(board=board, player=X, last_move=None,
                     mini_winners=[EMPTY] * 9, winner=None)
    wins, blocks, preferred = mini_tactical_filter(s)
    check(wins == [], f"no X mini win (got {wins})")
    check(blocks == [11], f"block O mini win at cell 11 (got {blocks})")
    check(preferred == [11], f"mini preferred blocks when no win (got {preferred})")


if __name__ == "__main__":
    test_win_in_1()
    test_block_in_1()
    test_neutral_empty_board()
    test_mini_tactical_filter()
    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for m in _failures:
            print(f"  - {m}")
        raise SystemExit(1)
    print("All tactics tests passed.")
