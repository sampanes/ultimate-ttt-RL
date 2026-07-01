"""
Cross-validate C++ engine against Python engine.

All of these work (no PYTHONPATH needed):
    python engine/cpp/test_engine.py
    python -m engine.cpp.test_engine   (requires engine/cpp/__init__.py)
    PYTHONPATH=. python engine/cpp/test_engine.py
"""
import sys
import os
import random

# Build absolute paths from __file__ so any invocation style works.
_here = os.path.dirname(os.path.abspath(__file__))          # .../engine/cpp/
_root = os.path.normpath(os.path.join(_here, "..", ".."))   # project root
sys.path.insert(0, os.path.join(_here, "build", "Release")) # C++ .pyd
sys.path.insert(0, _root)                                   # engine package

from uttt_engine import GameState as CppGame, X, O, DRAW, NO_WINNER
from engine.game import GameState as PyGame
from engine.constants import EMPTY

def random_game_cpp():
    gs = CppGame()
    moves = []
    while not gs.is_over():
        valid = gs.valid_moves()
        assert valid, f"No valid moves but game not over: winner={gs.winner}"
        move = random.choice(valid)
        ok = gs.make_move(move)
        assert ok, f"make_move({move}) failed"
        moves.append(move)
    return gs.winner, moves

def replay_in_python(moves):
    gs = PyGame()
    for move in moves:
        ok, _ = gs.make_move(move)
        assert ok, f"Python rejected move {move}"
    return gs.winner

def map_winner(w):
    """Python uses None for no winner; C++ uses -1. Map Python winner to int."""
    if w is None: return NO_WINNER
    return w

# -- Checkpoint 1: basic construction ------------------------------------------
print("Checkpoint 1: construction + first valid moves")
gs = CppGame()
assert gs.player == X
assert gs.last_move == -1
assert gs.winner == NO_WINNER
assert not gs.is_over()
moves = gs.valid_moves()
assert len(moves) == 81, f"Expected 81 first moves, got {len(moves)}"
print(f"  OK -- {len(moves)} valid first moves")

# -- Checkpoint 2: single move --------------------------------------------------
print("Checkpoint 2: single move routing")
gs = CppGame()
gs.make_move(40)  # center cell, mini-board 4
assert gs.player == O
assert gs.last_move == 40
valid = gs.valid_moves()
# cell 40 -> get_next_mini: row=4, col=4, local_row=1, local_col=1 -> mini 4
# mini 4 cells: rows 3-5, cols 3-5 -> indices 30,31,32,39,40,41,48,49,50
# cell 40 is taken, so 8 remain
assert len(valid) == 8, f"Expected 8 valid moves after center, got {len(valid)}: {valid}"
print(f"  OK -- {len(valid)} valid moves after playing center")

# -- Checkpoint 3: full random games -------------------------------------------
print("Checkpoint 3: 1000 random games complete without assertion errors")
winners = {X: 0, O: 0, DRAW: 0}
for _ in range(1000):
    w, _ = random_game_cpp()
    assert w in (X, O, DRAW), f"Invalid winner: {w}"
    winners[w] += 1
print(f"  OK -- X:{winners[X]} O:{winners[O]} Draw:{winners[DRAW]}")

# -- Checkpoint 4: cross-validate vs Python ------------------------------------
print("Checkpoint 4: cross-validate 500 games vs Python engine")
mismatches = 0
for i in range(500):
    cpp_winner, moves = random_game_cpp()
    py_winner = map_winner(replay_in_python(moves))
    if cpp_winner != py_winner:
        mismatches += 1
        print(f"  MISMATCH game {i}: cpp={cpp_winner} py={py_winner} moves={moves[:10]}...")

if mismatches == 0:
    print(f"  OK -- all 500 games match Python engine exactly")
else:
    print(f"  FAILED -- {mismatches}/500 mismatches")

print("\nAll checkpoints done.")
