"""Generate golden test vectors for the JS engine port (M3 correctness gate).

Records complete game traces from the Python engine so the JS implementation
can be verified move-by-move.  No torch required.

Output: docs/play/golden_vectors.json

Each trace entry has:
  moves          -- flat list of cell indices played in order
  states         -- one entry per ply BEFORE the move is applied:
      board        (81 ints)
      player       (1=X, 2=O)
      lastMove     (-1 or 0-80)
      miniWinners  (9 ints: 0=empty,1=X,2=O,3=draw)
      constraint   (-1=free, 0-8=forced mini)
      winner       (null, 1, 2, or 3)
      validMoves   (list of legal cell indices)

Run:
    python -m scripts.gen_golden_vectors
    python -m scripts.gen_golden_vectors --games 100 --seed 7 --out docs/play/golden_vectors.json
"""

import argparse
import json
import os

import random as _rnd

from engine.game import GameState
from engine.rules import rule_utl_valid_moves, rule_utl_get_next_mini
from engine.constants import X, O, EMPTY, DRAW


def _constraint(game) -> int:
    """Derive the JS `constraint` field from the Python GameState.

    JS uses constraint = -1 (free) or 0-8 (forced mini).
    Python tracks this implicitly via `last_move`.
    """
    if game.last_move is None or game.last_move == -1:
        return -1
    forced = rule_utl_get_next_mini(game.last_move)
    # If that mini is already won/full, it's free choice.
    if game.mini_winners[forced] != EMPTY:
        return -1
    cells = [game.board[i] for i in _mini_cells(forced)]
    if all(c != EMPTY for c in cells):
        return -1
    return forced


def _mini_cells(mini_idx: int):
    mr, mc = divmod(mini_idx, 3)
    return [(mr * 3 + lr) * 9 + mc * 3 + lc
            for lr in range(3) for lc in range(3)]


def _snapshot(game) -> dict:
    valid = rule_utl_valid_moves(game.board, game.last_move, game.mini_winners)
    return {
        "board":       list(game.board),
        "player":      game.player,
        "lastMove":    -1 if game.last_move is None else game.last_move,
        "miniWinners": list(game.mini_winners),
        "constraint":  _constraint(game),
        "winner":      game.winner,   # None, X(1), O(2), DRAW(3)
        "validMoves":  sorted(valid),
    }


def generate_traces(n_games: int = 50, seed: int = 0) -> list:
    _rnd.seed(seed)
    traces = []

    for _ in range(n_games):
        game   = GameState()
        moves  = []
        states = []

        while not game.is_over():
            valid = rule_utl_valid_moves(game.board, game.last_move, game.mini_winners)
            if not valid:
                break
            states.append(_snapshot(game))
            idx = _rnd.choice(valid)
            moves.append(idx)
            game.make_move(idx)

        # Append terminal state so winner/validMoves=[] can be checked too.
        states.append(_snapshot(game))
        traces.append({"moves": moves, "states": states})

    return traces


def main():
    ap = argparse.ArgumentParser(
        description="Generate golden engine vectors for the JS port (M3)."
    )
    ap.add_argument("--games", type=int, default=50,
                    help="Number of full games to record (default 50).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed (default 0; use a different seed to extend coverage).")
    ap.add_argument("--out", type=str, default="docs/play/golden_vectors.json",
                    help="Output path (default: docs/play/golden_vectors.json).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Generating {args.games} game traces (seed={args.seed})...")
    traces = generate_traces(args.games, args.seed)

    total_plies = sum(len(t["moves"]) for t in traces)
    output = {
        "version":     "1.0",
        "description": "Golden vectors from Python engine for JS correctness gate.",
        "n_games":     len(traces),
        "total_plies": total_plies,
        "seed":        args.seed,
        "traces":      traces,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, separators=(",", ":"))   # compact, no whitespace

    kb = os.path.getsize(args.out) // 1024
    print(f"Written: {args.out}  ({kb} KB, {len(traces)} games, {total_plies} plies)")
    print("Run the JS tests: open docs/play/test_engine.html in a browser.")


if __name__ == "__main__":
    main()
