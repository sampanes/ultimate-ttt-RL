"""Build a GOLD-seeded endgame position suite for blunder-rate grading.

WHY GOLD SEEDING?
  A strong agent rarely enters its own losing endgames organically -- so
  grading only the positions that come up in candidate-vs-opponent games
  skews toward won positions where the candidate is already ahead. The
  blunder rate looks good because the hard cases are absent.

  GOLD seeding fixes this: play RANDOM-vs-RANDOM games to collect ALL
  endgame position types (won / drawn / lost for the player to move).
  The random agents have no bias -- they'll walk into losing endgames just
  as readily as winning ones.

OUTPUT FORMAT (JSON list of objects):
  {
    "board":         [81 ints, 0=empty 1=X 2=O],
    "mini_winners":  [9 ints, 0=empty 1=X 2=O 3=draw],
    "player":        1 or 2 (who is to move),
    "last_move":     int or null,
    "value":         +1/0/-1 from player's perspective,
    "optimal_moves": [list of cell indices that achieve value]
  }

USAGE
  # Build (no torch needed):
  python -m scripts.build_endgame_suite --out suite.json

  # Larger suite -- takes ~10 min on a fast box at max_empty 20:
  python -m scripts.build_endgame_suite --out suite.json \\
      --n_games 2000 --max_empty 20 --budget 200000

  # Then grade any checkpoint against it (needs torch):
  python -m scripts.grade_agent \\
      --suite suite.json \\
      --checkpoint models/league_pg/best.pt --network medium

GATE
  A suite is only useful if it has reasonable balance across value categories.
  The script prints a breakdown at the end. Aim for at least 100 won + 50 drawn
  positions; lost positions are rare (< 10 % is normal) because even random games
  rarely produce a provably-lost position that the solver can verify quickly.
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

from engine.game import _PyGameState
from engine.constants import EMPTY, X, O, DRAW
from engine.rules import rule_utl_valid_moves, rule_utl_get_next_mini, _MINI_INDICES
from engine.solver import AlphaBetaSolver


# --------------------------------------------------------------------------- #
# Suite entry type
# --------------------------------------------------------------------------- #

@dataclass
class SuiteEntry:
    board:         List[int]         # 81 ints
    mini_winners:  List[int]         # 9 ints
    player:        int               # X or O
    last_move:     Optional[int]     # global cell idx of last move, or None
    value:         int               # +1/0/-1 from player's perspective
    optimal_moves: List[int]         # all moves that achieve value


def state_from_entry(entry: "SuiteEntry") -> _PyGameState:
    """Reconstruct a _PyGameState from a SuiteEntry (for cold-call grading)."""
    return _PyGameState(
        board=list(entry.board),
        player=entry.player,
        last_move=entry.last_move,
        mini_winners=list(entry.mini_winners),
        winner=None,
    )


# --------------------------------------------------------------------------- #
# Dedup key (transpositions only; no D4 for v1 -- the working set is small)
# --------------------------------------------------------------------------- #

def _dedup_key(board, player, last_move, mini_winners):
    if last_move is None:
        constraint = None
    else:
        c = rule_utl_get_next_mini(last_move)
        if mini_winners[c] != EMPTY:
            constraint = None
        elif all(board[i] != EMPTY for i in _MINI_INDICES[c]):
            constraint = None
        else:
            constraint = c
    return (bytes(board), player, constraint)


# --------------------------------------------------------------------------- #
# Optimal-move enumeration
# --------------------------------------------------------------------------- #

def _get_optimal_moves(
    state: _PyGameState,
    optimal_value: int,
    budget: int,
    solver: AlphaBetaSolver,
) -> Optional[List[int]]:
    """Return all moves from state that achieve optimal_value, or None if any solve fails."""
    valid = rule_utl_valid_moves(state.board, state.last_move, state.mini_winners)
    result = []
    for mv in valid:
        s = state.clone()
        _, w = s.make_move(mv)
        if w is not None:
            if w == state.player:
                mv_val = 1    # mover won
            elif w == DRAW:
                mv_val = 0
            else:
                mv_val = -1   # can't happen in UTTT (mover can't lose on their own move)
        else:
            r = solver.solve(s, budget)
            if not r.exact:
                return None   # couldn't verify this branch -- skip the whole position
            mv_val = -r.value  # negate: opponent's value -> mover's value
        if mv_val == optimal_value:
            result.append(mv)
    return result or None


# --------------------------------------------------------------------------- #
# Suite builder
# --------------------------------------------------------------------------- #

def build_suite(
    n_games:   int          = 1000,
    max_empty: int          = 15,
    budget:    int          = 100_000,
    seed:      Optional[int] = None,
    verbose:   bool         = True,
) -> List[SuiteEntry]:
    """Play n_games random games, collect and solve all endgame positions.

    Returns a list of SuiteEntry objects with proven values and optimal moves.
    Positions are deduplicated by (board, player, constraint) key -- transpositions
    across different games are stored only once.
    """
    if seed is not None:
        random.seed(seed)

    solver   = AlphaBetaSolver()
    seen     = set()
    suite: List[SuiteEntry] = []
    counts   = {1: 0, 0: 0, -1: 0}
    skipped  = 0   # positions the solver couldn't prove within budget
    t0       = time.time()

    for game_idx in range(n_games):
        if verbose and (game_idx + 1) % max(1, n_games // 10) == 0:
            elapsed = time.time() - t0
            print(
                f"  game {game_idx+1}/{n_games} | "
                f"suite={len(suite)} (+1:{counts[1]} 0:{counts[0]} -1:{counts[-1]}) | "
                f"skipped={skipped} | TT={len(solver._tt)} | {elapsed:.1f}s"
            )

        state = _PyGameState()
        while state.winner is None:
            empty = state.board.count(EMPTY)

            if empty <= max_empty:
                key = _dedup_key(state.board, state.player,
                                 state.last_move, state.mini_winners)
                if key not in seen:
                    r = solver.solve(state, budget)
                    if r.exact:
                        opt = _get_optimal_moves(state, r.value, budget, solver)
                        if opt is not None:
                            seen.add(key)
                            suite.append(SuiteEntry(
                                board=list(state.board),
                                mini_winners=list(state.mini_winners),
                                player=state.player,
                                last_move=state.last_move,
                                value=r.value,
                                optimal_moves=opt,
                            ))
                            counts[r.value] += 1
                        else:
                            skipped += 1  # optimal_moves sub-solve failed
                    else:
                        skipped += 1      # root solve exceeded budget

            valid = rule_utl_valid_moves(state.board, state.last_move, state.mini_winners)
            if not valid:
                break
            state.make_move(random.choice(valid))

    if verbose:
        elapsed = time.time() - t0
        total = len(suite)
        print(f"\nSuite built: {total} positions  ({elapsed:.1f}s)")
        print(f"  +1 won : {counts[1]}  ({100*counts[1]/max(total,1):.0f}%)")
        print(f"   0 draw: {counts[0]}  ({100*counts[0]/max(total,1):.0f}%)")
        print(f"  -1 lost: {counts[-1]}  ({100*counts[-1]/max(total,1):.0f}%)")
        print(f"  Skipped (budget exceeded): {skipped}")
        print(f"  TT entries: {len(solver._tt)}")
        if counts[-1] == 0 and total > 0:
            print("  NOTE: no lost positions -- try --max_empty 20+ "
                  "or --n_games 2000+ for a more balanced suite.")
    return suite


# --------------------------------------------------------------------------- #
# Suite I/O
# --------------------------------------------------------------------------- #

def save_suite(suite: List[SuiteEntry], path: str) -> None:
    data = [asdict(e) for e in suite]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Saved: {path}  ({len(suite)} entries)")


def load_suite(path: str) -> List[SuiteEntry]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    suite = [SuiteEntry(**d) for d in data]
    counts = {1: 0, 0: 0, -1: 0}
    for e in suite:
        counts[e.value] += 1
    print(f"Loaded: {path}  ({len(suite)} entries, "
          f"+1:{counts[1]} won  0:{counts[0]} drawn  -1:{counts[-1]} lost)")
    return suite


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Build a GOLD-seeded endgame position suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--out",       required=True,
                    help="Output JSON path for the suite.")
    ap.add_argument("--n_games",   type=int, default=1000,
                    help="Random games to generate from (default 1000).")
    ap.add_argument("--max_empty", type=int, default=15,
                    help="Only collect positions with <= this many empty cells (default 15). "
                         "UTTT games rarely reach <= 10 empty organically; 15-20 is a good range.")
    ap.add_argument("--budget",    type=int, default=100_000,
                    help="Solver node budget per position (default 100k). "
                         "Increase for deeper/harder positions; decrease for speed.")
    ap.add_argument("--seed",      type=int, default=None,
                    help="RNG seed for reproducible suite (default: unseeded).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite an existing suite file.")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        ap.error(f"{args.out} already exists -- pass --overwrite to replace it.")

    print(f"Building GOLD endgame suite:")
    print(f"  n_games={args.n_games}  max_empty={args.max_empty}  budget={args.budget:,}")
    if args.seed is not None:
        print(f"  seed={args.seed}")

    suite = build_suite(
        n_games=args.n_games,
        max_empty=args.max_empty,
        budget=args.budget,
        seed=args.seed,
        verbose=True,
    )

    if not suite:
        print("[!] Suite is empty -- no positions were solved within budget.")
        print("    Try --max_empty 20 or --budget 500000.")
        return

    save_suite(suite, args.out)


if __name__ == "__main__":
    main()
