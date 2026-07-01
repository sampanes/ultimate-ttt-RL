"""Grade a trained agent using the alpha-beta endgame oracle.

Implements GRADING_AND_ORACLE.md Part 6 as a runnable script.

WHAT IT DOES
  Plays `--games` games between the candidate net and `--opponent`. At every
  candidate-to-move position with <= `--max_empty` empty cells, the oracle
  tries to prove whether the candidate's actual move was optimal. Reports:

    attempted   -- candidate positions within the empty-cell filter
    proven      -- subset where BOTH solves finished within --budget nodes
    blunders    -- proven positions where the candidate played suboptimally
    blunder_rate = blunders / proven

  The oracle only counts what it CAN certify; inexact solves are skipped, not
  called clean. A low rate on a large proven set is a strong signal.

KNOWN LIMITATION (organic mode):
  Strong agents rarely enter organically losing endgames, so the proven set
  skews toward positions X is already winning.

  FIX: use --suite to grade against a pre-built GOLD-seeded position suite.
  The suite was generated from random-vs-random games, so it covers all
  outcome types (won / drawn / lost for the mover) without selection bias.
  Build one with: python -m scripts.build_endgame_suite --out suite.json

SUITE MODE (--suite path.json):
  Skips organic game play entirely. Loads a pre-built GOLD suite, cold-calls
  the agent on each position, and checks the chosen move against the stored
  optimal_moves set. No --opponent, --games, or --budget needed.

CANDIDATE SELECTION -- two paths:

  1. Arena finalist by id (reads arch from arena_state.json, no --network needed):
       --candidate arena:21@hof
       --candidate arena:22@latest

  2. Direct checkpoint with explicit architecture preset:
       --checkpoint models/league_pg/best.pt --network medium

  Exactly one of --candidate or --checkpoint is required. --network is ignored
  when --candidate is used.

EXAMPLE (home box, needs torch):
    python -m scripts.grade_agent --candidate arena:21@hof \\
        --games 5000 --opponent center --max_empty 26 --budget 300000

    python -m scripts.grade_agent --candidate arena:21@hof \\
        --tactical --games 5000 --opponent center --max_empty 26

    python -m scripts.grade_agent --checkpoint models/league_pg/best.pt \\
        --network medium --games 5000 --opponent center --max_empty 26
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from agents import get_agent
from agents.agent_base import ModelConfigCNN
from agents.neural_net_agent_pg import NeuralNetAgentPG
from engine.game import GameState
from engine.constants import X, O, DRAW, EMPTY
from engine.solver import grade_move, clear_tt
from scripts.train_league import NETWORK_CONFIGS
from scripts.benchmark_suite import resolve_candidate, _build_base_candidate
from scripts.build_endgame_suite import load_suite, state_from_entry


# --------------------------------------------------------------------------- #
# Agent loaders
# --------------------------------------------------------------------------- #

def _build_arena_candidate(spec, device, tactical=False):
    """Load an arena finalist by spec (e.g. 'arena:21@hof').
    Reads architecture from models/arena/arena_state.json via the benchmark
    suite resolver -- no --network preset required."""
    candidate_spec = resolve_candidate(spec)
    agent = _build_base_candidate(candidate_spec, torch.device(device), tactical)
    return agent, str(candidate_spec.checkpoint)


def _build_pg_agent(checkpoint, network, device, tactical=False):
    net = NETWORK_CONFIGS[network]
    cfg = ModelConfigCNN(
        **net,
        learning_rate=1e-4,
        label="league_pg",
        model_dir="models/league_pg",
        device=device,
    )
    agent = NeuralNetAgentPG(cfg=cfg, model_path=checkpoint, tactical=tactical)
    agent.set_eval(True)
    return agent


# --------------------------------------------------------------------------- #
# Play loop: collect (pre-move state, move) at candidate's turns
# --------------------------------------------------------------------------- #

def play_and_collect(candidate, opponent, games, max_empty):
    """Play `games` games, alternating who is X.

    Returns a list of (_PyGameState, move) tuples -- one per candidate-to-move
    position with <= max_empty empty cells. Both state and move are from BEFORE
    make_move was called, so the oracle sees the same position the agent saw.
    """
    history = []
    for g in range(games):
        cand_side = X if g % 2 == 0 else O
        game = GameState()

        while not game.is_over():
            is_candidate = (game.player == cand_side)
            mover = candidate if is_candidate else opponent

            move = mover.select_move(game)

            # Snapshot AFTER select_move (read-only) but BEFORE make_move.
            if is_candidate:
                empty = sum(1 for c in game.board if c == EMPTY)
                if empty <= max_empty:
                    history.append((game.clone(), move))

            valid, _ = game.make_move(move)
            if not valid:
                raise ValueError(f"Illegal move {move} from {mover.name}")

    return history


# --------------------------------------------------------------------------- #
# Grade the collected positions
# --------------------------------------------------------------------------- #

def grade_history(history, budget):
    """Oracle-grade every (state, move) pair.

    Returns (attempted, proven, blunders):
      attempted -- total positions in history
      proven    -- positions where both solves finished within budget
      blunders  -- proven positions where candidate played suboptimally
    """
    attempted = len(history)
    proven = 0
    blunders = 0
    for state, move in history:
        g = grade_move(state, move, budget)
        if g is None:
            continue   # one or both solves exceeded budget -- skip
        proven += 1
        if g.is_blunder:
            blunders += 1
    return attempted, proven, blunders


# --------------------------------------------------------------------------- #
# GOLD suite grading: cold-call the agent on pre-solved positions
# --------------------------------------------------------------------------- #

def grade_from_suite(candidate, suite):
    """Grade an agent against a pre-built GOLD endgame suite.

    No games are played. Each position is reconstructed from the suite and
    fed directly to candidate.select_move. The chosen move is compared against
    the pre-computed optimal_moves set.

    Blunder (WDL definition):
      value +1 (won): played move not in optimal_moves -> gives up the win
      value  0 (drawn): played move not in optimal_moves -> lets opponent win
      value -1 (lost): ALL moves lose -> excluded from blunder counting

    Returns (gradable, blunders, by_value) where by_value maps each value
    (+1/0/-1) to a [count, blunder_count] pair.
    """
    by_value = {1: [0, 0], 0: [0, 0], -1: [0, 0]}
    with torch.no_grad():
        for entry in suite:
            v = entry.value
            by_value[v][0] += 1
            if v == -1:
                continue   # all moves lose; nothing to penalise
            state = state_from_entry(entry)
            move  = candidate.select_move(state)
            if move not in entry.optimal_moves:
                by_value[v][1] += 1
    gradable = by_value[1][0] + by_value[0][0]
    blunders = by_value[1][1] + by_value[0][1]
    return gradable, blunders, by_value


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Grade a net against the alpha-beta oracle (GRADING_AND_ORACLE Part 6)."
    )
    ap.add_argument("--suite", type=str, default="",
                    help="Pre-built GOLD endgame suite JSON (from build_endgame_suite.py). "
                         "When set, organic play is skipped; the agent is cold-called on "
                         "each suite position and its move is checked against optimal_moves. "
                         "Build with: python -m scripts.build_endgame_suite --out suite.json")
    ap.add_argument("--candidate", type=str, default="",
                    help="Arena finalist spec (e.g. 'arena:21@hof'). "
                         "Reads arch from arena_state.json; --network is ignored.")
    ap.add_argument("--checkpoint", type=str, default="",
                    help="Candidate net weights (.pt). Used with --network.")
    ap.add_argument("--network", type=str, default="medium",
                    choices=list(NETWORK_CONFIGS),
                    help="Architecture preset for --checkpoint (ignored with --candidate).")
    ap.add_argument("--games", type=int, default=200,
                    help="Games to play (more games = larger proven set).")
    ap.add_argument("--opponent", type=str, default="winblock",
                    help="Opponent key from agents.AGENT_FACTORIES (default: winblock).")
    ap.add_argument("--max_empty", type=int, default=26,
                    help="Grade only positions with <= this many empty cells (default 26). "
                         "UTTT games rarely reach <= 15 empty cells; 26 is the usable floor.")
    ap.add_argument("--budget", type=int, default=100_000,
                    help="Node budget per oracle solve call (default 100000).")
    ap.add_argument("--tactical", action="store_true",
                    help="Enable 1-ply tactical lookahead on the candidate.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducibility (default 0).")
    ap.add_argument("--device", type=str, default="",
                    help="'cpu' or 'cuda' (default: auto).")
    args = ap.parse_args()

    if not args.candidate and not args.checkpoint:
        ap.error("Supply --candidate arena:N@hof  OR  --checkpoint path/to/net.pt --network arch")
    if args.candidate and args.checkpoint:
        ap.error("--candidate and --checkpoint are mutually exclusive")
    if args.checkpoint and not os.path.isfile(args.checkpoint):
        ap.error(f"checkpoint not found: {args.checkpoint}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("Alpha-beta oracle grading  (GRADING_AND_ORACLE.md Part 6)")
    if args.candidate:
        try:
            candidate, ckpt_path = _build_arena_candidate(args.candidate, device, args.tactical)
        except Exception as e:
            ap.error(str(e))
        print(f"  candidate  : {args.candidate}  ({ckpt_path})")
    else:
        candidate = _build_pg_agent(args.checkpoint, args.network, device, args.tactical)
        print(f"  candidate  : {args.checkpoint}  ({args.network})")
    if not args.suite:
        print(f"  opponent   : {args.opponent}")
        print(f"  games      : {args.games}   max_empty: {args.max_empty}   budget: {args.budget}")
    if args.tactical:
        print("  tactical   : ON (1-ply lookahead on candidate)")

    # ---- GOLD suite mode (--suite) -----------------------------------------
    if args.suite:
        if not os.path.isfile(args.suite):
            ap.error(f"suite not found: {args.suite}")
        suite = load_suite(args.suite)
        if not suite:
            print("[!] Suite is empty.")
            return
        print(f"  mode       : GOLD suite grading ({len(suite)} pre-solved positions)")

        t0 = time.perf_counter()
        gradable, blunders, by_value = grade_from_suite(candidate, suite)
        elapsed = time.perf_counter() - t0

        n_won,   bl_won   = by_value[1]
        n_drawn, bl_drawn = by_value[0]
        n_lost,  _        = by_value[-1]
        rate = 100.0 * blunders / gradable if gradable > 0 else float("nan")

        print(f"\n=== GOLD suite grading results ===")
        print(f"  suite size  : {len(suite)}  (+1:{n_won} won  0:{n_drawn} drawn  -1:{n_lost} lost)")
        print(f"  gradable    : {gradable}  (won + drawn; lost positions excluded)")
        print(f"  blunders    : {blunders}  ({bl_won} in won pos, {bl_drawn} in drawn pos)")
        if gradable > 0:
            print(f"  blunder rate: {rate:.2f}%")
        print(f"  grade time  : {elapsed:.1f}s")

        if gradable == 0:
            print("\n  Inconclusive: no gradable positions in suite "
                  "(all positions are value=-1 or suite is empty).")
        elif blunders == 0:
            print(f"\n  Clean: zero blunders in {gradable} GOLD-seeded positions.")
        else:
            print(f"\n  {blunders} blunder(s) in {gradable} GOLD-seeded positions.")
        return

    # ---- organic play mode (default) ---------------------------------------
    try:
        opponent = get_agent(args.opponent)
    except ValueError as e:
        ap.error(str(e))

    # ---- play games --------------------------------------------------------
    t0 = time.perf_counter()
    history = play_and_collect(candidate, opponent, args.games, args.max_empty)
    play_elapsed = time.perf_counter() - t0

    print(f"\n  Collected {len(history)} candidate positions in {play_elapsed:.1f}s"
          f"  ({args.games} games, <= {args.max_empty} empty cells)")

    if not history:
        print("  Nothing to grade -- try raising --max_empty or --games.")
        return

    # ---- grade positions ---------------------------------------------------
    clear_tt()
    t1 = time.perf_counter()
    attempted, proven, blunders = grade_history(history, args.budget)
    grade_elapsed = time.perf_counter() - t1

    not_proven = attempted - proven
    blunder_rate = 100.0 * blunders / proven if proven > 0 else float("nan")
    proven_pct = 100.0 * proven / attempted if attempted > 0 else 0.0

    print(f"\n=== Oracle grading results ===")
    print(f"  positions attempted : {attempted}")
    print(f"  proven by oracle    : {proven}  ({proven_pct:.0f}%)  "
          f"  budget-exceeded (skipped): {not_proven}")
    print(f"  blunders            : {blunders}")
    if proven > 0:
        print(f"  blunder rate        : {blunder_rate:.2f}%  "
              f"(blunders / proven positions)")
    print(f"  grade time          : {grade_elapsed:.1f}s")

    print()
    if proven == 0:
        print("  Inconclusive: no positions were solved within budget.")
        print("  Try raising --budget or lowering --max_empty.")
    elif blunders == 0:
        print(f"  Clean: zero blunders in {proven} proven positions.")
    else:
        print(f"  {blunders} blunder(s) found.  "
              f"Raise --max_empty / --games for a larger proven set,")
        print(f"  or inspect verbose output with --budget to find patterns.")

    # ---- proven-rate note --------------------------------------------------
    if not_proven > attempted * 0.5:
        print(f"\n  NOTE: {not_proven}/{attempted} positions exceeded budget.")
        print(f"  Either raise --budget or lower --max_empty for a better coverage ratio.")


if __name__ == "__main__":
    main()
