"""Does more search actually make the gen-22 teacher STRONGER?

RESULT_SEARCH_DISAGREEMENT.md measured that every doubling of simulations
changes the chosen move at a flat ~14-16% rate out to 800 sims, while the visit
distribution and the root value both converge. That is the signature of a search
reshuffling among near-equivalent moves.

tools/adjudicate_move_disagreement.py then tried to referee the individual
disagreements and could not: its independent referee (gregory) sits at chance on
the subset where it commits, and its two same-network signals point in opposite
directions. Refereeing individual moves needs an oracle stronger than the thing
being tested, and no such oracle exists here.

So stop refereeing and just play. This runs the teacher against ITSELF at two
simulation counts over fixed openings with colors swapped. No referee, no
averaged-Q artifact, no circularity: if deeper search plays better, it wins.

    score > 0.5  ->  deeper search is genuinely stronger; target churn tracks
                     target improvement, and the distillation study has a real
                     independent variable.
    score ~ 0.5  ->  deeper search reshuffles among equals. The sim-count axis
                     is orthogonal to quality and the effort belongs in search
                     QUALITY (symmetry folding, transpositions, tree reuse,
                     solved-node propagation -- all unimplemented, MCTS_STATUS.md).

Sequencing note: run the WIDEST gap first (800 vs 200, two doublings). If even
that is at chance, every narrower rung is settled and no further games are
needed. Only if it separates is there a curve worth mapping.

    python -m tools.teacher_sim_ladder --pairs 800:200 --games 400
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from engine.constants import DRAW, O, X
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from scripts.expert_iter import _eval_openings, _play_fixed_match
from tools.analyze_search_disagreement import MCTS, build_model

# One seed per pair so a rerun of a single rung reproduces exactly, and so two
# different rungs never share an opening set by accident.
PAIR_SEEDS = {
    (100, 50): 7701,
    (200, 100): 7702,
    (400, 200): 7703,
    (800, 400): 7704,
    (800, 200): 7705,
    (400, 100): 7706,
    (800, 50): 7707,
    (200, 50): 7708,
}


def make_move_fn(model, device, n_sims):
    """A raw-argmax MCTS mover at a fixed simulation count.

    wave_size mirrors the analysis tool: MCTS.search clamps eff_wave to
    n_sims // _MIN_WAVES, and the 16-wave floor is load-bearing (mcts.py records
    1 wave -> 0.00 strength), so this is the largest batch that does not degrade
    the search being measured.
    """
    eff = max(1, n_sims // MCTS._MIN_WAVES)
    mcts = MCTS(model, device, n_sims=n_sims, c_puct=1.5,
                add_dirichlet_at_root=False, wave_size=eff)

    def move_fn(state, _move_num):
        pi, _root = mcts.search(state.clone())
        return int(pi.argmax())

    return move_fn


def play_match_detailed(move_fn_a, move_fn_b, n_games, seed):
    """`_play_fixed_match` that also hands back the per-game outcomes.

    The promotion gate's version returns only the aggregate score, which forces
    a binomial variance assumption. Draws contribute exactly 0.5 with no
    variance at all, so that assumption is conservative and the real interval is
    narrower -- worth knowing when a result lands near 0.5, which is precisely
    where this tool operates.

    Logic is a line-for-line mirror of scripts.expert_iter._play_fixed_match
    (same openings, same per-game reseed, same color swap). A parity check
    against it runs in --parity-check mode.
    """
    openings = _eval_openings(n_games, seed)
    py_state = random.getstate()
    np_state = np.random.get_state()
    outcomes = []
    try:
        for opening_idx, opening in enumerate(openings):
            for a_side in (X, O):
                if len(outcomes) >= n_games:
                    break
                game_seed = seed + opening_idx * 2 + (0 if a_side == X else 1)
                random.seed(game_seed)
                np.random.seed(game_seed & 0xFFFFFFFF)
                state = GameState()
                for move in opening:
                    ok, _ = state.make_move(move)
                    if not ok:
                        raise RuntimeError(
                            f"fixed opening contains illegal move {move}")
                move_num = len(opening)
                while not state.is_over():
                    fn = move_fn_a if state.player == a_side else move_fn_b
                    move = fn(state, move_num)
                    valid = rule_utl_valid_moves(
                        state.board, state.last_move, state.mini_winners)
                    if not isinstance(move, (int, np.integer)) or move not in valid:
                        raise RuntimeError(
                            f"evaluation agent returned illegal move {move}")
                    state.make_move(int(move))
                    move_num += 1
                if state.winner == a_side:
                    outcomes.append(1.0)
                elif state.winner == DRAW:
                    outcomes.append(0.5)
                else:
                    outcomes.append(0.0)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return outcomes


def outcome_ci(outcomes, z=1.96):
    """Normal-approximation CI using the OBSERVED variance of the scores.

    Unlike a binomial interval this credits draws for carrying no variance, so
    a draw-heavy match gets the tighter interval it has earned.
    """
    n = len(outcomes)
    if n == 0:
        return 0.0, (0.0, 1.0), 0.0
    arr = np.asarray(outcomes, dtype=np.float64)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean, (max(0.0, mean - z * se), min(1.0, mean + z * se)), se


def wilson_ci(score, n, z=1.96):
    """Wilson interval, which behaves near 0.5 and does not run off [0,1].

    Draws count as half a win, so this treats the score as a binomial
    proportion over n games -- slightly conservative, since a draw carries less
    variance than a coin flip.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = min(max(score, 0.0), 1.0)
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/expert_iter_v2/teacher.pt")
    ap.add_argument("--pairs", nargs="+", default=["800:200"],
                    help="deep:shallow sim pairs, e.g. 800:200 400:200")
    ap.add_argument("--games", type=int, default=400,
                    help="games per pair; colors are swapped within the run")
    ap.add_argument("--seed-override", type=int, default=0,
                    help="use this opening seed instead of the PAIR_SEEDS entry. "
                         "A different seed draws a DIFFERENT opening set, so two "
                         "runs of the same pair can be pooled for more power; "
                         "reusing the same seed just replays identical games.")
    ap.add_argument("--parity-check", action="store_true",
                    help="also run the promotion gate's own harness and assert "
                         "the aggregate score matches (doubles the cost)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="results/teacher_sim_ladder.json")
    args = ap.parse_args()

    pairs = []
    for spec in args.pairs:
        deep_s, shallow_s = spec.split(":")
        deep, shallow = int(deep_s), int(shallow_s)
        if deep <= shallow:
            raise SystemExit(f"[X] --pairs wants deep:shallow, got {spec}")
        pairs.append((deep, shallow))

    model = build_model(args.checkpoint, args.device)

    results = {}
    for deep, shallow in pairs:
        seed = args.seed_override or PAIR_SEEDS.get((deep, shallow),
                                                    7700 + deep + shallow)
        deep_fn = make_move_fn(model, args.device, deep)
        shallow_fn = make_move_fn(model, args.device, shallow)

        t0 = time.time()
        outcomes = play_match_detailed(deep_fn, shallow_fn, args.games, seed)
        dt = time.time() - t0

        if args.parity_check:
            ref = _play_fixed_match(deep_fn, shallow_fn, args.games, seed=seed)
            got = float(np.mean(outcomes))
            if abs(ref - got) > 1e-12:
                raise SystemExit(f"[X] parity: {got} != gate harness {ref}")
            print(f"  [OK] parity with _play_fixed_match: {got:.6f}")

        score, (lo, hi), se = outcome_ci(outcomes)
        wlo, whi = wilson_ci(score, args.games)
        wins = sum(1 for o in outcomes if o == 1.0)
        draws = sum(1 for o in outcomes if o == 0.5)
        losses = sum(1 for o in outcomes if o == 0.0)
        separated = lo > 0.5
        key = f"{deep}v{shallow}"
        doublings = math.log2(deep / shallow)
        results[key] = {
            "deep_sims": deep, "shallow_sims": shallow,
            "games": args.games, "seed": seed,
            "score_for_deep": score,
            "ci95": [lo, hi], "se": se,
            "ci95_binomial": [wlo, whi],
            "wins": wins, "draws": draws, "losses": losses,
            "outcomes": outcomes,
            "separated_from_chance": separated,
            "doublings": doublings,
            "per_doubling_edge": (score - 0.5) / doublings,
            "seconds": dt,
        }
        verdict = "SEPARATES" if separated else "at chance"
        print(f"  {key:>10}  {score:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"W{wins}/D{draws}/L{losses}  {dt:.0f}s  -- {verdict}")
        print(f"             per doubling {(score - 0.5) / doublings:+.4f}"
              f"   (binomial CI would be [{wlo:.4f}, {whi:.4f}])")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "checkpoint": args.checkpoint,
        "games_per_pair": args.games,
        "note": ("Self-play head to head, fixed openings, colors swapped, "
                 "raw argmax over the visit counts, no Dirichlet noise. "
                 "score_for_deep > 0.5 means the deeper search won."),
        "pairs": results,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
