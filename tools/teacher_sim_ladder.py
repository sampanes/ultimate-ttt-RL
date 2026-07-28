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

SOLVE A/B. An arm may carry a `+solve` suffix, which turns on solved-node
propagation (agents/mcts.py). Pairing an arm against itself at the SAME sim
count is then the cleanest possible search-quality test: identical network,
identical budget, identical openings, one flag different.

    python -m tools.teacher_sim_ladder --pairs 800+solve:800 --games 400

Search-cost counters (expansions, network evaluations, wall clock per move) are
read off the MCTS objects afterwards, so the strength number and the cost it was
bought at come out of the same run rather than a separate benchmark that might
not match.
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
from tools import provenance
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

# Solve-on vs solve-off at a matched simulation count. Separate namespace so a
# solve A/B never silently reuses a sim-ladder rung's opening set.
SOLVE_AB_SEEDS = {50: 7801, 100: 7802, 200: 7803, 400: 7804, 800: 7805}


def parse_arm(spec):
    """'800' -> (800, False);  '800+solve' -> (800, True)."""
    solve = False
    if spec.endswith("+solve"):
        solve, spec = True, spec[:-len("+solve")]
    return int(spec), solve


def arm_label(sims, solve):
    return f"{sims}+solve" if solve else str(sims)


def make_move_fn(model, device, n_sims, solve=False):
    """A raw-argmax MCTS mover at a fixed simulation count.

    wave_size mirrors the analysis tool: MCTS.search clamps eff_wave to
    n_sims // _MIN_WAVES, and the 16-wave floor is load-bearing (mcts.py records
    1 wave -> 0.00 strength), so this is the largest batch that does not degrade
    the search being measured.

    Returns (move_fn, mcts) so the caller can read the cost counters off the
    same object that played the games.
    """
    eff = max(1, n_sims // MCTS._MIN_WAVES)
    mcts = MCTS(model, device, n_sims=n_sims, c_puct=1.5,
                add_dirichlet_at_root=False, wave_size=eff, solve=solve)

    def move_fn(state, _move_num):
        pi, _root = mcts.search(state.clone())
        return int(pi.argmax())

    return move_fn, mcts


def cost_stats(mcts):
    """Per-move search cost, as the report asks for it.

    Wall clock comes from the MCTS's own accumulator, not the match elapsed
    time: a match's total is shared by both arms and cannot be split afterwards.
    """
    n = max(1, mcts.stat_searches)
    return {
        "moves": mcts.stat_searches,
        "expanded_nodes_per_move": mcts.stat_expansions / n,
        "nn_evals_per_move": mcts.stat_nn_evals / n,
        "nn_batches_per_move": mcts.stat_nn_batches / n,
        "terminal_probes_per_move": mcts.stat_probes / n,
        "solved_roots": mcts.stat_solved_roots,
        "solved_root_rate": mcts.stat_solved_roots / n,
        "seconds_per_move": mcts.stat_seconds / n,
    }


def equal_wall_clock(score, deep, shallow):
    """Strength per unit compute, reported but NOT used to gate anything.

    Equal-simulation and equal-time are different questions. Solving buys
    correctness at every budget; whether it also buys strength per SECOND
    depends on how the redirected budget trades off, and the two must not be
    conflated -- a correctness fix that costs time is still a correctness fix.
    So this block exists to be read alongside the gates, never inside them.

    strength_per_second is the deep arm's edge over chance divided by its
    per-move cost. It is a ratio of two measured things, not a model, and it is
    only comparable BETWEEN rows of the same table.
    """
    edge = score - 0.5
    sd, ss = deep["seconds_per_move"], shallow["seconds_per_move"]
    return {
        "strength_delta": edge,
        "wall_clock_ratio": (sd / ss) if ss else None,
        "neural_eval_delta": deep["nn_evals_per_move"] - shallow["nn_evals_per_move"],
        "expansion_delta": (deep["expanded_nodes_per_move"]
                            - shallow["expanded_nodes_per_move"]),
        "probe_delta": (deep["terminal_probes_per_move"]
                        - shallow["terminal_probes_per_move"]),
        "strength_per_second": (edge / sd) if sd else None,
        "deep_seconds_per_move": sd,
        "shallow_seconds_per_move": ss,
        "time_normalised_edge": (edge / (sd / ss)) if ss and sd else None,
        "gating": "REPORT ONLY -- never an input to the four pilot gates",
    }


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
        a_s, b_s = spec.split(":")
        a, b = parse_arm(a_s), parse_arm(b_s)
        # Equal sims is legal ONLY when the two arms differ in some other way
        # (currently: solving). Otherwise the "pair" is a net against itself.
        if a[0] < b[0] or a == b:
            raise SystemExit(
                f"[X] --pairs wants deep:shallow or an equal-sims solve A/B, "
                f"got {spec}")
        pairs.append((a, b))

    model = build_model(args.checkpoint, args.device)

    results = {}
    for (deep, deep_solve), (shallow, shallow_solve) in pairs:
        if deep == shallow:
            default_seed = SOLVE_AB_SEEDS.get(deep, 7800 + deep)
        else:
            default_seed = PAIR_SEEDS.get((deep, shallow), 7700 + deep + shallow)
        seed = args.seed_override or default_seed
        deep_fn, deep_mcts = make_move_fn(model, args.device, deep, deep_solve)
        shallow_fn, shallow_mcts = make_move_fn(model, args.device, shallow,
                                                shallow_solve)

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
        a_label, b_label = arm_label(deep, deep_solve), arm_label(shallow, shallow_solve)
        key = f"{a_label}v{b_label}"
        doublings = math.log2(deep / shallow) if deep != shallow else 0.0
        results[key] = {
            "deep_sims": deep, "shallow_sims": shallow,
            "deep_solve": deep_solve, "shallow_solve": shallow_solve,
            "games": args.games, "seed": seed,
            "score_for_deep": score,
            "ci95": [lo, hi], "se": se,
            "ci95_binomial": [wlo, whi],
            "wins": wins, "draws": draws, "losses": losses,
            "outcomes": outcomes,
            "separated_from_chance": separated,
            "doublings": doublings,
            "per_doubling_edge": ((score - 0.5) / doublings if doublings else None),
            "seconds": dt,
            "cost_deep": cost_stats(deep_mcts),
            "cost_shallow": cost_stats(shallow_mcts),
            "equal_wall_clock": equal_wall_clock(
                score, cost_stats(deep_mcts), cost_stats(shallow_mcts)),
        }
        verdict = "SEPARATES" if separated else "at chance"
        print(f"  {key:>16}  {score:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"W{wins}/D{draws}/L{losses}  {dt:.0f}s  -- {verdict}")
        if doublings:
            print(f"             per doubling {(score - 0.5) / doublings:+.4f}"
                  f"   (binomial CI would be [{wlo:.4f}, {whi:.4f}])")
        else:
            print(f"             equal budget, so this is pure search quality"
                  f"   (binomial CI would be [{wlo:.4f}, {whi:.4f}])")
        for tag, m in ((a_label, deep_mcts), (b_label, shallow_mcts)):
            c = cost_stats(m)
            print(f"             {tag:>12}: {c['expanded_nodes_per_move']:7.1f} "
                  f"expanded  {c['nn_evals_per_move']:7.1f} nn-evals  "
                  f"{c['terminal_probes_per_move']:7.1f} probes  "
                  f"{c['seconds_per_move'] * 1000:7.1f} ms/move  "
                  f"solved-root {c['solved_root_rate']:.3f}")
        e = results[key]["equal_wall_clock"]
        print(f"             equal-wall-clock (report only): "
              f"strength {e['strength_delta']:+.4f}  "
              f"time x{(e['wall_clock_ratio'] or float('nan')):.3f}  "
              f"nn {e['neural_eval_delta']:+.1f}  "
              f"exp {e['expansion_delta']:+.1f}  "
              f"edge/s {(e['strength_per_second'] or float('nan')):+.3f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "checkpoint": args.checkpoint,
        "games_per_pair": args.games,
        "note": ("Self-play head to head, fixed openings, colors swapped, "
                 "raw argmax over the visit counts, no Dirichlet noise. "
                 "score_for_deep > 0.5 means the deeper search won."),
        "pairs": results,
        "provenance": provenance.build(),
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
