"""Benchmark a trained net against a deep-MCTS compute oracle.

This is the concrete realization of GRADING_AND_ORACLE.md **Part 9** -- the
deep-MCTS oracle. It does NOT replace the alpha-beta endgame oracle (Part 6, exact proof)
or the external-bot benchmark (Part 8, gene-pool-independent). It is a THIRD,
complementary instrument: any-position, tunable-accuracy, EMPIRICAL strength.

Why it matters right now: the `--parallel 64` strength run trained a medium net
through the whole curriculum, but its headline ELO (4437) is **closed-loop
self-play inflation** -- meaningless as an absolute rating. Playing that net
against a strong, fixed deep-MCTS opponent gives an honest, reproducible number
that cannot inflate: the oracle searches the same fixed budget every game.

What you get: a W/D/L record of the candidate vs an MCTS opponent at a chosen
simulation budget. With `--sim_ladder` you sweep the budget and watch for the
**crossover** -- the sim count at which the candidate stops winning. That crossover
is the honest strength reading.

CAVEATS (read GRADING_AND_ORACLE.md Part 9 and agents/mcts.py):
  * The oracle is STRONG, not PROVEN-optimal. Beating it is reassuring, not QED;
    the alpha-beta oracle still certifies. Losing to it is a real, harvestable signal.
  * The ConvNet value head is unbounded (no tanh) and trained on a shaped return,
    so MCTS mixes that scale with clean +/-1 terminals (a known, non-inverting
    quality caveat). If the oracle is weirdly weak at high sims, suspect that --
    not a sign bug (the sign convention is verified in agents/test_mcts.py).

CANDIDATE MODES:
  --candidate_sims 0   (default) raw policy, no search -- deterministic
  --candidate_sims N   wraps candidate in MCTS-N; combine with --dirichlet
                       and/or --candidate_temperature for stochastic games

Run on the home box (needs torch + a checkpoint). Examples:

  # Raw candidate vs oracle ladder (deterministic baseline):
  python -m scripts.benchmark_vs_mcts --checkpoint models/league_pg/best.pt \
      --network medium --games 40 --oracle_sims 800

  # MCTS candidate vs MCTS oracle -- stochastic, diverse games:
  python -m scripts.benchmark_vs_mcts --checkpoint models/league_pg/best.pt \
      --network medium --candidate_sims 100 --dirichlet \
      --games 40 --sim_ladder 100,400,1600

  # Strength curve with stochastic oracle:
  python -m scripts.benchmark_vs_mcts --checkpoint models/league_pg/best.pt \
      --network medium --games 20 --sim_ladder 100,400,1600 --oracle_temperature 0.5
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from agents.agent_base import ModelConfigCNN
from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.mcts import MCTSAgent
from engine.game import GameState
from engine.constants import X, O, DRAW

# Single source of architecture truth.
from scripts.train_league import NETWORK_CONFIGS


def _build_pg_agent(checkpoint, network, device, tactical=False):
    """Load a PG checkpoint at the given architecture. Warns loudly on a
    shape mismatch (NeuralNetAgentPG.model_load_magic surfaces it)."""
    net = NETWORK_CONFIGS[network]
    cfg = ModelConfigCNN(
        **net,
        learning_rate=1e-4,
        label="league_pg",
        model_dir="models/league_pg",
        device=device,
    )
    agent = NeuralNetAgentPG(cfg=cfg, model_path=checkpoint, tactical=tactical)
    agent.set_eval(True)  # eval -> deterministic argmax, no gradient accumulation
    return agent


def play_match(candidate, oracle, games):
    """Play `games` between candidate and oracle, alternating who moves first
    (even games: candidate is X; odd: candidate is O) for a balanced, fully
    deterministic record. Returns (wins, draws, losses) from the candidate's
    perspective."""
    wins = draws = losses = 0
    for g in range(games):
        cand_side = X if g % 2 == 0 else O
        game = GameState()
        for a in (candidate, oracle):
            getattr(a, "clear_history", lambda: None)()

        while not game.is_over():
            mover = candidate if game.player == cand_side else oracle
            move = mover.select_move(game)
            valid, _ = game.make_move(move)
            if not valid:
                raise ValueError(f"Illegal move {move} from {mover.name}")

        if game.winner == cand_side:
            wins += 1
        elif game.winner == DRAW:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses


def _report(label, wins, draws, losses, elapsed):
    total = wins + draws + losses
    wr = 100.0 * wins / total if total else 0.0
    dr = 100.0 * draws / total if total else 0.0
    lr = 100.0 * losses / total if total else 0.0
    print(f"\n=== {label} ===")
    print(f"  games   : {total}   ({elapsed:.1f}s, {elapsed / max(total,1):.2f}s/game)")
    print(f"  candidate W/D/L : {wins}/{draws}/{losses}")
    print(f"  win-rate  : {wr:.1f}%   draw-rate: {dr:.1f}%   loss-rate: {lr:.1f}%")
    return wr


def main():
    ap = argparse.ArgumentParser(description="Benchmark a net vs a deep-MCTS oracle (GRADING_AND_ORACLE Part 9).")
    ap.add_argument("--checkpoint", type=str, default="",
                    help="Candidate net weights (.pt). REQUIRED -- e.g. models/league_pg/best.pt")
    ap.add_argument("--network", type=str, default="medium", choices=list(NETWORK_CONFIGS),
                    help="Candidate architecture (best.pt from the strength run is 'medium').")
    ap.add_argument("--games", type=int, default=40, help="Games per match (split evenly across colors).")
    ap.add_argument("--oracle_sims", type=int, default=800, help="MCTS simulations/move for the oracle.")
    ap.add_argument("--oracle_checkpoint", type=str, default="",
                    help="Policy/value net for the oracle (default: same as --checkpoint).")
    ap.add_argument("--oracle_network", type=str, default="",
                    help="Oracle architecture (default: same as --network).")
    ap.add_argument("--c_puct", type=float, default=1.5, help="PUCT exploration constant (oracle and candidate MCTS).")
    ap.add_argument("--sim_ladder", type=str, default="",
                    help="Comma list of sim counts to sweep, e.g. 100,400,1600 (overrides --oracle_sims).")
    ap.add_argument("--candidate_sims", type=int, default=0,
                    help="MCTS sims for the candidate (default 0 = raw policy). "
                         "> 0 wraps candidate in MCTSAgent; combine with --dirichlet for stochastic games.")
    ap.add_argument("--candidate_temperature", type=float, default=0.0,
                    help="Move-selection temperature for the candidate (default 0 = argmax). "
                         "> 0 samples proportional to visit^(1/T), breaking determinism.")
    ap.add_argument("--dirichlet", action="store_true",
                    help="Add Dirichlet noise at the candidate MCTS root (only meaningful with --candidate_sims > 0).")
    ap.add_argument("--dir_alpha", type=float, default=0.3,
                    help="Dirichlet alpha (default 0.3; AlphaZero uses ~0.03 for chess, ~0.3 for small branching).")
    ap.add_argument("--dir_eps", type=float, default=0.25,
                    help="Dirichlet mixing weight (default 0.25; AlphaZero uses 0.25).")
    ap.add_argument("--oracle_temperature", type=float, default=0.0,
                    help="Move-selection temperature for the oracle (default 0 = deterministic argmax).")
    ap.add_argument("--wave_size", type=int, default=1,
                    help="Batched leaf eval wave size for MCTS (1=serial original; "
                         "8-16 good for the oracle, 16-32 for candidate MCTS).")
    ap.add_argument("--tactical", action="store_true",
                    help="Give the CANDIDATE 1-ply tactical lookahead at eval (default off -> raw net).")
    ap.add_argument("--seed", type=int, default=0, help="Seed torch/numpy/random (search is deterministic anyway).")
    ap.add_argument("--device", type=str, default="",
                    help="'cpu' / 'cuda' (default: auto). MCTS on a small CNN is fine on CPU.")
    args = ap.parse_args()

    if not args.checkpoint:
        ap.error("--checkpoint is required (e.g. models/league_pg/best.pt). "
                 "The candidate's strength is meaningless without its weights.")
    if not os.path.isfile(args.checkpoint):
        ap.error(f"--checkpoint not found: {args.checkpoint}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    oracle_ckpt = args.oracle_checkpoint or args.checkpoint
    oracle_net = args.oracle_network or args.network

    print("Deep-MCTS oracle benchmark (GRADING_AND_ORACLE.md Part 9)")
    print("  This is EMPIRICAL strength: the oracle is strong, not proven-optimal.")
    print(f"  device={device}  candidate={args.checkpoint} ({args.network})")
    print(f"  oracle net={oracle_ckpt} ({oracle_net})  c_puct={args.c_puct}")

    base_candidate = _build_pg_agent(args.checkpoint, args.network, device, tactical=args.tactical)
    if args.candidate_sims > 0:
        candidate = MCTSAgent(
            base_candidate,
            n_sims=args.candidate_sims,
            c_puct=args.c_puct,
            temperature=args.candidate_temperature,
            add_dirichlet_at_root=args.dirichlet,
            dir_alpha=args.dir_alpha,
            dir_eps=args.dir_eps,
            wave_size=args.wave_size,
        )
        noise_tag = f"  dirichlet: alpha={args.dir_alpha} eps={args.dir_eps}" if args.dirichlet else ""
        print(f"  candidate: MCTS-{args.candidate_sims}  temperature={args.candidate_temperature}{noise_tag}")
    else:
        candidate = base_candidate
        if args.tactical:
            print("  candidate: 1-ply tactical lookahead ENABLED")

    sims_list = ([int(s) for s in args.sim_ladder.split(",") if s.strip()]
                 if args.sim_ladder else [args.oracle_sims])

    curve = []
    for sims in sims_list:
        oracle_inner = _build_pg_agent(oracle_ckpt, oracle_net, device)
        oracle = MCTSAgent(oracle_inner, n_sims=sims, c_puct=args.c_puct,
                           temperature=args.oracle_temperature,
                           wave_size=args.wave_size)
        t0 = time.perf_counter()
        w, d, l = play_match(candidate, oracle, args.games)
        wr = _report(f"candidate vs MCTS-{sims}", w, d, l, time.perf_counter() - t0)
        curve.append((sims, w, d, l, wr))

    if len(curve) > 1:
        print("\n=== strength curve (candidate win-rate as the oracle deepens) ===")
        print("  sims    W/D/L      win%")
        crossover = None
        for sims, w, d, l, wr in curve:
            print(f"  {sims:<6}  {w}/{d}/{l:<5}  {wr:.1f}%")
            if crossover is None and wr < 50.0:
                crossover = sims
        if crossover is not None:
            print(f"\n  Crossover: candidate drops below 50% win-rate at ~{crossover} sims.")
        else:
            print("\n  Candidate held >=50% at every sim count tested -- raise the ladder.")


if __name__ == "__main__":
    main()
