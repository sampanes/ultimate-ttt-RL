"""verify_recompute_parity.py -- prove the collect-then-recompute learn step is numerically
equivalent to the trusted in-graph batched path (learn_from_trajectories).

WHY THIS EXISTS (THROUGHPUT.md Part C): the recompute refactor decouples gradient-step count
from the self-play batch size, but its sharp edge is silent reward/state MISALIGNMENT -- if a
stored (state, action) does not line up with the reward it trains on, the loss still falls
while strength quietly rots. This script is the alignment oracle.

The in-graph learn_from_trajectories path is trusted-correct. So if a single FULL-BATCH
recompute step reproduces its loss terms (actor / value / entropy / total) to float tolerance
on the SAME trajectories, the stored (state, valid, action) <-> (log_prob, value, reward)
alignment is verified: with weights unchanged and ConvNet having no BatchNorm/Dropout, a fresh
forward must reproduce the collection-time terms EXACTLY, so a misaligned state would produce a
different log_prob and the totals would diverge -> FAIL.

Runs in clean fp32 (TF32 OFF) so the differing GEMM batch shapes between the per-ply collection
forwards and the full-batch recompute forward cannot introduce ~1e-3 reduced-precision deltas
that masquerade as a real mismatch. This tests the math + alignment, not TF32 kernels. (Real
training keeps TF32 on via train_league's set_float32_matmul_precision('high').)

    python -m scripts.verify_recompute_parity --games 60 [--network small] [--tol 1e-3]

Exit 0 = PASS (parity within tol). Exit 1 = FAIL (prints per-term deltas). No checkpoint is
written and weights are never stepped.
"""
import argparse

import torch
# Clean fp32 -- must be set before any matmul runs.
torch.set_float32_matmul_precision('highest')
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass

from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.agent_base import ModelConfigCNN
from scripts.train_league import ParallelGameRunner, NETWORK_CONFIGS
from scripts.league_manager import LeagueManager


def main():
    ap = argparse.ArgumentParser(description="Recompute-vs-in-graph parity check (THROUGHPUT Part C).")
    ap.add_argument("--games", type=int, default=60, help="Self-play games to collect for the check.")
    ap.add_argument("--network", default="small", choices=["small", "medium", "large"])
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--entropy_coef", type=float, default=0.05)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-3, help="Max allowed abs delta per loss term.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    net = NETWORK_CONFIGS[args.network]
    cfg = ModelConfigCNN(**net, learning_rate=1e-4, label="league_pg", model_dir="models/league_pg")
    active = NeuralNetAgentPG(cfg=cfg, model_path=None)   # fresh random weights, no checkpoint touched
    active.set_eval(False)  # train mode (ConvNet has no train-only stochastic layers -> deterministic)

    league = LeagueManager(population=[active], model_dir="models/league_pg/archive")
    league.set_stage(0)

    print(f"Collecting {args.games} self-play games (network={args.network}, fp32, seed {args.seed})...")
    opponents = []
    for _ in range(args.games):
        opp = league.sample_opponent(active)
        if hasattr(opp, "set_eval"):
            opp.set_eval(True)
        opponents.append(opp)
    trajectories = ParallelGameRunner(active, opponents).run(collect_inputs=True)

    # Buffer-length sanity: the recompute capture must align 1:1 with the reward/log_prob buffers.
    bad = [i for i, t in enumerate(trajectories)
           if not (len(t.states) == len(t.valids) == len(t.actions)
                   == len(t.rewards) == len(t.log_probs))]
    if bad:
        print(f"FAIL: {len(bad)} trajectories have misaligned buffer lengths (e.g. idx {bad[:5]}).")
        for i in bad[:3]:
            t = trajectories[i]
            print(f"  traj {i}: states={len(t.states)} valids={len(t.valids)} "
                  f"actions={len(t.actions)} rewards={len(t.rewards)} log_probs={len(t.log_probs)}")
        raise SystemExit(1)

    # Trusted path: loss terms WITHOUT stepping (weights stay put for the recompute pass).
    collected = active.learn_from_trajectories(
        trajectories, gamma=args.gamma, entropy_coef=args.entropy_coef,
        value_coef=args.value_coef, update=False, return_components=True)
    # Recompute path: single full-batch loss terms, no step, deterministic order.
    recompute = active.learn_from_trajectories_recompute(
        trajectories, gamma=args.gamma, entropy_coef=args.entropy_coef,
        value_coef=args.value_coef, minibatch_size=0, update=False, return_components=True)

    if not isinstance(collected, dict) or not collected or not isinstance(recompute, dict) or not recompute:
        print(f"FAIL: no transitions collected (collected={collected}, recompute={recompute}). "
              f"Increase --games.")
        raise SystemExit(1)

    print(f"\nTransitions: in-graph N={collected.get('N')}  recompute N={recompute.get('N')}")
    print(f"{'term':<10}{'in-graph':>18}{'recompute':>18}{'abs delta':>14}")
    worst = 0.0
    for k in ("actor", "value", "entropy", "total"):
        a, b = collected[k], recompute[k]
        d = abs(a - b)
        worst = max(worst, d)
        print(f"{k:<10}{a:>18.8f}{b:>18.8f}{d:>14.3e}")

    n_match = collected.get("N") == recompute.get("N")
    ok = n_match and worst <= args.tol
    print(f"\nworst abs delta = {worst:.3e}  (tol {args.tol:.0e})  | N match: {n_match}")
    print("=" * 62)
    if ok:
        print("PARITY PASS -- recompute path == in-graph path within tolerance.")
        print("Alignment verified: stored (state,valid,action) reproduce the trusted loss terms,")
        print("so --recompute is safe to enable for training.")
        raise SystemExit(0)
    print("PARITY FAIL -- recompute diverges from the trusted in-graph path.")
    print("Do NOT enable --recompute: a delta here means the stored")
    print("(state/valid/action) <-> (log_prob/value/reward) alignment is broken.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
