"""verify_opponent_batch_parity.py -- prove ParallelGameRunner's --batch_opponents path is
outcome-identical to the per-slot opponent loop, game for game.

WHY THIS EXISTS: --batch_opponents groups the opponent forward passes (clones, archives,
nn_big8, lottery) by weight identity and runs one batched argmax forward per group instead
of one select_move call per opponent move. That's a real throughput lever (it removes an
unbatched, Python-driven forward per opponent move), but its sharp edge is a SILENT change
of play -- if the batched forward or the grouping ever picked a different move, self-play
strength would drift while nothing errored. This script is that oracle.

WHY IT MUST BE EXACT: every NN opponent that reaches ParallelGameRunner is in eval mode and
picks its move by deterministic argmax over masked logits (see each agent's select_move
eval branch / batch_select_moves_eval). argmax is reorder-invariant and consumes no RNG, so
grouping/reordering opponent slots CANNOT change any chosen move. Stochastic / non-NN
opponents (random, winblock/center fallbacks, MixedAgent) are not batchable and stay in a
per-slot loop in slot order in BOTH modes, so their RNG stream is untouched either way.
Therefore the two modes must produce byte-identical games.

HOW: build one active agent (frozen -- run() only forwards, never learns) and one fixed
opponent list, then play the same batch twice, reseeding all RNG identically before each
run so the active agent's multinomial sampling, the X/O side assignment, and the stochastic
opponents' draws all line up. Compare per-game winner, side, reward sequence, AND the full
move (action) sequence. Any divergence = FAIL.

Runs in clean fp32 (TF32 OFF), same as verify_recompute_parity: batched vs single-position
GEMMs differ at ~1e-7, which can only matter at an argmax near-tie; fp32 makes even that
vanishingly unlikely and keeps the test about logic, not reduced-precision kernels.

    python -m scripts.verify_opponent_batch_parity --games 80 [--network small] [--stage 4]

Exit 0 = PASS (all games identical). Exit 1 = FAIL (prints the first divergence). No
checkpoint is written and weights are never stepped.
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


def _reseed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batchable_count(opponents):
    n = 0
    for opp in opponents:
        if hasattr(opp, "batch_select_moves_eval"):
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Opponent-batching parity check (ParallelGameRunner --batch_opponents).")
    ap.add_argument("--games", type=int, default=80, help="Self-play games per run.")
    ap.add_argument("--network", default="small", choices=["small", "medium", "large"])
    ap.add_argument("--stage", type=int, default=4,
                    help="Curriculum stage for opponent sampling. Higher stages draw more NN "
                         "opponents (clones/archives), which is exactly what batching targets.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _reseed(args.seed)

    net = NETWORK_CONFIGS[args.network]
    cfg = ModelConfigCNN(**net, learning_rate=1e-4, label="league_pg", model_dir="models/league_pg")
    active = NeuralNetAgentPG(cfg=cfg, model_path=None)   # fresh random weights, no checkpoint touched
    active.set_eval(False)  # train mode = the real parallel path (batch_select_moves samples)

    league = LeagueManager(population=[active], model_dir="models/league_pg/archive")
    try:
        league.set_stage(args.stage)
    except Exception:
        league.set_stage(0)

    # Build ONE fixed opponent list and reuse it for both runs so the two runs see identical
    # opponents (same objects, same weights). run() calls clear_history() on them and builds
    # fresh GameStates internally, so reuse is safe.
    opponents = []
    for _ in range(args.games):
        opp = league.sample_opponent(active)
        if hasattr(opp, "set_eval"):
            opp.set_eval(True)
        opponents.append(opp)

    n_batchable = _batchable_count(opponents)
    print(f"Opponents: {len(opponents)} total, {n_batchable} batchable (NN, eval argmax), "
          f"{len(opponents) - n_batchable} per-slot (stochastic/non-NN). "
          f"network={args.network} stage={args.stage} fp32 seed={args.seed}")
    if n_batchable == 0:
        print("WARNING: no batchable opponents were sampled -- the batched path is never "
              "exercised. Raise --stage or --games so clones/archives/nn_big8 appear.")

    # Run A: per-slot opponent loop (the trusted default).
    _reseed(args.seed)
    traj_a = ParallelGameRunner(active, opponents).run(collect_inputs=True, batch_opponents=False)
    # Run B: batched opponent forwards. Reseed identically first.
    _reseed(args.seed)
    traj_b = ParallelGameRunner(active, opponents).run(collect_inputs=True, batch_opponents=True)

    if len(traj_a) != len(traj_b):
        print(f"FAIL: game count differs ({len(traj_a)} vs {len(traj_b)}).")
        raise SystemExit(1)

    mismatches = []
    for i, (a, b) in enumerate(zip(traj_a, traj_b)):
        acts_a = [int(x) for x in a.actions]
        acts_b = [int(x) for x in b.actions]
        rew_a = [round(float(r), 6) for r in a.rewards]
        rew_b = [round(float(r), 6) for r in b.rewards]
        if (a.winner != b.winner or a.active_side != b.active_side
                or acts_a != acts_b or rew_a != rew_b):
            mismatches.append((i, a, b, acts_a, acts_b, rew_a, rew_b))

    print(f"\nGames compared: {len(traj_a)}   mismatches: {len(mismatches)}")
    print("=" * 62)
    if not mismatches:
        print("PARITY PASS -- every game is byte-identical with --batch_opponents on vs off.")
        print("Grouping + batched argmax does not change a single opponent move.")
        print("--batch_opponents is safe to enable for training.")
        raise SystemExit(0)

    print(f"PARITY FAIL -- {len(mismatches)} game(s) diverged. First:")
    i, a, b, acts_a, acts_b, rew_a, rew_b = mismatches[0]
    print(f"  game {i}: winner {a.winner} vs {b.winner}, side {a.active_side} vs {b.active_side}")
    print(f"    actions off: {acts_a}")
    print(f"    actions on : {acts_b}")
    if rew_a != rew_b:
        print(f"    rewards off: {rew_a}")
        print(f"    rewards on : {rew_b}")
    print("Do NOT enable --batch_opponents: a divergence means the grouping or the batched")
    print("argmax picked a different opponent move than the per-slot select_move loop.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
