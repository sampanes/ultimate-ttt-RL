"""AlphaZero-style self-play training for Ultimate Tic-Tac-Toe.

Core loop per iteration:
  1. Self-play: play `--games_per_iter` games using MCTS with Dirichlet noise.
     Each move records (board_tensor, visit_pi, outcome_z).
  2. Add examples to a fixed-size replay buffer.
  3. Train: sample random minibatches, optimize
       policy_loss (cross-entropy vs visit pi) + value_coef * value_loss (MSE vs z).
  4. Checkpoint + log metrics.

Policy target: MCTS visit count distribution (not the sampled action).
Value target: game outcome from that position's player perspective
              (+1 win, -1 loss, 0 draw).

Relationship to train_league.py:
  train_league uses policy-gradient actor-critic from *sampled* game rollouts.
  train_alphazero uses MCTS-improved *planned* policy targets. The two are
  complementary: league-training produces the initial policy; AZ refines it with
  search. Start from a league checkpoint via --checkpoint.

value_tanh recommendation:
  Use --value_tanh. Without tanh the value head is unbounded (trained on shaped
  returns), which means terminal values (+/-1) and leaf values use different scales
  -- the known MCTS quality caveat in agents/mcts.py. --value_tanh fixes this for
  new runs. Start from a *fresh* model or a checkpoint trained with --value_tanh;
  loading an untanhed checkpoint and immediately enabling --value_tanh produces
  garbage (the value head output range changes). A warning is printed if omitted.

Examples:
  # Fresh AZ run, small net, verify it runs before long commitment:
  python -m scripts.train_alphazero --network small --value_tanh --iters 3 --games_per_iter 10

  # Full run seeded from a league checkpoint:
  python -m scripts.train_alphazero \\
    --checkpoint models/league_pg/best.pt \\
    --network medium --value_tanh \\
    --n_sims 200 --wave_size 8 \\
    --games_per_iter 100 --buffer_size 50000 \\
    --train_steps 200 --batch_size 256 \\
    --iters 0
"""

import argparse
import collections
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agents.agent_base import ModelConfigCNN, board_to_tensor_from_gamestate
from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.mcts import MCTS
from engine.game import GameState
from engine.constants import X, O, DRAW
from scripts.train_league import NETWORK_CONFIGS, prune_versions, DEFAULT_ELO
from scripts.trainer_base import append_metrics, clear_metrics_log

# A single training example produced by self-play.
# x     : (7, 9, 9) float32 board tensor (cpu)
# pi    : (81,) float32 visit distribution (sums to 1 over legal moves)
# z     : float -- game outcome from that position's player perspective (+1/-1/0)
Example = Tuple[torch.Tensor, np.ndarray, float]


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self._buf: collections.deque = collections.deque(maxlen=maxlen)

    def extend(self, examples: List[Example]):
        self._buf.extend(examples)

    def sample(self, n: int) -> List[Example]:
        return random.sample(self._buf, min(n, len(self._buf)))

    def __len__(self):
        return len(self._buf)


@torch.no_grad()
def collect_game(model, device, n_sims: int, c_puct: float,
                 dir_alpha: float, dir_eps: float,
                 wave_size: int, temperature_moves: int) -> List[Example]:
    """Play one self-play game using MCTS; return a list of (x, pi, z) examples."""
    mcts = MCTS(model, device, n_sims=n_sims, c_puct=c_puct,
                add_dirichlet_at_root=True, dir_alpha=dir_alpha, dir_eps=dir_eps,
                wave_size=wave_size)

    state = GameState()
    trajectory = []  # (tensor, pi, player)
    move_num = 0

    while not state.is_over():
        pi, _ = mcts.search(state)

        x = board_to_tensor_from_gamestate(state).cpu()
        trajectory.append((x, pi.copy(), state.player))

        if move_num < temperature_moves:
            # Proportional sampling with temperature=1 for early moves (exploration).
            pi_sum = pi.sum()
            if pi_sum > 0:
                probs = pi / pi_sum
            else:
                probs = np.ones(81) / 81
            move = int(np.random.choice(81, p=probs))
        else:
            move = int(np.argmax(pi))

        state.make_move(move)
        move_num += 1

    # Assign outcome z from each position's player perspective.
    winner = state.winner
    examples: List[Example] = []
    for x, pi, player in trajectory:
        if winner == DRAW:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0
        examples.append((x, pi, z))

    return examples


def train_on_examples(model, optimizer, examples: List[Example],
                      value_coef: float, device: str) -> Tuple[float, float, float]:
    """Run one gradient step on a batch. Returns (total_loss, policy_loss, value_loss)."""
    model.train()

    xs = torch.stack([e[0] for e in examples]).to(device)       # (B, 7, 9, 9)
    pis = torch.tensor(np.stack([e[1] for e in examples]),
                       dtype=torch.float32, device=device)       # (B, 81)
    zs = torch.tensor([e[2] for e in examples],
                      dtype=torch.float32, device=device)        # (B,)

    policy_logits, values = model.forward_both(xs)               # (B,81), (B,)
    if policy_logits.dim() == 1:
        policy_logits = policy_logits.unsqueeze(0)
        values = values.unsqueeze(0)

    # Policy: cross-entropy between visit distribution and log_softmax of logits.
    # Using KL-divergence form: -sum(pi * log_softmax) = CE(pi, logits).
    log_probs = F.log_softmax(policy_logits, dim=-1)        # (B, 81)
    policy_loss = -(pis * log_probs).sum(dim=-1).mean()     # scalar

    # Value: MSE between predicted value and outcome z.
    value_loss = F.mse_loss(values, zs)

    loss = policy_loss + value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


def _eval_winrate(agent: NeuralNetAgentPG, n_games: int = 50) -> float:
    """Quick winrate vs random to track learning progress."""
    from agents.random_agent import RandomAgent
    from engine.constants import X, O, DRAW
    rand = RandomAgent()
    wins = 0
    for g in range(n_games):
        state = GameState()
        agent_side = X if g % 2 == 0 else O
        agent.set_eval(True)
        while not state.is_over():
            mover = agent if state.player == agent_side else rand
            move = mover.select_move(state)
            state.make_move(move)
        if state.winner == agent_side:
            wins += 1
    agent.set_eval(False)
    return wins / n_games


def main():
    ap = argparse.ArgumentParser(description="AlphaZero-style self-play training.")
    ap.add_argument("--checkpoint", type=str, default="",
                    help="Starting weights (.pt). Leave empty for fresh random init. "
                         "Load a league checkpoint to jump-start policy quality.")
    ap.add_argument("--network", type=str, default="small", choices=list(NETWORK_CONFIGS),
                    help="Network architecture (small/medium/large).")
    ap.add_argument("--value_tanh", action=argparse.BooleanOptionalAction, default=False,
                    help="Apply tanh to value head (strongly recommended). "
                         "Calibrates output to [-1,1], fixing the MCTS value-scale mismatch. "
                         "Start from a fresh model or a --value_tanh checkpoint.")
    ap.add_argument("--iters", type=int, default=0,
                    help="Training iterations (0 = run forever until Ctrl+C).")
    ap.add_argument("--games_per_iter", type=int, default=50,
                    help="Self-play games per iteration.")
    ap.add_argument("--n_sims", type=int, default=100,
                    help="MCTS simulations per move during self-play.")
    ap.add_argument("--wave_size", type=int, default=1,
                    help="Batched leaf-eval wave size (1=serial; 8+ speeds up GPU).")
    ap.add_argument("--c_puct", type=float, default=1.5)
    ap.add_argument("--dir_alpha", type=float, default=0.3,
                    help="Dirichlet alpha for root noise (~0.3 for UTTT).")
    ap.add_argument("--dir_eps", type=float, default=0.25,
                    help="Dirichlet mixing weight (AlphaZero uses 0.25).")
    ap.add_argument("--temperature_moves", type=int, default=20,
                    help="Number of moves to sample proportionally (exploration); "
                         "after this, use argmax (exploitation).")
    ap.add_argument("--buffer_size", type=int, default=100_000,
                    help="Maximum replay buffer size (examples, not games).")
    ap.add_argument("--train_steps", type=int, default=100,
                    help="Gradient steps per iteration.")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="SGD batch size drawn from the replay buffer.")
    ap.add_argument("--value_coef", type=float, default=1.0,
                    help="Weight on value loss. 1.0 = equal weight (AlphaZero default).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model_dir", type=str, default="models/alphazero")
    ap.add_argument("--keep_versions", type=int, default=5,
                    help="Number of version_NNN.pt files to keep.")
    ap.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=True,
                    help="Save models/alphazero/best.pt whenever eval winrate improves. DEFAULT ON. Pass --no-save_best to disable.")
    ap.add_argument("--eval_games", type=int, default=40,
                    help="Games vs random per eval (0 = skip eval).")
    ap.add_argument("--eval_every", type=int, default=5,
                    help="Evaluate every N iterations.")
    ap.add_argument("--device", type=str, default="",
                    help="'cpu' / 'cuda' (default: auto).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed torch/numpy/random (weight init, self-play sampling).")
    ap.add_argument("--no_metrics", action="store_true",
                    help="Skip writing to loss_logs/metrics_log.jsonl.")
    args = ap.parse_args()

    if not args.value_tanh:
        print("[!] --value_tanh not set. Value head is unbounded (trained on shaped returns).")
        print("    MCTS will mix that scale with clean +/-1 terminal values.")
        print("    For best AlphaZero quality, use --value_tanh on a fresh model.")

    if args.seed is not None:
        import random as _r
        _r.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seeded RNGs: {args.seed}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.model_dir, exist_ok=True)

    net = NETWORK_CONFIGS[args.network]
    cfg = ModelConfigCNN(
        **net,
        learning_rate=args.lr,
        label="alphazero",
        model_dir=args.model_dir,
        value_tanh=args.value_tanh,
    )
    agent = NeuralNetAgentPG(cfg=cfg, model_path=None)
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            ap.error(f"--checkpoint not found: {args.checkpoint}")
        agent.seed_from_checkpoint(args.checkpoint)
        print(f"Seeded weights from {args.checkpoint}")

    agent.model.to(device)
    agent.device = device
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)

    if not args.no_metrics:
        clear_metrics_log()

    buffer = ReplayBuffer(args.buffer_size)
    version_idx = 0
    best_wr = 0.0

    print(f"AlphaZero training | net={args.network} | device={device} | "
          f"n_sims={args.n_sims} wave={args.wave_size} | value_tanh={args.value_tanh}")
    print(f"  games_per_iter={args.games_per_iter} | train_steps={args.train_steps} | "
          f"batch_size={args.batch_size} | buffer_size={args.buffer_size}")

    iteration = 0
    try:
        while args.iters == 0 or iteration < args.iters:
            t0 = time.perf_counter()

            # --- Self-play ---
            agent.model.eval()
            new_examples: List[Example] = []
            for g in range(args.games_per_iter):
                exs = collect_game(
                    model=agent.model,
                    device=device,
                    n_sims=args.n_sims,
                    c_puct=args.c_puct,
                    dir_alpha=args.dir_alpha,
                    dir_eps=args.dir_eps,
                    wave_size=args.wave_size,
                    temperature_moves=args.temperature_moves,
                )
                new_examples.extend(exs)
            buffer.extend(new_examples)

            # --- Train ---
            total_loss = policy_loss_sum = value_loss_sum = 0.0
            steps_done = 0
            for _ in range(args.train_steps):
                if len(buffer) < args.batch_size:
                    break
                batch = buffer.sample(args.batch_size)
                tl, pl, vl = train_on_examples(
                    agent.model, optimizer, batch,
                    value_coef=args.value_coef, device=device,
                )
                total_loss += tl
                policy_loss_sum += pl
                value_loss_sum += vl
                steps_done += 1
            agent.model.eval()

            avg_loss   = total_loss / max(steps_done, 1)
            avg_pol    = policy_loss_sum / max(steps_done, 1)
            avg_val    = value_loss_sum / max(steps_done, 1)
            elapsed    = time.perf_counter() - t0

            # --- Checkpoint ---
            version_path = os.path.join(args.model_dir, f"version_{version_idx:03d}.pt")
            agent.save(version_path)
            prune_versions(args.model_dir, args.keep_versions)
            version_idx += 1

            # --- Eval ---
            wr = float('nan')
            if args.eval_games > 0 and (iteration % args.eval_every == 0):
                wr = _eval_winrate(agent, args.eval_games)
                if args.save_best and wr > best_wr:
                    best_wr = wr
                    best_path = os.path.join(args.model_dir, "best.pt")
                    agent.save(best_path)

            # --- Metrics ---
            if not args.no_metrics:
                append_metrics(
                    loss=avg_loss,
                    epsilon=float('nan'),
                    winrate=wr if not np.isnan(wr) else 0.0,
                    value_loss=avg_val,
                    t=time.time(),
                    policy_loss=avg_pol,
                    games_total=(iteration + 1) * args.games_per_iter,
                    buffer=len(buffer),
                )

            wr_str = f"{wr*100:.1f}%" if not (isinstance(wr, float) and np.isnan(wr)) else "--"
            print(
                f"iter {iteration+1:4d} | "
                f"buf={len(buffer):6d} | "
                f"loss={avg_loss:.4f} (pol={avg_pol:.4f} val={avg_val:.4f}) | "
                f"wr_vs_rand={wr_str} | "
                f"{elapsed:.1f}s"
            )

            iteration += 1

    except KeyboardInterrupt:
        print("\nInterrupted. Saving final checkpoint...")
        agent.save(os.path.join(args.model_dir, "final.pt"))
        print("Saved.")


if __name__ == "__main__":
    main()
