"""Expert iteration: distill a strong MCTS teacher into a student net.

WHY (2026-07-04): two AlphaZero-from-scratch runs (M4a/M4b) flatlined or
regressed. Root cause was NOT the paradigm alone -- agents/mcts.py had an
inverted virtual-loss sign plus an unclamped wave size that together made
wave-batched search WEAKER than the raw policy it wrapped, poisoning every
visit target (both fixed, see agents/mcts.py + agents/test_mcts.py). But even
with search fixed, bootstrapping from random weights at ~1 game/s is
data-starved. This script starts from PROVEN strength instead:

    teacher = MCTS(teacher_sims) over models/league_pg/best.pt
    (benchmark 2026-06-30: 16-sim MCTS beats raw best.pt 100%;
     edge re-verified 0.875-0.925 after the wave fixes)

Loop, forever (Ctrl+C safe, --resume picks up where it left off):
  1. GENERATE: teacher self-play games via train_alphazero.collect_game
     (Dirichlet root noise + early-move temperature for diversity, tactical
     win-in-1 / losing-move ground truth applied to targets). Positions are
     appended to an in-RAM window AND written to disk shards for resume.
  2. TRAIN: student (fresh net, tanh value head) supervised on the window:
     CE on visit distributions + MSE on game outcomes.
  3. GATE (wall-clock paced, dashboard-friendly): student raw vs random /
     WinBlockAgent / teacher raw, plus the search invariant (MCTS over
     student must score >= 0.5 vs raw student -- below that, halt and debug).
  4. PROMOTE: when MCTS(student) beats MCTS(teacher) >= --promote_thresh,
     the student's weights BECOME the teacher and generation increments.
     The shipped artifact is always student + search at eval time.

Metrics go to loss_logs/metrics_log.jsonl -- the AZ dashboard
(gui/alphazero/index.html) renders them unchanged: 'winrate' = student raw
vs random, wr_heur = vs win/block bot, wr_anchor = vs CURRENT TEACHER raw,
mcts_edge = search invariant, teacher_gen = promotions so far.

Run (cmd, from repo root; start_goat.bat / stop_goat.bat wrap this):
    set CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv\\Scripts\\python -m scripts.expert_iter --resume
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from agents.agent_base import ModelConfigCNN
from engine.constants import DRAW
from agents.deterministics import WinBlockAgent
from agents.mcts import MCTS
from agents.neural_net_agent_pg import NeuralNetAgentPG
from agents.random_agent import RandomAgent
from scripts.train_alphazero import (NETWORK_CONFIGS, ReplayBuffer,
                                     _play_match, _policy_move, collect_game,
                                     train_on_examples)
from scripts.trainer_base import append_metrics, clear_metrics_log, format_elapsed


# --------------------------------------------------------------------------- #
# Move-function builders (gauntlet protocol: sample first plies for variety)
# --------------------------------------------------------------------------- #

def _raw_fn(model, device, sample_moves=6):
    return lambda s, mn: _policy_move(model, device, s, sample_moves, mn)


def _search_fn(model, device, sims, sample_moves=6):
    mcts = MCTS(model, device, n_sims=sims, c_puct=1.5,
                add_dirichlet_at_root=False, wave_size=64)

    def f(state, move_num):
        pi, _ = mcts.search(state)
        if move_num < sample_moves:
            s = pi.sum()
            p = pi / s if s > 0 else np.ones(81) / 81
            return int(np.random.choice(81, p=p))
        return int(np.argmax(pi))
    return f


def _agent_fn(agent):
    return lambda s, mn: agent.select_move(s)


# --------------------------------------------------------------------------- #
# Shard store: every generation block is persisted so --resume never loses data
# --------------------------------------------------------------------------- #

class ShardStore:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _path(self, idx):
        return os.path.join(self.data_dir, f"shard_{idx:05d}.pt")

    def write(self, idx, examples, teacher_gen):
        xs = torch.stack([e[0] for e in examples])
        pis = torch.from_numpy(np.stack([e[1] for e in examples]))
        zs = torch.tensor([e[2] for e in examples], dtype=torch.float32)
        torch.save({"x": xs, "pi": pis, "z": zs, "teacher_gen": teacher_gen},
                   self._path(idx))

    def load_window(self, last_idx, max_examples):
        """Newest-first reload into (examples, count) until max_examples."""
        examples = []
        idx = last_idx
        while idx >= 0 and len(examples) < max_examples:
            p = self._path(idx)
            if os.path.isfile(p):
                d = torch.load(p, map_location="cpu", weights_only=False)
                for i in range(d["x"].shape[0]):
                    examples.append((d["x"][i], d["pi"][i].numpy(),
                                     float(d["z"][i])))
            idx -= 1
        examples.reverse()   # oldest first, so the deque evicts oldest
        return examples


# --------------------------------------------------------------------------- #
# Checkpoint shells
# --------------------------------------------------------------------------- #

def _make_agent(network, device, value_tanh, model_dir, lr):
    cfg = ModelConfigCNN(**NETWORK_CONFIGS[network], learning_rate=lr,
                         label="expert_iter", model_dir=model_dir,
                         value_tanh=value_tanh)
    a = NeuralNetAgentPG(cfg=cfg, model_path=None)
    a.model.to(device)
    a.device = device
    a.model.eval()
    return a


def _save_teacher(path, model, value_tanh, gen):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in
                               model.state_dict().items()},
                "value_tanh": value_tanh, "gen": gen}, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Expert iteration: MCTS teacher -> student distillation.")
    ap.add_argument("--network", choices=list(NETWORK_CONFIGS), default="medium")
    ap.add_argument("--teacher_ckpt", type=str,
                    default=os.path.join("models", "league_pg", "best.pt"),
                    help="Gen-0 teacher weights (ignored on --resume with state).")
    ap.add_argument("--teacher_tanh", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Gen-0 teacher has a tanh value head (league best.pt "
                         "does NOT; promoted students always do).")
    ap.add_argument("--teacher_sims", type=int, default=200,
                    help="MCTS sims per teacher move during generation.")
    ap.add_argument("--games_per_block", type=int, default=16,
                    help="Teacher self-play games per generate/train block.")
    ap.add_argument("--train_steps", type=int, default=100,
                    help="SGD steps per block.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value_coef", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=200_000,
                    help="Max positions in the training window.")
    ap.add_argument("--min_window", type=int, default=5_000,
                    help="Do not train until this many positions exist.")
    ap.add_argument("--dir_eps", type=float, default=0.10,
                    help="Dirichlet mix at the teacher's search root.")
    ap.add_argument("--dir_alpha", type=float, default=0.3)
    ap.add_argument("--temperature_moves", type=int, default=10)
    ap.add_argument("--gate_min", type=float, default=5.0,
                    help="Minutes between gate evals (raw matches + edge probe).")
    ap.add_argument("--gate_games", type=int, default=40)
    ap.add_argument("--edge_sims", type=int, default=64,
                    help="Sims for the search-invariant probe.")
    ap.add_argument("--edge_games", type=int, default=20)
    ap.add_argument("--promote_min", type=float, default=30.0,
                    help="Minutes between promotion matches (search vs search).")
    ap.add_argument("--promote_sims", type=int, default=64)
    ap.add_argument("--promote_games", type=int, default=40)
    ap.add_argument("--promote_thresh", type=float, default=0.55,
                    help="MCTS(student) score vs MCTS(teacher) to promote.")
    ap.add_argument("--model_dir", type=str,
                    default=os.path.join("models", "expert_iter"))
    ap.add_argument("--blocks", type=int, default=0, help="0 = run forever.")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from model_dir state if present, else fresh.")
    ap.add_argument("--no_metrics", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.model_dir, exist_ok=True)
    state_path = os.path.join(args.model_dir, "state.json")
    teacher_path = os.path.join(args.model_dir, "teacher.pt")
    student_path = os.path.join(args.model_dir, "student.pt")
    resume_path = os.path.join(args.model_dir, "resume.pt")
    store = ShardStore(os.path.join(args.model_dir, "data"))

    resuming = args.resume and os.path.isfile(state_path)

    # ---- teacher ----------------------------------------------------------
    if resuming:
        payload = torch.load(teacher_path, map_location="cpu",
                             weights_only=False)
        teacher_tanh = payload["value_tanh"]
        teacher_gen = payload["gen"]
        teacher = _make_agent(args.network, device, teacher_tanh,
                              args.model_dir, args.lr)
        teacher.model.load_state_dict(payload["state_dict"])
    else:
        teacher_tanh = args.teacher_tanh
        teacher_gen = 0
        teacher = _make_agent(args.network, device, teacher_tanh,
                              args.model_dir, args.lr)
        # STRICT load: a partially-transferred gen-0 teacher would silently
        # generate garbage expert data. Arch mismatch must be a hard error.
        payload = torch.load(args.teacher_ckpt, map_location="cpu",
                             weights_only=False)
        sd = payload["state_dict"] if (isinstance(payload, dict)
                                       and "state_dict" in payload) else payload
        teacher.model.load_state_dict(sd)
        _save_teacher(teacher_path, teacher.model, teacher_tanh, teacher_gen)
        print(f"Gen-0 teacher seeded from {args.teacher_ckpt}")
    teacher.model.eval()

    # ---- student (always tanh: clean [-1,1] value for search) -------------
    student = _make_agent(args.network, device, True, args.model_dir, args.lr)
    optimizer = torch.optim.Adam(student.model.parameters(), lr=args.lr)

    block = 0
    shard_idx = -1
    games_total = 0
    steps_total = 0
    buffer = ReplayBuffer(args.window)

    if resuming:
        with open(state_path) as f:
            st = json.load(f)
        block = st["block"]
        shard_idx = st["shard_idx"]
        games_total = st["games_total"]
        steps_total = st["steps_total"]
        teacher_gen = st["teacher_gen"]
        sd = torch.load(student_path, map_location=device, weights_only=False)
        student.model.load_state_dict(sd["state_dict"])
        if os.path.isfile(resume_path):
            opt_sd = torch.load(resume_path, map_location="cpu",
                                weights_only=False)
            optimizer.load_state_dict(opt_sd["optimizer"])
        buffer.extend(store.load_window(shard_idx, args.window))
        print(f"Resumed: block {block}, {games_total} games, "
              f"{len(buffer)} positions reloaded, teacher gen {teacher_gen}")
    else:
        if not args.no_metrics:
            clear_metrics_log()

    heur = WinBlockAgent()
    rnd = RandomAgent()
    last_gate_t = 0.0      # fire the first gate after the first block
    last_promote_t = time.time()
    t_start = time.time()

    def save_all():
        student.save(student_path, verbose=False)
        torch.save({"optimizer": optimizer.state_dict()}, resume_path)
        with open(state_path, "w") as f:
            json.dump({"block": block, "shard_idx": shard_idx,
                       "games_total": games_total, "steps_total": steps_total,
                       "teacher_gen": teacher_gen}, f)

    print(f"Expert iteration on {device} | network={args.network} | "
          f"teacher_sims={args.teacher_sims} | window={args.window}")

    try:
        while args.blocks == 0 or block < args.blocks:
            t0 = time.perf_counter()

            # ---- 1. GENERATE ------------------------------------------------
            teacher.model.eval()
            new_examples = []
            draws = 0
            moves_total = 0
            with torch.no_grad():
                for _ in range(args.games_per_block):
                    exs, winner, gstats = collect_game(
                        model=teacher.model,
                        device=device,
                        n_sims=args.teacher_sims,
                        c_puct=1.5,
                        dir_alpha=args.dir_alpha,
                        dir_eps=args.dir_eps,
                        wave_size=64,   # MCTS clamps to n_sims // 16 internally
                        temperature_moves=args.temperature_moves,
                        use_tactics=True,
                        opponent_fn=None,
                    )
                    new_examples.extend(exs)
                    moves_total += gstats["moves"]
                    if winner == DRAW:
                        draws += 1
            games_total += args.games_per_block
            shard_idx += 1
            store.write(shard_idx, new_examples, teacher_gen)
            buffer.extend(new_examples)

            pis = np.stack([e[1] for e in new_examples])
            pi_ent = float(-(pis * np.log(pis + 1e-12)).sum(axis=1).mean())
            gen_secs = time.perf_counter() - t0

            # ---- 2. TRAIN ---------------------------------------------------
            avg_loss = avg_pol = avg_val = float("nan")
            if len(buffer) >= args.min_window:
                tl = pl = vl = 0.0
                for _ in range(args.train_steps):
                    batch = buffer.sample(args.batch_size)
                    l, p, v = train_on_examples(student.model, optimizer, batch,
                                                value_coef=args.value_coef,
                                                device=device)
                    tl += l
                    pl += p
                    vl += v
                avg_loss = tl / args.train_steps
                avg_pol = pl / args.train_steps
                avg_val = vl / args.train_steps
                steps_total += args.train_steps
            student.model.eval()

            # ---- 3. GATE (wall-clock paced) ----------------------------------
            extra = {
                "sp_draws": round(draws / args.games_per_block, 3),
                "pi_ent": round(pi_ent, 3),
                "avg_len": round(moves_total / args.games_per_block, 1),
                "teacher_gen": teacher_gen,
            }
            wr_random = float("nan")
            gate_line = ""
            with torch.no_grad():
                if time.time() - last_gate_t >= args.gate_min * 60:
                    tg0 = time.perf_counter()
                    stu_fn = _raw_fn(student.model, device)
                    wr_random = _play_match(stu_fn, _agent_fn(rnd),
                                            args.gate_games)
                    extra["wr_heur"] = round(_play_match(
                        stu_fn, _agent_fn(heur), args.gate_games), 4)
                    extra["wr_anchor"] = round(_play_match(
                        stu_fn, _raw_fn(teacher.model, device),
                        args.gate_games), 4)
                    extra["mcts_edge"] = round(_play_match(
                        _search_fn(student.model, device, args.edge_sims),
                        stu_fn, args.edge_games), 4)
                    extra["gauntlet_secs"] = round(time.perf_counter() - tg0, 1)
                    last_gate_t = time.time()
                    gate_line = (f"     gate | rand={wr_random*100:.0f}% | "
                                 f"heur={extra['wr_heur']*100:.0f}% | "
                                 f"vs_teacher={extra['wr_anchor']*100:.0f}% | "
                                 f"edge={extra['mcts_edge']*100:.0f}% | "
                                 f"{extra['gauntlet_secs']}s")
                    if extra["mcts_edge"] < 0.5:
                        gate_line += "\n     [!] edge < 0.5 -- search losing " \
                                     "to raw student; investigate if persistent"

                # ---- 4. PROMOTE? --------------------------------------------
                if (time.time() - last_promote_t >= args.promote_min * 60
                        and len(buffer) >= args.min_window
                        and not np.isnan(avg_loss)):
                    tp0 = time.perf_counter()
                    score = _play_match(
                        _search_fn(student.model, device, args.promote_sims),
                        _search_fn(teacher.model, device, args.promote_sims),
                        args.promote_games)
                    last_promote_t = time.time()
                    extra["promote_score"] = round(score, 4)
                    if score >= args.promote_thresh:
                        teacher_gen += 1
                        teacher.model.load_state_dict(student.model.state_dict())
                        teacher_tanh = True
                        _save_teacher(teacher_path, teacher.model,
                                      teacher_tanh, teacher_gen)
                        extra["teacher_gen"] = teacher_gen
                        print(f"[**] PROMOTION: student beat teacher "
                              f"{score*100:.0f}% -> teacher gen {teacher_gen}")
                    else:
                        print(f"     promotion match: {score*100:.0f}% "
                              f"(need {args.promote_thresh*100:.0f}%) | "
                              f"{time.perf_counter()-tp0:.0f}s")

            # ---- metrics + state --------------------------------------------
            if not args.no_metrics:
                append_metrics(
                    loss=avg_loss,
                    epsilon=float("nan"),
                    winrate=wr_random,
                    value_loss=avg_val,
                    t=time.time(),
                    policy_loss=avg_pol,
                    games_total=games_total,
                    buffer=len(buffer),
                    extra=extra,
                )
            block += 1
            save_all()

            elapsed = time.perf_counter() - t0
            loss_str = f"{avg_loss:.4f}" if not np.isnan(avg_loss) else "warmup"
            print(f"block {block:5d} | gen {teacher_gen} | "
                  f"games={games_total:6d} | buf={len(buffer):6d} | "
                  f"loss={loss_str} | ent={pi_ent:.2f} | "
                  f"draws={draws}/{args.games_per_block} | "
                  f"gen {gen_secs:.0f}s / total {elapsed:.0f}s")
            if gate_line:
                print(gate_line)

    except KeyboardInterrupt:
        print("\n[!] Interrupted -- saving state...")
    finally:
        save_all()
        print(f"State saved to {args.model_dir} "
              f"({format_elapsed(time.time() - t_start)} this session). "
              f"Relaunch with --resume to continue.")


if __name__ == "__main__":
    main()
