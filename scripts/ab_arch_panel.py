"""Score two or more offline-trained students on the fixed external panel.

Every match uses the repo's standard `_play_fixed_match` harness: diverse
reproducible openings, colors swapped, per-game RNG reset, and a fixed seed per
opponent so all arms face byte-identical games. Anchors are the frozen
non-gene-pool opponents (GRADING_AND_ORACLE.md) -- never the lottery, which is
a constant function and voids any number measured against it.

Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before running with --sims on CUDA.

Usage:
    python -m scripts.ab_arch_panel --models models/ab_arch/plain.pt \
        models/ab_arch/modern.pt --games 300 --sims 50
"""
import argparse
import json
import os
import time

import torch

from agents.agent_base import ModelConfigCNN
from agents.neural_net_agent_3 import ConvNet
from agents.deterministics import WinBlockAgent
from agents.gregory import GregoryAgent
from agents.random_agent import RandomAgent
from scripts.expert_iter import (_play_fixed_match, _raw_fn, _search_fn,
                                 _agent_fn)

torch.set_float32_matmul_precision("high")

# Fixed per-opponent seeds -- identical to the promotion gate's, so numbers
# here are directly comparable to the ones in state.json / RESULT_* docs.
SEEDS = {"random": 4401, "winblock": 3301, "gregory_d3": 8801,
         "gregory_d4": 8802, "teacher": 5501, "h2h": 7701}


def load_student(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfigCNN(value_tanh=ck.get("value_tanh", True),
                         model_dir="models/_offline", **ck["arch"])
    model = ConvNet(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    name = ck.get("arch_name") or os.path.splitext(os.path.basename(path))[0]
    return name, model, ck


def load_teacher(path, device):
    from scripts.train_league import NETWORK_CONFIGS
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfigCNN(value_tanh=ck.get("value_tanh", True),
                         model_dir="models/_offline", **NETWORK_CONFIGS["arena22"])
    model = ConvNet(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="External panel for an arch A/B.")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--teacher", type=str,
                    default="models/expert_iter_v2/teacher.pt")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--sims", type=int, default=0,
                    help="If >0, also run each arm as net+MCTS at this many "
                         "sims vs gregory(d3).")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    device = args.device
    arms = [load_student(p, device) for p in args.models]
    teacher = load_teacher(args.teacher, device) if args.teacher else None

    opponents = {
        "random": _agent_fn(RandomAgent()),
        "winblock": _agent_fn(WinBlockAgent()),
        "gregory_d3": _agent_fn(GregoryAgent(depth=3)),
        "gregory_d4": _agent_fn(GregoryAgent(depth=4)),
    }
    if teacher is not None:
        opponents["teacher"] = _raw_fn(teacher, device, sample_moves=0)

    results = {}
    print(f"\npanel: {args.games} games/cell, fixed openings, colors swapped\n")
    header = f"{'arm':<10}{'params':>10}" + "".join(
        f"{k:>13}" for k in opponents)
    print(header)
    print("-" * len(header))

    with torch.no_grad():
        for name, model, ck in arms:
            row = {}
            fn = _raw_fn(model, device, sample_moves=0)
            for opp_name, opp_fn in opponents.items():
                t0 = time.time()
                row[opp_name] = _play_fixed_match(
                    fn, opp_fn, args.games, seed=SEEDS[opp_name])
                row[f"_{opp_name}_secs"] = round(time.time() - t0, 1)
            results[name] = row
            print(f"{name:<10}{ck['params']:>10,}" + "".join(
                f"{row[k]:>13.3f}" for k in opponents), flush=True)

        # Direct head-to-head: the incumbent (arm 0) against every challenger.
        # This is the M2_5 tie-breaker rule -- when panel cells disagree, a
        # fixed-opening color-swapped h2h decides.
        if len(arms) >= 2:
            n0, m0, _ = arms[0]
            f0 = _raw_fn(m0, device, sample_moves=0)
            print(f"\nraw head-to-head vs {n0} (>0.5 means the challenger "
                  f"loses; {args.games} games, seed {SEEDS['h2h']}):")
            results["h2h"] = {}
            for name, model, _ in arms[1:]:
                s = _play_fixed_match(
                    f0, _raw_fn(model, device, sample_moves=0),
                    args.games, seed=SEEDS["h2h"])
                results["h2h"][name] = s
                verdict = "challenger WINS" if s < 0.5 else "incumbent holds"
                print(f"  {n0} vs {name:<12}{s:>8.3f}   {verdict}", flush=True)

        # Deployed player: net + search, the thing that actually ships.
        if args.sims > 0:
            print(f"\nnet + MCTS-{args.sims} vs gregory(d3):")
            greg = opponents["gregory_d3"]
            for name, model, _ in arms:
                t0 = time.time()
                s = _play_fixed_match(
                    _search_fn(model, device, args.sims, sample_moves=0),
                    greg, args.games, seed=SEEDS["gregory_d3"])
                secs = time.time() - t0
                results[name][f"mcts{args.sims}_gregory_d3"] = s
                results[name][f"mcts{args.sims}_secs"] = round(secs, 1)
                print(f"  {name:<10}{s:>8.3f}   ({secs:.0f}s "
                      f"= {secs / args.games * 1000:.0f} ms/game)", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
