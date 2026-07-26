"""Pocket-champion challenge: 300-game panel + the h2h tie-breaker.

CHAMPIONS.md's promotion rule is "beats the incumbent on the M2 panel", amended
2026-07-11 so that when panel modes disagree a direct 300-game fixed-opening
colour-swapped raw head-to-head decides. The M2 suite runs only 9 openings x 2
games = 18 per cell, which is a coarse screen; this runs the deciding numbers at
300 games and adds the gene-pool-independent alpha-beta rulers that the M2
anchor list does not carry.

Both nets are played RAW (argmax, no search, no sampling), which is how the
pocket model is graded and how it ships.

    python -m scripts.pocket_challenge
"""
from __future__ import annotations

import argparse

import torch

from agents.agent_base import ModelConfigCNN
from agents.deterministics import WinBlockAgent
from agents.gregory import GregoryAgent
from agents.neural_net_agent_3 import ConvNet
from agents.random_agent import RandomAgent
from scripts.expert_iter import _agent_fn, _play_fixed_match, _raw_fn

# The incumbent, identified by the SHA-256 recorded in CHAMPIONS.md.
INCUMBENT = dict(
    label="arena:21@hof (incumbent)",
    path="models/arena/hall_of_fame/06-26-26_elo1819.pt",
    arch=dict(conv_channels=[32, 128, 32, 32], fc_hidden_sizes=[128, 256, 512]),
    value_tanh=False,
)
CHALLENGER = dict(
    label="squeeze/gen22 (challenger)",
    path="models/ab_arch/squeeze.pt",
    arch=dict(conv_channels=[56, 56, 56, 56], fc_hidden_sizes=[256],
              head_squeeze=2),
    value_tanh=True,
)

# Same per-opponent seeds the promotion gate uses, so these numbers are
# comparable with everything else in the repo.
SEEDS = {"random": 4401, "winblock": 3301, "gregory_d3": 8801,
         "gregory_d4": 8802, "h2h": 9901}


def load(spec, device):
    cfg = ModelConfigCNN(value_tanh=spec["value_tanh"],
                         model_dir="models/_pocket", **spec["arch"])
    model = ConvNet(cfg).to(device)
    raw = torch.load(spec["path"], map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(sd)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    return model, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    opponents = {
        "random": _agent_fn(RandomAgent()),
        "winblock": _agent_fn(WinBlockAgent()),
        "gregory_d3": _agent_fn(GregoryAgent(depth=3)),
        "gregory_d4": _agent_fn(GregoryAgent(depth=4)),
    }

    nets = {}
    print(f"panel: {args.games} games/cell, raw argmax, fixed openings, "
          f"colours swapped\n")
    head = f"{'net':<28}{'params':>10}" + "".join(
        f"{k:>12}" for k in opponents)
    print(head)
    print("-" * len(head))

    rows = {}
    with torch.no_grad():
        for spec in (INCUMBENT, CHALLENGER):
            model, nparams = load(spec, args.device)
            nets[spec["label"]] = model
            fn = _raw_fn(model, args.device, sample_moves=0)
            row = {k: _play_fixed_match(fn, opp, args.games, seed=SEEDS[k])
                   for k, opp in opponents.items()}
            rows[spec["label"]] = row
            print(f"{spec['label']:<28}{nparams:>10,}" + "".join(
                f"{row[k]:>12.3f}" for k in opponents), flush=True)

        inc = _raw_fn(nets[INCUMBENT["label"]], args.device, sample_moves=0)
        chal = _raw_fn(nets[CHALLENGER["label"]], args.device, sample_moves=0)
        h2h = _play_fixed_match(chal, inc, args.games, seed=SEEDS["h2h"])

    print(f"\ntie-breaker: challenger vs incumbent, {args.games} games, "
          f"seed {SEEDS['h2h']}")
    print(f"  challenger scores {h2h:.4f}  "
          f"({'CHALLENGER WINS' if h2h > 0.5 else 'incumbent holds'})")

    won = sum(1 for k in opponents
              if rows[CHALLENGER["label"]][k] > rows[INCUMBENT["label"]][k])
    print(f"\n  panel cells won by challenger: {won}/{len(opponents)}")
    print(f"  h2h: {'PASS' if h2h > 0.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
