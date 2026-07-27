"""Where does the honest ruler still have headroom?

The promotion gate can only detect progress on a panel that is not saturated.
winblock is effectively exhausted (the gen-22 teacher scores 0.927 there) and
gregory-d3 is heading the same way (0.813). When the ruler saturates the gate
stops measuring strength and starts selecting noise -- which is exactly how the
564k-game plateau happened (RESULT_GATE_PLATEAU.md).

So the ladder has to stay ahead of the lineage. This measures a set of nets
against gregory at increasing depth and reports both the score and what the
panel costs, because a ruler nobody can afford to run is not a ruler.

    python -m scripts.ruler_ladder --depths 3,4,5 --games 300
"""
from __future__ import annotations

import argparse
import time

import torch

from agents.agent_base import ModelConfigCNN
from agents.gregory import GregoryAgent
from agents.neural_net_agent_3 import ConvNet
from scripts.expert_iter import _agent_fn, _play_fixed_match, _raw_fn
from scripts.train_alphazero import NETWORK_CONFIGS

# Seeds follow the promotion gate's convention: d3=8801, d4=8802, and one per
# deeper rung so a given depth always draws the same openings.
DEPTH_SEEDS = {1: 8799, 2: 8800, 3: 8801, 4: 8802, 5: 8803, 6: 8804, 7: 8805}

NETS = [
    dict(label="gen-22 teacher (oracle)",
         path="models/expert_iter_v2/teacher.pt",
         arch=NETWORK_CONFIGS["arena22"], value_tanh=True),
    dict(label="pocket:squeeze-gen22",
         path="models/ab_arch/squeeze.pt",
         arch=dict(conv_channels=[56, 56, 56, 56], fc_hidden_sizes=[256],
                   head_squeeze=2),
         value_tanh=True),
]


def load(spec, device):
    cfg = ModelConfigCNN(value_tanh=spec["value_tanh"],
                         model_dir="models/_ladder", **spec["arch"])
    model = ConvNet(cfg).to(device)
    raw = torch.load(spec["path"], map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(sd)
    model.eval()
    return model, sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", type=str, default="3,4,5")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    depths = [int(d) for d in args.depths.split(",") if d.strip()]

    print(f"raw argmax, {args.games} games/cell, fixed openings, "
          f"colours swapped\n")
    hdr = f"{'net':<26}{'params':>10}" + "".join(f"{'d'+str(d):>9}" for d in depths)
    print(hdr)
    print("-" * len(hdr))

    costs = {}
    with torch.no_grad():
        for spec in NETS:
            model, n = load(spec, args.device)
            fn = _raw_fn(model, args.device, sample_moves=0)
            row = []
            for d in depths:
                t0 = time.perf_counter()
                s = _play_fixed_match(fn, _agent_fn(GregoryAgent(depth=d)),
                                      args.games, seed=DEPTH_SEEDS[d])
                costs.setdefault(d, []).append(time.perf_counter() - t0)
                row.append(s)
            print(f"{spec['label']:<26}{n:>10,}"
                  + "".join(f"{s:>9.3f}" for s in row), flush=True)

    print(f"\n{'depth':>6} {'panel secs':>12} {'headroom (1 - best)':>21}")
    for d in depths:
        print(f"{d:>6} {max(costs[d]):>12.0f}")
    print("\nA rung is usable in the gate while it is both AFFORDABLE (a panel "
          "must\nbe a small fraction of --promote_min) and UNSATURATED (scores "
          "well under\n1.0, with room for the lineage to climb).")


if __name__ == "__main__":
    main()
