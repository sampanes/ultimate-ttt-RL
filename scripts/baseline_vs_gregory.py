"""S1 pre-step (STRENGTH_NEXT.md): baseline the raw net vs shallow gregory.

Decides whether GregoryAgent(depth=2) joins the expert_iter training mix or
whether the d3-in-mix / d4-becomes-ruler variant is needed instead:

    score <  0.55 -> ENABLE the d2 slice (--greg_mix 0.10, donated equally
                     from --opp_mix and --rnd_mix; see PENDING.md runbook)
    score >= 0.55 -> the d2 slice teaches little; STOP and ping the authoring
                     box for the d3-in-mix + d4-ruler variant (STRENGTH_NEXT
                     S1 pre-step -- do not improvise it, the ruler swap
                     changes panel cost ~10x)

CPU-only and read-only: never touches the GPU, the run state, or the metrics
log, so it is safe to run WHILE the goat run is live. (Worst case a promotion
overwrites teacher.pt at the exact moment of the read -- just rerun.) d2 is
~1 ms/move, so the default 300-game panel takes well under a minute.

Run (cmd, from repo root):
    .venv\\Scripts\\python -m scripts.baseline_vs_gregory

Search-on variant -- measures the DEPLOYED player (net + MCTS), not the raw
head the promotion gate scores. Add --sims N to play net + N-sim MCTS vs
gregory. Still CPU-only / GPU-safe, just slower (50 sims x 300 games ~ tens of
minutes). Example (the page's Hard mode is 50 sims):
    .venv\\Scripts\\python -m scripts.baseline_vs_gregory --which student --sims 50 --depths 3

Seeds match the promotion panel's gregory seed (8801), so a --depths 3 run
here is directly comparable to the promote_gregory series in the metrics log.
"""

import argparse
import json
import os
import time

import torch

from agents.gregory import GregoryAgent
from scripts.expert_iter import (_agent_fn, _make_agent, _play_fixed_match,
                                 _raw_fn, _search_fn)


def main():
    ap = argparse.ArgumentParser(
        description="Raw-net baseline vs shallow gregory (STRENGTH_NEXT S1 "
                    "pre-step).")
    ap.add_argument("--model_dir", type=str,
                    default=os.path.join("models", "expert_iter_v2"))
    ap.add_argument("--which", choices=["teacher", "student"],
                    default="teacher",
                    help="teacher = last promoted net (the lineage's raw net "
                         "of record); student = current training weights.")
    ap.add_argument("--depths", type=str, default="2",
                    help="Comma-separated gregory depths to measure.")
    ap.add_argument("--games", type=int, default=300,
                    help="Fixed color-swapped openings per depth (300 -> "
                         "SE ~0.03, same as the promotion panel).")
    ap.add_argument("--network", type=str, default=None,
                    help="Architecture override; default reads it from "
                         "model_dir/state.json.")
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu (default) so a live goat run is undisturbed.")
    ap.add_argument("--sims", type=int, default=0,
                    help="MCTS sims per move over the net. 0 (default) = raw "
                         "policy argmax, the promotion-gate number. >0 runs the "
                         "deployed search player (50 = the page's Hard mode); "
                         "still CPU-only, just slower.")
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(2)  # good citizen: a live goat run owns this box

    network = args.network
    if network is None:
        with open(os.path.join(args.model_dir, "state.json")) as f:
            network = json.load(f)["network"]

    ckpt = os.path.join(args.model_dir, f"{args.which}.pt")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    # teacher.pt carries value_tanh/gen metadata; student.pt (agent.save
    # format) does not, but students always have a tanh head. The value head
    # is irrelevant to raw argmax policy play anyway.
    value_tanh = bool(payload.get("value_tanh", True))
    gen = payload.get("gen")
    agent = _make_agent(network, args.device, value_tanh, args.model_dir,
                        lr=1e-3)
    agent.model.load_state_dict(payload["state_dict"])
    agent.model.eval()
    if args.sims > 0:
        move_fn = _search_fn(agent.model, args.device, args.sims, sample_moves=0)
        who = f"MCTS({args.sims}) {args.which}"
    else:
        move_fn = _raw_fn(agent.model, args.device, sample_moves=0)
        who = f"Raw {args.which}"

    gen_str = f" gen {gen}" if gen is not None else ""
    print(f"{who}{gen_str} [{network}] vs gregory | "
          f"{args.games} fixed color-swapped openings per depth | "
          f"device={args.device}")

    d2_score = None
    with torch.no_grad():
        for d in [int(x) for x in args.depths.split(",")]:
            t0 = time.perf_counter()
            score = _play_fixed_match(
                move_fn, _agent_fn(GregoryAgent(depth=d)), args.games, seed=8801)
            print(f"  vs gregory(d{d}): {score:.3f}   "
                  f"({time.perf_counter() - t0:.0f}s)")
            if d == 2:
                d2_score = score

    if d2_score is not None and args.sims == 0:
        if d2_score < 0.55:
            print("\nVERDICT: d2 slice has headroom -> enable it "
                  "(--greg_mix 0.10 --opp_mix 0.30 --rnd_mix 0.10; "
                  "see the PENDING.md S1 runbook).")
        else:
            print("\nVERDICT: raw net already scores >= 0.55 vs d2 -- the "
                  "slice teaches little. STOP: ping the authoring box for "
                  "the d3-in-mix + d4-ruler variant (STRENGTH_NEXT S1).")


if __name__ == "__main__":
    main()
