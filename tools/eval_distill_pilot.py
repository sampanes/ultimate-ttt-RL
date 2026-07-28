"""Evaluate the distillation pilot: does an 800-sim teacher beat a 50-sim one?

Primary result is games, not loss. Loss is not even comparable across the arms:
the 800-sim policy target is measurably softer (mean entropy 2.029 vs 1.977
bits, because 50 sims quantises any move below ~1/50 to exactly zero), so it
carries a higher cross-entropy floor by construction.

Three evaluations, in increasing cost:

  --primary    student_800 vs student_50, head to head, PAIRED WITHIN SEED.
               Each seed's two arms started from identical weights and saw the
               same batch order, so their difference isolates the target. Runs
               all three seed pairs and pools.

  --policy     Agreement with the 800-sim policy over the training positions,
               overall and on the 50-vs-800 disagreement subset, broken out by
               phase, branching factor, tactical status and teacher value gap.
               No games; reads index.npz.

  --ladder     Every student against the frozen external anchors. Expensive
               (gregory d4 dominates); sized by --ladder-games.

    python -m tools.eval_distill_pilot --primary --policy
    python -m tools.eval_distill_pilot --ladder --ladder-games 300
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch

from agents.agent_base import ModelConfigCNN
from agents.deterministics import WinBlockAgent
from agents.gregory import GregoryAgent
from agents.neural_net_agent_3 import ConvNet
from agents.random_agent import RandomAgent
from scripts.expert_iter import _agent_fn, _raw_fn
from scripts.train_student_offline import AB_ARCHS, load_corpus
from tools.teacher_sim_ladder import outcome_ci, play_match_detailed

# Frozen in EXPERIMENT_DISTILL_PILOT.json before any target was generated.
ANCHOR_SEEDS = {"random": 4401, "winblock": 3301,
                "gregory_d3": 8801, "gregory_d4": 8802}
H2H_BASE_SEED = 9901


def load_student(path, device, arch="squeeze"):
    cfg = ModelConfigCNN(value_tanh=True, model_dir="models/_pilot",
                         **AB_ARCHS[arch])
    model = ConvNet(cfg).to(device)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(sd)
    model.eval()
    return model


def fmt_ci(score, lo, hi):
    return f"{score:.4f} [{lo:.4f}, {hi:.4f}]"


def run_primary(args, device, results):
    a, b = args.arm_a, args.arm_b
    print(f"\nPRIMARY: student_{a} vs student_{b}, {args.games} games per seed, "
          f"paired within seed\n")
    pooled = []
    per_seed = {}
    for seed in args.seeds:
        p800 = os.path.join(args.pilot, args.students_dir, f"s{seed}_sims{a}.pt")
        p50 = os.path.join(args.pilot, args.students_dir, f"s{seed}_sims{b}.pt")
        for p in (p800, p50):
            if not os.path.exists(p):
                raise SystemExit(f"[X] missing student {p}")
        m800, m50 = load_student(p800, device, args.arch), load_student(p50, device, args.arch)
        match_seed = H2H_BASE_SEED + seed
        with torch.no_grad():
            f800 = _raw_fn(m800, device, sample_moves=0)
            f50 = _raw_fn(m50, device, sample_moves=0)
            t0 = time.time()
            outcomes = play_match_detailed(f800, f50, args.games, match_seed)
        score, (lo, hi), se = outcome_ci(outcomes)
        w = sum(1 for o in outcomes if o == 1.0)
        d = sum(1 for o in outcomes if o == 0.5)
        ll = sum(1 for o in outcomes if o == 0.0)
        # Per-game outcomes and the match seed are persisted so a later run can
        # be paired GAME BY GAME against this one. `_eval_openings(n, seed)`
        # depends on the seed alone and emits a prefix-stable stream, the
        # per-game reseed is seed + opening_idx*2 + side, and students move with
        # sample_moves=0, so two runs sharing match_seed play game i from the
        # identical opening, colour and RNG. Without this vector only seed-level
        # pairing is defensible -- see tools/pooled_estimator.pairing_status.
        per_seed[seed] = {"score": score, "ci95": [lo, hi], "se": se,
                          "wins": w, "draws": d, "losses": ll,
                          "match_seed": match_seed, "outcomes": outcomes,
                          "seconds": time.time() - t0}
        pooled.extend(outcomes)
        print(f"  seed {seed:>3}  {fmt_ci(score, lo, hi)}  W{w}/D{d}/L{ll}")

    score, (lo, hi), se = outcome_ci(pooled)
    z = (score - 0.5) / se if se > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    spread = max(v["score"] for v in per_seed.values()) - \
        min(v["score"] for v in per_seed.values())
    print(f"\n  POOLED  {fmt_ci(score, lo, hi)}  n={len(pooled)}  "
          f"z={z:.2f}  p={p:.4f}")
    print(f"  across-seed spread {spread:.4f} "
          f"(training variance -- if this dwarfs the effect, the effect is "
          f"not resolved)")
    verdict = (f"student_{a} > student_{b}, SEPARATED" if lo > 0.5 else
               f"student_{b} > student_{a}, SEPARATED" if hi < 0.5 else
               "UNRESOLVED -- interval covers 0.5")
    print(f"  verdict: {verdict}")
    results[f"primary_{a}_vs_{b}"] = {
        "games_per_seed": args.games, "per_seed": per_seed,
        "pooled_score": score, "pooled_ci95": [lo, hi], "pooled_n": len(pooled),
        "z": z, "p": p, "across_seed_spread": spread, "verdict": verdict,
    }


def run_policy(args, device, results):
    print("\nPOLICY: agreement with the 800-sim target over the training set\n")
    X, P800, _Z = load_corpus(os.path.join(args.pilot, "sims800"), 0, verbose=False)
    _X50, P50, _Z50 = load_corpus(os.path.join(args.pilot, "sims50"), 0, verbose=False)
    idx = np.load(os.path.join(args.pilot, "index.npz"), allow_pickle=True)

    tgt800 = P800.argmax(1)
    tgt50 = P50.argmax(1)
    disagree = (tgt800 != tgt50).numpy()
    vgap = np.abs(idx["q_root_800"] - idx["q_root_50"])
    print(f"  positions {len(tgt800):,}  disagreement subset "
          f"{disagree.sum():,} ({disagree.mean():.3f})")

    preds = {}
    for seed, arm in itertools.product(args.seeds, args.arms):
        path = os.path.join(args.pilot, args.students_dir, f"s{seed}_sims{arm}.pt")
        model = load_student(path, device, args.arch)
        out = []
        with torch.no_grad():
            for start in range(0, X.shape[0], 4096):
                xb = X[start:start + 4096].to(device)
                logits, _v = model.forward_both(xb)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                # Mask to legal moves -- channel 3 of the planes IS the mask.
                mask = xb[:, 3].reshape(xb.shape[0], 81) > 0.5
                logits = logits.masked_fill(~mask, float("-inf"))
                out.append(logits.argmax(1).cpu())
        preds[(seed, arm)] = torch.cat(out).numpy()

    strata = {
        "all": np.ones(len(tgt800), dtype=bool),
        "disagreement": disagree,
        "agreement": ~disagree,
    }
    for name in ("early", "mid", "late"):
        strata[f"phase:{name}"] = (idx["phase"] == name)
    for name in ("1", "2-4", "5-8", "9+"):
        strata[f"legal:{name}"] = (idx["legal_bucket"] == name)
    strata["tactical:mate_in_1"] = idx["immediate_win"].astype(bool)
    strata["tactical:none"] = ~idx["immediate_win"].astype(bool)
    hi_gap = vgap >= np.quantile(vgap, 0.9)
    strata["value_gap:top10pct"] = hi_gap
    strata["value_gap:rest"] = ~hi_gap

    t800 = tgt800.numpy()
    rows = {}
    print(f"  {'stratum':<24}{'n':>8}{'arm50 ->800tgt':>16}"
          f"{'arm800->800tgt':>16}{'delta':>10}")
    for sname, mask in strata.items():
        n = int(mask.sum())
        if n == 0:
            continue
        a50 = float(np.mean([(preds[(s, args.arm_b)][mask] == t800[mask]).mean()
                             for s in args.seeds]))
        a800 = float(np.mean([(preds[(s, args.arm_a)][mask] == t800[mask]).mean()
                              for s in args.seeds]))
        rows[sname] = {"n": n, f"arm{args.arm_b}_agree_800tgt": a50,
                       f"arm{args.arm_a}_agree_800tgt": a800, "delta": a800 - a50}
        print(f"  {sname:<24}{n:>8,}{a50:>16.4f}{a800:>16.4f}"
              f"{a800 - a50:>+10.4f}")
    print("\n  (arm800 fitting its own target better is EXPECTED and is not "
          "evidence of strength;\n   the informative part is where the gap "
          "concentrates.)")
    results["policy"] = rows


def run_ladder(args, device, results):
    print(f"\nLADDER: every student vs the frozen anchors, "
          f"{args.ladder_games} games/cell\n")
    opponents = {
        "random": _agent_fn(RandomAgent()),
        "winblock": _agent_fn(WinBlockAgent()),
        "gregory_d3": _agent_fn(GregoryAgent(depth=3)),
        "gregory_d4": _agent_fn(GregoryAgent(depth=4)),
    }
    names = [k for k in opponents if k in args.anchors] if args.anchors \
        else list(opponents)
    head = f"  {'student':<20}" + "".join(f"{k:>13}" for k in names)
    print(head)
    print("  " + "-" * (len(head) - 2))
    rows = {}
    for seed, arm in itertools.product(args.seeds, args.arms):
        tag = f"s{seed}_sims{arm}"
        model = load_student(os.path.join(args.pilot, args.students_dir,
                                          f"{tag}.pt"), device, args.arch)
        row = {}
        with torch.no_grad():
            fn = _raw_fn(model, device, sample_moves=0)
            for k in names:
                oc = play_match_detailed(fn, opponents[k], args.ladder_games,
                                         ANCHOR_SEEDS[k])
                sc, (lo, hi), _se = outcome_ci(oc)
                row[k] = {"score": sc, "ci95": [lo, hi]}
        rows[tag] = row
        print(f"  {tag:<20}" + "".join(f"{row[k]['score']:>13.4f}"
                                       for k in names), flush=True)

    # Summarise whatever arms this invocation actually ran -- the ladder is
    # often re-run for one new arm at a time, and a missing arm is not an error.
    print()
    for k in names:
        parts = []
        for arm in args.arms:
            got = [rows[f"s{s}_sims{arm}"][k]["score"] for s in args.seeds
                   if f"s{s}_sims{arm}" in rows]
            if got:
                parts.append(f"{arm}-sim {float(np.mean(got)):.4f}")
        print(f"  {k:<14} arm mean: " + " | ".join(parts))
    results.setdefault("ladder", {}).update(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", default="models/distill_pilot")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--arms", nargs="+", default=["50", "800"])
    ap.add_argument("--arch", default="squeeze", choices=list(AB_ARCHS),
                    help="student architecture; must match how they were trained")
    ap.add_argument("--students-dir", default="students",
                    help="subdirectory of --pilot holding the checkpoints")
    ap.add_argument("--arm-a", default="800", help="primary h2h: side A")
    ap.add_argument("--arm-b", default="50", help="primary h2h: side B")
    ap.add_argument("--games", type=int, default=800,
                    help="head-to-head games PER SEED")
    ap.add_argument("--ladder-games", type=int, default=300)
    ap.add_argument("--anchors", nargs="*", default=[])
    ap.add_argument("--primary", action="store_true")
    ap.add_argument("--policy", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--output", default="results/distill_pilot_eval.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not (args.primary or args.policy or args.ladder):
        raise SystemExit("[X] pick at least one of --primary --policy --ladder")

    results = {}
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as fh:
            results = json.load(fh)

    if args.primary:
        run_primary(args, args.device, results)
    if args.policy:
        run_policy(args, args.device, results)
    if args.ladder:
        run_ladder(args, args.device, results)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
