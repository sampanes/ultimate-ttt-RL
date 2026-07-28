"""Remove the target-temperature confound from the distillation pilot.

The pilot found student_50 beating student_800, and the diagnosis is that the
two arms differed in TWO variables, not one:

  teacher quality    800 sims is +0.019/doubling stronger (RESULT_TEACHER_SIM_LADDER)
  target temperature 800 sims produces a SOFTER visit distribution

The second is a mechanical consequence of PUCT, not a property of the position.
The exploration bonus scales with sqrt(N_total), so more simulations spray more
absolute visits onto moves already known to be inferior. Measured on the pilot's
mate-in-1 positions: the 800-sim target puts a visit on EVERY legal move (11.53
of 11.53) and keeps only 0.693 mass on the win, where the 50-sim target touches
3.82 of 11.53 and keeps 0.825. Both pick the same move 93% of the time -- the
deeper teacher is not wrong, it is diluted.

This rewrites an arm's targets at pi ** (1/T), renormalised, choosing T so the
arm's mean top-move mass matches a reference arm. With both arms then at the
same temperature, a remaining difference is teacher QUALITY alone.

Sharpening cannot recreate the 50-sim sparsity pattern exactly -- a 1/800 visit
raised to a power is small but nonzero, where a 0-visit move is structurally
absent. Matching mean top-move mass is the closest honest control, and the
residual tail is reported so it is not mistaken for an exact match.

    python -m tools.sharpen_distill_corpus --arm 800 --match-arm 50
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from scripts.train_student_offline import load_corpus


def sharpen(pi, t):
    """pi ** (1/t), renormalised. t < 1 sharpens, t > 1 softens."""
    out = pi.clamp_min(0.0) ** (1.0 / t)
    return out / out.sum(1, keepdim=True).clamp_min(1e-12)


def solve_temperature(pi, target_mass, lo=0.05, hi=1.0, iters=60):
    """Bisect for the T whose sharpened mean top-move mass hits target_mass.

    Mean top-move mass is monotone decreasing in T, so bisection is safe.
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = float(sharpen(pi, mid).max(1).values.mean())
        if got > target_mass:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_temperature_per_position(pi, target_mass, lo=0.02, hi=8.0, iters=50):
    """Per-position T_i matching each row's top-move mass to target_mass[i].

    A GLOBAL temperature is the wrong control here. The 800-sim target is
    sharper than the 50-sim one on average (0.4906 vs 0.4659) yet dramatically
    softer on forced wins (0.693 vs 0.825): more simulations sharpen ambiguous
    positions, where the extra search genuinely discriminates, and soften
    decisive ones, where PUCT's sqrt(N) exploration keeps sampling moves the
    search already knows are lost. One scalar cannot undo an effect that points
    in opposite directions by position type.

    Matching row by row removes the temperature difference exactly, leaving
    only WHICH move each teacher ranks first and how it orders the rest -- the
    teacher-quality signal the pilot is supposed to measure.

    Monotone in T, so vectorised bisection is safe. Rows already at the target
    converge to T=1 on their own; no special-casing needed.
    """
    n = pi.shape[0]
    lo_t = torch.full((n, 1), lo, dtype=torch.float64)
    hi_t = torch.full((n, 1), hi, dtype=torch.float64)
    p64 = pi.double().clamp_min(0.0)
    tgt = target_mass.double().reshape(-1, 1)
    for _ in range(iters):
        mid = 0.5 * (lo_t + hi_t)
        s = p64 ** (1.0 / mid)
        got = (s / s.sum(1, keepdim=True).clamp_min(1e-300)).max(1,
                                                                 keepdim=True).values
        too_sharp = got > tgt
        lo_t = torch.where(too_sharp, mid, lo_t)
        hi_t = torch.where(too_sharp, hi_t, mid)
    return (0.5 * (lo_t + hi_t))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", default="models/distill_pilot")
    ap.add_argument("--arm", type=int, default=800, help="arm to rewrite")
    ap.add_argument("--match-arm", type=int, default=50,
                    help="arm whose mean top-move mass is the target")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="explicit T; 0 solves for it")
    ap.add_argument("--per-position", action="store_true",
                    help="match each row's top-move mass individually. Correct "
                         "here: the sharpness gap points in OPPOSITE directions "
                         "for tactical vs ambiguous positions, so no single "
                         "scalar T can remove it.")
    ap.add_argument("--suffix", default="sharp")
    ap.add_argument("--shard-size", type=int, default=10000)
    args = ap.parse_args()

    src = os.path.join(args.pilot, f"sims{args.arm}")
    ref = os.path.join(args.pilot, f"sims{args.match_arm}")
    X, PI, Z = load_corpus(src, 0, verbose=False)
    _Xr, PIr, _Zr = load_corpus(ref, 0, verbose=False)

    target = float(PIr.max(1).values.mean())
    if args.per_position:
        tvec = solve_temperature_per_position(PI, PIr.max(1).values)
        s = PI.double().clamp_min(0.0) ** (1.0 / tvec)
        PS = (s / s.sum(1, keepdim=True).clamp_min(1e-300)).float()
        t = float(tvec.median())
        print(f"per-position T: median {t:.4f}  "
              f"p10 {float(tvec.quantile(0.10)):.4f}  "
              f"p90 {float(tvec.quantile(0.90)):.4f}")
    else:
        t = args.temperature or solve_temperature(PI, target)
        PS = sharpen(PI, t)

    idx = np.load(os.path.join(args.pilot, "index.npz"), allow_pickle=True)
    m1 = torch.from_numpy(idx["immediate_win"].astype(bool))

    print(f"arm sims={args.arm} -> temperature T={t:.4f} "
          f"(matching sims{args.match_arm} top-move mass {target:.4f})")
    print(f"  top-move mass   before {PI.max(1).values.mean():.4f}  "
          f"after {PS.max(1).values.mean():.4f}  ref {target:.4f}")
    print(f"  mate-in-1 mass  before {PI[m1].max(1).values.mean():.4f}  "
          f"after {PS[m1].max(1).values.mean():.4f}  "
          f"ref {PIr[m1].max(1).values.mean():.4f}")
    eff_nz = (PS > 1e-4).sum(1).float().mean()
    print(f"  moves above 1e-4: before {(PI > 1e-4).sum(1).float().mean():.2f}  "
          f"after {eff_nz:.2f}  ref {(PIr > 1e-4).sum(1).float().mean():.2f}")
    print(f"  argmax preserved by sharpening: "
          f"{(PS.argmax(1) == PI.argmax(1)).float().mean():.4f} (must be 1.0000)")

    out = os.path.join(args.pilot, f"sims{args.arm}_{args.suffix}")
    data_dir = os.path.join(out, "data")
    os.makedirs(data_dir, exist_ok=True)
    for old in glob.glob(os.path.join(data_dir, "shard_*.pt")):
        os.remove(old)
    n = X.shape[0]
    for s, start in enumerate(range(0, n, args.shard_size)):
        stop = min(start + args.shard_size, n)
        torch.save({"x": X[start:stop], "pi": PS[start:stop],
                    "z": Z[start:stop], "teacher_gen": 22},
                   os.path.join(data_dir, f"shard_{s:05d}.pt"))
    with open(os.path.join(out, "sharpen.json"), "w", encoding="utf-8") as fh:
        json.dump({"source_arm": args.arm, "match_arm": args.match_arm,
                   "per_position": bool(args.per_position),
                   "temperature": t, "target_top_mass": target,
                   "achieved_top_mass": float(PS.max(1).values.mean())}, fh,
                  indent=2)
    print(f"wrote {out} -- {n:,} examples")


if __name__ == "__main__":
    main()
