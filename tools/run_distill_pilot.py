"""Train the paired distillation-pilot students: 3 seeds x 2 teacher arms.

The pairing is the whole design. Within a seed, the two arms get:

  the same initial weights   (--init_seed, verified by printed fingerprint)
  the same batch order       (--seed drives the index permutation)
  the same symmetry draws    (--seed + 1 drives the dihedral RNG)
  the same optimizer schedule and step count
  the same 50,000 positions with byte-identical x and z

Only the policy target differs. Across seeds the init and order change, so the
spread over seeds measures training variance directly rather than assuming it
is negligible.

This driver refuses to proceed if a seed's two arms report different init
fingerprints, because that silently destroys the pairing and the result would
look like an arm effect.

    python -m tools.run_distill_pilot
    python -m tools.run_distill_pilot --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

FINGERPRINT = re.compile(r"init_fingerprint=(-?\d+\.\d+)")


def train_one(corpus, out, init_seed, seed, steps, batch_size, lr, lr_min,
              half_life, value_coef, snapshot_every, dry_run):
    cmd = [
        sys.executable, "-u", "-m", "scripts.train_student_offline",
        "--corpus", corpus,
        "--arch", "squeeze",
        "--out", out,
        "--init_seed", str(init_seed),
        "--seed", str(seed),
        "--steps", str(steps),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--lr_min", str(lr_min),
        "--lr_half_life_steps", str(half_life),
        "--value_coef", str(value_coef),
        "--snapshot_every", str(snapshot_every),
    ]
    if dry_run:
        print("  DRY " + " ".join(cmd))
        return None, 0.0
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"[X] training failed for {out}")
    m = FINGERPRINT.search(proc.stdout)
    tail = [ln for ln in proc.stdout.splitlines() if "loss" in ln.lower()]
    if tail:
        print(f"    {tail[-1].strip()}")
    return (float(m.group(1)) if m else None), dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", default="models/distill_pilot")
    ap.add_argument("--arms", type=int, nargs="+", default=[50, 800])
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    # Defaults below are the values frozen in EXPERIMENT_DISTILL_PILOT.json.
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr_min", type=float, default=1e-4)
    ap.add_argument("--lr_half_life_steps", type=int, default=15000)
    ap.add_argument("--value_coef", type=float, default=1.0)
    ap.add_argument("--snapshot_every", type=int, default=5000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.pilot, "students")
    os.makedirs(out_dir, exist_ok=True)

    runs, fingerprints = {}, {}
    for seed in args.seeds:
        print(f"seed {seed}")
        for arm in args.arms:
            corpus = os.path.join(args.pilot, f"sims{arm}")
            out = os.path.join(out_dir, f"s{seed}_sims{arm}.pt")
            print(f"  arm sims={arm} -> {out}")
            fp, dt = train_one(
                corpus, out, seed, seed, args.steps, args.batch_size, args.lr,
                args.lr_min, args.lr_half_life_steps, args.value_coef,
                args.snapshot_every, args.dry_run)
            runs[f"s{seed}_sims{arm}"] = {"out": out, "seconds": dt,
                                          "init_fingerprint": fp}
            fingerprints.setdefault(seed, {})[arm] = fp
            if not args.dry_run:
                print(f"    done in {dt / 60:.1f} min  fingerprint={fp}")

        if not args.dry_run:
            got = set(fingerprints[seed].values())
            if len(got) != 1 or None in got:
                raise SystemExit(
                    f"[X] seed {seed}: arms started from DIFFERENT weights "
                    f"{fingerprints[seed]}. The pairing is broken -- an arm "
                    f"effect measured now would be confounded with init.")
            print(f"  [OK] seed {seed} arms share init {got.pop()}")

    if args.dry_run:
        return
    payload = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "arms": args.arms, "seeds": args.seeds,
               "steps": args.steps, "batch_size": args.batch_size,
               "lr": args.lr, "lr_min": args.lr_min,
               "lr_half_life_steps": args.lr_half_life_steps,
               "value_coef": args.value_coef, "runs": runs}
    path = os.path.join(args.pilot, "students", "training_log.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
