"""Re-search the frozen pilot positions and dump per-child Q and visit stats.

The distillation corpora store only the visit-count policy and the ROOT value.
Every question in the target-extraction study is about the CHILDREN -- whether
the move deep search picks also has the best child Q, how often visit rank and
Q rank disagree, and how big the top-two Q gap is when they do. None of that is
recoverable from what is on disk, so the search is replayed.

This is not new game generation. The positions are the frozen sample, replayed
bit-for-bit; MCTS.search already returns the root, and each child already
carries N, W, Q(), prior and solved. Nothing in agents/mcts.py is modified.

THE LOAD-BEARING CHECK. Replaying only means something if the replay is the
SAME search that produced the frozen targets. If it is not, the extracted Q
belongs to a different tree than the pi it will be paired with, and every
downstream transform is built on a mismatch. So the extractor recomputes the
visit policy and compares it to the corpus pi, and by default a single
mismatched row is fatal. `--allow-drift` exists only to characterize a failure,
never to proceed past one.

    python -m tools.extract_child_q --pilot models/distill_pilot --sims 800 \
        --limit 200 --out results/child_q/smoke_800.npz
    python -m tools.extract_child_q --pilot models/distill_pilot --sims 50 800 \
        --out results/child_q
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from agents.mcts import MCTS
from tools.make_distill_corpus import (build_model, load_pool_xz,
                                       sample_positions)

# Sampling inputs are the pilot's, not ours to choose: they come from the
# corpus manifest so "the same positions" is read from the record rather than
# retyped here and quietly drifting.
NEEDED = ("corpus", "max_shards", "positions", "seed", "checkpoint", "c_puct")


def load_manifest(pilot):
    with open(os.path.join(pilot, "manifest.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    missing = [k for k in NEEDED if k not in m]
    if missing:
        raise SystemExit(f"[X] {pilot}/manifest.json lacks {missing}")
    return m


def corpus_pi(pilot, sims, suffix=""):
    """The frozen visit-count targets this replay has to reproduce."""
    shards = sorted(glob.glob(os.path.join(
        pilot, f"sims{sims}{suffix}", "data", "shard_*.pt")))
    if not shards:
        raise SystemExit(f"[X] no shards for arm sims{sims}{suffix} under {pilot}")
    return torch.cat([torch.load(p, map_location="cpu",
                                 weights_only=False)["pi"] for p in shards]).numpy()


def extract_arm(pilot, m, sims, solve, limit, device, allow_drift):
    frozen = corpus_pi(pilot, sims)
    n_total = frozen.shape[0]
    n = min(limit, n_total) if limit else n_total

    rng_unused = __import__("random").Random(m["seed"])
    pool_x, pool_z = load_pool_xz(m["corpus"], m["max_shards"], rng_unused)
    sample = sample_positions(pool_x, pool_z, m["positions"], m["seed"])
    if len(sample) != n_total:
        raise SystemExit(
            f"[X] resampled {len(sample)} positions but the arm holds "
            f"{n_total}. The manifest and the shards disagree.")

    model = build_model(m["checkpoint"], device)
    eff = max(1, sims // MCTS._MIN_WAVES)
    mcts = MCTS(model, device, n_sims=sims, c_puct=m["c_puct"],
                add_dirichlet_at_root=False, wave_size=eff, solve=solve)

    # NaN, not 0, for illegal moves: 0 is a legitimate Q (a dead-even position)
    # and conflating the two would silently make every illegal move look drawn.
    child_q = np.full((n, 81), np.nan, dtype=np.float32)
    child_n = np.zeros((n, 81), dtype=np.int32)
    child_prior = np.full((n, 81), np.nan, dtype=np.float32)
    child_solved = np.zeros((n, 81), dtype=np.int8)
    root_value = np.zeros(n, dtype=np.float32)
    pi_replay = np.zeros((n, 81), dtype=np.float32)

    drift, t0 = [], time.time()
    for k in range(n):
        _pool_idx, st, _legal, _filled, _z = sample[k]
        pi, root = mcts.search(st.clone())
        pi_replay[k] = pi
        root_value[k] = root.Q()
        for mv, ch in root.children.items():
            child_q[k, mv] = ch.Q()
            child_n[k, mv] = ch.N
            child_prior[k, mv] = ch.prior
            # solved is None for an unproven child, and -1/+1 for a proven
            # win/loss. None means "no proof", which is 0 here -- distinct from
            # both proof outcomes rather than collapsed into one of them.
            child_solved[k, mv] = 0 if ch.solved is None else ch.solved

        # Compare against the frozen target in the SAME normalization it was
        # stored in; make_distill_corpus writes raw visit counts as pi.
        if not np.array_equal(pi.astype(np.float32), frozen[k]):
            drift.append(k)
            if not allow_drift:
                raise SystemExit(
                    f"[X] row {k}: the replayed search does not reproduce the "
                    f"frozen sims{sims} target. The extracted child Q would "
                    f"belong to a different tree than the pi it is paired "
                    f"with, so every transform built on it would be invalid. "
                    f"Refusing to continue. (--allow-drift to characterize.)")
        if (k + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  sims={sims}  {k + 1:,}/{n:,}  {el:.0f}s  "
                  f"({el / (k + 1) * 1000:.1f} ms/pos, "
                  f"eta {el / (k + 1) * (n - k - 1) / 60:.0f} min)", flush=True)

    dt = time.time() - t0
    print(f"  sims={sims} wave={eff}  {dt:.0f}s  "
          f"({dt / max(n, 1) * 1000:.1f} ms/pos)  drift={len(drift)}/{n}")
    return {"child_q": child_q, "child_n": child_n, "child_prior": child_prior,
            "child_solved": child_solved, "root_value": root_value,
            "pi_replay": pi_replay,
            "drift_rows": np.array(drift, dtype=np.int64),
            "n": n, "seconds": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="models/distill_pilot")
    ap.add_argument("--sims", type=int, nargs="+", default=[50, 800])
    ap.add_argument("--solve", action="store_true",
                    help="replay with solved-node propagation. OFF by default "
                         "so the extraction matches the frozen 0.4108 corpora "
                         "and 'solved vs unsolved' stays out of this study.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True,
                    help="directory, or an .npz path when a single --sims")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-drift", action="store_true")
    args = ap.parse_args()

    m = load_manifest(args.pilot)
    print(f"pilot {args.pilot}: {m['positions']:,} positions, seed {m['seed']}, "
          f"teacher {os.path.basename(m['checkpoint'])}, c_puct {m['c_puct']}")

    single = len(args.sims) == 1 and args.out.endswith(".npz")
    os.makedirs(os.path.dirname(args.out) if single else args.out,
                exist_ok=True)

    for s in args.sims:
        print(f"--- arm sims={s} solve={args.solve} ---")
        res = extract_arm(args.pilot, m, s, args.solve, args.limit,
                          args.device, args.allow_drift)
        n, dt = res.pop("n"), res.pop("seconds")
        path = args.out if single else os.path.join(args.out, f"sims{s}.npz")
        np.savez_compressed(path, **res)
        print(f"[OK] wrote {path}  ({n:,} rows, {dt / 60:.1f} min)")


if __name__ == "__main__":
    main()
