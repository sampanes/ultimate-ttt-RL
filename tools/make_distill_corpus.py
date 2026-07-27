"""Matched distillation corpora: same positions, teachers differing only in sims.

Generates the pilot described in RESULT_TEACHER_SIM_LADDER.md's closing section.
For one shared set of positions drawn from the existing gen-22 corpus, it
replays the frozen gen-22 teacher's MCTS at each requested simulation count and
writes one training corpus per arm. No new self-play: the positions already
exist, only the targets are recomputed.

The arms are identical in every respect except the policy target:

  x  identical across arms (byte-for-byte -- asserted at write time)
  z  identical across arms; the ORIGINAL corpus game outcome is carried through
  pi THE TREATMENT -- the arm's own full visit distribution

WHY z IS NOT THE SEARCH VALUE. Using each arm's root Q as the value target would
make the arms differ in two ways at once, and it is not what the real pipeline
does: expert iteration pairs visit-count policy targets with game-outcome value
targets. Carrying the original z through isolates the policy target as the sole
independent variable and keeps the pilot faithful to the production setup. Each
arm's root Q is still recorded in the index as `q_root_<sims>` so a value-target
variant can be tried later without regenerating anything.

WHY THE SAMPLE IS NOT PHASE-STRATIFIED. tools/analyze_search_disagreement.py
stratifies equally across early/mid/late because it is measuring per-phase
effects. A TRAINING corpus must not: a phase-balanced student learns a different
function from the one the real pipeline produces. This samples uniformly at
random from the pool, preserving the corpus's natural phase distribution.

Output layout, directly consumable by scripts/train_student_offline.py:

    <out>/sims50/data/shard_*.pt      {x, pi, z, teacher_gen}
    <out>/sims800/data/shard_*.pt     {x, pi, z, teacher_gen}
    <out>/index.npz                   per-position strata + q_root per arm
    <out>/manifest.json               what was generated and from what

    python -m tools.make_distill_corpus --positions 50000 --sims 50 800
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import time

import numpy as np
import torch

from engine.rules import rule_utl_valid_moves
from tools.analyze_search_disagreement import (
    LEGAL_BUCKETS, MCTS, build_model, legal_bucket, phase_of, state_from_planes)
from tools.summarize_search_disagreement import forced_target_state, tactical_flags


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pool_xz(corpus, max_shards, rng):
    """Pool of (planes, outcome). load_pool in the analysis tool drops z."""
    shards = sorted(glob.glob(os.path.join(corpus, "data", "shard_*.pt"))
                    or glob.glob(os.path.join(corpus, "shard_*.pt")))
    if not shards:
        raise SystemExit(f"[X] no shards under {corpus} or {corpus}/data")
    pick = shards if len(shards) <= max_shards else rng.sample(shards, max_shards)
    pick = sorted(pick)
    xs, zs = [], []
    for p in pick:
        d = torch.load(p, map_location="cpu", weights_only=False)
        xs.append(d["x"])
        zs.append(d["z"])
    pool_x, pool_z = torch.cat(xs), torch.cat(zs)
    print(f"pool: {pool_x.shape[0]:,} positions from {len(pick)} shards")
    return pool_x, pool_z


def sample_positions(pool_x, pool_z, n_wanted, seed):
    """Uniform random sample of reconstructable positions, natural distribution."""
    rng = random.Random(seed)
    order = list(range(pool_x.shape[0]))
    rng.shuffle(order)
    kept, rejects = [], 0
    for i in order:
        if len(kept) >= n_wanted:
            break
        rec = state_from_planes(pool_x[i])
        if rec is None:
            rejects += 1
            continue
        st, legal, filled = rec
        kept.append((i, st, legal, filled, float(pool_z[i])))
    if len(kept) < n_wanted:
        raise SystemExit(f"[X] pool exhausted: {len(kept):,} < {n_wanted:,}. "
                         f"Raise --max-shards.")
    print(f"sampled {len(kept):,} positions (reconstruction rejects: {rejects})")
    return kept


def write_arm(out_dir, xs, pis, zs, teacher_gen, shard_size):
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for old in glob.glob(os.path.join(data_dir, "shard_*.pt")):
        os.remove(old)
    n = xs.shape[0]
    for s, start in enumerate(range(0, n, shard_size)):
        stop = min(start + shard_size, n)
        torch.save({"x": xs[start:stop], "pi": pis[start:stop],
                    "z": zs[start:stop], "teacher_gen": teacher_gen},
                   os.path.join(data_dir, f"shard_{s:05d}.pt"))
    print(f"  wrote {out_dir} -- {n:,} examples in "
          f"{(n + shard_size - 1) // shard_size} shards")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="models/corpus_gen22")
    ap.add_argument("--checkpoint", default="models/expert_iter_v2/teacher.pt")
    ap.add_argument("--out", default="models/distill_pilot")
    ap.add_argument("--positions", type=int, default=50000)
    ap.add_argument("--sims", type=int, nargs="+", default=[50, 800])
    ap.add_argument("--max-shards", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true",
                    help="skip arms whose corpus already has the right count")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    pool_x, pool_z = load_pool_xz(args.corpus, args.max_shards, rng)
    sample = sample_positions(pool_x, pool_z, args.positions, args.seed)

    xs = torch.stack([pool_x[i] for i, _, _, _, _ in sample]).contiguous()
    zs = torch.tensor([z for _, _, _, _, z in sample], dtype=torch.float32)

    model = build_model(args.checkpoint, args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    teacher_gen = int(payload.get("gen", 0) or 0)

    # Per-position strata, computed once -- arm-independent by construction.
    strata = {"pos_id": [], "phase": [], "legal_count": [], "legal_bucket": [],
              "forced_target": [], "immediate_win": [], "mini_win": []}
    for pool_idx, st, legal, filled, _z in sample:
        flags = tactical_flags(st, legal)
        strata["pos_id"].append(pool_idx)
        strata["phase"].append(phase_of(filled))
        strata["legal_count"].append(len(legal))
        strata["legal_bucket"].append(legal_bucket(len(legal)))
        strata["forced_target"].append(forced_target_state(st))
        strata["immediate_win"].append(bool(flags[0]))
        strata["mini_win"].append(bool(flags[1]))

    index = {k: np.array(v) for k, v in strata.items()}
    timings = {}

    total_est = sum(args.positions * (0.0384 * s / 50.0) for s in args.sims)
    print(f"estimated search cost: {total_est / 3600.0:.1f}h for "
          f"{len(args.sims)} arms x {args.positions:,} positions")

    for s in args.sims:
        arm_dir = os.path.join(args.out, f"sims{s}")
        done = sorted(glob.glob(os.path.join(arm_dir, "data", "shard_*.pt")))
        if args.resume and done:
            have = sum(torch.load(p, map_location="cpu",
                                  weights_only=False)["x"].shape[0] for p in done)
            if have == args.positions:
                print(f"resume: sims={s} already complete ({have:,}), skipping")
                continue

        # eff_wave mirrors MCTS.search's own clamp; the 16-wave floor is
        # load-bearing (mcts.py records 1 wave -> 0.00 strength).
        eff = max(1, s // MCTS._MIN_WAVES)
        mcts = MCTS(model, args.device, n_sims=s, c_puct=1.5,
                    add_dirichlet_at_root=False, wave_size=eff)
        pis = torch.zeros((len(sample), 81), dtype=torch.float32)
        qroot = np.zeros(len(sample), dtype=np.float32)

        t0 = time.time()
        for k, (_pool_idx, st, _legal, _filled, _z) in enumerate(sample):
            pi, root = mcts.search(st.clone())
            pis[k] = torch.as_tensor(pi, dtype=torch.float32)
            qroot[k] = root.Q()
            if (k + 1) % 5000 == 0:
                el = time.time() - t0
                print(f"  sims={s}  {k + 1:,}/{len(sample):,}  {el:.0f}s  "
                      f"({el / (k + 1) * 1000:.1f} ms/pos, "
                      f"eta {el / (k + 1) * (len(sample) - k - 1) / 60:.0f} min)")
        dt = time.time() - t0
        timings[str(s)] = dt
        print(f"  sims={s} wave={eff}  {dt:.0f}s  ({dt / len(sample) * 1000:.1f} ms/pos)")

        write_arm(arm_dir, xs, pis, zs, teacher_gen, args.shard_size)
        index[f"q_root_{s}"] = qroot

    np.savez_compressed(os.path.join(args.out, "index.npz"), **index)

    # x must be identical across arms -- that is the whole design. Verify it
    # from what actually landed on disk rather than trusting the write path.
    hashes = {}
    for s in args.sims:
        shards = sorted(glob.glob(os.path.join(args.out, f"sims{s}",
                                               "data", "shard_*.pt")))
        h = hashlib.sha256()
        for p in shards:
            h.update(torch.load(p, map_location="cpu",
                                weights_only=False)["x"].numpy().tobytes())
        hashes[str(s)] = h.hexdigest()
    distinct = set(hashes.values())
    if len(distinct) != 1:
        raise SystemExit(f"[X] arms do not share identical x planes: {hashes}")
    print(f"[OK] all arms share identical x planes ({hashes[str(args.sims[0])][:16]})")

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": args.corpus,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "teacher_gen": teacher_gen,
        "positions": args.positions,
        "sims": args.sims,
        "max_shards": args.max_shards,
        "seed": args.seed,
        "shard_size": args.shard_size,
        "sampling": "uniform random, natural phase distribution (NOT stratified)",
        "value_target": "original corpus game outcome z, identical across arms",
        "policy_target": "arm's own full MCTS visit distribution",
        "dirichlet": False,
        "c_puct": 1.5,
        "x_sha256_shared": hashes[str(args.sims[0])],
        "search_seconds": timings,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
