"""Does more search actually change the training target?

The proposed controlled-distillation study ({50,100,200} teacher sims, one fixed
student) only has an independent variable if the teacher's TARGETS differ across
sim counts. This measures that directly, and costs no new self-play: it replays
MCTS at several sim counts over positions already sitting in the gen-22 corpus.

Decision rule this is built to serve:
  * 50->100 and 100->200 disagree substantially  -> run the distillation study.
  * 200->800 is near-identical                   -> drop the 800 arm.
  * even 100->200 barely differs                 -> do not run it; the teacher
    has no headroom at this sim range, and the effort belongs in search QUALITY
    (symmetry folding, transpositions, tree reuse -- see MCTS_STATUS.md).

Two implementation facts worth knowing before reading the numbers:

1. BATCH SIZE IS BOUNDED BY SEARCH SEMANTICS, NOT VRAM. MCTS.search clamps
   eff_wave = min(wave_size, n_sims // 16); the 16-wave floor exists because
   collapsing waves destroys search quality (mcts.py records 38 waves -> 0.95,
   10 -> 0.70-0.80, 1 -> 0.00). So the largest legal batch is n_sims//16: 3 at
   50 sims, 50 at 800. Forcing a bigger batch would degrade the very targets
   being measured. A 6.77M net at wave<=50 uses tens of MB, so the card is
   nowhere near full and cannot be -- wall clock is the constraint here, not
   memory. Filling it would need batching across independent positions, i.e.
   the eval-server machinery, which is out of scope for a pre-check.

2. DIRICHLET NOISE IS OFF. We are measuring what search depth does to the
   target, so root exploration noise is excluded; it would add variance
   unrelated to sim count.

Positions are reconstructed from the stored (7,9,9) planes. Note channel 4 maps
a DRAWN mini-board to -1.0, identical to an O-won one, so mini_winners is NOT
recoverable from it -- it is recomputed from the board planes using the engine's
own rule_utl_check_mini_win. Channel 3 (legal moves) is then a free correctness
check: any position whose recomputed legal mask disagrees with the stored one is
dropped rather than silently measured.

    python -m tools.analyze_search_disagreement --sample-size 10000 \
        --sims 50 100 200 800 --output results/disagreement
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

from agents.agent_base import ModelConfigCNN
from agents.mcts import MCTS
from agents.neural_net_agent_3 import ConvNet
from engine.constants import EMPTY, X, O
from engine.game import _PyGameState
from engine.rules import _MINI_INDICES, rule_utl_check_mini_win, rule_utl_valid_moves
from scripts.train_alphazero import NETWORK_CONFIGS

# Game phase by occupied cells. Fixed bands rather than data terciles so the
# buckets mean the same thing across runs and corpora.
PHASE_BANDS = [("early", 0, 20), ("mid", 21, 45), ("late", 46, 81)]
LEGAL_BUCKETS = [("1", 1, 1), ("2-4", 2, 4), ("5-8", 5, 8), ("9+", 9, 81)]


def phase_of(filled):
    for name, lo, hi in PHASE_BANDS:
        if lo <= filled <= hi:
            return name
    return "late"


def legal_bucket(n):
    for name, lo, hi in LEGAL_BUCKETS:
        if lo <= n <= hi:
            return name
    return "9+"


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #

def state_from_planes(x):
    """Rebuild a playable state from the stored (7,9,9) planes, or None.

    Returns None when the recomputed legal-move mask disagrees with channel 3,
    which is the guard against a silently wrong reconstruction.
    """
    xp = x[0].numpy() if hasattr(x[0], "numpy") else x[0]
    op = x[1].numpy() if hasattr(x[1], "numpy") else x[1]
    board = [EMPTY] * 81
    for i in range(81):
        r, c = divmod(i, 9)
        if xp[r, c] > 0.5:
            board[i] = X
        elif op[r, c] > 0.5:
            board[i] = O

    player = X if float(x[2, 0, 0]) > 0 else O

    ch5 = x[5].numpy() if hasattr(x[5], "numpy") else x[5]
    last_move = None
    if ch5.max() > 0.5:
        flat = int(np.argmax(ch5))
        last_move = (flat // 9) * 9 + (flat % 9)

    # Recomputed, NOT read from channel 4 -- see module docstring.
    mini_winners = [EMPTY] * 9
    for m, idxs in enumerate(_MINI_INDICES):
        w = rule_utl_check_mini_win([board[i] for i in idxs])
        if w != EMPTY:
            mini_winners[m] = w

    st = _PyGameState(board=board, player=player, last_move=last_move,
                      mini_winners=mini_winners)

    recomputed = set(rule_utl_valid_moves(st.board, st.last_move,
                                          st.mini_winners))
    ch3 = x[3].numpy() if hasattr(x[3], "numpy") else x[3]
    stored = {(r * 9 + c) for r in range(9) for c in range(9) if ch3[r, c] > 0.5}
    if recomputed != stored or not recomputed:
        return None
    return st, sorted(recomputed), sum(1 for b in board if b != EMPTY)


# --------------------------------------------------------------------------- #
# Divergences
# --------------------------------------------------------------------------- #

def js_divergence(p, q, legal):
    """Jensen-Shannon divergence in BITS (base 2), so it lies in [0, 1].

    Well-defined when either policy puts zero mass on a move, which is why the
    decision threshold is expressed on this rather than on KL.
    """
    p, q = p[legal], q[legal]
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return float("nan")
    p, q = p / ps, q / qs
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def kl_divergence(p, q, legal, eps=1e-9):
    """KL(p||q) in bits over the legal support, epsilon-smoothed.

    Reported for continuity with the training loss, but JS is the number the
    decision rule uses: KL blows up wherever the lower-sim search never visited
    a move the higher-sim search likes, which is exactly the interesting case.
    """
    p, q = p[legal].astype(np.float64), q[legal].astype(np.float64)
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return float("nan")
    p, q = p / ps, (q / qs) + eps
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


# --------------------------------------------------------------------------- #

def load_pool(corpus, max_shards, rng):
    # ShardStore writes into <model_dir>/data, so accept either the run
    # directory or the shard directory itself.
    shards = sorted(glob.glob(os.path.join(corpus, "data", "shard_*.pt"))
                    or glob.glob(os.path.join(corpus, "shard_*.pt")))
    if not shards:
        raise SystemExit(f"[X] no shards under {corpus} or {corpus}/data")
    pick = shards if len(shards) <= max_shards else rng.sample(shards, max_shards)
    xs = []
    for p in pick:
        d = torch.load(p, map_location="cpu", weights_only=False)
        xs.append(d["x"])
    pool = torch.cat(xs)
    print(f"pool: {pool.shape[0]:,} positions from {len(pick)} shards")
    return pool


def build_sample(corpus, max_shards, sample_size, seed):
    """The stratified sample, as a deterministic function of (corpus, seed).

    Factored out so the analysis pass can recover the EXACT positions -- and
    therefore the boards, which are not written to the per-position table --
    without re-running any search. Both callers must consume the RNG in the
    same order, so they share this one implementation rather than duplicating
    it. Returns (sample, rejects) with sample[i] keyed by pos_id == i.
    """
    rng = random.Random(seed)
    pool = load_pool(corpus, max_shards, rng)
    order = list(range(pool.shape[0]))
    rng.shuffle(order)
    per_phase = sample_size // len(PHASE_BANDS)
    buckets = {name: [] for name, _, _ in PHASE_BANDS}
    rejected = 0
    for i in order:
        if all(len(v) >= per_phase for v in buckets.values()):
            break
        rec = state_from_planes(pool[i])
        if rec is None:
            rejected += 1
            continue
        st, legal, filled = rec
        ph = phase_of(filled)
        if len(buckets[ph]) < per_phase:
            buckets[ph].append((st, legal, filled))
    sample = [t for v in buckets.values() for t in v]
    return sample, rejected, buckets


def build_model(ckpt_path, device):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tanh = payload.get("value_tanh", True)
    sd = payload.get("state_dict", payload)
    cfg = ModelConfigCNN(value_tanh=tanh, model_dir="models/_disagree",
                         **NETWORK_CONFIGS["arena22"])
    model = ConvNet(cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    gen = payload.get("gen", "?")
    print(f"teacher: gen {gen}, value_tanh={tanh}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/expert_iter_v2/teacher.pt")
    ap.add_argument("--corpus", default="models/corpus_gen22")
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--sims", type=int, nargs="+", default=[50, 100, 200, 800])
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--output", default="results/disagreement")
    ap.add_argument("--max-shards", type=int, default=200,
                    help="Shards to draw the sample pool from.")
    ap.add_argument("--js-threshold", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    model = build_model(args.checkpoint, args.device)
    sims = sorted(args.sims)

    # ---- stratified sample (shared with the analysis pass) ----------------
    sample, rejected, buckets = build_sample(
        args.corpus, args.max_shards, args.sample_size, args.seed)
    print(f"sampled {len(sample):,} positions "
          + ", ".join(f"{k}={len(v)}" for k, v in buckets.items())
          + f"  (reconstruction rejects: {rejected})")
    if not sample:
        raise SystemExit("[X] no usable positions")

    # ---- search each position at every sim count --------------------------
    n = len(sample)
    pis = {s: np.zeros((n, 81), dtype=np.float32) for s in sims}
    qs = {s: np.zeros(n, dtype=np.float32) for s in sims}
    t_start = time.time()
    for s in sims:
        eff = max(1, s // MCTS._MIN_WAVES)      # the clamp, made explicit
        mcts = MCTS(model, args.device, n_sims=s, c_puct=1.5,
                    add_dirichlet_at_root=False, wave_size=eff)
        t0 = time.time()
        for k, (st, _legal, _f) in enumerate(sample):
            pi, root = mcts.search(st.clone())
            pis[s][k] = pi
            qs[s][k] = root.Q()
        dt = time.time() - t0
        print(f"  sims={s:>4} wave={eff:>3}  {dt:>7.1f}s  "
              f"({dt / n * 1000:.1f} ms/position)", flush=True)
    print(f"search total {time.time() - t_start:.1f}s")

    # ---- pairwise metrics -------------------------------------------------
    pairs = list(zip(sims, sims[1:]))
    rows = []
    for k, (st, legal, filled) in enumerate(sample):
        row = {"pos_id": k, "phase": phase_of(filled), "filled_cells": filled,
               "n_legal": len(legal)}
        for s in sims:
            row[f"argmax_{s}"] = int(np.argmax(pis[s][k]))
            row[f"rootq_{s}"] = float(qs[s][k])
        for a, b in pairs:
            tag = f"{a}_{b}"
            same = int(np.argmax(pis[a][k]) == np.argmax(pis[b][k]))
            js = js_divergence(pis[a][k], pis[b][k], legal)
            row[f"same_argmax_{tag}"] = same
            row[f"js_{tag}"] = js
            row[f"kl_{tag}"] = kl_divergence(pis[a][k], pis[b][k], legal)
            row[f"dq_{tag}"] = float(qs[b][k] - qs[a][k])
            row[f"disagree_{tag}"] = int(
                (not same) or (js == js and js >= args.js_threshold))
        rows.append(row)

    # ---- summary ----------------------------------------------------------
    def agg(subset, tag):
        if not subset:
            return None
        js = np.array([r[f"js_{tag}"] for r in subset], dtype=np.float64)
        js = js[~np.isnan(js)]
        kl = np.array([r[f"kl_{tag}"] for r in subset], dtype=np.float64)
        kl = kl[~np.isnan(kl)]
        dq = np.array([abs(r[f"dq_{tag}"]) for r in subset])
        return {
            "n": len(subset),
            "top_move_agreement": float(np.mean([r[f"same_argmax_{tag}"] for r in subset])),
            "selected_move_changed": float(1 - np.mean([r[f"same_argmax_{tag}"] for r in subset])),
            "js_mean": float(js.mean()) if js.size else None,
            "js_median": float(np.median(js)) if js.size else None,
            "js_p90": float(np.percentile(js, 90)) if js.size else None,
            "kl_mean": float(kl.mean()) if kl.size else None,
            "kl_median": float(np.median(kl)) if kl.size else None,
            "abs_root_value_delta_mean": float(dq.mean()),
            "abs_root_value_delta_median": float(np.median(dq)),
            "disagreement_fraction": float(np.mean([r[f"disagree_{tag}"] for r in subset])),
        }

    summary = {
        "checkpoint": args.checkpoint, "corpus": args.corpus,
        "sample_size": n, "sims": sims, "seed": args.seed,
        "js_threshold": args.js_threshold, "js_units": "bits (base 2), range [0,1]",
        "dirichlet_noise": False,
        "wave_sizes": {str(s): max(1, s // MCTS._MIN_WAVES) for s in sims},
        "reconstruction_rejects": rejected,
        "overall": {}, "by_phase": {}, "by_legal_moves": {},
    }
    for a, b in pairs:
        tag = f"{a}_{b}"
        summary["overall"][tag] = agg(rows, tag)
        summary["by_phase"][tag] = {
            ph: agg([r for r in rows if r["phase"] == ph], tag)
            for ph, _, _ in PHASE_BANDS}
        summary["by_legal_moves"][tag] = {
            lb: agg([r for r in rows if legal_bucket(r["n_legal"]) == lb], tag)
            for lb, _, _ in LEGAL_BUCKETS}

    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    disagree_any = [r for r in rows
                    if any(r[f"disagree_{a}_{b}"] for a, b in pairs)]
    _write_table(rows, os.path.join(args.output, "per_position"))
    _write_table(disagree_any, os.path.join(args.output, "disagreement_positions"))
    # Full visit distributions, as requested -- not just the selected move.
    np.savez_compressed(os.path.join(args.output, "policies.npz"),
                        **{f"pi_{s}": pis[s] for s in sims},
                        **{f"rootq_{s}": qs[s] for s in sims})

    # ---- report -----------------------------------------------------------
    print(f"\n{'pair':>10} {'top-move agree':>15} {'JS mean':>9} {'JS med':>8} "
          f"{'|dV| mean':>10} {'disagree':>9}")
    print("-" * 66)
    for a, b in pairs:
        o = summary["overall"][f"{a}_{b}"]
        print(f"{a}->{b:<5} {o['top_move_agreement']:>15.3f} "
              f"{o['js_mean']:>9.4f} {o['js_median']:>8.4f} "
              f"{o['abs_root_value_delta_mean']:>10.4f} "
              f"{o['disagreement_fraction']:>9.3f}")
    print(f"\ndisagreement subset (argmax differs OR JS >= {args.js_threshold}): "
          f"{len(disagree_any):,}/{n:,} = {len(disagree_any)/n:.1%}")
    print(f"wrote {args.output}/")


def _write_table(rows, stem):
    """Parquet when pyarrow is available, compressed CSV otherwise."""
    if not rows:
        open(stem + ".csv", "w").write("")
        return
    cols = list(rows[0].keys())
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table({c: [r[c] for r in rows] for c in cols}),
                       stem + ".parquet")
        print(f"  wrote {stem}.parquet ({len(rows):,} rows)")
    except ImportError:
        import csv
        import gzip
        with gzip.open(stem + ".csv.gz", "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {stem}.csv.gz ({len(rows):,} rows) "
              f"-- pyarrow not installed, so no .parquet")


if __name__ == "__main__":
    main()
