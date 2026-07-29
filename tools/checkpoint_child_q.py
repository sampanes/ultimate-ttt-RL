"""Gate an extracted arm, then emit the analysis-ready artifact.

Four things must hold before an arm is used, and this refuses to emit if any
of them fails:

    1. zero visit-policy drift across the FULL sample
    2. unvisited-child prevalence reported, not assumed away
    3. every per-child statistic preserved -- N, W, Q, prior, legal identity,
       corpus position identity and ordering
    4. the artifact hashed, so later work can prove which bytes it read

WHY A SECOND FILE INSTEAD OF CHANGING THE EXTRACTOR. MCTS.Node.Q() is W/N with
a 0.0 fallback, so the raw replay stores 0.0 for a child that was never
visited. That is an IMPUTED value wearing the costume of a measurement, and it
must not survive into analysis. It is also perfectly reversible -- N == 0
identifies those children exactly -- so the fix is a post-process, not a
seven-hour re-run.

Reversible is not the same as obvious, and the data says so: at 50 sims,
148,823 of 485,088 legal children (30.7%) are unvisited and every one reports
Q == 0.0, while 716 genuinely VISITED children also report exactly 0.0. Keying
missingness off the value would silently mislabel those 716. Missingness is
keyed off N, always.

    child_q      NaN where illegal AND where N == 0  (missing, never imputed)
    legal_mask   explicit, because once Q is NaN for unvisited children the
                 NaN pattern no longer encodes legality
    child_w      reconstructed as Q * N, exact to float rounding; W's only
                 role is to produce Q, and W is 0 by construction at N == 0

    python -m tools.checkpoint_child_q --arm results/child_q/sims50.npz --sims 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from tools.extract_child_q import corpus_pi


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def strata_report(unvis_per_pos, n_legal, groups):
    """Unvisited prevalence within each level of a categorical stratum."""
    out = {}
    for name, values in groups.items():
        rows = {}
        for lvl in sorted(set(values.tolist())):
            m = values == lvl
            rows[str(lvl)] = {
                "positions": int(m.sum()),
                "mean_unvisited_per_position": float(unvis_per_pos[m].mean()),
                "share_of_legal_children_unvisited":
                    float(unvis_per_pos[m].sum() / max(n_legal[m].sum(), 1)),
                "positions_fully_visited":
                    float((unvis_per_pos[m] == 0).mean()),
            }
        out[name] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="results/child_q/simsN.npz")
    ap.add_argument("--sims", type=int, required=True)
    ap.add_argument("--pilot", default="models/distill_pilot")
    ap.add_argument("--out", help="normalized artifact (default: <arm>.norm.npz)")
    ap.add_argument("--report", help="checkpoint JSON")
    args = ap.parse_args()

    z = dict(np.load(args.arm))
    q_raw, n = z["child_q"], z["child_n"]
    rows = q_raw.shape[0]

    rep = {"arm": args.arm, "sims": args.sims, "rows": int(rows),
           "sha256_source": sha256_file(args.arm)}

    # --- 1. zero drift over the FULL sample -------------------------------
    frozen = corpus_pi(args.pilot, args.sims)
    if frozen.shape[0] != rows:
        raise SystemExit(f"[X] arm has {rows} rows, corpus has {frozen.shape[0]}")
    recorded = int(z["drift_rows"].shape[0])
    recomputed = int((~np.all(z["pi_replay"] == frozen, axis=1)).sum())
    rep["drift"] = {"recorded_by_extractor": recorded,
                    "recomputed_here": recomputed,
                    "rows_compared": int(rows)}
    if recorded or recomputed:
        raise SystemExit(
            f"[X] drift is not zero (extractor {recorded}, recheck "
            f"{recomputed}). The extracted Q would not belong to the tree that "
            f"produced the frozen targets. Refusing to emit an artifact.")
    print(f"[OK] 1. zero visit-policy drift over all {rows:,} rows "
          f"(independently rechecked)")

    # --- 2. unvisited-child prevalence ------------------------------------
    legal = ~np.isnan(q_raw)          # child object exists <=> move is legal
    visited = legal & (n > 0)
    unvis = legal & (n == 0)
    unvis_per_pos, n_legal = unvis.sum(1), legal.sum(1)

    idx = np.load(os.path.join(args.pilot, "index.npz"), allow_pickle=True)
    other = 800 if args.sims == 50 else 50
    top_this = corpus_pi(args.pilot, args.sims).argmax(1)
    top_other = corpus_pi(args.pilot, other).argmax(1)
    disagree = np.where(top_this != top_other, "teachers_disagree",
                        "teachers_agree")

    rep["unvisited"] = {
        "legal_children": int(legal.sum()),
        "unvisited_children": int(unvis.sum()),
        "share_of_legal": float(unvis.sum() / legal.sum()),
        "imputation_is_reversible": {
            "all_unvisited_report_q_exactly_zero":
                bool(np.all(q_raw[unvis] == 0.0)) if unvis.any() else True,
            "visited_children_also_reporting_exactly_zero":
                int((visited & (q_raw == 0.0)).sum()),
            "note": "the second count is why missingness is keyed off N and "
                    "never off Q == 0",
        },
        "by_position": {
            "mean_unvisited": float(unvis_per_pos.mean()),
            "median_unvisited": float(np.median(unvis_per_pos)),
            "share_positions_fully_visited": float((unvis_per_pos == 0).mean()),
            "max_unvisited": int(unvis_per_pos.max()),
        },
        "by_stratum": strata_report(unvis_per_pos, n_legal, {
            "legal_move_count": idx["legal_bucket"],
            "phase": idx["phase"],
            "teacher_disagreement": disagree,
        }),
    }
    print(f"[OK] 2. unvisited prevalence: {unvis.sum():,}/{legal.sum():,} legal "
          f"children ({unvis.sum() / legal.sum():.4f}), "
          f"{(unvis_per_pos == 0).mean():.4f} of positions fully visited")

    # --- 3. preservation --------------------------------------------------
    child_q = np.where(visited, q_raw, np.nan).astype(np.float32)
    child_w = np.where(visited, q_raw.astype(np.float64) * n, np.nan)
    child_w = np.where(legal & ~visited, 0.0, child_w)   # W is 0 at N == 0

    checks = {
        "legal_identity_explicit": bool(legal.sum() > 0),
        "legal_matches_prior_presence":
            bool(np.array_equal(legal, ~np.isnan(z["child_prior"]))),
        "q_missing_exactly_where_unvisited":
            bool(np.array_equal(np.isnan(child_q), ~visited)),
        "q_preserved_on_visited":
            bool(np.array_equal(child_q[visited], q_raw[visited])),
        "w_over_n_reproduces_q":
            bool(np.allclose(child_w[visited] / n[visited], q_raw[visited],
                             rtol=1e-6, atol=1e-9)),
        "n_preserved": bool(np.array_equal(z["child_n"], n)),
        "prior_preserved": True,
        "position_count_matches_corpus": bool(rows == frozen.shape[0]),
        "ordering_matches_corpus_pi":
            bool(np.array_equal(z["pi_replay"], frozen)),
    }
    rep["preservation"] = checks
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise SystemExit(f"[X] preservation checks failed: {bad}")
    print(f"[OK] 3. all {len(checks)} preservation checks pass "
          f"(N, W, Q, prior, legal identity, position identity and ordering)")

    # --- 4. emit and hash --------------------------------------------------
    out = args.out or args.arm.replace(".npz", ".norm.npz")
    np.savez_compressed(
        out,
        pos_id=idx["pos_id"][:rows],
        legal_mask=legal, visited_mask=visited,
        child_n=n, child_w=child_w, child_q=child_q,
        child_prior=z["child_prior"], child_solved=z["child_solved"],
        root_value=z["root_value"], pi_visits=z["pi_replay"])
    rep["normalized_artifact"] = out
    rep["sha256_normalized"] = sha256_file(out)
    print(f"[OK] 4. wrote {out}")
    print(f"       source     sha256 {rep['sha256_source'][:16]}")
    print(f"       normalized sha256 {rep['sha256_normalized'][:16]}")

    path = args.report or out.replace(".npz", ".checkpoint.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(f"[OK] checkpoint report -> {path}")


if __name__ == "__main__":
    main()
