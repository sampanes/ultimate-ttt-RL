"""Answer the three characterization questions before any transform is chosen.

    1. When 800 changes the top move, does its chosen move also have the best
       child Q?
    2. How often does visit-count rank disagree with child-Q rank?
    3. Is the student reversal concentrated in positions with very small Q gaps?

This is a DESCRIPTIVE pass. It fits nothing, tunes nothing, and picks no
threshold. The reference scale is the already-measured 0.013 swapped-move value
gap from RESULT_SEARCH_DISAGREEMENT.md, fixed before this distribution was
seen; the point of quoting it is that it was NOT chosen from these numbers.

TWO TRAPS THIS HANDLES EXPLICITLY.

An unvisited child reports Q() == 0.0, because Q() is W/N with a 0.0 fallback.
Zero is also a perfectly ordinary Q for a dead-even move, so an unvisited child
is indistinguishable from a balanced one by value alone. At 800 sims every
legal child gets a visit and it does not matter; at 50 sims it does. Every
statistic here masks to N > 0 and reports how many children that discards.

Question 3 cannot be answered from targets alone -- "the student reversal" is a
property of trained students, not of the corpus. What this file can measure is
where the two teachers' targets DISAGREE as a function of Q gap. That is a
necessary condition, not the reversal itself, and it is labelled as such rather
than quietly promoted.

    python -m tools.characterize_visit_vs_q --child-q results/child_q \
        --pilot models/distill_pilot --output results/child_q/characterization.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Fixed in advance, from RESULT_SEARCH_DISAGREEMENT.md's adjudication of moves
# the two budgets swap. Quoted as a RULER, never as a cut chosen from the
# distribution below.
REFERENCE_GAP = 0.013

# Reported so the shape of the distribution is visible around the ruler without
# any one of them being privileged as "the" threshold.
GAP_BINS = [0.0, 0.005, 0.013, 0.03, 0.10, 0.30, 2.01]


def ranks(vals, mask):
    """Competition ranks over the masked entries, best (largest) first."""
    out = np.full(vals.shape, -1, dtype=np.int32)
    for i in range(vals.shape[0]):
        idx = np.flatnonzero(mask[i])
        if idx.size == 0:
            continue
        order = idx[np.argsort(-vals[i, idx], kind="stable")]
        out[i, order] = np.arange(order.size, dtype=np.int32)
    return out


def kendall_tau(a, b, mask):
    """Per-row Kendall tau-b between two rankings, NaN when under 2 items."""
    n = a.shape[0]
    tau = np.full(n, np.nan)
    for i in range(n):
        idx = np.flatnonzero(mask[i])
        if idx.size < 2:
            continue
        x, y = a[i, idx].astype(float), b[i, idx].astype(float)
        conc = disc = 0
        for p in range(idx.size):
            for q in range(p + 1, idx.size):
                dx, dy = x[p] - x[q], y[p] - y[q]
                s = dx * dy
                if s > 0:
                    conc += 1
                elif s < 0:
                    disc += 1
        tot = conc + disc
        if tot:
            tau[i] = (conc - disc) / tot
    return tau


def load(child_q_dir, sims):
    z = np.load(os.path.join(child_q_dir, f"sims{sims}.npz"))
    return {k: z[k] for k in z.files}


def per_row_stats(d):
    q, n = d["child_q"], d["child_n"]
    legal = ~np.isnan(q)
    visited = legal & (n > 0)
    qm = np.where(visited, q, -np.inf)
    nm = np.where(legal, n, -1)

    top_visit = nm.argmax(1)
    top_q = qm.argmax(1)
    rows = np.arange(q.shape[0])

    # Best Q among moves OTHER than the visit-chosen one: how much value the
    # visit target gives up, negative when the visit pick is genuinely best.
    other = qm.copy()
    other[rows, top_visit] = -np.inf
    best_other = other.max(1)
    q_of_top_visit = qm[rows, top_visit]

    srt = np.sort(qm, axis=1)[:, ::-1]
    top_two_gap = srt[:, 0] - srt[:, 1]
    top_two_gap[~np.isfinite(top_two_gap)] = np.nan

    return {
        "legal": legal, "visited": visited,
        "top_visit": top_visit, "top_q": top_q,
        "visit_is_q_best": top_visit == top_q,
        # >0 means the visit pick is Q-best by this much; <0 means the search
        # most-visited a move its own value function does not rank first.
        "top_q_gap": q_of_top_visit - best_other,
        "top_two_q_gap": top_two_gap,
        "n_visited": visited.sum(1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child-q", default="results/child_q")
    ap.add_argument("--pilot", default="models/distill_pilot")
    ap.add_argument("--output")
    args = ap.parse_args()

    a50, a800 = load(args.child_q, 50), load(args.child_q, 800)
    s50, s800 = per_row_stats(a50), per_row_stats(a800)
    idx = np.load(os.path.join(args.pilot, "index.npz"), allow_pickle=True)
    n = a800["child_q"].shape[0]
    out = {"n_positions": int(n), "reference_gap": REFERENCE_GAP,
           "reference_gap_provenance":
               "RESULT_SEARCH_DISAGREEMENT.md swapped-move value gap, fixed "
               "before this distribution was measured"}

    # Coverage of the Q()==0 trap.
    out["unvisited_legal_children"] = {
        "sims50": {"count": int((s50["legal"] & ~s50["visited"]).sum()),
                   "share_of_legal": float((s50["legal"] & ~s50["visited"]).sum()
                                           / s50["legal"].sum())},
        "sims800": {"count": int((s800["legal"] & ~s800["visited"]).sum()),
                    "share_of_legal": float((s800["legal"] & ~s800["visited"]).sum()
                                            / s800["legal"].sum())},
        "note": "excluded from every statistic below; Q() is 0.0 for these, "
                "which is indistinguishable from a genuinely even move"}

    # --- Q1: when 800 changes the top move, is its pick also Q-best? ---
    changed = s50["top_visit"] != s800["top_visit"]
    q1 = {"n_changed": int(changed.sum()),
          "share_changed": float(changed.mean())}
    for tag, m in (("changed", changed), ("unchanged", ~changed), ("all", np.ones(n, bool))):
        q1[tag] = {
            "n": int(m.sum()),
            "800_pick_is_q_best": float(s800["visit_is_q_best"][m].mean()),
            "50_pick_is_q_best": float(s50["visit_is_q_best"][m].mean()),
            "median_top_q_gap_800": float(np.nanmedian(s800["top_q_gap"][m])),
            "median_top_two_q_gap_800": float(np.nanmedian(s800["top_two_q_gap"][m])),
        }
    # The sharpest form: on changed positions, does 800's new move beat the
    # move 50 preferred, by 800's OWN child values?
    rows = np.arange(n)
    q800 = np.where(s800["visited"], a800["child_q"], np.nan)
    adv = q800[rows, s800["top_visit"]] - q800[rows, s50["top_visit"]]
    q1["on_changed_800_pick_minus_50_pick_by_800_q"] = {
        "median": float(np.nanmedian(adv[changed])),
        "mean": float(np.nanmean(adv[changed])),
        "share_positive": float(np.nanmean(adv[changed] > 0)),
        "share_within_reference_gap":
            float(np.nanmean(np.abs(adv[changed]) < REFERENCE_GAP)),
    }
    out["q1_top_move_change_vs_q"] = q1

    # --- Q2: how often does visit rank disagree with Q rank? ---
    q2 = {}
    for tag, d, s in (("sims50", a50, s50), ("sims800", a800, s800)):
        rv = ranks(np.where(s["visited"], d["child_n"], -1).astype(np.float64),
                   s["visited"])
        rq = ranks(np.where(s["visited"], d["child_q"], -np.inf), s["visited"])
        tau = kendall_tau(rv, rq, s["visited"])
        q2[tag] = {
            "argmax_disagreement_rate": float(1.0 - s["visit_is_q_best"].mean()),
            "kendall_tau_mean": float(np.nanmean(tau)),
            "kendall_tau_median": float(np.nanmedian(tau)),
            "share_tau_below_0.5": float(np.nanmean(tau < 0.5)),
            "mean_visited_children": float(s["n_visited"].mean()),
        }
    out["q2_visit_vs_q_rank"] = q2

    # --- Q3: where do the two budgets disagree, as a function of Q gap? ---
    gap = s800["top_two_q_gap"]
    bins, lab = [], []
    for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        m = (gap >= lo) & (gap < hi)
        lab.append(f"[{lo:g},{hi:g})")
        bins.append({
            "n": int(m.sum()),
            "share_of_positions": float(m.mean()),
            "top_move_change_rate": float(changed[m].mean()) if m.any() else None,
            "800_pick_is_q_best": float(s800["visit_is_q_best"][m].mean()) if m.any() else None,
            "mate_in_1_share": float(idx["immediate_win"][:n][m].mean()) if m.any() else None,
        })
    out["q3_disagreement_by_q_gap"] = {
        "bins": dict(zip(lab, bins)),
        "share_top_two_gap_below_reference":
            float(np.nanmean(gap < REFERENCE_GAP)),
        "median_top_two_gap": float(np.nanmedian(gap)),
        "caveat": "This locates TARGET disagreement, not the student reversal. "
                  "The reversal is a property of trained students and cannot "
                  "be read off the corpus; this is a necessary condition only.",
    }

    print(json.dumps({k: v for k, v in out.items()
                      if k.startswith(("q1", "q2", "q3", "unvisited"))},
                     indent=2, default=str)[:4000])
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
