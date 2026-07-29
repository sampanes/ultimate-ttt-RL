"""Characterize visit-versus-Q disagreement in the frozen pilot corpora.

DESCRIPTIVE ONLY. This fits nothing, tunes nothing and selects no threshold. It
exists so that a target transformation, if one is ever preregistered, is chosen
from an observed failure mode rather than from a candidate list by inertia.

Three parts, in the order the owner specified:

  1. WITHIN each arm: does the visit policy agree with the arm's own child
     values? Visit argmax versus max-Q visited child, Spearman rank correlation
     over visited legal moves, visit mass placed outside the top-Q move, and the
     top-visit move's Q deficit from the best Q available -- stratified by
     tactical status, phase, branching factor and visit coverage.

  2. BETWEEN the arms: what changed from 50 to 800 sims. Top-move change rate,
     the 800 top-two Q gap, the Q advantage of the 800 pick over the 50 pick,
     whether the 50 pick was even visited at 800, whether 800's visit and Q
     argmaxes agree, and the JS divergence between the two visit policies.

  3. The changed-top-move positions split on the FIXED 0.013 reference, plus the
     continuous distribution behind the bins.

THE 800 TREE IS THE COMPARISON SURFACE. Every cross-arm value comparison is
evaluated in the 800 tree, because at 800 sims every legal child is visited
while at 50 sims 30.7% are not. Averaging a 50-Q against an 800-Q would average
a censored quantity against a complete one and call the result a value gap.

MISSING IS MISSING. The normalized artifacts carry NaN wherever N == 0, because
MCTS.Node.Q() is W/N with a 0.0 fallback and an unvisited child would otherwise
be indistinguishable from a dead-even one. Every statistic here masks to visited
children and reports how many it discarded. Run tools/checkpoint_child_q.py
first; this refuses to read the raw extractor output.

0.013 IS A RULER, NOT A CUT. It is the swapped-move value gap already measured
in RESULT_SEARCH_DISAGREEMENT.md, fixed before this distribution was seen. The
three bins are explanatory, not a claim that 0.013 is the right training
threshold.

    python -m tools.characterize_visit_vs_q --child-q results/child_q \
        --pilot models/distill_pilot --output results/child_q/characterization.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Fixed in advance, from RESULT_SEARCH_DISAGREEMENT.md's adjudication of the
# moves the two budgets swap. Quoted as a scale, never as a cut chosen here.
REFERENCE_GAP = 0.013

# Quantiles reported for every continuous quantity, so the shape is visible
# without any single point being privileged as a threshold.
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def qsummary(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            **{f"p{int(q * 100)}": float(np.quantile(v, q)) for q in QUANTILES}}


def nanstat(fn, v):
    """None rather than a NaN warning when a stratum has nothing defined --
    single-legal-move positions have no rank correlation to report."""
    v = np.asarray(v, dtype=np.float64)
    return float(fn(v)) if np.isfinite(v).any() else None


def f4(x, w=11):
    return f"{'':>{w}}" if x is None else f"{x:>{w}.4f}"


def avg_rank(v):
    """Average ranks, ascending, ties shared. Visit counts tie constantly at
    low budgets, and competition ranks would invent an ordering among moves the
    search never distinguished."""
    n = v.size
    order = np.argsort(v, kind="stable")
    sv = v[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rho, or NaN when either side is entirely tied (undefined)."""
    if a.size < 2:
        return np.nan
    ra, rb = avg_rank(a), avg_rank(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def js_divergence(p, q):
    """Jensen-Shannon divergence in BITS, so it is bounded in [0, 1] and two
    policies can be compared against that ceiling rather than an open scale."""
    m = 0.5 * (p + q)
    out = 0.0
    for d in (p, q):
        nz = d > 0
        out += 0.5 * float(np.sum(d[nz] * np.log2(d[nz] / m[nz])))
    return out


def load_arm(child_q_dir, sims):
    path = os.path.join(child_q_dir, f"sims{sims}.norm.npz")
    if not os.path.exists(path):
        raise SystemExit(
            f"[X] {path} not found. Run tools.checkpoint_child_q on the raw "
            f"arm first -- the raw extractor stores Q()==0.0 for unvisited "
            f"children, and reading it here would silently treat 148k missing "
            f"values as measured zeros.")
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    pi = d["pi_visits"].astype(np.float64)
    tot = pi.sum(1, keepdims=True)
    d["pi_norm"] = np.divide(pi, tot, out=np.zeros_like(pi), where=tot > 0)
    # PERSPECTIVE -- the single easiest thing to get backwards here.
    #
    # child_q is stored in the CHILD's to_play frame, and the child's mover is
    # the OPPONENT of the player choosing the move. MCTS._best_child scores
    # -c.Q() for exactly this reason. So the value of a move TO THE MOVER is
    # the negation, and every comparison against the visit counts must use it.
    #
    # Comparing visits against raw child_q measures agreement between what the
    # search played and what would be good FOR THE OPPONENT, which produces a
    # near-perfect inversion (Spearman -0.74) that reads as a catastrophic
    # search defect and is nothing but the sign. The tell is mate-in-1: a
    # winning child is terminal at -1 in its own frame, so on raw child_q the
    # forced win looks like the worst move on the board.
    d["child_v"] = -d["child_q"]
    return d


def within_arm(d):
    """Part 1 statistics, per position."""
    q, n = d["child_v"], d["child_n"]      # mover's frame -- see load_arm
    legal, visited = d["legal_mask"], d["visited_mask"]
    rows = q.shape[0]
    pi = d["pi_norm"]

    qm = np.where(visited, q, -np.inf)
    nm = np.where(legal, n, -1)
    top_visit = nm.argmax(1)
    top_q = qm.argmax(1)
    r = np.arange(rows)

    best_q = qm.max(1)
    q_of_top_visit = np.where(visited[r, top_visit], q[r, top_visit], np.nan)

    srt = np.sort(qm, axis=1)[:, ::-1]
    top_two_gap = srt[:, 0] - srt[:, 1]
    top_two_gap[~np.isfinite(top_two_gap)] = np.nan

    rho = np.full(rows, np.nan)
    for i in range(rows):
        idx = np.flatnonzero(visited[i])
        if idx.size >= 2:
            rho[i] = spearman(n[i, idx].astype(np.float64),
                              q[i, idx].astype(np.float64))

    n_legal = legal.sum(1)
    n_vis = visited.sum(1)
    return {
        "top_visit": top_visit, "top_q": top_q,
        "visit_is_q_best": top_visit == top_q,
        # >= 0 by construction: how much value the most-visited move gives up
        # against the best value the search itself found.
        "q_deficit": best_q - q_of_top_visit,
        "top_two_q_gap": top_two_gap,
        "spearman": rho,
        # Visit mass the target puts anywhere other than the best-Q move.
        "mass_outside_top_q": 1.0 - pi[r, top_q],
        "n_legal": n_legal, "n_visited": n_vis,
        "coverage": n_vis / np.maximum(n_legal, 1),
        "legal": legal, "visited": visited,
    }


def coverage_bucket(cov):
    out = np.full(cov.shape, "<0.50", dtype=object)
    out[cov >= 0.50] = "0.50-0.75"
    out[cov >= 0.75] = "0.75-1.00"
    out[cov >= 0.9999] = "1.00 (complete)"
    return np.array(out)


def strata(idx, s, rows):
    return {
        "all": np.ones(rows, dtype=bool),
        **{f"phase:{p}": idx["phase"][:rows] == p
           for p in ("early", "mid", "late")},
        **{f"legal:{b}": idx["legal_bucket"][:rows] == b
           for b in ("1", "2-4", "5-8", "9+")},
        "tactical:mate_in_1": idx["immediate_win"][:rows].astype(bool),
        "tactical:none": ~idx["immediate_win"][:rows].astype(bool),
        **{f"coverage:{c}": coverage_bucket(s["coverage"]) == c
           for c in ("1.00 (complete)", "0.75-1.00", "0.50-0.75", "<0.50")},
    }


def sign_check(arms, idx, rows, out):
    """Refuse to report anything if the value frame is inverted.

    On a mate-in-1 the winning move is TERMINAL at -1 in the child's own frame,
    which makes it the maximum in the mover's frame. A search that reliably
    finds mates therefore cannot disagree with the mover-frame best value most
    of the time. If it appears to, the sign is wrong -- which is precisely the
    bug the first run of this file shipped with, and it read as a catastrophic
    search defect rather than as an analysis error.
    """
    _d, s = arms[800]
    mate = idx["immediate_win"][:rows].astype(bool)
    if not mate.any():
        return None
    agree = float(s["visit_is_q_best"][mate].mean())
    print(f"\n[sign check] over {int(mate.sum()):,} mate-in-1 positions, the "
          f"800-sim visit argmax is the mover-frame best value "
          f"{agree:.4f} of the time")
    if agree < 0.5:
        raise SystemExit(
            "[X] the value frame is INVERTED. child_q is stored in the child's "
            "to_play frame and must be negated before it is compared with "
            "anything the mover chose (see load_arm). Refusing to emit tables.")
    out["sign_check_mate_in_1_agreement"] = agree
    return agree


def part1(arms, idx, rows, out):
    print("\n" + "=" * 78)
    print("PART 1 -- visit versus Q disagreement WITHIN each arm")
    print("=" * 78)
    block = {}
    for sims, (d, s) in arms.items():
        cens = int((s["legal"] & ~s["visited"]).sum())
        # V = the child value in the MOVER's frame (-child_q). Named apart from
        # Q throughout so a reader cannot mistake which sign is on the table.
        head = (f"  {'stratum':<22}{'n':>7}{'visit!=Vbest':>14}"
                f"{'spearman':>11}{'mass off V':>12}{'V deficit':>11}")
        print(f"\n  arm sims={sims}   "
              f"{cens:,} of {int(s['legal'].sum()):,} legal children unvisited "
              f"({cens / s['legal'].sum():.4f}) -- excluded from every column")
        print(head)
        print("  " + "-" * (len(head) - 2))
        rowsout = {}
        for name, m in strata(idx, s, rows).items():
            if not m.any():
                continue
            rowsout[name] = {
                "n": int(m.sum()),
                "argmax_disagreement_rate":
                    float(1.0 - s["visit_is_q_best"][m].mean()),
                "spearman_mean": nanstat(np.nanmean, s["spearman"][m]),
                "spearman_median": nanstat(np.nanmedian, s["spearman"][m]),
                "spearman_undefined_share":
                    float(np.mean(~np.isfinite(s["spearman"][m]))),
                "mass_outside_top_v_mean":
                    nanstat(np.nanmean, s["mass_outside_top_q"][m]),
                "v_deficit_mean": nanstat(np.nanmean, s["q_deficit"][m]),
                "v_deficit_median": nanstat(np.nanmedian, s["q_deficit"][m]),
                "share_v_deficit_over_reference":
                    float(np.nanmean(s["q_deficit"][m] > REFERENCE_GAP)),
                "mean_visited_children": float(s["n_visited"][m].mean()),
            }
            v = rowsout[name]
            print(f"  {name:<22}{v['n']:>7,}{v['argmax_disagreement_rate']:>14.4f}"
                  f"{f4(v['spearman_mean'])}{f4(v['mass_outside_top_v_mean'], 12)}"
                  f"{f4(v['v_deficit_mean'])}")
        block[f"sims{sims}"] = {
            "unvisited_legal_children": cens,
            "unvisited_share_of_legal": float(cens / s["legal"].sum()),
            "value_frame": "mover (-child_q); see load_arm",
            "v_deficit_distribution": qsummary(s["q_deficit"]),
            "spearman_distribution": qsummary(s["spearman"]),
            "top_two_v_gap_distribution": qsummary(s["top_two_q_gap"]),
            "by_stratum": rowsout,
        }
    out["part1_within_arm"] = block


def part2(arms, idx, rows, out):
    print("\n" + "=" * 78)
    print("PART 2 -- what changed from 50 to 800 sims (evaluated in the 800 tree)")
    print("=" * 78)
    d50, s50 = arms[50]
    d800, s800 = arms[800]
    r = np.arange(rows)
    mv50, mv800 = s50["top_visit"], s800["top_visit"]
    changed = mv50 != mv800

    q800 = d800["child_v"]                 # mover's frame -- see load_arm
    vis800 = d800["visited_mask"]
    # Q of each arm's pick, both read off the 800 tree. NaN if 800 never
    # visited that move -- which is itself one of the reported quantities, not
    # something to paper over with a fallback value.
    q800_of_800 = np.where(vis800[r, mv800], q800[r, mv800], np.nan)
    q800_of_50 = np.where(vis800[r, mv50], q800[r, mv50], np.nan)
    adv = q800_of_800 - q800_of_50
    fifty_visited_at_800 = vis800[r, mv50]

    js = np.empty(rows)
    for i in range(rows):
        js[i] = js_divergence(d50["pi_norm"][i], d800["pi_norm"][i])

    block = {
        "positions": int(rows),
        "top_move_changed": int(changed.sum()),
        "top_move_change_rate": float(changed.mean()),
        "fifty_pick_visited_at_800": {
            "all": float(fifty_visited_at_800.mean()),
            "on_changed": float(fifty_visited_at_800[changed].mean()),
        },
        "argmax_agreement_within_800": {
            "visit_argmax_equals_q_argmax": float(s800["visit_is_q_best"].mean()),
            "on_changed": float(s800["visit_is_q_best"][changed].mean()),
            "on_unchanged": float(s800["visit_is_q_best"][~changed].mean()),
        },
        "top_two_q_gap_800": {
            "all": qsummary(s800["top_two_q_gap"]),
            "on_changed": qsummary(s800["top_two_q_gap"][changed]),
            "on_unchanged": qsummary(s800["top_two_q_gap"][~changed]),
        },
        "q_advantage_of_800_pick_over_50_pick": {
            "on_changed": qsummary(adv[changed]),
            "share_positive": float(np.nanmean(adv[changed] > 0)),
            "share_negative": float(np.nanmean(adv[changed] < 0)),
            "share_within_reference": float(
                np.nanmean(np.abs(adv[changed]) < REFERENCE_GAP)),
        },
        "js_divergence_bits": {
            "all": qsummary(js),
            "on_changed": qsummary(js[changed]),
            "on_unchanged": qsummary(js[~changed]),
            "note": "base 2, so 1.0 is the maximum for disjoint supports",
        },
    }
    b = block
    print(f"\n  top move changed          {b['top_move_changed']:,} / {rows:,} "
          f"({b['top_move_change_rate']:.4f})")
    print(f"  50's pick visited at 800  {b['fifty_pick_visited_at_800']['all']:.4f} "
          f"overall, {b['fifty_pick_visited_at_800']['on_changed']:.4f} on changed")
    a = b["argmax_agreement_within_800"]
    print(f"  800 visit argmax == Q argmax   {a['visit_argmax_equals_q_argmax']:.4f} "
          f"(changed {a['on_changed']:.4f}, unchanged {a['on_unchanged']:.4f})")
    g = b["top_two_q_gap_800"]
    print(f"  800 top-two Q gap         median {g['all']['p50']:.4f}  "
          f"changed {g['on_changed']['p50']:.4f}  "
          f"unchanged {g['on_unchanged']['p50']:.4f}")
    v = b["q_advantage_of_800_pick_over_50_pick"]
    print(f"  Q800(800 pick) - Q800(50 pick) on changed: "
          f"median {v['on_changed']['p50']:+.4f}  mean {v['on_changed']['mean']:+.4f}")
    print(f"      positive {v['share_positive']:.4f}  "
          f"negative {v['share_negative']:.4f}  "
          f"|adv| < {REFERENCE_GAP} {v['share_within_reference']:.4f}")
    j = b["js_divergence_bits"]
    print(f"  JS(pi50, pi800) bits      median {j['all']['p50']:.4f}  "
          f"changed {j['on_changed']['p50']:.4f}  "
          f"unchanged {j['on_unchanged']['p50']:.4f}")
    out["part2_fifty_vs_eight_hundred"] = block
    return changed, adv


def part3(changed, adv, out):
    print("\n" + "=" * 78)
    print(f"PART 3 -- changed-top-move positions split on the fixed "
          f"{REFERENCE_GAP} reference")
    print("=" * 78)
    a = adv[changed]
    finite = np.isfinite(a)
    bins = {
        "negative": a < 0.0,
        "near_equivalent": (a >= 0.0) & (a < REFERENCE_GAP),
        "meaningful": a >= REFERENCE_GAP,
    }
    rowsout = {}
    total = int(finite.sum())
    print(f"\n  {int(changed.sum()):,} changed positions, {total:,} with a "
          f"defined advantage "
          f"({int((~finite).sum()):,} undefined -- 800 never visited the 50 pick)")
    print(f"\n  {'bin':<18}{'n':>9}{'share':>10}{'median adv':>13}")
    print("  " + "-" * 48)
    for name, m in bins.items():
        m = m & finite
        rowsout[name] = {
            "n": int(m.sum()),
            "share_of_changed_with_defined_adv":
                float(m.sum() / total) if total else None,
            "median_advantage": (float(np.median(a[m])) if m.any() else None),
        }
        print(f"  {name:<18}{int(m.sum()):>9,}"
              f"{rowsout[name]['share_of_changed_with_defined_adv']:>10.4f}"
              f"{(rowsout[name]['median_advantage'] or 0.0):>13.4f}")
    out["part3_changed_split"] = {
        "reference_gap": REFERENCE_GAP,
        "reference_gap_provenance":
            "RESULT_SEARCH_DISAGREEMENT.md swapped-move value gap, fixed before "
            "this distribution was measured",
        "changed_positions": int(changed.sum()),
        "defined_advantage": total,
        "undefined_advantage": int((~finite).sum()),
        "bins": rowsout,
        "continuous_distribution": qsummary(a),
        "caveat": "The bins are explanatory. They are not evidence that 0.013 "
                  "is the right training threshold, and this file selects no "
                  "threshold at all.",
    }
    c = out["part3_changed_split"]["continuous_distribution"]
    print(f"\n  continuous: mean {c['mean']:+.4f}  p10 {c['p10']:+.4f}  "
          f"p50 {c['p50']:+.4f}  p90 {c['p90']:+.4f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child-q", default="results/child_q")
    ap.add_argument("--pilot", default="models/distill_pilot")
    ap.add_argument("--output", default="results/child_q/characterization.json")
    args = ap.parse_args()

    arms = {}
    for sims in (50, 800):
        d = load_arm(args.child_q, sims)
        arms[sims] = (d, within_arm(d))
    rows = arms[50][0]["child_q"].shape[0]
    if arms[800][0]["child_q"].shape[0] != rows:
        raise SystemExit("[X] the two arms hold different position counts")

    idx = np.load(os.path.join(args.pilot, "index.npz"), allow_pickle=True)
    out = {"positions": int(rows), "reference_gap": REFERENCE_GAP}

    sign_check(arms, idx, rows, out)
    part1(arms, idx, rows, out)
    changed, adv = part2(arms, idx, rows, out)
    part3(changed, adv, out)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
