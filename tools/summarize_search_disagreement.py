"""Interpretation layer for analyze_search_disagreement.

Adds what is needed to actually act on the pre-check rather than eyeball it:
bootstrap confidence intervals, strata the search pass did not record, a
diagnostic dump of the most-disagreeing positions, and explicit decision flags.

It recomputes every metric from `policies.npz` (the full visit distributions)
rather than re-parsing the summary, and recovers the BOARDS -- which the search
pass never wrote down -- by replaying the sampler. That replay is exact because
sampling is a deterministic function of (corpus, seed, max_shards, sample_size),
and both passes call the same build_sample(), so the two cannot drift apart.
Running it costs no search.

    python -m tools.summarize_search_disagreement --input results/disagreement
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np

from engine.constants import DRAW, EMPTY, O, X
from engine.rules import _MINI_INDICES, rule_utl_get_next_mini
from tools.analyze_search_disagreement import (
    LEGAL_BUCKETS, PHASE_BANDS, build_sample, js_divergence, kl_divergence,
    legal_bucket, phase_of)


# --------------------------------------------------------------------------- #
# Strata the search pass did not record
# --------------------------------------------------------------------------- #

def mini_of(move):
    """Which mini-board (0-8) does flat board index `move` (0-80) live in?

    The board is row-major 9x9, so move = r*9 + c and the mini-board is
    (r//3)*3 + (c//3). This is NOT `move // 9`, which is the board ROW: the two
    agree only for moves in the leftmost mini of each band, so a row-based
    version looks plausible on spot checks and is wrong on 2/3 of the board.
    That exact mistake invalidated the mini_win_available and forced_target
    strata of every run before 2026-07-27; see the INVALID_STRATA note below.
    """
    return (move // 27) * 3 + (move % 9) // 3

def forced_target_state(st):
    """Is the mini-board the mover is SENT to open, won, drawn, or absent?

    This separates constrained positions from free ones, which is the axis most
    likely to govern how much search buys: with one legal mini-board the tree is
    narrow and deep, with a free choice it is wide and shallow.
    """
    if st.last_move is None:
        return "none"
    # The mini you are SENT to is the LOCAL cell of the last move,
    # (r%3)*3 + (c%3) -- not `last_move % 9`, which is the column.
    target = rule_utl_get_next_mini(st.last_move)
    w = st.mini_winners[target]
    if w in (X, O):
        return "won"
    cells = [st.board[i] for i in _MINI_INDICES[target]]
    if w == DRAW or all(c != EMPTY for c in cells):
        return "drawn"
    return "open"


def tactical_flags(st, legal):
    """Cheap one-move tactics: can the mover end the game, or win a mini?

    Only what is genuinely cheap -- one clone-and-move per legal move. Deeper
    threat detection (does the OPPONENT win next) needs a second ply and is
    deliberately left out.
    """
    immediate_win = False
    mini_win = False
    mover = st.player
    for m in legal:
        s = st.clone()
        s.make_move(m)
        if s.winner == mover:
            immediate_win = True
            mini_win = True
            break
        k = mini_of(m)
        if s.mini_winners[k] == mover and st.mini_winners[k] == EMPTY:
            mini_win = True
    return immediate_win, mini_win


def render_board(st):
    rows = []
    for r in range(9):
        cells = []
        for c in range(9):
            v = st.board[r * 9 + c]
            cells.append("." if v == EMPTY else ("X" if v == X else "O"))
            if c % 3 == 2 and c != 8:
                cells.append("|")
        rows.append(" ".join(cells))
        if r % 3 == 2 and r != 8:
            rows.append("-" * len(rows[-1]))
    return rows


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_ci(values, stat, n_boot, rng, alpha=0.05):
    """Percentile bootstrap CI over POSITIONS.

    This quantifies sampling variability of the position set only. The searches
    themselves are deterministic here (no Dirichlet noise, fixed net), so there
    is no search randomness for it to capture -- which is the correct reading:
    the interval says 'if I drew another 10k positions from this corpus', not
    'if I re-ran the search'.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    n = v.size
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draws[b] = stat(v[rng.integers(0, n, n)])
    return {"point": float(stat(v)),
            "lo": float(np.percentile(draws, 100 * alpha / 2)),
            "hi": float(np.percentile(draws, 100 * (1 - alpha / 2))),
            "n": int(n)}


def pair_stats(js, kl_ab, kl_ba, dq, changed, n_boot, rng):
    return {
        "top_move_disagreement_rate": bootstrap_ci(changed, np.mean, n_boot, rng),
        "js_mean": bootstrap_ci(js, np.mean, n_boot, rng),
        "js_median": bootstrap_ci(js, np.median, n_boot, rng),
        "root_value_change_mean_signed": bootstrap_ci(dq, np.mean, n_boot, rng),
        "root_value_change_mean_abs": bootstrap_ci(np.abs(dq), np.mean, n_boot, rng),
        # Reported, but NOT used by the decision rule: KL is unstable wherever
        # the lower-sim search never visited a move the higher-sim search likes,
        # which is precisely the interesting case. Epsilon-smoothed, both ways.
        "kl_ab_mean_smoothed": bootstrap_ci(kl_ab, np.mean, n_boot, rng),
        "kl_ba_mean_smoothed": bootstrap_ci(kl_ba, np.mean, n_boot, rng),
        "js_p90": float(np.nanpercentile(js, 90)) if len(js) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="results/disagreement")
    ap.add_argument("--corpus", default="models/corpus_gen22")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--max-shards", type=int, default=200)
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--js-threshold", type=float, default=0.05)
    ap.add_argument("--n-diagnostic", type=int, default=100)
    # Decision thresholds, exposed rather than buried.
    ap.add_argument("--differ-move-rate", type=float, default=0.02,
                    help="CI LOWER bound on move-change rate must exceed this "
                         "for targets to count as materially different.")
    ap.add_argument("--differ-js-median", type=float, default=0.005,
                    help="CI lower bound on median JS must exceed this too.")
    ap.add_argument("--identical-move-rate", type=float, default=0.05,
                    help="CI UPPER bound below this (with the JS test) means "
                         "near-identical targets.")
    ap.add_argument("--identical-js-median", type=float, default=0.01)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    npz = np.load(os.path.join(args.input, "policies.npz"))
    sims = sorted(int(k.split("_")[1]) for k in npz.files if k.startswith("pi_"))
    pis = {s: npz[f"pi_{s}"] for s in sims}
    qs = {s: npz[f"rootq_{s}"] for s in sims}
    n = pis[sims[0]].shape[0]
    print(f"loaded {n:,} positions, sims {sims}")

    sample, rejects, _ = build_sample(args.corpus, args.max_shards,
                                      args.sample_size, args.seed)
    if len(sample) != n:
        raise SystemExit(
            f"[X] sample replay gave {len(sample)} positions but policies.npz "
            f"has {n}. The corpus or the seed/shard args differ from the "
            f"search run -- refusing to join mismatched data.")
    print(f"replayed the identical sample ({rejects} reconstruction rejects)")

    # ---- per-position strata + metrics ------------------------------------
    meta = []
    for k, (st, legal, filled) in enumerate(sample):
        imm, miniw = tactical_flags(st, legal)
        meta.append({
            "pos_id": k, "phase": phase_of(filled), "filled_cells": filled,
            "n_legal": len(legal), "legal_bucket": legal_bucket(len(legal)),
            "forced_target": forced_target_state(st),
            "immediate_win_available": int(imm),
            "mini_win_available": int(miniw),
        })

    pairs = list(zip(sims, sims[1:]))
    per_pair = {}
    for a, b in pairs:
        js = np.empty(n); kab = np.empty(n); kba = np.empty(n)
        dq = np.empty(n); ch = np.empty(n)
        for k, (st, legal, _f) in enumerate(sample):
            js[k] = js_divergence(pis[a][k], pis[b][k], legal)
            kab[k] = kl_divergence(pis[a][k], pis[b][k], legal)
            kba[k] = kl_divergence(pis[b][k], pis[a][k], legal)
            dq[k] = qs[b][k] - qs[a][k]
            ch[k] = float(np.argmax(pis[a][k]) != np.argmax(pis[b][k]))
        per_pair[(a, b)] = dict(js=js, kab=kab, kba=kba, dq=dq, ch=ch)

    # ---- summary with CIs, overall and by stratum -------------------------
    out = {
        "input": args.input, "sample_size": n, "sims": sims,
        "n_bootstrap": args.n_boot, "js_units": "bits (base 2), range [0,1]",
        "ci": "percentile bootstrap over positions, 95%",
        "overall": {}, "by_phase": {}, "by_legal_bucket": {},
        "by_forced_target": {}, "by_tactical": {},
    }

    def slice_stats(mask, a, b):
        d = per_pair[(a, b)]
        if mask.sum() == 0:
            return None
        return pair_stats(d["js"][mask], d["kab"][mask], d["kba"][mask],
                          d["dq"][mask], d["ch"][mask], args.n_boot, rng)

    all_mask = np.ones(n, dtype=bool)
    for a, b in pairs:
        tag = f"{a}_{b}"
        out["overall"][tag] = slice_stats(all_mask, a, b)
        out["by_phase"][tag] = {
            ph: slice_stats(np.array([m["phase"] == ph for m in meta]), a, b)
            for ph, _, _ in PHASE_BANDS}
        out["by_legal_bucket"][tag] = {
            lb: slice_stats(np.array([m["legal_bucket"] == lb for m in meta]), a, b)
            for lb, _, _ in LEGAL_BUCKETS}
        out["by_forced_target"][tag] = {
            ft: slice_stats(np.array([m["forced_target"] == ft for m in meta]), a, b)
            for ft in ("open", "won", "drawn", "none")}
        out["by_tactical"][tag] = {
            "immediate_win_available": slice_stats(
                np.array([bool(m["immediate_win_available"]) for m in meta]), a, b),
            "no_immediate_win": slice_stats(
                np.array([not m["immediate_win_available"] for m in meta]), a, b),
            "mini_win_available": slice_stats(
                np.array([bool(m["mini_win_available"]) for m in meta]), a, b),
        }

    # ---- decision flags ---------------------------------------------------
    def differ(tag):
        s = out["overall"][tag]
        return (s["top_move_disagreement_rate"]["lo"] > args.differ_move_rate
                and s["js_median"]["lo"] > args.differ_js_median)

    def near_identical(tag):
        s = out["overall"][tag]
        return (s["top_move_disagreement_rate"]["hi"] < args.identical_move_rate
                and s["js_median"]["hi"] < args.identical_js_median)

    have = {f"{a}_{b}" for a, b in pairs}
    dec = {
        "thresholds": {
            "differ_move_rate_ci_lo": args.differ_move_rate,
            "differ_js_median_ci_lo": args.differ_js_median,
            "identical_move_rate_ci_hi": args.identical_move_rate,
            "identical_js_median_ci_hi": args.identical_js_median,
        },
        # The pre-check can only establish that an independent variable EXISTS.
        # Confirming a target difference is an IMPROVEMENT needs the gen-22
        # teacher measured at each sim count against the external ladder, which
        # has not been run. So the *_supported flags are provisional.
        "teacher_separation_measured": False,
        "teacher_separation_note":
            "Differing targets prove the independent variable is real, not "
            "that higher-sim targets are BETTER. Run the gen-22 teacher-side "
            "ladder at each sim count before attributing any student result to "
            "target quality.",
    }
    if "50_100" in have:
        dec["distillation_50_100_supported"] = bool(differ("50_100"))
    if "100_200" in have:
        dec["distillation_100_200_supported"] = bool(differ("100_200"))
    if "200_800" in have:
        # One-way inference that does NOT need teacher strength: targets that
        # are indistinguishable cannot train a student differently.
        dec["drop_800_arm"] = bool(near_identical("200_800"))
        dec["drop_800_arm_basis"] = (
            "200->800 targets near-identical" if dec["drop_800_arm"]
            else "200->800 targets differ materially; arm retained")
    out["decisions"] = dec

    with open(os.path.join(args.input, "summary_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- diagnostic dump: the most-disagreeing positions -------------------
    maxjs = np.nanmax(np.stack([per_pair[p]["js"] for p in pairs]), axis=0)
    top = np.argsort(-np.nan_to_num(maxjs))[:args.n_diagnostic]
    diag = []
    for k in top:
        st, legal, _f = sample[int(k)]
        entry = dict(meta[int(k)])
        entry["max_js_across_pairs"] = float(maxjs[k])
        entry["board"] = render_board(st)
        entry["to_play"] = "X" if st.player == X else "O"
        entry["last_move"] = st.last_move
        entry["legal_moves"] = legal
        entry["per_sim"] = {
            str(s): {
                "selected_move": int(np.argmax(pis[s][k])),
                "root_value": float(qs[s][k]),
                "policy": {str(m): round(float(pis[s][k][m]), 4)
                           for m in legal if pis[s][k][m] > 0},
            } for s in sims}
        entry["pairwise_js"] = {f"{a}_{b}": float(per_pair[(a, b)]["js"][k])
                                for a, b in pairs}
        diag.append(entry)
    with open(os.path.join(args.input, "diagnostic_top_js.json"), "w") as f:
        json.dump(diag, f, indent=2)

    # ---- strata table for the enriched columns ----------------------------
    cols = list(meta[0].keys())
    with gzip.open(os.path.join(args.input, "position_strata.csv.gz"),
                   "wt", newline="") as f:
        f.write(",".join(cols) + "\n")
        for m in meta:
            f.write(",".join(str(m[c]) for c in cols) + "\n")

    # ---- report -----------------------------------------------------------
    print(f"\n{'pair':>9}  {'move-change rate (95% CI)':>30}  "
          f"{'JS median (95% CI)':>28}  {'|dV| mean':>10}")
    print("-" * 84)
    for a, b in pairs:
        s = out["overall"][f"{a}_{b}"]
        r, j = s["top_move_disagreement_rate"], s["js_median"]
        v = s["root_value_change_mean_abs"]
        print(f"{a}->{b:<4}  {r['point']:>10.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
              f"      {j['point']:>10.4f} [{j['lo']:.4f}, {j['hi']:.4f}]"
              f"  {v['point']:>10.4f}")

    print("\nby forced-target mini-board (move-change rate):")
    hdr = f"  {'pair':>9}" + "".join(f"{k:>10}" for k in ("open", "won", "drawn", "none"))
    print(hdr)
    for a, b in pairs:
        row = out["by_forced_target"][f"{a}_{b}"]
        cells = "".join(
            (f"{row[k]['top_move_disagreement_rate']['point']:>10.3f}"
             if row[k] else f"{'-':>10}") for k in ("open", "won", "drawn", "none"))
        print(f"  {a}->{b:<4}" + cells)

    print("\ndecisions:")
    for k, v in dec.items():
        if k.startswith(("distillation", "drop_800", "teacher_separation_m")):
            print(f"  {k:<36} {v}")
    print(f"\nwrote summary_analysis.json, diagnostic_top_js.json "
          f"({len(diag)} positions), position_strata.csv.gz")


if __name__ == "__main__":
    main()
