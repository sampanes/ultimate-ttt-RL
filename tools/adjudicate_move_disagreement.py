"""When deeper search changes its mind, is the later move actually better?

RESULT_SEARCH_DISAGREEMENT.md established that every doubling of simulations
changes the chosen move at a roughly constant ~13-14% rate out to 800 sims. That
is a statement about CHURN, not about IMPROVEMENT. A search that reshuffles
among near-equivalent moves produces exactly the same signature. This tool tests
the missing half.

For every position where the 200- and 800-sim searches pick different moves, it
adjudicates the two candidate moves three ways:

  1. VALUE DELTA (primary). Play each candidate, then evaluate the resulting
     child with a deeper search (default 1600 sims). The mover's value is
     -Q(child), since Q is stored from the child's to-move perspective.
     delta = v(800's move) - v(200's move); positive means the deeper look
     prefers what the 800-sim search chose.

  2. DEEP ROOT PREFERENCE. Run the deep adjudicator at the root and see which
     candidate its own argmax lands on, if either.

  3. INDEPENDENT VOTE. Ask gregory -- a completely different engine, depth-
     limited alpha-beta with a hand-written heuristic -- which candidate it
     prefers.

WHY THREE. Signals 1 and 2 use THE SAME NETWORK with more simulations, so they
are partly circular: a 1600-sim search is closer to the 800-sim search than to
the 200-sim one almost by construction, because both are marching toward the
same deep-search limit. They demonstrate CONVERGENCE, which is necessary but not
sufficient for correctness. Signal 3 breaks that circularity with a different
evaluation function entirely -- it is weaker than the teacher (the gen-22 net
scores 0.638 against gregory-d4, i.e. it wins), so treat it as a noisy but
genuinely independent referee rather than ground truth.

Read them together. Value delta positive AND gregory above chance is real
evidence that deeper search improves the move. Value delta positive with
gregory at chance is consistent with mere convergence toward the net's own deep
preference, which may or may not be better play.

    python -m tools.adjudicate_move_disagreement --limit 800
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from agents.gregory import GregoryAgent
from engine.constants import DRAW
from tools.analyze_search_disagreement import (
    PHASE_BANDS, LEGAL_BUCKETS, MCTS, build_model, build_sample, legal_bucket,
    phase_of)
from tools.summarize_search_disagreement import (
    bootstrap_ci, forced_target_state, render_board, tactical_flags)


def child_value_for_mover(state, move, mcts):
    """Value of `move` from the MOVER's perspective, per the deep adjudicator.

    Terminal children are scored exactly rather than searched -- a move that
    ends the game needs no opinion. A move can only win or draw for the mover,
    never lose, so the loss branch is unreachable but kept explicit.
    """
    child = state.clone()
    child.make_move(move)
    if child.is_over():
        if child.winner == DRAW or child.winner is None:
            return 0.0, True
        return (1.0 if child.winner == state.player else -1.0), True
    _pi, root = mcts.search(child)
    # Q at the child is from the CHILD's to-move player (the opponent), so the
    # mover's value is its negation.
    return -float(root.Q()), False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="results/disagreement")
    ap.add_argument("--checkpoint", default="models/expert_iter_v2/teacher.pt")
    ap.add_argument("--corpus", default="models/corpus_gen22")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--max-shards", type=int, default=200)
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--pair", type=int, nargs=2, default=[200, 800],
                    help="The (shallow, deep) sim pair whose disagreements are "
                         "adjudicated.")
    ap.add_argument("--adjudicator-sims", type=int, default=1600)
    ap.add_argument("--gregory-depth", type=int, default=4,
                    help="Independent referee. Deeper is a better judge but "
                         "costs more; d5 is ~4x d4 and no stronger "
                         "(RESULT_RULER_LADDER.md).")
    ap.add_argument("--limit", type=int, default=800,
                    help="Adjudicate at most this many disagreement positions "
                         "(random subsample, seeded). 0 = all.")
    ap.add_argument("--no-root-preference", action="store_true",
                    help="Skip signal 2, saving ~1/3 of the search cost.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    lo, hi = args.pair
    out_dir = args.output or args.input
    rng = np.random.default_rng(args.seed)

    # An adjudicator must be DEEPER than both candidates or it is not
    # adjudicating anything -- judging an 800-sim move with a 400-sim search
    # just asks the shallower opinion to grade the deeper one.
    if args.adjudicator_sims <= max(lo, hi):
        raise SystemExit(
            f"[X] --adjudicator-sims {args.adjudicator_sims} is not deeper "
            f"than the candidates {lo}/{hi}. Use at least {2 * max(lo, hi)}.")

    npz = np.load(os.path.join(args.input, "policies.npz"))
    for s in (lo, hi):
        if f"pi_{s}" not in npz.files:
            raise SystemExit(f"[X] policies.npz has no pi_{s}; run "
                             f"analyze_search_disagreement with --sims {s}")
    pi_lo, pi_hi = npz[f"pi_{lo}"], npz[f"pi_{hi}"]

    sample, _rej, _b = build_sample(args.corpus, args.max_shards,
                                    args.sample_size, args.seed)
    if len(sample) != pi_lo.shape[0]:
        raise SystemExit("[X] sample replay does not match policies.npz")

    disagree = [k for k in range(len(sample))
                if int(np.argmax(pi_lo[k])) != int(np.argmax(pi_hi[k]))]
    print(f"{lo}v{hi} argmax disagreements: {len(disagree):,} / "
          f"{len(sample):,} ({len(disagree)/len(sample):.1%})")
    if args.limit and len(disagree) > args.limit:
        disagree = sorted(rng.choice(disagree, args.limit, replace=False).tolist())
        print(f"adjudicating a seeded subsample of {len(disagree):,}")

    model = build_model(args.checkpoint, args.device)
    adj = MCTS(model, args.device, n_sims=args.adjudicator_sims, c_puct=1.5,
               add_dirichlet_at_root=False,
               wave_size=max(1, args.adjudicator_sims // MCTS._MIN_WAVES))
    greg = GregoryAgent(depth=args.gregory_depth)

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for i, k in enumerate(disagree):
            st, legal, filled = sample[k]
            m_lo = int(np.argmax(pi_lo[k]))
            m_hi = int(np.argmax(pi_hi[k]))
            v_lo, term_lo = child_value_for_mover(st, m_lo, adj)
            v_hi, term_hi = child_value_for_mover(st, m_hi, adj)

            root_pref = None
            if not args.no_root_preference:
                _p, r = adj.search(st.clone())
                best = int(np.argmax(_p))
                root_pref = ("deep" if best == m_hi else
                             "shallow" if best == m_lo else "other")

            gm = int(greg.select_move(st.clone()))
            greg_pref = ("deep" if gm == m_hi else
                         "shallow" if gm == m_lo else "other")

            imm, miniw = tactical_flags(st, legal)
            rows.append({
                "pos_id": k, "phase": phase_of(filled), "n_legal": len(legal),
                "legal_bucket": legal_bucket(len(legal)),
                "forced_target": forced_target_state(st),
                "immediate_win_available": int(imm),
                "mini_win_available": int(miniw),
                "move_shallow": m_lo, "move_deep": m_hi,
                "v_shallow": v_lo, "v_deep": v_hi,
                "delta": v_hi - v_lo,
                "deep_move_better": int(v_hi > v_lo),
                "terminal_child": int(term_lo or term_hi),
                "root_pref": root_pref, "gregory_pref": greg_pref,
            })
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(disagree)}  {el:.0f}s  "
                      f"({el/(i+1):.2f}s/pos)", flush=True)

    n = len(rows)
    delta = np.array([r["delta"] for r in rows])
    better = np.array([r["deep_move_better"] for r in rows], dtype=float)

    def rate(key, val, subset):
        if not subset:
            return None
        return float(np.mean([r[key] == val for r in subset]))

    summary = {
        "pair": [lo, hi], "adjudicator_sims": args.adjudicator_sims,
        "gregory_depth": args.gregory_depth,
        "n_disagreements_total": int(len(sample) * 0 + len(disagree)),
        "n_adjudicated": n,
        "circularity_warning":
            "value_delta and root_preference use the SAME network with more "
            "simulations, so they measure convergence toward that network's "
            "deep-search limit, not correctness. gregory_preference is the "
            "independent referee (different engine, but weaker than the "
            "teacher). Read them together.",
        "value_delta": {
            "deep_move_better_rate": bootstrap_ci(better, np.mean, args.n_boot, rng),
            "mean_delta": bootstrap_ci(delta, np.mean, args.n_boot, rng),
            "median_delta": bootstrap_ci(delta, np.median, args.n_boot, rng),
        },
        "root_preference": {
            p: rate("root_pref", p, rows) for p in ("deep", "shallow", "other")
        } if not args.no_root_preference else None,
        "gregory_preference": {
            p: rate("gregory_pref", p, rows) for p in ("deep", "shallow", "other")
        },
        "by_phase": {}, "by_legal_bucket": {}, "by_forced_target": {},
        "by_tactical": {},
    }

    def strat(subset):
        if not subset:
            return None
        d = np.array([r["delta"] for r in subset])
        b = np.array([r["deep_move_better"] for r in subset], dtype=float)
        return {"n": len(subset),
                "deep_move_better_rate": float(b.mean()),
                "mean_delta": float(d.mean()),
                "gregory_deep": rate("gregory_pref", "deep", subset),
                "gregory_shallow": rate("gregory_pref", "shallow", subset)}

    for ph, _, _ in PHASE_BANDS:
        summary["by_phase"][ph] = strat([r for r in rows if r["phase"] == ph])
    for lb, _, _ in LEGAL_BUCKETS:
        summary["by_legal_bucket"][lb] = strat(
            [r for r in rows if r["legal_bucket"] == lb])
    for ft in ("open", "won", "drawn", "none"):
        summary["by_forced_target"][ft] = strat(
            [r for r in rows if r["forced_target"] == ft])
    summary["by_tactical"] = {
        "immediate_win_available": strat(
            [r for r in rows if r["immediate_win_available"]]),
        "no_immediate_win": strat(
            [r for r in rows if not r["immediate_win_available"]]),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"adjudication_{lo}v{hi}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    import csv
    import gzip
    with gzip.open(os.path.join(out_dir, f"adjudication_{lo}v{hi}.csv.gz"),
                   "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- report -----------------------------------------------------------
    vd = summary["value_delta"]
    r = vd["deep_move_better_rate"]
    m = vd["mean_delta"]
    print(f"\nadjudicated {n:,} positions where {lo} and {hi} sims disagree "
          f"({time.time()-t0:.0f}s)")
    print(f"\n  PRIMARY (same net, {args.adjudicator_sims} sims -- convergence, "
          f"not correctness)")
    print(f"    deeper move is better: {r['point']:.3f} "
          f"[{r['lo']:.3f}, {r['hi']:.3f}]   (0.500 = coin flip)")
    print(f"    mean value delta:      {m['point']:+.4f} "
          f"[{m['lo']:+.4f}, {m['hi']:+.4f}]")
    if summary["root_preference"]:
        rp = summary["root_preference"]
        print(f"    deep adjudicator's own pick: {hi}-move {rp['deep']:.3f} | "
              f"{lo}-move {rp['shallow']:.3f} | neither {rp['other']:.3f}")
    gp = summary["gregory_preference"]
    print(f"\n  INDEPENDENT (gregory d{args.gregory_depth}, different engine)")
    print(f"    prefers {hi}-move {gp['deep']:.3f} | {lo}-move "
          f"{gp['shallow']:.3f} | neither {gp['other']:.3f}")
    denom = gp["deep"] + gp["shallow"]
    if denom > 0:
        print(f"    head-to-head where it picks one of them: "
              f"{gp['deep']/denom:.3f} for the deeper move (0.500 = chance)")
    print(f"\nwrote adjudication_{lo}v{hi}.json / .csv.gz")


if __name__ == "__main__":
    main()
