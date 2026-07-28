"""Did solved-node propagation fix the forced-win dilution it was built for?

RESULT_DISTILL_PILOT.md measured the defect precisely, on these exact positions:

    mate-in-1 (n=1,144)   top-move mass   50 sims 0.8251   800 sims 0.6930
                          nonzero moves   50 sims 3.82     800 sims 11.53 of 11.53

PUCT's exploration bonus scales with sqrt(N_total), so a bigger budget puts more
ABSOLUTE visits on moves already known to lose. The deeper teacher picked the
same move 93% of the time -- it was not wrong, it was diluted, and the student
inherited the dilution.

This re-measures the same quantities with solving on and off, on the SAME frozen
positions, so the before/after numbers are directly comparable to the published
ones rather than to a fresh sample.

Both modes are searched here even though the solve=off targets at 50 and 800
already exist on disk as the pilot corpora. Re-running them buys two things that
reading them back cannot:

  * PARITY. Every solve=off target is compared to the stored bytes, so the guard
    that this change did not perturb the default path is complete rather than
    sampled. If it fails, every before/after comparison below is void.
  * An exact 'network evaluations avoided' figure, as a per-position difference
    between the two modes on identical positions, rather than a number inferred
    from an assumption about what an unsolved simulation would have cost.

Beyond target shape it reports proof COVERAGE (which positions get solved at
all, by tactical class and game phase), proof TIMING (whether a proof lands
before visits have already been spent elsewhere), and RECONCILIATION impact (how
often the returned-policy correction changes the target, and whether the change
was right). Adjudication of the last one is exact: the referee is the proof, and
proofs are checked against exhaustive minimax in agents/test_mcts.py.

Positions are enriched for tactics on purpose: mate-in-1 is only 2.29% of the
corpus, so a uniform subsample would leave far too few to say anything about the
stratum where the whole effect lives. ALL mate-in-1 positions are taken plus a
random draw of the rest, and overall figures are reweighted back to the corpus's
natural rate. Per-stratum numbers are the primary reading.

    python -m tools.measure_solved_targets --nontactical 4000
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import time

import numpy as np
import torch

from engine.rules import rule_utl_valid_moves
from tools import provenance
from tools.analyze_search_disagreement import (
    MCTS, build_model, js_divergence, legal_bucket, phase_of, state_from_planes)

# Coverage strata. The pilot's index only flags mate-in-1; 'other_tactical' is
# recomputed here (see tactical_class) so coverage can distinguish a narrow
# forced-win correction from a generally active search mechanism.
STRATA = ("mate_in_1", "other_tactical", "non_tactical")

# When a proof lands, as an ordered category rather than a raw count.
PROOF_BUCKETS = ("root_expansion", "simulation_1_to_10", "simulation_11_to_50",
                 "simulation_51_plus", "unsolved")


def proof_bucket(proof_sim):
    """Bucket a proof by WHEN it arrived. Expansion is its own category.

    'root_expansion' is not "simulation 0". Expansion runs an exact one-ply
    probe over every legal move, so proofs found there are produced by
    PREPROCESSING, before a single simulation has been spent -- no visits can
    have been misallocated yet, and no neural evaluation was needed to find
    them. Folding those into a numeric sim-0 bin hides the mechanism, making a
    cheap deterministic check look like a fast search result.

    Sim indices >= 1 are wave-granularity upper bounds (see the granularity
    note emitted alongside): the wave path can only record a proof at a wave
    boundary, so a proof credited to sim k arrived somewhere in that wave.
    """
    if proof_sim is None:
        return "unsolved"
    if proof_sim == 0:
        return "root_expansion"
    if proof_sim <= 10:
        return "simulation_1_to_10"
    if proof_sim <= 50:
        return "simulation_11_to_50"
    return "simulation_51_plus"


def load_pilot(pilot, arm):
    """Concatenated (x, pi) for one arm, in shard order == index.npz row order."""
    shards = sorted(glob.glob(os.path.join(pilot, f"sims{arm}", "data",
                                           "shard_*.pt")))
    if not shards:
        raise SystemExit(f"[X] no shards for arm {arm} under {pilot}")
    xs, pis = [], []
    for p in shards:
        d = torch.load(p, map_location="cpu", weights_only=False)
        xs.append(d["x"])
        pis.append(d["pi"])
    return torch.cat(xs), torch.cat(pis)


def winning_moves(state, legal):
    """The immediately game-winning moves, via the engine (no rule duplication)."""
    mover = state.player
    out = []
    for m in legal:
        probe = state.clone()
        probe.make_move(m)
        if probe.winner == mover:
            out.append(m)
    return out


def backfill_decision(summary, deep_sims=800):
    """Should the expensive deep arm be rerun just to recover proof-timing buckets?

    Written BEFORE the coverage numbers exist, for the same reason the estimator
    was: a rerun costing ~55 minutes is exactly the kind of call that gets
    rationalised after the fact in whichever direction is convenient.

    The owner's rule, transcribed. Backfill 50 and 200 (cheap, ~17 min combined)
    if non-expansion proofs are materially present at all. Rerun the deep arm
    ONLY if timing resolution could actually change the interpretation, i.e.
    either:

      (a) more than 10% of deep-arm proofs land AFTER root expansion, or
      (b) late proofs account for a meaningful fraction of proof-corrected
          target changes.

    (b) has no number attached in the instruction, so one is fixed here rather
    than left to judgement at reading time: LATE_SHARE_OF_CHANGES = 0.10, the
    same bar as (a). Reported explicitly as a chosen threshold, not a measured
    one, so it can be argued with in advance instead of after.

    Note what is NOT the trigger: mere absence of the middle buckets. Those are
    secondary instrumentation. `visits_off_proven_at_proof` already measures
    pre-proof target distortion DIRECTLY, rather than using arrival time as a
    proxy for it, so the interpretation usually does not depend on the split.
    """
    POST_EXPANSION_SHARE = 0.10
    LATE_SHARE_OF_CHANGES = 0.10

    timing = (summary.get("proof_timing") or {}).get(str(deep_sims)) or {}
    recon = (summary.get("reconciliation") or {}).get(str(deep_sims)) or {}
    n_solved = timing.get("n_solved") or 0

    # root_expansion and unsolved survive in the old schema via
    # proof_at_sim_0_rate, so (a) is answerable even without the buckets.
    # That field is computed over the SOLVED records only -- `(psim == 0).mean()`
    # where psim comes from `sr`, not from every searched root -- so it is
    # already the share of proofs from expansion and needs no rescaling. An
    # earlier version of this function multiplied it by n_searched/n_solved and
    # produced a NEGATIVE post-expansion share, which would have suppressed
    # both the rerun and the cheap backfill. The 10% thresholds below are
    # untouched; only the denominator was wrong.
    at0 = timing.get("proof_at_sim_0_rate")
    share = timing.get("share_of_proofs_from_expansion")
    if share is None:
        share = at0
    post = (1.0 - share) if share is not None else None

    out = {
        "deep_sims": deep_sims,
        "thresholds": {"post_expansion_share": POST_EXPANSION_SHARE,
                       "late_share_of_corrected_changes": LATE_SHARE_OF_CHANGES,
                       "second_threshold_is_a_choice": (
                           "the instruction said 'a meaningful fraction' "
                           "without a number; 0.10 fixed here in advance")},
        "measured": {"n_solved": n_solved,
                     "share_from_expansion": share,
                     "share_after_expansion": post,
                     "n_proof_corrected_changes": recon.get("n_changed")},
        "buckets_present": bool(timing.get("when_proved")),
    }

    if post is None:
        out["verdict"] = "UNDECIDABLE -- no proof timing recorded for this arm"
        out["backfill_cheap_arms"] = True
        out["rerun_deep_arm"] = False
        return out

    trigger_a = post > POST_EXPANSION_SHARE
    # (b) is only computable once the buckets exist, which is the point of the
    # cheap backfill: run 50 and 200 first, then re-ask.
    late = None
    if timing.get("when_proved") and recon.get("n_changed"):
        w = timing["when_proved"]
        late_n = w.get("simulation_11_to_50", 0) + w.get("simulation_51_plus", 0)
        late = late_n / recon["n_changed"]
    out["measured"]["late_proofs_per_corrected_change"] = late
    trigger_b = late is not None and late > LATE_SHARE_OF_CHANGES

    out["trigger_a_post_expansion_exceeds_10pct"] = trigger_a
    out["trigger_b_late_proofs_material"] = trigger_b
    out["backfill_cheap_arms"] = bool(post > 0.0)
    out["rerun_deep_arm"] = bool(trigger_a or trigger_b)
    out["verdict"] = (
        f"rerun the {deep_sims} arm: " +
        ", ".join([t for t, ok in
                   (("post-expansion proofs exceed 10%", trigger_a),
                    ("late proofs are a material share of corrections",
                     trigger_b)) if ok])
        if out["rerun_deep_arm"] else
        f"do NOT rerun the {deep_sims} arm -- {post:.1%} of proofs land after "
        f"root expansion, below the 10% bar, and timing resolution cannot "
        f"change the interpretation")
    return out


def tactical_class(state, legal):
    """mate_in_1 / other_tactical / non_tactical, plus the winning-move set.

    'other_tactical' means the position is sharp without being won outright:
    either some legal move hands the opponent an immediate win (we can blunder),
    or some legal move claims a mini-board. Both are decided by playing the move
    on a clone, so the mini-board test compares mini_winners before and after
    rather than doing index arithmetic at all. That is why this file was never
    touched by the mini-board indexing bug that invalidated the summarizer's
    mini_win stratum (see ERRATA_MINI_INDEX_BUG.md) -- there is no index to get
    wrong. Keep it that way.
    """
    mover = state.player
    wins, mini, blunder = [], False, False
    for m in legal:
        probe = state.clone()
        probe.make_move(m)
        if probe.winner == mover:
            wins.append(m)
            continue
        if probe.winner is None:
            if any(a != b for a, b in zip(probe.mini_winners,
                                          state.mini_winners)):
                mini = True
            if not blunder:
                reply = rule_utl_valid_moves(probe.board, probe.last_move,
                                             probe.mini_winners)
                opp = probe.player
                for r in reply:
                    p2 = probe.clone()
                    p2.make_move(r)
                    if p2.winner == opp:
                        blunder = True
                        break
    if wins:
        return "mate_in_1", wins
    if mini or blunder:
        return "other_tactical", []
    return "non_tactical", []


def entropy_bits(p, legal):
    q = np.asarray(p, dtype=np.float64)[legal]
    s = q.sum()
    if s <= 0:
        return float("nan")
    q = q / s
    q = q[q > 0]
    return float(-np.sum(q * np.log2(q)))


def policy_stats(pi_rows, positions, win_sets):
    """Target-shape metrics for one (sims, mode) arm over one stratum."""
    if not positions:
        return None
    top, ent, nz, wmass, wacc = [], [], [], [], []
    for k, (_st, legal, _f) in positions:
        p = pi_rows[k]
        top.append(float(p[legal].max()))
        ent.append(entropy_bits(p, legal))
        nz.append(int((p[legal] > 0).sum()))
        wins = win_sets.get(k)
        if wins:
            wmass.append(float(sum(p[m] for m in wins)))
            wacc.append(float(int(np.argmax(p)) in wins))
    out = {
        "n": len(positions),
        "top_move_mass": float(np.mean(top)),
        "policy_entropy_bits": float(np.nanmean(ent)),
        "nonzero_moves": float(np.mean(nz)),
    }
    if wmass:
        out["winning_move_mass"] = float(np.mean(wmass))
        out["winning_move_argmax_accuracy"] = float(np.mean(wacc))
    return out


def pair_stats(pi_a, pi_b, positions):
    if not positions:
        return None
    changed, js = [], []
    for k, (_st, legal, _f) in positions:
        changed.append(float(np.argmax(pi_a[k]) != np.argmax(pi_b[k])))
        js.append(js_divergence(pi_a[k], pi_b[k], legal))
    js = np.array(js, dtype=np.float64)
    js = js[~np.isnan(js)]
    return {
        "n": len(positions),
        "top_move_disagreement": float(np.mean(changed)),
        "js_mean": float(js.mean()) if js.size else None,
        "js_median": float(np.median(js)) if js.size else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", default="models/distill_pilot")
    ap.add_argument("--checkpoint", default="models/expert_iter_v2/teacher.pt")
    ap.add_argument("--sims", type=int, nargs="+", default=[50, 200, 800])
    ap.add_argument("--nontactical", type=int, default=4000,
                    help="random non-tactical positions to add to ALL the "
                         "mate-in-1 ones")
    ap.add_argument("--max-tactical", type=int, default=0,
                    help="cap the tactical positions too (0 = all of them). "
                         "For smoke tests; the real run wants all 1,144.")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--output", default="results/solved_targets")
    ap.add_argument("--reuse-policies", action="store_true",
                    help="load <output>/policies.npz and re-run only the "
                         "analysis. No search, no GPU, seconds not hours.")
    ap.add_argument("--small-gap", type=float, default=0.05,
                    help="|q_root_800 - q_root_50| below this counts as a "
                         "near-equivalent swap")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--decide-backfill", metavar="SUMMARY_JSON",
                    help="read a landed summary.json and print the "
                         "pre-registered proof-timing backfill decision; runs "
                         "no search and exits")
    args = ap.parse_args()

    if args.decide_backfill:
        with open(args.decide_backfill, encoding="utf-8") as fh:
            d = backfill_decision(json.load(fh), deep_sims=max(args.sims))
        print(json.dumps(d, indent=2))
        print(f"\ncheap backfill (50, 200): {d['backfill_cheap_arms']}")
        print(f"rerun deep arm          : {d['rerun_deep_arm']}")
        print(f"verdict: {d['verdict']}")
        return

    os.makedirs(args.output, exist_ok=True)
    idx = np.load(os.path.join(args.pilot, "index.npz"), allow_pickle=True)
    tactical = idx["immediate_win"].astype(bool)
    n_all = tactical.shape[0]

    # The stored arms double as the solve=off ground truth for 50 and 800.
    stored = {}
    x_planes = None
    for s in (50, 800):
        if s in args.sims and os.path.isdir(os.path.join(args.pilot, f"sims{s}")):
            x_planes, pi = load_pilot(args.pilot, s)
            stored[s] = pi.numpy()
    if x_planes is None:
        raise SystemExit("[X] need at least one stored arm to recover the planes")
    if x_planes.shape[0] != n_all:
        raise SystemExit(f"[X] index has {n_all} rows, corpus has {x_planes.shape[0]}")

    # ---- subset: every tactical position, plus a random draw of the rest ----
    rng = random.Random(args.seed)
    tact_ids = [i for i in range(n_all) if tactical[i]]
    rest_ids = [i for i in range(n_all) if not tactical[i]]
    rng.shuffle(rest_ids)
    if args.max_tactical:
        rng.shuffle(tact_ids)
        tact_ids = sorted(tact_ids[:args.max_tactical])
    keep = sorted(tact_ids + rest_ids[:args.nontactical])
    natural_tactical_rate = float(tactical.mean())
    print(f"corpus {n_all:,} positions, mate-in-1 rate {natural_tactical_rate:.4%}")
    print(f"subset {len(keep):,} = {len(tact_ids):,} tactical "
          f"+ {min(args.nontactical, len(rest_ids)):,} non-tactical")

    positions = []       # (row_index_into_subset_arrays, (state, legal, filled))
    dropped = 0
    for j, i in enumerate(keep):
        rec = state_from_planes(x_planes[i])
        if rec is None:
            dropped += 1
            continue
        positions.append((j, rec))
    if dropped:
        print(f"[!] {dropped} positions failed reconstruction and were dropped")

    # Tactical class and winning-move sets, computed once; arm-independent.
    # The pilot's index only flags mate-in-1, so 'other_tactical' is derived
    # here rather than read from it.
    win_sets, tclass = {}, {}
    t0 = time.time()
    for j, (st, legal, _f) in positions:
        cls, wins = tactical_class(st, legal)
        tclass[j] = cls
        if wins:
            win_sets[j] = wins
    counts = {c: sum(1 for v in tclass.values() if v == c) for c in STRATA}
    print(f"tactical classes in {time.time() - t0:.0f}s: "
          + "  ".join(f"{k}={v:,}" for k, v in counts.items()))
    flagged = int(sum(1 for j, _ in positions if tactical[keep[j]]))
    if len(win_sets) != flagged:
        print(f"[!] {len(win_sets)} positions have a winning move but the "
              f"pilot index flags {flagged}; using the recomputed set")

    arms, costs = {}, {}
    cache_path = os.path.join(args.output, "policies.npz")
    if args.reuse_policies:
        if not os.path.isfile(cache_path):
            raise SystemExit(f"[X] --reuse-policies but no {cache_path}")
        cached = np.load(cache_path)
        want = np.array([keep[j] for j, _ in positions])
        if not np.array_equal(cached["rows"], want):
            raise SystemExit(
                "[X] cached policies cover different positions than this "
                "invocation samples. Match --nontactical / --max-tactical / "
                "--seed to the run that produced them.")
        arms = {k[len("pi_"):]: cached[k] for k in cached.files
                if k.startswith("pi_")}
        costs = {k: {"reused_from_cache": True} for k in arms}
        prior = os.path.join(args.output, "summary.json")
        if os.path.isfile(prior):
            with open(prior, encoding="utf-8") as fh:
                costs.update({k: v for k, v in json.load(fh)
                              .get("cost", {}).items() if k in arms})
        print(f"reusing cached arms {sorted(arms)} -- analysis only, no search")

    per_position = {}
    parity = {}
    if not args.reuse_policies:
        model = build_model(args.checkpoint, args.device)

        # Warm the CUDA context and cuDNN autotuner before anything is timed --
        # otherwise whichever arm runs first eats the startup cost and the
        # solve-on/solve-off comparison picks up an artefact, not a signal.
        warm = MCTS(model, args.device, n_sims=800, c_puct=1.5,
                    add_dirichlet_at_root=False,
                    wave_size=max(1, 800 // MCTS._MIN_WAVES), solve=False)
        for _j, (st, _legal, _f) in positions[:5]:
            warm.search(st.clone())

    # ---- search each (sims, mode) ------------------------------------------
    # The solve=off arms at 50 and 800 exist on disk already, and running them
    # again is not free. It is worth it twice over: it makes 'neural evaluations
    # avoided' an exact per-position difference instead of an assumption, and it
    # turns the parity guard from a 200-position sample into complete
    # verification that solve=False still reproduces the frozen pilot corpus
    # byte for byte. The stored arms become the thing checked AGAINST, not a
    # shortcut around the work.
    for mode in ("off", "on") if not args.reuse_policies else ():
        for s in args.sims:
            key = f"{s}_{mode}"
            eff = max(1, s // MCTS._MIN_WAVES)
            mcts = MCTS(model, args.device, n_sims=s, c_puct=1.5,
                        add_dirichlet_at_root=False, wave_size=eff,
                        solve=(mode == "on"))
            rows = np.zeros((len(positions), 81), dtype=np.float32)
            recs = []
            t0 = time.time()
            for r, (_j, (st, _legal, _f)) in enumerate(positions):
                before = (mcts.stat_nn_evals, mcts.stat_expansions)
                pi, _root = mcts.search(st.clone())
                rows[r] = pi
                if mode == "on":
                    recs.append(dict(mcts.last))
                else:
                    # solve=off keeps no proof record, but the per-position net
                    # cost is what makes 'evaluations avoided' measurable rather
                    # than assumed, so capture it directly.
                    recs.append({
                        "nn_evals": mcts.stat_nn_evals - before[0],
                        "expansions": mcts.stat_expansions - before[1],
                        "raw_argmax": int(pi.argmax()),
                    })
            dt = time.time() - t0
            arms[key] = rows
            per_position[key] = recs

            if mode == "off" and s in stored:
                ref = stored[s][[keep[j] for j, _ in positions]]
                bad = int(np.sum(~np.all(rows == ref, axis=1)))
                if bad:
                    raise SystemExit(
                        f"[X] PARITY FAILED at {s} sims: {bad}/{len(rows)} "
                        f"solve=False targets differ from the frozen pilot "
                        f"corpus. Solving perturbed the default path -- every "
                        f"comparison in this run would be invalid.")
                parity[s] = len(rows)
                print(f"  [OK] parity: all {len(rows):,} solve=off targets at "
                      f"{s} sims are bit-identical to the frozen pilot corpus")

            n = max(1, mcts.stat_searches)
            costs[key] = {
                "moves": mcts.stat_searches,
                "expanded_nodes_per_move": mcts.stat_expansions / n,
                "nn_evals_per_move": mcts.stat_nn_evals / n,
                "nn_batches_per_move": mcts.stat_nn_batches / n,
                "terminal_probes_per_move": mcts.stat_probes / n,
                "solved_root_rate": mcts.stat_solved_roots / n,
                "seconds_per_move": mcts.stat_seconds / n,
                "wall_seconds": dt,
            }
            print(f"  {key:>10}  {dt:>7.0f}s  "
                  f"{costs[key]['seconds_per_move'] * 1000:6.1f} ms/move  "
                  f"{costs[key]['expanded_nodes_per_move']:7.1f} expanded  "
                  f"{costs[key]['nn_evals_per_move']:7.1f} nn-evals  "
                  f"solved-root {costs[key]['solved_root_rate']:.3f}", flush=True)

    # Rows are indexed by position order within the subset.
    row_class = [tclass[j] for j, _ in positions]
    row_phase = [str(idx["phase"][keep[j]]) for j, _ in positions]
    phases = sorted(set(row_phase))
    by_row = [(r, rec) for r, (_j, rec) in enumerate(positions)]
    row_tactical = [(r, rec) for r, (j, rec) in enumerate(positions)
                    if tactical[keep[j]]]
    row_other = [(r, rec) for r, (j, rec) in enumerate(positions)
                 if not tactical[keep[j]]]
    win_by_row = {r: win_sets[j] for r, (j, _rec) in enumerate(positions)
                  if j in win_sets}

    out = {
        "pilot": args.pilot, "checkpoint": args.checkpoint,
        "sims": args.sims, "seed": args.seed,
        "corpus_positions": int(n_all),
        "natural_tactical_rate": natural_tactical_rate,
        "subset": {"total": len(positions), "tactical": len(row_tactical),
                   "non_tactical": len(row_other)},
        "note": ("Subset is ENRICHED for tactics; per-stratum numbers are "
                 "primary. 'overall_reweighted' mixes the two strata at the "
                 "corpus's natural mate-in-1 rate."),
        "cost": costs,
        "targets": {}, "pairs": {},
    }

    for key, rows in arms.items():
        t = policy_stats(rows, row_tactical, win_by_row)
        o = policy_stats(rows, row_other, win_by_row)
        rw = {}
        if t and o:
            w = natural_tactical_rate
            for m in ("top_move_mass", "policy_entropy_bits", "nonzero_moves"):
                rw[m] = w * t[m] + (1 - w) * o[m]
        out["targets"][key] = {"tactical": t, "non_tactical": o,
                               "overall_reweighted": rw}

    sims = sorted(args.sims)
    for mode in ("off", "on"):
        for a, b in list(zip(sims, sims[1:])) + [(sims[0], sims[-1])]:
            ka, kb = f"{a}_{mode}", f"{b}_{mode}"
            if ka not in arms or kb not in arms:
                continue
            out["pairs"][f"{a}_{b}_{mode}"] = {
                "all": pair_stats(arms[ka], arms[kb], by_row),
                "tactical": pair_stats(arms[ka], arms[kb], row_tactical),
                "non_tactical": pair_stats(arms[ka], arms[kb], row_other),
            }

    # ---- (1) proof coverage, (2) proof timing, (3) reconciliation impact -----
    # All three read the per-search records, so they are only available for the
    # arms actually searched in this run.
    if per_position:
        out["parity_positions_verified"] = parity
        out["proof_coverage"] = {}
        out["proof_timing"] = {}
        out["reconciliation"] = {}

        for s in args.sims:
            key = f"{s}_on"
            recs = per_position.get(key)
            if not recs:
                continue
            solved = np.array([r["root_solved"] is not None for r in recs])

            cov = {"overall": float(solved.mean()), "n": int(len(recs))}
            for c in STRATA:
                m = np.array([rc == c for rc in row_class])
                cov[c] = {"n": int(m.sum()),
                          "solved_rate": float(solved[m].mean()) if m.any() else None}
            cov["by_phase"] = {}
            for ph in phases:
                m = np.array([str(p) == ph for p in row_phase])
                cov["by_phase"][ph] = {
                    "n": int(m.sum()),
                    "solved_rate": float(solved[m].mean()) if m.any() else None}
            # Which way the proofs go -- a search that only ever proves its own
            # losses is a very different mechanism from one that finds wins.
            outcomes = [r["root_solved"] for r in recs if r["root_solved"] is not None]
            cov["proven_win"] = int(sum(1 for v in outcomes if v == 1))
            cov["proven_draw"] = int(sum(1 for v in outcomes if v == 0))
            cov["proven_loss"] = int(sum(1 for v in outcomes if v == -1))
            out["proof_coverage"][str(s)] = cov

            sr = [r for r in recs if r["root_solved"] is not None]
            if sr:
                psim = np.array([r["proof_sim"] for r in sr], dtype=np.float64)
                off = np.array([r["visits_off_proven_at_proof"] for r in sr
                                if r["visits_off_proven_at_proof"] is not None],
                               dtype=np.float64)
                refu = np.array([r["visits_on_refuted_at_proof"] for r in sr],
                                dtype=np.float64)
                off_recs = per_position.get(f"{s}_off")
                avoided = None
                if off_recs:
                    avoided = np.array(
                        [off_recs[i]["nn_evals"] - recs[i]["nn_evals"]
                         for i, r in enumerate(recs)
                         if r["root_solved"] is not None], dtype=np.float64)
                # WHEN the proof arrived, as explicit categories. Denominator
                # is every searched root, so the buckets and 'unsolved' sum to
                # 1 and coverage is readable straight off this table.
                buckets = {b: 0 for b in PROOF_BUCKETS}
                for r in recs:
                    buckets[proof_bucket(
                        r["proof_sim"] if r["root_solved"] is not None
                        else None)] += 1
                n_all = max(1, len(recs))

                out["proof_timing"][str(s)] = {
                    "n_searched": len(recs),
                    "n_solved": len(sr),
                    "when_proved": buckets,
                    "when_proved_rate": {b: buckets[b] / n_all
                                         for b in PROOF_BUCKETS},
                    "proved_before_any_simulation":
                        buckets["root_expansion"] / n_all,
                    "share_of_proofs_from_expansion":
                        (buckets["root_expansion"] / len(sr)) if sr else None,
                    "bucket_semantics":
                        ("root_expansion = found by the exact one-ply probe "
                         "during expansion, before simulation 1 and with zero "
                         "neural evaluations spent on it; it is preprocessing, "
                         "NOT a sim-0 search result"),
                    "proof_sim_mean": float(psim.mean()),
                    "proof_sim_median": float(np.median(psim)),
                    "proof_at_sim_0_rate": float((psim == 0).mean()),
                    "proof_within_10pct_of_budget": float((psim <= 0.1 * s).mean()),
                    "visits_off_proven_at_proof_mean":
                        float(off.mean()) if off.size else None,
                    "visits_off_proven_at_proof_median":
                        float(np.median(off)) if off.size else None,
                    "visits_off_proven_zero_rate":
                        float((off == 0).mean()) if off.size else None,
                    "visits_on_refuted_at_proof_mean": float(refu.mean()),
                    "visits_on_refuted_final_mean": float(np.mean(
                        [r["visits_on_refuted_final"] for r in sr])),
                    # Exact, not modelled: the same positions were searched with
                    # solving off, so this is a measured difference.
                    "nn_evals_avoided_mean":
                        float(avoided.mean()) if avoided is not None else None,
                    "nn_evals_on_mean": float(np.mean([r["nn_evals"] for r in sr])),
                    "proof_sim_granularity_note":
                        f"wave path records proofs at wave boundaries "
                        f"(eff_wave={max(1, s // MCTS._MIN_WAVES)}), so "
                        f"proof_sim is an upper bound except for sim 0, which "
                        f"is exact",
                }

            changed_idx = [i for i, r in enumerate(recs) if r["reconciled"]]
            changed = [recs[i] for i in changed_idx]
            # Adjudication here is EXACT, not heuristic. The referee is the proof
            # itself, and the proofs are checked against exhaustive minimax in
            # agents/test_mcts.py. This is a different thing from
            # tools/adjudicate_move_disagreement.py, which failed because it
            # needed an oracle stronger than the thing under test.
            def _tally(pred):
                return int(sum(1 for r in changed if pred(r)))
            out["reconciliation"][str(s)] = {
                "n": len(recs),
                "n_changed": len(changed),
                "changed_rate": len(changed) / max(1, len(recs)),
                "changed_by_class": {
                    c: int(sum(1 for i in changed_idx if row_class[i] == c))
                    for c in STRATA},
                "adjudicated": {
                    "raw_was_proven_loss": _tally(
                        lambda r: r["raw_argmax_solved"] == 1),
                    "corrected_is_proven_win": _tally(
                        lambda r: r["corrected_argmax_solved"] == -1),
                    "raw_loss_to_corrected_win": _tally(
                        lambda r: r["raw_argmax_solved"] == 1
                        and r["corrected_argmax_solved"] == -1),
                    "strict_improvement": _tally(
                        lambda r: r["corrected_argmax_solved"] == -1
                        or r["raw_argmax_solved"] == 1),
                    "neither_side_proven": _tally(
                        lambda r: r["raw_argmax_solved"] is None
                        and r["corrected_argmax_solved"] is None),
                },
            }

    # ---- the pilot's two remaining named failure cases ----------------------
    # Both are defined on the solve=OFF arms, because that is where they were
    # observed. The point is to look at what solving does to those SAME rows.
    lo_s, hi_s = sims[0], sims[-1]
    ka, kb = f"{lo_s}_off", f"{hi_s}_off"
    if ka in arms and kb in arms:
        corpus_rows = [keep[j] for j, _ in positions]
        vgap = np.abs(idx[f"q_root_{hi_s}"][corpus_rows]
                      - idx[f"q_root_{lo_s}"][corpus_rows]) \
            if f"q_root_{hi_s}" in idx.files and f"q_root_{lo_s}" in idx.files \
            else None

        same_but_diffuse, small_gap_swap = [], []
        for r, rec in by_row:
            legal = rec[1]
            a_top = float(arms[ka][r][legal].max())
            b_top = float(arms[kb][r][legal].max())
            same = np.argmax(arms[ka][r]) == np.argmax(arms[kb][r])
            if same and b_top < a_top:
                same_but_diffuse.append((r, rec))
            if (not same) and vgap is not None and vgap[r] < args.small_gap:
                small_gap_swap.append((r, rec))

        out["failure_cases"] = {
            "definition": {
                "same_move_but_deeper_is_more_diffuse":
                    f"argmax({lo_s})==argmax({hi_s}) and top-move mass at "
                    f"{hi_s} is LOWER -- the deeper search agrees but hedges",
                "top_move_swap_at_small_value_gap":
                    f"argmax differs and |q_root_{hi_s} - q_root_{lo_s}| < "
                    f"{args.small_gap} -- churn among near-equivalent moves",
            },
            "counts": {"same_move_but_deeper_is_more_diffuse": len(same_but_diffuse),
                       "top_move_swap_at_small_value_gap": len(small_gap_swap)},
            "targets": {},
        }
        for name, subset in (("same_move_but_deeper_is_more_diffuse", same_but_diffuse),
                             ("top_move_swap_at_small_value_gap", small_gap_swap)):
            out["failure_cases"]["targets"][name] = {
                key: policy_stats(rows_, subset, win_by_row)
                for key, rows_ in arms.items()}

    # Persist the raw per-search records. The aggregates above answer the
    # questions asked today; this file is what makes a question asked TOMORROW
    # answerable without re-running the search. Skipping it once already cost a
    # full re-run when the proof-timing buckets were requested after the fact.
    if per_position:
        rec_path = os.path.join(args.output, "records.json.gz")
        with gzip.open(rec_path, "wt", encoding="utf-8") as fh:
            json.dump({"row_class": list(row_class),
                       "row_phase": [str(p) for p in row_phase],
                       "arms": {k: v for k, v in per_position.items()}},
                      fh)
        out["records_file"] = rec_path

    # The policy cache must land BEFORE provenance runs: provenance re-derives
    # the parity count by reloading this file, deliberately not trusting the
    # inline check that produced the log line. Build the block second, then
    # write the summary last so it carries the verdict.
    if not args.reuse_policies:
        np.savez_compressed(cache_path,
                            keep=np.array(keep),
                            rows=np.array([keep[j] for j, _ in positions]),
                            **{f"pi_{k}": v for k, v in arms.items()})

    out["provenance"] = provenance.build(sims=min(args.sims),
                                         measure_out=args.output)

    with open(os.path.join(args.output, "summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # ---- report -------------------------------------------------------------
    print("\nTARGET SHAPE -- mate-in-1 positions "
          f"(n={len(row_tactical)}); the pilot measured 0.8251 / 0.6930 here")
    print(f"{'arm':>10} {'top-move mass':>14} {'entropy bits':>13} "
          f"{'nonzero':>8} {'win mass':>9} {'win argmax':>11}")
    print("-" * 70)
    for key in sorted(arms, key=lambda k: (k.split('_')[1], int(k.split('_')[0]))):
        t = out["targets"][key]["tactical"]
        if not t:
            continue
        print(f"{key:>10} {t['top_move_mass']:>14.4f} "
              f"{t['policy_entropy_bits']:>13.4f} {t['nonzero_moves']:>8.2f} "
              f"{t.get('winning_move_mass', float('nan')):>9.4f} "
              f"{t.get('winning_move_argmax_accuracy', float('nan')):>11.4f}")

    print(f"\nTARGET SHAPE -- non-tactical (n={len(row_other)}); "
          f"must NOT regress")
    print(f"{'arm':>10} {'top-move mass':>14} {'entropy bits':>13} {'nonzero':>8}")
    print("-" * 48)
    for key in sorted(arms, key=lambda k: (k.split('_')[1], int(k.split('_')[0]))):
        o = out["targets"][key]["non_tactical"]
        if not o:
            continue
        print(f"{key:>10} {o['top_move_mass']:>14.4f} "
              f"{o['policy_entropy_bits']:>13.4f} {o['nonzero_moves']:>8.2f}")

    print("\nPER-DOUBLING CHURN (all subset positions)")
    print(f"{'pair':>16} {'top-move disagree':>18} {'JS mean':>9} {'JS median':>10}")
    print("-" * 56)
    for k in out["pairs"]:
        a = out["pairs"][k]["all"]
        print(f"{k:>16} {a['top_move_disagreement']:>18.4f} "
              f"{a['js_mean']:>9.4f} {a['js_median']:>10.4f}")

    print("\nSEARCH COST per move -- on the ENRICHED subset, so solving looks "
          "better here\nthan it would on natural play; the ladder run is the "
          "authority for that.")
    print(f"{'arm':>10} {'expanded':>10} {'nn-evals':>10} {'probes':>9} "
          f"{'ms/move':>9} {'solved root':>12}")
    print("-" * 64)
    for key in sorted(costs, key=lambda k: (k.split('_')[1], int(k.split('_')[0]))):
        c = costs[key]
        if "seconds_per_move" not in c:
            continue
        print(f"{key:>10} {c['expanded_nodes_per_move']:>10.1f} "
              f"{c['nn_evals_per_move']:>10.1f} "
              f"{c['terminal_probes_per_move']:>9.1f} "
              f"{c['seconds_per_move'] * 1000:>9.1f} "
              f"{c['solved_root_rate']:>12.3f}")

    fc = out.get("failure_cases")
    if fc:
        for name, n_sub in fc["counts"].items():
            print(f"\nFAILURE CASE: {name}  (n={n_sub})")
            print(f"{'arm':>10} {'top-move mass':>14} {'entropy bits':>13} "
                  f"{'nonzero':>8}")
            print("-" * 48)
            for key in sorted(arms, key=lambda k: (k.split('_')[1],
                                                   int(k.split('_')[0]))):
                t = fc["targets"][name].get(key)
                if not t:
                    continue
                print(f"{key:>10} {t['top_move_mass']:>14.4f} "
                      f"{t['policy_entropy_bits']:>13.4f} "
                      f"{t['nonzero_moves']:>8.2f}")

    if out.get("proof_coverage"):
        print("\n(1) PROOF COVERAGE -- fraction of ROOTS proven")
        hdr = (f"{'sims':>6} {'overall':>9} " +
               " ".join(f"{c:>15}" for c in STRATA) +
               "   " + " ".join(f"{p:>7}" for p in phases))
        print(hdr)
        print("-" * len(hdr))
        for s in args.sims:
            c = out["proof_coverage"].get(str(s))
            if not c:
                continue
            cells = " ".join(
                (f"{c[k]['solved_rate']:>15.4f}" if c[k]["solved_rate"] is not None
                 else f"{'-':>15}") for k in STRATA)
            ph = " ".join(
                (f"{c['by_phase'][p]['solved_rate']:>7.3f}"
                 if c["by_phase"][p]["solved_rate"] is not None else f"{'-':>7}")
                for p in phases)
            print(f"{s:>6} {c['overall']:>9.4f} {cells}   {ph}")
            print(f"       of which  win {c['proven_win']:,}  "
                  f"draw {c['proven_draw']:,}  loss {c['proven_loss']:,}")

    if out.get("proof_timing"):
        print("\n(2) PROOF TIMING -- do proofs land before the target is distorted?")
        print(f"{'sims':>6} {'n':>7} {'proof sim':>10} {'@sim 0':>8} "
              f"{'visits off proven':>18} {'zero':>7} {'nn avoided':>11}")
        print("-" * 72)
        for s in args.sims:
            t = out["proof_timing"].get(str(s))
            if not t:
                continue
            av = t["nn_evals_avoided_mean"]
            print(f"{s:>6} {t['n_solved']:>7,} "
                  f"{t['proof_sim_median']:>10.1f} "
                  f"{t['proof_at_sim_0_rate']:>8.3f} "
                  f"{(t['visits_off_proven_at_proof_mean'] or 0):>18.2f} "
                  f"{(t['visits_off_proven_zero_rate'] or 0):>7.3f} "
                  f"{(av if av is not None else float('nan')):>11.1f}")

        print("\n    WHEN the proof arrived (share of all searched roots).")
        print("    root_expansion = exact one-ply probe, before simulation 1 --")
        print("    preprocessing, not a fast search result.")
        hdr = " ".join(f"{b:>19}" for b in PROOF_BUCKETS)
        print(f"{'sims':>6} {hdr}")
        print("-" * (7 + 20 * len(PROOF_BUCKETS)))
        for s in args.sims:
            t = out["proof_timing"].get(str(s))
            if not t or "when_proved_rate" not in t:
                continue
            row = " ".join(f"{t['when_proved_rate'][b]:>19.4f}"
                           for b in PROOF_BUCKETS)
            print(f"{s:>6} {row}")

    if out.get("reconciliation"):
        print("\n(3) RECONCILIATION -- raw visit argmax vs proof-corrected argmax")
        print(f"{'sims':>6} {'changed':>9} {'rate':>8} {'raw was loss':>13} "
              f"{'fixed to win':>13} {'strict improve':>15}")
        print("-" * 68)
        for s in args.sims:
            r = out["reconciliation"].get(str(s))
            if not r:
                continue
            a = r["adjudicated"]
            print(f"{s:>6} {r['n_changed']:>9,} {r['changed_rate']:>8.4f} "
                  f"{a['raw_was_proven_loss']:>13,} "
                  f"{a['raw_loss_to_corrected_win']:>13,} "
                  f"{a['strict_improvement']:>15,}")

    print(f"\nwrote {args.output}/summary.json"
          + ("" if args.reuse_policies else " and policies.npz"))


if __name__ == "__main__":
    main()
