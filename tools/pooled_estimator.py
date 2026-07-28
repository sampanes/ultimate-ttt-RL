"""Pre-registered estimator for the solved-node distillation pilot.

Defined before any solved target exists, so the analysis cannot be chosen to
suit the number it produces.

THE STATISTIC
-------------
Per seed s, the 800-solve student plays the 50-solve student head to head over
paired openings with colours swapped. A game scores 1 / 0.5 / 0. The primary
figure pools GAMES across the three matched seed pairs:

    pooled = sum_s (wins_s + 0.5 * draws_s) / sum_s games_s

not the equal-weight mean of three percentages. The two coincide only when
every pair contributes the same number of games; if any pair is extended (the
frozen manifest explicitly allows extending beyond 800 games when a CI still
overlaps) the equal-weight mean would silently overweight the short pairs.

VARIANCE
--------
Binomial/Wilson is wrong here. A draw scores exactly 0.5 with zero variance,
and this matchup draws at roughly 30%, so a binomial interval is needlessly
wide. Variance is taken from the observed outcome distribution:

    mean = (w + 0.5d) / n
    E[x^2] = (w + 0.25d) / n
    var  = E[x^2] - mean^2

THE ESTIMATOR HIERARCHY (fixed by the owner before results exist)
-----------------------------------------------------------------
`pooled >= 0.4554` AND the recovery-delta CI excludes zero. "The delta CI" is
ambiguous in a way that changes the answer, so all four reads are computed and
each is assigned a fixed role that CANNOT be reassigned after seeing them:

  PRIMARY      pooled solved result vs the fixed published baseline 0.4108.
               Tests the exact preregistered intervention claim: does solved
               propagation recover at least half the measured penalty. This
               and only this is the pass/fail gate.

  CONFIRMATORY paired by seed. Statistically the most efficient analysis,
               because each solved seed is matched to the corresponding old
               seed and the large seed-specific student-strength effect
               cancels. The baseline seed spread is 0.132, roughly 3x the
               effect being chased, so this is where the signal is cleanest.

  SENSITIVITY  two-sample at game level: baseline treated as an estimate
               carrying its own measured sampling error (~1.4x wider).

  WARNING      seed-level random-effects / two-sample. REPORTED, NEVER A GATE.
               With three highly heterogeneous seeds it asks a broader
               replication-across-training-seeds question that this pilot was
               never powered to answer, and it will fail even for a real
               effect. It must not be allowed to retroactively invalidate the
               pilot; see `scope_of_inference`.

THE PAIRED STATISTIC, PREREGISTERED EXACTLY
-------------------------------------------
    For each seed:
        delta_seed = solved_score_seed - baseline_score_seed

    Primary paired estimate:
        weighted pooled delta across paired games
    Secondary descriptive estimate:
        arithmetic mean of the three seed-level deltas

Pairing is claimed at two possible resolutions and the code will not overstate
which one it has:

  game level  requires the two evaluations to have used IDENTICAL opening
              schedules. That holds iff they share a match seed, because
              `_eval_openings(n, seed)` depends on the seed alone and emits a
              prefix-stable stream, `game_seed = seed + opening_idx*2 + side`
              is likewise prefix-stable, and students move with
              `sample_moves=0` (deterministic). Under that condition game i of
              the old run and game i of the new run share opening, colour and
              per-game RNG, and the per-game difference is a genuine paired
              observation. Requires both runs' per-game outcome vectors.

  seed level  the fallback. The two matches are then treated as independent
              samples within each seed, so var(delta_s) = se_new^2 + se_base^2.
              Still cancels the seed effect, just not the opening effect.

`pairing.claimed` records which was actually used, per seed. Games whose
opening schedules were not verified identical are NEVER counted as paired.

    python -m tools.pooled_estimator --results FILE.json
    python -m tools.pooled_estimator --demo
    python -m tools.pooled_estimator --demo-at-threshold
"""
from __future__ import annotations

import argparse
import json
import math

# From RESULT_DISTILL_PILOT.md / results/distill_pilot_eval.json -- the
# solve-OFF reversal this pilot must shrink. Stored as raw W/D/L so every
# derived figure below has ONE source of truth and cannot drift from the
# published numbers.
BASELINE_COUNTS = {                       # seed -> [wins, draws, losses]
    "11": [193, 147, 460],
    "22": [297, 150, 353],
    "33": [261, 173, 366],
}
BASELINE_POOLED = 0.4108                  # published, cross-checked below
BASELINE_GAMES_PER_SEED = 800
BASELINE_PENALTY = 0.5 - BASELINE_POOLED  # 0.0892
SUCCESS_POOLED = 0.4554                   # half the penalty recovered
BASELINE_EVAL_FILE = "results/distill_pilot_eval.json"

# tools/eval_distill_pilot.py: play_match_detailed(..., H2H_BASE_SEED + seed).
# Equality of this value between the two runs is exactly the condition that
# licenses game-level pairing.
H2H_BASE_SEED = 9901
BASELINE_MATCH_SEEDS = {s: H2H_BASE_SEED + int(s) for s in BASELINE_COUNTS}

Z = 1.959963984540054                     # 95% normal
T2 = 4.302652729911275                    # t(0.975, df=2)

SCOPE_OF_INFERENCE = (
    "The experiment can establish recovery for these preregistered matched "
    "training seeds. It cannot precisely estimate how consistently that "
    "recovery generalizes across arbitrary new training seeds.")


def score_stats(w, d, l):
    """Mean, variance and SE of the per-game score from raw W/D/L counts."""
    n = w + d + l
    if n == 0:
        return {"n": 0, "score": None, "se": None, "var": None}
    mean = (w + 0.5 * d) / n
    var = max((w + 0.25 * d) / n - mean * mean, 0.0)
    return {"n": n, "wins": w, "draws": d, "losses": l,
            "score": mean, "var": var, "se": math.sqrt(var / n)}


BASELINE_STATS = {s: score_stats(*c) for s, c in BASELINE_COUNTS.items()}
BASELINE_PER_SEED = {s: v["score"] for s, v in BASELINE_STATS.items()}
BASELINE_PER_SEED_SE = {s: v["se"] for s, v in BASELINE_STATS.items()}
BASELINE_POOLED_STATS = score_stats(
    sum(c[0] for c in BASELINE_COUNTS.values()),
    sum(c[1] for c in BASELINE_COUNTS.values()),
    sum(c[2] for c in BASELINE_COUNTS.values()))
BASELINE_POOLED_SE_GAMELEVEL = BASELINE_POOLED_STATS["se"]   # 0.008970

# Guard: if the stored counts ever stop reproducing the published headline,
# every delta below is silently measured against the wrong thing.
assert abs(BASELINE_POOLED_STATS["score"] - BASELINE_POOLED) < 5e-5, (
    "BASELINE_COUNTS no longer reproduce the published 0.4108")


def ci(mean, se, crit=Z):
    if se is None:
        return None
    return {"lo": mean - crit * se, "hi": mean + crit * se,
            "excludes_zero": (mean - crit * se) * (mean + crit * se) > 0}


def heterogeneity(per_seed):
    """Cochran's Q and I^2 across the seed pairs.

    High heterogeneity means the seeds disagree about the effect, and a pooled
    number is then a summary of disagreeing things rather than a sharper
    estimate of one thing. With 3 seeds this has almost no power -- it is a
    tripwire, not a test, and per the estimator hierarchy it never gates.
    """
    pts = [(s["score"], s["se"]) for s in per_seed.values() if s.get("se")]
    if len(pts) < 2:
        return {"status": "insufficient"}
    wts = [1.0 / (se * se) for _, se in pts]
    fixed = sum(w * m for (m, _), w in zip(pts, wts)) / sum(wts)
    q = sum(w * (m - fixed) ** 2 for (m, _), w in zip(pts, wts))
    df = len(pts) - 1
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    scores = [m for m, _ in pts]
    return {"cochran_q": q, "df": df, "i_squared": i2,
            "inverse_variance_mean": fixed,
            "spread": max(scores) - min(scores),
            "note": ("I^2 > 0.5 means the seeds disagree materially and the "
                     "pooled figure should not be read as one effect. Only 3 "
                     "seeds, so absence of heterogeneity proves little.")}


def _normalise(entry):
    """Accept either [w, d, l] or a dict with counts / outcomes / match_seed."""
    if isinstance(entry, (list, tuple)):
        return {"wins": entry[0], "draws": entry[1], "losses": entry[2],
                "outcomes": None, "match_seed": None}
    out = dict(entry)
    oc = out.get("outcomes")
    if oc is not None and "wins" not in out:
        out["wins"] = sum(1 for x in oc if x == 1.0)
        out["draws"] = sum(1 for x in oc if x == 0.5)
        out["losses"] = sum(1 for x in oc if x == 0.0)
    out.setdefault("outcomes", None)
    out.setdefault("match_seed", None)
    return out


def pairing_status(entries, baseline_outcomes):
    """Decide, per seed, the strongest pairing resolution actually earned.

    Deliberately conservative: game-level pairing is claimed ONLY when the
    match seeds are known, equal, and both per-game vectors are in hand. A
    missing match seed is treated as unverified, not as a pass.
    """
    per_seed, levels = {}, []
    for s, e in entries.items():
        want = BASELINE_MATCH_SEEDS.get(s)
        got = e.get("match_seed")
        have_new = e.get("outcomes") is not None
        have_base = bool(baseline_outcomes) and s in baseline_outcomes
        if got is None:
            why, level = "match_seed not recorded by the new run", "seed"
        elif want is not None and got != want:
            why, level = (f"match seed {got} != baseline {want}: different "
                          f"opening schedule, pairing NOT claimed"), "seed"
        elif not have_new:
            why, level = "new run did not persist per-game outcomes", "seed"
        elif not have_base:
            why, level = ("baseline per-game outcomes unavailable -- rerun "
                          "tools.eval_distill_pilot to regenerate them; the "
                          "match is deterministic so it reproduces exactly"), "seed"
        else:
            n = min(len(e["outcomes"]), len(baseline_outcomes[s]))
            why, level = (f"identical opening schedule (seed {got}); first {n} "
                          f"games pair one-to-one"), "game"
        per_seed[s] = {"level": level, "reason": why,
                       "match_seed_new": got, "match_seed_baseline": want}
        levels.append(level)
    overall = "game" if levels and all(x == "game" for x in levels) else "seed"
    return {"claimed": overall, "per_seed": per_seed,
            "condition": ("game-level pairing requires an identical opening "
                          "schedule, which holds iff the match seed and the "
                          "deterministic move policy match; see module header"),
            "note": ("games that did not use identical opening schedules are "
                     "never counted as paired")}


def paired_delta(entries, per_seed, pairing, baseline_outcomes):
    """The preregistered paired statistic.

    Primary   : weighted pooled delta across PAIRED games.
    Secondary : arithmetic mean of the seed-level deltas (descriptive only).
    """
    rows = {}
    for s, v in per_seed.items():
        if not v["n"] or s not in BASELINE_PER_SEED:
            continue
        lvl = pairing["per_seed"].get(s, {}).get("level", "seed")
        d = v["score"] - BASELINE_PER_SEED[s]
        if lvl == "game":
            a = entries[s]["outcomes"]
            b = baseline_outcomes[s]
            n = min(len(a), len(b))
            diffs = [a[i] - b[i] for i in range(n)]
            m = sum(diffs) / n
            var = max(sum((x - m) ** 2 for x in diffs) / n, 0.0)
            rows[s] = {"level": "game", "n_paired": n, "delta": m,
                       "se": math.sqrt(var / n),
                       "delta_seed_level_for_contrast": d}
        else:
            se = math.sqrt(v["se"] ** 2 + BASELINE_PER_SEED_SE[s] ** 2)
            rows[s] = {"level": "seed", "n_paired": v["n"], "delta": d,
                       "se": se}
    if not rows:
        return None

    # PRIMARY paired estimate: weighted by paired games, not equal-weight.
    tot = sum(r["n_paired"] for r in rows.values())
    pooled = sum(r["delta"] * r["n_paired"] for r in rows.values()) / tot
    # Variance of a fixed-weight linear combination of independent seed deltas.
    var = sum((r["n_paired"] / tot) ** 2 * r["se"] ** 2 for r in rows.values())
    se = math.sqrt(var)

    # SECONDARY descriptive: unweighted mean of the three seed deltas. Carries
    # a df=2 t interval, which is wide by construction -- it is a description
    # of how the three seeds sit, not the estimate the gate leans on.
    ds = [r["delta"] for r in rows.values()]
    mean = sum(ds) / len(ds)
    if len(ds) > 1:
        v = sum((x - mean) ** 2 for x in ds) / (len(ds) - 1)
        se_t = math.sqrt(v / len(ds))
        arith = {"delta": mean, "se": se_t, "df": len(ds) - 1,
                 **(ci(mean, se_t, T2) or {}),
                 "interval": "t(0.975, df=2) = 4.3027, wide by construction"}
    else:
        arith = {"delta": mean, "se": None}

    return {
        "definition": "delta_seed = solved_score_seed - baseline_score_seed",
        "pairing_level": pairing["claimed"],
        "efficiency_note": (
            "Pairing always removes the seed-level term, which is the point: "
            "against read 4 the gain is enormous (0.0407 of baseline spread "
            "eliminated). Whether it also beats the PRIMARY read depends on "
            "resolution. At GAME level it does, because the opening is held "
            "fixed and the per-game difference has far less variance than "
            "either score. At SEED level it cannot, because it honestly "
            "carries the baseline's within-seed sampling error that PRIMARY "
            "assumes away -- it is then a more defensible interval, not a "
            "narrower one."),
        "primary_weighted_pooled_delta": {
            "delta": pooled, "se": se, "n_paired": tot, **(ci(pooled, se) or {}),
            "weights": "paired games per seed",
        },
        "secondary_arithmetic_mean_delta": arith,
        "per_seed": rows,
    }


def analyse(per_seed_counts, baseline_outcomes=None):
    entries = {str(s): _normalise(e) for s, e in per_seed_counts.items()}
    baseline_outcomes = {str(k): v for k, v in (baseline_outcomes or {}).items()}

    per_seed = {s: score_stats(e["wins"], e["draws"], e["losses"])
                for s, e in entries.items()}
    pooled = score_stats(
        sum(v["wins"] for v in per_seed.values() if v["n"]),
        sum(v["draws"] for v in per_seed.values() if v["n"]),
        sum(v["losses"] for v in per_seed.values() if v["n"]))
    p = pooled["score"]

    bs = list(BASELINE_PER_SEED.values())
    base_mean = sum(bs) / len(bs)
    base_var = sum((x - base_mean) ** 2 for x in bs) / (len(bs) - 1)
    se_base_seedlevel = math.sqrt(base_var / len(bs))
    se_base_gamelevel = BASELINE_POOLED_SE_GAMELEVEL

    pairing = pairing_status(entries, baseline_outcomes)

    out = {
        "primary_statistic": "game-weighted pooled score, 800-solve vs 50-solve",
        "estimator_roles": {
            "PRIMARY": "1_pooled_vs_fixed_baseline -- the only pass/fail gate",
            "CONFIRMATORY": "2_paired_by_seed -- most efficient, cancels seed effect",
            "SENSITIVITY": "3_two_sample_gamelevel",
            "GENERALIZATION_WARNING": ("4_seed_level -- report only, never a "
                                       "gate, cannot invalidate the pilot"),
        },
        "scope_of_inference": SCOPE_OF_INFERENCE,
        "per_seed": per_seed,
        "pooled": pooled,
        "equal_weight_mean_for_contrast": (
            sum(v["score"] for v in per_seed.values() if v["n"])
            / max(1, sum(1 for v in per_seed.values() if v["n"]))),
        "heterogeneity": heterogeneity(per_seed),
        "pairing": pairing,
        "baseline": {"pooled": BASELINE_POOLED,
                     "counts": BASELINE_COUNTS,
                     "per_seed": BASELINE_PER_SEED,
                     "per_seed_se": BASELINE_PER_SEED_SE,
                     "games_per_seed": BASELINE_GAMES_PER_SEED,
                     "game_level_se": se_base_gamelevel,
                     "seed_level_se": se_base_seedlevel,
                     "seed_spread": max(bs) - min(bs),
                     "match_seeds": BASELINE_MATCH_SEEDS,
                     "source": BASELINE_EVAL_FILE},
    }
    if p is None:
        return out

    absolute_recovery = p - BASELINE_POOLED
    out["recovery"] = {
        "absolute_recovery": absolute_recovery,
        "fraction_recovered": absolute_recovery / BASELINE_PENALTY,
        "remaining_penalty": 0.5 - p,
        "reading_guide": (
            "These three are reported together so a value near 0.456 cannot be "
            "called simply 'success' or simply 'continued reversal'. It is "
            "both: roughly half the penalty recovered, while the deeper-target "
            "student still loses the head to head."),
    }

    se_new = pooled["se"]
    d1 = ci(absolute_recovery, se_new)
    se3 = math.sqrt(se_new ** 2 + se_base_gamelevel ** 2)
    d3 = ci(absolute_recovery, se3)
    se4 = math.sqrt(se_new ** 2 + se_base_seedlevel ** 2)
    d4 = ci(absolute_recovery, se4)
    d2 = paired_delta(entries, per_seed, pairing, baseline_outcomes)

    out["delta_reads"] = {
        "1_pooled_vs_fixed_baseline": {
            "role": "PRIMARY", "delta": absolute_recovery, "se": se_new,
            **(d1 or {}),
            "assumes": "baseline 0.4108 is exact; the preregistered rule"},
        "2_paired_by_seed": dict(role="CONFIRMATORY", **(d2 or {})),
        "3_two_sample_gamelevel": {
            "role": "SENSITIVITY", "delta": absolute_recovery, "se": se3,
            **(d3 or {}),
            "assumes": ("baseline is an estimate carrying its measured "
                        "game-level sampling error")},
        "4_two_sample_seedlevel": {
            "role": "GENERALIZATION_WARNING", "delta": absolute_recovery,
            "se": se4, **(d4 or {}),
            "assumes": ("baseline error is the spread across TRAINING RUNS. "
                        "Answers 'would this replicate under new seeds', which "
                        "three heterogeneous seeds cannot resolve"),
            "gating": ("NEVER a gate. Failing here is expected even for a real "
                       "effect and does not invalidate the pilot.")},
    }

    passes_level = p >= SUCCESS_POOLED
    passes_ci = bool(d1 and d1["excludes_zero"] and absolute_recovery > 0)
    pw = (d2 or {}).get("primary_weighted_pooled_delta") or {}
    out["prereg"] = {
        "rule": f"pooled >= {SUCCESS_POOLED} AND recovery-delta CI excludes 0",
        "gate_estimator": "1_pooled_vs_fixed_baseline (PRIMARY) only",
        "pooled": p,
        "threshold": SUCCESS_POOLED,
        "passes_level": passes_level,
        "passes_ci": passes_ci,
        "MATERIALLY_SHRINKS": bool(passes_level and passes_ci),
        "confirmatory_agrees": bool(pw.get("excludes_zero")
                                    and (pw.get("delta") or 0) > 0),
        "sensitivity_agrees": bool(d3 and d3["excludes_zero"]
                                   and absolute_recovery > 0),
        "generalization_warning_agrees": bool(d4 and d4["excludes_zero"]
                                              and absolute_recovery > 0),
        "resolution_note": (
            f"the threshold delta is {SUCCESS_POOLED - BASELINE_POOLED:+.4f}; "
            f"PRIMARY resolves {Z * se_new:.4f}, "
            f"CONFIRMATORY resolves "
            f"{(Z * pw['se']) if pw.get('se') else float('nan'):.4f}, "
            f"SENSITIVITY resolves {Z * se3:.4f}, "
            f"WARNING resolves {Z * se4:.4f}"),
    }
    return out


def report(a):
    print(f"{'seed':>6} {'games':>7} {'W':>5} {'D':>5} {'L':>5} "
          f"{'score':>8} {'se':>7} {'base':>7} {'delta':>8}")
    for s, v in a["per_seed"].items():
        if not v["n"]:
            continue
        b = a["baseline"]["per_seed"].get(s)
        base_txt = f"{b:.4f}" if b is not None else "--"
        delta_txt = f"{v['score'] - b:+.4f}" if b is not None else "--"
        print(f"{s:>6} {v['n']:>7} {v['wins']:>5} {v['draws']:>5} {v['losses']:>5} "
              f"{v['score']:>8.4f} {v['se']:>7.4f} {base_txt:>7} {delta_txt:>8}")
    p = a["pooled"]
    print(f"{'POOLED':>6} {p['n']:>7} {p['wins']:>5} {p['draws']:>5} {p['losses']:>5} "
          f"{p['score']:>8.4f} {p['se']:>7.4f} "
          f"{a['baseline']['pooled']:>7.4f} "
          f"{p['score'] - a['baseline']['pooled']:>+8.4f}")
    print(f"\nequal-weight mean (contrast only): "
          f"{a['equal_weight_mean_for_contrast']:.4f}")

    h = a["heterogeneity"]
    print(f"heterogeneity  Q={h.get('cochran_q', 0):.3f} df={h.get('df')} "
          f"I^2={h.get('i_squared', 0):.3f}  spread={h.get('spread', 0):.4f}")

    pr = a["pairing"]
    print(f"\npairing claimed: {pr['claimed'].upper()} LEVEL")
    for s, v in pr["per_seed"].items():
        print(f"  seed {s:>3}  {v['level']:<5}  {v['reason']}")

    r = a["recovery"]
    print(f"\nabsolute_recovery   {r['absolute_recovery']:+.4f}")
    print(f"fraction_recovered  {r['fraction_recovered']:+.4f}  "
          f"(of the {BASELINE_PENALTY:.4f} penalty)")
    print(f"remaining_penalty   {r['remaining_penalty']:+.4f}  "
          f"(distance from 0.5 parity)")

    print("\ndelta reads:")
    for k, v in a["delta_reads"].items():
        if not v:
            continue
        role = v.get("role", "")
        if k == "2_paired_by_seed":
            w = v.get("primary_weighted_pooled_delta")
            if not w:
                print(f"  {k:<28} {role:<22} unavailable")
                continue
            print(f"  {k:<28} {role:<22} {w['delta']:+.4f} "
                  f"[{w['lo']:+.4f}, {w['hi']:+.4f}] "
                  f"excl0={w['excludes_zero']}  n={w['n_paired']} "
                  f"({v['pairing_level']}-paired)")
            s2 = v["secondary_arithmetic_mean_delta"]
            print(f"  {'  (secondary, descriptive)':<28} {'':<22} "
                  f"{s2['delta']:+.4f}" +
                  (f" [{s2['lo']:+.4f}, {s2['hi']:+.4f}]" if "lo" in s2 else ""))
        elif "lo" in v:
            print(f"  {k:<28} {role:<22} {v['delta']:+.4f} "
                  f"[{v['lo']:+.4f}, {v['hi']:+.4f}] "
                  f"excl0={v['excludes_zero']}")

    g = a["prereg"]
    print(f"\nPREREG  {g['rule']}")
    print(f"  gate estimator: {g['gate_estimator']}")
    print(f"  pooled {g['pooled']:.4f} vs {g['threshold']:.4f} -> "
          f"level={g['passes_level']}  ci={g['passes_ci']}")
    print(f"  MATERIALLY_SHRINKS = {g['MATERIALLY_SHRINKS']}")
    print(f"  confirmatory agrees            = {g['confirmatory_agrees']}")
    print(f"  sensitivity agrees             = {g['sensitivity_agrees']}")
    print(f"  generalization warning agrees  = "
          f"{g['generalization_warning_agrees']}  (not a gate)")
    print(f"  resolution: {g['resolution_note']}")
    print(f"\nSCOPE: {a['scope_of_inference']}")


def _synth(scores, n=800, draw_rate=0.30):
    """W/D/L counts reproducing a target score at a plausible draw rate."""
    counts = {}
    for s, sc in scores.items():
        d = round(n * draw_rate)
        w = round(sc * n - 0.5 * d)
        counts[s] = [w, d, n - d - w]
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="JSON: {seed: [w,d,l]} or "
                                      "{seed: {wins,draws,losses,outcomes,match_seed}}")
    ap.add_argument("--baseline-outcomes",
                    help="JSON: {seed: [per-game score,...]} from a baseline "
                         "replay; enables game-level pairing")
    ap.add_argument("--demo", action="store_true",
                    help="run the estimator on the baseline itself (delta ~ 0)")
    ap.add_argument("--demo-at-threshold", action="store_true",
                    help="power probe: every seed shifted to exactly clear "
                         "the preregistered threshold")
    ap.add_argument("--output")
    args = ap.parse_args()

    base_oc = None
    if args.demo:
        counts = _synth(BASELINE_PER_SEED)
    elif args.demo_at_threshold:
        shift = SUCCESS_POOLED - BASELINE_POOLED
        counts = _synth({s: v + shift for s, v in BASELINE_PER_SEED.items()})
    elif args.results:
        counts = {str(k): v for k, v in
                  json.load(open(args.results, encoding="utf-8")).items()}
        if args.baseline_outcomes:
            base_oc = json.load(open(args.baseline_outcomes, encoding="utf-8"))
    else:
        raise SystemExit("[X] need --results, --demo or --demo-at-threshold")

    a = analyse(counts, base_oc)
    report(a)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(a, fh, indent=2)
        print(f"\n[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
