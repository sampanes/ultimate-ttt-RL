"""Read the anchor-ladder matches and rule on whether the ladder is ORDERED.

A ladder is only a ruler if its rungs are in the order they claim to be. This
program has already been burned once by an ordering that looked obvious --
800-simulation teachers distil WORSE than 50-simulation ones -- so "more time is
stronger" gets measured at every adjacent pair rather than assumed at any.

Each rung is one doubling of thinking time, which makes the scores directly
comparable to each other: they all answer "what is 2x the clock worth here?"

    python -m tools.ladder_report

Verdicts, and why the middle one is not a failure:

    ORDERED         the longer budget wins, and the CI excludes 0.5
    NOT SEPARATED   the longer budget is ahead but the CI includes 0.5 --
                    the rungs are too close for this many games to tell apart,
                    which is a fact about the ladder, not an error
    INVERTED        the longer budget LOSES, CI excluding 0.5. The rung is
                    unusable as a ruler and the ladder must be cut below it.
"""
from __future__ import annotations

import argparse
import json
import os

from tools import engine_registry as reg

# (tag, longer-budget engine, shorter-budget engine, budget_ms pair)
RUNGS = [
    ("ladder_B500_A250", "anchor_B", "anchor_A", (500, 250)),
    ("ladder_final_B500", "final", "anchor_B", (1000, 500)),
    ("ladder_C2000_final", "anchor_C", "final", (2000, 1000)),
    ("ladder_D4000_C2000", "anchor_D", "anchor_C", (4000, 2000)),
]

TARGET_LO, TARGET_HI = 0.25, 0.75


def load(outdir, tag):
    path = os.path.join(outdir, f"{tag}.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verdict(score, lo, hi):
    if lo > 0.5:
        return "ORDERED"
    if hi < 0.5:
        return "INVERTED"
    return "NOT SEPARATED"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results/arena_1s")
    args = ap.parse_args()

    print("=" * 78)
    print("  TIME-BUDGET ANCHOR LADDER -- ordering validation")
    print("=" * 78)
    print("  Every rung is one doubling of thinking time. Same network, same")
    print("  engine, same openings (seed %d), colours swapped.\n"
          % reg.SEEDS["ladder"])

    rows, missing = [], []
    for tag, hi_eng, lo_eng, (hi_ms, lo_ms) in RUNGS:
        doc = load(args.outdir, tag)
        if doc is None:
            missing.append(tag)
            continue
        r = doc["results"]["h2h"]
        s, (ci_lo, ci_hi) = r["score_for_a"], r["ci95"]
        rows.append({
            "tag": tag, "hi": hi_eng, "lo": lo_eng,
            "hi_ms": hi_ms, "lo_ms": lo_ms,
            "score": s, "lo95": ci_lo, "hi95": ci_hi,
            "w": r["wins"], "d": r["draws"], "l": r["losses"],
            "n": len(r["outcomes"]), "minutes": r["seconds"] / 60.0,
            "verdict": verdict(s, ci_lo, ci_hi),
            "a_report": r["player_a"]["report"],
            "b_report": r["player_b"]["report"],
        })

    print(f"  {'doubling':>16s}  {'score':>7s}  {'95% CI':>18s}  "
          f"{'W/D/L':>12s}  {'n':>4s}  verdict")
    print("  " + "-" * 74)
    for x in rows:
        pair = f"{x['lo_ms']} -> {x['hi_ms']} ms"
        ci = f"[{x['lo95']:.4f}, {x['hi95']:.4f}]"
        wdl = f"{x['w']}/{x['d']}/{x['l']}"
        print(f"  {pair:>16s}  {x['score']:7.4f}  {ci:>18s}  {wdl:>12s}  "
              f"{x['n']:4d}  {x['verdict']}")
    if missing:
        print(f"\n  [!] not yet run: {', '.join(missing)}")

    # -- what the clock actually buys -------------------------------------
    print("\n  Simulations the engine actually achieves, by budget:")
    print(f"    {'budget':>8s}  {'new sims':>9s}  {'inherited':>10s}  "
          f"{'nn-evals':>9s}  {'nn-evals/s':>11s}  {'p99 ms':>8s}")
    seen = {}
    for x in rows:
        for eng, ms, rep in ((x["hi"], x["hi_ms"], x["a_report"]),
                             (x["lo"], x["lo_ms"], x["b_report"])):
            if ms in seen or not rep.get("moves"):
                continue
            seen[ms] = True
            per, thr = rep["per_move"], rep["throughput"]
            print(f"    {ms:6d}ms  {per['simulations']:9.0f}  "
                  f"{per['inherited_simulations']:10.0f}  "
                  f"{per['neural_evaluations']:9.0f}  "
                  f"{thr['neural_evals_per_second']:11.0f}  "
                  f"{rep['latency_ms']['p99']:8.1f}")

    # -- the deployment agent's place on the ladder ------------------------
    print("\n  Where the 1,000 ms deployment agent sits:")
    finals = []
    for x in rows:
        if x["hi"] == "final":
            finals.append((x["lo"], x["lo_ms"], x["score"],
                           x["lo95"], x["hi95"]))
        elif x["lo"] == "final":
            finals.append((x["hi"], x["hi_ms"], 1.0 - x["score"],
                           1.0 - x["hi95"], 1.0 - x["lo95"]))
    for eng, ms, s, lo, hi in sorted(finals, key=lambda t: t[1]):
        inband = TARGET_LO <= s <= TARGET_HI
        print(f"    vs {eng:9s} ({ms:4d} ms)   final scores {s:.4f}  "
              f"[{lo:.4f}, {hi:.4f}]   "
              f"{'IN BAND' if inband else 'outside 0.25-0.75'}")

    usable = [t for t in finals if TARGET_LO <= t[2] <= TARGET_HI]
    print("\n  " + "=" * 74)
    if not rows:
        print("  no rungs available yet")
    elif any(x["verdict"] == "INVERTED" for x in rows):
        bad = [x for x in rows if x["verdict"] == "INVERTED"]
        print("  [X] LADDER IS NOT ORDERED. Inverted rung(s): "
              + ", ".join(f"{b['lo_ms']}->{b['hi_ms']} ms" for b in bad))
        print("      Cut the ladder below the inversion. A rung that is not")
        print("      ordered cannot serve as a ruler for anything.")
    elif usable:
        # Prefer the HARDEST anchor still inside the band: it leaves the most
        # headroom before a future candidate saturates it, which is exactly how
        # gregory(d4) died.
        eng, ms, s, _lo, _hi = max(usable, key=lambda t: t[1])
        print(f"  [OK] ladder ordered where measured. PRIMARY ANCHOR: "
              f"{eng} ({ms} ms), final scores {s:.4f}")
        print(f"       Hardest rung still inside {TARGET_LO}-{TARGET_HI}, so "
              f"it has the most headroom")
        print(f"       before a future candidate saturates it -- which is how "
              f"gregory(d4) died.")
    else:
        print("  [!] no measured anchor puts the 1,000 ms agent inside "
              f"{TARGET_LO}-{TARGET_HI}.")
        print("      Every rung is either too easy or too hard; the ladder "
              "needs a rung between the two closest.")
    print("  " + "=" * 74)


if __name__ == "__main__":
    main()
