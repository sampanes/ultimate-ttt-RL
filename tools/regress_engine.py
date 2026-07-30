"""Regression gate for the frozen final engine. Exits non-zero on a failure.

    python -m tools.regress_engine                 # the gate, ~10 min
    python -m tools.regress_engine --games 4       # a smoke run

WHAT IT GUARDS, and why each check is shaped the way it is.

1. LATENCY  p99 <= 1000 ms, max <= 1250 ms. The frozen operational requirement.

2. TREE-REUSE ADOPTION, measured against its own structural ceiling rather than
   a flat number. The only unavoidable miss is the first move of each game, when
   no tree exists yet, so the ceiling is `1 - games/moves` and a fixed floor
   like 0.94 would mean different things at different game counts. The gate is
   therefore "adoption is within 1 point of the ceiling", which is
   game-count-independent and reads as: nothing except the first move fails to
   adopt.

3. INHERITED SIMULATIONS, as a ratio to new ones. Adoption alone is a hollow
   statistic -- a bug that re-rooted correctly but dropped the subtree's
   statistics would keep adoption at 0.957 and inherit nothing, and check 2
   would not notice. The frozen ratio is 3149/3877 = 0.81; the floor is 0.5.

4. THROUGHPUT, but ONLY when the environment matches the frozen baseline.
   This closes a hole the first three cannot: under a deadline, latency is
   pinned by construction, so an engine that became three times slower would
   still show p99 ~= 1000 and sail through checks 1-3 while playing far weaker.
   Simulations per move is the statistic that actually degrades -- but it is
   also machine-dependent, so gating it on different silicon would produce
   false failures. It is enforced on the reference box and explicitly skipped,
   loudly, anywhere else.

Deliberately NOT gated: playing strength. A strength regression needs a match
against a fixed opponent, which is hours; that is the anchor ladder's job, not
this gate's. What this catches is the cheap, common failure -- a change that
breaks the deadline or quietly disables reuse -- in about ten minutes.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

from tools import engine_registry as reg
from tools.arena_1s import TimedPlayer, latency_report, play_match, print_report

# Measured 2026-07-29 over 240 games on the reference box, engine `final`.
FROZEN = {
    "p99_ms": 998.7,
    "reuse_rate": 0.9569,
    "inherited_sims_per_move": 3149.5,
    "new_sims_per_move": 3877.0,
    "neural_evals_per_move": 2461.0,
    "neural_evals_per_second": 2997.0,
}

ADOPTION_SLACK = 0.01     # points below the structural ceiling
INHERIT_FLOOR = 0.50      # inherited / new simulations
THROUGHPUT_FLOOR = 0.70   # of the frozen per-move network evaluations


class Gate:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, got, want, detail=""):
        self.rows.append((name, bool(ok), got, want, detail))
        return ok

    def skip(self, name, why):
        self.rows.append((name, None, None, None, why))

    def report(self):
        print("\n" + "=" * 78)
        print("  REGRESSION GATE")
        print("=" * 78)
        failed = 0
        for name, ok, got, want, detail in self.rows:
            mark = "SKIP" if ok is None else ("OK" if ok else "X ")
            failed += ok is False
            line = f"  [{mark}] {name:<34s}"
            if ok is not None:
                line += f" {got:>10}   need {want}"
            print(line)
            if detail:
                print(f"         {detail}")
        n = sum(r[1] is not None for r in self.rows)
        print("=" * 78)
        print(f"  {n - failed}/{n} checks passed"
              + (f", {failed} FAILED" if failed else ""))
        return failed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="final",
                    help="registry engine to gate (default: the promoted one)")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--warmup-games", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default="results/arena_1s")
    ap.add_argument("--tag", default="regression")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    drift = reg.env_drift()
    if drift:
        print("[!] environment differs from the frozen baseline:")
        for k, v in drift.items():
            print(f"      {k}: frozen {v['frozen']} -> now {v['now']}")

    gc.disable()
    print(f"[!] automatic cyclic GC off; collecting at game boundaries")

    # The mirror is the same frozen engine on both sides, so the score is 0.5
    # by construction and is not the measurement. What is measured is what this
    # engine costs over realistic positions.
    pa = TimedPlayer(f"engine:{args.engine}", device)
    pb = TimedPlayer(f"engine:{args.engine}", device)
    pb.name = pa.name + "-mirror"
    print(f"gating engine:{args.engine}  fingerprint "
          f"{pa.provenance['fingerprint']}  "
          f"({pa.net_info['params']:,} params)")
    print(f"{args.games} games + {args.warmup_games} warmup, "
          f"{pa.budget_ms:.0f} ms per move, device {device}\n")

    t0 = time.time()
    play_match(pa, pb, args.games, reg.SEEDS["headline"],
               warmup=args.warmup_games, gc_mode="deferred")
    dt = time.time() - t0
    rep = latency_report(pa)
    print_report(pa, rep)

    g = Gate()
    lat, per, thr = rep["latency_ms"], rep["per_move"], rep["throughput"]
    tr = rep["tree_reuse"]

    g.check("latency p99 <= 1000 ms", lat["p99"] <= reg.REQUIREMENT["p99_ms"],
            f"{lat['p99']:.1f}", f"<= {reg.REQUIREMENT['p99_ms']:.0f}",
            f"frozen baseline was {FROZEN['p99_ms']} ms")
    g.check("latency max <= 1250 ms", lat["max"] <= reg.REQUIREMENT["max_ms"],
            f"{lat['max']:.1f}", f"<= {reg.REQUIREMENT['max_ms']:.0f}")

    if pa.reuse:
        # One unavoidable miss per game: the first move has no prior tree.
        ceiling = 1.0 - args.games / max(tr["moves"], 1)
        floor = ceiling - ADOPTION_SLACK
        other = {k: v for k, v in tr["miss_reason"].items()
                 if v and k != "no_tree"}
        g.check("tree reuse adoption", tr["reuse_rate"] >= floor,
                f"{tr['reuse_rate']:.4f}", f">= {floor:.4f}",
                f"structural ceiling {ceiling:.4f} (one first move per game); "
                f"other misses: {other or 'none'}")

        new_sims = per["simulations"]
        ratio = per["inherited_simulations"] / max(new_sims, 1e-9)
        g.check("inherited / new simulations", ratio >= INHERIT_FLOOR,
                f"{ratio:.3f}", f">= {INHERIT_FLOOR:.2f}",
                f"{per['inherited_simulations']:.0f} inherited vs "
                f"{new_sims:.0f} new; adoption without inheritance would be "
                f"a hollow pass")
    else:
        g.skip("tree reuse adoption", f"engine:{args.engine} has reuse off")

    if drift:
        g.skip("network evaluations per move",
               "environment differs from the frozen baseline -- throughput is "
               "machine-dependent and is not gated off the reference box")
    elif args.engine != "final":
        g.skip("network evaluations per move",
               f"frozen throughput is recorded for `final`, not "
               f"`{args.engine}`")
    else:
        want = FROZEN["neural_evals_per_move"] * THROUGHPUT_FLOOR
        g.check("network evaluations per move",
                per["neural_evaluations"] >= want,
                f"{per['neural_evaluations']:.0f}", f">= {want:.0f}",
                f"frozen {FROZEN['neural_evals_per_move']:.0f}/move. Under a "
                f"deadline the p99 cannot see a slowdown -- this can")

    failed = g.report()
    print(f"\n  {rep['moves']:,} moves in {dt / 60:.1f} min")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"engine": args.engine, "games": args.games,
                   "config": pa.config(), "report": rep,
                   "frozen": FROZEN, "environment": reg.environment(),
                   "environment_drift": drift, "seconds": dt,
                   "checks": [{"name": n, "pass": o, "got": got, "need": w,
                               "detail": d} for n, o, got, w, d in g.rows],
                   "PASS": failed == 0}, fh, indent=2)
    print(f"  wrote {out}")

    if failed:
        print("\n[X] REGRESSION -- the frozen engine no longer meets its "
              "own baseline")
        sys.exit(1)
    print("\n[OK] engine matches its frozen baseline")


if __name__ == "__main__":
    main()
