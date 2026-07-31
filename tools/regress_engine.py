"""Regression gate for the frozen final engine. Exits non-zero on a failure.

    python -m tools.regress_engine                 # the gate, ~10 min
    python -m tools.regress_engine --games 4       # a smoke run

WHAT IT GUARDS, and why each check is shaped the way it is.

1. LATENCY  p99 <= 1000 ms, max <= 1250 ms. The frozen operational requirement.

1b. RESERVE HEADROOM, which is check 1 stated as a cause instead of a verdict.
   The search honours its own deadline: over 5,517 measured moves it never
   exceeded 981.1 ms against a 980 ms target, and the worst chunk on the moves
   that DID blow the requirement was 3.3 ms. Every p99 miss so far has come
   from the work the caller waits for and the search does not time -- re-rooting
   and releasing the discarded tree. The reserve exists to absorb exactly that,
   so `overhead p99 <= reserve` fails before the requirement does and names the
   thing to fix. It is how `pocket` was found to miss p99 by 2.3 ms while its
   search was blameless.

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
   would not notice.

   The floor is 0.35, and the reason it is not 0.5 is check 0 below. How much a
   search inherits depends on how well it predicted the OPPONENT'S reply, which
   depends on whether the opponent shares its network. Measured: `final`
   inherits 0.81 of its new simulations against a same-network opponent and
   only 0.48 against a different one. A 0.5 floor calibrated on the first case
   fails legitimately on the second.

0. THE OPPONENT IS NEVER A MIRROR, and this is the check that all the others
   sit on. An engine playing a byte-identical copy of itself predicts its
   replies far better than it predicts anyone else's, so it adopts a much
   larger subtree and does much less re-rooting work. Measured on `pocket`:
   3,707 inherited simulations per move against itself, 2,224 against `final`,
   and an overhead p99 of 18.77 ms against 23.06 -- the difference between
   passing the requirement and failing it. This gate shipped as a mirror and
   PASSED an engine that fails against a real opponent. A deployment
   requirement measured against the single most favourable opponent in
   existence is not measuring deployment.

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

# Measured 2026-07-29 over 240 games on the reference box, engine `final`
# against `original` -- a SAME-NETWORK opponent, which is the favourable case
# for inheritance. `inherited_sims_per_move` in particular does not transfer to
# a cross-network opponent: the same engine inherits 1,551.9 against `pocket`.
FROZEN = {
    "p99_ms": 998.7,
    "reuse_rate": 0.9569,
    "inherited_sims_per_move": 3149.5,
    "new_sims_per_move": 3877.0,
    "neural_evals_per_move": 2461.0,
    "neural_evals_per_second": 2997.0,
}

ADOPTION_SLACK = 0.01     # points below the structural ceiling
# 0.35, not 0.50: inheritance depends on whether the opponent shares the
# engine's network. Measured for `final`: 0.81 same-network, 0.48 cross-network.
# The floor has to clear the cross-network case or the gate fails honestly-
# behaving engines. It is still far below anything a working re-root produces.
INHERIT_FLOOR = 0.35      # inherited / new simulations
THROUGHPUT_FLOOR = 0.70   # of the frozen per-move network evaluations

# A gate must not play an engine against a copy of itself -- see check 0. What
# inflates inheritance is shared WEIGHTS, not a shared name or config, so the
# default is derived from the checkpoint: an engine is gated against a standing
# opponent that does not use its network. Pinning `{"final": "pocket"}` by name
# would have left `original` and every anchor rung facing a same-network
# opponent, which is the defect wearing a different hat.
CROSS_NET_OPPONENT = {reg.GEN22: "pocket", reg.POCKET: "final"}
FALLBACK_OPPONENT = "final"


def choose_opponent(engine, requested=None, allow_mirror=False):
    """The engine this gate plays against. Never the engine itself.

    See check 0. Extracted so the rule can be tested without playing games.
    """
    if requested is None:
        ckpt = reg.ENGINES[engine].get("ckpt")
        requested = CROSS_NET_OPPONENT.get(ckpt, FALLBACK_OPPONENT)
    opponent = requested
    if opponent == engine and not allow_mirror:
        raise SystemExit(
            f"[X] engine and opponent are both '{opponent}'. An engine "
            f"predicts its own replies far better than anyone else's, so a "
            f"mirror adopts a much larger subtree and understates re-rooting "
            f"cost -- measured at 4.3 ms of overhead p99, enough to turn a "
            f"FAIL into a PASS. Pass --opponent <other> or, if a mirror is "
            f"genuinely what you want, --allow-mirror.")
    return opponent


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
    ap.add_argument("--opponent", default=None,
                    help="engine to play against. Defaults to a DIFFERENT "
                         "network -- never a mirror, which understates "
                         "re-rooting cost by ~4 ms of overhead p99")
    ap.add_argument("--allow-mirror", action="store_true",
                    help="permit engine == opponent. The result is not a "
                         "deployment latency measurement; see check 0")
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

    # The score is not the measurement -- 10 games cannot resolve strength and
    # this gate never claims to. What is measured is what the engine COSTS over
    # realistic positions, against an opponent it cannot predict.
    opponent = choose_opponent(args.engine, args.opponent, args.allow_mirror)

    pa = TimedPlayer(f"engine:{args.engine}", device)
    pb = TimedPlayer(f"engine:{opponent}", device)
    if opponent == args.engine:
        pb.name = pa.name + "-mirror"
    print(f"gating engine:{args.engine}  fingerprint "
          f"{pa.provenance['fingerprint']}  "
          f"({pa.net_info['params']:,} params)")
    print(f"opponent engine:{opponent}  ({pb.net_info['params']:,} params)"
          + ("   [!] MIRROR -- not a deployment measurement"
             if opponent == args.engine else ""))
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

    # Check 1 says whether the requirement was met. This says WHY, before it
    # stops being met. The search holds its own deadline to within a
    # millisecond, so a p99 miss is never search overrun -- it is the work the
    # caller waits for and the search does not time: re-rooting and releasing
    # the discarded tree. That is what the reserve is held back for, so
    # `overhead p99 <= reserve` is the same statement as check 1, expressed as
    # something actionable.
    over_p99 = rep["overhead_ms"]["p99"]
    reserve = pa.mcts.reserve_ms
    g.check("reserve covers caller-side work", over_p99 <= reserve,
            f"{over_p99:.2f}", f"<= {reserve:.0f}",
            f"re-root + release outside the search's deadline; "
            f"{reserve - over_p99:+.2f} ms of margin. This is the check that "
            f"fires FIRST -- p99 misses are caused here, not in the search")

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
        json.dump({"engine": args.engine, "opponent": opponent,
                   "mirror": opponent == args.engine, "games": args.games,
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
