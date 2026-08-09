"""Stage 1 of #41: what the CUDA graph does to throughput, before any strength.

TWO SEPARATE QUESTIONS, MEASURED SEPARATELY, because one workload cannot answer
both honestly.

  probe   IDENTICAL POSITIONS. Both arms search the same fixed set of real
          positions for the full deadline. Nothing diverges, so sims/move and
          nn-evals/move are strictly comparable. What it does not carry is
          tree reuse -- every search starts from a bare root -- so it is a
          throughput measurement, not a picture of deployment.
  match   REAL GAMES against the same opponent from the same paired openings,
          with tree reuse on. Latency percentiles and the frozen p99
          requirement are read off this, because a requirement written against
          deployment has to be measured in deployment. Positions DO diverge
          the moment the two arms choose differently, which is unavoidable and
          is why the throughput numbers are taken from `probe`.

THE QUESTION THIS IS ASKING. `RESULT_KERNEL_TRACE.md` predicted about 195 ms per
move recoverable. The interesting outcome is not whether the wave gets faster
-- that was already measured on a bench -- but whether the recovered time turns
into additional useful search. If sims/move rises by far less than the wave
speedup implies, the search's own bookkeeping is absorbing it, and that is a
result worth having rather than a disappointment.

NO STRENGTH CLAIM IS MADE HERE. Throughput has already failed to predict
strength once on this code (RESULT_MODEL_SIZE). Stage 2 is a separate match.

    python -m tools.graph_ab --engine pocket_r35 --games 8 --positions 60
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from tools import engine_registry
from tools.arena_1s import (TimedPlayer, latency_report, play_match,
                            phase_of)
from tools.runlock import gpu_baseline, single_instance

OUT_DIR = os.path.join("results", "graph_ab")
AB_SEED = engine_registry.SEEDS["confirm"]

# Pre-registered from RESULT_KERNEL_TRACE.md, so stage 1 checks a prediction
# rather than describing whatever it happens to see.
PREDICTED_MS_RAW = 279.3          # measured bench saving, per move
PREDICTED_MS_DISCOUNTED = 195.5   # after the 30% efficiency discount
SEARCH_MS = 913.0                 # search time in a 1000 ms move


def build(engine, graph, device):
    """`graph` is ignored when `engine` is a registry name that already pins it.

    Both arms are named registry engines -- `pocket_r35` and `pocket_graph` --
    rather than a base plus an ad-hoc override, because the candidate differs
    in TWO declared keys, not one: the graph flag and the larger reserve the
    graph forces. Comparing `pocket_r35+graph=1` against `pocket_r35` would
    have measured a candidate that cannot ship, since that arm keeps a 35 ms
    reserve against a 30.9 ms overhead p99.
    """
    return TimedPlayer("engine:%s" % engine, device)


def sample_positions(engine, games, device, seed):
    """Real positions from real play, spread across the game.

    Taken from a match the graph is not involved in, so both arms see the same
    inputs and neither had a hand in choosing them.
    """
    pa = build(engine, False, device)
    pb = build(engine, False, device)
    seen = []

    orig = pa.move

    def rec(state, move_num):
        seen.append((state.clone(), phase_of(int(np.count_nonzero(state.board)))))
        return orig(state, move_num)

    pa.move = rec
    play_match(pa, pb, games, seed, warmup=1, gc_mode="deferred")
    return seen


def probe_arm(engine, graph, device, positions, reps=1):
    """Search each fixed position for the full deadline, from a bare root."""
    p = build(engine, graph, device)
    params = p.net_info["params"]
    rows = []
    for state, phase in positions:
        for _ in range(reps):
            p.mcts.reset_stats()
            t0 = time.perf_counter()
            with torch.no_grad():
                pi, root = p.mcts.search(state.clone())
            dt = (time.perf_counter() - t0) * 1000.0
            last = p.mcts.last
            rows.append({
                "phase": phase,
                "search_ms": dt,
                "sims": last["simulations_completed"],
                "nn": last["neural_evaluations"],
                "expansions": last["nodes_expanded"],
                "root_n": last["root_n"],
                "argmax": int(np.argmax(pi)),
            })
    graphed = p.mcts.graph_wave
    return {
        "engine": engine,
        "params": params,
        "graph": bool(graph),
        "graph_replays": graphed.replays if graphed else 0,
        "positions": len(positions),
        "rows": rows,
        "search_ms": float(np.mean([r["search_ms"] for r in rows])),
        "sims_per_move": float(np.mean([r["sims"] for r in rows])),
        "nn_per_move": float(np.mean([r["nn"] for r in rows])),
        "expansions_per_move": float(np.mean([r["expansions"] for r in rows])),
        "root_visits_per_move": float(np.mean([r["root_n"] for r in rows])),
        "wave_ms": float(np.mean([r["search_ms"] * 8.0 / max(1, r["nn"])
                                  for r in rows])),
        "fingerprint": (p.provenance or {}).get("fingerprint"),
    }


def match_arm(path):
    """The match half, read from a `tools.regress_engine` run.

    Not replayed here. That tool IS the frozen latency gate -- same opponent
    rule, same seed, same warmup, same deferred GC -- and re-implementing the
    match in this file would produce a second, unblessed latency number that
    could disagree with the one that actually decides the requirement. It
    already did: the first version of this tool defaulted to a 2000 ms
    opponent and failed the INCUMBENT at p99 1011 ms, which the gate passes at
    986.2.
    """
    with open(path) as fh:
        d = json.load(fh)
    rep = dict(d["report"])
    rep["engine"] = d["engine"]
    rep["opponent"] = d["opponent"]
    rep["gate_pass"] = d["PASS"]
    rep["fingerprint"] = d["config"]["fingerprint"]
    rep["reserve_ms"] = d["config"]["reserve_ms"]
    rep["source"] = path
    return rep


def pct_change(a, b):
    return 100.0 * (b - a) / a if a else float("nan")


def report_arms(arms):
    """N engines on ONE set of positions -- for the model-size question.

    `RESULT_MODEL_SIZE` concluded that a 39x parameter cut buys only 1.24x more
    search, and that conclusion was measured on an engine spending most of a
    wave issuing tiny CUDA operations. Removing the dispatch changes what a
    parameter costs, so the ratio has to be re-measured rather than carried
    forward.
    """
    print()
    print("=" * 74)
    print("ARMS ON IDENTICAL POSITIONS")
    print("=" * 74)
    print("  %-26s %10s %12s %12s" % ("engine", "params", "nn/move",
                                      "sims/move"))
    for a in arms:
        print("  %-26s %10s %12.1f %12.1f"
              % (a["engine"], "{:,}".format(a["params"]),
                 a["nn_per_move"], a["sims_per_move"]))
    print()
    by = {a["engine"]: a for a in arms}

    def ratio(small, big, label):
        if small in by and big in by and by[big]["nn_per_move"]:
            r = by[small]["nn_per_move"] / by[big]["nn_per_move"]
            print("  %-38s %.3fx" % (label, r))
            return r
        return None

    return {
        "eager": ratio("pocket_r35", "final",
                       "172k / 6.77M search, graph OFF"),
        "graphed": ratio("pocket_graph", "final+graph=1",
                         "172k / 6.77M search, graph ON"),
    }


def report_probe(off, on):
    print()
    print("=" * 74)
    print("STAGE 1a -- IDENTICAL POSITIONS, %d of them, bare root each time"
          % off["positions"])
    print("=" * 74)
    rows = [
        ("search ms/move", off["search_ms"], on["search_ms"]),
        ("sims/move", off["sims_per_move"], on["sims_per_move"]),
        ("nn-evals/move", off["nn_per_move"], on["nn_per_move"]),
        ("expansions/move", off["expansions_per_move"],
         on["expansions_per_move"]),
        ("root visits/move", off["root_visits_per_move"],
         on["root_visits_per_move"]),
        ("wave ms (nn/8)", off["wave_ms"], on["wave_ms"]),
    ]
    print("  %-22s %12s %12s %10s" % ("", "graph off", "graph on", "change"))
    for name, a, b in rows:
        print("  %-22s %12.2f %12.2f %9.1f%%" % (name, a, b, pct_change(a, b)))
    print()
    print("  graph replays: %d" % on["graph_replays"])
    if off["graph_replays"]:
        print("  [X] the OFF arm replayed a graph -- the arms are not "
              "distinct")


def report_match(off, on, requirement):
    print()
    print("=" * 74)
    print("STAGE 1b -- REAL GAMES from the frozen gate, vs %s"
          % off["opponent"])
    print("=" * 74)
    print("  %s (reserve %.0f ms)  ->  %s (reserve %.0f ms)"
          % (off["engine"], off["reserve_ms"], on["engine"], on["reserve_ms"]))
    print("  sources: %s | %s" % (off["source"], on["source"]))
    for key, sub in (("latency_ms", ("p50", "p95", "p99", "max")),
                     ("overhead_ms", ("mean", "p99", "max")),
                     ("per_move", ("simulations", "neural_evaluations",
                                   "nodes_expanded",
                                   "inherited_simulations"))):
        print("  %s" % key)
        for k in sub:
            a = off.get(key, {}).get(k)
            b = on.get(key, {}).get(k)
            if a is None or b is None:
                continue
            print("    %-22s %12.2f %12.2f %9.1f%%"
                  % (k, a, b, pct_change(a, b)))
    print()
    for label, rep in (("graph off", off), ("graph on", on)):
        p99 = rep["latency_ms"]["p99"]
        mx = rep["latency_ms"]["max"]
        ok = p99 <= requirement["p99_ms"] and mx <= requirement["max_ms"]
        print("  %-10s p99 %8.2f (<= %.0f)  max %8.2f (<= %.0f)  %s"
              % (label, p99, requirement["p99_ms"], mx,
                 requirement["max_ms"], "PASS" if ok else "*** FAIL ***"))
    # A p99 miss here is never search overrun -- the search holds its own
    # deadline to within a millisecond. It is the work the caller waits for
    # OUTSIDE that deadline: re-rooting and releasing the discarded tree. A
    # bigger tree costs more of both, which is precisely what more search buys.
    # So the reserve has to be sized to the arm, exactly as pocket -> pocket_r35
    # was. This prints the size it needs.
    print()
    print("  reserve needed = overhead p99 + worst-chunk p99:")
    for label, rep in (("graph off", off), ("graph on", on)):
        need = rep["overhead_ms"]["p99"] + rep["worst_chunk_ms"]["p99"]
        print("    %-10s %6.2f + %5.2f = %6.2f ms" %
              (label, rep["overhead_ms"]["p99"],
               rep["worst_chunk_ms"]["p99"], need))


def report_verdict(probe_off, probe_on, match_on, requirement):
    print()
    print("=" * 74)
    print("STAGE 1 VERDICT")
    print("=" * 74)
    gain = pct_change(probe_off["nn_per_move"], probe_on["nn_per_move"])
    # Against the PRE-REGISTERED prediction, not against the wave time. The
    # first draft divided the nn gain by the change in `wave_ms`, but wave_ms
    # is defined as search_ms/nn -- the two are algebraically the same
    # measurement and their ratio is near 1 by construction, which is not a
    # finding. What can be checked is #40's prediction: the device sequence
    # costs 279.3 ms of a ~913 ms search (195.5 discounted), so removing it
    # should buy 913/(913-x) more waves.
    for label, saved in (("as measured on the bench", PREDICTED_MS_RAW),
                         ("after the 30%% discount", PREDICTED_MS_DISCOUNTED)):
        want = 100.0 * (SEARCH_MS / (SEARCH_MS - saved) - 1.0)
        print("  predicted %+6.1f%% (%s: %.1f ms/move of %.0f)"
              % (want, label, saved, SEARCH_MS))
    print("  MEASURED  %+6.1f%% network evaluations per move, identical "
          "positions" % gain)
    print("  MEASURED  %+6.1f%% in real games with tree reuse"
          % pct_change(
              (match_on.get("_off_nn") or float("nan")),
              match_on["per_move"]["neural_evaluations"]))
    p99 = match_on["latency_ms"]["p99"]
    mx = match_on["latency_ms"]["max"]
    lat_ok = p99 <= requirement["p99_ms"] and mx <= requirement["max_ms"]
    print("  latency requirement with the graph on: %s"
          % ("PASS" if lat_ok else "FAIL -- stage 2 must not run"))
    print()
    print("  Throughput is NOT strength. Stage 2 is a separate paired match")
    print("  at equal wall clock, and it is the only thing that can promote.")
    return {"nn_per_move_pct_identical_positions": gain,
            "nn_per_move_pct_real_games": pct_change(
                match_on.get("_off_nn") or float("nan"),
                match_on["per_move"]["neural_evaluations"]),
            "predicted_pct_raw": 100.0 * (
                SEARCH_MS / (SEARCH_MS - PREDICTED_MS_RAW) - 1.0),
            "predicted_pct_discounted": 100.0 * (
                SEARCH_MS / (SEARCH_MS - PREDICTED_MS_DISCOUNTED) - 1.0),
            "latency_pass": bool(lat_ok)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="pocket_r35")
    ap.add_argument("--candidate", default="pocket_graph")
    ap.add_argument("--match-baseline",
                    default=os.path.join("results", "arena_1s",
                                         "graph-control.json"))
    ap.add_argument("--match-candidate",
                    default=os.path.join("results", "arena_1s",
                                         "graph-candidate.json"))
    # `final` is what tools/regress_engine picks for this checkpoint, and the
    # latency requirement has to be measured under the conditions the gate uses.
    # The first run of this tool used anchor_C at 2000 ms; that opponent builds
    # a far larger tree, and under deferred GC the collections it produces
    # pushed worst-chunk p99 to 90 ms and failed BOTH arms -- including the
    # frozen incumbent, which passes the real gate. A control arm that fails is
    # a broken harness, not a finding.
    ap.add_argument("--opponent", default="final")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--position-games", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=AB_SEED)
    ap.add_argument("--arms", default="",
                    help="comma-separated engine specs to probe on one set of "
                         "positions instead of running the two-arm A/B; for "
                         "the model-size re-test (#43)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "graph_ab.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[X] the graph path needs CUDA")
    os.makedirs(OUT_DIR, exist_ok=True)

    with single_instance("graph_ab"):
        base = gpu_baseline()
        if base:
            print("[..] GPU load before starting: %.0f%% mean, %.0f%% peak"
                  % (base["mean_pct"], base["max_pct"]))

        print("[..] sampling positions from %d games" % args.position_games)
        pool = sample_positions(args.baseline, args.position_games,
                                args.device, args.seed)
        step = max(1, len(pool) // args.positions)
        positions = pool[::step][:args.positions]
        print("[..] %d positions of %d sampled" % (len(positions), len(pool)))

        if args.arms:
            names = [s.strip() for s in args.arms.split(",") if s.strip()]
            arms = []
            for n in names:
                print("[..] probe: %s" % n)
                arms.append(probe_arm(n, "graph=1" in n, args.device,
                                      positions))
            ratios = report_arms(arms)
            for a in arms:
                a.pop("rows", None)
            with open(args.out, "w") as fh:
                json.dump({"arms": arms, "ratios": ratios,
                           "positions": len(positions),
                           "seed": args.seed,
                           "git_head": engine_registry.git_head(),
                           "gpu_baseline_before_run": base}, fh, indent=2,
                          default=str)
            print("\n[OK] wrote %s" % args.out)
            return

        print("[..] probe: %s" % args.baseline)
        p_off = probe_arm(args.baseline, False, args.device, positions)
        print("[..] probe: %s" % args.candidate)
        p_on = probe_arm(args.candidate, True, args.device, positions)
        report_probe(p_off, p_on)

        m_off = match_arm(args.match_baseline)
        m_on = match_arm(args.match_candidate)
        req = engine_registry.REQUIREMENT
        report_match(m_off, m_on, req)
        m_on["_off_nn"] = m_off["per_move"]["neural_evaluations"]
        verdict = report_verdict(p_off, p_on, m_on, req)

        # Rows are bulky and only the aggregates are read; keep the file small.
        for d in (p_off, p_on):
            d.pop("rows", None)
        res = {
            "baseline": args.baseline, "candidate": args.candidate,
            "opponent": m_off["opponent"], "seed": args.seed,
            "positions": len(positions),
            "git_head": engine_registry.git_head(),
            "environment": engine_registry.environment(),
            "gpu_baseline_before_run": base,
            "probe": {"off": p_off, "on": p_on},
            "match": {"off": m_off, "on": m_on},
            "verdict": verdict,
        }
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print("\n[OK] wrote %s" % args.out)


if __name__ == "__main__":
    main()
