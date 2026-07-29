"""The 1,000 ms arena -- playing strength under a fixed move deadline.

THE OBJECTIVE THIS MEASURES. The product is not the raw network and not the
teacher-imitation quality of a student. It is the complete network-plus-search
agent returning a move inside a deadline. So the only promotion criterion here
is win rate at an IDENTICAL wall-clock budget, and every cost metric exists to
explain a strength number rather than to stand in for one.

    A change that raises simulations per second and does not raise strength is
    not an improvement. It is not promoted.

FROZEN OPERATIONAL REQUIREMENT. Fixed before any implementation was compared,
so no later result can move the bar:

    p99 move latency  <=  1000 ms      <- the requirement
    max move latency  <=  1250 ms      <- reported, and a hard reject

The strict p99 is the requirement because a mean is the wrong statistic for a
deadline: a player averaging 900 ms that occasionally takes 1.8 s has not met
the objective, and the mean cannot see that. The absolute cap is carried
alongside because p99 over a few thousand moves still permits a handful of
arbitrarily long ones, and a single 3 s move is a visible product failure.

REFERENCE HARDWARE is this box (CUDA). The browser port is a separate question
and is deliberately out of scope; latency there is a different measurement on
different silicon, and mixing the two would make neither number mean anything.

WHY THE CLOCK, NOT A SIMULATION COUNT. Under a deadline, simulations completed
is an OUTPUT. A bigger network that is stronger per simulation can be weaker per
second, and a fixed-simulation benchmark cannot see that trade at all -- it is
the single most likely way to pick the wrong model for deployment.

DETERMINISM, HONESTLY. Openings, colours, and per-game RNG are fixed by seed and
students play at temperature 0, so the only nondeterminism is machine timing
jitter deciding how many simulations fit. That is inherent to the thing being
measured, not a defect to engineer away, and it is why `simulations_completed`
is recorded for every move: a match can be replayed exactly by pinning those
counts. Latency is therefore reported as a distribution, never as one number.

    python -m tools.arena_1s --mode bench  --games 40
    python -m tools.arena_1s --mode h2h    --games 200 \
        --player-a "ms=1000,reuse=1" --player-b "ms=1000,reuse=0"
    python -m tools.arena_1s --mode anchor --games 100 --anchors gregory_d4
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time

import numpy as np
import torch

from agents.agent_base import ModelConfigCNN
from agents.deterministics import WinBlockAgent
from agents.gregory import GregoryAgent
from agents.mcts import MCTS, TreeReuseSearcher
from agents.neural_net_agent_3 import ConvNet
from agents.random_agent import RandomAgent
from engine.constants import DRAW, O, X
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from scripts.expert_iter import _agent_fn, _eval_openings
from scripts.train_alphazero import NETWORK_CONFIGS
from scripts.train_student_offline import AB_ARCHS
from tools import provenance
from tools.analyze_search_disagreement import PHASE_BANDS
from tools.teacher_sim_ladder import outcome_ci

# Frozen 2026-07-28, before the first implementation comparison.
REQUIREMENT = {
    "budget_ms": 1000,
    "p99_ms": 1000.0,
    "max_ms": 1250.0,
    "frozen": "2026-07-28, before any candidate was benchmarked",
}

# Own namespace, so an arena run can never silently share an opening set with
# the distillation pilot (9901) or the teacher sim ladder (77xx/78xx).
ARENA_BASE_SEED = 6100
ANCHOR_SEEDS = {"random": 6141, "winblock": 6142,
                "gregory_d3": 6143, "gregory_d4": 6144}

DEFAULT_CKPT = "models/expert_iter_v2/teacher.pt"


def load_net(ckpt, arch, device):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    tanh = payload.get("value_tanh", True)
    sd = payload.get("state_dict", payload)
    shape = NETWORK_CONFIGS[arch] if arch in NETWORK_CONFIGS else AB_ARCHS[arch]
    cfg = ModelConfigCNN(value_tanh=tanh, model_dir="models/_arena", **shape)
    model = ConvNet(cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, {
        "checkpoint": ckpt, "arch": arch, "value_tanh": bool(tanh),
        "gen": payload.get("gen", "?"),
        "params": sum(p.numel() for p in model.parameters()),
    }


def parse_spec(spec):
    """'ms=1000,wave=8,reuse=1' -> a dict of typed player options."""
    out = {}
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        if "=" not in part:
            raise SystemExit(f"[X] player option '{part}' is not key=value")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    unknown = set(out) - {"ckpt", "arch", "ms", "sims", "wave", "cpuct",
                          "reuse", "solve", "maxsims", "name", "reserve",
                          "bexp"}
    if unknown:
        raise SystemExit(f"[X] unknown player options: {sorted(unknown)}")
    return out


class TimedPlayer:
    """A network plus search, under a move deadline, recording what it cost."""

    def __init__(self, spec, device, count_nodes=False):
        o = parse_spec(spec)
        self.ckpt = o.get("ckpt", DEFAULT_CKPT)
        self.arch = o.get("arch", "arena22")
        # ms and sims are mutually exclusive: one is a deadline, the other is a
        # fixed budget, and a player cannot be under both at once.
        if "ms" in o and "sims" in o:
            raise SystemExit("[X] give a player ms= or sims=, not both")
        self.budget_ms = float(o["ms"]) if "ms" in o else None
        self.n_sims = int(o.get("sims", 800))
        self.wave = int(o.get("wave", 8))
        self.c_puct = float(o.get("cpuct", 1.5))
        self.reuse = o.get("reuse", "0") == "1"
        self.solve = o.get("solve", "1") == "1"
        self.max_sims = int(o.get("maxsims", 200_000))
        self.reserve = float(o["reserve"]) if "reserve" in o else None
        # bexp=0 forces the old per-leaf expansion, so the batching change can
        # be judged by STRENGTH at equal wall clock rather than by its own
        # throughput figure.
        self.bexp = (o["bexp"] == "1") if "bexp" in o else None

        self.model, self.net_info = load_net(self.ckpt, self.arch, device)
        self.mcts = MCTS(self.model, device, n_sims=self.n_sims,
                         c_puct=self.c_puct, add_dirichlet_at_root=False,
                         wave_size=self.wave, solve=self.solve,
                         time_budget_ms=self.budget_ms,
                         max_sims=self.max_sims, reserve_ms=self.reserve,
                         batched_expand=self.bexp)
        self.searcher = TreeReuseSearcher(self.mcts, enabled=self.reuse,
                                          count_nodes=count_nodes)
        self.name = o.get("name") or self._auto_name()
        self.records = []
        self.policies = []
        self.recording = True

    def _auto_name(self):
        budget = (f"{self.budget_ms:.0f}ms" if self.budget_ms
                  else f"{self.n_sims}sims")
        bits = [self.arch, budget, f"w{self.wave}"]
        if self.reuse:
            bits.append("reuse")
        if self.solve:
            bits.append("solve")
        if self.mcts.batched_expand:
            bits.append("bexp")
        return "-".join(bits)

    def config(self):
        return {"name": self.name, "budget_ms": self.budget_ms,
                "batched_expand": self.mcts.batched_expand,
                "reserve_ms": self.mcts.reserve_ms,
                "n_sims": None if self.budget_ms else self.n_sims,
                "wave_size": self.wave, "c_puct": self.c_puct,
                "tree_reuse": self.reuse, "solve": self.solve,
                "max_sims": self.max_sims, **self.net_info}

    def new_game(self):
        self.searcher.reset()

    def reset_counters(self):
        """Drop everything accumulated so far. Used to discard the warmup."""
        self.records.clear()
        self.policies.clear()
        self.mcts.reset_stats()
        self.searcher.reset_stats()

    def move(self, state, move_num):
        # OUTER timer. The requirement is about the move the caller waits for,
        # which includes re-rooting and any instrumentation -- not just the
        # interval MCTS.search times itself over. Reporting the inner number
        # against a deadline would quietly exclude work the deadline covers.
        t0 = time.perf_counter()
        pi, _root = self.searcher.search(state)
        mv = int(pi.argmax())
        move_ms = (time.perf_counter() - t0) * 1000.0
        if self.recording:
            rec = self.mcts.last
            self.records.append((
                move_ms, rec["elapsed_ms"], rec["simulations_completed"],
                rec["neural_evaluations"], rec["nodes_expanded"],
                rec["tree_nodes_reused"] or 0, rec["inherited_simulations"],
                rec["transposition_hits"], mv,
                int(np.count_nonzero(state.board)), move_num,
                rec["worst_chunk_ms"],
            ))
            self.policies.append(pi.copy())
        return mv


class AnchorPlayer:
    """A fixed external opponent. Timed too, so the arena's own overhead and
    the anchor's cost are visible rather than folded into the match total."""

    def __init__(self, name, agent):
        self.name = name
        self.fn = _agent_fn(agent)
        self.records = []
        self.policies = []
        self.recording = True
        self.searcher = None

    def config(self):
        return {"name": self.name, "kind": "anchor"}

    def new_game(self):
        pass

    def reset_counters(self):
        self.records.clear()

    def move(self, state, move_num):
        t0 = time.perf_counter()
        mv = self.fn(state, move_num)
        ms = (time.perf_counter() - t0) * 1000.0
        if self.recording:
            self.records.append((ms, ms, 0, 0, 0, 0, 0, 0, int(mv),
                                 int(np.count_nonzero(state.board)), move_num,
                                 0.0))
        return int(mv)


REC_COLS = ("move_ms", "search_ms", "simulations_completed",
            "neural_evaluations", "nodes_expanded", "tree_nodes_reused",
            "inherited_simulations", "transposition_hits", "chosen_move",
            "filled", "move_num", "worst_chunk_ms")


def play_match(pa, pb, n_games, seed, warmup=0, gc_mode="deferred"):
    """Paired openings, colours swapped, per-game reseed.

    Line-for-line the same game protocol as
    tools.teacher_sim_ladder.play_match_detailed -- same `_eval_openings`, same
    per-game seed, same swap -- so an arena score is comparable with every
    result already published from that harness. The two additions are the
    per-game tree reset (a searcher must never carry a tree between games) and
    the warmup, whose moves are played for real and then discarded: the first
    CUDA forward passes pay one-off autotune and allocator costs that would
    otherwise land entirely in the p99 the requirement is written against.
    """
    total = n_games + warmup
    openings = _eval_openings(total, seed)
    py_state, np_state = random.getstate(), np.random.get_state()
    outcomes = []
    played = 0
    try:
        for opening_idx, opening in enumerate(openings):
            for a_side in (X, O):
                if played >= total:
                    break
                is_warmup = played < warmup
                for p in (pa, pb):
                    p.new_game()
                    if not is_warmup and p.recording is False:
                        # First real game: throw away everything the warmup
                        # accumulated, including the cumulative MCTS counters
                        # and the reuse tallies. Gating only the record list
                        # would leave warmup misses in the reuse rate.
                        p.reset_counters()
                    p.recording = not is_warmup
                game_seed = seed + opening_idx * 2 + (0 if a_side == X else 1)
                random.seed(game_seed)
                np.random.seed(game_seed & 0xFFFFFFFF)
                state = GameState()
                for mv in opening:
                    ok, _ = state.make_move(mv)
                    if not ok:
                        raise RuntimeError(f"illegal opening move {mv}")
                move_num = len(opening)
                while not state.is_over():
                    p = pa if state.player == a_side else pb
                    mv = p.move(state, move_num)
                    legal = rule_utl_valid_moves(state.board, state.last_move,
                                                 state.mini_winners)
                    if mv not in legal:
                        raise RuntimeError(f"{p.name} returned illegal move {mv}")
                    state.make_move(mv)
                    move_num += 1
                played += 1
                # Collect between GAMES, never between moves: in deployment the
                # opponent's turn is genuinely free time, but in this harness
                # both players share one process, so a collect between moves
                # would land inside the other player's budget and corrupt its
                # latency. A game boundary belongs to neither.
                if gc_mode == "deferred":
                    gc.collect()
                if is_warmup:
                    continue
                outcomes.append(1.0 if state.winner == a_side else
                                0.5 if state.winner == DRAW else 0.0)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        for p in (pa, pb):
            p.recording = True
    return outcomes


def phase_of(filled):
    for name, lo, hi in PHASE_BANDS:
        if lo <= filled <= hi:
            return name
    return "late"


def latency_report(player):
    """Percentiles, throughput, and the frozen-requirement verdict."""
    if not player.records:
        return {"moves": 0}
    arr = np.array(player.records, dtype=np.float64)
    col = {k: arr[:, i] for i, k in enumerate(REC_COLS)}
    ms = col["move_ms"]
    sec = ms / 1000.0
    rep = {
        "moves": int(arr.shape[0]),
        # Everything the caller waits for that the search did not time itself
        # over: re-rooting, instrumentation, the argmax. Reported so a p99 miss
        # can be attributed to search or to the wrapper around it.
        "overhead_ms": {
            "mean": float((ms - col["search_ms"]).mean()),
            "max": float((ms - col["search_ms"]).max()),
        },
        # A chunk is atomic, so its duration is the floor on how far a search
        # can overrun the deadline no matter how good the predictor is.
        "worst_chunk_ms": {
            "mean": float(col["worst_chunk_ms"].mean()),
            "p99": float(np.percentile(col["worst_chunk_ms"], 99)),
            "max": float(col["worst_chunk_ms"].max()),
        },
        "latency_ms": {
            "mean": float(ms.mean()),
            "p50": float(np.percentile(ms, 50)),
            "p95": float(np.percentile(ms, 95)),
            "p99": float(np.percentile(ms, 99)),
            "max": float(ms.max()),
        },
        "per_move": {
            "simulations": float(col["simulations_completed"].mean()),
            "neural_evaluations": float(col["neural_evaluations"].mean()),
            "nodes_expanded": float(col["nodes_expanded"].mean()),
            "inherited_simulations": float(col["inherited_simulations"].mean()),
            "transposition_hits": float(col["transposition_hits"].mean()),
        },
        "throughput": {
            "simulations_per_second":
                float(col["simulations_completed"].sum() / max(sec.sum(), 1e-9)),
            "neural_evals_per_second":
                float(col["neural_evaluations"].sum() / max(sec.sum(), 1e-9)),
        },
    }
    budget = player.config().get("budget_ms")
    if budget:
        over = ms > REQUIREMENT["p99_ms"]
        rep["requirement"] = {
            "p99_ms": rep["latency_ms"]["p99"],
            "p99_pass": rep["latency_ms"]["p99"] <= REQUIREMENT["p99_ms"],
            "max_ms": rep["latency_ms"]["max"],
            "max_pass": rep["latency_ms"]["max"] <= REQUIREMENT["max_ms"],
            "moves_over_budget": int(over.sum()),
            "share_over_budget": float(over.mean()),
        }
        rep["requirement"]["PASS"] = bool(rep["requirement"]["p99_pass"]
                                          and rep["requirement"]["max_pass"])
    # "How many simulations does it actually achieve in one second" is a
    # per-phase question: the branching factor collapses toward the end of a
    # game, so one average would describe no real position.
    by_phase = {}
    phases = np.array([phase_of(int(f)) for f in col["filled"]])
    for name, _lo, _hi in PHASE_BANDS:
        m = phases == name
        if not m.any():
            continue
        by_phase[name] = {
            "moves": int(m.sum()),
            "simulations": float(col["simulations_completed"][m].mean()),
            "neural_evaluations": float(col["neural_evaluations"][m].mean()),
            "latency_p50_ms": float(np.percentile(ms[m], 50)),
            "latency_p99_ms": float(np.percentile(ms[m], 99)),
        }
    rep["by_phase"] = by_phase
    if player.searcher is not None:
        rep["tree_reuse"] = player.searcher.stats()
        m = player.mcts
        # A proven root returns at once, so these moves cost almost nothing and
        # pull the mean latency down without being a speed improvement. Reported
        # so the latency distribution can be read honestly.
        rep["early_stop_rate"] = (m.stat_early_stops / m.stat_searches
                                  if m.stat_searches else 0.0)
    return rep


def print_report(player, rep):
    if not rep.get("moves"):
        print(f"  {player.name}: no moves recorded")
        return
    lat, per, thr = rep["latency_ms"], rep["per_move"], rep["throughput"]
    print(f"\n  {player.name}   {rep['moves']:,} moves")
    print(f"    latency ms   p50 {lat['p50']:7.1f}  p95 {lat['p95']:7.1f}  "
          f"p99 {lat['p99']:7.1f}  max {lat['max']:7.1f}  "
          f"mean {lat['mean']:7.1f}   "
          f"(non-search overhead mean {rep['overhead_ms']['mean']:.2f} ms, "
          f"max {rep['overhead_ms']['max']:.2f})")
    print(f"    per move     {per['simulations']:8.1f} sims  "
          f"{per['neural_evaluations']:8.1f} nn-evals  "
          f"{per['nodes_expanded']:8.1f} expanded  "
          f"{per['inherited_simulations']:8.1f} inherited")
    print(f"    throughput   {thr['simulations_per_second']:8.1f} sims/s  "
          f"{thr['neural_evals_per_second']:8.1f} nn-evals/s")
    if rep.get("early_stop_rate"):
        print(f"    early stop   {rep['early_stop_rate']:.3f} of moves returned "
              f"on a proven root (near-zero latency, not a speedup)")
    if "by_phase" in rep and rep["by_phase"]:
        cells = "  ".join(f"{k} {v['simulations']:.0f}"
                          for k, v in rep["by_phase"].items())
        print(f"    sims/move by phase   {cells}")
    if "tree_reuse" in rep and rep["tree_reuse"]["moves"]:
        t = rep["tree_reuse"]
        miss = ", ".join(f"{k} {v}" for k, v in t["miss_reason"].items() if v)
        ret = ("n/a" if t["node_retention"] is None
               else f"{t['node_retention']:.3f}")
        print(f"    tree reuse   rate {t['reuse_rate']:.3f}  "
              f"inherited {t['inherited_sims_per_move']:.1f} sims/move  "
              f"node retention {ret}")
        if miss:
            print(f"                 misses: {miss}")
    if "requirement" in rep:
        r = rep["requirement"]
        verdict = "PASS" if r["PASS"] else "FAIL"
        w = rep["worst_chunk_ms"]
        print(f"    worst chunk  mean {w['mean']:6.1f} ms  p99 {w['p99']:6.1f}  "
              f"max {w['max']:6.1f}   (atomic: the floor on any overrun)")
        print(f"    requirement  [{verdict}] p99 {r['p99_ms']:.1f} <= "
              f"{REQUIREMENT['p99_ms']:.0f} and max {r['max_ms']:.1f} <= "
              f"{REQUIREMENT['max_ms']:.0f}  "
              f"({r['moves_over_budget']} moves over budget)")


def save_records(path, players):
    arrays = {}
    for tag, p in players.items():
        if p.records:
            arrays[f"{tag}_records"] = np.array(p.records, dtype=np.float64)
        if p.policies:
            arrays[f"{tag}_root_policy"] = np.array(p.policies, dtype=np.float32)
    if not arrays:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays["columns"] = np.array(REC_COLS)
    np.savez_compressed(path, **arrays)
    return path


def build_anchor(name):
    return AnchorPlayer(name, {
        "random": lambda: RandomAgent(),
        "winblock": lambda: WinBlockAgent(),
        "gregory_d3": lambda: GregoryAgent(depth=3),
        "gregory_d4": lambda: GregoryAgent(depth=4),
    }[name]())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["bench", "h2h", "anchor"], default="bench")
    ap.add_argument("--player-a", default=f"ms={REQUIREMENT['budget_ms']}")
    ap.add_argument("--player-b", default=f"ms={REQUIREMENT['budget_ms']}")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--warmup-games", type=int, default=2,
                    help="played for real, then discarded -- the first CUDA "
                         "passes pay one-off costs that would otherwise land "
                         "in the p99")
    ap.add_argument("--anchors", nargs="+", default=["gregory_d4"],
                    choices=list(ANCHOR_SEEDS))
    ap.add_argument("--seed", type=int, default=ARENA_BASE_SEED)
    ap.add_argument("--gc", choices=["deferred", "auto", "off"],
                    default="deferred",
                    help="deferred (default): automatic cyclic collection off "
                         "during play, one explicit collect at each GAME "
                         "boundary. Safe because TreeReuseSearcher.release() "
                         "breaks the tree cycles itself, so refcounting "
                         "reclaims trees and the collect is only insurance "
                         "against third-party cycles. auto: CPython defaults, "
                         "whose gen-2 scans walk the whole live tree mid-chunk "
                         "and own the latency tail. off: never collect -- "
                         "diagnostic only, leaks any cycle we do not break.")
    ap.add_argument("--count-nodes", action="store_true",
                    help="two subtree walks per move; instrumentation only, "
                         "off during a timing run")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--outdir", default="results/arena_1s")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if args.gc != "auto":
        gc.disable()
        print(f"[!] automatic cyclic GC off (--gc {args.gc}); "
              f"{'collecting at game boundaries' if args.gc == 'deferred' else 'NEVER collecting -- diagnostic'}")

    payload = {"tag": args.tag, "mode": args.mode, "games": args.games,
               "warmup_games": args.warmup_games, "seed": args.seed,
               "device": args.device, "requirement": REQUIREMENT,
               "provenance": provenance.build()}

    pa = TimedPlayer(args.player_a, args.device, count_nodes=args.count_nodes)
    print(f"A: {pa.name}  ({pa.net_info['params']:,} params, "
          f"gen {pa.net_info['gen']})")

    if args.mode == "bench":
        # The same configuration on both sides. The score is 0.5 by
        # construction and is NOT the point: this measures latency, throughput
        # and per-phase simulation counts over realistic positions, which is
        # the baseline every later candidate is compared against.
        pb = TimedPlayer(args.player_a, args.device,
                         count_nodes=args.count_nodes)
        pb.name = pa.name + "-mirror"
        pairs = [(pa, pb, args.seed, "self")]
    elif args.mode == "h2h":
        pb = TimedPlayer(args.player_b, args.device,
                         count_nodes=args.count_nodes)
        print(f"B: {pb.name}  ({pb.net_info['params']:,} params, "
              f"gen {pb.net_info['gen']})")
        pairs = [(pa, pb, args.seed, "h2h")]
    else:
        pairs = [(pa, build_anchor(a), ANCHOR_SEEDS[a], a) for a in args.anchors]

    results = {}
    for a, b, seed, label in pairs:
        t0 = time.time()
        outcomes = play_match(a, b, args.games, seed, warmup=args.warmup_games,
                              gc_mode=args.gc)
        dt = time.time() - t0
        score, (lo, hi), se = outcome_ci(outcomes)
        w = sum(1 for o in outcomes if o == 1.0)
        d = sum(1 for o in outcomes if o == 0.5)
        ll = sum(1 for o in outcomes if o == 0.0)
        print(f"\n{label}: {a.name} vs {b.name}")
        print(f"  score for A  {score:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"W{w}/D{d}/L{ll}  n={len(outcomes)}  {dt / 60:.1f} min")
        ra, rb = latency_report(a), latency_report(b)
        print_report(a, ra)
        print_report(b, rb)
        results[label] = {
            "score_for_a": score, "ci95": [lo, hi], "se": se,
            "wins": w, "draws": d, "losses": ll, "outcomes": outcomes,
            "seed": seed, "seconds": dt,
            "player_a": {"config": a.config(), "report": ra},
            "player_b": {"config": b.config(), "report": rb},
        }
        rec = save_records(os.path.join(args.outdir,
                                        f"{args.tag}_{label}_moves.npz"),
                           {"a": a, "b": b})
        if rec:
            results[label]["move_records"] = rec

    if args.device == "cuda":
        payload["cuda_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        print(f"\n  CUDA peak allocated {payload['cuda_peak_mb']:.0f} MB")

    payload["results"] = results
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
