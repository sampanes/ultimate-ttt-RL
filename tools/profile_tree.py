"""Where the 1,000 ms goes inside the Python tree.

The case for looking here is now made from two independent directions, and
neither of them is a guess about what ought to be slow:

  * a 39x smaller network buys only 1.13x more simulations, so at most 12.7% of
    a simulation is the network and >=87% is everything around it;
  * network throughput DEGRADES 25% as the tree grows (3,329 -> 2,503
    nn-evals/s from 250 ms to 4,000 ms) even though the forward pass is
    identical -- the extra cost is per-simulation bookkeeping on longer paths.

Both larger costs this program already removed are gone (plane building, 46%
-> 3.6%; the per-leaf `_expand_from_logits` round trips, replaced by
`_expand_wave`), so the tree is what remains.

WHY SAMPLING, NOT cProfile. `_best_child` is called tens of thousands of times
per move. cProfile adds ~1-2 us per call, which lands as tens of percent of the
budget ON THE VERY FUNCTION being judged, and would manufacture the conclusion
that selection dominates. A sampling profiler pays a small uniform cost instead
and cannot bias one function against another. The price is that it gives
percentages rather than exact call counts, which is why `--mode count` exists
separately: exact counts from a short run whose TIMING is admittedly distorted,
crossed against a microbenchmark of each operation in isolation. Three methods,
different biases; agreement is the evidence, not any one table.

MCTS IS NOT EDITED. `agents/mcts.py` is frozen and its sha256 is a hard gate in
the registry, so this tool instruments from outside the process's own stack.
The per-line report is only meaningful against those exact bytes, so a source
mismatch is fatal here rather than a warning.

    python -m tools.profile_tree --mode all --engine final --games 12

Modes:
    sample      clean line-level profile of the real 1,000 ms workload
    count       exact operation counts per move (timing distorted, counts not)
    microbench  cost of each operation in isolation, no games
    all         all three, plus the reconciliation between them
"""

import argparse
import collections
import json
import linecache
import math
import os
import random
import statistics
import sys
import threading
import time

import numpy as np
import torch

from agents import mcts as mcts_mod
from agents.mcts import MCTS, MCTSNode, TreeReuseSearcher
from engine.constants import X, O
from engine.game import GameState
from engine.rules import rule_utl_valid_moves
from tools import engine_registry
from tools.arena_1s import TimedPlayer, phase_of, play_match

OUT_DIR = os.path.join("results", "profile_tree")

# Frozen namespace: a profiling run must not be able to leak into the opening
# set of a match anyone reads a score off. No score is ever read off these.
PROFILE_SEED = engine_registry.SEEDS["profile"]

# (file basename, function) -> the operation the owner named. Anything absent
# is reported under its own name rather than swept into an "other" bucket --
# an unexplained 8% is exactly the thing a profile exists to surface.
OPERATION = {
    ("mcts.py", "_best_child"):             "child scoring / best-child",
    ("mcts.py", "Q"):                       "child scoring / best-child",
    ("mcts.py", "U"):                       "child scoring / best-child",
    ("mcts.py", "<lambda>"):                "child scoring / best-child",
    ("mcts.py", "_run_wave"):               "selection traversal",
    ("mcts.py", "_run_sim"):                "selection traversal",
    ("mcts.py", "_clone"):                  "selection traversal",
    ("mcts.py", "_backup"):                 "backup traversal",
    ("mcts.py", "_expand_wave"):            "expansion",
    ("mcts.py", "_expand_from_logits"):     "expansion",
    ("mcts.py", "_expand"):                 "expansion",
    ("mcts.py", "__init__"):                "node allocation",
    ("mcts.py", "_mark_terminal_children"): "proof propagation",
    ("mcts.py", "_solve_from_children"):    "proof propagation",
    ("mcts.py", "_propagate_solved"):       "proof propagation",
    ("mcts.py", "_mark_solved"):            "proof propagation",
    ("mcts.py", "_note_proof"):             "proof propagation",
    ("mcts.py", "release"):                 "tree release",
    ("mcts.py", "_count"):                  "tree node counting",
    ("mcts.py", "_adopt"):                  "tree reuse: adopt",
    ("mcts.py", "_run_budget"):             "budget loop control",
    ("mcts.py", "search"):                  "search(): pi + record",
    ("mcts.py", "_finish_record"):          "search(): pi + record",
    ("mcts.py", "_begin_record"):           "search(): pi + record",
    ("rules.py", "rule_utl_valid_moves"):   "legal-child iteration",
    ("agent_base.py", "wave_planes"):       "plane build",
    ("agent_base.py", "board_to_tensor_from_gamestate"): "plane build",
    ("agent_base.py", "_has_fill_planes"):  "plane build",
    ("neural_net_agent_3.py", "forward_both"): "network forward",
    ("neural_net_agent_3.py", "forward"):      "network forward",
    ("neural_net_agent_3.py", "_forward_trunk"): "network forward",
}

# Functions whose interesting structure is INSIDE them -- one bucket would hide
# the split the port decision turns on (is `_expand_wave` the mask arithmetic,
# the node allocation loop, or the proof probes?).
LINE_SPLIT = ("_run_wave", "_expand_wave", "_best_child", "_backup")


def classify(filename, func):
    base = os.path.basename(filename)
    op = OPERATION.get((base, func))
    if op is not None:
        return op
    if base.startswith("arena_1s") or base.startswith("profile_tree"):
        return "harness"
    return "other: %s:%s" % (base, func)


def assert_frozen_sources():
    """Refuse to profile bytes the line numbers do not describe."""
    bad = []
    for path, want in engine_registry.ENGINE_SOURCES.items():
        got = engine_registry.sha256_file(path)
        if got != want:
            bad.append((path, want[:12], (got or "MISSING")[:12]))
    if bad:
        print("[X] engine sources have drifted from the frozen baseline.")
        for path, want, got in bad:
            print("    %-32s frozen %s  now %s" % (path, want, got))
        print("    A per-line profile of different bytes is not a profile of")
        print("    the frozen engine. Recover with:")
        print("      git worktree add ../uttt-anchor arena-1s-baseline")
        raise SystemExit(2)


# ----------------------------------------------------------------------
# Mode 1: sampling profiler
# ----------------------------------------------------------------------

class StackSampler(threading.Thread):
    """Snapshot the main thread's stack on a randomized tick.

    Cost is one `sys._current_frames()` plus a short frame walk per tick, paid
    by a second thread, so it is uniform across the code being measured rather
    than proportional to call count. The frame recorded while a C call is in
    flight is still the Python line that made the call, which is the wanted
    attribution: a `state.make_move` line should carry its C++ engine time.

    THE INTERVAL IS RANDOMIZED, and that is not a detail. This search is
    strictly periodic -- a wave of 8 selections, one forward pass, one
    expansion, repeat, hundreds of times a second -- and a fixed-period sampler
    aimed at a periodic workload aliases: it can land on the forward pass every
    time and report 100%, or never and report 0%. A uniform jitter of +/-50%
    around the mean interval destroys any such lock-in.

    ACHIEVED RATE IS NOT THE REQUESTED RATE. Windows sleeps in ~7-16 ms
    granules, so a 1,000 Hz request delivers ~130 Hz. That is harmless as long
    as nothing divides by the nominal figure -- every duration here is derived
    from `wall / samples`, which is exact whatever the OS did. The rate is
    reported so the reader can size the error bars rather than infer them.

    Its RNG is private. Drawing jitter from the `random` module would consume
    from the same Mersenne Twister that `play_match` reseeds per game, which
    would change the openings themselves -- a profiler that alters the workload
    it profiles.
    """

    def __init__(self, target_tid, ctx, hz=500):
        super().__init__(daemon=True, name="tree-profiler-sampler")
        self.tid = target_tid
        self.ctx = ctx
        self.hz = hz
        self.interval = 1.0 / hz
        self.leaf = collections.Counter()    # (phase, file, func, line) -> n
        self.pairs = collections.Counter()   # (leaf op, caller op) -> n
        self.samples = 0
        self._rng = random.Random(0xC0FFEE)
        self._halt = threading.Event()
        self.wall = 0.0

    def run(self):
        t_start = time.perf_counter()
        frames = sys._current_frames
        jitter = self._rng.uniform
        while not self._halt.is_set():
            time.sleep(self.interval * jitter(0.5, 1.5))
            f = frames().get(self.tid)
            if f is None:
                continue
            phase = self.ctx["phase"]
            co = f.f_code
            self.leaf[(phase, co.co_filename, co.co_name, f.f_lineno)] += 1
            back = f.f_back
            self.pairs[(classify(co.co_filename, co.co_name),
                        "-" if back is None
                        else classify(back.f_code.co_filename,
                                      back.f_code.co_name))] += 1
            self.samples += 1
        self.wall = time.perf_counter() - t_start

    def stop(self):
        self._halt.set()
        self.join(timeout=5.0)


def instrument_phase(player, ctx):
    """Tag every sample with the game stage the move belongs to."""
    orig = player.move

    def move(state, move_num):
        ctx["phase"] = phase_of(int(np.count_nonzero(state.board)))
        try:
            return orig(state, move_num)
        finally:
            ctx["phase"] = "outside"
    player.move = move


def run_sample(engine, games, hz, device, seed):
    ctx = {"phase": "outside"}
    pa = TimedPlayer("engine:%s" % engine, device)
    pb = TimedPlayer("engine:%s" % engine, device)
    for p in (pa, pb):
        instrument_phase(p, ctx)
    print("  engine %s  fingerprint %s  %s params"
          % (engine, (pa.provenance or {}).get("fingerprint"),
             format(pa.net_info["params"], ",")))
    print("  %d games of self-play at %s ms, sampling at %d Hz"
          % (games, pa.budget_ms, hz))

    sampler = StackSampler(threading.get_ident(), ctx, hz=hz)
    t0 = time.perf_counter()
    sampler.start()
    try:
        play_match(pa, pb, games, seed, warmup=2, gc_mode="deferred")
    finally:
        sampler.stop()
    wall = time.perf_counter() - t0

    moves = len(pa.records) + len(pb.records)
    sims = sum(r[2] for r in pa.records) + sum(r[2] for r in pb.records)
    in_search = sum(n for (ph, _f, _fn, _l), n in sampler.leaf.items()
                    if ph != "outside")
    # Self-calibrating: whatever the sampler actually achieved, one sample is
    # one (wall / samples) slice of real time. This is exact even when the tick
    # ran late, which is why the nominal Hz is never used as a divisor.
    per_sample_ms = (wall * 1000.0 / sampler.samples) if sampler.samples else 0.0

    return {
        "engine": engine,
        "fingerprint": (pa.provenance or {}).get("fingerprint"),
        "games": games, "moves": moves, "wall_s": wall,
        "sims_per_move": sims / moves if moves else 0.0,
        "samples": sampler.samples, "requested_hz": hz,
        "achieved_hz": sampler.samples / wall if wall else 0.0,
        "in_search_samples": in_search,
        "per_sample_ms": per_sample_ms,
        "ms_per_move": in_search * per_sample_ms / moves if moves else 0.0,
        "leaf": {"%s|%s|%s|%d" % k: v for k, v in sampler.leaf.items()},
        "pairs": {"%s <- %s" % k: v for k, v in sampler.pairs.items()},
    }


def report_sample(res, top_lines=8):
    leaf = collections.Counter()
    for key, n in res["leaf"].items():
        phase, filename, func, line = key.rsplit("|", 3)
        leaf[(phase, filename, func, int(line))] = n
    in_search = res["in_search_samples"]
    if not in_search:
        print("  [X] no in-search samples")
        return {}

    by_op = collections.Counter()
    by_op_phase = collections.defaultdict(collections.Counter)
    by_line = collections.defaultdict(collections.Counter)
    phase_totals = collections.Counter()
    for (phase, filename, func, line), n in leaf.items():
        if phase == "outside":
            continue
        op = classify(filename, func)
        by_op[op] += n
        by_op_phase[op][phase] += n
        phase_totals[phase] += n
        if func in LINE_SPLIT:
            by_line[func][(filename, line)] += n

    ms_move = res["ms_per_move"]
    print("")
    print("  %-34s %8s %10s %8s" % ("operation", "share", "ms/move", "samples"))
    print("  " + "-" * 64)
    for op, n in by_op.most_common():
        share = n / in_search
        print("  %-34s %7.1f%% %9.1f %8d"
              % (op[:34], 100.0 * share, share * ms_move, n))
    print("  " + "-" * 64)
    print("  %-34s %7.1f%% %9.1f %8d" % ("TOTAL in search", 100.0, ms_move,
                                         in_search))
    # One binomial standard error at a 10% share, so nobody reads a 1.5-point
    # gap between two buckets as a ranking.
    se = 100.0 * math.sqrt(0.10 * 0.90 / in_search)
    print("  one standard error at a 10%% share is %.2f points (%d samples)"
          % (se, in_search))

    phases = [p for p in ("early", "mid", "late") if phase_totals[p]]
    if len(phases) > 1:
        print("")
        print("  share of in-search time by game stage")
        print("  %-34s %s" % ("operation",
                              " ".join("%9s" % p for p in phases)))
        print("  " + "-" * (34 + 10 * len(phases)))
        for op, _n in by_op.most_common():
            cells = " ".join("%8.1f%%" % (100.0 * by_op_phase[op][p]
                                          / phase_totals[p]) for p in phases)
            print("  %-34s %s" % (op[:34], cells))
        print("  moves are %s" % ", ".join(
            "%s %d samples" % (p, phase_totals[p]) for p in phases))

    for func in LINE_SPLIT:
        rows = by_line.get(func)
        if not rows:
            continue
        print("")
        print("  inside %s()" % func)
        for (filename, line), n in rows.most_common(top_lines):
            src = linecache.getline(filename, line).strip()
            print("    %6.2f%% %5.1f ms  L%-5d %s"
                  % (100.0 * n / in_search, n / in_search * ms_move, line,
                     src[:80]))

    return {op: n / in_search for op, n in by_op.items()}


# ----------------------------------------------------------------------
# Mode 2: exact operation counts
# ----------------------------------------------------------------------

class Counts:
    def __init__(self):
        self.best_child = 0
        self.valid_calls = 0
        self.children_created = 0
        self.backup_calls = 0
        self.backup_steps = 0
        self.child_scan = 0   # children examined by _best_child (the real k)


def run_count(engine, games, device, seed):
    """Exact per-move operation counts.

    The wrappers below cost real time and the resulting LATENCY is therefore
    not the frozen engine's -- fewer simulations fit in 1,000 ms. That is
    acceptable and stated: this run is read for counts per simulation, which
    the deadline does not change, never for a duration.
    """
    C = Counts()
    orig_bc = MCTS._best_child
    orig_bk = MCTS._backup
    orig_valid = mcts_mod.rule_utl_valid_moves

    def bc(self, node):
        C.best_child += 1
        C.child_scan += len(node.children)
        return orig_bc(self, node)

    def bk(self, node, leaf_value):
        C.backup_calls += 1
        n, cur = 0, node
        while cur is not None:
            n += 1
            cur = cur.parent
        C.backup_steps += n
        return orig_bk(self, node, leaf_value)

    def valid(board, last_move, mini_winners):
        v = orig_valid(board, last_move, mini_winners)
        C.valid_calls += 1
        C.children_created += len(v)
        return v

    MCTS._best_child = bc
    MCTS._backup = bk
    mcts_mod.rule_utl_valid_moves = valid
    try:
        pa = TimedPlayer("engine:%s" % engine, device, count_nodes=True)
        pb = TimedPlayer("engine:%s" % engine, device, count_nodes=True)
        play_match(pa, pb, games, seed, warmup=2, gc_mode="deferred")
    finally:
        MCTS._best_child = orig_bc
        MCTS._backup = orig_bk
        mcts_mod.rule_utl_valid_moves = orig_valid

    moves = len(pa.records) + len(pb.records)
    sims = sum(r[2] for r in pa.records) + sum(r[2] for r in pb.records)
    sa, sb = pa.searcher.stats(), pb.searcher.stats()
    prev = pa.searcher.stat_prev_nodes + pb.searcher.stat_prev_nodes
    kept = pa.searcher.stat_reused_nodes + pb.searcher.stat_reused_nodes
    probes = pa.mcts.stat_probes + pb.mcts.stat_probes
    expansions = pa.mcts.stat_expansions + pb.mcts.stat_expansions
    nn = pa.mcts.stat_nn_evals + pb.mcts.stat_nn_evals
    batches = pa.mcts.stat_nn_batches + pb.mcts.stat_nn_batches

    out = {
        "moves": moves, "sims_per_move": sims / moves,
        "best_child_per_move": C.best_child / moves,
        "best_child_per_sim": C.best_child / sims if sims else 0.0,
        "mean_children_scanned": C.child_scan / C.best_child if C.best_child else 0,
        "mean_depth": C.best_child / sims if sims else 0.0,
        "backup_steps_per_move": C.backup_steps / moves,
        "backup_steps_per_sim": C.backup_steps / sims if sims else 0.0,
        "nodes_created_per_move": C.children_created / moves,
        "valid_calls_per_move": C.valid_calls / moves,
        "expansions_per_move": expansions / moves,
        "probes_per_move": probes / moves,
        "nn_evals_per_move": nn / moves,
        "nn_batches_per_move": batches / moves,
        "mean_wave": nn / batches if batches else 0.0,
        "nodes_released_per_move": (prev - kept) / moves,
        "nodes_prev_per_move": prev / moves,
        "reuse_rate": (sa["reuse_rate"] + sb["reuse_rate"]) / 2.0,
    }
    print("")
    for k in ("sims_per_move", "mean_depth", "best_child_per_move",
              "mean_children_scanned", "backup_steps_per_move",
              "nodes_created_per_move", "valid_calls_per_move",
              "expansions_per_move", "probes_per_move", "nn_evals_per_move",
              "mean_wave", "nodes_released_per_move", "reuse_rate"):
        print("    %-26s %12.1f" % (k, out[k]))
    print("")
    print("    NOTE: instrumented, so sims/move is BELOW the frozen engine's.")
    print("    Read the per-simulation ratios, not the per-move totals.")
    return out


# ----------------------------------------------------------------------
# Mode 3: microbenchmark of each operation in isolation
# ----------------------------------------------------------------------

def _sample_states(n=200, seed=7):
    """Reachable mid-game positions, so branching factors are realistic."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = GameState()
        depth = rng.randint(4, 45)
        for _ in range(depth):
            v = rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
            if not v or s.is_over():
                break
            s.make_move(rng.choice(v))
        if not s.is_over():
            out.append(s)
    return out


def _timeit(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6      # microseconds per call


def run_microbench(device):
    m = MCTS(model=None, device=device, n_sims=1, c_puct=1.5, wave_size=8,
             solve=True)
    states = _sample_states(120)
    ks = [len(rule_utl_valid_moves(s.board, s.last_move, s.mini_winners))
          for s in states]
    print("")
    print("    legal moves per position: mean %.1f  median %d  p90 %d"
          % (statistics.mean(ks), int(statistics.median(ks)),
             sorted(ks)[int(0.9 * len(ks))]))

    out = {}

    # best_child across realistic branching factors. Priors and visit counts
    # are varied so the max() cannot short-circuit on a degenerate tie.
    print("")
    print("    %-30s %10s" % ("operation", "us/call"))
    print("    " + "-" * 42)
    bc_by_k = {}
    for k in (2, 4, 6, 9, 12, 20):
        parent = MCTSNode(to_play=X)
        parent.N = 500
        for i in range(k):
            c = MCTSNode(parent=parent, prior=1.0 / k, move=i, to_play=O)
            c.N = 1 + i * 3
            c.W = 0.1 * i
            parent.children[i] = c
        bc_by_k[k] = _timeit(lambda: m._best_child(parent), 20000)
        print("    %-30s %10.2f" % ("best_child, k=%d" % k, bc_by_k[k]))
    out["best_child_us_by_k"] = bc_by_k

    # backup over a realistic path length
    chain = MCTSNode(to_play=X)
    node = chain
    for d in range(12):
        child = MCTSNode(parent=node, prior=0.1, move=d,
                         to_play=O if d % 2 == 0 else X)
        node.children[d] = child
        node = child
    out["backup_us_depth12"] = _timeit(lambda: m._backup(node, 0.3), 20000)
    out["backup_us_per_step"] = out["backup_us_depth12"] / 13.0
    print("    %-30s %10.2f" % ("backup, depth 12", out["backup_us_depth12"]))
    print("    %-30s %10.3f" % ("  per step", out["backup_us_per_step"]))

    parent = MCTSNode(to_play=X)
    out["node_alloc_us"] = _timeit(
        lambda: MCTSNode(parent=parent, prior=0.5, move=3, to_play=O), 200000)
    print("    %-30s %10.3f" % ("MCTSNode()", out["node_alloc_us"]))

    st = states[0]
    out["valid_moves_us"] = _timeit(
        lambda: rule_utl_valid_moves(st.board, st.last_move, st.mini_winners),
        50000)
    print("    %-30s %10.3f" % ("rule_utl_valid_moves", out["valid_moves_us"]))

    out["clone_us"] = _timeit(lambda: st.clone(), 50000)
    print("    %-30s %10.3f" % ("state.clone()", out["clone_us"]))

    mv = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)[0]

    def clone_move():
        c = st.clone()
        c.make_move(mv)
    out["clone_move_us"] = _timeit(clone_move, 50000)
    print("    %-30s %10.3f" % ("clone + make_move (probe)",
                                out["clone_move_us"]))

    # release over a tree the size the engine actually discards
    def build(depth, branch):
        root = MCTSNode(to_play=X)
        frontier, total = [root], 1
        for _ in range(depth):
            nxt = []
            for n in frontier:
                for i in range(branch):
                    c = MCTSNode(parent=n, prior=0.1, move=i, to_play=O)
                    n.children[i] = c
                    nxt.append(c)
                    total += 1
            frontier = nxt
        return root, total

    root, total = build(4, 6)
    t0 = time.perf_counter()
    TreeReuseSearcher.release(root)
    rel = (time.perf_counter() - t0) * 1e6
    out["release_us_per_node"] = rel / total
    out["release_tree_nodes"] = total
    print("    %-30s %10.3f" % ("release, per node",
                                out["release_us_per_node"]))
    return out


# ----------------------------------------------------------------------
# Reconciliation
# ----------------------------------------------------------------------

def reconcile(shares, counts, bench, ms_per_move):
    """Counts x isolated cost, against the sampled share. Two methods agreeing
    is the evidence; either one alone is a story."""
    k = counts["mean_children_scanned"]
    by_k = bench["best_child_us_by_k"]
    keys = sorted(by_k)
    lo = max([x for x in keys if x <= k], default=keys[0])
    hi = min([x for x in keys if x >= k], default=keys[-1])
    bc_us = (by_k[lo] if hi == lo else
             by_k[lo] + (by_k[hi] - by_k[lo]) * (k - lo) / (hi - lo))

    model = {
        "child scoring / best-child":
            counts["best_child_per_move"] * bc_us,
        "backup traversal":
            counts["backup_steps_per_move"] * bench["backup_us_per_step"],
        "node allocation":
            counts["nodes_created_per_move"] * bench["node_alloc_us"],
        "legal-child iteration":
            counts["valid_calls_per_move"] * bench["valid_moves_us"],
        "proof propagation":
            counts["probes_per_move"] * bench["clone_move_us"],
        "tree release":
            counts["nodes_released_per_move"] * bench["release_us_per_node"],
    }
    print("")
    print("  %-30s %10s %10s %8s" % ("operation", "modelled", "sampled",
                                     "ratio"))
    print("  " + "-" * 62)
    for op, us in sorted(model.items(), key=lambda kv: -kv[1]):
        modelled_ms = us / 1000.0
        sampled_ms = shares.get(op, 0.0) * ms_per_move
        ratio = (modelled_ms / sampled_ms) if sampled_ms > 0 else float("nan")
        print("  %-30s %9.1f %10.1f %8.2f"
              % (op, modelled_ms, sampled_ms, ratio))
    print("")
    print("  A ratio near 1 means the two methods agree. Below 1 means the")
    print("  sampler sees cost the isolated benchmark does not -- cache misses")
    print("  over a real 20k-node tree, which a tight loop over one hot node")
    print("  never pays. Treat the sampled column as the truth and this one as")
    print("  the check that nothing is wildly misattributed.")
    return {op: v / 1000.0 for op, v in model.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="all",
                    choices=("sample", "count", "microbench", "all"))
    ap.add_argument("--engine", default="final",
                    help="registry engine to profile (default: final)")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--count-games", type=int, default=4)
    ap.add_argument("--hz", type=int, default=500,
                    help="requested sample rate; Windows delivers ~130 Hz "
                         "whatever is asked, and every figure self-calibrates")
    ap.add_argument("--seed", type=int, default=PROFILE_SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--tag", default="tree")
    args = ap.parse_args()

    assert_frozen_sources()
    if args.engine in engine_registry.LADDER_EXEMPT:
        print("[!] %s is an evaluation-only ladder rung, not the deployment "
              "configuration" % args.engine)

    payload = {"engine": args.engine, "seed": args.seed, "hz": args.hz,
               "environment": engine_registry.environment(),
               "environment_drift": engine_registry.env_drift(),
               "git_head": engine_registry.git_head()}
    shares, counts, bench, sample = {}, None, None, None

    if args.mode in ("sample", "all"):
        print("=== 1. sampling profile, real %s ms workload ===" % args.engine)
        sample = run_sample(args.engine, args.games, args.hz, args.device,
                            args.seed)
        print("  %d samples over %.1f s (%.0f Hz achieved of %d requested;"
              " the OS sleep granularity sets this, nothing divides by the"
              " request)"
              % (sample["samples"], sample["wall_s"], sample["achieved_hz"],
                 sample["requested_hz"]))
        print("  %d moves, %.0f sims/move, %.1f ms/move in search"
              % (sample["moves"], sample["sims_per_move"],
                 sample["ms_per_move"]))
        shares = report_sample(sample)
        payload["sample"] = sample
        payload["shares"] = shares

    if args.mode in ("count", "all"):
        print("")
        print("=== 2. exact operation counts (timing distorted) ===")
        counts = run_count(args.engine, args.count_games, args.device,
                           args.seed + 1)
        payload["counts"] = counts

    if args.mode in ("microbench", "all"):
        print("")
        print("=== 3. isolated cost per operation ===")
        bench = run_microbench("cpu")
        payload["microbench"] = bench

    if args.mode == "all" and shares and counts and bench:
        print("")
        print("=== 4. reconciliation: counts x isolated cost vs sampled ===")
        payload["modelled_ms"] = reconcile(shares, counts, bench,
                                           sample["ms_per_move"])

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print("")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
