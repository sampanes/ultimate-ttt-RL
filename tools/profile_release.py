"""#46a -- does `release()` have to be on the move-critical path at all?

`RESULT_NATIVE_SELECT.md` closed on this: the reserve has gone 20 -> 35 -> 50 ->
95 ms across four engines, every millisecond of it caller-side work outside the
search's own deadline, and it grows with every throughput win because the thing
it pays for is a walk over a tree that throughput makes bigger. `pocket_sel`
failed the frozen latency requirement at the reserve it inherited and needed 95.

THE QUESTION IS NOT "how do we make the walk faster". It is whether the walk
needs to happen inside the move at all. Discarded subtrees could be detached in
O(1) and destroyed at the game boundary, next to the cyclic collect that already
runs there. What decides that is memory, and memory is what this tool measures:

    retired nodes per move          how much is dropped
    retired bytes per move          what that is worth in RAM
    retained nodes after adopt      what re-rooting keeps
    release ms vs retired nodes     the scaling law, as a fitted slope
    release p50 / p95 / p99 / max   the tail, which is what the reserve pays for
    peak live tree bytes            the floor any design has to hold
    projected peak, whole game      the number the design decision turns on

    python -m tools.profile_release --engine pocket_sel --games 6

THREE ARMS, for the reason `tools/profile_selection` needs four: no single run
supplies an unperturbed tail AND a node count.

  clean      no instrumentation at all. The wall, the throughput and the
             latency percentiles every other arm is priced against.
  timed      `release` and `_adopt` wrapped for time only. They are called ONCE
             PER MOVE, not 150,000 times, so this wrapper is free -- these are
             the percentiles to quote.
  counted    `release` replaced by an instrumented copy that counts nodes inside
             its own walk, and `_adopt` followed by a walk of the survivor.
             Supplies every count and the ms-vs-nodes slope. Its own timings run
             a few percent high and the report says by how much, measured
             against the timed arm rather than assumed.

MEMORY IS MEASURED WITHOUT THE GPU. A process-RSS delta taken around a real
search would also capture whatever the CUDA caching allocator did that second,
which is not a tree. So bytes-per-node is measured on synthetic trees built by
the REAL `_build_children_mirrored` -- same objects, same order, no device --
at the branching factor and expanded fraction the game arm actually observed.
The structural `sys.getsizeof` accounting is computed alongside it as a second
opinion; the two are printed together and a disagreement is flagged.

WHAT THIS TOOL DOES NOT DO: it does not change the engine. Everything here
either wraps or replaces a function for the duration of a run, so the numbers
describe `pocket_sel` exactly as it was measured in #45a.
"""

import argparse
import ctypes
import gc
import json
import os
import sys
import time

import subprocess

import numpy as np
import torch

from agents.mcts import MCTS, MCTSNode, TreeReuseSearcher
from tools import engine_registry
from tools.arena_1s import TimedPlayer, latency_report, play_match
from tools.profile_tree import assert_frozen_sources
from tools.runlock import gpu_baseline, single_instance

OUT_DIR = os.path.join("results", "profile_release")

# Instrumented, unscored. Same discipline as `select`: no result is read off
# these openings, and they are kept out of any namespace one is.
RELEASE_SEED = engine_registry.SEEDS["release"]


# ----------------------------------------------------------------------
# Process memory. Windows first, because that is the reference box.
# ----------------------------------------------------------------------

class _MEMCOUNTERS(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS_EX. `PrivateUsage` is the commit charge, which
    is the honest one for "how much memory is this holding": the working set
    can be trimmed by the OS without anything being freed."""
    _fields_ = [("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


_WIN_PROBE = None


def _win_probe():
    """Bind GetProcessMemoryInfo once, WITH argtypes.

    THE ARGTYPES ARE NOT OPTIONAL. `GetCurrentProcess` returns the pseudo-handle
    0xFFFFFFFFFFFFFFFF; left at ctypes' default `c_int` restype it comes back as
    a 32-bit -1, gets passed as a 32-bit argument, and the call fails -- quietly
    returning 0, which this tool would have read as "the tree costs nothing".
    """
    global _WIN_PROBE
    if _WIN_PROBE is None:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.GetCurrentProcess.argtypes = []
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MEMCOUNTERS),
                       ctypes.c_uint32]
        _WIN_PROBE = (fn, k32.GetCurrentProcess())
    return _WIN_PROBE


def memory_now():
    """(private_bytes, peak_working_set_bytes). Zeros where unavailable.

    `PrivateUsage` is the commit charge, which is the honest answer to "how much
    is this process holding": the working set can be trimmed by the OS without
    anything having been freed.

    Nothing above this line depends on the platform, so a Linux box gets the
    rusage figure and a clear zero for the peak rather than an exception.
    """
    if sys.platform == "win32":
        fn, handle = _win_probe()
        c = _MEMCOUNTERS()
        c.cb = ctypes.sizeof(c)
        if fn(handle, ctypes.byref(c), c.cb):
            return int(c.PrivateUsage), int(c.PeakWorkingSetSize)
        return 0, 0
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_maxrss) * 1024, 0
    except Exception:
        return 0, 0


# ----------------------------------------------------------------------
# The instrumented release. A DELIBERATE COPY -- see the drift test.
# ----------------------------------------------------------------------

def counting_release(root, keep=None, out=None):
    """`TreeReuseSearcher.release`, plus a tally of what it walked.

    THIS IS A COPY OF PRODUCTION CODE AND COPIES DRIFT. It exists because the
    count has to come from inside the walk -- counting before would need a
    second full traversal (which perturbs the very thing being measured) and
    counting after is impossible, since `release` is what erases the structure.
    `tools/test_profile_release.py` builds real mirrored trees, runs this and
    the real `release` on identical copies, and requires the resulting object
    graphs to be indistinguishable and the count to match an independent walk.
    Without that test this function is an unverified fork of the code under
    measurement.

    `out` receives [nodes, expanded]: `expanded` is the subset that had children
    to drop, which is also the subset carrying a mirror, and the two are needed
    separately because their per-node memory differs by an order of magnitude.
    """
    n_nodes = 0
    n_expanded = 0
    stack = [root]
    while stack:
        n = stack.pop()
        if n is keep:
            continue
        n_nodes += 1
        kids = n.children
        if kids:
            n_expanded += 1
        n.children = {}
        n.parent = None
        if n.sel is not None:
            n.sel = None
            n.kids = None
            n.selN = None
            n.selW = None
            n.selS = None
        stack.extend(kids.values())
    if out is not None:
        out[0] = n_nodes
        out[1] = n_expanded
    return n_nodes


def count_subtree(node):
    """Nodes in a subtree, including the root. Read-only."""
    n, stack = 0, [node]
    while stack:
        n += 1
        stack.extend(stack.pop().children.values())
    return n


# ----------------------------------------------------------------------
# What a node costs in bytes
# ----------------------------------------------------------------------

# CPython allocates in 16-byte size classes and `sys.getsizeof` reports neither
# the GC header nor that rounding, so every structural figure below is a LOWER
# bound. It is carried as a cross-check on the measured number, never as the
# number.
_GC_HEAD = 16


def _sizeof(obj):
    try:
        return sys.getsizeof(obj)
    except TypeError:
        return 0


def structural_bytes(root):
    """Sum `sys.getsizeof` over everything a subtree owns.

    The one thing getsizeof cannot see is the C++ side of a `ChildArray`: two
    std::vectors, int32 moves and float64 priors, plus their headers. Those are
    added analytically and marked as such in the report.
    """
    total = 0
    n_nodes = 0
    stack = [root]
    while stack:
        n = stack.pop()
        n_nodes += 1
        total += _sizeof(n) + _GC_HEAD
        total += _sizeof(n.children) + _GC_HEAD
        sel = n.sel
        if sel is not None:
            k = len(n.kids)
            total += _sizeof(n.kids) + _GC_HEAD
            total += _sizeof(sel) + _GC_HEAD
            for arr in (n.selN, n.selW, n.selS):
                total += _sizeof(arr) + _GC_HEAD
            # std::vector<int32_t> mv_ and std::vector<double> pr_, each with a
            # 24-byte control block, plus the allocator's own bookkeeping.
            total += 2 * 24 + k * 4 + k * 8
        stack.extend(n.children.values())
    return total, n_nodes


def _synthetic_tree(mcts, depth, branching, next_to_play=1):
    """A tree built by the engine's own child constructor, on the CPU.

    `_build_children_mirrored` is called directly rather than reimplemented, so
    the object graph is production's: same node type, same slots written, same
    numpy columns, same `ChildArray`. Only the priors are invented.

    Shape matters as much as contents. A uniform tree of branching `b` is
    1/b expanded nodes and (b-1)/b leaves, and `b` is taken from the game arm
    as retired-nodes-over-retired-expanded -- so the synthetic tree's mix of
    cheap leaves and expensive mirrored parents matches the real one by
    construction rather than by assumption.
    """
    row = np.full(81, 1.0 / max(branching, 1), dtype=np.float32)
    valid = list(range(branching))
    root = MCTSNode(to_play=next_to_play)
    frontier = [root]
    mirror = mcts._mirror
    for _ in range(depth):
        nxt = []
        for node in frontier:
            if mirror:
                mcts._build_children_mirrored(node, valid, row, next_to_play)
            else:
                # The same loop `_expand_children` runs for an unmirrored
                # engine. Kept here so `native_select=False` really does
                # produce an unmirrored tree -- calling the mirrored builder
                # regardless would make every "no mirror" comparison a
                # comparison of a tree with itself.
                for mv in valid:
                    node.children[mv] = MCTSNode(parent=node,
                                                 prior=float(row[mv]),
                                                 move=mv,
                                                 to_play=next_to_play)
            nxt.extend(node.children.values())
        frontier = nxt
    return root, frontier


def depth_for(branching, want_nodes=300000):
    """Deepest uniform tree of this branching that stays near `want_nodes`.

    Size is not free of consequences: too small and the per-node figure is
    dominated by the allocator's arena granularity, too large and the run
    starts measuring page faults. A few tens of thousands of nodes is also
    roughly what one move retires, which is the regime being priced.
    """
    depth, total = 1, 1 + branching
    while total * branching <= want_nodes and depth < 12:
        depth += 1
        total += branching ** depth
    return depth


def measure_bytes_per_node(branching, device="cpu", reps=5, depth=None):
    """Bytes of process commit per node, measured on real objects, no GPU.

    TREES ARE BUILT CUMULATIVELY AND KEPT ALIVE, and that is the whole design of
    this measurement. Build-then-free-then-build reports ZERO, because the
    second tree is served out of the arenas the first one left behind -- the
    first version of this function did exactly that and reported 0.0 bytes per
    node against a structural estimate of 391. Holding every tree means each new
    one has to be committed for real, which is also precisely the regime a
    deferred-retirement queue would put the process in.

    THIS MUST RUN IN A FRESH PROCESS. After a match the interpreter is sitting
    on hundreds of megabytes of freed tree, and the first few trees built here
    would be absorbed by it. `main` spawns `--measure-bytes` for that reason.

    The first tree is excluded: it pays for numpy's small-array caches, pybind's
    type machinery and the arena the interpreter would have grown anyway.
    """
    branching = max(2, int(branching))
    if depth is None:
        depth = depth_for(branching)
    mcts = MCTS(model=None, device=device, n_sims=1, c_puct=1.5, wave_size=8,
                solve=True, batched_expand=True, native_select=True)
    gc.collect()
    held = []
    marks = [memory_now()[0]]
    struct = n_nodes = 0
    for _ in range(reps):
        root, _leaves = _synthetic_tree(mcts, depth, branching)
        held.append(root)
        marks.append(memory_now()[0])
        if not struct:
            struct, n_nodes = structural_bytes(root)
    deltas = [(marks[i + 1] - marks[i]) / n_nodes for i in range(reps)]
    before_free = marks[-1]
    for root in held:
        TreeReuseSearcher.release(root)
    held.clear()
    gc.collect()
    freed_at = memory_now()[0]
    # CUMULATIVE, not the median of the per-tree deltas. The interpreter
    # commits memory in 256 KB arenas, so an individual tree's delta is
    # quantised -- measured at 336 to 534 bytes/node for identical trees. The
    # total over several trees divides that granularity away. The first delta
    # is the warmup and is reported but never used.
    span = reps - 1 if reps > 1 else 1
    cumulative = (marks[-1] - marks[1]) / float(n_nodes * span)
    return {
        "nodes_per_tree": int(n_nodes),
        "trees_held": reps,
        "branching": branching,
        "depth": depth,
        "bytes_per_node": float(cumulative),
        "first_tree_bytes_per_node": float(deltas[0]),
        "all_deltas_bytes_per_node": [float(d) for d in deltas],
        "freed_bytes_per_node": (before_free - freed_at) / (n_nodes * reps),
        "structural_bytes_per_node": struct / n_nodes,
        "held_total_bytes": before_free - marks[0],
    }


def spawn_bytes_per_node(branching):
    """Run `measure_bytes_per_node` in a clean interpreter and parse it back.

    Named so it can be found in a process list, and it exits on its own -- no
    teardown to document, no PID file, because it lives for a few seconds and
    holds no port and no GPU context.
    """
    cmd = [sys.executable, "-m", "tools.profile_release",
           "--measure-bytes", "%.4f" % branching]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit("[X] the memory subprocess failed:\n%s"
                         % (res.stderr or res.stdout))
    for line in reversed(res.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit("[X] the memory subprocess printed no JSON:\n%s"
                     % res.stdout)


# ----------------------------------------------------------------------
# Deployment arm
# ----------------------------------------------------------------------

class Probe:
    """Per-move release and adopt records, collected during a real match."""

    def __init__(self):
        self.release_ms = []
        self.release_nodes = []
        self.release_expanded = []
        self.adopt_ms = []
        self.retained = []
        self.live_after = []
        # Retired nodes accumulated within the current game, and the running
        # maximum across games. This is the projection the design turns on: it
        # is exactly what a whole-game deferred queue would be holding at the
        # moment before the boundary collect.
        self.game_cum = 0
        self.game_peak = 0
        self.per_game_cum = []
        self.game_cum_expanded = 0
        self.game_peak_expanded = 0

    def new_game(self):
        if self.game_cum:
            self.per_game_cum.append(self.game_cum)
        self.game_cum = 0
        self.game_cum_expanded = 0

    def finish(self):
        self.new_game()


def instrument(player, probe, counting):
    """Wrap `release` and `_adopt` on this player's searcher. Returns restore.

    Patched on the INSTANCE, not the class: the opponent has a searcher too and
    its re-rooting is not what is being measured. `release` is a staticmethod on
    the class, so the instance attribute shadows it -- and `search()` calls
    `self.release(...)`, which finds the shadow. Verified in the test file
    rather than assumed, because a patch that silently missed would report a
    release cost of zero.

    EVERY PROBE IS GATED ON `player.recording`. `play_match` plays warmup games
    for real and then discards them, because the first CUDA forward passes pay
    one-off autotune costs; a warmup move in a p99 is
    `uttt-warmup-in-numerator-not-denominator` all over again, and the tail is
    the entire point of this tool.

    Restores use `del`, not a write-back. Assigning a bound method into an
    instance's own `__dict__` makes the instance point at an object that points
    back at it, and this tool runs with the cyclic collector off.
    """
    searcher = player.searcher
    real_release = TreeReuseSearcher.release
    real_adopt = searcher._adopt
    pc = time.perf_counter

    def release(root, keep=None):
        if not player.recording:
            return real_release(root, keep)
        out = [0, 0]
        t0 = pc()
        if counting:
            counting_release(root, keep, out)
        else:
            real_release(root, keep)
        dt = (pc() - t0) * 1000.0
        probe.release_ms.append(dt)
        probe.release_nodes.append(out[0])
        probe.release_expanded.append(out[1])
        probe.game_cum += out[0]
        probe.game_cum_expanded += out[1]
        probe.game_peak = max(probe.game_peak, probe.game_cum)
        probe.game_peak_expanded = max(probe.game_peak_expanded,
                                       probe.game_cum_expanded)

    def adopt(state):
        if not player.recording:
            return real_adopt(state)
        t0 = pc()
        node, reason = real_adopt(state)
        dt = (pc() - t0) * 1000.0
        probe.adopt_ms.append(dt)
        # The retained walk is the expensive half of this instrument and it is
        # only in the counting arm. It is also the only way to get "what did
        # re-rooting actually keep" -- `stat_inherited_sims` is simulations,
        # which is not nodes.
        probe.retained.append(count_subtree(node) if (counting and node) else 0)
        return node, reason

    searcher.release = release
    searcher._adopt = adopt

    def restore():
        del searcher.release
        del searcher._adopt
    return restore


def watch_new_game(player, probe):
    """Hook `new_game` so the per-game accumulator resets on the boundary."""
    orig = type(player).new_game

    def new_game():
        probe.new_game()
        return orig(player)
    player.new_game = new_game

    def restore():
        del player.new_game
    return restore


def watch_live_tree(player, probe):
    """Count the tree the search LEAVES BEHIND, once per move.

    This is the floor: whatever design replaces `release`, the live tree is
    memory the engine has to hold anyway. Only in the counting arm -- it is a
    full walk inside the move.
    """
    orig = type(player).move

    def move(state, move_num):
        mv = orig(player, state, move_num)
        if player.recording:
            root = player.searcher._root
            probe.live_after.append(
                count_subtree(root) if root is not None else 0)
        return mv
    player.move = move

    def restore():
        del player.move
    return restore


def run_game(engine, opponent, games, device, seed, mode):
    """One arm of deployment play. `mode` in {clean, timed, counted}.

    The opponent is a DIFFERENT network, for the reason `tools/regress_engine`
    check 0 gives: a mirror predicts its own replies, adopts a much larger
    subtree, and therefore retires a much smaller one. Measuring retirement
    against a mirror would understate exactly the quantity this tool exists to
    size.
    """
    pa = TimedPlayer("engine:%s" % engine, device)
    pb = TimedPlayer("engine:%s" % opponent, device)
    probe = Probe()
    undo = []
    if mode != "clean":
        undo.append(instrument(pa, probe, counting=(mode == "counted")))
        undo.append(watch_new_game(pa, probe))
        if mode == "counted":
            undo.append(watch_live_tree(pa, probe))
    t0 = time.time()
    try:
        play_match(pa, pb, games, seed, warmup=2, gc_mode="deferred")
    finally:
        for fn in reversed(undo):
            fn()
    probe.finish()
    rep = latency_report(pa)
    return {
        "engine": engine,
        "opponent": opponent,
        "mode": mode,
        "games": games,
        "seconds": time.time() - t0,
        "fingerprint": (pa.provenance or {}).get("fingerprint"),
        "reserve_ms": pa.mcts.reserve_ms,
        "budget_ms": pa.budget_ms,
        "report": rep,
        "probe": {
            "release_ms": probe.release_ms,
            "release_nodes": probe.release_nodes,
            "release_expanded": probe.release_expanded,
            "adopt_ms": probe.adopt_ms,
            "retained": probe.retained,
            "live_after": probe.live_after,
            "per_game_cum": probe.per_game_cum,
            "game_peak": probe.game_peak,
            "game_peak_expanded": probe.game_peak_expanded,
        },
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def pcts(vals):
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "sum": float(a.sum())}


def fit_slope(nodes, ms):
    """ms = intercept + slope * nodes, least squares, with R^2.

    The slope is the number the design argument needs: if release is linear in
    nodes with a small intercept then it is pure per-node work and nothing about
    it can be optimised in place that deferral would not remove entirely.
    """
    x = np.asarray(nodes, dtype=np.float64)
    y = np.asarray(ms, dtype=np.float64)
    keep = x > 0
    x, y = x[keep], y[keep]
    if x.size < 3 or x.std() == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"us_per_node": float(slope) * 1000.0,
            "intercept_ms": float(intercept),
            "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
            "n": int(x.size)}


def mb(byts):
    return byts / (1024.0 * 1024.0)


def render(payload):
    arms = {a["mode"]: a for a in payload["arms"]}
    clean = arms.get("clean")
    timed = arms.get("timed")
    counted = arms.get("counted")
    mem = payload["memory"]
    bpn = mem["bytes_per_node"]
    out = {}

    print()
    print("=" * 78)
    print("#46a  RELEASE PATH -- %s vs %s, %d games"
          % (payload["engine"].upper(), payload["opponent"],
             payload["games"]))
    print("=" * 78)
    for mode in ("clean", "timed", "counted"):
        a = arms.get(mode)
        if not a:
            continue
        r = a["report"]
        print("  %-9s %8.1f ms/move  %8.1f nn-evals  p99 %7.1f  overhead p99"
              " %6.2f" % (mode, r["latency_ms"]["mean"],
                          r["per_move"]["neural_evaluations"],
                          r["latency_ms"]["p99"], r["overhead_ms"]["p99"]))
    if clean and counted and clean["report"]["per_move"]["neural_evaluations"]:
        lost = 100.0 * (1.0 - counted["report"]["per_move"]
                        ["neural_evaluations"]
                        / clean["report"]["per_move"]["neural_evaluations"])
        print("  the counting arm costs %.1f%% of the search; the timed arm's"
              % lost)
        print("  percentiles are the ones to quote.")

    # ---- the tail, from the timed arm ----
    src = timed or counted
    rel = pcts(src["probe"]["release_ms"])
    ado = pcts(src["probe"]["adopt_ms"])
    print()
    print("  RELEASE AND ADOPT, ms per move (%s arm, %d moves)"
          % (src["mode"], rel.get("n", 0)))
    print("  %-14s %8s %8s %8s %8s %8s"
          % ("", "mean", "p50", "p95", "p99", "max"))
    for name, d in (("release", rel), ("_adopt", ado)):
        if not d.get("n"):
            continue
        print("  %-14s %8.2f %8.2f %8.2f %8.2f %8.2f"
              % (name, d["mean"], d["p50"], d["p95"], d["p99"], d["max"]))
    over = src["report"]["overhead_ms"]
    print("  caller-side overhead p99 %.2f ms against a %.0f ms reserve;"
          % (over["p99"], src["reserve_ms"]))
    if rel.get("n"):
        print("  release alone is %.0f%% of that p99."
              % (100.0 * rel["p99"] / over["p99"]))
    out["release_ms"] = rel
    out["adopt_ms"] = ado
    out["overhead_p99"] = over["p99"]
    out["reserve_ms"] = src["reserve_ms"]

    if timed and counted:
        t = pcts(timed["probe"]["release_ms"])
        c = pcts(counted["probe"]["release_ms"])
        if t["mean"]:
            print("  the counter inside the walk adds %.1f%% to release's own"
                  % (100.0 * (c["mean"] / t["mean"] - 1.0)))
            print("  mean (%.2f -> %.2f ms), which is what the slope below"
                  % (t["mean"], c["mean"]))
            print("  carries.")

    # ---- counts and scaling, from the counting arm ----
    if not counted:
        return out
    p = counted["probe"]
    nodes = pcts(p["release_nodes"])
    expanded = pcts(p["release_expanded"])
    retained = pcts([v for v in p["retained"] if v])
    live = pcts(p["live_after"])
    slope = fit_slope(p["release_nodes"], p["release_ms"])
    moves = max(nodes.get("n", 0), 1)

    print()
    print("  WHAT A MOVE RETIRES (counting arm, %d moves)" % moves)
    print("    retired nodes / move          %12.0f" % nodes["mean"])
    print("      of which expanded           %12.0f  (%.1f%%, these carry the"
          % (expanded["mean"],
             100.0 * expanded["mean"] / max(nodes["mean"], 1e-9)))
    print("                                               native mirror)")
    if retained.get("n"):
        print("    retained after adopt          %12.0f  (%.1f%% of the tree"
              % (retained["mean"],
                 100.0 * retained["mean"]
                 / max(nodes["mean"] + retained["mean"], 1e-9)))
        print("                                               that existed)")
    print("    live tree after the search    %12.0f" % live["mean"])
    print("    retired nodes / move   p95    %12.0f" % nodes["p95"])
    print("    retired nodes / move   max    %12.0f" % nodes["max"])
    out["retired_nodes"] = nodes
    out["retired_expanded"] = expanded
    out["retained_after_adopt"] = retained
    out["live_after_search"] = live

    if slope:
        print()
        print("  RELEASE SCALES WITH NODES, and with essentially nothing else")
        print("    %.3f us per node, intercept %+.3f ms, R^2 %.4f, n=%d"
              % (slope["us_per_node"], slope["intercept_ms"], slope["r2"],
                 slope["n"]))
        if slope["r2"] >= 0.9:
            print("    A fit this tight means there is no fixed cost worth")
            print("    attacking and no per-node work worth shaving: the only")
            print("    lever is not doing the walk here.")
        out["slope"] = slope

    # ---- memory ----
    print()
    print("  BYTES PER NODE -- %d trees of %s nodes held at once, branching %d,"
          % (mem["trees_held"], "{:,}".format(mem["nodes_per_tree"]),
             mem["branching"]))
    print("  built by the engine's own `_build_children_mirrored`, fresh")
    print("  process, no GPU")
    print("    marginal commit / node        %12.1f bytes  <- used below" % bpn)
    print("    first tree (warmup, unused)   %12.1f bytes"
          % mem["first_tree_bytes_per_node"])
    print("    returned when all are freed   %12.1f bytes"
          % mem["freed_bytes_per_node"])
    print("    structural (getsizeof, lower bound) %6.1f bytes"
          % mem["structural_bytes_per_node"])
    gap = abs(bpn - mem["structural_bytes_per_node"]) / max(bpn, 1e-9)
    print("    the structural walk cannot see the GC header, the 16-byte size")
    print("    class rounding or the C++ vectors, so it is expected LOW; it is")
    print("    %.0f%% under the measured figure." % (100.0 * gap))
    if gap > 0.60:
        print("    [!] that is further apart than the unseen terms explain.")
    out["memory"] = mem

    retired_bytes = nodes["mean"] * bpn
    live_bytes = live["mean"] * bpn
    peak_game = counted["probe"]["game_peak"] * bpn
    per_game = counted["probe"]["per_game_cum"]
    print()
    print("  MEMORY, AND THE DESIGN DECISION")
    print("    retired bytes / move          %12.2f MB" % mb(retired_bytes))
    print("    live tree after the search    %12.2f MB" % mb(live_bytes))
    print("    peak live tree (max move)     %12.2f MB" % mb(live["max"] * bpn))
    print("    PROJECTED PEAK IF NOTHING IS FREED UNTIL THE GAME BOUNDARY")
    print("      worst game observed         %12.2f MB  (%s nodes)"
          % (mb(peak_game), "{:,}".format(counted["probe"]["game_peak"])))
    if per_game:
        print("      mean over %2d games          %12.2f MB"
              % (len(per_game), mb(float(np.mean(per_game)) * bpn)))
    print("      plus the live tree          %12.2f MB total"
          % mb(peak_game + live["max"] * bpn))
    out["retired_bytes_per_move"] = retired_bytes
    out["live_bytes"] = live_bytes
    out["deferred_peak_bytes"] = peak_game + live["max"] * bpn
    out["per_game_retired_nodes"] = per_game

    proc = payload.get("process_memory") or {}
    if proc.get("private_bytes"):
        print()
        print("    for scale: this process holds %.0f MB right now, peak"
              " working set %.0f MB"
              % (mb(proc["private_bytes"]), mb(proc["peak_working_set"])))
    out["process_memory"] = proc
    return out


def verdict(payload, derived):
    """Which of the owner's four designs the measurement selects."""
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    peak = derived.get("deferred_peak_bytes")
    if peak is None:
        print("  no counting arm -- nothing to decide on")
        return {}
    rel = derived["release_ms"]
    budget = mb(peak)
    print("  Deferring every discarded subtree to the game boundary would hold")
    print("  %.0f MB at the worst moment of the worst game measured." % budget)
    fits = budget <= 1024.0
    if fits:
        print("  That fits. DESIGN 1: detach on the move path, append the")
        print("  retired roots to a list, destroy them at the game boundary")
        print("  next to the collect that already runs there. The move-path")
        print("  operation becomes a list append.")
    else:
        print("  That does NOT fit comfortably. DESIGN 2: a bounded retired")
        print("  queue, reclaimed outside the deadline-critical section with a")
        print("  node watermark, sized from the numbers above.")
    print()
    print("  What it buys: release is %.2f ms at p99 and %.2f ms at its worst,"
          % (rel["p99"], rel["max"]))
    print("  out of a caller-side overhead p99 of %.2f ms against a %.0f ms"
          % (derived.get("overhead_p99", float("nan")),
             payload.get("reserve_ms", 0)))
    print("  reserve. Removing it from the move path is worth that much of the")
    print("  reserve directly, and the reserve is thinking time.")
    return {"deferred_peak_mb": budget, "design": 1 if fits else 2,
            "fits_in_one_gb": bool(fits)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="pocket_sel")
    ap.add_argument("--opponent", default="final",
                    help="a DIFFERENT network. A mirror adopts a larger "
                         "subtree and therefore retires a smaller one, which "
                         "understates every number here")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=RELEASE_SEED)
    ap.add_argument("--branching", type=float, default=0.0,
                    help="override the measured average branching used for "
                         "the synthetic memory trees")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--tag", default="release46")
    ap.add_argument("--rerender", default="")
    ap.add_argument("--measure-bytes", type=float, default=0.0,
                    help="internal: run ONLY the per-node memory measurement, "
                         "at this branching, and print it as JSON. Spawned as "
                         "a subprocess because a process that has just played "
                         "a match is sitting on the freed arenas of every tree "
                         "it retired, and the measurement would be absorbed "
                         "by them")
    args = ap.parse_args()

    if args.measure_bytes:
        print(json.dumps(measure_bytes_per_node(int(round(
            args.measure_bytes)))))
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.rerender:
        with open(args.rerender) as fh:
            payload = json.load(fh)
        payload["derived"] = render(payload)
        payload["verdict"] = verdict(payload, payload["derived"])
        return

    assert_frozen_sources()

    with single_instance("profile_release"):
        base = gpu_baseline()
        if base:
            print("[..] GPU load before starting: %.0f%% mean, %.0f%% peak"
                  % (base["mean_pct"], base["max_pct"]))
        # Deployment conditions, the same ones the frozen p99 was measured
        # under: automatic cyclic collection off, a collect at game boundaries.
        gc.disable()
        print("[!] automatic cyclic GC off; collecting at boundaries only")

        payload = {"tag": args.tag, "engine": args.engine,
                   "opponent": args.opponent, "games": args.games,
                   "seed": args.seed, "device": args.device, "arms": [],
                   "git_head": engine_registry.git_head(),
                   "environment": engine_registry.environment(),
                   "environment_drift": engine_registry.env_drift(),
                   "gpu_baseline_before_run": base}
        path = os.path.join(OUT_DIR, "%s.json" % args.tag)

        def save():
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)

        for mode in ("clean", "timed", "counted"):
            print("[..] %s / %s vs %s" % (mode, args.engine, args.opponent))
            arm = run_game(args.engine, args.opponent, args.games,
                           args.device, args.seed, mode)
            payload["arms"].append(arm)
            payload["reserve_ms"] = arm["reserve_ms"]
            save()

        counted = payload["arms"][-1]
        nodes = counted["probe"]["release_nodes"]
        expanded = counted["probe"]["release_expanded"]
        # Average children per expanded node, from the run itself. Every node
        # except the retired roots is somebody's child, so the ratio is the
        # branching factor the memory trees have to be built at -- guessing it
        # would put a guess inside the number the design decision turns on.
        tot_n, tot_e = float(sum(nodes)), float(sum(expanded))
        branching = args.branching or (tot_n / tot_e if tot_e else 8.0)
        print("[..] measured branching %.2f children per expanded node"
              % branching)
        payload["branching_measured"] = branching
        print("[..] measuring bytes per node in a FRESH process (no GPU)")
        payload["memory"] = spawn_bytes_per_node(branching)
        priv, peak = memory_now()
        payload["process_memory"] = {"private_bytes": priv,
                                     "peak_working_set": peak}
        save()

        derived = render(payload)
        derived["overhead_p99"] = (payload["arms"][1]["report"]["overhead_ms"]
                                   ["p99"])
        payload["derived"] = derived
        payload["verdict"] = verdict(payload, derived)
        save()
        print()
        print("[OK] wrote %s" % path)


if __name__ == "__main__":
    main()
