"""#44 -- where the 1,000 ms goes now that the wave is graphed.

`RESULT_TREE_PROFILE.md` priced `_best_child` at about 108 ms/move, measured on
`pocket_r35` before the CUDA graph. That is no longer the quantity the
native-selection decision turns on: `pocket_graph` runs about 30% more network
evaluations in the same second and therefore builds a materially bigger tree.
Both terms of `calls x cost-per-call` may have moved, and a share-of-total table
cannot say which.

    python -m tools.profile_selection --mode all

THE TIMER IS THE SAME ONE AS #39, deliberately: `ExclusiveTimer` from
`tools/profile_tree.py`. The stack sampler in that file is NOT used -- it is
GIL-biased 6.4x toward C, which is exactly the axis this question sits on.

FOUR ARMS PER ENGINE, because no single run can supply both factors of
`ms/move = calls/move x us/call` without distorting one of them:

  clean      no instrumentation. Supplies the wall time every share is taken
             of, and prices the other arms by how much search they cost.
  timed      every target wrapped. Supplies us/call. Its own call counts are
             NOT used: it loses ~30% of the search, so it makes ~30% fewer
             calls than the engine really does.
  counting   the same targets wrapped with an increment and no clock, ~1% of
             the search. Supplies calls/move directly.
  in-situ    ONLY `_best_child` wrapped. Prices that wrapper where it is used
             (about 2x the tight-loop calibration) and, more usefully, measures
             the SLOPE of search rate against the cost of the selection path --
             which is the lever a native primitive would pull.

An earlier version scaled every row by the ratio of simulation rates instead of
counting. That is the estimator `uttt-warmup-in-numerator-not-denominator`
warns against: a descent into a proven subtree is a simulation that costs almost
nothing, so the ratio moves with proof structure as well as throughput.

ALSO:

  parent attribution   `state.make_move` is called from two places that must
                       not be added together -- the descent, and the terminal
                       probes. The timer records (caller, callee), so the probes
                       can also be rolled up INCLUSIVE of the clone and
                       make_move they cause, which is the number their
                       recoverable ceiling has to be read off.
  fixed positions      The primary workload is a fixed position set searched
                       from a bare root, not self-play. A same-network match
                       inflates tree-reuse inheritance about 1.7x
                       (RESULT_GRAPH_WAVE, "what this does NOT license"), so its
                       trees are larger than deployment's and every per-node
                       cost measured on them is overstated.
  two engines          `pocket_r35` and `pocket_graph` on the SAME positions, so
                       "it costs more because there are more calls" is separable
                       from "each call is slower because the tree is bigger".
                       One engine cannot answer that.
  real deferred GC     `gc.disable()` plus a collect at boundaries, which is
                       what `tools/regress_engine` does and what the frozen
                       latency requirement was measured under. Note that
                       `tools/graph_ab` did NOT do this -- it passed
                       gc_mode="deferred" to `play_match`, which only adds the
                       boundary collect and leaves automatic collection on, and
                       it never released the previous position's tree. Both of
                       its arms were affected equally so its ratios stand, but
                       its absolute per-move figures were not taken under
                       deployment conditions.

Modes:
    fixed   N fixed positions, bare root, full deadline. No tree reuse, so
            `_adopt` and `release` do not appear -- they cannot, from a bare
            root -- and everything else is measured without inheritance.
    game    real games against `final`, tree reuse on. This exists for
            `_adopt` and `release`, which are deployment-only costs, and it
            uses a CROSS-NETWORK opponent because a mirror understates
            re-rooting (uttt-mirror-gate-hides-reroot-cost).
    all     both.
"""

import argparse
import collections
import gc
import json
import os
import time

import numpy as np
import torch

from agents import mcts as mcts_mod
from agents.graph_wave import GraphedWave
from agents.mcts import MCTS, TreeReuseSearcher
from engine.game import GameState
from tools import engine_registry
from tools.arena_1s import TimedPlayer, phase_of, play_match
from tools.profile_tree import ExclusiveTimer, assert_frozen_sources
from tools.runlock import gpu_baseline, single_instance

OUT_DIR = os.path.join("results", "profile_selection")

# Frozen namespace. No score is ever read off an instrumented run, and the
# openings must not leak into a namespace one is.
SELECT_SEED = engine_registry.SEEDS["select"]

# The forward pass is a bound method on the model instance, so it is wrapped
# per player rather than on a class.
FORWARD = "device: network forward"

# (report name, holder, attribute). MOD is the mcts module, whose module-global
# names are what mcts.py's own call sites resolve -- patching `engine.rules`
# would also catch the arena's legality check, which is harness, not engine.
WRAP_TARGETS = [
    ("_best_child",               "MCTS",  "_best_child"),
    ("backup",                    "MCTS",  "_backup"),
    ("terminal probes",           "MCTS",  "_mark_terminal_children"),
    ("proof: backward induction", "MCTS",  "_solve_from_children"),
    ("proof: propagation",        "MCTS",  "_propagate_solved"),
    ("node creation",             "MCTS",  "_expand_children"),
    ("wave loop",                 "MCTS",  "_run_wave"),
    ("device: graphed wave",      "MCTS",  "_expand_wave_graphed"),
    ("device: graph replay",      "GRAPHW", "run"),
    ("device: eager wave",        "MCTS",  "_expand_wave"),
    ("device: plane build + H2D", "MOD",   "wave_planes"),
    ("legal moves",               "MOD",   "rule_utl_valid_moves"),
    ("state clone",               "MOD",   "_clone"),
    ("state.make_move",           "STATE", "make_move"),
    ("tree reuse: adopt",         "REUSE", "_adopt"),
    ("tree release",              "REUSE", "release"),
]

# Everything the host spends getting a wave onto the device and back. The eager
# and graphed arms split this differently -- eager pays plane build, forward and
# `_expand_wave`; graphed pays all three inside one wrapped call -- so only the
# SUM is comparable across arms.
DEVICE_OPS = ("device: graphed wave", "device: graph replay",
              "device: eager wave", "device: plane build + H2D", FORWARD)

# Deployment-only: they exist because a tree was carried across a move.
REUSE_OPS = ("tree reuse: adopt", "tree release")

# One definition, shared by `patch` and its tests. A second copy in the test
# file is how a target silently stops being wrapped.
HOLDERS = {"MCTS": MCTS, "MOD": mcts_mod, "REUSE": TreeReuseSearcher,
           "STATE": GameState, "GRAPHW": GraphedWave}

# Report order, owner's list first. Anything measured but unlisted is printed
# after it under its own name rather than swept into "everything else" -- an
# unexplained row is the thing a profile exists to surface.
ORDER = [
    "_best_child",
    "terminal probes",
    "node creation",
    "backup",
    "tree reuse: adopt",
    "tree release",
    "state.make_move",
    "state clone",
    "legal moves",
    "proof: backward induction",
    "proof: propagation",
    "wave loop",
    "device: plane build + H2D",
    FORWARD,
    "device: eager wave",
    "device: graphed wave",
    "device: graph replay",
]


class AttributedTimer(ExclusiveTimer):
    """`ExclusiveTimer` plus who called whom, and the inclusive total.

    The exclusive column is the one that sums to the whole, so it stays the
    default. But an exclusive `terminal probes` row is nearly meaningless on its
    own: the probe's real cost is the clone and the make_move it performs, both
    of which are wrapped separately and therefore subtracted out of it. What can
    be recovered by removing the probes is the INCLUSIVE figure, and what a
    native selection primitive can recover from `_best_child` is the EXCLUSIVE
    one. Reporting only one of the two would answer only one of the questions.

    THE PER-CALL PRICE IS ITSELF THE BUDGET. This wrapper is entered about
    150,000 times per move, so every attribute lookup inside it is worth ~0.02
    ms/move. The bindings below are hoisted into the closure and the caller
    tally is one combined `[calls, seconds]` list rather than two dictionaries,
    which took the priced overhead from 1.11 us/call to well under that. It
    matters because the overhead is what forces the whole report to be scaled.
    """

    def __init__(self, ctx=None):
        super().__init__(ctx)
        self.inclusive = collections.defaultdict(float)
        self.calls_from = {}      # (caller, callee) -> [calls, seconds]

    def wrap(self, name, fn):
        st = self._stack
        total = self.total
        incl = self.inclusive
        calls = self.calls
        cfrom = self.calls_from
        ctx = self.ctx
        pc = time.perf_counter

        def inner(*a, **kw):
            caller = st[-1][0] if st else "-"
            frame = [name, pc(), 0.0]
            st.append(frame)
            try:
                return fn(*a, **kw)
            finally:
                st.pop()
                dt = pc() - frame[1]
                if ctx["on"]:
                    total[name] += dt - frame[2]
                    incl[name] += dt
                    calls[name] += 1
                    e = cfrom.get((caller, name))
                    if e is None:
                        cfrom[(caller, name)] = [1, dt]
                    else:
                        e[0] += 1
                        e[1] += dt
                if st:
                    st[-1][2] += dt
        return inner

    def calibrate(self, n=200000):
        """Price the wrapper, and split it at the measurement boundary.

        MOST OF A WRAPPER'S COST IS NOT INSIDE THE INTERVAL IT MEASURES. The
        clock starts after the frame is built and stops before the accumulators
        are written, so argument packing, the closure entry, the caller lookup
        and the four dictionary updates all happen OUTSIDE `dt`. They cost the
        search real time, but they were never in `raw_exclusive` and subtracting
        them from it is a double charge -- which is exactly what drove
        `_best_child` to a negative per-call cost on the first full run.

        So two prices come out of here:

            inside   what a wrapped call adds to its OWN recorded time. This is
                     what a row has to be charged.
            outside  the rest. It lands in the recorded time of whatever wrapped
                     function was on the stack above it, so it is charged to
                     PARENTS, once per nested call.
        """
        def noop():
            return None
        wrapped = self.wrap("_calibration", noop)
        was, self.ctx["on"] = self.ctx["on"], True
        t0 = time.perf_counter()
        for _ in range(n):
            wrapped()
        hot = time.perf_counter() - t0
        inside = self.total["_calibration"]
        t0 = time.perf_counter()
        for _ in range(n):
            noop()
        cold = time.perf_counter() - t0
        self.ctx["on"] = was
        for d in (self.total, self.calls, self.inclusive):
            d.pop("_calibration", None)
        self.calls_from.pop(("-", "_calibration"), None)
        self.inside_us = max(0.0, (inside - cold) / n * 1e6)
        return max(0.0, (hot - cold) / n * 1e6)


class CountingTimer(AttributedTimer):
    """Counts calls and nothing else, as cheaply as Python allows.

    WHY A THIRD ARM. `ms/move` for an operation is `calls/move x us/call`, and
    the two factors want opposite instruments. `us/call` needs a clock, which is
    expensive. `calls/move` needs an increment, which is nearly free -- and it is
    the factor that a scaling assumption would otherwise have to supply, because
    a deadline-bound search that has been slowed down makes FEWER calls.

    The alternative was to scale every row by the ratio of simulation rates, and
    that is exactly the estimator `uttt-warmup-in-numerator-not-denominator`
    warns off: a descent into a proven subtree costs almost nothing and is still
    counted as a simulation, so sims/move swings with proof structure rather
    than with throughput. Here it swung 35-57% across identical openings. The
    counting arm removes the need for the ratio at all.
    """

    def wrap(self, name, fn):
        calls = self.calls
        ctx = self.ctx

        def inner(*a, **kw):
            if ctx["on"]:
                calls[name] += 1
            return fn(*a, **kw)
        return inner


def patch(timer, player, only=None):
    """Wrap every target, or just `only`. Returns a restore callable.

    `GameState` is a pybind11 type and `make_move` is a plain function in its
    `__dict__`, so it patches like any other class attribute -- checked in
    tools/test_profile_selection.py rather than assumed, because a pybind type
    that refused `setattr` would fail silently into "make_move costs 0 ms".

    `only` exists for the in-situ calibration: wrapping ONE function and
    measuring how much search that alone costs prices that wrapper under the
    conditions it is actually used in.
    """
    saved = []
    for name, key, attr in WRAP_TARGETS:
        if only is not None and name not in only:
            continue
        holder = HOLDERS[key]
        raw = holder.__dict__.get(attr, getattr(holder, attr))
        is_static = isinstance(raw, staticmethod)
        fn = raw.__func__ if is_static else raw
        saved.append((holder, attr, raw))
        wrapped = timer.wrap(name, fn)
        setattr(holder, attr, staticmethod(wrapped) if is_static else wrapped)

    # The forward pass is wrapped on the INSTANCE, which shadows the class
    # method. Restoring must delete that shadow rather than write the bound
    # method back: an nn.Module holding a bound method of itself in its own
    # __dict__ is a reference cycle, and this profile runs with the cyclic
    # collector disabled.
    model = player.model
    wrap_fwd = only is None or FORWARD in only
    shadowed = "forward_both" in vars(model)
    raw_fwd = model.forward_both
    if wrap_fwd:
        model.forward_both = timer.wrap(FORWARD, raw_fwd)

    def restore():
        for holder, attr, raw in saved:
            setattr(holder, attr, raw)
        if not wrap_fwd:
            return
        if shadowed:
            model.forward_both = raw_fwd
        else:
            del model.forward_both
    return restore


def gate(player, ctx, count):
    """Open the accumulator only for the player under measurement.

    THE `finally` IS LOAD-BEARING. `tools/profile_tree` instruments both players
    of a self-play match, so the flag is rewritten on every move and never has
    to be closed. Here the opponent is a different engine and is NOT
    instrumented, so a gate left open after our move would charge every one of
    `final`'s 1,000 ms searches to this arm's totals.
    """
    orig = player.move

    def move(state, move_num):
        ctx["on"] = bool(count and player.recording)
        try:
            return orig(state, move_num)
        finally:
            ctx["on"] = False
    player.move = move


def build(engine, device):
    """`count_nodes` stays OFF.

    It is two full subtree walks per move, inside the interval the move is
    timed over, and it is not wrapped -- so it would both steal search from the
    deadline and land in "everything else". `tools/profile_tree` turns it on
    only in its counts mode for the same reason.
    """
    return TimedPlayer("engine:%s" % engine, device)


def positions_for(engine, games, device, seed, want):
    """Real positions from real play, spread across the game.

    Sampled from a match neither arm takes part in as the graph arm, so the
    inputs are the same for everyone and no arm chose them.
    """
    pa = build(engine, device)
    pb = build(engine, device)
    seen = []
    orig = pa.move

    def rec(state, move_num):
        seen.append((state.clone(),
                     phase_of(int(np.count_nonzero(state.board)))))
        return orig(state, move_num)

    pa.move = rec
    play_match(pa, pb, games, seed, warmup=1, gc_mode="deferred")
    step = max(1, len(seen) // want)
    return seen[::step][:want]


# ----------------------------------------------------------------------
# Driver 1: fixed positions, bare root
# ----------------------------------------------------------------------

def run_fixed(engine, positions, device, wrapped, only=None, warmup=2,
              counting=False):
    """Search each position for the full deadline. No reuse, no divergence.

    THE WARMUP IS NOT OPTIONAL, and it is the same reason `play_match` has one:
    the first CUDA forward passes pay cudnn autotune and allocator costs that
    belong to no move in particular. `play_match(warmup=N)` covers the game
    driver; this loop bypasses it entirely, so it has to do its own. Warmup
    positions are searched for real with the accumulator SHUT and are not part
    of any denominator.
    """
    p = build(engine, device)
    ctx = {"on": False}
    timer = restore = None
    if wrapped:
        timer = (CountingTimer if counting else AttributedTimer)(ctx)
        restore = patch(timer, p, only=only)
    rows = []
    try:
        for state, _phase in positions[:warmup]:
            p.mcts.reset_stats()
            with torch.no_grad():
                _pi, root = p.mcts.search(state.clone())
            TreeReuseSearcher.release(root)
            gc.collect()
        for state, phase in positions:
            p.mcts.reset_stats()
            ctx["on"] = True
            t0 = time.perf_counter()
            with torch.no_grad():
                _pi, root = p.mcts.search(state.clone())
            dt = (time.perf_counter() - t0) * 1000.0
            ctx["on"] = False
            last = p.mcts.last
            rows.append({"phase": phase, "search_ms": dt,
                         "sims": last["simulations_completed"],
                         "nn": last["neural_evaluations"],
                         "expansions": last["nodes_expanded"],
                         "root_n": last["root_n"]})
            # Between positions, not inside one: the tree is cyclic garbage and
            # a collector pause landing mid-search is exactly what deferred GC
            # exists to move out of the measurement.
            TreeReuseSearcher.release(root)
            gc.collect()
    finally:
        if restore is not None:
            restore()

    out = summarize(p, rows, timer, len(rows),
                    float(np.mean([r["search_ms"] for r in rows])))
    out["workload"] = "fixed"
    out["only"] = sorted(only) if only else None
    out["counting"] = bool(counting)
    return out


# ----------------------------------------------------------------------
# Driver 2: real games, tree reuse on
# ----------------------------------------------------------------------

def run_game(engine, opponent, games, device, seed, wrapped,
             counting=False):
    """Deployment shape: reuse on, a real opponent, `move_ms` as the total.

    The opponent is a DIFFERENT network on purpose. A mirror inherits about 1.7x
    more of its own tree than a cross-network match does, and every per-node
    cost here scales with tree size, so a mirror would inflate exactly the rows
    this run exists to measure.
    """
    pa = build(engine, device)
    pb = build(opponent, device)
    ctx = {"on": False}
    timer = restore = None
    if wrapped:
        timer = (CountingTimer if counting else AttributedTimer)(ctx)
        restore = patch(timer, pa)
    gate(pa, ctx, True)
    gate(pb, ctx, False)
    try:
        play_match(pa, pb, games, seed, warmup=2, gc_mode="deferred")
    finally:
        if restore is not None:
            restore()

    rows = [{"phase": phase_of(r[9]), "search_ms": r[0], "sims": r[2],
             "nn": r[3], "expansions": r[4], "root_n": 0,
             "inner_ms": r[1], "inherited": r[6]} for r in pa.records]
    out = summarize(pa, rows, timer, len(rows),
                    float(np.mean([r["search_ms"] for r in rows])))
    out["workload"] = "game"
    out["counting"] = bool(counting)
    out["opponent"] = opponent
    out["inner_ms_per_move"] = float(np.mean([r["inner_ms"] for r in rows]))
    out["inherited_per_move"] = float(np.mean([r["inherited"] for r in rows]))
    out["reuse"] = pa.searcher.stats()
    return out


def summarize(player, rows, timer, moves, total_ms):
    """One arm's numbers. `total_ms` is the wall the rows are shares OF."""
    out = {
        "engine": player.engine_name,
        "fingerprint": (player.provenance or {}).get("fingerprint"),
        "params": player.net_info["params"],
        "graph": bool(player.mcts.graph_wave is not None),
        "graph_requested": bool(player.mcts.graph_wave_requested),
        "budget_ms": player.budget_ms,
        "reserve_ms": player.mcts.reserve_ms,
        "wave": player.wave,
        "moves": moves,
        "wall_ms_per_move": total_ms,
        "sims_per_move": float(np.mean([r["sims"] for r in rows])),
        "nn_per_move": float(np.mean([r["nn"] for r in rows])),
        "expansions_per_move": float(np.mean([r["expansions"] for r in rows])),
        "wrapped": timer is not None,
    }
    if timer is None:
        return out

    # RAW sums only. The per-call price is NOT applied here, because the price
    # this report uses is derived from the clean arm and that arm is not in
    # scope yet -- see `price_arm`.
    ops = {}
    # Keyed on `calls`, NOT on `total`: the counting timer never writes a
    # duration, so iterating `total` gave it an empty `ops` and `per_move`
    # silently fell back to the timed arm's own call counts -- the exact
    # extrapolation the counting arm exists to remove. Caught by the residual
    # check, which jumped to 22% of the move.
    for name in timer.calls:
        calls = timer.calls[name]
        nested = sum(e[0] for (caller, _c), e in timer.calls_from.items()
                     if caller == name)
        ops[name] = {
            "calls": calls,
            "calls_per_move": calls / moves if moves else 0.0,
            "nested_calls": nested,
            "raw_exclusive_ms_per_move": (timer.total[name] * 1000.0 / moves
                                          if moves else 0.0),
            "raw_inclusive_ms_per_move": (timer.inclusive[name] * 1000.0 / moves
                                          if moves else 0.0),
        }
    out["ops"] = ops
    out["calls_per_move"] = (sum(timer.calls.values()) / moves
                             if moves else 0.0)
    out["top_level_calls_per_move"] = (
        sum(e[0] for (caller, _c), e in timer.calls_from.items()
            if caller == "-") / moves if moves else 0.0)
    out["calibrated_us_per_call"] = timer.calibrate()
    out["inside_us_per_call"] = timer.inside_us
    out["callers"] = {
        "%s -> %s" % key: {"calls": entry[0],
                           "raw_ms_per_move": (entry[1] * 1000.0 / moves
                                               if moves else 0.0)}
        for key, entry in timer.calls_from.items()}
    return out


def insitu_price(solo, clean, op):
    """What one wrapper costs, measured where it is used.

    `ExclusiveTimer.calibrate()` times a wrapped zero-argument no-op in a tight
    loop, and that is a biased lower bound: the branch predictor and the
    instruction cache are perfectly warm, the argument tuple is empty, and the
    accumulator dictionaries hold one key. Same trap as
    `in-process-sampler-gil-bias` -- a calibration workload has to span the axis
    the instrument is used across, and a tight-loop no-op does not span this
    one. Measured here at about 5x the tight-loop figure.

    `solo` is a run of the same positions with ONLY `op` wrapped. The search is
    deadline-bound, so the wrapper does not make a move longer -- it makes fewer
    simulations fit -- and the shortfall against the clean arm is exactly what
    that one wrapper cost:

        cost_ms_per_move = wall x (1 - solo_sims / clean_sims)
        us_per_call      = cost_ms_per_move / calls_per_move

    This also measures something the report wants for its own sake: how much
    search one microsecond on this code path is worth. See `elasticity`.
    """
    o = (solo.get("ops") or {}).get(op)
    if not o or not o["calls_per_move"]:
        return None
    k = scale_of(clean, solo)
    if not k or k != k:
        return None
    cost = solo["wall_ms_per_move"] * (1.0 - 1.0 / k)
    return {"op": op, "us_per_call": cost * 1000.0 / o["calls_per_move"],
            "cost_ms_per_move": cost, "calls_per_move": o["calls_per_move"],
            # Both units are carried. The slope is quoted in network
            # evaluations because that is the work unit this project settled
            # on; simulations are kept alongside so the two can be compared.
            "nn_solo": solo["nn_per_move"], "nn_clean": clean["nn_per_move"],
            "sims_solo": solo["sims_per_move"],
            "sims_clean": clean["sims_per_move"],
            "nn_lost_pct": 100.0 * (1.0 - solo["nn_per_move"]
                                    / clean["nn_per_move"]),
            "sims_lost_pct": 100.0 * (1.0 - solo["sims_per_move"]
                                      / clean["sims_per_move"])}


_INSIDE_US = None


def measure_inside_us():
    """The wrapper's in-interval cost, measured on demand and memoised."""
    global _INSIDE_US
    if _INSIDE_US is None:
        t = AttributedTimer()
        t.calibrate()
        _INSIDE_US = t.inside_us
    return _INSIDE_US


def price_arm(arm, clean):
    """Subtract the instrument. It acts in three places, so it comes off three.

    INSIDE. What a wrapped call adds to its own recorded time -- see
    `AttributedTimer.calibrate`. Charged to the row, once per call.

    OUTSIDE, CHARGED TO THE PARENT. The rest of a wrapper's cost happens before
    the clock starts and after it stops, so it is invisible to that call's own
    row but perfectly visible in the recorded time of whatever wrapped function
    was above it on the stack. Charged once per NESTED call, to the caller. At
    top level it lands in "everything else", which needs no explicit charge
    because the residual is a remainder.

    DIFFUSE. Those two together still do not account for all the search the
    instrument cost -- typically about a fifth of it. The rest is not
    attributable to any call site: several hundred thousand extra frames and
    dictionary writes a second push on the instruction and data caches, and that
    slows the UNWRAPPED code too. It comes off as a proportional deflation.

    THE FIRST VERSION OF THIS CHARGED THE WHOLE MEASURED PRICE PER CALL and
    drove `_best_child` to -1.113 us/call on the first full run -- a wrapper
    cannot cost more than the total time recorded for the call it wraps, so the
    negative was the model telling us it was double-charging.

    The scaled total equals the clean arm's wall time by construction, so an
    accounting mistake anywhere above has somewhere to surface; the report
    prints that drift.
    """
    k = scale_of(clean, arm)
    wall = arm["wall_ms_per_move"]
    instrument = wall * (1.0 - 1.0 / k) if k and k == k else 0.0
    # `inside` is a property of the wrapper's own code, not of the run, so a
    # result file written before the split existed can be re-priced without
    # replaying the games. `wrap()` is unchanged; only `calibrate()` grew the
    # second number.
    inside = arm.get("inside_us_per_call")
    if inside is None:
        inside = arm["inside_us_per_call"] = measure_inside_us()
    outside = max(0.0, arm["calibrated_us_per_call"] - inside)
    moves = arm["moves"]

    direct = 0.0
    for o in arm["ops"].values():
        o["_price_us"] = inside
        o["_direct_ms"] = (o["calls_per_move"] * inside
                           + o["nested_calls"] / moves * outside) / 1000.0
        direct += o["_direct_ms"]
    available = wall - direct
    deflate = ((wall - instrument) / available) if available > 0 else 1.0

    for o in arm["ops"].values():
        o["exclusive_ms_per_move"] = ((o["raw_exclusive_ms_per_move"]
                                       - o["_direct_ms"]) * deflate)
        # Inclusive keeps the children's real time but drops their inside cost,
        # which is theirs and already removed from their own rows.
        o["inclusive_ms_per_move"] = ((o["raw_inclusive_ms_per_move"]
                                       - o["calls_per_move"] * inside / 1000.0
                                       - o["nested_calls"] / moves
                                       * (inside + outside) / 1000.0) * deflate)
        o["us_per_call"] = ((o["exclusive_ms_per_move"] * 1000.0
                             / o["calls_per_move"])
                            if o["calls_per_move"] else 0.0)
    for c in arm["callers"].values():
        per = c["calls"] / moves * inside / 1000.0
        c["ms_per_move"] = (c["raw_ms_per_move"] - per) * deflate

    arm["scale"] = k
    arm["scale_sims"] = scale_of(clean, arm, "sims_per_move")
    arm["instrument_ms_per_move"] = instrument
    arm["direct_ms_per_move"] = direct
    arm["diffuse_ms_per_move"] = instrument - direct
    arm["deflate"] = deflate
    arm["wrapper_us_per_call"] = arm["calibrated_us_per_call"]
    arm["inside_us"] = inside
    arm["outside_us"] = outside
    arm["implied_us_per_call"] = ((instrument * 1000.0 / arm["calls_per_move"])
                                  if arm["calls_per_move"] else 0.0)
    arm["wrapper_overhead_ms_per_move"] = instrument
    arm["measured_ms_per_move"] = wall - instrument
    return arm


def elasticity(price, bc_ms_per_move, clean):
    """What removing the selection cost would buy, from a MEASURED slope.

    The in-situ run added a known number of microseconds to every `_best_child`
    call and the search lost a measured number of simulations. That is a
    measured derivative of search rate with respect to the cost of this exact
    code path, and #45 proposes to move along the same line in the other
    direction. It is far better evidence than "this row is N ms, so N ms is
    available", because it already contains whatever second-order effects the
    deadline predictor and the wave structure impose.
    """
    if not price or not price["cost_ms_per_move"]:
        return None
    nn_per_ms = (price["nn_clean"] - price["nn_solo"]) \
        / price["cost_ms_per_move"]
    sims_per_ms = (price["sims_clean"] - price["sims_solo"]) \
        / price["cost_ms_per_move"]
    return {
        "nn_per_ms_of_selection": nn_per_ms,
        "sims_per_ms_of_selection": sims_per_ms,
        "best_child_ms_per_move": bc_ms_per_move,
        "nn_if_selection_were_free": clean["nn_per_move"]
        + nn_per_ms * bc_ms_per_move,
        "pct_more_search": (100.0 * nn_per_ms * bc_ms_per_move
                            / clean["nn_per_move"])
        if clean["nn_per_move"] else float("nan"),
        "pct_more_sims": (100.0 * sims_per_ms * bc_ms_per_move
                          / clean["sims_per_move"])
        if clean["sims_per_move"] else float("nan"),
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def scale_of(clean, dirty, unit="nn_per_move"):
    """Ratio of work rates -- used ONLY to price the instrument.

    Under a wall clock, instrumentation does not make a move take longer; it
    makes less search fit inside it, and the shortfall is what it consumed.

    THE UNIT IS NETWORK EVALUATIONS, NOT SIMULATIONS.
    `uttt-warmup-in-numerator-not-denominator`: a descent into a proven subtree
    is counted as a simulation and costs almost nothing, so sims/move moves with
    proof structure as well as with throughput -- 35-57% across identical
    openings. Measured here on the same positions, `pocket_r35` and
    `pocket_graph` differ by +7.3% in sims and +28.8% in network evaluations.
    Neither unit is perfect (the fraction of leaves needing the net is not
    fixed either), so the report prints both and flags a disagreement.

    DO NOT USE THIS TO SCALE A ROW. Per-operation call rates come from the
    counting arm, which measures them directly at a fiftieth of the
    perturbation.
    """
    a = dirty.get(unit) or 0.0
    return (clean[unit] / a) if a else float("nan")


def per_move(arm, count, name):
    """`calls/move x us/call`, each factor from the arm that measures it best.

    calls/move  from the counting arm: an increment per call, ~1% of the search,
                and no scaling assumption anywhere.
    us/call     from the timed arm, which is heavily perturbed in total but
                whose PER-CALL figure is not -- the perturbation shows up as
                fewer calls, not as a slower call.

    Falls back to the timed arm's own call rate when there is no counting arm,
    which understates the row.
    """
    o = (arm.get("ops") or {}).get(name)
    if not o:
        return 0.0, 0.0
    calls = o["calls_per_move"]
    if count:
        c = (count.get("ops") or {}).get(name)
        if c and c["calls_per_move"]:
            # The counting arm is cheap but not free -- about 7% of the search
            # -- so its call rate is still slightly below the engine's. That
            # last step up is the same work-rate ratio the timed arm needed,
            # applied to a third of the gap.
            calls = c["calls_per_move"] * count.get("scale", 1.0)
    return calls, calls * o["us_per_call"] / 1000.0


def report_arm(arm, clean, count=None):
    """One arm's table. `ms/move` is calls-from-the-counting-arm x us/call."""
    price_arm(arm, clean)
    ops = arm.get("ops") or {}
    k = arm["scale"]
    wall = clean["wall_ms_per_move"]
    print()
    print("=" * 78)
    print("%s -- %s, %s params, graph %s"
          % (arm["engine"].upper(), arm["workload"],
             "{:,}".format(arm["params"]), "ON" if arm["graph"] else "off"))
    print("=" * 78)
    print("  clean    %8.1f ms/move wall %8.1f sims %8.1f nn-evals"
          % (wall, clean["sims_per_move"], clean["nn_per_move"]))
    if count:
        print("  counting %8.1f ms/move wall %8.1f sims %8.1f nn-evals"
              " (%.1f%% slower, corrected)"
              % (count["wall_ms_per_move"], count["sims_per_move"],
                 count["nn_per_move"],
                 100.0 * (1.0 - 1.0 / count.get("scale", 1.0))))
    print("  timed    %8.1f ms/move wall %8.1f sims %8.1f nn-evals"
          " (%.1f%% slower)"
          % (arm["wall_ms_per_move"], arm["sims_per_move"], arm["nn_per_move"],
             100.0 * (1.0 - 1.0 / k)))
    ks = arm["scale_sims"]
    if ks == ks and abs(ks - k) > 0.05 * k:
        print("  [!] the two work units disagree about how much the instrument")
        print("      cost: %.1f%% by network evaluations, %.1f%% by"
              % (100.0 * (1.0 - 1.0 / k), 100.0 * (1.0 - 1.0 / ks)))
        print("      simulations. Network evaluations is the one charged;")
        print("      a free descent is not a unit of work.")
    print("  %s wrapper entries a move, at %.3f us inside the interval they"
          % ("{:,.0f}".format(arm["calls_per_move"]), arm["inside_us"]))
    print("  time and %.3f us outside it (charged to the caller); the"
          % arm["outside_us"])
    print("  unattributable remainder comes off as a %.3fx deflation of"
          % arm["deflate"])
    print("  us/call -- cache pressure on code the instrument does not wrap.")
    print()
    # Whether the counting arm actually supplied anything, not merely whether
    # one was passed: an empty `ops` silently falls back to the timed arm's own
    # call counts, which is the extrapolation this design exists to remove.
    used_count = bool(count and any(
        o.get("calls_per_move") for o in (count.get("ops") or {}).values()))
    print("  calls/move is from the %s; us/call from the timed run; ms/move is"
          % ("counting arm" if used_count
             else "TIMED arm -- NO USABLE COUNTING ARM"))
    print("  their product, so no row is scaled by a simulation-rate ratio.")
    if count and not used_count:
        print("  [X] a counting arm was supplied but carried no call counts;")
        print("      every row below understates the engine by whatever the")
        print("      timed arm lost.")
    print()
    print("  %-28s %11s %9s %10s %7s"
          % ("operation", "calls/move", "us/call", "ms/move", "share"))
    print("  " + "-" * 70)
    named = [n for n in ORDER if n in ops]
    named += sorted(n for n in ops if n not in ORDER)
    rows, measured, unresolved = {}, 0.0, []
    for name in named:
        calls, ms = per_move(arm, count, name)
        rows[name] = {"calls_per_move": calls, "ms_per_move": ms,
                      "us_per_call": ops[name]["us_per_call"]}
        measured += ms
        if not ops[name]["calls"]:
            continue
        # A row whose own per-call cost is comparable to the price charged to
        # it cannot be resolved by this instrument. Flagged, never clamped:
        # clamping hides that the number is not a measurement.
        flag = " "
        if ops[name]["us_per_call"] < 2.0 * ops[name]["_price_us"]:
            flag = "*"
            unresolved.append(name)
        print("  %-27s%s %11.1f %9.3f %10.2f %6.1f%%"
              % (name, flag, calls, ops[name]["us_per_call"], ms,
                 100.0 * ms / wall))
    resid = wall - measured
    print("  " + "-" * 70)
    print("  %-28s %11s %9s %10.2f %6.1f%%"
          % ("everything else", "-", "-", resid, 100.0 * resid / wall))
    print("  %-28s %11s %9s %10.2f %6.1f%%"
          % ("TOTAL (clean arm's wall)", "-", "-", wall, 100.0))
    # The residual USED to be an identity -- rows were scaled so the total came
    # back to the wall by construction. It is not one any more: the call rates
    # come from a different run than the per-call costs, so the two only
    # reconcile if the counting arm's call ratio really matches the search-rate
    # ratio. That makes a wild residual a genuine warning rather than a
    # rounding artifact.
    if resid < 0 or resid > 0.25 * wall:
        print("  [!] residual is %.1f%% of the move. The rows are built from"
              % (100.0 * resid / wall))
        print("      one run's call counts and another's per-call costs; a")
        print("      residual this size means those two runs disagree about")
        print("      how much work a move contains.")
    absent = [n for n in ORDER if n in ops and not ops[n]["calls"]]
    if absent:
        print("  never called: %s" % ", ".join(absent))
    if unresolved:
        print("  * per-call cost is under 2x the price charged to that row, so")
        print("    it sits at the resolution floor and is a bound, not a")
        print("    measurement: %s" % ", ".join(unresolved))

    def scaled(name, field):
        o = ops.get(name)
        if not o or not o["calls_per_move"]:
            return 0.0
        calls, _ms = per_move(arm, count, name)
        return o[field] * calls / o["calls_per_move"]

    print()
    print("  ROLL-UPS (inclusive -- these OVERLAP the table above)")
    incl = scaled("terminal probes", "inclusive_ms_per_move")
    excl = rows.get("terminal probes", {}).get("ms_per_move", 0.0)
    print("    terminal probes, with the clone and make_move they cause"
          "  %7.2f ms/move" % incl)
    print("      the probe's own loop is %.2f of that; the rest is charged to"
          " the clone and make_move rows" % excl)
    dev = sum(rows[n]["ms_per_move"] for n in DEVICE_OPS if n in rows)
    print("    whole device path (planes, forward, mask, softmax, pulls)"
          "       %7.2f ms/move" % dev)
    reuse = sum(scaled(n, "inclusive_ms_per_move") for n in REUSE_OPS
                if n in ops)
    print("    _adopt + release, the work OUTSIDE the search's own deadline"
          "    %7.2f ms/move" % reuse)
    tree = sum(v["ms_per_move"] for n, v in rows.items() if n not in DEVICE_OPS)
    print("    everything that is not the device path"
          "                       %7.2f ms/move" % tree)

    callers = arm.get("callers") or {}
    mm = {k2: v for k2, v in callers.items()
          if k2.endswith("-> state.make_move")}
    if mm:
        cscale = ((rows.get("state.make_move", {}).get("calls_per_move", 0.0)
                   / ops["state.make_move"]["calls_per_move"])
                  if ops.get("state.make_move", {}).get("calls_per_move")
                  else 1.0)
        print()
        print("  state.make_move by caller (the descent and the probes are")
        print("  different budgets and must not be added together)")
        for key, v in sorted(mm.items(), key=lambda kv: -kv[1]["ms_per_move"]):
            print("    %-42s %10.1f calls %8.2f ms/move"
                  % (key.split(" -> ")[0],
                     v["calls"] / arm["moves"] * cscale,
                     v["ms_per_move"] * cscale))
    consistent = cross_checks(arm)
    return {"scale": k, "consistent": consistent, "rows": rows,
            "residual_ms_per_move": resid,
            "device_ms_per_move": dev, "reuse_ms_per_move": reuse,
            "non_device_ms_per_move": tree,
            "probes_inclusive_ms_per_move": incl}


def factors(arm, clean, row):
    """`_best_child` ms/move as the product of three independent factors.

    ms/move = simulations/move  x  calls/simulation  x  us/call

    This is the decomposition the question needs. "It costs more" can mean the
    engine runs more simulations (a throughput fact, and the whole point of the
    graph), that each simulation descends further (a tree-shape fact), or that
    each descent step got slower (a cache fact). Only the third would be an
    argument against porting selection, and only the third is invisible in a
    share-of-total table.
    """
    if not row or not clean.get("sims_per_move"):
        return None
    return {
        "sims_per_move": clean["sims_per_move"],
        "calls_per_sim": row["calls_per_move"] / clean["sims_per_move"],
        "us_per_call": row["us_per_call"],
        "calls_per_move_scaled": row["calls_per_move"],
        "ms_per_move_scaled": row["ms_per_move"],
    }


def report_selection(arms, cleans, derived):
    """The question #44 was actually asked: what happened to `_best_child`?"""
    print()
    print("=" * 78)
    print("_BEST_CHILD -- MORE CALLS, OR SLOWER CALLS?")
    print("=" * 78)
    print("  ms/move = simulations/move x descents/simulation x us/descent")
    print()
    print("  %-26s %10s %10s %9s %10s"
          % ("", "sims/move", "calls/sim", "us/call", "ms/move"))
    rows = {}
    for arm in arms:
        f = factors(arm, cleans[arm_key(arm)],
                    derived[arm_key(arm)]["rows"].get("_best_child"))
        if f is None:
            continue
        rows[arm_key(arm)] = f
        print("  %-26s %10.1f %10.3f %9.3f %10.2f"
              % ("%s (%s)" % (arm["engine"], arm["workload"]),
                 f["sims_per_move"], f["calls_per_sim"], f["us_per_call"],
                 f["ms_per_move_scaled"]))
    print()
    print("  RESULT_TREE_PROFILE measured 108.0 ms/move on `pocket_r35` at")
    print("  4,929 sims/move, in a same-network match. It is not comparable")
    print("  line for line -- it charged the wrapper at the tight-loop price,")
    print("  which the in-situ section below shows is roughly half the real")
    print("  one, so it OVERSTATES this row. The controlled comparison is the")
    print("  pair below: both arms, same positions, same instrument.")

    out = {}
    for workload in ("fixed", "game"):
        a = rows.get("pocket_r35/%s" % workload)
        b = rows.get("pocket_graph/%s" % workload)
        if not (a and b):
            continue
        d = {"sims": pct(a["sims_per_move"], b["sims_per_move"]),
             "per_sim": pct(a["calls_per_sim"], b["calls_per_sim"]),
             "us": pct(a["us_per_call"], b["us_per_call"]),
             "ms": pct(a["ms_per_move_scaled"], b["ms_per_move_scaled"])}
        out[workload] = d
        print()
        print("  pocket_r35 -> pocket_graph, %s" % workload)
        print("    simulations per move  %+7.1f%%  more search fits in the"
              " second" % d["sims"])
        print("    descents per sim      %+7.1f%%  the tree got deeper"
              % d["per_sim"])
        print("    us per descent        %+7.1f%%  cost of one descent step"
              % d["us"])
        print("    ---------------------------------")
        print("    ms per move           %+7.1f%%" % d["ms"])
        verdicts = []
        if abs(d["us"]) < 10.0:
            verdicts.append("the per-call cost did NOT materially change")
        else:
            verdicts.append("the per-call cost moved %+.1f%%" % d["us"])
        verdicts.append("so the growth is %s"
                        % ("almost entirely more calls"
                           if abs(d["us"]) < abs(d["sims"] + d["per_sim"]) / 3
                           else "a mix of more calls and slower calls"))
        print("    -> %s, %s." % (verdicts[0], verdicts[1]))
    return out


def cross_checks(arm):
    """Counts that MUST agree if the attribution is right.

    Each of these is an identity in `agents/mcts.py`, not an expectation:
    the descent calls `state.make_move` exactly once per `_best_child`, the
    probes clone once per legal move they examine, and `_mark_terminal_children`
    runs once per expanded leaf. A mismatch means a wrapper is catching calls
    from somewhere this report does not think they come from, which would move
    real milliseconds into the wrong row.
    """
    callers = arm.get("callers") or {}
    ops = arm.get("ops") or {}
    moves = arm["moves"]

    def cs(key):
        return callers.get(key, {}).get("calls", 0) / moves if moves else 0.0

    checks = [
        ("descent make_move == _best_child",
         cs("wave loop -> state.make_move"),
         ops.get("_best_child", {}).get("calls_per_move", 0.0)),
        ("probe clone == probe make_move",
         cs("terminal probes -> state clone"),
         cs("terminal probes -> state.make_move")),
        ("terminal probes == legal-move calls",
         ops.get("terminal probes", {}).get("calls_per_move", 0.0),
         ops.get("legal moves", {}).get("calls_per_move", 0.0)),
    ]
    print()
    print("  internal consistency (identities in agents/mcts.py, not"
          " expectations)")
    ok = True
    for label, a, b in checks:
        agree = abs(a - b) <= max(1.0, 0.01 * max(a, b))
        ok = ok and agree
        print("    %-38s %10.1f %10.1f  %s"
              % (label, a, b, "ok" if agree else "*** MISMATCH ***"))
    return ok


def pct(a, b):
    return 100.0 * (b - a) / a if a else float("nan")


def arm_key(arm):
    return "%s/%s" % (arm["engine"], arm["workload"])


def verdict(arms, derived, cleans):
    """Is selection still the dominant recoverable host term?"""
    print()
    print("=" * 78)
    print("#44 VERDICT")
    print("=" * 78)
    cand = [a for a in arms
            if a["engine"] == "pocket_graph" and a["workload"] == "game"]
    cand = cand or [a for a in arms if a["engine"] == "pocket_graph"]
    if not cand:
        print("  no pocket_graph arm was run")
        return {}
    arm = cand[0]
    d = derived[arm_key(arm)]
    rank = sorted(((v["ms_per_move"], n) for n, v in d["rows"].items()
                   if n not in DEVICE_OPS), reverse=True)
    print("  host cost of %s (%s), scaled, largest first:"
          % (arm["engine"], arm["workload"]))
    for ms, name in rank[:6]:
        print("    %-30s %7.2f ms/move" % (name, ms))
    bc = d["rows"].get("_best_child", {}).get("ms_per_move", 0.0)
    probe = d["probes_inclusive_ms_per_move"]
    dev = d["device_ms_per_move"]
    # The CLEAN arm's wall, the same denominator the tables use. The timed
    # arm's own wall is not it -- under a clock, moves that stop early on a
    # proof land differently between runs, so the two differ by a few percent.
    total = cleans[arm_key(arm)]["wall_ms_per_move"]
    print()
    print("  _best_child          %7.2f ms/move  %5.1f%% of the move"
          % (bc, 100.0 * bc / total))
    print("  terminal probes      %7.2f ms/move  %5.1f%%  (inclusive)"
          % (probe, 100.0 * probe / total))
    print("  whole device path    %7.2f ms/move  %5.1f%%"
          % (dev, 100.0 * dev / total))
    top = rank[0][1] if rank else None
    print()
    if top == "_best_child":
        print("  `_best_child` IS the largest recoverable host term. #36 is a")
        print("  selection-engine project: port the PUCT primitive only.")
    else:
        print("  `_best_child` is NOT the largest host term -- %s is, at"
              " %.2f ms/move." % (top, rank[0][0]))
        print("  Re-read the scope of #36 before porting selection.")
    return {"dominant_host_op": top,
            "best_child_ms_per_move": bc,
            "probes_inclusive_ms_per_move": probe,
            "device_ms_per_move": dev,
            "move_ms": total,
            "ranked": [{"op": n, "ms_per_move": ms} for ms, n in rank]}


def report_insitu(prices, cleans):
    """The price of the selection wrapper, and the slope it hands #45."""
    print()
    print("=" * 78)
    print("IN-SITU PRICE OF ONE WRAPPER, AND WHAT A MICROSECOND OF SELECTION"
          " BUYS")
    print("=" * 78)
    print("  One run per arm with ONLY `_best_child` wrapped. The search is")
    print("  deadline-bound, so the wrapper does not lengthen a move -- it")
    print("  costs simulations, and the shortfall prices it exactly.")
    print()
    print("  %-24s %10s %11s %10s %11s"
          % ("", "us/call", "ms/move", "sims lost", "vs tight loop"))
    out = {}
    for key, pr in sorted(prices.items()):
        if pr is None:
            continue
        cal = pr.get("calibrated_us_per_call") or float("nan")
        print("  %-24s %10.3f %11.2f %9.1f%% %10.1fx"
              % (key, pr["us_per_call"], pr["cost_ms_per_move"],
                 pr["sims_lost_pct"], pr["us_per_call"] / cal))
        out[key] = pr
    print()
    print("  The tight-loop calibration is the LOWER bound it was always")
    print("  documented to be. Note what it is and is not used for: the row")
    print("  charge is the INSIDE half of it, which is small and directly")
    print("  measured; this column is the whole per-call cost, most of which")
    print("  never lands in any row and is removed as the deflation.")
    return out


def report_solo(solos, cleans, counts):
    """`_best_child` read off the run that barely perturbs it.

    The full run wraps sixteen functions and loses about 30% of its search, so
    every row in it carries a 1.4x scaling and a 0.9x deflation. The solo run
    wraps ONE function and loses about 5%, so its `_best_child` number needs
    almost no correction at all. When the two disagree, this is the one to
    believe -- it is the same measurement with a fifth of the perturbation.
    """
    print()
    print("=" * 78)
    print("_BEST_CHILD, MEASURED WITH ONLY ITSELF WRAPPED")
    print("=" * 78)
    print("  %-24s %10s %10s %9s %10s %8s"
          % ("", "own calls", "us/call", "own ms", "ms/move", "of move"))
    out = {}
    for solo in solos:
        clean = cleans.get(arm_key(solo))
        if clean is None:
            continue
        price_arm(solo, clean)
        o = solo["ops"].get("_best_child")
        if not o:
            continue
        calls, ms = per_move(solo, counts.get(arm_key(solo)), "_best_child")
        out[solo["engine"]] = {
            "calls_per_move": calls,
            "us_per_call": o["us_per_call"],
            "ms_per_move": ms,
            "share_of_move": ms / clean["wall_ms_per_move"],
            "instrument_pct": 100.0 * (1.0 - 1.0 / solo["scale"]),
        }
        print("  %-24s %10.1f %10.3f %9.2f %10.2f %7.1f%%"
              % (solo["engine"], o["calls_per_move"], o["us_per_call"],
                 o["exclusive_ms_per_move"], ms,
                 100.0 * ms / clean["wall_ms_per_move"]))
    print()
    print("  Instrument cost in these runs: %s -- against ~30%% in the full"
          % ", ".join("%s %.1f%%" % (e, v["instrument_pct"])
                      for e, v in out.items()))
    print("  table above, which is why these are the numbers to quote.")
    return out


def report_elasticity(el, engine):
    if not el:
        return
    print()
    print("  Measured slope for %s: one millisecond off the selection path is"
          % engine)
    print("  worth %.1f network evaluations a move. `_best_child` costs"
          % el["nn_per_ms_of_selection"])
    print("  %.1f ms, so a free selection primitive is worth about %.0f more"
          % (el["best_child_ms_per_move"],
             el["nn_per_ms_of_selection"] * el["best_child_ms_per_move"]))
    print("  network evaluations a move -- %+.1f%% (%+.1f%% by simulations)."
          % (el["pct_more_search"], el["pct_more_sims"]))
    print("  That is the CEILING: a native primitive is not free -- it still")
    print("  has to read the mirrored arrays and cross the boundary once per")
    print("  descent -- and it is one arm of one run, not an interval.")


def render(payload):
    derived = {}
    arms = payload["arms"]
    cleans = {a["engine"] + "/" + a["workload"]: a for a in payload["clean"]}
    solos = payload.get("solo") or []
    counts = {arm_key(c): c for c in payload.get("count") or []}
    for key, c in counts.items():
        if key in cleans:
            c["scale"] = scale_of(cleans[key], c)
    prices = {}
    for solo in solos:
        clean = cleans.get(arm_key(solo))
        if clean is None:
            continue
        pr = insitu_price(solo, clean, "_best_child")
        if pr:
            pr["calibrated_us_per_call"] = solo["calibrated_us_per_call"]
            # One wrapper's price is a property of the wrapper, so the fixed
            # run's figure is reused for the same engine's game arm.
            prices[solo["engine"]] = pr
    by_op = {e: {"_best_child": p} for e, p in prices.items()}

    for arm in arms:
        clean = cleans[arm_key(arm)]
        derived[arm_key(arm)] = report_arm(arm, clean, counts.get(arm_key(arm)))
    payload["best_child"] = report_selection(arms, cleans, derived)
    payload["insitu"] = report_insitu(prices, cleans) if prices else {}
    solo_bc = report_solo(solos, cleans, counts) if solos else {}
    payload["best_child_solo"] = solo_bc

    el = {}
    for engine, pr in prices.items():
        bc = solo_bc.get(engine)
        clean = cleans.get("%s/fixed" % engine)
        if not (bc and clean):
            continue
        # The solo figure, not the full run's: the elasticity multiplies it, so
        # the more perturbed number would be amplified rather than averaged.
        e = elasticity(pr, bc["ms_per_move"], clean)
        el[engine] = e
        report_elasticity(e, engine)
    payload["elasticity"] = el
    payload["derived"] = derived
    payload["verdict"] = verdict(arms, derived, cleans)
    return payload


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="all", choices=("fixed", "game", "all"))
    ap.add_argument("--arms", default="pocket_r35,pocket_graph")
    ap.add_argument("--opponent", default="final",
                    help="cross-network opponent for --mode game; a mirror "
                         "inflates inherited trees and overstates every "
                         "per-node cost")
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--position-games", type=int, default=3)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SELECT_SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--tag", default="select")
    ap.add_argument("--rerender", default="",
                    help="re-print the report from a saved JSON, no games")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.rerender:
        with open(args.rerender) as fh:
            payload = json.load(fh)
        render(payload)
        return

    assert_frozen_sources()
    names = [s.strip() for s in args.arms.split(",") if s.strip()]

    with single_instance("profile_selection"):
        base = gpu_baseline()
        if base:
            print("[..] GPU load before starting: %.0f%% mean, %.0f%% peak"
                  % (base["mean_pct"], base["max_pct"]))
        # Deployment conditions. `gc.disable()` is what tools/regress_engine
        # does and what the frozen p99 was measured under; passing
        # gc_mode="deferred" alone only ADDS a boundary collect and leaves
        # automatic collection running.
        gc.disable()
        print("[!] automatic cyclic GC off; collecting at boundaries only")

        payload = {"tag": args.tag, "mode": args.mode, "arms": [], "clean": [],
                   "solo": [], "count": [],
                   "seed": args.seed, "positions": args.positions,
                   "games": args.games, "opponent": args.opponent,
                   "device": args.device,
                   "git_head": engine_registry.git_head(),
                   "environment": engine_registry.environment(),
                   "environment_drift": engine_registry.env_drift(),
                   "gpu_baseline_before_run": base}
        path = os.path.join(OUT_DIR, "%s.json" % args.tag)

        def save():
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)

        positions = []
        if args.mode in ("fixed", "all"):
            print("[..] sampling positions from %d games" % args.position_games)
            positions = positions_for(names[0], args.position_games,
                                      args.device, args.seed, args.positions)
            print("[..] %d fixed positions" % len(positions))
            payload["n_positions"] = len(positions)
            for name in names:
                for wrapped in (False, True):
                    print("[..] fixed / %s / %s" % (name, "wrapped" if wrapped
                                                    else "clean"))
                    res = run_fixed(name, positions, args.device, wrapped)
                    payload["arms" if wrapped else "clean"].append(res)
                    save()
                # Counting arm: an increment per call, no clock. It supplies
                # every row's calls/move at ~1% perturbation, so no row has to
                # be scaled by a simulation-rate ratio.
                print("[..] fixed / %s / counting" % name)
                payload["count"].append(
                    run_fixed(name, positions, args.device, True,
                              counting=True))
                save()
                # In-situ calibration: the same positions with ONLY
                # `_best_child` wrapped, which prices that wrapper where it is
                # used and measures how much search a millisecond on the
                # selection path is worth.
                print("[..] fixed / %s / in-situ _best_child only" % name)
                payload["solo"].append(
                    run_fixed(name, positions, args.device, True,
                              only={"_best_child"}))
                save()

        if args.mode in ("game", "all"):
            for name in names:
                for wrapped in (False, True):
                    print("[..] game / %s vs %s / %s"
                          % (name, args.opponent,
                             "wrapped" if wrapped else "clean"))
                    res = run_game(name, args.opponent, args.games,
                                   args.device, args.seed + 1, wrapped)
                    payload["arms" if wrapped else "clean"].append(res)
                    save()
                print("[..] game / %s vs %s / counting"
                      % (name, args.opponent))
                payload["count"].append(
                    run_game(name, args.opponent, args.games, args.device,
                             args.seed + 1, True, counting=True))
                save()

        render(payload)
        save()
        print()
        print("[OK] wrote %s" % path)


if __name__ == "__main__":
    main()
