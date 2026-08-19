"""#47 -- what the one-ply terminal probes COST, and what they BUY.

`RESULT_SELECTION_PROFILE.md` priced the probes at 118.8 ms/move inclusive on
`pocket_graph` and called the recoverable part "roughly 42 ms of Python loop
plus pybind transitions". Two things have happened since: native selection and
deferred retirement both landed, the engine now runs materially more search per
second, and `pocket_defer` was promoted. Every per-move figure in that sentence
was measured on a different engine.

But re-pricing the cost is only half a decision, and it is the half that cannot
answer the question. A native implementation that makes 35,000 terminal checks
cheap is worth building only if those checks are doing enough proving to be
worth keeping AT ALL. So this tool measures BENEFIT DENSITY alongside cost, and
then ablates the whole subsystem.

    python -m tools.probe_ablation --mode all

WHAT `solve=0` ACTUALLY TURNS OFF, stated up front because the arm is named for
the probes and removes more than them: the one-ply probes at expansion, the
marking of terminal nodes met during descent, the proof propagation those feed,
the proven-win/refuted-loss short circuit in selection (`ChildArray` is
constructed with the flag, so the native selector drops it too), the early
return once the root is proven, and the one-hot root policy correction. It is
the ablation of the FEATURE. The probes are the only thing that manufactures
proofs at expansion, so there is no way to remove them alone today -- and
building a knob for that is a decision this measurement is supposed to inform,
not a prerequisite for it.

ARMS, and the `repeat` one is the one most easily forgotten:

  clean      `pocket_defer`, no instrumentation. The wall time, search rate and
             latency every share is taken of.
  counting   the same engine with counters and no clock: probe roots, legal
             children probed, terminal hits, node proofs, propagation depth,
             and pybind crossings split by whether a probe caused them.
  timed      a clock on the probe loop, the two primitives it calls and the #48
             predicate, and nothing else. Supplies us/probed child; its own
             counts are not used, because an instrumented deadline-bound search
             does less work and therefore makes fewer calls.
  off        `pocket_defer+solve=0`, no instrumentation. The ablation.
  repeat     `pocket_defer` AGAIN, no instrumentation.
  sel_*      the same clean/counting/timed triple against `pocket_filter`, the
             #48 candidate. `--mode gate`.

#48 ANSWERED THE QUESTION THIS TOOL ASKED, and the answer was the third option:
neither keep the probes as they are nor delete them, but SKIP them where
`could_end` proves they cannot find anything. That predicate now lives in
`agents.mcts` and is imported here rather than copied, and `--mode gate` is
what checks that the shipped version buys what `--mode filter` predicted.

THE REPEAT ARM IS THE POINT. Move disagreement between ON and OFF is
meaningless without knowing what two runs of the SAME engine disagree on. These
searches are wall-clock bound, so the simulation count differs run to run and
so, sometimes, does the argmax. Reporting "ON and OFF chose differently on 12%
of positions" without the ON-vs-ON floor underneath it would be reporting the
clock. The floor is measured here and every disagreement is quoted against it.

FIXED POSITIONS, BARE ROOT, and that is the right workload for once. The
question is what a probe does inside a search, not what it does to a game, and
the brief says explicitly to do this first as a throughput/behaviour study
rather than an arena match. It also makes the ON and OFF arms answerable on
IDENTICAL inputs, which a game cannot do -- the moment the two arms differ, the
positions differ, and a disagreement rate stops meaning anything.

WHAT THIS TOOL CANNOT SAY. A disagreement has no sign. If ON and OFF pick
different moves, this measures how often, not which was better. That is a
strength question and it costs a match; the point of running this first is to
find out whether one is worth funding.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import time

import numpy as np
import torch

from agents import mcts as _mcts_mod
from agents.mcts import MCTS, TreeReuseSearcher, could_end
from engine.game import GameState
from tools import engine_registry
from tools.arena_1s import TimedPlayer, phase_of, play_match
from tools.profile_selection import AttributedTimer
from tools.profile_tree import assert_frozen_sources
from tools.runlock import gpu_baseline, single_instance

OUT_DIR = os.path.join("results", "probe_ablation")

# Instrumented and unscored, like `profile`, `select` and `release`.
PROBE_SEED = engine_registry.SEEDS["probe"]

# The arms are derived from whatever is deployed, not from a hard-coded name.
# This tool exists to ask a question ABOUT the baseline, so it has to follow it.
ON_SPEC = "engine:%s" % engine_registry.DEPLOYED
OFF_SPEC = "engine:%s+solve=0" % engine_registry.DEPLOYED

# #48c's gate arms. BOTH SIDES LITERAL, unlike the two above, and this is the
# registry's stated rule rather than an inconsistency: a published comparison
# has to keep building the same two engines forever. Deriving the base from
# `DEPLOYED` would have been fine on the day it ran and self-defeating the day
# after -- `pocket_filter` was promoted on the strength of this measurement, so
# a `DEPLOYED`-derived base would now build the candidate against itself and
# report a dead heat.
#
# THE POINT OF PUTTING THEM IN THIS FILE rather than in a new gate tool is that
# the base arms and the selective arms are then measured by the same
# instrument, priced by the same `calibrate()`, on the same positions. A
# separately written gate would compare two engines through two instruments and
# call the difference an optimisation.
GATE_BASE_SPEC = "engine:pocket_defer"
SEL_SPEC = "engine:pocket_filter"

# The equal-clock ablation is CONFOUNDED and cannot be un-confounded in place:
# the OFF arm gets 18.9% more search, so a move that changes might have changed
# because the probes were removed or because the search got deeper. These two
# arms remove the clock. `ms=0` makes the search simulation-bound, at which
# point both arms do the SAME amount of work and any disagreement is the
# feature's own decision effect.
#
# It is 800 simulations because that is the sim-bound default and `sims` is not
# a key the deployment engine pins, so it cannot be overridden without changing
# what the base engine declares. That is roughly a tenth of the tree the
# engine builds in a second, which FAVOURS the probes -- less search means more
# room for a one-ply proof to matter -- so a null here is the strong direction.
FIXED_ON_SPEC = "engine:%s+ms=0" % engine_registry.DEPLOYED
FIXED_OFF_SPEC = "engine:%s+ms=0,solve=0" % engine_registry.DEPLOYED

# Wrapped by the `timed` arm. Deliberately three names and not sixteen: the
# whole-engine breakdown is tools/profile_selection's job, and every extra
# wrapper here costs search that the probe rows would then be scaled against.
TIMED_TARGETS = (
    ("terminal probes", MCTS, "_mark_terminal_children"),
    ("state clone", GameState, "clone"),
    ("state.make_move", GameState, "make_move"),
    # #48. The predicate is what the selective arm pays INSTEAD of the loop, so
    # leaving it unwrapped would let its cost hide inside "terminal probes" and
    # make the saving look bigger than it is. It is nested inside the probe, so
    # `price()` charges the probe its whole wrapper -- see the `nested` term.
    ("could_end", _mcts_mod, "could_end"),
)


# ----------------------------------------------------------------------
# Counters
# ----------------------------------------------------------------------

class ProbeCounters:
    """Everything the benefit question needs, counted and not inferred.

    THE CROSSING COUNT IS MEASURED, NOT ASSERTED. Reading the loop suggests
    four transitions into C++ per probed child -- the copy constructor behind
    `clone`, `_CppGameState.make_move`, the `_raw_winner()` behind the `winner`
    property that `make_move` reads for its return tuple, and the second
    `_raw_winner()` behind the `probe.winner` test. A fifth on a terminal hit,
    from `_terminal_value`. That reading is worth exactly nothing until the
    counters agree with it, because "obviously it does N" is how a profile ends
    up measuring the wrong thing.
    """

    # `roots` is roots CONSIDERED and `scanned` is roots the loop actually ran
    # on. They are equal for any engine without the #48 filter, and keeping
    # them as two fields rather than deriving one from the other is what lets
    # the same instrument read both engines: `children` then means "children
    # actually probed" on both sides, which is the quantity every per-child
    # figure below divides by.
    FIELDS = ("roots", "scanned", "roots_skipped", "children", "hits",
              "roots_with_hit", "roots_proved",
              "propagations", "levels", "clone_probe", "clone_other",
              "make_probe", "make_other", "winner_probe", "winner_other")

    def __init__(self):
        for f in self.FIELDS:
            setattr(self, f, 0)

    def crossings(self):
        return {"probe": self.clone_probe + self.make_probe + self.winner_probe,
                "other": self.clone_other + self.make_other + self.winner_other}

    def as_dict(self):
        d = {f: getattr(self, f) for f in self.FIELDS}
        d["crossings"] = self.crossings()
        return d


def instrument(counters, ctx):
    """Patch the probe path with counters. Returns a restore callable.

    `GameState` is a plain Python subclass of the pybind type -- `winner` is a
    property IT defines, and `clone`/`make_move` are its own methods -- so all
    three patch normally. `_CppGameState` is untouched: the crossings are
    counted at the boundary the Python side owns.
    """
    in_probe = [False]
    saved = []

    raw_probe = MCTS.__dict__["_mark_terminal_children"]

    def counting_probe(self, node, state):
        if not ctx["on"]:
            return raw_probe(self, node, state)
        kids = node.children
        counters.roots += 1
        skips0 = self.stat_probe_skips
        in_probe[0] = True
        try:
            raw_probe(self, node, state)
        finally:
            in_probe[0] = False
        if self.stat_probe_skips != skips0:
            # #48 skipped this root, so nothing was probed. Attributing its
            # children to `children` would credit the filtered engine with work
            # it did not do and make every per-child figure below meaningless.
            # The skip is read off production's OWN counter rather than
            # re-evaluating the predicate here, which would be an instrument
            # that agrees with itself.
            counters.roots_skipped += 1
            return
        counters.scanned += 1
        counters.children += len(kids)
        # A child is only `solved` here if this call proved it: children are
        # created by the expansion that immediately precedes this, so there is
        # no earlier proof to confuse with one of ours.
        hits = sum(1 for c in kids.values() if c.solved is not None)
        counters.hits += hits
        counters.roots_with_hit += bool(hits)
        counters.roots_proved += node.solved is not None

    raw_prop = MCTS.__dict__["_propagate_solved"]

    def counting_prop(self, node):
        if not ctx["on"]:
            return raw_prop(self, node)
        # The chain production will actually consider, captured with
        # production's own stopping rule rather than a copy of its body.
        chain = []
        cur = node.parent
        while cur is not None and cur.solved is None:
            chain.append(cur)
            cur = cur.parent
        raw_prop(self, node)
        counters.propagations += 1
        counters.levels += sum(1 for n in chain if n.solved is not None)

    raw_clone = GameState.__dict__["clone"]

    def counting_clone(self):
        if ctx["on"]:
            if in_probe[0]:
                counters.clone_probe += 1
            else:
                counters.clone_other += 1
        return raw_clone(self)

    raw_make = GameState.__dict__["make_move"]

    def counting_make(self, idx):
        if ctx["on"]:
            if in_probe[0]:
                counters.make_probe += 1
            else:
                counters.make_other += 1
        return raw_make(self, idx)

    raw_winner = GameState.__dict__["winner"]
    fget = raw_winner.fget

    def counting_winner(self):
        if ctx["on"]:
            if in_probe[0]:
                counters.winner_probe += 1
            else:
                counters.winner_other += 1
        return fget(self)

    for holder, attr, new in ((MCTS, "_mark_terminal_children", counting_probe),
                              (MCTS, "_propagate_solved", counting_prop),
                              (GameState, "clone", counting_clone),
                              (GameState, "make_move", counting_make),
                              (GameState, "winner", property(counting_winner))):
        saved.append((holder, attr, holder.__dict__[attr]))
        setattr(holder, attr, new)

    def restore():
        for holder, attr, raw in saved:
            setattr(holder, attr, raw)
    return restore


# NOTE ON `could_end`. This tool PROPOSED the predicate in #47 and carried its
# own copy; #48 moved it into the engine and this module now imports it (see
# the import block at the top). The direction matters: a local copy would leave
# the false-negative count below measuring a function production does not run,
# which is exactly the failure this instrument exists to prevent. Re-running
# `--mode filter` therefore now measures the shipped predicate.


class FilterCounters:
    FIELDS = ("roots", "passed", "found", "false_negatives",
              "passed_and_found", "children_all", "children_passed")

    def __init__(self):
        for f in self.FIELDS:
            setattr(self, f, 0)

    def as_dict(self):
        return {f: getattr(self, f) for f in self.FIELDS}


def instrument_filter(counters, ctx):
    """Evaluate `could_end` at every probe root, then run the real probe.

    Records what the filter WOULD have skipped and, critically, whether it ever
    would have skipped a root the probe found something in.
    """
    raw = MCTS.__dict__["_mark_terminal_children"]

    def checked(self, node, state):
        if not ctx["on"]:
            return raw(self, node, state)
        kids = node.children
        n = len(kids)
        keep = could_end(state.mini_winners, state.player)
        counters.roots += 1
        counters.children_all += n
        if keep:
            counters.passed += 1
            counters.children_passed += n
        raw(self, node, state)
        found = any(c.solved is not None for c in kids.values())
        if found:
            counters.found += 1
            if keep:
                counters.passed_and_found += 1
            else:
                counters.false_negatives += 1

    MCTS._mark_terminal_children = checked

    def restore():
        MCTS._mark_terminal_children = raw
    return restore


def patch_timed(timer):
    saved = []
    for name, holder, attr in TIMED_TARGETS:
        raw = holder.__dict__[attr]
        saved.append((holder, attr, raw))
        setattr(holder, attr, timer.wrap(name, raw))

    def restore():
        for holder, attr, raw in saved:
            setattr(holder, attr, raw)
    return restore


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def positions_for(spec, games, device, seed, want):
    """Real positions from real play, sampled by an engine neither arm is.

    Uses the DEPLOYED engine with probes ON, which is the distribution the
    question is about: what the probes do in the positions the shipping engine
    actually reaches. An `off` engine would sample a different game tree and
    then the ablation would be answering about positions it does not visit.
    """
    pa = TimedPlayer(spec, device)
    pb = TimedPlayer(spec, device)
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


def run_arm(spec, positions, device, label, counters=None, timer=None,
            filt=None, warmup=2):
    """Search each position for the full deadline, from a bare root.

    No tree reuse and no divergence, so both arms see byte-identical inputs and
    a disagreement is attributable to the search rather than to the game having
    gone somewhere else.
    """
    p = TimedPlayer(spec, device)
    ctx = {"on": False}
    restore = None
    if counters is not None:
        restore = instrument(counters, ctx)
    elif filt is not None:
        restore = instrument_filter(filt, ctx)
    elif timer is not None:
        timer.ctx = ctx
        restore = patch_timed(timer)
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
                pi, root = p.mcts.search(state.clone())
            dt = (time.perf_counter() - t0) * 1000.0
            ctx["on"] = False
            last = p.mcts.last
            rows.append({
                "phase": phase,
                "search_ms": dt,
                "move": int(np.argmax(pi)),
                "sims": last["simulations_completed"],
                "nn": last["neural_evaluations"],
                "expansions": last["nodes_expanded"],
                "probes": last["probes"],
                "probe_roots": last["probe_roots"],
                "probe_skips": last["probe_skips"],
                "root_solved": last["root_solved"],
                "root_n": last["root_n"],
            })
            TreeReuseSearcher.release(root)
            gc.collect()
    finally:
        if restore is not None:
            restore()

    out = {"label": label, "spec": spec, "rows": rows,
           "budget_ms": p.mcts.time_budget_ms,
           "reserve_ms": p.mcts.reserve_ms,
           "solve": p.mcts.solve,
           "fingerprint": p.provenance.get("fingerprint"),
           "params": p.net_info["params"]}
    if counters is not None:
        out["counters"] = counters.as_dict()
    if filt is not None:
        out["filter"] = filt.as_dict()
    if timer is not None:
        # `ExclusiveTimer` accumulates `time.perf_counter()` deltas, which are
        # SECONDS. The wrapper price is in microseconds. Storing the raw floats
        # under a `_ms` name and then subtracting a millisecond correction from
        # them produced a NEGATIVE probe cost on the first run -- so the
        # conversion happens once, here, at the only place that knows the unit.
        out["timer"] = {
            "exclusive_ms": {k: v * 1000.0 for k, v in timer.total.items()},
            "inclusive_ms": {k: v * 1000.0 for k, v in timer.inclusive.items()},
            "calls": dict(timer.calls),
            "calls_from": {"%s>%s" % k: v[0]
                           for k, v in timer.calls_from.items()},
            "inside_us": getattr(timer, "inside_us", None),
            "total_us": getattr(timer, "price_us", None),
        }
    return out


def price(timer_blob, moves):
    """Subtract the instrument from the probe rows. This is not optional.

    The probe loop calls `clone` and `make_move` about 35,000 times a move, and
    both are wrapped. At roughly a microsecond of wrapper apiece that is tens of
    milliseconds per move of pure instrument sitting INSIDE the interval the
    probe is timed over. Reporting the raw number would price the profiler and
    call it the probe.

    `calibrate` splits the wrapper price at the measurement boundary
    (`tools/profile_selection.AttributedTimer.calibrate` explains why): `inside`
    is what a wrapped call adds to its OWN recorded time, and the rest lands in
    whatever wrapped function is above it on the stack. So a nested call charges
    its parent the WHOLE price, and a function charges itself only `inside`.
    """
    inside = timer_blob.get("inside_us") or 0.0
    total = timer_blob.get("total_us") or 0.0
    outside = max(0.0, total - inside)
    n_probe = timer_blob["calls"].get("terminal probes", 0)
    nested = (timer_blob["calls_from"].get("terminal probes>state clone", 0)
              + timer_blob["calls_from"].get(
                  "terminal probes>state.make_move", 0)
              # #48: one more nested wrapper, and it fires on EVERY root rather
              # than per child. On the filtered engine it is the largest of the
              # three by call count on skipped roots, so omitting it would
              # under-price the instrument exactly where the saving is claimed.
              + timer_blob["calls_from"].get("terminal probes>could_end", 0))
    incl = timer_blob["inclusive_ms"].get("terminal probes", 0.0)
    excl = timer_blob["exclusive_ms"].get("terminal probes", 0.0)
    wrapper = 1e-3 * (inside * n_probe + total * nested)
    # The guard that would have caught the seconds-for-milliseconds bug on the
    # first run instead of after it printed "-1.905 us per probed child". A
    # correction larger than the thing being corrected is never a small error.
    if n_probe and wrapper >= incl:
        raise SystemExit(
            "[X] the instrument correction (%.1f) is larger than the measured "
            "probe time (%.1f).\n    Either the timer is not in milliseconds "
            "or the wrapper price is wrong; both make every probe figure "
            "meaningless." % (wrapper, incl))
    return {
        "calls": n_probe,
        "nested_calls": nested,
        # Passed through so `units` can split the per-ROOT nested wrapper from
        # the two per-CHILD ones without re-reading the timer blob.
        "timer_calls": dict(timer_blob["calls_from"]),
        "raw_inclusive_ms": incl,
        "raw_exclusive_ms": excl,
        # Inclusive holds the nested calls, so it pays their whole price.
        "inclusive_ms": incl - wrapper,
        # Exclusive already had the nested DT subtracted, but not the part of
        # each nested wrapper that ran outside its own clock.
        "exclusive_ms": excl - 1e-3 * (inside * n_probe + outside * nested),
        "wrapper_ms": wrapper,
        "moves": moves,
    }


# ----------------------------------------------------------------------
# Derivation
# ----------------------------------------------------------------------

def rate(arm):
    """Search rate, composition-robust.

    `nn/move` is not usable here and this is the tool where that bites hardest.
    A search whose root is proven returns almost immediately, so it contributes
    a near-zero nn count to the mean -- and the ablation's whole point is that
    one arm can do that and the other cannot. Quoting nn/move would score the
    probes for finishing early, which buys nothing at a fixed deadline: the
    same proven move is simply returned sooner.

    So: nn PER SECOND OF SEARCH, times the deadline the engine is allowed.
    `uttt-deferred-retirement-wins` has this replicating to 1.7-6.1% where
    nn/move moved 22% on one engine.
    """
    rows = arm["rows"]
    nn = sum(r["nn"] for r in rows)
    sec = sum(r["search_ms"] for r in rows) / 1000.0
    # A fixed-SIMULATION arm has no deadline, so there is no clock to multiply
    # by. Its nn_full is defined as its nn/move, which is the honest reading:
    # the arm is not competing for time, it is doing a fixed amount of work.
    deadline = (((arm["budget_ms"] - arm["reserve_ms"]) / 1000.0)
                if arm["budget_ms"] else (sec / len(rows) if rows else 0.0))
    nn_s = nn / sec if sec else 0.0
    return {
        "moves": len(rows),
        "nn_per_move": nn / len(rows) if rows else 0.0,
        "nn_per_second": nn_s,
        "nn_full": nn_s * deadline,
        "sims_per_move": float(np.mean([r["sims"] for r in rows])) if rows
                         else 0.0,
        "expansions_per_move": float(np.mean([r["expansions"] for r in rows]))
                               if rows else 0.0,
        "probes_per_move": float(np.mean([r["probes"] for r in rows]))
                           if rows else 0.0,
        "p50_ms": float(np.percentile([r["search_ms"] for r in rows], 50))
                  if rows else 0.0,
        "p99_ms": float(np.percentile([r["search_ms"] for r in rows], 99))
                  if rows else 0.0,
        "max_ms": max((r["search_ms"] for r in rows), default=0.0),
        "solved_roots": sum(1 for r in rows if r["root_solved"] is not None),
        "early_returns": sum(1 for r in rows
                             if r["root_solved"] is not None
                             and r["search_ms"] < 0.5 * arm["budget_ms"]),
    }


def disagreement(a, b):
    """Fraction of positions where the two arms chose a different move."""
    pairs = list(zip(a["rows"], b["rows"]))
    n = len(pairs)
    diff = sum(1 for x, y in pairs if x["move"] != y["move"])
    by_phase = collections.defaultdict(lambda: [0, 0])
    for x, y in pairs:
        e = by_phase[x["phase"]]
        e[0] += x["move"] != y["move"]
        e[1] += 1
    return {"n": n, "differ": diff, "rate": diff / n if n else 0.0,
            "by_phase": {k: {"differ": v[0], "n": v[1],
                             "rate": v[0] / v[1] if v[1] else 0.0}
                         for k, v in sorted(by_phase.items())}}


def units(count_arm, timed_arm, clean_arm):
    """The owner's probe work-unit list, one row each."""
    c = count_arm["counters"]
    moves = len(count_arm["rows"])
    p = price(timed_arm["timer"], len(timed_arm["rows"]))
    # us/child needs BOTH factors from a consistent place: the clock from the
    # timed arm and children-per-root from the counting arm. The timed arm's
    # own call count is not usable as a rate -- it lost search to the wrapper
    # and therefore probed less -- but the nested call count IS the number of
    # children it probed, exactly, since the loop clones once per child.
    scanned = c["scanned"] or c["roots"]
    per_root = c["children"] / scanned if scanned else 0.0
    # The probe loop clones once and make_moves once per PROBED child, so the
    # count of those two nested wrappers is exactly twice the children probed.
    # `could_end` is nested too but fires once per ROOT, so it is subtracted
    # out rather than folded into the child count -- getting this wrong would
    # inflate the denominator on the filtered arm and report a per-child cost
    # that fell for a reason that is not a per-child cost.
    per_root_nested = p["timer_calls"].get("terminal probes>could_end", 0)
    timed_children = (p["nested_calls"] - per_root_nested) / 2.0
    incl_us = (p["inclusive_ms"] * 1000.0 / timed_children
               if timed_children else 0.0)
    excl_us = (p["exclusive_ms"] * 1000.0 / timed_children
               if timed_children else 0.0)
    cross = c["crossings"]
    # The predicate's own cost, from the same timed arm and priced the same
    # way. `could_end` calls nothing that is wrapped, so it pays only the
    # `inside` half of the wrapper and its exclusive time needs no nesting
    # correction.
    tb = timed_arm["timer"]
    n_pred = tb["calls"].get("could_end", 0)
    pred_ms = tb["exclusive_ms"].get("could_end", 0.0)
    pred_us = (pred_ms * 1000.0 / n_pred - (tb.get("inside_us") or 0.0)
               if n_pred else 0.0)
    return {
        "probe_roots_per_move": c["roots"] / moves if moves else 0.0,
        "probe_roots_scanned_per_move": scanned / moves if moves else 0.0,
        "probe_roots_skipped_per_move": (c["roots_skipped"] / moves
                                         if moves else 0.0),
        "scan_rate": scanned / c["roots"] if c["roots"] else 0.0,
        "children_probed_per_move": c["children"] / moves if moves else 0.0,
        "children_per_root": per_root,
        "clones_per_move": (c["clone_probe"] + c["clone_other"]) / moves
                           if moves else 0.0,
        "clones_from_probes_per_move": c["clone_probe"] / moves if moves else 0,
        "make_move_per_move": (c["make_probe"] + c["make_other"]) / moves
                              if moves else 0.0,
        "make_move_from_probes_per_move": c["make_probe"] / moves if moves
                                          else 0.0,
        "us_per_probed_child_inclusive": incl_us,
        "us_per_probed_child_python_loop": excl_us,
        "crossings_per_probed_child": (cross["probe"] / c["children"]
                                       if c["children"] else 0.0),
        "crossings_from_probes_per_move": cross["probe"] / moves if moves
                                          else 0.0,
        "crossings_elsewhere_per_move": cross["other"] / moves if moves else 0.0,
        # Benefit density, in four levels of increasing strength. Only the
        # first three are answerable here; the fourth is the ablation.
        "hit_rate_per_child": c["hits"] / c["children"] if c["children"] else 0.0,
        "roots_with_any_hit": (c["roots_with_hit"] / c["roots"] if c["roots"]
                               else 0.0),
        "expansions_that_produce_a_proof": (c["roots_proved"] / c["roots"]
                                            if c["roots"] else 0.0),
        "propagations_per_move": c["propagations"] / moves if moves else 0.0,
        "levels_per_propagation": (c["levels"] / c["propagations"]
                                   if c["propagations"] else 0.0),
        "searches_with_a_proven_root": (
            sum(1 for r in clean_arm["rows"] if r["root_solved"] is not None)
            / len(clean_arm["rows"]) if clean_arm["rows"] else 0.0),
        # ms/move = children/move x us/child, and the two factors come from
        # DIFFERENT arms on purpose -- #44's rule. The clock has to come from
        # the timed arm, but that arm lost search to the wrapper and therefore
        # probed fewer children, so its own per-move count understates the
        # engine's. The counting arm supplies the count at ~1% perturbation.
        # Dividing the timed arm's total by its own move count would report the
        # instrumented engine's probe bill, not the deployed one's.
        "probe_ms_per_move_inclusive": incl_us * c["children"] / moves / 1000.0
                                       if moves else 0.0,
        "probe_ms_per_move_python_loop": excl_us * c["children"] / moves
                                         / 1000.0 if moves else 0.0,
        "instrument_ms_per_move_subtracted": (p["wrapper_ms"] / p["moves"]
                                              if p["moves"] else 0.0),
        "raw_probe_ms_per_move_before_pricing": (p["raw_inclusive_ms"]
                                                 / p["moves"]
                                                 if p["moves"] else 0.0),
        # What the filter itself costs, priced separately and INCLUDED in the
        # probe row above (it runs inside the probe's interval). Quoted on its
        # own because the honest question about a filter is not "how much did
        # the loop shrink" but "how much did the loop shrink NET of what the
        # test cost", and on a filtered engine the predicate fires on every one
        # of ~4,100 roots a move whether it admits or not.
        "predicate_us_per_root": pred_us,
        "predicate_ms_per_move": pred_us * c["roots"] / moves / 1000.0
                                 if moves else 0.0,
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def pct(a, b):
    return 100.0 * (a / b - 1.0) if b else float("nan")


def gate(payload, arms):
    """#48c. Did the filter buy what its own measurement predicted?

    THE PREDICTION IS NOT A HOPE, it is arithmetic on #47's numbers, and it is
    written down here so the run either meets it or has to explain itself. The
    probes were 126.9 ms of a 921.2 ms search and the filter keeps 17.52% of
    the per-child work, so it should hand back 82.48% of 13.8% of the search
    -- and removing a share `s` of a fixed clock buys 1/(1-s) - 1 more work,
    which is +12.8%.

    Falling well short of that does not mean the filter failed. It means the
    cost moved somewhere the probe row was hiding it -- the predicate, the
    `mini_winners` crossing it needs, or a term that was never probe-specific
    at all -- and the brief is explicit that finding out where comes BEFORE
    adding another optimisation.
    """
    on, sel = payload["rate"]["clean"], payload["rate"]["sel_clean"]
    u = payload.get("units")
    us = payload.get("units_selective")
    out = {
        "nn_full_gain": (sel["nn_full"] / on["nn_full"] - 1.0
                         if on["nn_full"] else 0.0),
        "sims_gain": (sel["sims_per_move"] / on["sims_per_move"] - 1.0
                      if on["sims_per_move"] else 0.0),
        "p99_ms": {"legacy": on["p99_ms"], "selective": sel["p99_ms"]},
        "max_ms": {"legacy": on["max_ms"], "selective": sel["max_ms"]},
    }
    if u and us:
        out["children_kept"] = (us["children_probed_per_move"]
                                / u["children_probed_per_move"]
                                if u["children_probed_per_move"] else 0.0)
        out["probe_ms_kept"] = (us["probe_ms_per_move_inclusive"]
                                / u["probe_ms_per_move_inclusive"]
                                if u["probe_ms_per_move_inclusive"] else 0.0)
        # The prediction, recomputed from THIS run's legacy arm rather than
        # from #47's stored numbers -- a gate that checks a hard-coded
        # expectation is checking the day it was written.
        share = (u["probe_ms_per_move_inclusive"]
                 / np.mean([r["search_ms"] for r in arms["clean"]["rows"]]))
        freed = share * (1.0 - out["children_kept"])
        out["probe_share_of_search"] = share
        out["predicted_nn_full_gain"] = (1.0 / (1.0 - freed) - 1.0
                                         if freed < 1.0 else float("nan"))
    return out


def derive(payload):
    """Recompute every derived section from the raw arms.

    Called by `main` after measuring AND by `--rerender`, so a mistake in the
    derivation can be fixed and re-applied to a run that already cost an hour
    of GPU. Nothing here reads a stored derived value.
    """
    arms = {a["label"]: a for a in payload["arms"]}
    payload["rate"] = {k: rate(v) for k, v in arms.items()}
    if "filter" in arms:
        payload["filter"] = arms["filter"]["filter"]
    if "counting" in arms and "timed" in arms and "clean" in arms:
        payload["units"] = units(arms["counting"], arms["timed"],
                                 arms["clean"])
    if all(k in arms for k in ("sel_counting", "sel_timed", "sel_clean")):
        payload["units_selective"] = units(arms["sel_counting"],
                                           arms["sel_timed"],
                                           arms["sel_clean"])
    if "clean" in arms and "sel_clean" in arms:
        payload["gate"] = gate(payload, arms)
    d = {}
    if "repeat" in arms and "clean" in arms:
        d["floor"] = disagreement(arms["clean"], arms["repeat"])
    if "off" in arms and "clean" in arms:
        d["on_off"] = disagreement(arms["clean"], arms["off"])
    if "fs_on" in arms and "fs_off" in arms:
        d["fixed_sims"] = disagreement(arms["fs_on"], arms["fs_off"])
    if d:
        payload["disagreement"] = d
    return payload


def render(payload):
    arms = {a["label"]: a for a in payload["arms"]}
    print()
    print("=" * 78)
    print("#47 -- terminal probes on %s: cost, benefit, and the ablation"
          % engine_registry.DEPLOYED)
    print("=" * 78)
    print("positions %d   device %s   seed %d   git %s"
          % (payload.get("n_positions", 0), payload["device"], payload["seed"],
             payload["git_head"]))

    UNIT_ROWS = [
        ("probe roots considered", "probe_roots_per_move", "%12.1f"),
        ("probe roots actually scanned", "probe_roots_scanned_per_move",
         "%12.1f"),
        ("  roots skipped", "probe_roots_skipped_per_move", "%12.1f"),
        ("legal children probed", "children_probed_per_move", "%12.1f"),
        ("  children per scanned root", "children_per_root", "%12.2f"),
        ("clones (all)", "clones_per_move", "%12.1f"),
        ("  of which from probes", "clones_from_probes_per_move", "%12.1f"),
        ("make_move (all)", "make_move_per_move", "%12.1f"),
        ("  of which from probes", "make_move_from_probes_per_move",
         "%12.1f"),
        ("pybind crossings from probes", "crossings_from_probes_per_move",
         "%12.1f"),
        ("pybind crossings elsewhere", "crossings_elsewhere_per_move",
         "%12.1f"),
    ]

    def unit_block(u, title):
        print()
        print("-- %s ------------------------------" % title)
        for label, key, fmt in UNIT_ROWS:
            print(("  %-30s" + fmt) % (label, u[key]))
        print()
        print("  %-30s%12.3f us" % ("per probed child, inclusive",
                                    u["us_per_probed_child_inclusive"]))
        print("  %-30s%12.3f us" % ("per probed child, Python loop",
                                    u["us_per_probed_child_python_loop"]))
        print("  %-30s%12.2f" % ("pybind crossings per child",
                                 u["crossings_per_probed_child"]))
        print("  %-30s%12.3f us" % ("per root, could_end()",
                                    u["predicate_us_per_root"]))
        print()
        print("  %-30s%12.1f ms" % ("probes per move, inclusive",
                                    u["probe_ms_per_move_inclusive"]))
        print("  %-30s%12.1f ms" % ("  of which the Python loop",
                                    u["probe_ms_per_move_python_loop"]))
        print("  %-30s%12.1f ms" % ("  of which could_end()",
                                    u["predicate_ms_per_move"]))
        print("  %-30s%12.1f ms  (subtracted)"
              % ("  instrument",
                 u["instrument_ms_per_move_subtracted"]))
        print("  %-30s%12.1f ms  (do not quote)"
              % ("  raw, before pricing",
                 u["raw_probe_ms_per_move_before_pricing"]))

    if "counting" in arms and "timed" in arms and "clean" in arms:
        u = payload["units"]
        unit_block(u, "probe work units, per move (LEGACY probes)")
    if "units_selective" in payload:
        unit_block(payload["units_selective"],
                   "probe work units, per move (SELECTIVE, #48)")

        print()
        print("-- benefit density -----------------------------------------")
        print("  %-38s%9.4f" % ("probed children that ARE terminal",
                                u["hit_rate_per_child"]))
        print("  %-38s%9.4f" % ("probe roots finding any terminal child",
                                u["roots_with_any_hit"]))
        print("  %-38s%9.4f" % ("expansions that produce a node proof",
                                u["expansions_that_produce_a_proof"]))
        print("  %-38s%9.4f" % ("searches whose ROOT is proven",
                                u["searches_with_a_proven_root"]))
        print("  %-38s%9.2f" % ("proof propagations per move",
                                u["propagations_per_move"]))
        print("  %-38s%9.2f" % ("levels climbed per propagation",
                                u["levels_per_propagation"]))

    if "clean" in arms and "off" in arms:
        on, off = payload["rate"]["clean"], payload["rate"]["off"]
        print()
        print("-- the ablation: probes ON vs OFF, equal clock --------------")
        print("  %-26s %12s %12s %10s" % ("", "ON", "OFF", "OFF vs ON"))
        for label, key, fmt in (
                ("nn/second x deadline", "nn_full", "%12.1f"),
                ("nn per second", "nn_per_second", "%12.1f"),
                ("nn per move", "nn_per_move", "%12.1f"),
                ("simulations per move", "sims_per_move", "%12.1f"),
                ("expansions per move", "expansions_per_move", "%12.1f"),
                ("children probed per move", "probes_per_move", "%12.1f"),
                ("search p50 ms", "p50_ms", "%12.1f"),
                ("search p99 ms", "p99_ms", "%12.1f"),
                ("search max ms", "max_ms", "%12.1f"),
                ("proven roots", "solved_roots", "%12.0f"),
                ("early returns", "early_returns", "%12.0f")):
            print(("  %-26s" + fmt + fmt + "   %+7.1f%%")
                  % (label, on[key], off[key], pct(off[key], on[key])))

    if "disagreement" in payload:
        d = payload["disagreement"]
        print()
        print("-- move disagreement, against its own noise floor -----------")
        for key, label in (("floor", "ON vs ON (floor)"),
                           ("on_off", "ON vs OFF, equal clock"),
                           ("fixed_sims", "ON vs OFF, equal SIMS")):
            if key not in d:
                continue
            e = d[key]
            print("  %-24s %3d / %3d = %.4f" % (label, e["differ"], e["n"],
                                                e["rate"]))
        if "floor" in d and "on_off" in d:
            print()
            print("  equal-clock excess over the floor: %+.4f"
                  % (d["on_off"]["rate"] - d["floor"]["rate"]))
            print("  (the equal-CLOCK row is CONFOUNDED: the OFF arm also gets")
            print("   more search. The equal-SIMS row is the feature's own")
            print("   decision effect, with throughput held fixed.)")
            print("  (and a disagreement has NO SIGN -- this says how often")
            print("   the move changes, not whether the change is better)")

    if "gate" in payload:
        g = payload["gate"]
        on, sel = payload["rate"]["clean"], payload["rate"]["sel_clean"]
        print()
        print("-- #48c GATE: legacy probes vs selective, equal clock --------")
        print("  %-26s %12s %12s %10s"
              % ("", "legacy", "selective", "vs legacy"))
        for label, key, fmt in (
                ("nn/second x deadline", "nn_full", "%12.1f"),
                ("nn per second", "nn_per_second", "%12.1f"),
                ("simulations per move", "sims_per_move", "%12.1f"),
                ("expansions per move", "expansions_per_move", "%12.1f"),
                ("children probed per move", "probes_per_move", "%12.1f"),
                ("search p50 ms", "p50_ms", "%12.1f"),
                ("search p99 ms", "p99_ms", "%12.1f"),
                ("search max ms", "max_ms", "%12.1f"),
                ("proven roots", "solved_roots", "%12.0f")):
            print(("  %-26s" + fmt + fmt + "   %+7.1f%%")
                  % (label, on[key], sel[key], pct(sel[key], on[key])))
        if "children_kept" in g:
            print()
            print("  %-40s%9.4f" % ("per-child work kept",
                                    g["children_kept"]))
            print("  %-40s%9.4f" % ("probe ms/move kept", g["probe_ms_kept"]))
            print("  %-40s%9.4f" % ("probes as a share of the search",
                                    g["probe_share_of_search"]))
            print("  %-40s%+8.1f%%" % ("PREDICTED nn_full gain",
                                       100 * g["predicted_nn_full_gain"]))
        print("  %-40s%+8.1f%%" % ("MEASURED nn_full gain",
                                   100 * g["nn_full_gain"]))
        print()
        print("  (the two arms are the SAME SEARCH at a fixed simulation")
        print("   count -- agents/test_probe_filter.py -- so this is not a")
        print("   different player being faster, it is the same player")
        print("   running more of itself. No strength match is owed.)")

    if "filter" in payload:
        f = payload["filter"]
        print()
        print("-- could_end(): a NECESSARY condition, checked on real probes --")
        print("  %-34s%10d" % ("probe roots seen", f["roots"]))
        print("  %-34s%10d  %.4f" % ("roots the filter would keep",
                                     f["passed"],
                                     f["passed"] / f["roots"] if f["roots"]
                                     else 0.0))
        print("  %-34s%10d  %.4f" % ("children the filter would probe",
                                     f["children_passed"],
                                     f["children_passed"] / f["children_all"]
                                     if f["children_all"] else 0.0))
        print("  %-34s%10d" % ("roots where the probe found one", f["found"]))
        print("  %-34s%10d  <- MUST BE ZERO"
              % ("FALSE NEGATIVES", f["false_negatives"]))
        keep = f["children_passed"] / f["children_all"] if f["children_all"] \
            else 1.0
        print()
        print("  a filtered probe does %.1f%% of the per-child work"
              % (100.0 * keep))
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="all",
                    choices=("units", "ab", "filter", "fixedsims", "gate",
                             "all"))
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--position-games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--rerender", default="")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.rerender:
        with open(args.rerender) as fh:
            payload = json.load(fh)
        render(derive(payload))
        with open(args.rerender, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return

    assert_frozen_sources()
    # The crossing counters patch `GameState.clone`, `.make_move` and the
    # `winner` property, all of which exist only on the C++-backed subclass.
    # On the pure-Python fallback there is no boundary to count and every
    # per-crossing figure below would be a fiction.
    if "winner" not in GameState.__dict__:
        raise SystemExit("[X] the C++ engine is not loaded. There are no "
                         "pybind crossings to count, and the probe cost on "
                         "the Python engine is not the deployed one.\n"
                         "    cmake --build engine/cpp/build --config Release")

    with single_instance("probe_ablation"):
        base = gpu_baseline()
        if base:
            print("[..] GPU load before starting: %.0f%% mean, %.0f%% peak"
                  % (base["mean_pct"], base["max_pct"]))
        # Deployment conditions, same as tools/regress_engine and
        # tools/profile_selection: automatic cyclic collection off, collect at
        # boundaries. A collector pause landing mid-search is exactly what the
        # deferred-GC design moved out of the measurement.
        gc.disable()
        print("[!] automatic cyclic GC off; collecting at boundaries only")

        payload = {"tag": args.tag, "mode": args.mode, "arms": [],
                   "seed": args.seed, "device": args.device,
                   "on_spec": ON_SPEC, "off_spec": OFF_SPEC,
                   "sel_spec": SEL_SPEC, "gate_base_spec": GATE_BASE_SPEC,
                   "deployed": engine_registry.DEPLOYED,
                   "git_head": engine_registry.git_head(),
                   "environment": engine_registry.environment(),
                   "environment_drift": engine_registry.env_drift(),
                   "gpu_baseline_before_run": base}
        path = os.path.join(OUT_DIR, "%s.json" % args.tag)

        def save():
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)

        print("[..] sampling positions from %d games" % args.position_games)
        positions = positions_for(ON_SPEC, args.position_games, args.device,
                                  args.seed, args.positions)
        payload["n_positions"] = len(positions)
        print("[..] %d fixed positions" % len(positions))

        want = []
        if args.mode in ("units", "gate", "all"):
            want += ["counting", "timed"]
        if args.mode in ("units", "ab", "gate", "all"):
            want += ["clean"]
        if args.mode in ("ab", "all"):
            want += ["off", "repeat"]
        if args.mode in ("filter", "all"):
            want += ["filter"]
        if args.mode in ("fixedsims", "all"):
            want += ["fs_on", "fs_off"]
        # #48c. Its own mode and NOT part of `all`, because the base arms mean
        # something different here: `all` measures whatever is deployed, the
        # gate measures a pinned pair. Folding it in would silently swap the
        # base the moment a promotion landed.
        #
        # The selective arms come LAST and are the same three arms in the same
        # order as the base side: an A/B whose two halves ran under different
        # instruments, or with the GPU in a different state, is not an A/B. The
        # clean arms are what the gate reads; the counting and timed arms are
        # what says WHERE the difference went.
        if args.mode == "gate":
            want += ["sel_counting", "sel_timed", "sel_clean"]

        SEL_ARMS = {"sel_counting", "sel_timed", "sel_clean"}
        base_spec = GATE_BASE_SPEC if args.mode == "gate" else ON_SPEC
        for label in want:
            print("[..] arm %s" % label)
            kw = {}
            spec = SEL_SPEC if label in SEL_ARMS else base_spec
            if label in ("counting", "sel_counting"):
                kw["counters"] = ProbeCounters()
            elif label in ("timed", "sel_timed"):
                t = AttributedTimer()
                t.price_us = t.calibrate()
                print("    wrapper priced at %.3f us/call (%.3f inside)"
                      % (t.price_us, t.inside_us))
                kw["timer"] = t
            elif label == "filter":
                kw["filt"] = FilterCounters()
            elif label == "off":
                spec = OFF_SPEC
            elif label == "fs_on":
                spec = FIXED_ON_SPEC
            elif label == "fs_off":
                spec = FIXED_OFF_SPEC
            payload["arms"].append(
                run_arm(spec, positions, args.device, label, **kw))
            save()

        render(derive(payload))
        save()
        print("[OK] wrote %s" % path)


if __name__ == "__main__":
    main()
