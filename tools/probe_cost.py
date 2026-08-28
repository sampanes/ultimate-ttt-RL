"""#49 -- what the RESIDUAL probe path is made of, primitive by primitive.

`pocket_filter` took the probe loop from 154.6 ms/move to 46.6 ms. The next
question is not "how big is it" -- the gate already answered that -- but "what
is it MADE OF", because the two remaining proposals (a fused native probe, and
removing the redundant `winner` crossing) are proposals about different pieces
of it, and only a split can say whether either is worth an API.

WHY A SEPARATE TOOL, AND WHY NOT MORE WRAPPERS. `tools/probe_ablation` prices
the probe by wrapping it. That instrument has now been caught under-pricing
this exact path twice: #47 predicted +16.0% from wrapper timings and the
uninstrumented ablation measured +18.9%; #48 predicted +16.7% and the
uninstrumented gate measured +20.6%. Ratios 1.18 and 1.24, same direction, same
signature. A per-call clock starts after the callee is entered and stops before
it returns, so what ~14,000 GameState allocations and frees a move do to the
allocator and to cache locality lands OUTSIDE every interval it measures, and
is charged instead to whatever runs next. Adding six more wrappers to split the
probe finer would make that worse, not better: each new wrapper both perturbs
the path and hides more of it.

So this tool does not wrap production. It REPLAYS the production loop body over
real captured probe roots in a tight loop, as a cumulative ladder in which each
rung adds exactly one operation. The cost of an operation is a DIFFERENCE of
two rungs, which needs no wrapper at all, and the whole ladder is validated
against the real bound method timed the same way (`production` below). If the
hand-written top rung disagrees with the real method, the ladder is wrong and
the run says so instead of reporting a decomposition of something else.

WHAT A LADDER STILL CANNOT SEE, stated up front. Replaying a loop in isolation
has a hot allocator, a warm cache and a small live heap. Production has none of
those. That gap is not a defect to be apologised for, it is a quantity, so
`--mode heap` measures it directly: the same ladder, run with a real ~40k-node
search tree alive versus a clean heap. The difference is the allocator/cache
spillover, promoted here from "miscellaneous overhead" to a row with a number.

    python -m tools.probe_cost --mode all

MODES
  boundary  price the primitives on THIS box: a pybind free function, a pybind
            bound method, a Python function, a property, a 2-tuple. Every
            "one crossing costs X" claim below resolves to a measurement here
            rather than to a remembered number from another machine.
  ladder    the cumulative rungs, over roots the deployed filter ADMITS, and
            again over all roots the legacy engine scanned. The two corpora
            answer a question the gate raised and did not settle: post-filter
            per-child cost is 8.02 us against the legacy 3.95, so either the
            surviving children are genuinely more expensive or something else
            moved. That is a fact about which children survive, and it is
            testable by running the identical ladder on both populations.
  heap      the same ladder with a live tree pinned, versus without.
  alloc     #49c. Object churn per move, measured and NOT optimised: GameState
            clones, MCTSNode creations, live pymalloc blocks at peak.
  crossing  #49b. The decision number: us saved per probed child by dropping
            `make_move`'s discarded tuple and the second `winner` read, and the
            ms/move that projects to at the post-filter call volume.

THE CORPUS IS CAPTURED FROM PRODUCTION, not synthesised. Random reachable
states would misrepresent both the child count and the board occupancy of the
roots the filter admits -- admitted roots are late positions with most
mini-boards decided, which is exactly where `make_move` does its most expensive
work. So `capture` hooks the real probe and reservoir-samples real roots.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time

import numpy as np
import torch

from agents import mcts as mcts_mod
from agents.mcts import MCTS, MCTSNode, TreeReuseSearcher, _clone, could_end
from agents import native_select as _ns
from engine.constants import X, O, DRAW
from engine.game import GameState
from tools import engine_registry
from tools.arena_1s import TimedPlayer, phase_of, play_match
from tools.profile_tree import assert_frozen_sources
from tools.runlock import gpu_baseline, single_instance

OUT_DIR = os.path.join("results", "probe_cost")

# Instrumented and unscored, like every other profiling namespace here.
COST_SEED = engine_registry.SEEDS["probe"]

# The engine whose residual is being split. Literal, not `DEPLOYED`: this
# measurement gets written into a document with a number in it, and a document
# that silently re-targets itself on the next promotion is not a record.
SPEC = "engine:pocket_filter"

# THE CALL VOLUME AND THE ENVELOPE ARE MEASURED HERE, on this run's own
# positions, and are NOT imported from the #48c gate. The first version of this
# tool multiplied unit costs measured on its own corpus by children/move from
# the gate run, which is the exact error its own docstring warns about: those
# two runs disagree by 35% on children probed per move, because a wall-clock
# search on a different sample of positions probes a different amount. So
# `main` runs `probe_ablation`'s clean/counting/timed arms on the same
# `positions` list the ladder corpus was captured from, and every ms/move below
# is that run's own us/child times that run's own count.
#
# These remain only as the fallback for `--rerender` of a payload written
# before the in-situ arms existed, and as the gate's published figures for
# comparison.
GATE_CHILDREN_PER_MOVE = 5806.1667
GATE_ROOTS_PER_MOVE = 5868.15
GATE_PROBE_MS = 46.5618

# The two replicated under-prices, used to state the ladder's lower-bound
# correction rather than leaving the reader to remember them.
UNDERPRICE_RATIOS = (1.18, 1.24)

_CPP = GameState.__mro__[1] if len(GameState.__mro__) > 1 else None


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------

def bench(fn, reps, warmup=2):
    """Run `fn` `reps` times, return (min, median, all) seconds.

    MINIMUM, not mean. A microbenchmark's noise is one-sided -- the scheduler,
    another process and a cache eviction can only ever make a pass slower --
    so the minimum is the least contaminated estimate of the cost of the code,
    and the median is reported beside it so a wide gap between them is visible
    rather than averaged away.
    """
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return min(out), float(np.median(out)), out


def primitive(fn, n, reps=7):
    """ns per call of `fn`, with an empty loop of the same shape subtracted."""
    rng = range(n)

    def hot():
        for _ in rng:
            fn()

    def cold():
        for _ in rng:
            pass

    h, _, _ = bench(hot, reps)
    c, _, _ = bench(cold, reps)
    return max(0.0, (h - c) / n * 1e9)


def boundary_prices(reps=5, n=120000):
    """The primitive floor on THIS box and THIS pybind/Python build.

    Every structural claim in the crossing analysis -- "the redundant read is
    one bound-method crossing plus a tuple" -- becomes a number here. The
    no-ops are the ones #45a added to `uttt_select` for exactly this purpose:
    same module, same call machinery, no work inside.
    """
    _ns.require("the boundary primitives")
    pr = _ns.Probe()
    noop0, noop1 = _ns.noop0, _ns.noop1

    st = GameState()
    st.make_move(40)

    def py_fn():
        return None

    class _Holder:
        def m(self):
            return None

        @property
        def p(self):
            return 0

    h = _Holder()

    # NOT `lambda: (True, None)`. CPython folds an all-constant tuple into a
    # single LOAD_CONST, so that spelling measures nothing and reported 0.7 ns
    # on the first run -- which would have removed a real term from the
    # crossing arithmetic. The second element has to be a runtime value, which
    # is also what `make_move` actually builds.
    _runtime = st
    _mw = list(st.mini_winners)

    def tup():
        return (True, _runtime)

    out = {
        "pybind free fn, 0 args": primitive(noop0, n, reps),
        "pybind free fn, 1 arg": primitive(lambda: noop1(1), n, reps),
        "pybind bound method, 0 args": primitive(pr.m0, n, reps),
        "pybind bound method, 1 arg": primitive(lambda: pr.m1(1), n, reps),
        "python function": primitive(py_fn, n, reps),
        "python bound method": primitive(h.m, n, reps),
        "python property get": primitive(lambda: h.p, n, reps),
        "2-tuple build": primitive(tup, n, reps),
        "GameState._raw_winner": primitive(st._raw_winner, n, reps),
        "GameState.winner (property)": primitive(lambda: st.winner, n, reps),
        "GameState(s) copy ctor": primitive(lambda: GameState(st), n // 4, reps),
        "GameState.clone()": primitive(st.clone, n // 4, reps),
        "_clone(state)": primitive(lambda: _clone(st), n // 4, reps),
        # The predicate's INPUT, priced separately from the predicate. This is
        # the row the first run made necessary: `could_end` came out at 1.62
        # us/root against a 0.25 us crossing floor, which is only explicable if
        # most of it is not the scan. `mini_winners` is a pybind property that
        # materialises a 9-element Python list -- a crossing, a list, and nine
        # boxed ints -- on every root, and it is charged to the predicate.
        "GameState.mini_winners (list)": primitive(lambda: st.mini_winners,
                                                   n // 2, reps),
        "GameState.player": primitive(lambda: st.player, n, reps),
        "could_end(materialised list)": primitive(
            lambda: could_end(_mw, st.player), n // 2, reps),
        "could_end(state.mini_winners)": primitive(
            lambda: could_end(st.mini_winners, st.player), n // 2, reps),
    }
    # The lambda wrappers above add one Python call each. Subtract it so a
    # "crossing" price is a crossing and not a crossing plus a lambda -- the
    # difference is ~40 ns, which is 10% of the number the decision rests on.
    lam = out["python function"]
    for k in ("pybind free fn, 1 arg", "pybind bound method, 1 arg",
              "python property get", "2-tuple build",
              "GameState.winner (property)", "GameState(s) copy ctor",
              "_clone(state)", "GameState.mini_winners (list)",
              "GameState.player", "could_end(materialised list)",
              "could_end(state.mini_winners)"):
        out[k] = max(0.0, out[k] - lam)
    return out


# ----------------------------------------------------------------------
# Corpus: real probe roots, captured from a real search
# ----------------------------------------------------------------------

class RootCapture:
    """Reservoir-sample real probe roots, split by what the filter did.

    Stores `(state.clone(), tuple(children keys), priors)` and NOT the node:
    a node holds a parent pointer, so keeping one keeps the whole tree, and a
    corpus of 3,000 of them would pin several gigabytes and change the very
    allocator behaviour this tool is trying to measure.
    """

    def __init__(self, want, seed):
        self.want = want
        self.rng = random.Random(seed)
        self.admitted = []
        self.skipped = []
        self.n_admitted = 0
        self.n_skipped = 0

    def offer(self, bucket, n_seen, state, kids):
        # Decide FIRST, copy second. Building the lists and cloning on every
        # offer would add an allocation to every one of ~90,000 probe roots a
        # search -- a capture pass that perturbs the very allocator behaviour
        # the heap arm goes on to measure.
        if len(bucket) < self.want:
            j = len(bucket)
            bucket.append(None)
        else:
            j = self.rng.randrange(n_seen)
            if j >= self.want:
                return
        bucket[j] = (state.clone(), tuple(kids),
                     tuple(c.prior for c in kids.values()))

    def hook(self, mcts):
        raw = MCTS.__dict__["_mark_terminal_children"]
        cap = self

        def watched(self, node, state):
            kids = node.children
            if kids:
                if could_end(state.mini_winners, state.player):
                    cap.n_admitted += 1
                    cap.offer(cap.admitted, cap.n_admitted, state, kids)
                else:
                    cap.n_skipped += 1
                    cap.offer(cap.skipped, cap.n_skipped, state, kids)
            return raw(self, node, state)

        MCTS._mark_terminal_children = watched

        def restore():
            MCTS._mark_terminal_children = raw
        return restore


def capture_roots(spec, device, positions, want, seed):
    p = TimedPlayer(spec, device)
    cap = RootCapture(want, seed)
    restore = cap.hook(p.mcts)
    try:
        for state, _phase in positions:
            with torch.no_grad():
                _pi, root = p.mcts.search(state.clone())
            TreeReuseSearcher.release(root)
            gc.collect()
    finally:
        restore()
    return p, cap


def rebuild(mcts, corpus):
    """Detached (node, state) pairs with a REAL native mirror attached.

    The deployed engine runs `native_select=1`, so `node.selS` exists and the
    probe reads it once per root and writes it on a hit. Benching against nodes
    without a mirror would price a loop the engine does not run.

    THE ROOTS ARE PARENTLESS, and that is a deliberate, priced simplification
    rather than an oversight. `_mark_solved` calls `_propagate_solved`, which
    walks ancestors; with `parent is None` it returns immediately. In
    production that walk is 105 propagations a move climbing 0.201 levels
    each -- about 21 `_solve_from_children` calls per move against 788 scanned
    roots -- so pricing it here would move the total by well under 0.1 ms and
    would require a synthetic ancestor chain whose branching factor is itself
    made up. The `solved-status writes` row therefore covers backward
    induction and the proof write, and says so.
    """
    out = []
    for state, moves, priors in corpus:
        node = MCTSNode(parent=None, prior=0.0, move=None, to_play=state.player)
        row = np.zeros(81, dtype=np.float64)
        for mv, pr in zip(moves, priors):
            row[mv] = pr
        next_to_play = O if state.player == X else X
        mcts._build_children_mirrored(node, list(moves), row, next_to_play)
        out.append((node, state))
    return out


def reset(pairs):
    """Undo any marking a previous rung did, so every rung sees fresh nodes.

    Without this the first rung that marks children leaves them marked, and
    every later rung measures `_solve_from_children` over an already-solved
    node -- a different function on a different input, reported as the same
    row.
    """
    for node, _state in pairs:
        node.solved = None
        for c in node.children.values():
            c.solved = None
            c.is_terminal = False
            c.terminal_value = 0.0
        if node.selS is not None:
            node.selS[:] = _ns.SOLVED_NONE


# ----------------------------------------------------------------------
# The ladder
# ----------------------------------------------------------------------
#
# Each rung is the previous rung plus exactly one operation. They are written
# out longhand rather than generated, because a rung built by composing
# closures would price the composition.

def r0_iterate(pairs, mcts):
    n = 0
    for node, state in pairs:
        for mv, child in node.children.items():
            n += 1
    return n


def r1_clone(pairs, mcts):
    n = 0
    for node, state in pairs:
        for mv, child in node.children.items():
            probe = _clone(state)
            mcts.stat_probes += 1
            n += 1
    return n


def r2_make_native(pairs, mcts):
    cpp_make = _CPP.make_move
    n = 0
    for node, state in pairs:
        for mv, child in node.children.items():
            probe = _clone(state)
            cpp_make(probe, mv)
            mcts.stat_probes += 1
            n += 1
    return n


def r3_raw_winner(pairs, mcts):
    """The CANDIDATE path: one crossing to move, one to read, no tuple.

    EVERYTHING ELSE IS r4 VERBATIM, including the `_terminal_value` call on a
    hit. It would be cheaper to inline the terminal value here -- `w` is
    already in hand and `_terminal_value` re-reads `probe.winner` twice -- and
    that is exactly why it is not done: folding a second saving into this rung
    would make the step attribute both of them to the crossing, and #49b is a
    price for ONE change. The inline is priced separately by `r3b`.
    """
    cpp_make = _CPP.make_move
    tv = mcts._terminal_value
    n = 0
    for node, state in pairs:
        sS = node.selS
        for mv, child in node.children.items():
            probe = _clone(state)
            cpp_make(probe, mv)
            mcts.stat_probes += 1
            n += 1
            if probe._raw_winner() != -1:
                child.is_terminal = True
                child.terminal_value = tv(probe, child.to_play)
                child.solved = int(child.terminal_value)
                if sS is not None:
                    sS[child.cidx] = child.solved
    return n


def r3b_inline_value(pairs, mcts):
    """r3 with `_terminal_value` inlined too. Priced, not proposed.

    Only fires on a hit, and hits are ~3% of probed children, so this exists to
    put a number on a second micro-optimisation rather than to leave it as an
    unpriced "while we are in here".
    """
    cpp_make = _CPP.make_move
    n = 0
    for node, state in pairs:
        sS = node.selS
        for mv, child in node.children.items():
            probe = _clone(state)
            cpp_make(probe, mv)
            mcts.stat_probes += 1
            n += 1
            w = probe._raw_winner()
            if w != -1:
                child.is_terminal = True
                child.terminal_value = (0.0 if w == DRAW else
                                        (1.0 if w == child.to_play else -1.0))
                child.solved = int(child.terminal_value)
                if sS is not None:
                    sS[child.cidx] = child.solved
    return n


def r4_current(pairs, mcts):
    """The SHIPPED path, replicated: tuple built and discarded, winner re-read."""
    tv = mcts._terminal_value
    n = 0
    for node, state in pairs:
        sS = node.selS
        for mv, child in node.children.items():
            probe = _clone(state)
            probe.make_move(mv)
            mcts.stat_probes += 1
            n += 1
            if probe.winner is not None:
                child.is_terminal = True
                child.terminal_value = tv(probe, child.to_play)
                child.solved = int(child.terminal_value)
                if sS is not None:
                    sS[child.cidx] = child.solved
    return n


def r5_solve(pairs, mcts):
    """r4 plus the backward induction and propagation the real method runs."""
    tv = mcts._terminal_value
    solve_from = mcts._solve_from_children
    mark = mcts._mark_solved
    n = 0
    for node, state in pairs:
        sS = node.selS
        for mv, child in node.children.items():
            probe = _clone(state)
            probe.make_move(mv)
            mcts.stat_probes += 1
            n += 1
            if probe.winner is not None:
                child.is_terminal = True
                child.terminal_value = tv(probe, child.to_play)
                child.solved = int(child.terminal_value)
                if sS is not None:
                    sS[child.cidx] = child.solved
        status = solve_from(node)
        if status is not None:
            mark(node, status)
    return n


def r_production(pairs, mcts):
    """The REAL bound method. Not a rung -- the check on the rungs.

    If `r5 + the predicate` and this disagree by more than a few percent, the
    replica has drifted from the code it claims to decompose and every
    difference above it is a difference between two things that are not the
    production loop.

    THE COMPARISON HAS TO ADD THE PREDICATE BACK. The real method runs
    `could_end` inside itself; `r5` does not. The first run reported the
    replica 13.5% BELOW production and that gap was almost exactly the
    predicate -- an apparent instrument failure that was really a missing term
    in the comparison.
    """
    f = mcts._mark_terminal_children
    for node, state in pairs:
        f(node, state)
    return 0


def r_predicate(pairs, mcts):
    """`could_end` per ROOT, which is what a skipped root pays instead."""
    n = 0
    for node, state in pairs:
        if could_end(state.mini_winners, state.player):
            n += 1
    return n


RUNGS = [
    ("iterate children", r0_iterate, "child"),
    ("+ clone", r1_clone, "child"),
    ("+ make_move (native)", r2_make_native, "child"),
    ("+ _raw_winner + mark", r3_raw_winner, "child"),
    ("+ tuple + winner property", r4_current, "child"),
    ("+ solve/propagate", r5_solve, "child"),
]

# Off the ladder: same inputs, priced for its own sake rather than as a step.
EXTRA = [("r3 with the terminal value inlined", r3b_inline_value)]

# What each STEP between two rungs isolates, in the owner's vocabulary.
STEPS = [
    ("Python child iteration", 0, "child"),
    ("clone (alloc + copy ctor)", 1, "child"),
    ("make_move native execution", 2, "child"),
    ("probe.winner readback", 3, "child"),
    ("tuple construction + redundant crossing", 4, "child"),
    ("solved-status writes / propagation", 5, "child"),
]


def run_ladder(pairs, mcts, reps, label, filtered):
    """`filtered` is what `mcts.probe_filter` must be for the production arm.

    On a corpus of roots the filter ADMITS, the flag makes no difference to
    what runs and it is left as the engine has it. On the all-roots corpus it
    makes all the difference: with the flag on, production skips half the
    corpus and comes out 76% cheaper than the replica -- which is not an
    instrument failure but a different function. The legacy loop is what that
    corpus is a corpus OF, so the flag is forced off for it.
    """
    n_children = sum(len(node.children) for node, _ in pairs)
    n_roots = len(pairs)
    out = {"label": label, "roots": n_roots, "children": n_children,
           "children_per_root": n_children / n_roots if n_roots else 0.0,
           "reps": reps, "production_filtered": filtered,
           "rungs": {}, "extra": {}, "steps": {}}
    for name, fn, _unit in RUNGS:
        reset(pairs)
        mn, med, _all = bench(lambda: fn(pairs, mcts), reps)
        out["rungs"][name] = {"min_s": mn, "median_s": med,
                              "us_per_child": mn / n_children * 1e6,
                              "spread": (med / mn - 1.0) if mn else 0.0}
    for name, fn in EXTRA:
        reset(pairs)
        mn, _med, _all = bench(lambda: fn(pairs, mcts), reps)
        out["extra"][name] = mn / n_children * 1e6
    reset(pairs)
    was = mcts.probe_filter
    mcts.probe_filter = filtered
    try:
        mn, med, _ = bench(lambda: r_production(pairs, mcts), reps)
    finally:
        mcts.probe_filter = was
    out["production_us_per_child"] = mn / n_children * 1e6
    out["production_spread"] = (med / mn - 1.0) if mn else 0.0
    reset(pairs)
    mn, med, _ = bench(lambda: r_predicate(pairs, mcts), reps)
    out["predicate_us_per_root"] = mn / n_roots * 1e6

    order = [n for n, _f, _u in RUNGS]
    last = 0.0
    for i, name in enumerate(order):
        cur = out["rungs"][name]["us_per_child"]
        out["steps"][STEPS[i][0]] = cur - last
        last = cur
    # The replica against the real thing, with the predicate added back ONLY
    # when production would run it. With the filter off the method never calls
    # `could_end`, so adding it would charge production for work it did not do
    # and report the replica 7.7% too expensive -- the mirror image of the
    # -13.5% the first run produced by leaving it out when it WAS run.
    per_child = out["children_per_root"] or 1.0
    top = out["rungs"][order[-1]]["us_per_child"]
    if filtered:
        top += out["predicate_us_per_root"] / per_child
    out["replica_us_per_child"] = top
    out["replica_vs_production"] = (top / out["production_us_per_child"] - 1.0
                                    if out["production_us_per_child"] else 0.0)
    return out


# ----------------------------------------------------------------------
# #49b -- the crossing
# ----------------------------------------------------------------------

def crossing(lad, prices, children_per_move):
    """us/child saved by the raw path, and the ms/move that projects to.

    Two independent readings, deliberately:

      measured    rung 4 minus rung 3, which is the whole difference between
                  the shipped body and the candidate body on real states.
      structural  the primitive prices added up: one bound-method crossing,
                  one Python function frame, two property gets, one 2-tuple.

    They are derived from different data. If they agree, the saving is
    understood; if they do not, something else is in that step and the
    difference has to be explained before an API is added for it.
    """
    saved = lad["steps"]["tuple construction + redundant crossing"]
    # THE SHIPPED BODY AND THE CANDIDATE BODY, EACH SPELLED OUT, and the
    # structural estimate is their difference. The first version of this
    # charged the saving as "one crossing + a frame + two bare property gets +
    # a tuple" and came out 54% under the measurement -- because a bare
    # property get (61 ns) is not what `GameState.winner` costs (379 ns): the
    # property runs `_raw_winner()` and translates the sentinel. Pricing a
    # composite by the cost of its cheapest part is how a structural check
    # ends up disagreeing with reality and being believed anyway.
    shipped = (prices["python bound method"]              # GameState.make_move
               + prices["pybind bound method, 1 arg"]     # _CppGameState.make_move
               + prices["GameState.winner (property)"]    # inside make_move
               + prices["2-tuple build"]                  # discarded
               + prices["GameState.winner (property)"])   # the loop's own test
    candidate = (prices["pybind bound method, 1 arg"]
                 + prices["GameState._raw_winner"])
    structural = (shipped - candidate) / 1000.0
    fused_extra = prices["pybind bound method, 1 arg"] / 1000.0
    out = {
        "measured_us_per_child": saved,
        "structural_us_per_child": structural,
        "structural_shipped_ns": shipped,
        "structural_candidate_ns": candidate,
        "agreement": (saved / structural if structural else 0.0),
        "projected_ms_per_move": saved * children_per_move / 1000.0,
        # The extra a FUSED native probe_make_move could take on top of the
        # raw path: it would collapse the two remaining crossings into one, so
        # its ceiling over the candidate is one bound-method crossing.
        "fused_extra_us_per_child": fused_extra,
        "fused_extra_ms_per_move": fused_extra * children_per_move / 1000.0,
        # And the second micro-optimisation, priced but not bundled.
        "inline_terminal_value_us_per_child": (
            lad["rungs"]["+ _raw_winner + mark"]["us_per_child"]
            - lad["extra"].get("r3 with the terminal value inlined", 0.0)
            if lad.get("extra") else 0.0),
        "children_per_move": children_per_move,
    }
    out["inline_terminal_value_ms_per_move"] = (
        out["inline_terminal_value_us_per_child"] * children_per_move / 1000.0)
    out["band"] = band_for(out["projected_ms_per_move"])
    return out


def band_for(ms):
    return ("archive" if ms < 2.0 else
            "probably archive unless trivial" if ms < 5.0 else
            "implement + parity/throughput" if ms < 10.0 else
            "real optimisation candidate")


def elsewhere(cross, insitu):
    """The SAME redundancy, at the call site nobody was pricing.

    `_run_wave`'s descent does

        state.make_move(node.move)      # builds (True, self.winner)
        ...                             # Python discards the tuple
        if state.winner is not None:    # and reads the winner again

    which is the probe's redundancy character for character. The counting arm
    says the descent runs it 53,080 times a move against the probe's 5,806 --
    9.1 to 1 -- and `winner_other` is 106,533, almost exactly twice
    `make_other`, which is what confirms the second read really is happening on
    every one of them rather than being an artefact of reading the source.

    This is NOT a scope expansion decided here. It is the term the re-ranking
    was asked to look for ("state.make_move during normal traversal"), priced
    with the number the probe measurement already produced, and reported so the
    decision about it is made on a measurement instead of on the assumption
    that the probe was the hot path.
    """
    u = insitu["units"]
    probe_calls = u["make_move_from_probes_per_move"]
    all_calls = u["make_move_per_move"]
    descent_calls = all_calls - probe_calls
    us = cross["measured_us_per_child"]
    out = {
        "us_per_call": us,
        "probe_calls_per_move": probe_calls,
        "descent_calls_per_move": descent_calls,
        "ratio_descent_to_probe": (descent_calls / probe_calls
                                   if probe_calls else 0.0),
        "probe_ms_per_move": us * probe_calls / 1000.0,
        "descent_ms_per_move": us * descent_calls / 1000.0,
        "combined_ms_per_move": us * all_calls / 1000.0,
    }
    # The structural check that the second read is real rather than read off
    # the source: every make_move outside the probe should carry two winner
    # reads, so this ratio must come back at ~2.0. A value near 1.0 would mean
    # the descent does NOT re-read and the whole descent estimate is void.
    c = insitu.get("counters") or {}
    if c.get("make_other"):
        out["winner_reads_per_make_move_elsewhere"] = (c["winner_other"]
                                                       / c["make_other"])
    out["descent_band"] = band_for(out["descent_ms_per_move"])
    out["combined_band"] = band_for(out["combined_ms_per_move"])
    out["share_of_search"] = (out["combined_ms_per_move"]
                              / insitu["search_ms_per_move"]
                              if insitu["search_ms_per_move"] else 0.0)
    return out


# ----------------------------------------------------------------------
# Is the WRAPPER the residual? The one test that can say so.
# ----------------------------------------------------------------------

def price_check(mcts, pairs, reps):
    """Time the real method twice on the SAME corpus: bare, and wrapped.

    THIS IS THE ONLY ARM THAT CAN SETTLE THE RESIDUAL, and it exists because
    the two obvious explanations were measured and refuted: pinning a 71,000
    node tree moved the loop -1.8%, and growing the working set 118x moved it
    +3.9%. Neither is a 2.5x gap. What has not been tested is the instrument
    itself.

    `probe_ablation` prices its wrappers with `AttributedTimer.calibrate`,
    which times a wrapper around a trivial Python no-op in a tight loop. The
    wrappers in question go around `GameState.clone` and `GameState.make_move`
    -- pybind bound methods reached through a Python subclass -- roughly 13,000
    times a move. If a wrapper costs more there than it does around a no-op,
    `price()` under-subtracts and every "probe ms/move" figure ever published
    here is too high by the difference.

    The test is direct. Run the real `_mark_terminal_children` over a fixed
    corpus with no instrument at all; run it again with exactly the wrappers
    `probe_ablation` installs; then hand the wrapped result to
    `probe_ablation.price` and ask whether it recovers the bare number.

      recovered / bare ~= 1.0   the pricing is sound, and the residual is
                                something about running inside a real search
                                that a replay cannot reproduce.
      recovered / bare  > 1.0   the correction is too small. The published
                                probe costs are inflated by that factor and
                                the ladder was right all along.

    Either answer is worth having. The second would be a correction to three
    result documents, which is exactly why it gets measured instead of argued.
    """
    from tools import probe_ablation as pa
    from tools.profile_selection import AttributedTimer

    n_children = sum(len(node.children) for node, _ in pairs)
    n_roots = len(pairs)

    reset(pairs)
    bare, _med, _ = bench(lambda: r_production(pairs, mcts), reps)
    bare_us = bare / n_children * 1e6

    t = AttributedTimer(ctx={"on": True})
    t.price_us = t.calibrate()
    restore = pa.patch_timed(t)
    try:
        reset(pairs)
        # ONE pass. The wrapped arm is what production's timed arm does, and
        # production times each root once; repeating would let the timer's own
        # dictionaries settle into a state the real run never reaches.
        r_production(pairs, mcts)
    finally:
        restore()

    blob = {
        "exclusive_ms": {k: v * 1000.0 for k, v in t.total.items()},
        "inclusive_ms": {k: v * 1000.0 for k, v in t.inclusive.items()},
        "calls": dict(t.calls),
        "calls_from": {"%s>%s" % k: v[0] for k, v in t.calls_from.items()},
        "inside_us": t.inside_us,
        "total_us": t.price_us,
    }
    p = pa.price(blob, 1)
    per_root_nested = blob["calls_from"].get("terminal probes>could_end", 0)
    timed_children = (p["nested_calls"] - per_root_nested) / 2.0
    recovered_us = (p["inclusive_ms"] * 1000.0 / timed_children
                    if timed_children else 0.0)
    return {
        "roots": n_roots, "children": n_children,
        "bare_us_per_child": bare_us,
        "raw_wrapped_us_per_child": (p["raw_inclusive_ms"] * 1000.0
                                     / timed_children if timed_children
                                     else 0.0),
        "wrapper_subtracted_us_per_child": (p["wrapper_ms"] * 1000.0
                                            / timed_children
                                            if timed_children else 0.0),
        "recovered_us_per_child": recovered_us,
        "recovered_over_bare": (recovered_us / bare_us if bare_us else 0.0),
        "calibrated_price_us": t.price_us,
        "calibrated_inside_us": t.inside_us,
        # What the wrapper WOULD have to cost for the priced answer to land on
        # the bare one. Compared against the calibration, this is the whole
        # claim in one number.
        "implied_price_us": (
            (p["raw_inclusive_ms"] - bare_us * timed_children / 1000.0)
            * 1000.0 / (p["calls"] * (t.inside_us / t.price_us)
                        + p["nested_calls"])
            if p["nested_calls"] and t.price_us else 0.0),
    }


# ----------------------------------------------------------------------
# Locality: is the residual allocator pressure, or first touch?
# ----------------------------------------------------------------------

def locality_curve(mcts, corpus, sizes, reps):
    """us/child of the SAME loop over working sets of increasing size.

    THIS IS THE ARM THAT DECIDES WHAT THE RESIDUAL IS. The ladder replays a
    small corpus dozens of times, so its states and nodes are resident and its
    allocator is in a tight steady state; production touches each root once,
    interleaved with 50,000 node creations, torch and numpy. The gap between
    them came out at 2.7x per child, and `--mode heap` already ruled out the
    obvious explanation -- pinning a real 55,000-node tree moved the loop by
    only 7.4%.

    So the remaining candidate is FIRST TOUCH, and it is testable: hold the
    code, the allocation rate and the live-heap size fixed, and vary only how
    much distinct memory a single pass walks. If us/child climbs with the
    working set, the residual is locality and the lever is touching less
    memory. If it is flat, locality is refuted too and the residual is
    somewhere neither this tool nor the wrapper can see -- which is itself
    worth knowing before another optimisation is designed against it.
    """
    out = []
    for n in sizes:
        if n > len(corpus):
            continue
        pairs = rebuild(mcts, corpus[:n])
        kids = sum(len(node.children) for node, _ in pairs)
        reset(pairs)
        mn, med, _ = bench(lambda: r4_current(pairs, mcts), reps)
        out.append({
            "roots": n, "children": kids,
            "us_per_child": mn / kids * 1e6,
            "spread": (med / mn - 1.0) if mn else 0.0,
            # Rough resident footprint of one pass: the states and the nodes it
            # walks. Quoted as an order of magnitude, not a cache model.
            "kb_touched": (n * 4 + kids * 4) * 0.1,
        })
        pairs = None
        gc.collect()
    return out


# ----------------------------------------------------------------------
# The in-situ envelope, on THIS run's positions
# ----------------------------------------------------------------------

def insitu_envelope(spec, device, positions):
    """What the wrapper-priced instrument says the whole probe path costs.

    Runs `tools.probe_ablation`'s own clean / counting / timed triple rather
    than a second implementation of them. Two reasons, and the second is the
    one that matters: the pricing of the wrapper (`price()`) and the
    count-from-one-arm-clock-from-another rule (`units()`) are subtle enough
    that a re-implementation here would be a second opinion masquerading as
    agreement, and the published 46.6 ms figure came out of exactly that code.
    Reusing it makes this number comparable to the gate's by construction.
    """
    from tools import probe_ablation as pa
    from tools.profile_selection import AttributedTimer

    clean = pa.run_arm(spec, positions, device, "clean")
    counting = pa.run_arm(spec, positions, device, "counting",
                          counters=pa.ProbeCounters())
    t = AttributedTimer()
    t.price_us = t.calibrate()
    print("    wrapper priced at %.3f us/call (%.3f inside)"
          % (t.price_us, t.inside_us))
    timed = pa.run_arm(spec, positions, device, "timed", timer=t)
    u = pa.units(counting, timed, clean)
    r = pa.rate(clean)
    return {
        "children_per_move": u["children_probed_per_move"],
        "roots_per_move": u["probe_roots_per_move"],
        "scanned_per_move": u["probe_roots_scanned_per_move"],
        "children_per_root": u["children_per_root"],
        "probe_ms_per_move": u["probe_ms_per_move_inclusive"],
        "predicate_ms_per_move": u["predicate_ms_per_move"],
        "predicate_us_per_root": u["predicate_us_per_root"],
        "us_per_probed_child": u["us_per_probed_child_inclusive"],
        "hit_rate_per_child": u["hit_rate_per_child"],
        "search_ms_per_move": float(np.mean([x["search_ms"]
                                             for x in clean["rows"]])),
        "nn_full": r["nn_full"],
        "sims_per_move": r["sims_per_move"],
        "p99_ms": r["p99_ms"],
        "gate_children_per_move": GATE_CHILDREN_PER_MOVE,
        "gate_probe_ms_per_move": GATE_PROBE_MS,
        "units": u,
        "counters": counting["counters"],
    }


# ----------------------------------------------------------------------
# #49c -- allocation and object churn
# ----------------------------------------------------------------------

def alloc_study(spec, device, positions, warmup=2):
    """Object churn per move, measured. NOT optimised -- the brief says measure.

    `sys.getallocatedblocks()` is a single C call returning live pymalloc
    blocks, so it can be read on the move path without perturbing it. It is a
    NET measure: it says how many blocks are alive, never how many were
    created. Gross creation is therefore counted directly for the two types
    that dominate -- GameState, via the clone that makes every one of them
    inside the search, and MCTSNode, via the engine's own
    `stat_nodes_created`. Anything claiming a total Python allocation count
    would be `tracemalloc`, which is a ~10x slowdown and would measure a
    search that does not run.
    """
    p = TimedPlayer(spec, device)
    n_clone = [0]
    raw_clone = GameState.__dict__["clone"]

    def counting_clone(self):
        n_clone[0] += 1
        return raw_clone(self)

    GameState.clone = counting_clone
    rows = []
    try:
        for state, _phase in positions[:warmup]:
            with torch.no_grad():
                _pi, root = p.mcts.search(state.clone())
            TreeReuseSearcher.release(root)
            gc.collect()
        for state, phase in positions:
            gc.collect()
            base_blocks = sys.getallocatedblocks()
            p.mcts.reset_stats()
            n_clone[0] = 0
            t0 = time.perf_counter()
            with torch.no_grad():
                pi, root = p.mcts.search(state.clone())
            dt = (time.perf_counter() - t0) * 1000.0
            peak_blocks = sys.getallocatedblocks()
            last = p.mcts.last
            TreeReuseSearcher.release(root)
            after_blocks = sys.getallocatedblocks()
            gc.collect()
            rows.append({
                "phase": phase,
                "search_ms": dt,
                "clones": n_clone[0],
                "nodes_created": p.mcts.stat_nodes_created,
                "expansions": last["nodes_expanded"],
                "sims": last["simulations_completed"],
                "probes": last["probes"],
                "blocks_base": base_blocks,
                "blocks_peak": peak_blocks,
                "blocks_after_release": after_blocks,
            })
    finally:
        GameState.clone = raw_clone

    def mean(k):
        return float(np.mean([r[k] for r in rows])) if rows else 0.0

    live = [r["blocks_peak"] - r["blocks_base"] for r in rows]
    held = [r["blocks_after_release"] - r["blocks_base"] for r in rows]
    return {
        "moves": len(rows),
        "rows": rows,
        "clones_per_move": mean("clones"),
        "nodes_created_per_move": mean("nodes_created"),
        "expansions_per_move": mean("expansions"),
        "probes_per_move": mean("probes"),
        "search_ms_per_move": mean("search_ms"),
        "peak_live_blocks": float(np.mean(live)) if live else 0.0,
        "peak_live_blocks_max": max(live) if live else 0,
        "blocks_held_after_release": float(np.mean(held)) if held else 0.0,
        "blocks_per_node": (float(np.mean(live)) / mean("nodes_created")
                            if mean("nodes_created") else 0.0),
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def render(payload):
    print()
    print("=" * 78)
    print("#49 -- the residual probe path on %s, split" % SPEC)
    print("=" * 78)
    print("git %s   device %s   seed %d"
          % (payload["git_head"], payload["device"], payload["seed"]))

    if "boundary" in payload:
        b = payload["boundary"]
        print()
        print("-- primitive floor on this box ------------------------------")
        for k, v in b.items():
            print("  %-34s%10.1f ns" % (k, v))
        print()
        print("  the shipped probe body, from primitives alone:")
        shipped = (b["python bound method"] + b["pybind bound method, 1 arg"]
                   + b["GameState.winner (property)"] + b["2-tuple build"]
                   + b["GameState.winner (property)"])
        candidate = (b["pybind bound method, 1 arg"]
                     + b["GameState._raw_winner"])
        print("    %-38s%10.1f ns/child" % ("make_move + winner + winner",
                                            shipped))
        print("    %-38s%10.1f ns/child" % ("cpp make_move + _raw_winner",
                                            candidate))
        print("    %-38s%10.1f ns/child" % ("difference", shipped - candidate))

    for key, title in (("ladder_admitted",
                        "LADDER: roots the filter ADMITS (the residual)"),
                       ("ladder_all",
                        "LADDER: all roots (what legacy scanned)")):
        lad = payload.get(key)
        if not lad:
            continue
        print()
        print("-- %s ---------------" % title)
        print("  %d roots, %d children, %.2f children/root, %d reps"
              % (lad["roots"], lad["children"], lad["children_per_root"],
                 lad["reps"]))
        print()
        print("  %-42s%12s%12s" % ("cumulative rung", "us/child", "step"))
        last = 0.0
        for i, (name, _f, _u) in enumerate(RUNGS):
            cur = lad["rungs"][name]["us_per_child"]
            print("  %-42s%12.4f%12.4f" % (name, cur, cur - last))
            last = cur
        for k, v in lad.get("extra", {}).items():
            print("  %-42s%12.4f" % (k, v))
        print("  %-42s%12.4f us/root" % ("could_end (per root)",
                                         lad["predicate_us_per_root"]))
        print("  %-42s%12.4f" % ("replica + predicate",
                                 lad.get("replica_us_per_child", 0.0)))
        print("  %-42s%12.4f  (filter %s)"
              % ("REAL _mark_terminal_children",
                 lad["production_us_per_child"],
                 "on" if lad.get("production_filtered") else "off"))
        print("  %-42s%+11.2f%%" % ("replica vs production",
                                    100 * lad["replica_vs_production"]))

    if payload.get("insitu"):
        i = payload["insitu"]
        print()
        print("-- in-situ envelope, THIS run's own positions ---------------")
        for label, key, fmt in (
                ("probe roots per move", "roots_per_move", "%12.1f"),
                ("  of which scanned", "scanned_per_move", "%12.1f"),
                ("children probed per move", "children_per_move", "%12.1f"),
                ("children per scanned root", "children_per_root", "%12.2f"),
                ("us per probed child", "us_per_probed_child", "%12.3f"),
                ("probe ms/move (wrapper-priced)", "probe_ms_per_move",
                 "%12.2f"),
                ("  of which could_end", "predicate_ms_per_move", "%12.2f"),
                ("search ms/move", "search_ms_per_move", "%12.1f"),
                ("nn/second x deadline", "nn_full", "%12.1f")):
            print(("  %-40s" + fmt) % (label, i[key]))
        print()
        print("  the #48c gate saw %.1f children/move and %.2f ms; this run"
              % (i["gate_children_per_move"], i["gate_probe_ms_per_move"]))
        print("  sees %.1f and %.2f. Wall-clock searches on a different"
              % (i["children_per_move"], i["probe_ms_per_move"]))
        print("  sample of positions probe a different amount, which is why")
        print("  every ms/move below uses THIS run's count, not the gate's.")

    if payload.get("split"):
        s = payload["split"]
        print()
        print("-- the residual probe path, split ---------------------------")
        print("  %-44s%10s%9s" % ("", "ms/move", "share"))
        for name, ms in s["rows"]:
            print("  %-44s%10.2f%8.1f%%"
                  % (name, ms, 100 * ms / s["insitu_ms"] if s["insitu_ms"]
                     else 0.0))
        print("  %-44s%10.2f" % ("-- ladder subtotal", s["ladder_ms"]))
        print("  %-44s%10.2f%8.1f%%"
              % ("allocator/cache spillover (residual)", s["spillover_ms"],
                 100 * s["spillover_ms"] / s["insitu_ms"]
                 if s["insitu_ms"] else 0.0))
        print("  %-44s%10.2f" % ("== in-situ wrapper-priced total",
                                 s["insitu_ms"]))
        print()
        print("  (the in-situ total is itself a LOWER bound: the two")
        print("   uninstrumented gates that checked it came in %.0f%% and %.0f%%"
              % (100 * (UNDERPRICE_RATIOS[0] - 1),
                 100 * (UNDERPRICE_RATIOS[1] - 1)))
        print("   above the wrapper-derived prediction. Do not quote the")
        print("   subtotal as an optimisation ceiling.)")

    if "crossing" in payload:
        c = payload["crossing"]
        print()
        print("-- #49b: the redundant winner crossing ----------------------")
        print("  %-40s%12.4f us/child" % ("measured (rung 4 - rung 3)",
                                          c["measured_us_per_child"]))
        print("  %-40s%12.4f us/child" % ("structural (primitives summed)",
                                          c["structural_us_per_child"]))
        print("  %-40s%12.2f" % ("agreement (measured/structural)",
                                 c["agreement"]))
        print("  %-40s%12.1f" % ("children probed per move",
                                 c["children_per_move"]))
        print("  %-40s%12.2f ms/move" % ("PROJECTED saving",
                                         c["projected_ms_per_move"]))
        print("  %-40s%12s" % ("decision band", c["band"]))
        print()
        print("  %-40s%12.2f ms/move" % ("a FUSED native probe would add",
                                         c["fused_extra_ms_per_move"]))
        print("  %-40s%12.2f ms/move"
              % ("inlining _terminal_value would add",
                 c["inline_terminal_value_ms_per_move"]))

    if "elsewhere" in payload:
        e = payload["elsewhere"]
        print()
        print("-- the SAME redundancy at the descent call site -------------")
        print("  %-40s%12.1f" % ("make_move from probes, per move",
                                 e["probe_calls_per_move"]))
        print("  %-40s%12.1f" % ("make_move in the descent, per move",
                                 e["descent_calls_per_move"]))
        print("  %-40s%12.1f to 1" % ("descent : probe",
                                      e["ratio_descent_to_probe"]))
        if "winner_reads_per_make_move_elsewhere" in e:
            print("  %-40s%12.2f  <- must be ~2.0"
                  % ("winner reads per descent make_move",
                     e["winner_reads_per_make_move_elsewhere"]))
        print()
        print("  %-40s%12.2f ms/move  %s"
              % ("probe site alone", e["probe_ms_per_move"], ""))
        print("  %-40s%12.2f ms/move  %s"
              % ("descent site alone", e["descent_ms_per_move"],
                 e["descent_band"]))
        print("  %-40s%12.2f ms/move  %s"
              % ("both sites", e["combined_ms_per_move"],
                 e["combined_band"]))
        print("  %-40s%12.2f%%" % ("as a share of the search",
                                   100 * e["share_of_search"]))

    if payload.get("price_check"):
        p = payload["price_check"]
        print()
        print("-- is the residual the INSTRUMENT? --------------------------")
        print("  %d roots / %d children, one wrapped pass"
              % (p["roots"], p["children"]))
        print("  %-40s%12.4f us/child" % ("bare (no instrument)",
                                          p["bare_us_per_child"]))
        print("  %-40s%12.4f us/child" % ("raw, wrapped",
                                          p["raw_wrapped_us_per_child"]))
        print("  %-40s%12.4f us/child" % ("  wrapper subtracted",
                                          p["wrapper_subtracted_us_per_child"]))
        print("  %-40s%12.4f us/child" % ("priced result",
                                          p["recovered_us_per_child"]))
        print("  %-40s%12.3f  <- 1.00 means the pricing is sound"
              % ("priced / bare", p["recovered_over_bare"]))
        print()
        print("  %-40s%12.3f us/call" % ("wrapper price, calibrated",
                                         p["calibrated_price_us"]))
        print("  %-40s%12.3f us/call" % ("wrapper price implied by the bare "
                                         "run", p["implied_price_us"]))

    if payload.get("locality"):
        print()
        print("-- is the residual allocator pressure, or first touch? ------")
        print("  %10s%12s%14s%12s" % ("roots", "children", "kB touched",
                                      "us/child"))
        for r in payload["locality"]:
            print("  %10d%12d%14.0f%12.4f"
                  % (r["roots"], r["children"], r["kb_touched"],
                     r["us_per_child"]))
        first, last = payload["locality"][0], payload["locality"][-1]
        print("  %-40s%+11.1f%%"
              % ("smallest to largest working set",
                 100 * (last["us_per_child"] / first["us_per_child"] - 1)
                 if first["us_per_child"] else 0.0))
        if payload.get("insitu"):
            print("  %-40s%12.4f us/child"
                  % ("production, wrapper-priced",
                     payload["insitu"]["us_per_probed_child"]))

    if "heap" in payload:
        h = payload["heap"]
        print()
        print("-- allocator/cache spillover, measured directly -------------")
        print("  %-40s%12.4f us/child" % ("clean heap", h["clean_us"]))
        print("  %-40s%12.4f us/child" % ("with a live search tree",
                                          h["loaded_us"]))
        print("  %-40s%12d" % ("live nodes pinned", h["pinned_nodes"]))
        print("  %-40s%+11.1f%%" % ("cost of heap pressure",
                                    100 * h["ratio"]))
        print("  %-40s%12.2f ms/move" % ("projected at post-filter volume",
                                         h["delta_ms_per_move"]))

    if "alloc" in payload:
        a = payload["alloc"]
        print()
        print("-- #49c: object churn per move (MEASURED, not optimised) ----")
        for label, key, fmt in (
                ("GameState clones created", "clones_per_move", "%12.1f"),
                ("MCTSNode objects created", "nodes_created_per_move",
                 "%12.1f"),
                ("expansions", "expansions_per_move", "%12.1f"),
                ("children probed", "probes_per_move", "%12.1f"),
                ("peak live pymalloc blocks", "peak_live_blocks", "%12.1f"),
                ("blocks still held after release",
                 "blocks_held_after_release", "%12.1f"),
                ("live blocks per node created", "blocks_per_node",
                 "%12.2f")):
            print(("  %-40s" + fmt) % (label, a[key]))
    print()


def build_split(payload):
    """Turn the ladder into ms/move against the in-situ envelope.

    Every factor comes from THIS payload: unit costs from the ladder, call
    counts from the counting arm, the envelope from the timed arm. Nothing is
    scaled by a number from another run.
    """
    lad = payload.get("ladder_admitted")
    ins = payload.get("insitu")
    if not lad or not ins:
        return None
    children = ins["children_per_move"]
    roots = ins["roots_per_move"]
    scanned = ins["scanned_per_move"]
    envelope = ins["probe_ms_per_move"]
    rows = []
    rows.append(("predicate (could_end, every root)",
                 lad["predicate_us_per_root"] * roots / 1000.0))
    prices = payload.get("boundary", {})
    cross_floor = prices.get("pybind bound method, 0 args", 0.0) / 1000.0
    crossing_ms = cross_floor * children / 1000.0
    for name, step in lad["steps"].items():
        ms = step * children / 1000.0
        if name in ("make_move native execution", "probe.winner readback"):
            # Split the crossing out of the two rungs that pay one, so
            # "pybind call overhead" is a row rather than a claim.
            rows.append((name, max(0.0, ms - crossing_ms)))
        else:
            rows.append((name, ms))
    # The two crossings the candidate path would still pay, priced once.
    rows.append(("pybind call overhead (2 crossings/child)", 2 * crossing_ms))
    # Skipped roots still enter the method, bump a counter and return.
    skipped = roots - scanned
    frame = prices.get("python bound method", 0.0) / 1000.0
    rows.append(("skipped-root method entry (%d roots)" % int(skipped),
                 frame * skipped / 1000.0))
    ladder_ms = sum(ms for _n, ms in rows)
    return {"rows": rows, "ladder_ms": ladder_ms,
            "insitu_ms": envelope,
            "children_per_move": children, "roots_per_move": roots,
            "spillover_ms": envelope - ladder_ms,
            "spillover_share": ((envelope - ladder_ms) / envelope
                                if envelope else 0.0)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="all",
                    choices=("boundary", "ladder", "heap", "alloc",
                             "crossing", "locality", "pricecheck", "all"))
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--position-games", type=int, default=3)
    ap.add_argument("--capture-positions", type=int, default=8)
    # The reservoir, per bucket. Bigger than the ladder needs on purpose: the
    # locality arm has to walk a working set large enough to fall out of cache,
    # and 2 x 1200 roots does not.
    ap.add_argument("--roots", type=int, default=4000)
    # The ladder's own corpus size, held FIXED and separate. If the ladder grew
    # with the reservoir, every unit cost in the split would silently depend on
    # a flag whose purpose is a different arm -- and the locality curve says
    # that dependence is real.
    ap.add_argument("--ladder-roots", type=int, default=1200)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--seed", type=int, default=COST_SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--tag", default="probe49")
    ap.add_argument("--rerender", default="")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.rerender:
        with open(args.rerender) as fh:
            payload = json.load(fh)
        payload["split"] = build_split(payload)
        render(payload)
        with open(args.rerender, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return

    assert_frozen_sources()
    if _CPP is None or "winner" not in GameState.__dict__:
        raise SystemExit("[X] the C++ engine is not loaded. Every crossing "
                         "price below would be a fiction.\n"
                         "    cmake --build engine/cpp/build --config Release")

    with single_instance("probe_cost"):
        base = gpu_baseline()
        if base:
            print("[..] GPU load before starting: %.0f%% mean, %.0f%% peak"
                  % (base["mean_pct"], base["max_pct"]))
        gc.disable()
        print("[!] automatic cyclic GC off; collecting at boundaries only")

        payload = {"tag": args.tag, "mode": args.mode, "spec": SPEC,
                   "seed": args.seed, "device": args.device,
                   "reps": args.reps,
                   "git_head": engine_registry.git_head(),
                   "environment": engine_registry.environment(),
                   "environment_drift": engine_registry.env_drift(),
                   "gpu_baseline_before_run": base}
        path = os.path.join(OUT_DIR, "%s.json" % args.tag)

        def save():
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)

        if args.mode in ("boundary", "ladder", "crossing", "heap", "all"):
            print("[..] pricing primitives")
            payload["boundary"] = boundary_prices()
            save()

        need_corpus = args.mode in ("ladder", "crossing", "heap", "locality",
                                    "pricecheck", "all")
        need_positions = need_corpus or args.mode in ("alloc", "all")
        positions = []
        if need_positions:
            print("[..] sampling positions from %d games" % args.position_games)
            from tools.probe_ablation import positions_for
            positions = positions_for(SPEC, args.position_games, args.device,
                                      args.seed, args.positions)
            print("[..] %d fixed positions" % len(positions))

        player = cap = None
        if need_corpus:
            print("[..] capturing probe roots from %d real searches"
                  % args.capture_positions)
            player, cap = capture_roots(SPEC, args.device,
                                        positions[:args.capture_positions],
                                        args.roots, args.seed)
            payload["corpus"] = {
                "admitted_seen": cap.n_admitted,
                "skipped_seen": cap.n_skipped,
                "admitted_kept": len(cap.admitted),
                "skipped_kept": len(cap.skipped),
                "admit_rate": (cap.n_admitted
                               / (cap.n_admitted + cap.n_skipped)
                               if (cap.n_admitted + cap.n_skipped) else 0.0),
            }
            print("[..] %d admitted / %d skipped roots seen (%.4f admit rate)"
                  % (cap.n_admitted, cap.n_skipped,
                     payload["corpus"]["admit_rate"]))
            save()

        if args.mode in ("ladder", "crossing", "heap", "locality",
                         "pricecheck", "all"):
            gc.collect()
            nl = args.ladder_roots
            adm = rebuild(player.mcts, cap.admitted[:nl])
            print("[..] ladder over %d ADMITTED roots" % len(adm))
            # `filtered` left as the engine has it: every root here passes, so
            # the flag changes nothing and the honest arm is the shipped one.
            payload["ladder_admitted"] = run_ladder(
                adm, player.mcts, args.reps, "admitted",
                filtered=player.mcts.probe_filter)
            save()
            allr = rebuild(player.mcts,
                           cap.admitted[:nl] + cap.skipped[:nl])
            print("[..] ladder over %d ALL roots" % len(allr))
            # Filter FORCED OFF. This corpus is the legacy population, and a
            # filtered production arm would skip half of it -- reported as the
            # replica being 76% too expensive, which is what the first run
            # said before this argument existed.
            payload["ladder_all"] = run_ladder(allr, player.mcts, args.reps,
                                               "all", filtered=False)
            allr = None
            gc.collect()
            save()

        if args.mode in ("pricecheck", "all"):
            print("[..] instrument price check")
            payload["price_check"] = price_check(player.mcts, adm, args.reps)
            save()

        if args.mode in ("locality", "all"):
            print("[..] locality curve")
            pool = cap.admitted + cap.skipped
            sizes = [n for n in (75, 150, 300, 600, 1200, 2400, 4800,
                                 len(pool)) if n <= len(pool)]
            payload["locality"] = locality_curve(player.mcts, pool,
                                                 sorted(set(sizes)),
                                                 args.reps)
            save()

        if args.mode in ("heap", "all"):
            print("[..] heap-pressure arm")
            # The all-roots corpus was released above, before the clean rung
            # is compared against anything. It is ~18,000 live nodes that did
            # NOT exist when that rung was measured, so leaving it alive would
            # put part of the very heap pressure being measured into the
            # loaded arm from a source that is the instrument itself.
            gc.collect()
            clean = payload["ladder_admitted"]["rungs"][
                "+ tuple + winner property"]["us_per_child"]
            # Pin a real tree. Not a synthetic list of objects: the thing being
            # tested is what a live 40k-node MCTS tree does to the allocator,
            # and a list of identical dummies has a different size class mix.
            player.mcts.reset_stats()
            with torch.no_grad():
                _pi, pinned = player.mcts.search(positions[0][0].clone())
            n_pinned = player.mcts.stat_nodes_created
            reset(adm)
            mn, _med, _ = bench(lambda: r4_current(adm, player.mcts),
                                args.reps)
            n_children = sum(len(n.children) for n, _ in adm)
            loaded = mn / n_children * 1e6
            TreeReuseSearcher.release(pinned)
            pinned = None
            gc.collect()
            payload["heap"] = {
                "clean_us": clean, "loaded_us": loaded,
                "pinned_nodes": int(n_pinned),
                "ratio": (loaded / clean - 1.0) if clean else 0.0,
                "delta_us_per_child": loaded - clean,
            }
            save()

        if args.mode in ("ladder", "crossing", "heap", "locality",
                         "pricecheck", "all"):
            # The envelope and the call volume, on the SAME positions. It runs
            # last of the search arms so the ladder never competes with a
            # deadline-bound search for the CPU.
            print("[..] in-situ envelope (clean / counting / timed)")
            adm = None
            del player, cap
            gc.collect()
            payload["insitu"] = insitu_envelope(SPEC, args.device, positions)
            save()
            payload["crossing"] = crossing(
                payload["ladder_admitted"], payload["boundary"],
                payload["insitu"]["children_per_move"])
            payload["elsewhere"] = elsewhere(payload["crossing"],
                                             payload["insitu"])
            if "heap" in payload:
                payload["heap"]["delta_ms_per_move"] = (
                    payload["heap"]["delta_us_per_child"]
                    * payload["insitu"]["children_per_move"] / 1000.0)
            save()

        if args.mode in ("alloc", "all"):
            print("[..] allocation / object churn arm")
            gc.collect()
            payload["alloc"] = alloc_study(SPEC, args.device, positions)
            save()

        payload["split"] = build_split(payload)
        render(payload)
        save()
        print("[OK] wrote %s" % path)


if __name__ == "__main__":
    main()
