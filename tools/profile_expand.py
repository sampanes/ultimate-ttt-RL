"""Sync versus transfer versus compute inside the expansion wave.

`tools/profile_tree.py --mode wrap` left one bucket undecomposed and it is the
largest one: `_expand_wave` plus `wave_planes`, together up to 40% of a move.
That bucket is a mixture of four different things with four different fixes --
GPU compute, host-to-device copies, device-to-host copies, and CPU time spent
blocked waiting for work queued earlier -- and a wrapper around the whole call
cannot tell them apart, because CUDA is asynchronous. The wall time of
`probs.cpu()` is not transfer time; it is mostly the moment the forward pass
that was launched three statements earlier finally becomes visible.

Getting this wrong picks the wrong optimization. Compute-bound says leave the
transfers alone. Sync-dominated with modest GPU time says look at launch
fragmentation and serial host/device dependencies. Genuinely transfer-bound
says cut round trips. Those are three different pieces of work.

TWO INDEPENDENT INSTRUMENTS, because one instrument is a story:

    events   CUDA events on the stream. Gives true GPU-side durations for each
             device operation without ever blocking the CPU. Cannot separate
             "waiting" from "copying" inside a `.cpu()`, because that call is
             one interval on the CPU clock.
    sync     an explicit `torch.cuda.synchronize()` immediately before the
             first `.cpu()`. That splits the same interval into pure wait and
             pure copy -- and PERTURBS the workload, which is why it is a
             separate build and never the number quoted alone.

They measure the same quantity by different routes. Agreement is the evidence.

BOTH ARE SAMPLED, one wave in `--sample-every` (default 100). Instrumenting
every wave cost as much as the wave itself: twelve event records plus six
queries came to ~0.6 ms on WDDM against a device section of ~0.6 ms, and even
at one wave in twenty the events build lost 25% of its simulations. CPU timing
is cheap and stays on every wave. The two populations are disjoint -- clean
waves give the CPU column, sampled waves the device column -- because a sampled
wave's CPU timings are inflated by the very instrument being priced.

THE PROFILING BUILD IS NOT THE DEPLOYMENT BUILD, and that is measured, not
assumed. The untouched engine runs FIRST, on the same openings, and its network
evaluations per move is the reference every per-move figure is scaled by. The
primary column is PER LEAF, which does not move with the perturbation. Scaling
by the instrumented run's own rate would shrink every figure by exactly the
instrument's cost -- a report that hides its own error.

Per leaf rather than per wave because waves per move is not stable: only a
minority of waves reach the network at all, the rest select entirely into nodes
already expanded, terminal or proven, and that fraction swings with how
tactical the game is. The report states it.

MCTS IS NOT EDITED -- `agents/mcts.py` is frozen and hash-gated. The two
functions that need splitting are replicated here, statement for statement, and
`tools/test_profile_expand.py` asserts the replicas produce bit-identical
children, priors and leaf values against the frozen originals. A replica that
has drifted is a profile of code nobody runs.

    python -m tools.profile_expand --mode all --engine pocket_r35 --games 12

Modes:
    perturb   what the instrumentation costs, in sims/move (runs first in
              `all`, because its untouched arm sets the reference leaf rate)
    events    CUDA events, no blocking sync (the primary measurement)
    sync      explicit drain before the first pull (the cross-check)
    all       all three, plus the decision tree
"""

import argparse
import collections
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from agents import agent_base
from agents import mcts as mcts_mod
from agents.mcts import MCTS, MCTSNode
from engine.constants import X, O
from engine.rules import rule_utl_valid_moves
from tools import engine_registry
from tools.arena_1s import TimedPlayer, play_match
from tools.profile_tree import assert_frozen_sources, instrument_player, new_ctx

OUT_DIR = os.path.join("results", "profile_expand")

EXPAND_SEED = engine_registry.SEEDS["expand"]

# Ordered exactly as the wave executes, so the table reads as a timeline.
# `kind` is what the number means, which differs per row and is the whole point:
#   host    CPU work, nothing queued
#   launch  CPU time to ENQUEUE an async op -- not the op's duration
#   drain   CPU time blocked waiting for previously queued work (sync build)
#   copy    a transfer; CPU column is the blocking call, GPU column the DMA
#   gpu     device compute; the CPU column is launch only
SEGMENTS = [
    ("host: plane fill",            "host"),
    ("H2D: planes",                 "copy"),
    ("network forward",             "gpu"),
    ("host: legal masks",           "host"),
    ("H2D: mask",                   "copy"),
    ("device: mask + softmax",      "gpu"),
    ("first host synchronization",  "drain"),
    ("D2H: probs",                  "copy"),
    ("D2H: values",                 "copy"),
    ("host: child construction",    "host"),
    ("host: make_move / probes",    "host"),
]
SEG_KIND = dict(SEGMENTS)
SEG_ORDER = [s for s, _k in SEGMENTS]

# Segments that carry a CUDA event pair. The host-side ones have nothing on the
# stream to time.
GPU_SEGMENTS = ("H2D: planes", "network forward", "H2D: mask",
                "device: mask + softmax", "D2H: probs", "D2H: values")

PHASES = ("early", "mid", "late")

# The frozen engine's wave size. Under a clock `_run_budget` advances
# `sims_done` by exactly WAVE_SIZE per chunk, so sims/WAVE_SIZE is the number of
# chunks -- against which the observed wave count says how many of them reached
# the network at all.
WAVE_SIZE = 8


class Slot:
    """One wave's worth of timings, plus the CUDA events that produced them.

    Events are REUSED from a ring rather than allocated per wave, and are read
    back only when the slot comes round again. By then several blocking `.cpu()`
    calls have gone by on the same stream, so every event is long complete and
    `elapsed_time` needs no synchronization of its own. That is what keeps the
    `events` build free of blocking syncs.
    """

    def __init__(self):
        self.use_events = False
        self.sampled = False
        self.ev = {}
        self.order = []
        self.cpu = {}
        self.phase = "mid"
        self.k = 0
        self.pending = False

    def rec(self, seg, which):
        if not self.use_events:
            return
        pair = self.ev.get(seg)
        if pair is None:
            pair = self.ev[seg] = (torch.cuda.Event(enable_timing=True),
                                   torch.cuda.Event(enable_timing=True))
        pair[which].record()
        if which == 0:
            self.order.append(seg)
            return
        # AN EVENT RECORDED ON AN IDLE STREAM IS NOT TIMESTAMPED WHEN YOU
        # RECORD IT. Under WDDM the driver batches submissions, so an end event
        # after a blocking `.cpu()` sits unsubmitted until something else
        # flushes the queue -- which is the NEXT wave. Its interval then
        # silently absorbs child construction, the terminal probes and the next
        # wave's whole selection descent. Measured before this line existed:
        # `D2H: values` came back at 342.8 ms/move for an 8-float copy, and the
        # six device segments summed to 901.9 ms of a 904.4 ms move, i.e. a
        # GPU claimed to be 100% busy while ~200 ms/move of pure-Python tree
        # work ran with nothing queued.
        #
        # `query()` is a non-blocking poll, and on WDDM polling forces the
        # command buffer to be submitted. It does not wait, so this is still a
        # build with no blocking synchronization in it.
        pair[1].query()


class ExpandProbe:
    """Instrumented replicas of `wave_planes` and `MCTS._expand_wave`.

    Installed by monkey-patch, removed in a finally. Both replicas are
    statement-for-statement copies of the frozen originals with timing boundaries
    inserted between statements -- no reordering, no fused operations, no
    changed dtypes. `test_profile_expand.py` proves that against the originals.
    """

    def __init__(self, mode, sample_every=20, ring=4):
        assert mode in ("events", "sync", "off")
        self.mode = mode
        # THE INSTRUMENT MUST NOT COST WHAT IT MEASURES. Twelve event records
        # plus six queries per wave came to ~0.6 ms on WDDM, against a wave
        # whose whole device section is ~0.6 ms -- it halved the simulation
        # count, which is the same trap the sampler fell into. So events are
        # SAMPLED. CPU timing stays on every wave (eleven perf_counter pairs,
        # under 1 us) and supplies the CPU column; the sampled waves supply the
        # device column and are excluded from the CPU column, because theirs is
        # inflated. Two disjoint populations, each read for what it is honest
        # about.
        self.sample_every = max(1, int(sample_every))
        self.ctx = new_ctx()
        self.cpu = collections.defaultdict(float)    # clean waves, seconds
        self.cpu_s = collections.defaultdict(float)  # sampled waves, seconds
        self.gpu = collections.defaultdict(float)    # sampled waves, ms
        self.waves = collections.Counter()           # phase -> clean waves
        self.waves_s = collections.Counter()         # phase -> sampled waves
        self.leaves = collections.Counter()          # phase -> clean leaves
        self.leaves_s = collections.Counter()        # phase -> sampled leaves
        self.ksizes = collections.Counter()          # k -> waves (all)
        self.ring = [Slot() for _ in range(ring)]
        self.idx = 0
        self.n_waves = 0
        self.slot = None
        self.saved = []

    # -- ring management ------------------------------------------------

    def _begin(self, phase, k):
        slot = self.ring[self.idx]
        self.idx = (self.idx + 1) % len(self.ring)
        if slot.pending:
            self._harvest(slot)
        slot.cpu.clear()
        del slot.order[:]
        slot.phase = phase
        slot.k = k
        slot.sampled = (self.n_waves % self.sample_every) == 0
        slot.use_events = slot.sampled and self.mode == "events"
        slot.pending = True
        self.n_waves += 1
        self.slot = slot
        return slot

    def _harvest(self, slot):
        ph = slot.phase
        into = self.cpu_s if slot.sampled else self.cpu
        for seg, sec in slot.cpu.items():
            into[(ph, seg)] += sec
        if slot.use_events and slot.order:
            for seg in slot.order:
                a, b = slot.ev[seg]
                self.gpu[(ph, seg)] += a.elapsed_time(b)
            first = slot.ev[slot.order[0]][0]
            last = slot.ev[slot.order[-1]][1]
            # Wall of the whole device timeline for this wave. Subtracting the
            # per-op sum gives GPU IDLE inside the wave -- launch fragmentation,
            # which is a different disease from either compute or transfer.
            self.gpu[(ph, "_span")] += first.elapsed_time(last)
        if slot.sampled:
            self.waves_s[ph] += 1
            self.leaves_s[ph] += slot.k
        else:
            self.waves[ph] += 1
            self.leaves[ph] += slot.k
        self.ksizes[slot.k] += 1
        slot.pending = False

    def flush(self):
        """Drain the ring at the end of the match.

        This is the only blocking synchronize the EVENTS build ever performs,
        and it happens after the last move, outside every measurement. (The
        sync build drains once per sampled wave, on purpose, and records
        nothing here.)
        """
        if any(s.pending and s.use_events for s in self.ring):
            torch.cuda.synchronize()
        for slot in self.ring:
            if slot.pending:
                self._harvest(slot)
        self.slot = None

    # -- replicas -------------------------------------------------------

    def wave_planes(self, states, device):
        """Replica of agents.agent_base.wave_planes, split at the H2D."""
        if not self.ctx["on"]:
            return agent_base.wave_planes(states, device)
        slot = self._begin(self.ctx["phase"], len(states))

        t0 = time.perf_counter()
        k = len(states)
        buf = np.empty((k, 7, 9, 9), dtype=np.float32)
        if k and agent_base._has_fill_planes(states[0]):
            for i, s in enumerate(states):
                s.fill_planes(buf[i])
        else:
            for i, s in enumerate(states):
                buf[i] = agent_base.board_to_tensor_from_gamestate(s).numpy()
        host = torch.from_numpy(buf)
        t1 = time.perf_counter()
        slot.cpu["host: plane fill"] = t1 - t0

        slot.rec("H2D: planes", 0)
        xs = host.to(device)
        slot.rec("H2D: planes", 1)
        slot.cpu["H2D: planes"] = time.perf_counter() - t1
        return xs

    def forward_both(self, orig, model, x):
        slot = self.slot
        if slot is None:
            # The single-leaf root expansion in `_expand`. Real work, but not
            # part of a wave, and folding it in would corrupt the per-leaf
            # figures the port decision is read off.
            return orig(model, x)
        t0 = time.perf_counter()
        slot.rec("network forward", 0)
        out = orig(model, x)
        slot.rec("network forward", 1)
        slot.cpu["network forward"] = time.perf_counter() - t0
        return out

    def expand_wave(self, mcts, to_eval, logits_b, values_b):
        """Replica of MCTS._expand_wave, split at every device boundary."""
        slot = self.slot
        if slot is None:
            return self._orig_expand_wave(mcts, to_eval, logits_b, values_b)

        t0 = time.perf_counter()
        k = len(to_eval)
        valids = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
                  for _, _, s in to_eval]
        mask = np.zeros((k, 81), dtype=bool)
        for i, v in enumerate(valids):
            mask[i, v] = True
        mask_host = torch.from_numpy(mask)
        t1 = time.perf_counter()
        slot.cpu["host: legal masks"] = t1 - t0

        slot.rec("H2D: mask", 0)
        mask_t = mask_host.to(logits_b.device)
        slot.rec("H2D: mask", 1)
        t2 = time.perf_counter()
        slot.cpu["H2D: mask"] = t2 - t1

        slot.rec("device: mask + softmax", 0)
        probs = F.softmax(logits_b.masked_fill(~mask_t, float("-inf")), dim=1)
        slot.rec("device: mask + softmax", 1)
        t3 = time.perf_counter()
        slot.cpu["device: mask + softmax"] = t3 - t2

        if self.mode == "sync" and slot.sampled:
            # PERTURBING ON PURPOSE, and only on sampled waves. With the stream
            # already drained the next `.cpu()` is a pure copy, so the wait and
            # the transfer stop being one number. Unsampled waves stay clean and
            # keep supplying the CPU column.
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            slot.cpu["first host synchronization"] = t4 - t3
        else:
            t4 = t3

        slot.rec("D2H: probs", 0)
        probs_host = probs.cpu()
        slot.rec("D2H: probs", 1)
        t5 = time.perf_counter()
        slot.cpu["D2H: probs"] = t5 - t4
        probs_np = probs_host.numpy()

        slot.rec("D2H: values", 0)
        values_host = values_b.reshape(-1).cpu()
        slot.rec("D2H: values", 1)
        t6 = time.perf_counter()
        slot.cpu["D2H: values"] = t6 - t5
        values_np = values_host.numpy()

        out = {}
        probe_s = 0.0
        for i, (_pi, node, state) in enumerate(to_eval):
            row = probs_np[i]
            next_to_play = O if state.player == X else X
            for mv in valids[i]:
                node.children[mv] = MCTSNode(parent=node, prior=float(row[mv]),
                                             move=mv, to_play=next_to_play)
            mcts.stat_expansions += 1
            if mcts.solve:
                tp = time.perf_counter()
                mcts._mark_terminal_children(node, state)
                probe_s += time.perf_counter() - tp
            out[id(node)] = float(values_np[i])
        t7 = time.perf_counter()
        slot.cpu["host: make_move / probes"] = probe_s
        slot.cpu["host: child construction"] = (t7 - t6) - probe_s

        self.slot = None
        return out

    # -- install / remove -----------------------------------------------

    def install(self, players):
        for p in players:
            if not p.mcts.batched_expand:
                raise SystemExit(
                    "[X] %s runs the per-leaf expansion path; this study only "
                    "describes _expand_wave. Profile a bexp=1 engine."
                    % p.name)
            instrument_player(p, self.ctx)

        probe = self
        self._orig_expand_wave = MCTS.__dict__["_expand_wave"]
        self.saved.append((mcts_mod, "wave_planes", mcts_mod.wave_planes))
        self.saved.append((MCTS, "_expand_wave", self._orig_expand_wave))
        mcts_mod.wave_planes = self.wave_planes

        # A PLAIN FUNCTION, not `probe.expand_wave`. A bound method stored on a
        # class is not a descriptor, so it is never re-bound on attribute
        # access and the MCTS instance would be silently swallowed into the
        # first parameter -- every argument shifted by one.
        def expand_wave(mcts, to_eval, logits_b, values_b):
            return probe.expand_wave(mcts, to_eval, logits_b, values_b)
        MCTS._expand_wave = expand_wave

        # One class-level patch covers both players; the wave guard in
        # forward_both keeps the root expansion out of the wave totals.
        cls = type(players[0].model)
        orig_fb = cls.__dict__["forward_both"]
        self.saved.append((cls, "forward_both", orig_fb))

        def forward_both(model, x):
            return probe.forward_both(orig_fb, model, x)
        cls.forward_both = forward_both

    def remove(self):
        for holder, attr, raw in reversed(self.saved):
            setattr(holder, attr, raw)
        del self.saved[:]


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------

def _play(engine, games, device, seed, probe=None):
    pa = TimedPlayer("engine:%s" % engine, device)
    pb = TimedPlayer("engine:%s" % engine, device)
    if probe is not None:
        probe.install([pa, pb])
    try:
        play_match(pa, pb, games, seed, warmup=2, gc_mode="deferred")
    finally:
        if probe is not None:
            probe.remove()
            probe.flush()
    moves = len(pa.records) + len(pb.records)
    sims = sum(r[2] for r in pa.records) + sum(r[2] for r in pb.records)
    nn = sum(r[3] for r in pa.records) + sum(r[3] for r in pb.records)
    search_ms = sum(r[1] for r in pa.records) + sum(r[1] for r in pb.records)
    return {"moves": moves, "sims": sims, "nn_evals": nn,
            "sims_per_move": sims / moves if moves else 0.0,
            "nn_per_move": nn / moves if moves else 0.0,
            "search_ms_per_move": search_ms / moves if moves else 0.0,
            "fingerprint": (pa.provenance or {}).get("fingerprint"),
            "params": pa.net_info["params"], "budget_ms": pa.budget_ms}


def run_instrumented(engine, games, device, seed, mode, sample_every=100):
    """Per-WAVE and per-LEAF costs. Deliberately not per-move.

    An instrumented run completes fewer simulations than the frozen engine, so
    its own waves-per-move is depressed by exactly the amount the instrument
    costs. Multiplying a per-wave cost by that depressed rate would quietly
    scale every figure down by the perturbation -- the measurement would hide
    its own error. Per-wave and per-leaf costs are properties of the wave and
    do not move; the caller converts to per-move using the UNTOUCHED build's
    wave rate, and says so.
    """
    probe = ExpandProbe(mode, sample_every=sample_every)
    stats = _play(engine, games, device, seed, probe)
    moves = stats["moves"]
    clean_w = sum(probe.waves.values())
    samp_w = sum(probe.waves_s.values())
    waves = clean_w + samp_w
    clean_leaves = sum(probe.leaves.values())
    samp_leaves = sum(probe.leaves_s.values())
    leaves = clean_leaves + samp_leaves
    mean_k = clean_leaves / clean_w if clean_w else 0.0
    mean_k_s = samp_leaves / samp_w if samp_w else 0.0

    # PER LEAF IS THE UNIT THAT TRAVELS. Waves per move swings by a factor of
    # two between games -- a proven subtree costs a descent and no network at
    # all, so a tactical game runs far more waves that evaluate nothing -- and
    # it also falls with the instrument's own cost. A leaf is a leaf.
    cpu_us, leaf_us, gpu_us, gpu_leaf, samp_us, samp_leaf = {}, {}, {}, {}, {}, {}
    for seg in SEG_ORDER:
        c = sum(probe.cpu[(ph, seg)] for ph in PHASES)
        if c and clean_w:
            cpu_us[seg] = c * 1e6 / clean_w
            leaf_us[seg] = c * 1e6 / clean_leaves if clean_leaves else 0.0
        s = sum(probe.cpu_s[(ph, seg)] for ph in PHASES)
        if s and samp_w:
            samp_us[seg] = s * 1e6 / samp_w
            samp_leaf[seg] = s * 1e6 / samp_leaves if samp_leaves else 0.0
        g = sum(probe.gpu[(ph, seg)] for ph in PHASES)
        if g and samp_w:
            gpu_us[seg] = g * 1000.0 / samp_w
            gpu_leaf[seg] = g * 1000.0 / samp_leaves if samp_leaves else 0.0
    span = sum(probe.gpu[(ph, "_span")] for ph in PHASES)

    by_phase = {}
    for ph in PHASES:
        w, ws = probe.waves[ph], probe.waves_s[ph]
        if not w:
            continue
        by_phase[ph] = {
            "waves": w, "sampled_waves": ws, "leaves": probe.leaves[ph],
            "mean_k": probe.leaves[ph] / w,
            "cpu_us_per_wave": {seg: probe.cpu[(ph, seg)] * 1e6 / w
                                for seg in SEG_ORDER
                                if probe.cpu[(ph, seg)]},
            "gpu_us_per_wave": {seg: probe.gpu[(ph, seg)] * 1000.0 / ws
                                for seg in GPU_SEGMENTS
                                if ws and probe.gpu[(ph, seg)]},
        }

    # Waves that evaluated nothing. `_run_budget` advances sims by exactly
    # WAVE_SIZE per chunk, but a chunk whose every selected leaf is already
    # expanded, terminal or proven never reaches `wave_planes`. The gap is a
    # measurement of the search, not an artifact: it is how much of the budget
    # goes on descents that never touch the network.
    expected = stats["sims"] / WAVE_SIZE
    return {
        "mode": mode, "sample_every": sample_every, **stats,
        "waves": waves, "clean_waves": clean_w, "sampled_waves": samp_w,
        "leaves": leaves, "mean_k": mean_k, "mean_k_sampled": mean_k_s,
        "own_waves_per_move": waves / moves if moves else 0.0,
        "own_leaves_per_move": leaves / moves if moves else 0.0,
        "evaluating_wave_fraction": waves / expected if expected else 0.0,
        "cpu_us_per_wave": cpu_us,
        "cpu_us_per_leaf": leaf_us,
        # Sampled-wave CPU. In the sync build this is where the drain and the
        # drained (therefore pure) copies live; in the events build it is only
        # useful as the price of the events themselves.
        "sampled_cpu_us_per_wave": samp_us,
        "sampled_cpu_us_per_leaf": samp_leaf,
        "gpu_us_per_wave": gpu_us,
        "gpu_us_per_leaf": gpu_leaf,
        "gpu_span_us_per_wave": span * 1000.0 / samp_w if samp_w else 0.0,
        "gpu_span_us_per_leaf": (span * 1000.0 / samp_leaves
                                 if samp_leaves else 0.0),
        "k_histogram": {str(k): n for k, n in sorted(probe.ksizes.items())},
        "by_phase": by_phase,
    }


GPU_BUSY_CEILING = 0.85


def gpu_credible(res):
    """False when the device column claims more of the wave than its structure
    allows.

    The wave is strictly serial -- select, copy up, forward, copy down, build
    children, repeat -- so the GPU is necessarily idle during every host
    segment. A device total approaching the whole wave is therefore not a busy
    GPU, it is deferred event timestamps, and it is the failure this study hit
    on its first run. Kept as a permanent gate rather than a thing to remember.
    """
    busy = sum(res["gpu_us_per_wave"].values())
    wave = sum(res["cpu_us_per_wave"].values())
    return not wave or busy / wave <= GPU_BUSY_CEILING


def report_perturb(arms):
    """What the instrumentation costs -- measured PER LEAF, not per simulation.

    SIMULATIONS PER MOVE IS NOT A PERTURBATION METRIC, and reading it as one
    would have reported a 25% instrument cost that does not exist. Under a wall
    clock a simulation is not a fixed unit of work: a descent that lands in an
    already-expanded or proven subtree costs no network evaluation at all and
    is nearly free, and the share of such descents is a property of the
    POSITIONS, not of the build. Measured here, that share ran from 35% to 57%
    across four arms that started from identical openings and then diverged --
    which by itself moves sims/move by a quarter.

    Wall time per network evaluation has no such problem: a leaf is a leaf. It
    came out flat, so the instrument is cheaper than the noise floor of the
    comparison.
    """
    print("")
    print("  %-10s %7s %8s %8s %9s %9s"
          % ("build", "moves", "sims/mv", "leaf/mv", "us/leaf", "free"))
    print("  " + "-" * 58)
    base = None
    for k in ("untouched", "off", "events", "sync"):
        v = arms.get(k)
        if not v:
            continue
        us = (1000.0 * v["search_ms_per_move"] / v["nn_per_move"]
              if v["nn_per_move"] else 0.0)
        if base is None:
            base = us
        print("  %-10s %7d %8.0f %8.0f %9.1f %8.1f%%"
              % (k, v["moves"], v["sims_per_move"], v["nn_per_move"], us,
                 100.0 * (1 - v["nn_per_move"] / v["sims_per_move"])))
    spread = [1000.0 * v["search_ms_per_move"] / v["nn_per_move"]
              for v in arms.values() if v.get("nn_per_move")]
    if spread and base:
        print("")
        print("  us/leaf spans %.1f to %.1f, a %.1f%% range -- that is the"
              % (min(spread), max(spread),
                 100.0 * (max(spread) - min(spread)) / min(spread)))
        print("  noise floor of this ladder, and the instrument is inside it.")
        print("  `off` is the replicas with CPU timing only, so the events row")
        print("  prices the events alone.")
        print("  DO NOT read the sims/move column as a perturbation. See the")
        print("  `free` column: it is the share of chunks that evaluated")
        print("  nothing, it swings by 20 points between arms, and it moves")
        print("  sims/move on its own.")


def report(res, ref_leaves):
    """`ref_leaves` is the UNTOUCHED build's network evaluations per move.

    Per-leaf is the primary column. Per-move is per-leaf times that reference,
    and it is presentational: leaves per move is a property of the GAME as much
    as of the engine (a proven subtree costs a descent and no evaluation at
    all), so it swings between openings in a way a per-leaf cost does not.
    """
    print("")
    print("  %d moves, %d waves (%d sampled), %d leaves, mean batch %.2f"
          % (res["moves"], res["waves"], res["sampled_waves"], res["leaves"],
             res["mean_k"]))
    print("  %.0f sims/move and %.0f leaves/move in THIS build; the per-move"
          % (res["sims_per_move"], res["own_leaves_per_move"]))
    print("  column is scaled by the untouched build's %.0f leaves/move."
          % ref_leaves)
    print("  %.0f%% of waves reached the network; the rest selected only into"
          % (100.0 * res["evaluating_wave_fraction"]))
    print("  nodes already expanded, terminal or proven.")
    print("")
    print("  %-28s %6s %9s %9s %9s %9s"
          % ("segment", "kind", "cpu us/lf", "gpu us/lf", "cpu ms/mv",
             "gpu ms/mv"))
    print("  " + "-" * 76)
    cpu_tot = gpu_tot = 0.0
    for seg in SEG_ORDER:
        cus = res["cpu_us_per_leaf"].get(seg, 0.0)
        gus = res["gpu_us_per_leaf"].get(seg, 0.0)
        if cus == 0.0 and gus == 0.0:
            continue
        cpu_tot += cus
        gpu_tot += gus
        print("  %-28s %6s %9.2f %9s %9.1f %9s"
              % (seg, SEG_KIND[seg], cus, "%.2f" % gus if gus else "-",
                 cus * ref_leaves / 1000.0,
                 "%.1f" % (gus * ref_leaves / 1000.0) if gus else "-"))
    print("  " + "-" * 76)
    print("  %-28s %6s %9.2f %9s %9.1f %9s"
          % ("TOTAL in the wave", "", cpu_tot,
             "%.2f" % gpu_tot if gpu_tot else "-",
             cpu_tot * ref_leaves / 1000.0,
             "%.1f" % (gpu_tot * ref_leaves / 1000.0) if gpu_tot else "-"))
    if res["gpu_span_us_per_leaf"]:
        span = res["gpu_span_us_per_leaf"]
        print("  %-28s %6s %9s %9.2f %9s %9.1f"
              % ("gpu span, first to last", "", "", span, "",
                 span * ref_leaves / 1000.0))
        print("  %-28s %6s %9s %9.2f %9s %9.1f"
              % ("gpu idle inside the wave", "", "", span - gpu_tot, "",
                 (span - gpu_tot) * ref_leaves / 1000.0))
        if not gpu_credible(res):
            print("")
            print("  [X] GPU BUSY IS %.0f%% OF THE WAVE. Not believable: the"
                  % (100.0 * gpu_tot / cpu_tot if cpu_tot else 0.0))
            print("      wave is strictly serial, so the device is idle for"
                  " every")
            print("      host segment above. Event timestamps are deferred --"
                  " see")
            print("      Slot.rec(). DO NOT QUOTE THE GPU COLUMN.")

    drained = res.get("sampled_cpu_us_per_leaf", {})
    if "first host synchronization" in drained:
        print("")
        print("  the same interval, with the stream drained first (sampled"
              " waves only)")
        print("    %-28s %9s %9s" % ("", "us/leaf", "ms/move"))
        for seg in ("first host synchronization", "D2H: probs", "D2H: values"):
            us = drained.get(seg, 0.0)
            print("    %-28s %9.2f %9.1f"
                  % (seg, us, us * ref_leaves / 1000.0))
        print("    %-28s %9d of %d waves"
              % ("measured on", res["sampled_waves"], res["waves"]))

    hist = res["k_histogram"]
    if hist:
        tot = sum(hist.values())
        print("")
        print("  batch size (leaves per forward pass)")
        for k, n in sorted(hist.items(), key=lambda kv: int(kv[0])):
            print("    k=%-3s %6d waves  %5.1f%%" % (k, n, 100.0 * n / tot))

    if res["by_phase"]:
        print("")
        print("  cpu us per wave by game stage")
        phs = [p for p in PHASES if p in res["by_phase"]]
        print("  %-28s %s" % ("segment", " ".join("%9s" % p for p in phs)))
        print("  " + "-" * (28 + 10 * len(phs)))
        for seg in SEG_ORDER:
            cells, any_v = [], False
            for p in phs:
                v = res["by_phase"][p]["cpu_us_per_wave"].get(seg, 0.0)
                any_v = any_v or v > 0
                cells.append("%9.1f" % v)
            if any_v:
                print("  %-28s %s" % (seg, " ".join(cells)))
        print("  %-28s %s" % ("mean batch k", " ".join(
            "%9.2f" % res["by_phase"][p]["mean_k"] for p in phs)))
        print("  %-28s %s" % ("waves", " ".join(
            "%9d" % res["by_phase"][p]["waves"] for p in phs)))


def decide(ev, sy, ref_leaves):
    """The owner's decision tree, evaluated against the two builds.

    Printed as a verdict rather than left to the reader: the whole point of the
    study is that the four costs imply four different next pieces of work.
    """
    def ms(d, keys):
        return sum(d.get(k, 0.0) for k in keys) * ref_leaves / 1000.0

    gpu = ev["gpu_us_per_leaf"]
    cpu = ev["cpu_us_per_leaf"]
    h2d = ms(gpu, ("H2D: planes", "H2D: mask"))
    d2h = ms(gpu, ("D2H: probs", "D2H: values"))
    compute = ms(gpu, ("network forward", "device: mask + softmax"))
    gpu_busy = h2d + d2h + compute
    span = ev["gpu_span_us_per_leaf"] * ref_leaves / 1000.0
    wave_cpu = ms(cpu, SEG_ORDER)
    host = ms(cpu, [g for g in SEG_ORDER if SEG_KIND[g] == "host"])
    sd = sy.get("sampled_cpu_us_per_leaf", {})
    drain = ms(sd, ("first host synchronization",))
    pure_d2h = ms(sd, ("D2H: probs", "D2H: values"))
    blocked = ms(cpu, ("D2H: probs", "D2H: values"))

    print("")
    print("  %-42s %10s" % ("quantity", "ms/move"))
    print("  " + "-" * 54)
    rows = [
        ("GPU compute (forward + softmax)", compute),
        ("GPU transfer H2D", h2d),
        ("GPU transfer D2H", d2h),
        ("GPU busy, total", gpu_busy),
        ("GPU span (first record to last)", span),
        ("GPU idle inside the wave", span - gpu_busy),
        ("CPU in the wave, total", wave_cpu),
        ("CPU host work (no device involved)", host),
        ("CPU inside the two .cpu() calls", blocked),
        ("  of which waiting [sync build]", drain),
        ("  of which copying [sync build]", pure_d2h),
    ]
    for label, v in rows:
        print("  %-42s %10.1f" % (label, v))

    # TWO INSTRUMENTS, ONE QUANTITY. The events build measures the two `.cpu()`
    # calls as a single blocked interval; the sync build splits the same
    # interval into a drain and two drained copies. Their totals are the same
    # physical time by two different routes, so a large disagreement means one
    # of them is wrong -- and there is no way to tell which from inside either.
    print("")
    print("  %-42s %10s" % ("reconciliation", "ms/move"))
    print("  " + "-" * 54)
    print("  %-42s %10.1f" % ("events: inside the two .cpu() calls", blocked))
    print("  %-42s %10.1f" % ("sync:   drain + drained copies",
                              drain + pure_d2h))
    if blocked > 0:
        ratio = (drain + pure_d2h) / blocked
        print("  %-42s %10.2f" % ("ratio", ratio))
        if not 0.7 <= ratio <= 1.4:
            print("  [!] the two builds disagree by more than 40%. One of them")
            print("      is measuring something else; do not quote either.")

    print("")
    frac = compute / gpu_busy if gpu_busy else 0.0
    print("  GPU compute is %.0f%% of GPU busy time." % (100.0 * frac))
    if drain + pure_d2h > 0:
        print("  The two .cpu() calls are %.0f%% waiting, %.0f%% copying."
              % (100.0 * drain / (drain + pure_d2h),
                 100.0 * pure_d2h / (drain + pure_d2h)))
    print("")
    if frac >= 0.60:
        print("  VERDICT: compute-bound. Do NOT restructure transfers first --")
        print("  the copies are not where the time is. Fewer forwards, or a")
        print("  cheaper one, is the only real lever on this bucket.")
    elif gpu_busy and (h2d + d2h) / gpu_busy >= 0.40:
        print("  VERDICT: transfer-bound. Cut round trips per wave.")
    elif span and (span - gpu_busy) > 0.25 * span:
        print("  VERDICT: launch-fragmented. The GPU is idle inside the wave;")
        print("  look at serial host/device dependencies, not copy volume.")
    else:
        print("  VERDICT: mixed. Read the table; no single cost dominates.")
    print("")
    print("  NOTE ON ATTRIBUTION: `.cpu()` is where earlier asynchronous work")
    print("  becomes visible. Its wall time is NOT transfer time and is not")
    print("  reported as such anywhere above.")
    return {label: v for label, v in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="all",
                    choices=("events", "sync", "perturb", "all"))
    ap.add_argument("--engine", default="pocket_r35")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--perturb-games", type=int, default=3)
    ap.add_argument("--sample-every", type=int, default=100,
                    help="record CUDA events (or drain) on 1 wave in N; the "
                         "rest stay clean and supply the CPU column. At 1-in-20"
                         " the events build still lost 25%% of its simulations"
                         " -- a WDDM flush is not cheap")
    ap.add_argument("--seed", type=int, default=EXPAND_SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--tag", default="expand")
    ap.add_argument("--replay", metavar="PATH",
                    help="re-report a saved study instead of running one. The "
                         "raw per-leaf costs are in the JSON, so presentation "
                         "must never require another 40 minutes of GPU")
    args = ap.parse_args()

    if args.replay:
        with open(args.replay) as fh:
            payload = json.load(fh)
        ref = payload.get("ref_waves_per_move")   # legacy studies
        if payload.get("perturb"):
            print("=== what the instrumentation costs ===")
            report_perturb(payload["perturb"])
            ref = payload["perturb"]["untouched"]["nn_per_move"]
        for key, title in (("events", "CUDA events, no blocking sync"),
                           ("sync", "explicit drain before the first pull")):
            if key in payload:
                print("")
                print("=== %s ===" % title)
                report(payload[key], ref or
                       payload[key]["own_leaves_per_move"])
        if "events" in payload and "sync" in payload:
            print("")
            print("=== where the wave actually goes ===")
            decide(payload["events"], payload["sync"], ref)
        return

    assert_frozen_sources()
    if args.device != "cuda":
        raise SystemExit("[X] this study is about CUDA traffic; --device cuda")

    payload = {"engine": args.engine, "seed": args.seed,
               "sample_every": args.sample_every,
               "environment": engine_registry.environment(),
               "environment_drift": engine_registry.env_drift(),
               "git_head": engine_registry.git_head()}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "%s.json" % args.tag)

    def save():
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

    ev = sy = None
    # THE UNTOUCHED BUILD RUNS FIRST, because its wave rate is the denominator
    # every per-move figure below is expressed in. Under a wall clock an
    # instrumented run simply completes fewer waves, so using each build's own
    # rate would scale its costs down by its own perturbation and the report
    # would conceal the very thing this section measures.
    ref = None
    if args.mode in ("perturb", "all"):
        print("=== 1. what the instrumentation costs ===")
        arms = {}
        arms["untouched"] = _play(args.engine, args.perturb_games,
                                  args.device, args.seed)
        for m in ("off", "events", "sync"):
            probe = ExpandProbe(m, sample_every=args.sample_every)
            arms[m] = _play(args.engine, args.perturb_games, args.device,
                            args.seed, probe)
        report_perturb(arms)
        ref = arms["untouched"]["nn_per_move"]
        print("  reference: %.0f network evaluations (leaves) per move" % ref)
        payload["perturb"] = arms
        payload["ref_waves_per_move"] = ref
        save()

    if args.mode in ("events", "all"):
        print("")
        print("=== 2. CUDA events, no blocking synchronization ===")
        res = run_instrumented(args.engine, args.games, args.device,
                               args.seed, "events", args.sample_every)
        payload["events"] = res
        save()
        report(res, ref if ref else res["own_leaves_per_move"])
        ev = res
        save()

    if args.mode in ("sync", "all"):
        print("")
        print("=== 3. explicit drain before the first pull (PERTURBING) ===")
        res = run_instrumented(args.engine, args.games, args.device,
                               args.seed, "sync", args.sample_every)
        payload["sync"] = res
        save()
        report(res, ref if ref else res["own_leaves_per_move"])
        sy = res
        save()

    if args.mode == "all" and ev and sy:
        print("")
        print("=== 4. where the wave actually goes ===")
        payload["decision"] = decide(ev, sy, ref)

    save()
    print("")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
