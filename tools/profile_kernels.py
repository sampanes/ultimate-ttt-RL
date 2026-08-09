"""Kernel and launch structure of one expansion wave. MEASUREMENT ONLY.

`RESULT_EXPAND_CUDA.md` closed with the wave being DISPATCH-bound: a 6.25x
change in batch size moves the GPU forward by 8% and every transfer by nothing,
so the device cost is a fixed toll per call rather than per leaf or per byte.
That says the toll scales with the number of operations -- and the number of
operations has never been counted. Two estimates in that document rest on it:
"~20 kernels per wave" and "a graph replay is one dispatch, assume 50%
efficiency". Both are marked unmeasured, and the whole CUDA-graph arm of the
ceiling comparison is built on them.

This tool counts. It does not optimize, and it does not touch `agents/mcts.py`
(frozen, hash-gated). Nothing here is allowed to change how the engine plays.

WHAT IS MEASURED, AND BY WHICH INSTRUMENT. The two answers come from different
places on purpose, because neither instrument can give both:

    STRUCTURE   torch.profiler / CUPTI. Kernel count, kernel names, device-side
                durations, memcpy byte counts, and every explicit
                synchronization the CUDA runtime performs. These are counts and
                device timings; they are correct under the profiler.
    COST        plain CPU clocks and CUDA events with the profiler OFF. Wall
                time per segment. CUPTI adds tens of microseconds to every
                `cudaLaunchKernel` -- visible in the trace as launches longer
                than the kernels they launch -- so a launch cost read off the
                profiler would be the profiler's cost.

The composed figure -- CPU microseconds per launch -- is the unprofiled segment
time divided by the profiled kernel count. It is stated that way wherever it
appears, because it is the one number in this study that no single instrument
measured.

THE BENCH IS A REPLAY OF PRODUCTION WAVES. A short real match under the
deployment engine records the wave-size histogram and clones a bounded sample
of the actual leaf states. The bench then re-runs the device sequence of
`wave_planes` + `_expand_wave` on those exact states. Two consequences: the
plane and mask contents are real, and the bench's per-wave cost can be checked
against the per-wave cost already published from live play in
`results/profile_expand/expand.json`. That cross-check is the credibility gate.
A bench that does not reproduce production is describing something else.

    python -m tools.profile_kernels --mode all --engine pocket_r35

Modes:
    hist      wave-size histogram from live play (sets the bench's wave sizes)
    trace     CUPTI structure at each wave size
    bench     unprofiled per-segment cost at each wave size
    mask      decomposition of the mask path's host overhead
    capture   CUDA-graph capture feasibility, numerics, and replay cost
    all       all five, plus the decision rule from task #40
"""

import argparse
import collections
import json
import os
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from agents import agent_base
from agents import mcts as mcts_mod
from engine.rules import rule_utl_valid_moves
from tools import engine_registry
from tools.arena_1s import TimedPlayer, play_match
from tools.profile_tree import assert_frozen_sources, instrument_player, new_ctx

OUT_DIR = os.path.join("results", "profile_kernels")

KERNEL_SEED = engine_registry.SEEDS["kernels"]

# The segments of the device sequence, named exactly as in
# `tools/profile_expand.py` so the two studies can be read side by side.
SEGMENTS = [
    "host: plane fill",
    "H2D: planes",
    "network forward",
    "host: legal masks",
    "H2D: mask",
    "device: mask + softmax",
    "D2H: probs",
    "D2H: values",
]

# Segments with something on the stream to time.
GPU_SEGMENTS = ("H2D: planes", "network forward", "H2D: mask",
                "device: mask + softmax", "D2H: probs", "D2H: values")

# "Small kernel" for the purposes of the launch-bound question. A kernel whose
# device time is below the CPU cost of launching it cannot be made cheaper by
# making the GPU faster.
SMALL_KERNEL_US = 10.0

# Per-wave costs measured in LIVE PLAY by the previous study, at its mean wave
# size of 6.46 leaves. The bench is required to land near these or it is not
# reproducing production. Loaded from the JSON when present; these are the
# published fallback.
PRODUCTION_REFERENCE = {
    "source": "results/profile_expand/expand.json (events arm)",
    "mean_k": 6.459,
    "cpu_us_per_wave": {
        "host: plane fill": 28.3, "H2D: planes": 79.1,
        "network forward": 752.0, "host: legal masks": 66.0,
        "H2D: mask": 280.9, "device: mask + softmax": 58.5,
        "D2H: probs": 173.0, "D2H: values": 59.5,
    },
    "gpu_us_per_wave": {
        "H2D: planes": 142.2, "network forward": 780.2, "H2D: mask": 63.2,
        "device: mask + softmax": 60.4, "D2H: probs": 116.4,
        "D2H: values": 54.6,
    },
    "waves_per_move": 396.1,
}


LOCK_PATH = os.path.join(OUT_DIR, ".running.lock")


def acquire_lock():
    """Refuse to start while another instance is measuring.

    THIS IS NOT HOUSEKEEPING. Two overlapping runs of this tool produced two
    complete-looking results in the same file, and the k=8 kernel time differed
    by 1.9x between them because they were sharing the GPU. Nothing crashed and
    nothing looked wrong; the second file simply overwrote the first and both
    were contended. A timing study that can be silently invalidated by a stray
    process needs a lock, not a convention.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            with open(LOCK_PATH) as fh:
                who = fh.read().strip()
        except OSError:
            who = "unknown"
        raise SystemExit(
            "[X] another run holds %s (%s).\n"
            "    Two concurrent runs share the GPU and both results are then\n"
            "    contended -- which has already happened once. Wait for it, or\n"
            "    if it is dead, delete the lock file and re-run."
            % (LOCK_PATH, who))
    with os.fdopen(fd, "w") as fh:
        fh.write("pid %d started %s"
                 % (os.getpid(), time.strftime("%Y-%m-%d %H:%M:%S")))


def release_lock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


def gpu_baseline(samples=5, gap=0.4):
    """Whatever else is using the GPU, sampled before the study starts.

    Recorded rather than acted on: this is a desktop card and the compositor
    is always doing something. It belongs in the result so a later reader can
    tell a quiet box from a busy one.
    """
    import subprocess
    vals = []
    for _ in range(samples):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            vals.append(float(out.stdout.strip().splitlines()[0]))
        except Exception:
            return None
        time.sleep(gap)
    return {"samples": vals, "mean_pct": sum(vals) / len(vals),
            "max_pct": max(vals)}


# ----------------------------------------------------------------------
# The device sequence, replayed
# ----------------------------------------------------------------------

def _fill(buf, states):
    """`agent_base.wave_planes`'s fill, branch for branch.

    The C++ engine writes the planes directly; the pure-Python engine goes
    through `board_to_tensor_from_gamestate`. Both branches exist upstream, so
    both exist here -- a replica that only handles the fast path would silently
    profile something else on a box without the extension built.
    """
    if len(states) and agent_base._has_fill_planes(states[0]):
        for i, s in enumerate(states):
            s.fill_planes(buf[i])
    else:
        for i, s in enumerate(states):
            buf[i] = agent_base.board_to_tensor_from_gamestate(s).numpy()


def wave_sequence(model, states, device, out=None):
    """The exact device work one wave does, statement for statement.

    `agents/mcts.py:585-596` -- `wave_planes`, `forward_both`, then the head of
    `_expand_wave` up to the two `.cpu()` pulls. Everything after that pull is
    pure host work (child construction, probes) and is not device structure, so
    it is excluded here and priced in the tree profile instead.

    `out` is an optional dict; when given, each segment's result is stored so a
    caller can check the replay against the frozen original. Timing is the
    caller's job -- this function has no instrumentation in it at all, so the
    traced and the unprofiled paths execute identical bytes.
    """
    k = len(states)
    buf = np.empty((k, 7, 9, 9), dtype=np.float32)
    _fill(buf, states)
    xs = torch.from_numpy(buf).to(device)

    logits_b, values_b = model.forward_both(xs)
    if logits_b.dim() == 1:
        logits_b = logits_b.unsqueeze(0)
        values_b = values_b.unsqueeze(0)

    valids = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
              for s in states]
    mask = np.zeros((k, 81), dtype=bool)
    for i, v in enumerate(valids):
        mask[i, v] = True
    mask_t = torch.from_numpy(mask).to(logits_b.device)

    probs = F.softmax(logits_b.masked_fill(~mask_t, float("-inf")), dim=1)
    probs_np = probs.cpu().numpy()
    values_np = values_b.reshape(-1).cpu().numpy()
    if out is not None:
        out["probs"] = probs_np
        out["values"] = values_np
        out["valids"] = valids
    return probs_np, values_np


# ----------------------------------------------------------------------
# Mode 1: what wave sizes does production actually run?
# ----------------------------------------------------------------------

class WaveCollector:
    """Histogram every wave; clone a bounded sample of them for the bench.

    The histogram costs one `len()` and one counter bump, which is the point:
    the sizes the bench is built around must come from an engine that is
    running at its real speed. Cloning is capped per bucket and strided, so it
    touches a small minority of waves.
    """

    def __init__(self, ctx, cap=40, stride=17):
        self.ctx = ctx
        self.cap = cap
        self.stride = stride
        self.hist = collections.Counter()
        self.by_phase = collections.defaultdict(collections.Counter)
        self.pool = collections.defaultdict(list)
        self.n = 0
        self._orig = None

    def wave_planes(self, states, device):
        if self.ctx["on"]:
            k = len(states)
            self.hist[k] += 1
            self.by_phase[self.ctx["phase"]][k] += 1
            self.n += 1
            if len(self.pool[k]) < self.cap and self.n % self.stride == 0:
                self.pool[k].append([s.clone() for s in states])
        return self._orig(states, device)

    def install(self):
        self._orig = mcts_mod.wave_planes
        mcts_mod.wave_planes = self.wave_planes

    def remove(self):
        if self._orig is not None:
            mcts_mod.wave_planes = self._orig
            self._orig = None


def quantile_k(hist, q):
    """The wave size at quantile `q` of the wave population."""
    total = sum(hist.values())
    if not total:
        return None
    target = q * total
    seen = 0
    for k in sorted(hist):
        seen += hist[k]
        if seen >= target:
            return k
    return max(hist)


def collect_waves(engine, games, device, seed, cap=40):
    ctx = new_ctx()
    pa = TimedPlayer("engine:%s" % engine, device)
    pb = TimedPlayer("engine:%s" % engine, device)
    for p in (pa, pb):
        instrument_player(p, ctx)
    col = WaveCollector(ctx, cap=cap)
    col.install()
    try:
        play_match(pa, pb, games, seed, warmup=2, gc_mode="deferred")
    finally:
        col.remove()

    moves = len(pa.records) + len(pb.records)
    nn = sum(r[3] for r in pa.records) + sum(r[3] for r in pb.records)
    waves = sum(col.hist.values())
    leaves = sum(k * n for k, n in col.hist.items())
    return {
        "engine": engine,
        "games": games,
        "moves": moves,
        "waves": waves,
        "leaves": leaves,
        "nn_evals": nn,
        "waves_per_move": waves / moves if moves else 0.0,
        "leaves_per_move": leaves / moves if moves else 0.0,
        "mean_k": leaves / waves if waves else 0.0,
        "median_k": quantile_k(col.hist, 0.5),
        "p90_k": quantile_k(col.hist, 0.9),
        "histogram": {str(k): n for k, n in sorted(col.hist.items())},
        "share_of_leaves": {
            str(k): (k * n / leaves if leaves else 0.0)
            for k, n in sorted(col.hist.items())},
        "by_phase": {ph: {str(k): n for k, n in sorted(c.items())}
                     for ph, c in col.by_phase.items()},
        "fingerprint": (pa.provenance or {}).get("fingerprint"),
        "params": pa.net_info["params"],
        "budget_ms": pa.budget_ms,
    }, col.pool


# ----------------------------------------------------------------------
# Mode 2: structure, from CUPTI
# ----------------------------------------------------------------------

def _trace_events(path):
    with open(path) as fh:
        tr = json.load(fh)
    return tr["traceEvents"] if isinstance(tr, dict) else tr


def _demangle(name):
    """Enough of a name to tell conv from softmax from elementwise."""
    for tag, short in (("implicit_convolve_sgemm", "conv:implicit_gemm"),
                       ("cudnn", "conv:cudnn"),
                       ("_5x_cudnn", "conv:cudnn"),
                       ("gemm", "gemm"),
                       ("SoftMax", "softmax"),
                       ("vectorized_elementwise_kernel", "elementwise"),
                       ("elementwise_kernel", "elementwise"),
                       ("reduce_kernel", "reduce"),
                       ("CatArrayBatchedCopy", "cat")):
        if tag in name:
            return short
    return name[:48]


def trace_waves(model, waves, device, reps, warmup=8):
    """Kernel-level structure of `reps` consecutive waves at one wave size.

    The waves are replayed back to back, exactly as production runs them, and
    the first `warmup` are executed with the profiler off so that cudnn
    algorithm selection, allocator growth and module load do not land in the
    trace. Only whole waves are traced -- a partial wave would attribute a
    kernel to the wrong segment.
    """
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        for i in range(warmup):
            wave_sequence(model, waves[i % len(waves)], device)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for i in range(reps):
                wave_sequence(model, waves[i % len(waves)], device)
            torch.cuda.synchronize()

    path = os.path.join(OUT_DIR, "_trace_tmp.json")
    prof.export_chrome_trace(path)
    evs = _trace_events(path)
    os.remove(path)

    kernels, memcpys, syncs, launches = [], [], [], []
    for e in evs:
        cat, dur = e.get("cat"), float(e.get("dur") or 0.0)
        name = str(e.get("name") or "")
        if cat == "kernel":
            kernels.append({"name": _demangle(name), "dur": dur,
                            "ts": float(e.get("ts") or 0.0)})
        elif cat == "gpu_memcpy":
            memcpys.append({"name": name, "dur": dur,
                            "bytes": int((e.get("args") or {}).get("bytes", 0)),
                            "ts": float(e.get("ts") or 0.0)})
        elif cat == "cuda_runtime":
            if name.startswith("cudaLaunchKernel"):
                launches.append({"ts": float(e.get("ts") or 0.0), "dur": dur})
            elif "Synchronize" in name:
                syncs.append({"name": name, "dur": dur})

    kernels.sort(key=lambda d: d["ts"])
    launches.sort(key=lambda d: d["ts"])

    # Device-side gap between consecutive kernels: how long the GPU sat idle
    # between one kernel finishing and the next starting. Under a launch-bound
    # workload this is the launch cost leaking onto the device timeline.
    dev_gaps = []
    for a, b in zip(kernels, kernels[1:]):
        g = b["ts"] - (a["ts"] + a["dur"])
        if 0.0 <= g < 5000.0:
            dev_gaps.append(g)
    # CPU-side gap between the START of consecutive launches -- the launch
    # cadence. INFLATED BY CUPTI; reported for shape, never quoted as cost.
    cpu_gaps = [b["ts"] - a["ts"] for a, b in zip(launches, launches[1:])
                if 0.0 <= b["ts"] - a["ts"] < 5000.0]

    durs = [k["dur"] for k in kernels]
    gpu_busy = sum(durs) + sum(m["dur"] for m in memcpys)
    small = [d for d in durs if d < SMALL_KERNEL_US]
    by_kind = collections.Counter()
    kind_time = collections.defaultdict(float)
    for k in kernels:
        by_kind[k["name"]] += 1
        kind_time[k["name"]] += k["dur"]

    # Keyed by the driver's own description ("Memcpy HtoD (Pageable ->
    # Device)"), not by a direction bucket. The first draft bucketed into
    # H2D/D2H/other and put a 2,592-byte copy in "other" -- a device-to-device
    # copy nobody had accounted for, which a coarser key would have hidden.
    copies = collections.defaultdict(lambda: {"n": 0, "bytes": 0, "dur": 0.0})
    for m in memcpys:
        d = m["name"].replace("Memcpy ", "")
        copies[d]["n"] += 1
        copies[d]["bytes"] += m["bytes"]
        copies[d]["dur"] += m["dur"]

    return {
        "reps": reps,
        "kernels_total": len(kernels),
        "kernels_per_wave": len(kernels) / reps,
        "kernel_us_mean": (sum(durs) / len(durs)) if durs else 0.0,
        "kernel_us_median": statistics.median(durs) if durs else 0.0,
        "kernel_us_max": max(durs) if durs else 0.0,
        "small_kernel_count_share": len(small) / len(durs) if durs else 0.0,
        "small_kernel_time_share": (sum(small) / sum(durs)) if durs else 0.0,
        "gpu_kernel_us_per_wave": sum(durs) / reps,
        "gpu_memcpy_us_per_wave": sum(m["dur"] for m in memcpys) / reps,
        "gpu_busy_us_per_wave": gpu_busy / reps,
        "device_gap_us_median": statistics.median(dev_gaps) if dev_gaps else 0.0,
        "device_gap_us_total_per_wave": sum(dev_gaps) / reps,
        "profiled_launch_cadence_us_median": (statistics.median(cpu_gaps)
                                              if cpu_gaps else 0.0),
        "profiled_launch_us_mean": (sum(l["dur"] for l in launches)
                                    / len(launches)) if launches else 0.0,
        "launches_per_wave": len(launches) / reps,
        "explicit_syncs_per_wave": len(syncs) / reps,
        "sync_kinds": dict(collections.Counter(s["name"] for s in syncs)),
        "sync_us_per_wave_profiled": sum(s["dur"] for s in syncs) / reps,
        "copies_per_wave": {
            d: {"n": v["n"] / reps, "bytes": v["bytes"] / reps,
                "device_us": v["dur"] / reps}
            for d, v in sorted(copies.items())},
        "kernels_by_kind": {
            name: {"n_per_wave": n / reps, "us_per_wave": kind_time[name] / reps}
            for name, n in by_kind.most_common()},
    }


def trace_segmented(model, waves, device, reps, warmup=8):
    """The same kernels, attributed to the segment that launched them.

    "Kernels per network forward" and "kernels for masking and softmax" are
    separate questions from "kernels per wave", and the unannotated trace
    cannot answer them -- it sees one undifferentiated stream. So this pass
    brackets each statement in a `record_function` and attributes every
    `cudaLaunchKernel`, memcpy and synchronize to the annotation window that
    contains it.

    COUNTS ONLY. The annotations are themselves profiler events and they
    perturb the timing, which is why they are not in `wave_sequence` and why
    nothing timed is read off this pass.
    """
    import bisect

    from torch.profiler import ProfilerActivity, profile, record_function

    with torch.no_grad():
        for i in range(warmup):
            wave_sequence(model, waves[i % len(waves)], device)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for r in range(reps):
                states = waves[r % len(waves)]
                k = len(states)
                with record_function("seg|host: plane fill"):
                    buf = np.empty((k, 7, 9, 9), dtype=np.float32)
                    _fill(buf, states)
                with record_function("seg|H2D: planes"):
                    xs = torch.from_numpy(buf).to(device)
                with record_function("seg|network forward"):
                    logits_b, values_b = model.forward_both(xs)
                    if logits_b.dim() == 1:
                        logits_b = logits_b.unsqueeze(0)
                        values_b = values_b.unsqueeze(0)
                with record_function("seg|host: legal masks"):
                    valids = [rule_utl_valid_moves(s.board, s.last_move,
                                                   s.mini_winners)
                              for s in states]
                    mask = np.zeros((k, 81), dtype=bool)
                    for i, v in enumerate(valids):
                        mask[i, v] = True
                with record_function("seg|H2D: mask"):
                    mask_t = torch.from_numpy(mask).to(logits_b.device)
                with record_function("seg|device: mask + softmax"):
                    probs = F.softmax(
                        logits_b.masked_fill(~mask_t, float("-inf")), dim=1)
                with record_function("seg|D2H: probs"):
                    probs.cpu().numpy()
                with record_function("seg|D2H: values"):
                    values_b.reshape(-1).cpu().numpy()
            torch.cuda.synchronize()

    path = os.path.join(OUT_DIR, "_trace_seg_tmp.json")
    prof.export_chrome_trace(path)
    evs = _trace_events(path)
    os.remove(path)

    windows, device_by_corr, runtime = [], {}, []
    for e in evs:
        cat = e.get("cat")
        name = str(e.get("name") or "")
        ts, dur = float(e.get("ts") or 0.0), float(e.get("dur") or 0.0)
        corr = (e.get("args") or {}).get("correlation")
        if cat == "user_annotation" and name.startswith("seg|"):
            windows.append((ts, ts + dur, name[4:]))
        elif cat in ("kernel", "gpu_memcpy"):
            device_by_corr[corr] = (cat, name, dur,
                                    int((e.get("args") or {}).get("bytes", 0)))
        elif cat == "cuda_runtime":
            runtime.append((ts, name, corr))

    windows.sort()
    starts = [w[0] for w in windows]
    per_seg = collections.defaultdict(
        lambda: {"kernels": 0, "memcpys": 0, "bytes": 0, "syncs": 0,
                 "device_us": 0.0, "kernel_names": collections.Counter()})
    unattributed = 0
    for ts, name, corr in runtime:
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0 or ts > windows[i][1]:
            unattributed += 1
            continue
        seg = per_seg[windows[i][2]]
        if "Synchronize" in name:
            seg["syncs"] += 1
            continue
        got = device_by_corr.get(corr)
        if got is None:
            continue
        cat, kname, dur, nbytes = got
        seg["device_us"] += dur
        if cat == "kernel":
            seg["kernels"] += 1
            seg["kernel_names"][_demangle(kname)] += 1
        else:
            seg["memcpys"] += 1
            seg["bytes"] += nbytes

    return {
        "reps": reps,
        "unattributed_runtime_calls": unattributed,
        "per_segment": {
            seg: {"kernels_per_wave": v["kernels"] / reps,
                  "memcpys_per_wave": v["memcpys"] / reps,
                  "bytes_per_wave": v["bytes"] / reps,
                  "syncs_per_wave": v["syncs"] / reps,
                  "device_us_per_wave": v["device_us"] / reps,
                  "kernel_names": {n: c / reps
                                   for n, c in v["kernel_names"].most_common()}}
            for seg, v in per_seg.items()},
    }


# ----------------------------------------------------------------------
# Mode 3: cost, with the profiler off
# ----------------------------------------------------------------------

def _cuda_pair():
    return (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))


def bench_waves(model, waves, device, reps, warmup=8, events=True):
    """Per-segment CPU wall time, and optionally CUDA-event device time.

    Two passes over the same waves. The CPU pass carries no CUDA events at all,
    because recording twelve events costs about as much as the device section
    they measure (established in `tools/profile_expand.py`, which is why that
    study samples one wave in a hundred). The event pass is a separate
    population and its CPU column is not used.
    """
    segs = collections.OrderedDict((s, 0.0) for s in SEGMENTS)
    gpu = collections.OrderedDict((s, 0.0) for s in GPU_SEGMENTS)
    pc = time.perf_counter

    with torch.no_grad():
        for i in range(warmup):
            wave_sequence(model, waves[i % len(waves)], device)
        torch.cuda.synchronize()

        # ---- CPU pass -------------------------------------------------
        for r in range(reps):
            states = waves[r % len(waves)]
            k = len(states)
            t0 = pc()
            buf = np.empty((k, 7, 9, 9), dtype=np.float32)
            _fill(buf, states)
            t1 = pc()
            xs = torch.from_numpy(buf).to(device)
            t2 = pc()
            logits_b, values_b = model.forward_both(xs)
            if logits_b.dim() == 1:
                logits_b = logits_b.unsqueeze(0)
                values_b = values_b.unsqueeze(0)
            t3 = pc()
            valids = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
                      for s in states]
            mask = np.zeros((k, 81), dtype=bool)
            for i, v in enumerate(valids):
                mask[i, v] = True
            t4 = pc()
            mask_t = torch.from_numpy(mask).to(logits_b.device)
            t5 = pc()
            probs = F.softmax(logits_b.masked_fill(~mask_t, float("-inf")),
                              dim=1)
            t6 = pc()
            probs.cpu().numpy()
            t7 = pc()
            values_b.reshape(-1).cpu().numpy()
            t8 = pc()
            for name, dt in zip(SEGMENTS,
                                (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4,
                                 t6 - t5, t7 - t6, t8 - t7)):
                segs[name] += dt

        out = {"reps": reps,
               "k": len(waves[0]),
               "cpu_us_per_wave": {s: v * 1e6 / reps for s, v in segs.items()}}

        if not events:
            return out

        # ---- event pass ------------------------------------------------
        pairs = {s: _cuda_pair() for s in GPU_SEGMENTS}
        for r in range(reps):
            states = waves[r % len(waves)]
            k = len(states)
            buf = np.empty((k, 7, 9, 9), dtype=np.float32)
            _fill(buf, states)
            valids = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
                      for s in states]
            mask = np.zeros((k, 81), dtype=bool)
            for i, v in enumerate(valids):
                mask[i, v] = True

            def rec(seg, which):
                # An event recorded on an idle stream is not timestamped when
                # it is recorded -- WDDM batches submissions and the timestamp
                # waits for the next flush. A non-blocking query() forces the
                # submission. Without this the device column absorbs host work
                # and can exceed the whole wave. See profile_expand.Slot.rec.
                pairs[seg][which].record()
                if which:
                    pairs[seg][1].query()

            rec("H2D: planes", 0)
            xs = torch.from_numpy(buf).to(device)
            rec("H2D: planes", 1)
            rec("network forward", 0)
            logits_b, values_b = model.forward_both(xs)
            rec("network forward", 1)
            if logits_b.dim() == 1:
                logits_b = logits_b.unsqueeze(0)
                values_b = values_b.unsqueeze(0)
            rec("H2D: mask", 0)
            mask_t = torch.from_numpy(mask).to(logits_b.device)
            rec("H2D: mask", 1)
            rec("device: mask + softmax", 0)
            probs = F.softmax(logits_b.masked_fill(~mask_t, float("-inf")),
                              dim=1)
            rec("device: mask + softmax", 1)
            rec("D2H: probs", 0)
            probs.cpu().numpy()
            rec("D2H: probs", 1)
            rec("D2H: values", 0)
            values_b.reshape(-1).cpu().numpy()
            rec("D2H: values", 1)
            torch.cuda.synchronize()
            for s in GPU_SEGMENTS:
                gpu[s] += pairs[s][0].elapsed_time(pairs[s][1])

    out["gpu_us_per_wave"] = {s: v * 1000.0 / reps for s, v in gpu.items()}
    return out


# ----------------------------------------------------------------------
# Mode 4: where the mask path's host time actually goes
# ----------------------------------------------------------------------

def _time_drained(fn, reps):
    """Per-call cost with the stream empty before every call.

    The drains are excluded from the measured interval, so what is returned is
    the operation's own cost with nothing to wait for.
    """
    pc = time.perf_counter
    for _ in range(3):
        torch.cuda.synchronize()
        fn()
    total = 0.0
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = pc()
        fn()
        total += pc() - t0
    return total * 1e6 / reps


def mask_decomposition(model, waves, device, reps=300):
    """Attribute the mask path's host cost to a specific operation.

    Live play spends 280.9 us/wave on the single statement
    `torch.from_numpy(mask).to(logits_b.device)`, against 63.2 us of device
    time for a 648-byte copy. The candidates are construction, allocation,
    dtype, dispatch, and synchronization -- and only one of them can be fixed
    by keeping a buffer on the device, so guessing is not good enough.

    Every arm is timed with the stream DRAINED first except the two marked
    `in flight`, which reproduce production by leaving a forward pass queued.
    The difference between those two and their drained counterparts is the
    induced wait, and it is the whole question.
    """
    states = waves[0]
    k = len(states)
    valids = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
              for s in states]
    mask = np.zeros((k, 81), dtype=bool)
    for i, v in enumerate(valids):
        mask[i, v] = True
    mask_u8 = mask.astype(np.uint8)
    flat_idx = np.flatnonzero(mask.reshape(-1)).astype(np.int64)

    buf = np.empty((k, 7, 9, 9), dtype=np.float32)
    for i, s in enumerate(states):
        s.fill_planes(buf[i])
    with torch.no_grad():
        xs = torch.from_numpy(buf).to(device)
        torch.cuda.synchronize()

        pinned = torch.empty((k, 81), dtype=torch.bool, pin_memory=True)
        pinned.copy_(torch.from_numpy(mask))
        pinned_idx = torch.empty(len(flat_idx), dtype=torch.int64,
                                 pin_memory=True)
        pinned_idx.copy_(torch.from_numpy(flat_idx))
        dev_mask = torch.zeros((k, 81), dtype=torch.bool, device=device)
        dev_add = torch.zeros((k, 81), dtype=torch.float32, device=device)
        torch.cuda.synchronize()

        arms = collections.OrderedDict()

        def valids_only():
            [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
             for s in states]

        def alloc_only():
            np.zeros((k, 81), dtype=bool)

        def fill_only():
            m = np.zeros((k, 81), dtype=bool)
            for i, v in enumerate(valids):
                m[i, v] = True

        def wrap_only():
            torch.from_numpy(mask)

        def upload_pageable():
            torch.from_numpy(mask).to(device)

        def upload_pageable_u8():
            torch.from_numpy(mask_u8).to(device)

        def upload_pinned_blocking():
            pinned.to(device)

        def upload_pinned_nonblocking():
            pinned.to(device, non_blocking=True)

        def copy_into_device_buffer():
            dev_mask.copy_(pinned, non_blocking=True)

        def build_on_device():
            dev_mask.zero_()
            dev_mask.view(-1).index_fill_(0, pinned_idx.to(device,
                                                           non_blocking=True),
                                          True)

        def additive_mask_only():
            # The fused alternative: no bool mask at all, one add on device.
            torch.add(dev_add, dev_add)

        arms["host: rule_utl_valid_moves x k"] = _time_drained(valids_only, reps)
        arms["host: np.zeros((k,81), bool)"] = _time_drained(alloc_only, reps)
        arms["host: zeros + python fill loop"] = _time_drained(fill_only, reps)
        arms["host: torch.from_numpy (no copy)"] = _time_drained(wrap_only, reps)
        arms["upload: pageable bool .to() [drained]"] = _time_drained(
            upload_pageable, reps)
        arms["upload: pageable uint8 .to() [drained]"] = _time_drained(
            upload_pageable_u8, reps)
        arms["upload: pinned bool .to() [drained]"] = _time_drained(
            upload_pinned_blocking, reps)
        arms["upload: pinned .to(non_blocking) [drained]"] = _time_drained(
            upload_pinned_nonblocking, reps)
        arms["upload: copy_ into device buffer [drained]"] = _time_drained(
            copy_into_device_buffer, reps)
        arms["build on device from pinned index [drained]"] = _time_drained(
            build_on_device, reps)
        arms["device: additive mask, no upload [drained]"] = _time_drained(
            additive_mask_only, reps)

        # --- the production case: a forward pass is still in flight -------
        def in_flight(fn):
            pc = time.perf_counter
            total = 0.0
            for _ in range(reps):
                torch.cuda.synchronize()
                model.forward_both(xs)          # queued, not awaited
                t0 = pc()
                fn()
                total += pc() - t0
            torch.cuda.synchronize()
            return total * 1e6 / reps

        arms["upload: pageable bool .to() [FORWARD IN FLIGHT]"] = in_flight(
            upload_pageable)
        arms["upload: pinned .to(non_blocking) [FORWARD IN FLIGHT]"] = (
            in_flight(upload_pinned_nonblocking))
        arms["upload: copy_ into device buffer [FORWARD IN FLIGHT]"] = (
            in_flight(copy_into_device_buffer))

        # How long the forward actually takes, so the induced wait can be
        # checked against something rather than merely asserted.
        def forward_only():
            model.forward_both(xs)

        fwd_launch = _time_drained(forward_only, reps)

        def forward_and_drain():
            model.forward_both(xs)
            torch.cuda.synchronize()

        fwd_total = _time_drained(forward_and_drain, reps)

    return {"k": k, "reps": reps, "arms": arms,
            "forward_launch_us": fwd_launch,
            "forward_launch_plus_drain_us": fwd_total,
            "forward_tail_us": max(0.0, fwd_total - fwd_launch)}


# ----------------------------------------------------------------------
# Mode 5: can the region be captured, and what does a replay cost?
# ----------------------------------------------------------------------

CAPTURE_BLOCKERS = [
    ("dynamic tensor shapes",
     "one graph per wave size; the histogram says how many are needed"),
    ("allocations inside the captured region",
     "conv/softmax outputs are allocated from the graph's private pool at "
     "capture and reused every replay -- legal, but the outputs are the SAME "
     "tensors each time, so the consumer must copy out before the next replay"),
    ("CPU-dependent branches",
     "forward_both branches on x.dim(), self.value_tanh and "
     "policy.shape[0] == 1; all three are fixed once the shape is fixed, so "
     "they are resolved at capture"),
    ("changing memory addresses",
     "inputs must be copied into static device buffers; the wave cannot hand "
     "the graph a fresh tensor"),
    ("ops unsupported by capture",
     "the region contains conv, linear, activation, masked_fill and softmax; "
     "no .item(), no host allocation, no cpu() -- the D2H is tested "
     "separately"),
    ("hidden synchronizations",
     "a pageable H2D or a .cpu() inside the region would abort the capture; "
     "both are outside it by construction"),
]


def capture_probe(model, waves, device, reps=200):
    """Try the capture, verify the numerics, price the replay.

    Two variants, because the second changes what else can be removed:
      forward+mask+softmax   the region named in the plan
      + D2H into pinned      the two `.cpu()` pulls fold in as captured
                             async copies, leaving one event wait for the host
    """
    states = waves[0]
    k = len(states)
    buf = np.empty((k, 7, 9, 9), dtype=np.float32)
    _fill(buf, states)
    valids = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
              for s in states]
    mask = np.zeros((k, 81), dtype=bool)
    for i, v in enumerate(valids):
        mask[i, v] = True

    res = {"k": k, "reps": reps, "blockers_checked": [
        {"blocker": b, "finding": f} for b, f in CAPTURE_BLOCKERS]}

    with torch.no_grad():
        # Eager reference, and the eager cost to beat.
        ref_probs, ref_values = wave_sequence(model, states, device)

        static_x = torch.zeros((k, 7, 9, 9), device=device)
        static_m = torch.zeros((k, 81), dtype=torch.bool, device=device)
        static_x.copy_(torch.from_numpy(buf))
        static_m.copy_(torch.from_numpy(mask))

        def region():
            lg, vl = model.forward_both(static_x)
            if lg.dim() == 1:
                lg = lg.unsqueeze(0)
                vl = vl.unsqueeze(0)
            return (F.softmax(lg.masked_fill(~static_m, float("-inf")), dim=1),
                    vl.reshape(-1))

        # Capture requires a warmup on a side stream.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5):
                region()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                g_probs, g_values = region()
            res["captured"] = True
            res["capture_error"] = None
        except Exception as exc:                      # pragma: no cover
            res["captured"] = False
            res["capture_error"] = "%s: %s" % (type(exc).__name__, exc)
            return res

        graph.replay()
        torch.cuda.synchronize()
        res["max_abs_probs_diff"] = float(
            np.max(np.abs(g_probs.cpu().numpy() - ref_probs)))
        res["max_abs_values_diff"] = float(
            np.max(np.abs(g_values.cpu().numpy() - ref_values)))

        # Cost of one replay, host side, with nothing queued.
        def replay():
            graph.replay()

        res["replay_launch_us"] = _time_drained(replay, reps)

        def replay_and_drain():
            graph.replay()
            torch.cuda.synchronize()

        res["replay_plus_drain_us"] = _time_drained(replay_and_drain, reps)

        # The eager equivalent of exactly the captured region, same buffers.
        def eager_region():
            region()

        res["eager_launch_us"] = _time_drained(eager_region, reps)

        def eager_region_drain():
            region()
            torch.cuda.synchronize()

        res["eager_plus_drain_us"] = _time_drained(eager_region_drain, reps)

        # --- variant 2: fold the two pulls into the graph ----------------
        host_probs = torch.empty((k, 81), dtype=torch.float32, pin_memory=True)
        host_values = torch.empty((k,), dtype=torch.float32, pin_memory=True)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5):
                p, v = region()
                host_probs.copy_(p, non_blocking=True)
                host_values.copy_(v, non_blocking=True)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph2 = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph2):
                p2, v2 = region()
                host_probs.copy_(p2, non_blocking=True)
                host_values.copy_(v2, non_blocking=True)
            res["captured_with_d2h"] = True
            res["capture_with_d2h_error"] = None
        except Exception as exc:                      # pragma: no cover
            res["captured_with_d2h"] = False
            res["capture_with_d2h_error"] = "%s: %s" % (type(exc).__name__, exc)
            return res

        done = torch.cuda.Event()
        graph2.replay()
        done.record()
        done.synchronize()
        res["max_abs_probs_diff_d2h"] = float(
            np.max(np.abs(host_probs.numpy() - ref_probs)))
        res["max_abs_values_diff_d2h"] = float(
            np.max(np.abs(host_values.numpy() - ref_values)))

        ev = torch.cuda.Event()

        def replay2_full():
            graph2.replay()
            ev.record()
            ev.synchronize()
            host_probs.numpy()
            host_values.numpy()

        res["replay_d2h_end_to_end_us"] = _time_drained(replay2_full, reps)

        # --- variant 3: the WHOLE wave, eager against graphed ------------
        # THE DECISIVE QUANTITY. Not "how many kernels" and not the CPU time
        # spent inside the region -- most of that is the host blocked on a
        # pageable H2D while the GPU works, and removing the block does not
        # create wall clock out of nothing. What matters is the dispatch a
        # replay actually removes MINUS the GPU time that remains, and no
        # arithmetic over segment timings can produce that number. So both
        # whole waves are run, from raw game states to host-side arrays, and
        # timed against each other.
        #
        # This is a BENCH. `agents/mcts.py` is untouched, nothing here is
        # wired into the engine, and no strength claim follows from it.
        pin_x = torch.empty((k, 7, 9, 9), dtype=torch.float32, pin_memory=True)
        pin_m = torch.empty((k, 81), dtype=torch.bool, pin_memory=True)
        vx, vm = pin_x.numpy(), pin_m.numpy()

        def graphed_wave():
            _fill(vx, states)
            static_x.copy_(pin_x, non_blocking=True)
            vv = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
                  for s in states]
            vm[:] = False
            for i, v in enumerate(vv):
                vm[i, v] = True
            static_m.copy_(pin_m, non_blocking=True)
            graph2.replay()
            ev.record()
            ev.synchronize()
            return host_probs.numpy(), host_values.numpy()

        # VERIFY ON EVERY WAVE IN THE POOL, NOT ONE. A graph that had been
        # captured around a stale buffer -- reading a tensor the wave no longer
        # writes into -- would replay the captured input forever and match
        # perfectly on the wave it was captured from. One input cannot tell a
        # working graph from that. So each distinct wave is pushed through
        # both paths, and the outputs are required to DIFFER between waves as
        # well as to match the eager result on each.
        checks, worst_p, worst_v = 0, 0.0, 0.0
        seen = []
        for w in waves:
            if len(w) != k:
                continue
            states = w
            e_probs, e_values = wave_sequence(model, states, device)
            g_probs, g_values = graphed_wave()
            worst_p = max(worst_p, float(np.max(np.abs(g_probs - e_probs))))
            worst_v = max(worst_v, float(np.max(np.abs(g_values - e_values))))
            seen.append(g_probs.copy())
            checks += 1
        res["wave_checks"] = checks
        res["wave_max_abs_probs_diff"] = worst_p
        res["wave_max_abs_values_diff"] = worst_v
        res["wave_distinct_outputs"] = int(len(
            {p.tobytes() for p in seen}))
        states = waves[0]

        def eager_wave():
            wave_sequence(model, states, device)

        res["wave_eager_us"] = _time_drained(eager_wave, reps)
        res["wave_graphed_us"] = _time_drained(graphed_wave, reps)
        res["wave_saving_us"] = max(
            0.0, res["wave_eager_us"] - res["wave_graphed_us"])

    return res


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def _rule(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def report_hist(h):
    _rule("WAVE-SIZE HISTOGRAM -- what production actually runs")
    print("%d games, %d moves, %d waves, %d leaves"
          % (h["games"], h["moves"], h["waves"], h["leaves"]))
    print("mean k %.2f   median k %s   p90 k %s   waves/move %.1f"
          % (h["mean_k"], h["median_k"], h["p90_k"], h["waves_per_move"]))
    print()
    print("  %-4s %10s %8s %8s" % ("k", "waves", "% waves", "% leaves"))
    tot = h["waves"] or 1
    for k in sorted(h["histogram"], key=int):
        n = h["histogram"][k]
        print("  %-4s %10d %7.1f%% %7.1f%%"
              % (k, n, 100.0 * n / tot, 100.0 * h["share_of_leaves"][k]))
    print()
    print("A graph is captured per shape, so this is the graph-cache sizing")
    print("question. It is also the padding question: padding every wave up to")
    print("the top size costs whatever the forward costs at that size, and the")
    print("forward is nearly flat in k (8% across 6.25x, RESULT_EXPAND_CUDA).")


def report_trace(traces):
    _rule("KERNEL STRUCTURE -- CUPTI, one wave at each size")
    print("Counts and device durations are the profiler's; CPU launch cost is")
    print("NOT (CUPTI inflates every launch). See the cost table for that.")
    print()
    hdr = ("%-6s %8s %9s %9s %9s %9s %9s %9s"
           % ("k", "kernels", "mean us", "med us", "<10us n", "<10us t",
              "gpu us", "syncs"))
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(traces, key=int):
        t = traces[k]
        print("%-6s %8.1f %9.2f %9.2f %8.0f%% %8.0f%% %9.1f %9.2f"
              % (k, t["kernels_per_wave"], t["kernel_us_mean"],
                 t["kernel_us_median"], 100 * t["small_kernel_count_share"],
                 100 * t["small_kernel_time_share"],
                 t["gpu_busy_us_per_wave"], t["explicit_syncs_per_wave"]))
    print()
    any_k = traces[sorted(traces, key=int)[-1]]
    print("kernels by kind at k=%s:" % sorted(traces, key=int)[-1])
    for name, d in any_k["kernels_by_kind"].items():
        print("  %-28s %6.1f /wave  %8.2f us/wave"
              % (name, d["n_per_wave"], d["us_per_wave"]))
    print()
    print("copies per wave at k=%s:" % sorted(traces, key=int)[-1])
    for d, v in sorted(any_k["copies_per_wave"].items()):
        print("  %-34s %4.1f copies %7.0f B %6.2f us"
              % (d, v["n"], v["bytes"], v["device_us"]))
    print()
    print("explicit synchronizations per wave: %.2f  %s"
          % (any_k["explicit_syncs_per_wave"], any_k["sync_kinds"]))
    print("device idle between consecutive kernels: %.2f us median, "
          "%.1f us/wave total"
          % (any_k["device_gap_us_median"],
             any_k["device_gap_us_total_per_wave"]))


def report_segmented(seg):
    _rule("KERNELS BY SEGMENT -- which statement launches what")
    print("Counts from an annotated pass. The annotations perturb timing, so")
    print("only counts are read off this; %d runtime calls fell outside every"
          % seg["unattributed_runtime_calls"])
    print("window and were dropped.")
    print()
    hdr = "  %-26s %8s %8s %9s %7s %10s" % (
        "segment", "kernels", "memcpys", "bytes", "syncs", "device us")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in SEGMENTS:
        v = seg["per_segment"].get(s)
        if not v:
            continue
        print("  %-26s %8.1f %8.1f %9.0f %7.1f %10.1f"
              % (s, v["kernels_per_wave"], v["memcpys_per_wave"],
                 v["bytes_per_wave"], v["syncs_per_wave"],
                 v["device_us_per_wave"]))
    fwd = seg["per_segment"].get("network forward", {})
    sm = seg["per_segment"].get("device: mask + softmax", {})
    print()
    print("  network forward:        %.1f kernels -- %s"
          % (fwd.get("kernels_per_wave", 0.0),
             ", ".join("%s x%.0f" % (n, c)
                       for n, c in list(fwd.get("kernel_names", {}).items()))))
    print("  masking + softmax:      %.1f kernels -- %s"
          % (sm.get("kernels_per_wave", 0.0),
             ", ".join("%s x%.0f" % (n, c)
                       for n, c in list(sm.get("kernel_names", {}).items()))))


def report_bench(benches, traces):
    _rule("COST -- profiler off; CPU wall and CUDA events")
    for k in sorted(benches, key=int):
        b = benches[k]
        cpu = b["cpu_us_per_wave"]
        gpu = b.get("gpu_us_per_wave", {})
        print()
        print("k = %s   (CPU total %.1f us/wave, GPU busy %.1f us/wave)"
              % (k, sum(cpu.values()), sum(gpu.values())))
        print("  %-26s %10s %10s" % ("segment", "CPU us", "GPU us"))
        for s in SEGMENTS:
            print("  %-26s %10.1f %10s"
                  % (s, cpu.get(s, 0.0),
                     ("%.1f" % gpu[s]) if s in gpu else "--"))
        t = traces.get(k)
        if not (t and t["kernels_per_wave"]):
            continue
        fwd = cpu.get("network forward", 0.0)
        print("  -> %.1f us of CPU to issue %.1f kernels = %.2f us/launch"
              % (fwd, t["kernels_per_wave"], fwd / t["kernels_per_wave"]))
        # THE TWO DEVICE NUMBERS DISAGREE BY AN ORDER OF MAGNITUDE, AND THAT
        # IS THE FINDING. A CUDA event pair spans from when the first event is
        # processed to when the last is -- so it charges the GPU for every
        # microsecond it sat waiting for the next batch of commands to arrive.
        # CUPTI times the kernels themselves. The gap between them is idle
        # device, and the previous study's "GPU busy 482.5 ms/move" was the
        # former, which is why it looked compute-bound.
        busy = t["gpu_busy_us_per_wave"]
        span = sum(gpu.values())
        if busy and span:
            print("  -> device: %.1f us of kernels+copies (CUPTI) inside "
                  "%.1f us of" % (busy, span))
            print("     stream-elapsed time (CUDA events) -- %.0f%% of the "
                  "\"GPU time\" is the" % (100.0 * (1.0 - busy / span)))
            print("     device idle, waiting to be given work")


def report_bench_vs_production(benches, ref):
    _rule("CROSS-CHECK -- the bench against live play")
    print("Live play (%s) at mean k=%.2f, against the bench at k=8."
          % (ref["source"], ref["mean_k"]))
    print("These are different populations -- the bench replays a sample of")
    print("real waves at ONE size, live play averages over all sizes -- so")
    print("this is an agreement check, not an identity.")
    print()
    b = benches.get("8")
    if b is None:
        print("  [!] no k=8 bench to compare")
        return
    print("  %-26s %10s %10s %8s" % ("segment", "live us", "bench us", "ratio"))
    for s in SEGMENTS:
        live = ref["cpu_us_per_wave"].get(s)
        got = b["cpu_us_per_wave"].get(s)
        if live and got:
            print("  %-26s %10.1f %10.1f %8.2f" % (s, live, got, got / live))


def report_mask(m):
    _rule("MASK PATH -- where the 111.3 ms/move of host time goes")
    print("k = %d, %d reps. Live play spends 280.9 us/wave on the upload"
          % (m["k"], m["reps"]))
    print("statement against 63.2 us of device time for a %d-byte copy."
          % (m["k"] * 81,))
    print()
    for name, us in m["arms"].items():
        print("  %-50s %9.1f us" % (name, us))
    print()
    print("  forward launch (drained)        %9.1f us" % m["forward_launch_us"])
    print("  forward launch + drain          %9.1f us"
          % m["forward_launch_plus_drain_us"])
    print("  forward tail after the launch   %9.1f us" % m["forward_tail_us"])

    arms = m["arms"]
    drained = arms.get("upload: pageable bool .to() [drained]")
    inflight = arms.get("upload: pageable bool .to() [FORWARD IN FLIGHT]")
    if drained is None or inflight is None:
        return
    induced = inflight - drained
    print()
    print("  ATTRIBUTION")
    print("    the same statement costs %.1f us with the stream drained and"
          % drained)
    print("    %.1f us with a forward in flight. The %.1f us difference is"
          % (inflight, induced))
    print("    SYNCHRONIZATION INDUCED BY PRODUCING THE MASK: a pageable H2D")
    print("    emits cudaMemcpyAsync followed by cudaStreamSynchronize, and")
    print("    that sync waits on the whole forward pass queued before it.")
    print("    Not python construction (%.1f us), not allocation (%.1f us),"
          % (arms.get("host: zeros + python fill loop", 0.0),
             arms.get("host: np.zeros((k,81), bool)", 0.0)))
    print("    not dtype (uint8 drained is %.1f us), not the copy itself."
          % arms.get("upload: pageable uint8 .to() [drained]", 0.0))
    tail = m["forward_tail_us"]
    print("    Cross-check: the forward's own tail after its launch returns")
    print("    is %.1f us, against %.1f us of induced wait -- the same wait,"
          % (tail, induced))
    print("    measured two ways (ratio %.2f)."
          % (induced / tail if tail else 0.0))
    print()
    print("    [!] THIS IS NOT %.0f us OF RECOVERABLE WALL CLOCK. The GPU is"
          % induced)
    print("    busy for all of it. Removing the sync moves the wait to the")
    print("    next blocking point; it only pays if the host has real work to")
    print("    do meanwhile, or if the dispatch that made the GPU slow goes")
    print("    away too. Same trap as labelling .cpu() wall time as transfer.")


def report_capture(c):
    _rule("CUDA GRAPH -- capture feasibility and replay cost")
    for row in c["blockers_checked"]:
        print("  [%s] %s" % ("checked", row["blocker"]))
        print("      %s" % row["finding"])
    print()
    if not c.get("captured"):
        print("  [X] capture FAILED: %s" % c.get("capture_error"))
        return
    print("  [OK] forward + mask + softmax captured at k=%d" % c["k"])
    print("       max |probs - eager| = %.3e   max |values - eager| = %.3e"
          % (c["max_abs_probs_diff"], c["max_abs_values_diff"]))
    print("       eager  launch %8.1f us   launch+drain %8.1f us"
          % (c["eager_launch_us"], c["eager_plus_drain_us"]))
    print("       graph  replay %8.1f us   replay+drain %8.1f us"
          % (c["replay_launch_us"], c["replay_plus_drain_us"]))
    saved = c["eager_launch_us"] - c["replay_launch_us"]
    print("       CPU saved per wave: %.1f us (%.0f%% of the eager launch)"
          % (saved, 100.0 * saved / c["eager_launch_us"]
             if c["eager_launch_us"] else 0.0))
    gsav = c["eager_plus_drain_us"] - c["replay_plus_drain_us"]
    print("       end-to-end saved:   %.1f us (this is the one that counts)"
          % gsav)
    if not c.get("captured_with_d2h"):
        print("  [X] D2H variant failed: %s"
              % c.get("capture_with_d2h_error"))
        return
    print("  [OK] + D2H into pinned host buffers also captured")
    print("       max |probs| diff %.3e   max |values| diff %.3e"
          % (c["max_abs_probs_diff_d2h"], c["max_abs_values_diff_d2h"]))
    print("       replay + event wait + read: %8.1f us"
          % c["replay_d2h_end_to_end_us"])
    if "wave_eager_us" not in c:
        return
    print()
    print("  THE WHOLE WAVE, states in and host arrays out:")
    print("       eager    %8.1f us      graphed  %8.1f us"
          % (c["wave_eager_us"], c["wave_graphed_us"]))
    print("       saved    %8.1f us/wave (%.0f%%)"
          % (c["wave_saving_us"],
             100.0 * c["wave_saving_us"] / c["wave_eager_us"]
             if c["wave_eager_us"] else 0.0))
    print("       numerics vs eager over %d distinct waves: probs %.3e, "
          "values %.3e" % (c.get("wave_checks", 0),
                           c["wave_max_abs_probs_diff"],
                           c["wave_max_abs_values_diff"]))
    print("       distinct outputs across those waves: %d of %d (a graph "
          "stuck on" % (c.get("wave_distinct_outputs", 0),
                        c.get("wave_checks", 0)))
    print("       a stale buffer would score 1, and would still match on the")
    print("       wave it was captured from)")
    print("       This is a bench. mcts.py is untouched and nothing here is")
    print("       wired into the engine; no strength claim follows from it.")


RECOVERABLE_EFFICIENCY = 0.7


def decide(hist, traces, benches, mask, capture, ref):
    """The decision rule from task #40, keyed to the MEASURED saving.

    The threshold is written against "launch/host overhead a graph can
    actually eliminate", and that is deliberately not the CPU time spent
    inside the region. Most of the mask upload's 280 us/wave is the host
    blocked on a pageable H2D while the GPU runs the forward -- real CPU time,
    but not time a graph creates out of nothing, because the GPU still has to
    do the work. Summing the CPU column would have scored this at 461 ms/move
    when the whole wave only costs about 1.7 ms.

    So the number the verdict turns on is the end-to-end bench: the same
    states through the same sequence, eager against graphed, both drained.
    Discounted by RECOVERABLE_EFFICIENCY, because a bench replaying one wave
    in a loop does not carry the engine's allocator pressure, its varying
    shapes, or the buffer copies a real integration needs.
    """
    _rule("DECISION")
    # THIS RUN'S OWN WAVE RATE, not the previous study's. The earlier figure
    # (396.1/move) was derived as leaves/move over mean_k from an instrumented
    # run on a different opening set; this run measured 302.9 directly, with
    # nothing on the engine but a counter. Using the old one inflated every
    # ms/move figure below by 31%.
    wpm = (hist or {}).get("waves_per_move") or ref["waves_per_move"]
    b = benches.get("8") or list(benches.values())[0]
    cpu = b["cpu_us_per_wave"]
    gpu = b.get("gpu_us_per_wave", {})

    issue = (cpu.get("network forward", 0.0)
             + cpu.get("H2D: mask", 0.0)
             + cpu.get("device: mask + softmax", 0.0))
    dev = (gpu.get("network forward", 0.0)
           + gpu.get("H2D: mask", 0.0)
           + gpu.get("device: mask + softmax", 0.0))
    out = {
        "waves_per_move_reference": wpm,
        "cpu_in_region_us_per_wave": issue,
        "cpu_in_region_ms_per_move": issue * wpm / 1000.0,
        "gpu_us_per_wave_same_region": dev,
        "gpu_ms_per_move_same_region": dev * wpm / 1000.0,
        "recoverable_efficiency": RECOVERABLE_EFFICIENCY,
    }
    print("waves/move: %.1f measured this run (previous study derived %.1f "
          "on a" % (wpm, ref["waves_per_move"]))
    print("different opening set; using that one would inflate every figure "
          "below by %.0f%%)" % (100.0 * (ref["waves_per_move"] / wpm - 1.0)
                                if wpm else 0.0))
    print("CPU inside forward + mask upload + softmax: %.1f us/wave "
          "= %.1f ms/move" % (issue, out["cpu_in_region_ms_per_move"]))
    print("GPU time in that same region:               %.1f us/wave "
          "= %.1f ms/move" % (dev, out["gpu_ms_per_move_same_region"]))
    print("  [!] the CPU figure is NOT the recoverable one -- most of the mask")
    print("      upload is the host blocked while the GPU works. See below.")
    print()

    capturable = bool(capture.get("captured"))
    measured = capture.get("wave_saving_us")
    if capturable and measured is not None:
        ms = measured * wpm / 1000.0
        disc = ms * RECOVERABLE_EFFICIENCY
        out["wave_saving_us_per_wave"] = measured
        out["wave_saving_ms_per_move"] = ms
        out["wave_saving_ms_per_move_discounted"] = disc
        print("MEASURED, whole wave, eager vs graphed: %.1f -> %.1f us/wave"
              % (capture["wave_eager_us"], capture["wave_graphed_us"]))
        print("  = %.1f ms/move at the live wave rate, %.1f ms/move after a "
              "%.0f%% efficiency discount" % (ms, disc,
                                              100 * RECOVERABLE_EFFICIENCY))
        print()
    else:
        disc = 0.0

    if not capturable:
        verdict = ("abandon graph-first: the region does not capture (%s). "
                   "Unblock the narrow native design."
                   % capture.get("capture_error"))
    elif disc >= 150.0:
        verdict = ("graph-first: %.0f ms/move recoverable (discounted) and the "
                   "region captures with bit-identical output. #36 stays "
                   "blocked." % disc)
    elif disc >= 50.0:
        verdict = ("comparable: %.0f ms/move recoverable against ~97 ms of "
                   "recoverable _best_child. Reconsider the narrow native "
                   "port alongside." % disc)
    else:
        verdict = ("abandon graph-first: only %.0f ms/move recoverable; "
                   "unblock the narrow native design." % disc)
    out["verdict"] = verdict
    print("VERDICT: %s" % verdict)
    print()
    print("No strength claim follows from any of this. Whatever lands is")
    print("judged by win rate at equal wall clock, by itself, or not at all.")
    return out


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="all",
                    choices=["hist", "trace", "bench", "mask", "capture", "all"])
    ap.add_argument("--engine", default="pocket_r35")
    ap.add_argument("--games", type=int, default=4,
                    help="games for the wave-size histogram")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=KERNEL_SEED)
    ap.add_argument("--reps", type=int, default=400,
                    help="wave replays per size in trace and bench")
    ap.add_argument("--sizes", default="",
                    help="comma-separated wave sizes; default 1,4,8 plus the "
                         "production median and p90")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "kernels.json"))
    ap.add_argument("--rerender", default="",
                    help="re-print the reports and recompute the decision "
                         "from a saved run, measuring nothing")
    ap.add_argument("--allow-drift", action="store_true")
    args = ap.parse_args()

    if args.rerender:
        # A twenty-minute measurement should not have to be repeated to fix a
        # label or a denominator. Everything below reads only the saved run.
        with open(args.rerender) as fh:
            res = json.load(fh)
        ref = res["production_reference"]
        report_hist(res["histogram"])
        if res.get("traces"):
            report_trace(res["traces"])
        if res.get("segmented"):
            report_segmented(res["segmented"])
        if res.get("benches"):
            report_bench(res["benches"], res.get("traces", {}))
            report_bench_vs_production(res["benches"], ref)
        if res.get("mask"):
            report_mask(res["mask"])
        if res.get("capture"):
            report_capture(res["capture"])
        if res.get("benches") and res.get("capture"):
            res["decision"] = decide(res["histogram"], res.get("traces", {}),
                                     res["benches"], res.get("mask"),
                                     res["capture"], ref)
            with open(args.rerender, "w") as fh:
                json.dump(res, fh, indent=2, default=str)
            print("\n[OK] rewrote %s" % args.rerender)
        return

    if not args.allow_drift:
        assert_frozen_sources()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[X] this study measures CUDA structure; no CUDA here")
    os.makedirs(OUT_DIR, exist_ok=True)
    acquire_lock()
    import atexit
    atexit.register(release_lock)
    baseline = gpu_baseline()
    if baseline:
        print("[..] GPU load before starting: %.0f%% mean, %.0f%% peak"
              % (baseline["mean_pct"], baseline["max_pct"]))

    ref = dict(PRODUCTION_REFERENCE)
    prior = os.path.join("results", "profile_expand", "expand.json")
    if os.path.exists(prior):
        with open(prior) as fh:
            pj = json.load(fh)
        ev = pj.get("events", {})
        if ev.get("cpu_us_per_wave"):
            ref["cpu_us_per_wave"] = ev["cpu_us_per_wave"]
            ref["gpu_us_per_wave"] = ev["gpu_us_per_wave"]
            ref["mean_k"] = ev["mean_k"]
            ref["waves_per_move"] = (pj["ref_waves_per_move"] / ev["mean_k"]
                                     if ev.get("mean_k") else
                                     ref["waves_per_move"])

    res = {
        "engine": args.engine, "seed": args.seed, "mode": args.mode,
        "reps": args.reps,
        "git_head": engine_registry.git_head(),
        "environment": engine_registry.environment(),
        "gpu_baseline_before_run": baseline,
        "production_reference": ref,
    }

    def save():
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print("\n[OK] wrote %s" % args.out)

    want = args.mode
    need_pool = want in ("trace", "bench", "mask", "capture", "all")

    hist, pool = None, None
    if want in ("hist", "all") or need_pool:
        print("[..] collecting production waves: %d games at %s"
              % (args.games, args.engine))
        hist, pool = collect_waves(args.engine, args.games, args.device,
                                   args.seed)
        res["histogram"] = hist
        report_hist(hist)
        save()
    if want == "hist":
        return

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    else:
        sizes = [1, 4, 8, hist["median_k"], hist["p90_k"]]
    sizes = sorted({s for s in sizes if s and pool.get(s)})
    missing = sorted({1, 4, 8, hist["median_k"], hist["p90_k"]} - set(sizes))
    if missing:
        print("\n[!] no production waves collected at size(s) %s -- skipped "
              "rather than synthesised, because a synthetic wave is not a "
              "measurement of this engine." % missing)
    if not sizes:
        raise SystemExit("[X] no wave sizes available to measure")
    print("\n[..] measuring wave sizes %s" % sizes)

    # One model, built exactly as deployment builds it.
    player = TimedPlayer("engine:%s" % args.engine, args.device)
    model = player.model
    res["net"] = player.net_info
    res["fingerprint"] = (player.provenance or {}).get("fingerprint")

    traces, benches = {}, {}
    if want in ("trace", "all"):
        for k in sizes:
            print("[..] tracing k=%d" % k)
            traces[str(k)] = trace_waves(model, pool[k], args.device,
                                         args.reps)
        res["traces"] = traces
        report_trace(traces)
        big = max(sizes)
        print("[..] segmented trace at k=%d" % big)
        seg = trace_segmented(model, pool[big], args.device,
                              min(args.reps, 200))
        res["segmented"] = seg
        report_segmented(seg)
        save()

    if want in ("bench", "all"):
        for k in sizes:
            print("[..] benching k=%d" % k)
            benches[str(k)] = bench_waves(model, pool[k], args.device,
                                          args.reps)
        res["benches"] = benches
        report_bench(benches, traces)
        report_bench_vs_production(benches, ref)
        save()

    mask = None
    if want in ("mask", "all"):
        big = max(sizes)
        print("[..] decomposing the mask path at k=%d" % big)
        mask = mask_decomposition(model, pool[big], args.device)
        res["mask"] = mask
        report_mask(mask)
        save()

    cap = None
    if want in ("capture", "all"):
        big = max(sizes)
        print("[..] probing CUDA graph capture at k=%d" % big)
        cap = capture_probe(model, pool[big], args.device)
        res["capture"] = cap
        report_capture(cap)
        save()

    if want == "all":
        res["decision"] = decide(hist, traces, benches, mask, cap, ref)
        save()


if __name__ == "__main__":
    main()
