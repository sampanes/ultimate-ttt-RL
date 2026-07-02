"""bench_throughput.py -- measure which throughput levers actually help on THIS hardware,
so a multi-week training run uses the best-known config instead of a guess.

Runs the real training scripts (train_league.py / train_alphazero.py) as timed, black-box
subprocesses -- NO changes to those scripts, so there is zero regression risk to the run
this is meant to protect. Each candidate is a fixed flag combination; the harness sizes the
work to a wall-clock target, runs it a few times, and reports mean +- stdev games/sec.

    python -m scripts.bench_throughput                       # both suites, ~5 min/candidate
    python -m scripts.bench_throughput --suite league --minutes 3
    python -m scripts.bench_throughput --only "batch_opponents|compile|amp"   # just the new levers
    python -m scripts.bench_throughput --quick               # 1 repeat, ~1 min/candidate smoke

METRIC: end-to-end games/sec = (games the harness told the run to play) / (subprocess wall
time). End-to-end on purpose -- self-play AND the learn step. A lever like --recompute only
touches the learn step, and a slow learn step eats throughput just as much as slow self-play.

METHOD per candidate:
  1. PROBE (tiny, timing discarded): estimate the rate so step 2 is sized sensibly.
  2. MEASURE: size the game count so one repeat ~= minutes*60/repeats seconds, run --repeats
     fresh subprocesses, time each. Report mean/stdev/n. (Each repeat is a fresh process, so
     ~2-5s of CUDA/cuDNN init is paid per repeat, not amortized -- sizing to ~100s keeps that
     to a few percent. This is stated in the report, not hidden.)

ISOLATION (never disturbs a real run or the live dashboard):
  * every subprocess gets --no_metrics (never writes loss_logs/metrics_log.jsonl) and its own
    scratch --model_dir under loss_logs/bench_throughput/ (gitignored).
  * league runs pass --seed 0 --seed_model "" --patience 0 so RNG/opponent sampling line up
    across candidates and a fresh clone doesn't FileNotFoundError on the default seed path.
  * HARD PRECONDITION: do NOT run this while a real long training run is using the GPU -- they
    contend for the one card and invalidate both. This is a "run it before the long run" tool.

A candidate that exits non-zero (e.g. CUDA OOM -- --parallel 512 is a known OOM on a 10GB card)
is recorded FAILED with the tail of stderr and the sweep continues.

Output: ONE markdown report (default bench_throughput_report.md, gitignored -- carries local
paths, paste-back only). Exit 0 always (a FAILED candidate is data, not a harness error).
"""
import argparse
import io
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

from scripts.home_batch import REPO, _CHILD_ENV, _ENV_PROBE

SCRATCH = os.path.join(REPO, "loss_logs", "bench_throughput")
_ENV = {**_CHILD_ENV, "COCKPIT_PROC_LABEL": "uttt-bench-throughput"}


# ------------------------------------------------------------------ candidates
# Each candidate: (label, module, base_flags, games_flag, parallel_hint)
#   games_flag  -- the CLI flag whose value the harness sets to size the work.
#   parallel_hint -- games/batch, used only to pick a representative probe size and to round
#                    the measured game count up to a whole number of batches (league only).
LEAGUE_MODULE = "scripts.train_league"
AZ_MODULE = "scripts.train_alphazero"

# Anchors for the non-parallel sweeps. 64 is the hardware-confirmed best for the RTX 3080
# (PENDING.md: 512 OOMs, 256 starves updates, 64 flies); override with --anchor_parallel.
DEF_ANCHOR_PARALLEL = 64
DEF_ANCHOR_WAVE = 16


def _league_base(parallel, network, extra=None):
    base = [
        "--no_metrics", "--seed", "0", "--seed_model", "",
        "--patience", "0", "--curriculum",
        "--network", network, "--parallel", str(parallel), "--chunks", "1",
    ]
    if extra:
        base += extra
    return base


def _az_base(wave, n_sims, network, extra=None):
    base = [
        "--no_metrics", "--seed", "0",
        "--network", network, "--n_sims", str(n_sims),
        "--wave_size", str(wave), "--iters", "1",
        # keep the fixed learn cost modest so self-play throughput isn't drowned by it
        "--train_steps", "50", "--eval_games", "0",
    ]
    if extra:
        base += extra
    return base


def build_candidates(anchor_parallel, anchor_wave):
    league = []
    # floor
    league.append(("league:seq[parallel=0]", LEAGUE_MODULE,
                   _league_base(0, "medium"), "--chunk_games", 0))
    # parallel sweep (512 deliberately omitted -- known OOM on 10GB)
    for p in (16, 32, 64, 128, 256):
        league.append((f"league:parallel={p}", LEAGUE_MODULE,
                       _league_base(p, "medium"), "--chunk_games", p))
    # network sweep at the anchor
    for net in ("small", "medium", "large"):
        league.append((f"league:network={net}", LEAGUE_MODULE,
                       _league_base(anchor_parallel, net), "--chunk_games", anchor_parallel))
    # recompute (decoupled update count) sweep at the anchor
    for mb in (0, 32, 64, 128):
        league.append((f"league:recompute,mb={mb}", LEAGUE_MODULE,
                       _league_base(anchor_parallel, "medium", ["--recompute", "--minibatch_size", str(mb)]),
                       "--chunk_games", anchor_parallel))
    # the NEW levers, individually and combined, at the anchor
    league.append((f"league:batch_opponents", LEAGUE_MODULE,
                   _league_base(anchor_parallel, "medium", ["--batch_opponents"]),
                   "--chunk_games", anchor_parallel))
    league.append((f"league:compile", LEAGUE_MODULE,
                   _league_base(anchor_parallel, "medium", ["--compile"]),
                   "--chunk_games", anchor_parallel))
    league.append((f"league:amp", LEAGUE_MODULE,
                   _league_base(anchor_parallel, "medium", ["--amp"]),
                   "--chunk_games", anchor_parallel))
    league.append((f"league:batch_opponents+compile+amp", LEAGUE_MODULE,
                   _league_base(anchor_parallel, "medium",
                                ["--batch_opponents", "--compile", "--amp"]),
                   "--chunk_games", anchor_parallel))

    az = []
    for w in (1, 4, 8, 16, 32, 64):
        az.append((f"az:wave_size={w}", AZ_MODULE,
                   _az_base(w, 100, "medium"), "--games_per_iter", 0))
    for s in (100, 200, 400):
        az.append((f"az:n_sims={s}", AZ_MODULE,
                   _az_base(anchor_wave, s, "medium"), "--games_per_iter", 0))
    for net in ("small", "medium", "large"):
        az.append((f"az:network={net}", AZ_MODULE,
                   _az_base(anchor_wave, 100, net), "--games_per_iter", 0))
    return league, az


# ------------------------------------------------------------------ subprocess
def _scratch(label):
    d = os.path.join(SCRATCH, re.sub(r"[^A-Za-z0-9_]+", "_", label))
    os.makedirs(d, exist_ok=True)
    return d


def run_once(module, base_flags, games_flag, n_games, label, timeout):
    """Run one training subprocess for n_games; return (rc, wall_seconds, stderr_tail)."""
    cmd = [sys.executable, "-m", module] + base_flags + [games_flag, str(n_games),
                                                         "--model_dir", _scratch(label)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=REPO, env=_ENV, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        wall = time.time() - t0
        tail = (p.stderr or "")[-800:]
        return p.returncode, wall, tail
    except subprocess.TimeoutExpired:
        return 124, time.time() - t0, f"TIMEOUT after {timeout}s"


class GpuSampler:
    """Best-effort nvidia-smi poller -- records mean/max GPU util% and mem MiB. No-op if
    nvidia-smi is missing. Runs only while active so it never touches an idle GPU."""
    def __init__(self, interval=3.0):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.util = []
        self.mem = []

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                line = (out.stdout or "").strip().splitlines()
                if line:
                    u, m = line[0].split(",")
                    self.util.append(float(u))
                    self.mem.append(float(m))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self):
        if not self.util:
            return None
        return {"util_mean": round(statistics.mean(self.util), 1),
                "util_max": round(max(self.util), 1),
                "mem_max_mib": round(max(self.mem), 1) if self.mem else None}


# ------------------------------------------------------------------ measurement
def bench_candidate(cand, args):
    label, module, base_flags, games_flag, phint = cand
    is_seq = phint == 0 and module == LEAGUE_MODULE
    is_az = module == AZ_MODULE

    # --- probe ---
    if is_az:
        probe_n = 5
    elif is_seq:
        probe_n = 16
    else:
        probe_n = max(phint, 32)

    print(f"  [{label}] probe ({games_flag} {probe_n})...", flush=True)
    rc, wall, tail = run_once(module, base_flags, games_flag, probe_n, label + "_probe",
                              timeout=args.probe_timeout)
    if rc != 0:
        print(f"    FAILED (rc={rc})", flush=True)
        return {"label": label, "status": f"FAILED(rc={rc})", "flags": base_flags,
                "stderr": tail, "gps": [], "n_games": None}
    rate = probe_n / wall if wall > 1e-6 else 1.0

    # --- size the measurement to the wall-clock target ---
    sec_per_repeat = max(10.0, (args.minutes * 60.0) / max(1, args.repeats))
    n = int(rate * sec_per_repeat)
    n = max(args.min_games, min(args.max_games, n))
    if is_az:
        n = max(5, min(n, args.az_max_games))
    elif not is_seq:
        # round UP to a whole number of batches so every measured run is steady-state
        n = max(phint, ((n + phint - 1) // phint) * phint)
    print(f"    probe rate ~{rate:.1f} games/s -> measure {games_flag} {n} x{args.repeats}",
          flush=True)

    # --- measure ---
    gps, walls = [], []
    sampler = GpuSampler() if args.gpu_sample else None
    for r in range(args.repeats):
        if sampler:
            sampler.start()
        rc, wall, tail = run_once(module, base_flags, games_flag, n, label,
                                  timeout=args.measure_timeout)
        if sampler:
            sampler.stop()
        if rc != 0:
            print(f"    repeat {r+1}/{args.repeats} FAILED (rc={rc})", flush=True)
            return {"label": label, "status": f"FAILED(rc={rc})", "flags": base_flags,
                    "stderr": tail, "gps": gps, "n_games": n}
        g = n / wall if wall > 1e-6 else 0.0
        gps.append(g)
        walls.append(wall)
        print(f"    repeat {r+1}/{args.repeats}: {g:.1f} games/s ({wall:.1f}s)", flush=True)

    return {"label": label, "status": "OK", "flags": base_flags, "gps": gps,
            "n_games": n, "wall_mean": statistics.mean(walls),
            "gpu": sampler.summary() if sampler else None}


# ------------------------------------------------------------------ report
def _fmt_row(res):
    if not res["gps"]:
        return (f"| {res['label']} | -- | -- | 0 | {res.get('n_games') or '--'} | -- "
                f"| {res['status']} |")
    mean = statistics.mean(res["gps"])
    sd = statistics.stdev(res["gps"]) if len(res["gps"]) > 1 else 0.0
    wall = res.get("wall_mean", 0.0)
    return (f"| {res['label']} | {mean:.1f} | {sd:.1f} | {len(res['gps'])} "
            f"| {res['n_games']} | {wall:.1f} | {res['status']} |")


def _recommend(results):
    ok = [r for r in results if r["status"] == "OK" and r["gps"]]
    if not ok:
        return "No candidate completed -- check the FAILED rows' stderr tails below."
    ok.sort(key=lambda r: statistics.mean(r["gps"]), reverse=True)
    top = ok[0]
    top_mean = statistics.mean(top["gps"])
    line = f"**Fastest: `{top['label']}` at {top_mean:.1f} games/s.**"
    if len(ok) > 1:
        second = ok[1]
        sm = statistics.mean(second["gps"])
        if sm > 0 and (top_mean - sm) / top_mean < 0.05:
            line += (f" Within ~5% of `{second['label']}` ({sm:.1f}) -- effectively a wash; "
                     f"prefer whichever has fewer moving parts.")
    return line


def write_report(path, env_raw, league_res, az_res, args):
    buf = io.StringIO()
    w = buf.write
    w("# Throughput benchmark report\n\n")
    w("_Generated by `scripts/bench_throughput.py`. Metric = end-to-end games/sec "
      "(self-play + learn), (configured games) / (subprocess wall time)._\n\n")
    w(f"Settings: minutes/candidate={args.minutes}, repeats={args.repeats}, "
      f"anchor_parallel={args.anchor_parallel}, anchor_wave={args.anchor_wave}, "
      f"gpu_sample={args.gpu_sample}.\n\n")
    w("> Each repeat is a fresh process, so ~2-5s CUDA/cuDNN init is paid per repeat "
      "(a few % at ~100s sizing). Numbers rank configs; they are not absolute records.\n\n")

    w("## Environment\n\n```\n")
    w(env_raw.strip() + "\n```\n\n")

    for title, res in (("League suite (`train_league.py`)", league_res),
                       ("AlphaZero suite (`train_alphazero.py`)", az_res)):
        if not res:
            continue
        w(f"## {title}\n\n")
        w("| candidate | games/s (mean) | stdev | n | games/repeat | s/repeat | status |\n")
        w("|---|--:|--:|--:|--:|--:|---|\n")
        for r in sorted(res, key=lambda r: (statistics.mean(r["gps"]) if r["gps"] else -1),
                        reverse=True):
            w(_fmt_row(r) + "\n")
        w("\n" + _recommend(res) + "\n\n")
        if args.gpu_sample:
            w("GPU sampling (util% mean/max, mem MiB max):\n\n")
            for r in res:
                if r.get("gpu"):
                    g = r["gpu"]
                    w(f"- `{r['label']}`: util {g['util_mean']}/{g['util_max']}, "
                      f"mem {g['mem_max_mib']}\n")
            w("\n")
        fails = [r for r in res if r["status"] != "OK"]
        if fails:
            w("<details><summary>FAILED candidates (stderr tail)</summary>\n\n")
            for r in fails:
                w(f"**{r['label']}** ({r['status']}):\n\n```\n{r.get('stderr','')}\n```\n\n")
            w("</details>\n\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Benchmark UTTT training throughput levers.")
    ap.add_argument("--suite", choices=["league", "alphazero", "both"], default="both")
    ap.add_argument("--minutes", type=float, default=5.0,
                    help="Soft wall-clock target per candidate (split across --repeats). The "
                         "'5 min' is just a default; raise it for lower variance.")
    ap.add_argument("--repeats", type=int, default=3, help="Measurement runs per candidate.")
    ap.add_argument("--quick", action="store_true",
                    help="Fast smoke of the whole matrix: --repeats 1 --minutes 1.")
    ap.add_argument("--only", type=str, default="",
                    help="Regex; only run candidates whose label matches (e.g. "
                         "'batch_opponents|compile|amp' for just the new levers).")
    ap.add_argument("--anchor_parallel", type=int, default=DEF_ANCHOR_PARALLEL,
                    help="--parallel used for the non-parallel league sweeps.")
    ap.add_argument("--anchor_wave", type=int, default=DEF_ANCHOR_WAVE,
                    help="--wave_size used for the non-wave AZ sweeps.")
    ap.add_argument("--gpu_sample", action="store_true",
                    help="Poll nvidia-smi during measure runs for util%%/mem (best-effort).")
    ap.add_argument("--min_games", type=int, default=16)
    ap.add_argument("--max_games", type=int, default=20000)
    ap.add_argument("--az_max_games", type=int, default=500,
                    help="Cap AZ games/iter (MCTS per-move cost makes big counts slow).")
    ap.add_argument("--probe_timeout", type=int, default=180)
    ap.add_argument("--measure_timeout", type=int, default=1800)
    ap.add_argument("--out", type=str, default="bench_throughput_report.md")
    args = ap.parse_args()

    if args.quick:
        args.repeats = 1
        args.minutes = 1.0

    print("=" * 70)
    print("UTTT throughput benchmark")
    print("PRECONDITION: no real training run should be using the GPU right now.")
    print("=" * 70, flush=True)

    # env probe (same probe home_batch uses)
    p = subprocess.run([sys.executable, "-c", _ENV_PROBE], cwd=REPO, env=_ENV,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    env_raw = ""
    for line in (p.stdout or "").splitlines():
        if line.startswith("ENVJSON:"):
            try:
                env_raw = json.dumps(json.loads(line[len("ENVJSON:"):]), indent=2)
            except Exception:
                env_raw = line
    if not env_raw:
        env_raw = (p.stdout or "") + (p.stderr or "")
    print("Environment:\n" + env_raw, flush=True)

    league_cands, az_cands = build_candidates(args.anchor_parallel, args.anchor_wave)
    if args.suite == "league":
        az_cands = []
    elif args.suite == "alphazero":
        league_cands = []
    if args.only:
        rx = re.compile(args.only)
        league_cands = [c for c in league_cands if rx.search(c[0])]
        az_cands = [c for c in az_cands if rx.search(c[0])]

    league_res, az_res = [], []
    if league_cands:
        print(f"\n--- League suite: {len(league_cands)} candidates ---", flush=True)
        for c in league_cands:
            league_res.append(bench_candidate(c, args))
    if az_cands:
        print(f"\n--- AlphaZero suite: {len(az_cands)} candidates ---", flush=True)
        for c in az_cands:
            az_res.append(bench_candidate(c, args))

    out_path = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    write_report(out_path, env_raw, league_res, az_res, args)
    print(f"\nReport written: {out_path}")
    print("Paste it back (gitignored -- carries local paths).")


if __name__ == "__main__":
    main()
