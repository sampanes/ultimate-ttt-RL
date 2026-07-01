"""home_batch.py -- one turnkey command at home, paste back one report.

The whole point: front-load every decision on the authoring box; at home you run
ONE command, it captures everything into `home_batch_report.md`, and you paste
that file back. No authoring at home.

    python -m scripts.home_batch                 # default: probe + sweep + seeds (the cheap answers)
    python -m scripts.home_batch --phase run     # just the long strength run (watch the dashboard)
    python -m scripts.home_batch --phase all     # cheap answers, then the long run

Phases:
  probe  env facts (torch/CUDA/GPU/VRAM/TF32/C++ engine) + tactics unit test +
         a tiny smoke train that proves the new flags run under real torch and
         that metrics carry `stage`/`elo`.
  sweep  value-weight sweep (--value_coef 0.25/0.5/1.0, same fixed seed) -- answers
         "is 0.5 still right after the 0c .mean() switch?" (RESULT_0c ask #3).
  seeds  N reproducible repeats at a fixed budget -- pins the 1477-vs-1728 spread
         (is it luck or real variance?).  (RESULT_0c ask #2 / GOAT_NEXT #2)
  run    the long batched strength run (--save_best --patience), streamed live so
         the dashboard Training tab shows it; exercises --restore_best as a side
         effect if it peaks-then-drifts.  (GOAT_NEXT #3)

Everything except the long run is captured to the report; the long run streams to
your terminal + the dashboard and its final summary is appended.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(REPO, "loss_logs", "metrics_log.jsonl")
REPORT = os.path.join(REPO, "home_batch_report.md")
SCRATCH = os.path.join(REPO, "loss_logs", "home_batch")

# Child env: force utf-8 so the agent's emoji prints can't crash a captured run,
# and label the process so the cockpit Processes tab shows a readable name.
_CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "COCKPIT_PROC_LABEL": "uttt-home-batch"}

_lines = []


def emit(s=""):
    print(s, flush=True)
    _lines.append(s)


def flush_report():
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\n[home_batch] report written -> {REPORT}", flush=True)


def _train_cmd(extra):
    return [sys.executable, "-m", "scripts.train_league"] + extra


def run_train(extra, capture=True):
    """Run train_league as a child. capture=True buffers output (short phases);
    capture=False streams to the terminal (the long run, so the dashboard + you
    can watch it live)."""
    cmd = _train_cmd(extra)
    if capture:
        p = subprocess.run(cmd, cwd=REPO, env=_CHILD_ENV, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout or "", p.stderr or ""
    p = subprocess.run(cmd, cwd=REPO, env=_CHILD_ENV)
    return p.returncode, "", ""


def run_parity(games, network):
    """Run the recompute-vs-in-graph parity check. Returns (passed, captured_output)."""
    p = subprocess.run(
        [sys.executable, "-m", "scripts.verify_recompute_parity",
         "--games", str(games), "--network", network],
        cwd=REPO, env=_CHILD_ENV, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return p.returncode == 0, (p.stdout or "") + (p.stderr or "")


def read_metrics():
    """Parse loss_logs/metrics_log.jsonl (written + cleared by each train run)
    into a compact summary. Read AFTER a child finishes, BEFORE the next starts."""
    if not os.path.isfile(METRICS):
        return None
    rows = []
    with open(METRICS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return None
    elos = [r["elo"] for r in rows if "elo" in r]
    losses = [r["loss"] for r in rows if "loss" in r]
    wrs = [r["winrate"] for r in rows if "winrate" in r]
    stages = [r["stage"] for r in rows if "stage" in r]
    # Value-head quality: raw MSE is comparable across a value_coef sweep (unlike the
    # blended loss); explained variance is scale-free. The mean over the run is steadier
    # than the last record for ranking coefs.
    vlosses = [r["value_loss"] for r in rows if "value_loss" in r]
    evs = [r["explained_var"] for r in rows if "explained_var" in r]
    return {
        "records": len(rows),
        "final_elo": round(elos[-1], 1) if elos else None,
        "peak_elo": round(max(elos), 1) if elos else None,
        "final_loss": round(losses[-1], 4) if losses else None,
        "min_loss": round(min(losses), 4) if losses else None,
        "final_winrate": round(wrs[-1], 4) if wrs else None,
        "final_stage": stages[-1] if stages else None,
        "has_stage_elo": bool(stages) and bool(elos),
        "mean_value_loss": round(sum(vlosses) / len(vlosses), 5) if vlosses else None,
        "final_value_loss": round(vlosses[-1], 5) if vlosses else None,
        "mean_explained_var": round(sum(evs) / len(evs), 4) if evs else None,
        "final_explained_var": round(evs[-1], 4) if evs else None,
        "has_value_metrics": bool(vlosses),
    }


def _scratch(name):
    d = os.path.join(SCRATCH, name)
    os.makedirs(d, exist_ok=True)
    return d


_ENV_PROBE = r'''
import json, sys, platform
info = {"python": sys.version.split()[0], "platform": platform.platform()}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        p = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(p.total_memory / (1024**3), 2)
        info["capability"] = "%d.%d" % (p.major, p.minor)
    try:
        info["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
except Exception as e:
    info["torch_error"] = repr(e)
try:
    import numpy
    info["numpy"] = numpy.__version__
except Exception as e:
    info["numpy_error"] = repr(e)
try:
    import engine.game as g
    info["cpp_engine_active"] = bool(getattr(g, "_CPP_ENGINE", False))
    info["cpp_build_dir"] = getattr(g, "_cpp_build", None)
except Exception as e:
    info["engine_error"] = repr(e)
print("ENVJSON:" + json.dumps(info))
'''


def _git(args):
    try:
        return subprocess.run(["git"] + args, cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    except Exception as e:
        return f"(git error: {e})"


def phase_probe():
    emit("## Phase: probe\n")

    # --- environment ---
    emit("### Environment")
    p = subprocess.run([sys.executable, "-c", _ENV_PROBE], cwd=REPO, env=_CHILD_ENV,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    env = {}
    for line in (p.stdout or "").splitlines():
        if line.startswith("ENVJSON:"):
            try:
                env = json.loads(line[len("ENVJSON:"):])
            except json.JSONDecodeError:
                pass
    emit("```json")
    emit(json.dumps(env, indent=2))
    emit("```")
    emit(f"- git: `{_git(['rev-parse', '--short', 'HEAD'])}` on "
         f"`{_git(['rev-parse', '--abbrev-ref', 'HEAD'])}`"
         f"{' (dirty)' if _git(['status', '--porcelain']) else ' (clean)'}")
    emit("")

    # --- tactics unit test (verifies the lookahead on THIS box's engine too) ---
    emit("### Tactics unit test")
    t = subprocess.run([sys.executable, "-m", "engine.test_tactics"], cwd=REPO,
                       env=_CHILD_ENV, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tac_ok = t.returncode == 0
    emit(f"- engine.test_tactics: **{'PASS' if tac_ok else 'FAIL'}** (exit {t.returncode})")
    if not tac_ok:
        emit("```")
        emit((t.stdout or "") + (t.stderr or ""))
        emit("```")
    emit("")

    # --- smoke trains: do the new flags run under real torch, with stage/elo? ---
    emit("### Smoke trains (does the new code run; are stage+elo logged?)")
    smoke = [
        ("sequential --fix_0c", ["--chunks", "1", "--chunk_games", "40", "--parallel", "0",
                                 "--fix_0c", "--network", "small", "--seed_model", "",
                                 "--seed", "0", "--model_dir", _scratch("smoke_seq")]),
        ("batched", ["--chunks", "1", "--chunk_games", "64", "--parallel", "16",
                     "--network", "small", "--seed_model", "", "--seed", "0",
                     "--model_dir", _scratch("smoke_par")]),
    ]
    smoke_results = []
    for name, extra in smoke:
        rc, out, err = run_train(extra)
        m = read_metrics()
        ok = rc == 0 and m is not None and m["has_stage_elo"]
        smoke_results.append((name, ok, rc, m))
        emit(f"- {name}: **{'OK' if ok else 'PROBLEM'}** "
             f"(exit {rc}; metrics: {m})")
        if rc != 0:
            emit("```")
            emit((err or out or "")[-1500:])
            emit("```")
    emit("")

    # --- recompute parity (THROUGHPUT Part C alignment oracle) ---
    emit("### Recompute parity (THROUGHPUT Part C)")
    par_ok, par_out = run_parity(60, "small")
    emit(f"- verify_recompute_parity: **{'PASS' if par_ok else 'FAIL'}** "
         f"(in-graph vs collect-then-recompute loss terms)")
    emit("```")
    emit("\n".join(par_out.strip().splitlines()[-12:]))
    emit("```")
    emit("")
    return {"env": env, "tactics_pass": tac_ok, "parity_pass": par_ok,
            "smoke": [{"name": n, "ok": ok, "rc": rc} for n, ok, rc, _ in smoke_results]}


def phase_recompute(network, parity_games, ab_chunks, ab_games, baseline_parallel,
                    big_parallel, minibatch, seed):
    emit("## Phase: recompute (THROUGHPUT Part C validation)\n")
    emit("Validates the collect-then-recompute learn path before any long run trusts it: "
         "(1) numerical PARITY vs the trusted in-graph path (the alignment gate), (2) it runs "
         "under real torch with stage/elo/EV logged, (3) a short A/B showing a big batch + "
         "minibatched updates is not worse than the safe small-batch config at equal games "
         "budget (the whole point: GPU-fat batch WITHOUT starving gradient updates).\n")

    # (1) parity gate -- if this fails, nothing else matters.
    emit("### 1. Parity gate")
    par_ok, par_out = run_parity(parity_games, network)
    emit(f"- verify_recompute_parity (--games {parity_games} --network {network}): "
         f"**{'PASS' if par_ok else 'FAIL'}**")
    emit("```")
    emit("\n".join(par_out.strip().splitlines()[-14:]))
    emit("```")
    if not par_ok:
        emit("\n**STOP -- parity failed. Do NOT enable --recompute.** The "
             "(state,action)<->reward alignment is broken; the A/B below would be meaningless.\n")
        return {"recompute": {"parity_pass": False}}
    emit("")

    # (2) smoke -- does --recompute run end-to-end with stage/elo/EV?
    emit("### 2. Smoke train (--recompute runs; stage/elo/EV logged?)")
    extra = ["--chunks", "1", "--chunk_games", str(ab_games), "--parallel", str(big_parallel),
             "--recompute", "--minibatch_size", str(minibatch), "--network", network,
             "--seed_model", "", "--seed", str(seed), "--curriculum",
             "--model_dir", _scratch("recompute_smoke")]
    rc, out, err = run_train(extra)
    m = read_metrics() or {}
    smoke_ok = rc == 0 and m.get("has_stage_elo") and m.get("has_value_metrics")
    emit(f"- --parallel {big_parallel} --recompute --minibatch_size {minibatch}: "
         f"**{'OK' if smoke_ok else 'PROBLEM'}** (exit {rc}; metrics: {m})")
    if rc != 0:
        emit("```")
        emit((err or out or "")[-1500:])
        emit("```")
    emit("")

    # (3) A/B at equal games budget -- directional, single seed.
    emit("### 3. A/B at equal games budget (single seed, short -- directional, not definitive)")
    configs = [
        ("baseline-ingraph", ["--parallel", str(baseline_parallel)]),
        ("recompute-bigbatch", ["--parallel", str(big_parallel),
                                "--recompute", "--minibatch_size", str(minibatch)]),
    ]
    ab_rows = []
    for name, cfg_extra in configs:
        extra = (["--chunks", str(ab_chunks), "--chunk_games", str(ab_games)] + cfg_extra +
                 ["--network", network, "--seed_model", "", "--seed", str(seed), "--curriculum",
                  "--patience", "0", "--model_dir", _scratch("recompute_ab_" + name)])
        t0 = time.time()
        rc, out, err = run_train(extra)
        dt = time.time() - t0
        m = read_metrics() or {}
        ab_rows.append({"config": name, "rc": rc, "secs": round(dt, 1), **m})
        emit(f"- {name}: exit {rc} | {dt:.1f}s | final_stage={m.get('final_stage')} "
             f"peak_elo={m.get('peak_elo')} final_wr={m.get('final_winrate')} "
             f"mean_EV={m.get('mean_explained_var')}")
        if rc != 0:
            emit("```")
            emit((err or out or "")[-1200:])
            emit("```")
    emit("")
    emit("| config | secs | final stage | peak ELO | final WR | mean EV |")
    emit("|---|---|---|---|---|---|")
    for r in ab_rows:
        emit(f"| {r['config']} | {r.get('secs')} | {r.get('final_stage')} | "
             f"{r.get('peak_elo')} | {r.get('final_winrate')} | {r.get('mean_explained_var')} |")
    emit("\n**Read:** --recompute is cleared for a long run when parity PASSED, the smoke logged "
         "stage/elo/EV, and the big-batch recompute row is not worse on stage/EV/WR than the "
         "baseline at equal games (ideally in <= the wall-time -- that's the GPU throughput it "
         "buys back). Then run long with `--parallel <big> --recompute --minibatch_size <mb>`.\n")
    return {"recompute": {"parity_pass": True, "smoke_ok": smoke_ok, "ab": ab_rows}}


def phase_sweep(value_coefs, chunks, games, parallel, seed):
    emit("## Phase: sweep (value-weight)\n")
    emit(f"Fixed seed {seed}, {chunks} chunks x {games} games, --parallel {parallel}, "
         f"--curriculum, batched path. Only --value_coef varies, so the comparison is "
         f"controlled. RESULT_HOME_BATCH found blended loss + peak ELO can't discriminate "
         f"coefs -- so rank by the value-head metrics: **explained variance** (scale-free, "
         f"higher=better critic) and raw **value MSE** (comparable across coefs; the blended "
         f"loss is NOT).\n")
    rows = []
    for vc in value_coefs:
        extra = ["--chunks", str(chunks), "--chunk_games", str(games),
                 "--parallel", str(parallel), "--network", "small", "--seed_model", "",
                 "--seed", str(seed), "--curriculum", "--value_coef", str(vc),
                 "--model_dir", _scratch(f"sweep_vc{vc}")]
        rc, out, err = run_train(extra)
        m = read_metrics() or {}
        rows.append({"value_coef": vc, "rc": rc, **m})
        emit(f"- value_coef={vc}: exit {rc} | peak_elo={m.get('peak_elo')} "
             f"final_elo={m.get('final_elo')} mean_EV={m.get('mean_explained_var')} "
             f"mean_value_mse={m.get('mean_value_loss')} final_wr={m.get('final_winrate')}")
        if rc != 0:
            emit("```")
            emit((err or out or "")[-1200:])
            emit("```")
    emit("")
    emit("| value_coef | mean EV | final EV | mean value-MSE | peak ELO | final ELO | final WR |")
    emit("|---|---|---|---|---|---|---|")
    for r in rows:
        emit(f"| {r['value_coef']} | {r.get('mean_explained_var')} | {r.get('final_explained_var')} | "
             f"{r.get('mean_value_loss')} | {r.get('peak_elo')} | {r.get('final_elo')} | "
             f"{r.get('final_winrate')} |")
    emit("")
    # Pick the coef with the best (highest) mean explained variance -- the principled answer
    # to ask #3 when ELO is in the noise. Fall back gracefully if metrics are absent.
    ranked = [r for r in rows if r.get("mean_explained_var") is not None]
    if ranked:
        best = max(ranked, key=lambda r: r["mean_explained_var"])
        emit(f"**Best critic by mean explained variance: value_coef={best['value_coef']} "
             f"(EV={best['mean_explained_var']}).** If EVs are within ~0.02 of each other the "
             f"sweep is still a wash -- keep 0.5. Confirm against final WR before changing the default.\n")
    else:
        emit("**No value-head metrics in the log** -- older build or empty run; keep value_coef=0.5.\n")
    return {"value_coef_sweep": rows}


def phase_seeds(seeds, chunks, games, parallel):
    emit("## Phase: seeds (peak-ELO spread)\n")
    emit(f"{len(seeds)} seeds, {chunks} chunks x {games} games each, --parallel {parallel}, "
         f"--curriculum, patience disabled (equal budget per seed). Measures seed "
         f"sensitivity at a FIXED budget -- characterizes variance, not the full-run peak.\n")
    rows = []
    for s in seeds:
        extra = ["--chunks", str(chunks), "--chunk_games", str(games),
                 "--parallel", str(parallel), "--network", "small", "--seed_model", "",
                 "--seed", str(s), "--curriculum", "--patience", "0",
                 "--model_dir", _scratch(f"seed_{s}")]
        rc, out, err = run_train(extra)
        m = read_metrics() or {}
        rows.append({"seed": s, "rc": rc, **m})
        emit(f"- seed={s}: exit {rc} | peak_elo={m.get('peak_elo')} "
             f"final_elo={m.get('final_elo')} final_stage={m.get('final_stage')}")
        if rc != 0:
            emit("```")
            emit((err or out or "")[-1200:])
            emit("```")
    peaks = [r.get("peak_elo") for r in rows if r.get("peak_elo") is not None]
    spread = None
    if peaks:
        import statistics
        spread = {"n": len(peaks), "min": min(peaks), "max": max(peaks),
                  "mean": round(statistics.mean(peaks), 1),
                  "stdev": round(statistics.stdev(peaks), 1) if len(peaks) > 1 else 0.0,
                  "range": round(max(peaks) - min(peaks), 1)}
        emit("")
        emit(f"**Peak-ELO spread:** {spread}")
    emit("")
    return {"seed_repeats": rows, "spread": spread}


def phase_run(network, parallel, chunks, chunk_games, patience, seed):
    emit("## Phase: run (long strength run)\n")
    extra = ["--parallel", str(parallel), "--network", network, "--chunks", str(chunks),
             "--chunk_games", str(chunk_games), "--seed_model", "", "--curriculum",
             "--save_best", "--patience", str(patience), "--restore_best"]
    if seed is not None:
        extra += ["--seed", str(seed)]
    cmd_str = "python -m scripts.train_league " + " ".join(extra)
    emit(f"Command (streamed live; watch the dashboard Training tab):\n\n```\n{cmd_str}\n```\n")
    flush_report()  # persist the command before the (long) run, in case it's interrupted
    rc, _, _ = run_train(extra, capture=False)
    m = read_metrics() or {}
    emit(f"- exit {rc} | final_elo={m.get('final_elo')} peak_elo={m.get('peak_elo')} "
         f"records={m.get('records')} final_stage={m.get('final_stage')}")
    best_path = os.path.join(REPO, "models", "league_pg", "best.pt")
    emit(f"- best.pt present: {os.path.isfile(best_path)} ({best_path})")
    emit("")
    return {"run": {"rc": rc, **m, "best_pt": os.path.isfile(best_path)}}


def main():
    ap = argparse.ArgumentParser(description="Turnkey home harness for UTTT-RL.")
    ap.add_argument("--phase", default="cheap",
                    choices=["probe", "sweep", "seeds", "cheap", "run", "recompute", "all"],
                    help="cheap = probe+sweep+seeds (default); all = cheap+run; "
                         "recompute = THROUGHPUT Part C validation (parity + smoke + A/B).")
    # sweep
    ap.add_argument("--value_coefs", default="0.25,0.5,1.0")
    ap.add_argument("--sweep_chunks", type=int, default=3)
    ap.add_argument("--sweep_games", type=int, default=1000)
    ap.add_argument("--sweep_parallel", type=int, default=64)
    ap.add_argument("--sweep_seed", type=int, default=0)
    # seeds
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--seed_chunks", type=int, default=5)
    ap.add_argument("--seed_games", type=int, default=1000)
    ap.add_argument("--seed_parallel", type=int, default=64)
    # run (the long strength run -- tune to your box)
    ap.add_argument("--run_network", default="medium")
    ap.add_argument("--run_parallel", type=int, default=64,
                    help="Batch size for the long run. NOTE: RESULT_STRENGTH_RUN found 512 OOMs the "
                         "10GiB 3080 (medium) AND that big batches starve gradient updates -- 256 "
                         "stalled at stage 0 (~8 updates/chunk) while 64 (~31 updates/chunk) cleared "
                         "stage 1. So the default is 64 (learning > TF32 throughput here). Raise it "
                         "only if you've fixed the update-vs-batch tradeoff (minibatch the learn step).")
    ap.add_argument("--run_chunks", type=int, default=40)
    ap.add_argument("--run_chunk_games", type=int, default=2000)
    ap.add_argument("--run_patience", type=int, default=8)
    ap.add_argument("--run_seed", type=int, default=None)
    # recompute (THROUGHPUT Part C validation -- small net, so big batches are VRAM-safe here)
    ap.add_argument("--rc_network", default="small")
    ap.add_argument("--rc_parity_games", type=int, default=80)
    ap.add_argument("--rc_ab_chunks", type=int, default=3)
    ap.add_argument("--rc_ab_games", type=int, default=1024)
    ap.add_argument("--rc_baseline_parallel", type=int, default=64,
                    help="A/B baseline = the safe in-graph small-batch config (RESULT_STRENGTH_RUN).")
    ap.add_argument("--rc_big_parallel", type=int, default=256,
                    help="A/B recompute = a fat batch (GPU throughput) that would have starved "
                         "updates under the in-graph path; --recompute decouples the step count.")
    ap.add_argument("--rc_minibatch", type=int, default=64,
                    help="SGD minibatch for the recompute A/B leg (sets #gradient-steps).")
    ap.add_argument("--rc_seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(SCRATCH, exist_ok=True)
    value_coefs = [float(x) for x in args.value_coefs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    emit(f"# home_batch report -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"phase = `{args.phase}`\n")

    summary = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "phase": args.phase}
    do = args.phase
    try:
        if do in ("probe", "cheap", "all"):
            summary.update(phase_probe())
            flush_report()
        if do in ("sweep", "cheap", "all"):
            summary.update(phase_sweep(value_coefs, args.sweep_chunks, args.sweep_games,
                                       args.sweep_parallel, args.sweep_seed))
            flush_report()
        if do in ("seeds", "cheap", "all"):
            summary.update(phase_seeds(seeds, args.seed_chunks, args.seed_games,
                                       args.seed_parallel))
            flush_report()
        if do in ("run", "all"):
            summary.update(phase_run(args.run_network, args.run_parallel, args.run_chunks,
                                     args.run_chunk_games, args.run_patience, args.run_seed))
            flush_report()
        if do == "recompute":
            summary.update(phase_recompute(args.rc_network, args.rc_parity_games, args.rc_ab_chunks,
                                           args.rc_ab_games, args.rc_baseline_parallel,
                                           args.rc_big_parallel, args.rc_minibatch, args.rc_seed))
            flush_report()
    except KeyboardInterrupt:
        emit("\n_(interrupted)_")

    # --- machine-readable summary (paste-back gives me exact numbers) ---
    emit("## Machine summary")
    emit("```json")
    emit(json.dumps(summary, indent=2, default=str))
    emit("```")
    flush_report()


if __name__ == "__main__":
    main()
