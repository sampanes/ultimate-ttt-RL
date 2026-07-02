# Throughput benchmark harness — spec + status

**Status: IMPLEMENTED.** `scripts/bench_throughput.py` is built and py_compile/`--help`
verified on the authoring box; the timed runs need torch + GPU (home box). The two levers
this plan originally listed as "out of scope, needs new code" — opponent-side batching and
`torch.compile`/AMP — were ALSO built and are now benchmarked candidates (see §6). What
remains is running it at home and reading the report.

Run it:
```
python -m scripts.bench_throughput                     # both suites, ~5 min/candidate
python -m scripts.bench_throughput --quick             # fast smoke of the whole matrix
python -m scripts.bench_throughput --only "batch_opponents|compile|amp"  # just new levers
```
Before enabling anything the benchmark likes, gate the correctness-sensitive levers:
`python -m scripts.verify_opponent_batch_parity` (must PASS) for `--batch_opponents`, and
`python -m scripts.verify_recompute_parity` for `--recompute`. AMP/compile have no exact
oracle (AMP changes numerics) — validate convergence on a real run.

_Original plan below; file:line references were verified against `actor-critical-league`._

**Why this exists:** before committing weeks of GPU time to a long training run
(league or AlphaZero), find out which of the throughput levers already sitting in
the codebase actually move the needle on *this* hardware (RTX 3080, 10GB), so the
long run uses the best-known config instead of a guess.

---

## 1. Goal and metric

**Primary metric: end-to-end games/sec**, i.e. wall-clock time to complete a fixed
number of full training cycles (self-play generation *and* the learn step),
divided by games played. Not just self-play speed — a lever like `--recompute`
only affects the learn step, and a slow learn step eats into overall throughput
just as much as slow self-play does. Measuring end-to-end avoids missing that.

**Secondary (nice-to-have, not required for v1): GPU utilization sampling**
(`nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` polled every
~3s during each run) — tells you *why* a config is fast or slow (GPU-bound vs
Python/engine-bound), which matters for deciding what to try next, but isn't
needed to rank the candidates. Skip in v1 if it adds meaningful implementation
time; add later if the rankings raise "why" questions.

**No existing games/sec metric to reuse.** Confirmed by survey:
`scripts/train_league.py` only shows a live tqdm bar (`unit="batch"`, batches
here means one simultaneous forward-pass round across `--parallel` games, not
one completed game — do not treat tqdm's printed rate as games/sec without
converting). `scripts/train_alphazero.py` prints iteration summaries with no
throughput label at all. Older trainers (`trainer_base.py:412`, `trainer.py:170`,
etc.) do print `games/sec`, but they're not the code path used for real training
runs anymore.

---

## 2. Design: zero changes to the training scripts

**v1 requires NO modifications to `train_league.py` or `train_alphazero.py`.**
Both are treated as black boxes, run to natural completion via
`subprocess.run(...)` (never killed mid-run), timed with `time.time()` before and
after. This is deliberate: the whole point of the exercise is to protect weeks of
real training time, so the benchmark harness should carry zero risk of
introducing a regression into the scripts that will run the actual long run.

Per-candidate procedure (two phases, mirrors the existing `home_batch.py`
subprocess pattern at `scripts/home_batch.py:61-71`):

1. **Probe** (fixed tiny size, e.g. `--chunks 1 --chunk_games 20` for league,
   `--iters 1 --games_per_iter 5` for AlphaZero): time it once, get a rough
   games/sec estimate. This run's timing is discarded — it exists only to size
   step 2.
2. **Measure**: using the probe's rate, compute a game/iteration count sized so
   one repeat takes roughly 90–120s (`games ≈ max(probe_games, rate * 100)`).
   Run `--repeats` (default 3) fresh subprocesses at that size, each timed
   independently. Report mean, stdev, and n across the repeats — this is the
   "average" the request asked for, and 3×~100s keeps each candidate to roughly
   5 minutes total including the probe.

**Known limitation, state it plainly in the report rather than hiding it:** each
repeat is a fresh process, so CUDA context init + cuDNN autotune (~1-5s) is paid
on every repeat, not amortized once. Sizing each repeat to ~100s keeps this to a
few percent of the measurement; if it ever matters more than that, the fix is a
persistent long-lived subprocess with in-process interval timestamps instead of
fresh processes per repeat — flagged as a v2 idea, not needed for v1.

**Isolation from the live dashboard / any real run in progress:**
- Every subprocess call passes `--no_metrics` (confirmed to exist on both
  scripts — `train_league.py:590`, `train_alphazero.py:234`) so the benchmark
  never writes to the shared `loss_logs/metrics_log.jsonl` that the arena/AZ
  dashboards read live.
- Every subprocess call passes its own scratch `--model_dir` (mirrors
  `home_batch.py`'s `_scratch(name)` helper at `scripts/home_batch.py:127-130`)
  so checkpoints never collide with a real run's `models/league_pg` or
  `models/alphazero`.
- **Hard precondition, put it at the top of the harness's `--help` and printed
  banner: do not run this while a real long training run is active on the same
  GPU.** They'd contend for the single 3080 and invalidate both measurements.
  This is a "run it before you kick off the long run" tool, not a background
  profiler.
- `--seed 0` (train_league) is used identically across every candidate in a
  sweep, so opponent sampling / Dirichlet noise / action sampling RNG streams
  line up and don't add unrelated variance between candidates.
- `--seed_model ""` must be passed explicitly for league runs — the flag's
  default (`scripts/train_league.py:576`) points at a checkpoint path that does
  not exist in a fresh clone and will `FileNotFoundError` otherwise (a known
  gotcha, already noted in `README.md`'s quickstart per project memory).

**Failure handling:** wrap each subprocess call in try/except; a non-zero exit
(e.g. CUDA OOM, which is already confirmed to happen at `--parallel 512` on this
GPU per `PENDING.md`'s home-box notes) records `FAILED` + the last ~500 chars of
stderr in the report and moves on to the next candidate. Never let one bad
config abort the whole sweep.

---

## 3. The harness: `scripts/bench_throughput.py`

New sibling script, not a new phase bolted onto `scripts/home_batch.py`.
Reasoning (from the survey): `home_batch.py` is already 510 lines / ~35 CLI
flags / a flat if-chain with no phase registry — every existing phase there is
about *training quality* (parity, sweeps, seeds, the long run itself), and its
`read_metrics()` helper is hard-wired to the shared `loss_logs/metrics_log.jsonl`
path, which this task deliberately avoids touching. A dedicated script keeps the
throughput concern decoupled and the file size sane. Reuse `home_batch.py`'s
small helpers by import if convenient (`_scratch`, the subprocess pattern), but
don't be precious about it — duplicating ~15 lines is cheaper than coupling two
unrelated concerns.

**CLI:**
```
python -m scripts.bench_throughput --suite league|alphazero|both  (default: both)
                                    --minutes 5     (soft per-candidate time target)
                                    --repeats 3
                                    --out bench_throughput_report.md
```

**Output:** ONE markdown report (gitignored, same convention as
`home_batch_report.md` — add `bench_throughput_report.md` to `.gitignore` next to
the existing `home_batch_report.md` entry at `.gitignore:43`). Contents:
- Env probe header: GPU name, VRAM, torch/CUDA version, whether the C++ engine
  loaded (`engine.game._CPP_ENGINE`) — same probe pattern as
  `scripts/home_batch.py:133-164`, can literally reuse `_ENV_PROBE`.
- One ranked table per suite (games/sec descending), columns: config label, mean
  games/sec, stdev, n repeats, status (OK/FAILED).
- A one-line plain-language recommendation per suite at the bottom. If the top
  two results are within ~5% of each other, say so explicitly and prefer
  whichever has simpler downstream implications (e.g. don't recommend
  `--recompute` over plain if they're a wash, since plain has zero extra moving
  parts).

---

## 4. Candidate list — League suite (`train_league.py`)

Fixed baseline settings for every row unless the column says otherwise:
`--no_metrics --seed 0 --seed_model "" --patience 0`.

| # | Candidate | Flags varied | Notes |
|---|---|---|---|
| 1 | Sequential baseline | `--parallel 0 --network medium` | The literal default; establishes the floor. Known slow. |
| 2 | Parallel sweep | `--parallel {16,32,64,128,256} --network medium` | 512 is a **known OOM** on this 10GB card at `medium` (confirmed in project history) — skip it, don't waste a slot finding out again. |
| 3 | Recompute sweep | `--parallel <best from #2> --recompute --minibatch_size {0,32,64,128}` | `--recompute` defaults OFF; parity was already verified separately (`verify_recompute_parity.py`) — this only measures speed, not correctness. |
| 4 | Network size | `--parallel <best from #2> --network {small,medium,large}` | Bigger net = slower forward/backward per step; find the actual wall-clock cost, not just "large is slower" qualitatively. |
| 5 (optional/diagnostic) | C++ engine on/off | see §6 | Low actionability — see note below. |

Baseline "best from #2" is carried forward into #3 and #4 so those sweeps aren't
also re-discovering the parallel optimum; run #2 first, read its winner, then
launch #3/#4.

## 5. Candidate list — AlphaZero suite (`train_alphazero.py`)

Fixed baseline settings: `--no_metrics --network medium --n_sims 100`.

| # | Candidate | Flags varied | Notes |
|---|---|---|---|
| 1 | Serial baseline | `--wave_size 1` | Confirmed default; byte-identical to pre-batching behavior. |
| 2 | Wave size sweep | `--wave_size {1,4,8,16,32,64}` | This is the AZ-specific lever — MCTS leaf batching (`agents/mcts.py:32-39,153-225`) is already fully implemented, just never benchmarked for wall-clock. |
| 3 | Sim budget | `--n_sims {100,200,400}` at best wave_size from #2 | Not purely a throughput knob (more sims = stronger play too), but the report should show the games/sec cost curve so the strength/speed tradeoff is an informed choice, not a guess. |
| 4 | Network size | `--network {small,medium,large}` at best settings from #2/#3 | Same rationale as league #4. |

---

## 6a. Levers that WERE out of scope and are now BUILT

The original plan (§6b below) listed opponent batching and torch.compile/AMP as follow-up
code, not v1 candidates. They have since been implemented and are benchmarked candidates:

- **Opponent batching** — `--batch_opponents` on `train_league.py` (commit `ee60478`).
  `ParallelGameRunner` groups NN opponents by weight identity and runs one batched argmax
  forward per group instead of one `select_move` per opponent move. Every NN opponent is
  deterministic argmax at eval time, so it's outcome-identical to the per-slot loop —
  certified by `scripts/verify_opponent_batch_parity.py` (GATE before enabling). This is the
  highest-value lever: it removes an unbatched, Python-driven forward per opponent move, and
  per the project's own finding the Python loop (not the GPU) is the bottleneck.
- **`--compile` / `--amp`** on `train_league.py` (commit `5b45fa4`). `--compile` compiles a
  separate `forward_both` callable (state_dict untouched → clone/save/load/ONNX-export safe).
  `--amp` is fp16 autocast + GradScaler. Both default OFF, both experimental — AMP changes
  numerics (no exact oracle; validate convergence), and both optimize the GPU forward, which
  may be marginal on a tiny net. That's what the benchmark is for.

## 6b. Still out of scope (original notes)

- **C++ engine vs Python engine A/B.** Per survey: the C++ engine (`engine/cpp/`)
  is not built on this authoring box at all (`engine/cpp/build/` doesn't exist
  here), but per project history it IS built and loaded on the home box already
  — meaning any real run there already benefits from it by default, and there's
  no code path to intentionally disable it (no env var / flag exists to force
  the Python fallback; would need a small addition to `engine/game.py`'s import
  guard, e.g. `if not os.environ.get("UTTT_FORCE_PYTHON_ENGINE"): try: import
  uttt_engine...`). Low actionability: nobody is going to choose to run without
  the C++ engine in production. Worth measuring once out of curiosity ("how much
  is it buying us") but not worth the sweep's time budget in v1 — do it last, if
  at all, and only as a single before/after data point rather than a full
  repeats-and-average measurement.
- **Batching the opponent's forward pass in `ParallelGameRunner`.** ~~Out of scope~~
  → **BUILT, see §6a** (`--batch_opponents`, parity-gated). Left here as a record of the
  original scoping call.
- **`torch.compile()` and AMP.** ~~Out of scope~~ → **BUILT, see §6a** (`--compile`,
  `--amp`, both opt-in/experimental). Left here as a record of the original scoping call.

## 7. Rough time budget

League suite: ~9 candidates (1 + 5 + 4 + 3, after picking the #2 winner to reuse
in #3/#4) × ~5 min ≈ 45 min. AlphaZero suite: ~13 candidates × ~5 min ≈ 65 min.
Both suites together: under 2 hours unattended, run overnight or alongside
something else — this is meant to be a "kick it off, walk away, read the report"
tool, matching the `home_batch.py` precedent.

## 8. Open questions for the implementer

1. Exact `--chunk_games` / `--games_per_iter` sizing formula in the probe→measure
   step (the `rate * 100` target above is a starting guess, not a tuned
   constant) — fine to hand-tune once against real hardware numbers rather than
   over-engineer a dynamic calibrator.
2. Whether to add the GPU-utilization sampling (§1 secondary metric) in v1 or
   defer it — recommend deferring unless it's cheap (a background thread calling
   `nvidia-smi` in a loop is genuinely simple, so there's no strong reason not
   to include it if time allows).
3. Whether `--repeats 3` is worth the wall-clock cost vs `--repeats 1` with a
   single longer measurement window — 3 short repeats give a variance signal at
   roughly the same total time cost as 1 long one; recommend keeping repeats
   unless the implementer finds variance is negligible after the first sweep,
   in which case drop to 1 for later re-runs.
