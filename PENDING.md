# Pending -- what still needs to happen

Machine-to-machine handoff. History is in git. This is only the open queue.

---

## S0+S1 runbook -- TRAINING BOX, run next (authored 2026-07-13)

S0 (gen/train time split in metrics) and S1 (gregory d2 training slice,
opt-in) are authored and py_compile-verified on the authoring box; runtime
verification is this runbook. Everything below is cmd from the repo root.

**1. Pull + regression suites** (CPU, fast; the goat run can stay live):

    git pull
    .venv\Scripts\python -m scripts.test_expert_iter
    .venv\Scripts\python -m agents.test_mcts

Expect 10/10 (was 9: new slice-layout test) and 10/10.

**2. S1 pre-step: baseline the raw net vs gregory d2** (CPU-only, read-only,
~1 min, does NOT require stopping the run):

    .venv\Scripts\python -m scripts.baseline_vs_gregory --depths 2,3

The script prints the verdict itself (STRENGTH_NEXT S1 threshold 0.55).
Sanity check: the d3 number should land near the latest `promote_gregory`
in the metrics log (~0.10-0.16 as of gen-7) -- same seed, same instrument.
- **Verdict "enable"** (expected): go to step 3.
- **Verdict "STOP"** (raw net >= 0.55 vs d2): do NOT enable; paste the
  numbers back and the authoring box designs the d3-in-mix + d4-ruler
  variant. Still do step 3's restart WITHOUT the new flags so S0 logging
  goes live.

**3. One graceful restart to activate S0 (+S1 if step 2 said enable).**
Edit `start_goat.bat` line 20 and append the mix flags after `--resume`
(goat_supervisor passes all args through to expert_iter unchanged):

    ... -m scripts.goat_supervisor --resume --greg_mix 0.10 --opp_mix 0.30 --rnd_mix 0.10 ...

Then run `start_goat.bat` (it stops the old instance gracefully and
resumes; buffer/state/baselines all survive -- the promotion panel size is
unchanged, so no baseline rebase fires). Startup misuse is guarded: the
script hard-errors if greg_mix_depth >= gregory_depth or the mixes sum > 1.

**4. Expectations (STRENGTH_NEXT rule 3 -- write them down, don't panic):**
head/winblock may wobble for a few checks after the mix change; judge only
the fixed 300-game panels. Success metric = the gregory(d3) panel slope
improving from +1-2 pts/gen toward +4, NOT promotion cadence. The gregory
no-regression gate is already armed (best 0.143), so a real regression
still blocks promotion.

**5. Paste back after ~1 full generation on the new mix** (RESULT_S1.md or
chat), the useful info in priority order:

  a. The full `baseline_vs_gregory` output (pre-step numbers + verdict).
  b. The `promote_gregory` series before vs after the change -- every
     promotion-check line carries it:

         findstr promote_gregory loss_logs\metrics_log.jsonl

  c. Proof the slice is live: any metrics line showing `greg_games` > 0
     (16-game blocks at 0.10 mix -> expect 1-3 most blocks, some 0s).
  d. The S0 split -- mean gen vs train seconds (this is the S4
     go/no-go data):

         .venv\Scripts\python -c "import json;rows=[json.loads(l) for l in open('loss_logs/metrics_log.jsonl',encoding='utf-8')];g=[(r['gen_secs'],r['train_secs']) for r in rows if 'gen_secs' in r];n=len(g);print(f'blocks={n} gen={sum(a for a,_ in g)/n:.1f}s train={sum(b for _,b in g)/n:.1f}s')"

  e. Any promotion/no-promotion console lines from the segment (the
     gate-fail reasons matter as much as the passes).

---

## S-queue -- long-horizon strength backlog (OPEN, authoring box)

Full analysis and reasoning: `STRENGTH_NEXT.md` (2026-07-13, written on the
training box, analysis only -- no training code touched there). Headline:
`GregoryAgent(depth=2)` joins the expert_iter opponent mix while d3 stays the
untouched honest ruler. Also queued there: root-value capture for blended
value targets, playout-cap randomization, cross-game batched generation
(generation GPU batches are only 12 positions today), gen/train timing in
metrics, auto teacher snapshots per gen, browser MCTS "Brutal" mode, and the
never-run champion(+search)-vs-gregory measurement. One change per run
segment, judged by the fixed 300-game panels only. Context: certification of
gen-6+ is deliberately deferred (2026-07-12) until the compounded margin
justifies interrupting the run.

Status 2026-07-13: **S0 DONE, S1 AUTHORED (opt-in)** -- see the runbook
above. S2 (blended value targets) is the next authoring-box item, judged
against S1's trajectory; S3/S4 wait on the S0 timing data.

---

## M5.5 -- DONE 2026-07-11: gen-5 certified as oracle champion + shipped

Gen 5 promoted 2026-07-11 (58% h2h | winblock 35% | random 82% | gregory 14%)
and the queue below executed same-day. Verdict: **PROMOTED** -- direct
head-to-head 0.698 vs `arena:22@hof` (300 games), M2 raw/mcts_25/mcts_100 all
up (0.856 best mode), tactical + GOLD-suite ties. Full numbers:
`RESULT_M2_5.md`; `CHAMPIONS.md` oracle row updated (rule amended: direct h2h
tie-breaker added, self-oracle retired as cross-net criterion). Shipped:
`docs/models/champion.onnx` + play-page opt-in picker (pocket stays default);
`turn_based_games` UTTT solo bot upgraded (champion via cross-origin fetch,
win/block fallback). Goat run resumed into gen-6 (~2.5h downtime).

Original queue text kept below for reference.

**Trigger: the next goat-train PROMOTION line (gen 5).** Executed on the home box.
Plumbing already committed and smoke-tested: `benchmarks/goat_certified.json`
(candidate manifest), `value_tanh` manifest support in `scripts/benchmark_suite.py`
(threads through `grade_agent` and `export_onnx` via the shared candidate builder).

1. **Stop the run** (`stop_goat.bat`) so certification gets the GPU to itself.
2. **Snapshot**: copy `models/expert_iter_v2/teacher.pt` ->
   `models/expert_iter_v2/certified/candidate.pt`; record gen + SHA-256.
   (Weights stay gitignored; the SHA is the identity of record, as with the
   arena checkpoints.)
3. **M2 panel** (same instrument as `RESULT_M2.md`):
   ```
   set CUBLAS_WORKSPACE_CONFIG=:4096:8
   python -m scripts.benchmark_suite --candidate benchmarks/goat_certified.json ^
     --anchors lottery,nn_big8,winblock,center,first ^
     --candidate-sims 0,25,100 --oracle-sims 400 ^
     --openings standard --out results/goat-gen5
   ```
4. **GOLD blunder rate**:
   `python -m scripts.grade_agent --suite gold_endgame_suite.json --candidate benchmarks/goat_certified.json`
5. **Restart the run** (`start_goat.bat`, resumes); the ship steps below are CPU-only.
6. **Champion test** vs the incumbent oracle `arena:22@hof` (numbers from
   `RESULT_M2.md` / `CHAMPIONS.md`): tactical aggregate > 0.844, holds its own
   400-sim oracle >= 0.500, GOLD blunder < 6.26%. Write `RESULT_M2_5.md`; if it
   wins, update the `CHAMPIONS.md` oracle row (new SHA, which metric improved).
7. **If champion -- ship page 1 (this repo's Pages)**: export fp32 ONNX via the
   manifest (`python -m scripts.export_onnx --candidate benchmarks/goat_certified.json
   --out-dir <staging>`), deploy as `docs/models/champion.onnx` +
   `docs/models/champion_config.json` (do NOT overwrite the pocket
   `model.onnx`/`model_config.json`); add an opt-in model picker to the play page
   (default stays the 5 MB pocket net; champion is a ~27 MB opt-in download).
   Spot-check torch-vs-ONNX policy parity as in M3.
8. **If champion -- ship page 2 (`turn_based_games` repo)**: upgrade the UTTT solo
   bot. ort-web from CDN; fetch `champion_config.json`/`champion.onnx`
   cross-origin from `https://sampanes.github.io/ultimate-ttt-RL/models/`
   (GitHub Pages sends `Access-Control-Allow-Origin: *`), so the model has one
   source of truth. Async brain behind the existing synchronous
   `computerMove(state, slot)` contract: return the cached result when ready,
   `null` while thinking (the solo poll loop already re-arms on null); existing
   heuristic keeps playing until the model finishes loading or if offline.
   No framework changes to `solo.js`/`app.js`.

---

## Throughput benchmark harness -- DONE + FIXES BAKED (home box, 2026-07-02/03)

**All three steps ran; full results + bug reports in `RESULT_PERF_BENCH.md`. The
authoring-box follow-ups were then done ON the home box (user-approved one-off):**
- `--batch_opponents`: parity PASS + 1.11x A/B (1.19x in-bench) -> **BAKED default ON**
  (`--no-batch_opponents` to disable)
- `--wave_size`: ~20x AZ self-play at 64 vs 1 (2.3 vs 0.1 games/s) -> **BAKED 64**; not
  yet saturated, consider benching 128
- `--parallel`: **BAKED default 64** (hardware-confirmed; 256 starves updates)
- `--compile`: TritonMissing on Windows -> enable_compile() now checks Triton up front
  and DEGRADES GRACEFULLY (eager forward + warning, run continues). Still default OFF.
- `--amp`: dtype crash FIXED (loss now built in fp32 across all three learn paths --
  autocast graph tensors are fp16, mse_loss needs matching dtypes at backward; .float()
  keeps the graph, no-op when AMP off). Re-gated: **0.99x, no speed win** (tiny net, GPU
  not the bottleneck) with convergence fine -> stays default OFF, re-gate on new hardware.
- `home_batch --phase perf`: A/B rows now pin every lever explicitly (an empty baseline
  would inherit the new batch_opponents default and compare the lever against itself).
- Both parity oracles re-run PASS after the edits (recompute worst delta 2.6e-08;
  opponent-batch 80/80 byte-identical).

Original queue text kept below for reference.

---

## Throughput benchmark harness -- IMPLEMENTED, needs a home-box run

`scripts/bench_throughput.py` is built (see `BENCH_THROUGHPUT_PLAN.md` for the full spec).
One command, one gitignored report; runs `train_league.py`/`train_alphazero.py` as timed
black-box subprocesses (zero changes to them) with `--no_metrics` + scratch model dirs so
it never disturbs the dashboard or a real run. Candidates: `--parallel` sweep, `--recompute`
x `--minibatch_size`, `--network` size, AlphaZero `--wave_size`/`--n_sims`, PLUS the two new
levers built this pass -- `--batch_opponents` and `--compile`/`--amp`.

**New levers built (default OFF, benchmark them, then gate before trusting in a long run):**
- `--batch_opponents` (train_league, commit `ee60478`): batches the opponent forward passes
  in `ParallelGameRunner` (was unbatched per-slot). Highest-value lever -- the Python loop is
  the bottleneck, not the GPU. GATE: `python -m scripts.verify_opponent_batch_parity` (proves
  it's byte-identical to the per-slot loop) must PASS.
- `--compile` / `--amp` (train_league, commit `5b45fa4`): torch.compile a separate
  forward_both callable (state_dict/clone/ONNX untouched) / fp16 autocast + GradScaler.
  Experimental; AMP has no exact oracle (changes numerics) -- validate convergence.

**Home-box steps (in order):**
```
python -m scripts.bench_throughput --quick              # fast smoke of the whole matrix
python -m scripts.bench_throughput                      # the real ~5-min/candidate run
python -m scripts.home_batch --phase perf               # A/B: batch_opponents/amp/compile
```
`bench_throughput` measures raw speed across the whole candidate matrix. `home_batch --phase perf`
is the turnkey **A/B for the three throughput levers** -- it runs the batch_opponents parity gate,
then plays the same seed/budget with each lever on vs off, reporting BOTH games/sec AND convergence
(peak ELO / WR / EV) in one `home_batch_report.md`. Use it to settle the AMP/compile question that
has no static oracle ("faster, and does it still learn?"). Read both reports, pick the fastest safe
config, use it for the long run, paste the reports back.

---

## Home-box runs (needs torch + `.pt` models)

Runs 1-4 and 6 done on the RTX 3080, 2026-06-30 -- results in `RESULT_HOME_QUEUE.md`.
Run 5 (AlphaZero validation) executes separately. Commands kept for reproducibility.

### 1. MCTS unit tests -- lock the sign  [DONE: 7/7 PASS]
```
python -m agents.test_mcts
```
Value-sign convention locked.

### 2. Recompute parity gate  [DONE: parity PASS, but keep opt-in]
```
python -m scripts.verify_recompute_parity
python -m scripts.home_batch --phase recompute
```
Parity PASSED (safe). Short A/B did NOT clear the "not worse" bar (recompute-bigbatch
worse on EV/WR and slower, at --parallel 256 which starves updates). Do NOT flip
`--recompute` default ON yet -- needs a longer/multi-seed A/B via `--minibatch_size`.

### 3. Value-coef sweep  [DONE: wash, keep 0.5]
```
python -m scripts.home_batch --phase sweep
```
All EVs within ~0.004 (inside the ~0.02 wash threshold) and near zero. Keep
`--value_coef 0.5`; a discriminating sweep needs runs long enough for EV to lift off zero.

### 4. Honest absolute rating for best.pt  [DONE: 0/40 at all depths]
```
python -m scripts.benchmark_vs_mcts \
  --checkpoint models/league_pg/best.pt --network medium \
  --games 40 --oracle_sims 800

python -m scripts.benchmark_vs_mcts \
  --checkpoint models/league_pg/best.pt --network medium \
  --sim_ladder 100,400,1600
```
Raw net loses 100% to MCTS over its own weights, crossover below 100 sims. Closed-loop
ELO 4437 is confirmed meaningless; search is the dominant untapped lever (M4 backing).

### 5. AlphaZero validation run  [DONE: loop HEALTHY -- GO; 2 bugs, see below]
```
python -m scripts.train_alphazero \
  --checkpoint models/league_pg/best.pt --network medium \
  --value_tanh --n_sims 200 --games_per_iter 50 --iters 20
```
20 iters completed (~4.4h). Loss 6.54->1.66, value loss 0.45->0.11 (tanh head stable,
no NaN) -> GO for a long AZ run. Caveat: seeding untanhed best.pt with --value_tanh is
the docstring's "produces garbage" combo (wr_vs_rand opens at 42%, heals to 55%); a real
run should start fresh or from a tanh-trained checkpoint. Full detail in RESULT_HOME_QUEUE.md.

**Two authoring-box bugs this run hit (fixed via runtime shim to complete tonight):**
1. `train_alphazero.py` calls `agent.save_model(path)` -- no such method on NeuralNetAgentPG;
   it is `save(path, verbose=True)`. Crashes at first checkpoint (lines 332, 343, 367).
   Fix: rename those three calls to `agent.save(...)`.
2. `train_alphazero.py:354` uses a literal em-dash as the wr_str fallback -- non-ASCII,
   renders as a garbage glyph on the Windows console. Fix: ASCII fallback (`"n/a"`).

### 6. GOLD endgame suite + grading  [DONE: 16.37% neutral blunder rate]
```
# Build once (no torch needed on authoring box, but needs the engine):
python -m scripts.build_endgame_suite --out suite.json --n_games 1000 --max_empty 15

# Grade best.pt against it:
python -m scripts.grade_agent \
  --suite suite.json --checkpoint models/league_pg/best.pt --network medium
```
375-position neutral suite; best.pt blunders 16.37% (55/336 gradable). First
opponent-neutral figure. `suite.json` is regenerable (seed 0), not committed -- adopt
it as a fixture if you want a standing GOLD set.

---

## Pending flag hardening (bake after home data confirms)

| Flag | Script | Gate | Action when done |
|---|---|---|---|
| `--recompute` | train_league | parity PASS + A/B not worse | default ON, keep `--no-recompute` |
| `--minibatch_size` | train_league | A/B confirms best value | surface as `_DEFAULT_MINIBATCH = N` |
| `--value_coef` | train_league | EV sweep picks winner | surface as `_DEFAULT_VALUE_COEF = N` |
| `--value_tanh` | train_alphazero | AZ run validates | default ON for AZ script, keep `--no-value_tanh` |
| `--wave_size` | train_alphazero / benchmark_vs_mcts | [DONE 2026-07-03] ~20x at 64, monotonic | BAKED default 64 |
| `--batch_opponents` | train_league | [DONE 2026-07-03] parity PASS + 1.11x A/B | BAKED default ON, `--no-batch_opponents` kept |
| `--compile` | train_league | [CLOSED 2026-07-03] Triton has no Windows build | stays OFF; enable_compile() now degrades gracefully instead of crashing |
| `--amp` | train_league | [DONE 2026-07-03] dtype bug fixed; A/B 0.99x, convergence fine | stays OFF (no speed win on RTX 3080); re-gate on new hardware |

Low-priority (all settled as of 2026-07-03):
- `--lr` defaults were already 1e-4 (train_league) / 1e-3 (train_alphazero)
- `--parallel` default 0 -> 64 in train_league [BAKED 2026-07-03]
- `--keep_versions` default was already 5

---

## Pending code (author on this box, torch-free)

### MCTS batched leaf eval  [DONE -- already shipped]
`agents/mcts.py` already has full `wave_size` support (virtual loss, leaf dedup,
one batched forward pass per wave) and `scripts/train_alphazero.py` already
exposes `--wave_size N`. No action needed; PENDING previously had two stale
entries for this.

### AlphaZero richer logging  [DONE]
`scripts/trainer_base.append_metrics` gained `t`, `policy_loss`, `games_total`,
`buffer` (additive/optional, byte-identical for legacy callers). Wired into
`scripts/train_alphazero.py`'s per-iteration `append_metrics(...)` call:
`t=time.time()`, `policy_loss=avg_pol`, `games_total=(iteration+1)*args.games_per_iter`,
`buffer=len(buffer)`. py_compile clean; runtime display verification is a home-box
step (start a short AZ run, confirm the dashboard shows elapsed/throughput/games/buffer
instead of "--").

### Throughput levers: opponent batching + AMP/compile  [DONE -- author box, home-gated]
`--batch_opponents` (ParallelGameRunner groups NN opponents by weight, one batched
argmax forward per group; parity oracle `verify_opponent_batch_parity.py`), plus opt-in
`--amp`/`--compile` on train_league. All default OFF (byte-identical). `batch_select_moves_eval`
added to the three NN agent classes. Commits `ee60478` (batching) + `5b45fa4` (amp/compile).
Benchmark + gate at home (see the throughput-harness section above), then bake defaults.

### Gregory curriculum integration
Wire `GregoryAgent` into `league_manager.py` stage 5-6 as an external anchor.
Goal: break the closed-loop stage-6 pool with a gene-pool-independent opponent.
File: `arena/league_manager.py` around line 217+ (stage-weighted opponent mix).

### GUI stat cards
Surface `total_games` and `best_elo` from `arena_state.json` as stat cards in the dashboard.
Files: `gui/arena/templates/index.html`, `gui/arena/arena.js`.
See `gui/BACKLOG.md` for full backlog.

### Part B: training throttle/pause knob
File-based `loss_logs/control.json` -> trainer polls between games and sleeps to target rate.
Zero = pause (holds in-memory state, resume instant).
Files: `scripts/train_league.py` (poll loop), `gui/arena/templates/index.html` (Training tab button).

---

## M3 -- next milestone (torch + home box)

**Authoring-box work DONE:**
- `scripts/export_onnx.py` -- exports arena:21@hof to ONNX + dynamic int8 (home runs it)
- `scripts/gen_golden_vectors.py` -- generates `docs/play/golden_vectors.json` from Python engine
- `docs/play/golden_vectors.json` -- 50-game, 2957-ply correctness fixture (seed 0)
- `docs/play/test_engine.html` -- browser golden-vector test runner (open to verify JS engine)
- `docs/play/uttt_engine.js` -- full game engine port (home box wrote)
- `docs/play/agent.js` -- ONNX inference wrapper + PUCT MCTS (Hard=50 sims, Easy/Medium=raw policy)
- `docs/play/index.html` + `docs/index.html` -- complete play page + landing page (home box wrote)

**Home-box steps -- DONE (2026-07-01):**
1. [DONE] Run export: `python -m scripts.export_onnx --candidate arena:21@hof --quantize`
   -> fp32 5035 KB (intermediate, gitignored), int8 1283 KB committed as docs/models/model_int8.onnx
2. [DONE] Committed as bba60f6 (feat(m3): add int8 ONNX pocket model)
3. [DONE] Parity: top-1 move matches torch in all sampled positions; value error < 1.1%
4. [DONE (partial)] Bundle bytes: 1283 KB int8 (M3 gate cleared). Python move latency 0.77ms CPU
   (from RESULT_M3_PREP.md). Cold download + browser memory + phone need browser test.
5. [DONE] gen_golden_vectors ran on home box with C++ engine -- identical to authoring-box
   committed JSON (no diff). Golden vector fixture is authoritative.
6. [DONE] GitHub Pages enabled and live -- confirmed tested by user directly on GitHub.

Live at `https://sampanes.github.io/ultimate-ttt-RL/play/` (and `/play/test_engine.html`
for the JS engine golden-vector suite).

**M3 exit gate: CLEARED.** Static page plays a complete legal game offline, reports model
hash, passes golden-vector suite, MCTS Hard mode responds in <1s.

See `SHIP_PLAN.md` M3 section for full exit gate.

## M4 -- AlphaZero long run (home box)

**M4a (2026-07-03): plateaued, archived to `models/alphazero_m4_flat/`.**
15h / 77k games / ~1,500 iters with every yardstick flat (wr vs random pinned 54-57%,
past-self gauntlet pinned 50%, win/block bot winning 95%+ vs the raw net).
Diagnostic (CPU, vs version_1545): net policy entropy ~= uniform at every ply, BUT
policy CE ~= target entropy -- the net fit its targets near-optimally; the MCTS visit
targets themselves were near-uniform. Self-sealing loop: weak value head -> unfocused
search -> soft targets -> uniform policy. Value head also miscalibrated (-0.98 on a
bland random position).

**M4b (2026-07-04): fresh run with loop-breaking changes (all in train_alphazero.py):**
- `--tactics` (default ON): ultimate win-in-1 -> one-hot target, no search;
  moves allowing an immediate opponent game-win zeroed from targets (engine/tactics
  ground truth; mini-board tactics deliberately NOT forced, per tactics.py docstring)
- `--opp_mix 0.30`: slice of games vs diverse opponents (past-self pool refreshed
  every 25 iters cap 10 / WinBlockAgent / random); only net positions recorded.
  Rationale: league-style enemy diversity; twin self-play was draw-heavy + narrow.
- defaults retuned: dir_eps 0.25->0.15, temperature_moves 20->10, value_tanh ON
- eval vs random now scores draws as 0.5; per-iter self-play draw rate logged (sp_draws)
- also new since 7/3: wall-clock gauntlet eval (5 min: day-one anchor / past self /
  win-block bot / MCTS-edge probe) + `--resume` (run_state.json + resume.pt)
- launched: `--network medium --n_sims 300 --games_per_iter 50 --iters 0`

Watch on the dashboard: wr_heur (win/block bot) rising is the primary signal the
tactical blindness is fixed; past-self off 50% means real iteration-over-iteration
growth; sp_draws falling means games are decisive enough to carry value signal.

**M4b addendum (2026-07-04, mid-run restart at iter 63 via --resume):** new per-iter
metrics logged so the plateau's smoking gun is visible live, plus dashboard support
(quality chart + kill-shot/sharpness cards):
- `pi_ent`: mean entropy of the iteration's policy targets (M4a sat pinned near
  ln(81)=4.39; falling = search is decisive). Dashboard shows "target sharpness"
  = 1 - pi_ent/ln(81).
- `tac_w` / `tac_d`: win-in-1 shortcuts taken and losing-move filters applied, per game
- `avg_len`: mean game length in plies
- `winrate` is now null (not 0.0) on non-eval iterations
Graceful-stop note: taskkill /F skips the finally-block resume save; send Ctrl+C
(GenerateConsoleCtrlEvent via AttachConsole) so resume.pt (optimizer + buffer) lands.
(`scripts/send_ctrl_c.py` is now a repo utility; stop_goat.bat wraps it.)

**M4 POSTMORTEM (2026-07-04): run killed; root cause found -- TWO MCTS WAVE BUGS.**
M4b's outsider metrics stayed flat and raw play degraded BELOW the day-one anchor
(0.325). Discriminator experiment: with search equalized, current = day-one (0.500)
-- the visit targets themselves were poison. Cause (both in agents/mcts.py wave path,
both FIXED + locked by agents/test_mcts.py, now 10/10):
1. Virtual-loss SIGN inverted: this tree stores W from the child's to-play
   perspective and selection scores -c.Q(), so VL must RAISE W. `W -= VL` turned
   virtual loss into virtual WIN -> whole waves collapsed onto one line.
   Evidence: MCTS(64 sims, wave 64) over league best.pt scored 0.000 (!) vs the raw
   net it wrapped; after the sign fix + clamp, 0.800-0.925.
2. No waves-per-search floor: leaf expansion is deferred to the end of each wave,
   so the tree deepens by ONE PLY PER WAVE. wave_size ~ n_sims = a one-ply breadth
   probe, not search. MCTS.search now clamps wave to n_sims // 16 (_MIN_WAVES).
   Sweep (edge vs raw net, post-sign-fix): 38 waves 0.95, 16 waves 0.80-0.925,
   10 waves 0.70-0.80, 5 waves 0.375-0.80, 1 wave 0.000.
Every M4 self-play game and gauntlet probe ran wave=64; the 6/30 "MCTS crushes
best.pt" benchmark predates wave batching (wave=1) -- that is why it was healthy.
NOTE: bench_throughput's "wave 64 = 2.3 games/s" was measuring this degenerate
shallow search; real AZ throughput after the clamp is lower. Re-bench if it matters.
train_alphazero now HALTS on mcts_edge < 0.5 twice in a row (search invariant).
The M4b net was NOT wasted: post-fix its edge is 0.95 -- weights in
models/alphazero (version_463 + resume.pt) if ever wanted.

---

## M5 -- Expert iteration v2, "goat-train" (ready 2026-07-05)

The first expert-iteration run is invalid for candidate selection. After 9,312
games and 26 promotions, its last-ten-gate mean against WinBlock was only 0.116.
The adversarial review and exact evidence are in
`RESULT_ADVERSARIAL_REVIEW_2026-07-05.md`.

The corrected `scripts/expert_iter.py` now:
- starts from the independently selected Arena-22 HOF architecture/checkpoint,
  and initializes the tanh student from those policy weights instead of random;
- generates 35% of games against WinBlock to cover the measured blind spot;
- injects local mini-board win/block targets in that WinBlock slice only
  (`--mini_tactic_opp`, default on), because second-pass review showed Arena-22
  MCTS still misses that exact heuristic pattern;
- masks policy loss exactly as inference does and applies exact D4 symmetry
  augmentation;
- promotes on raw-inference strength over fixed color-swapped openings, plus
  absolute WinBlock improvement and a random-opponent regression guard;
- requires 1,000 current-teacher games between promotions, clears stale replay
  on promotion, and reloads only current-generation shards after resume.

`start_goat.bat` uses the new default `models/expert_iter_v2`; the old
`models/expert_iter` evidence is preserved and will not be resumed accidentally.
The milestone is no longer "promotion count." It is sustained raw WinBlock and
fixed-panel improvement, followed by the full M1 benchmark suite.
