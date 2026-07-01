# Pending — what still needs to happen

Machine-to-machine handoff. History is in git. This is only the open queue.

---

## Home-box runs (needs torch + `.pt` models)

Runs 1-4 and 6 done on the RTX 3080, 2026-06-30 -- results in `RESULT_HOME_QUEUE.md`.
Run 5 (AlphaZero validation) executes separately. Commands kept for reproducibility.

### 1. MCTS unit tests — lock the sign  [DONE: 7/7 PASS]
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
| `--wave_size` | train_alphazero / benchmark_vs_mcts | benchmark confirms sweet spot | surface as `_WAVE_SIZE = N` |

Low-priority (ready now, no gate needed):
- `--lr 1e-4` → `_DEFAULT_LR = 1e-4` in train_league (and `1e-3` in train_alphazero)
- `--parallel 0` default → `64` in train_league (hardware-confirmed best for RTX 3080)
- `--keep_versions 5` → `_DEFAULT_KEEP_VERSIONS = 5`

---

## Pending code (author on this box, torch-free)

### Gregory curriculum integration
Wire `GregoryAgent` into `league_manager.py` stage 5-6 as an external anchor.
Goal: break the closed-loop stage-6 pool with a gene-pool-independent opponent.
File: `arena/league_manager.py` around line 217+ (stage-weighted opponent mix).

### GUI stat cards
Surface `total_games` and `best_elo` from `arena_state.json` as stat cards in the dashboard.
Files: `gui/arena/templates/index.html`, `gui/arena/arena.js`.
See `gui/BACKLOG.md` for full backlog.

### MCTS batched leaf eval (Phase 6 prerequisite)
Collect N leaves, run one NN forward, expand all. ~5-10x speedup for deep-MCTS oracle runs and Phase 6 architecture search.
File: `agents/mcts.py` — add `wave_size` parameter to `MCTS.search()`.

### Part B: training throttle/pause knob
File-based `loss_logs/control.json` → trainer polls between games and sleeps to target rate.
Zero = pause (holds in-memory state, resume instant).
Files: `scripts/train_league.py` (poll loop), `gui/arena/templates/index.html` (Training tab button).

---

## M3 — next milestone (torch + home box)

**Authoring-box work DONE:**
- `scripts/export_onnx.py` — exports arena:21@hof to ONNX + dynamic int8 (home runs it)
- `scripts/gen_golden_vectors.py` — generates `docs/play/golden_vectors.json` from Python engine
- `docs/play/golden_vectors.json` — 50-game, 2957-ply correctness fixture (seed 0)
- `docs/play/test_engine.html` — browser golden-vector test runner (open to verify JS engine)
- `docs/play/uttt_engine.js` — full game engine port (home box wrote)
- `docs/play/agent.js` — ONNX inference wrapper + PUCT MCTS (Hard=50 sims, Easy/Medium=raw policy)
- `docs/play/index.html` + `docs/index.html` — complete play page + landing page (home box wrote)

**Home-box steps remaining:**
1. Run export: `python -m scripts.export_onnx --candidate arena:21@hof --quantize`
   → writes `docs/models/model_int8.onnx` + updates `model_config.json`
2. Commit model: `git add docs/models/ && git commit -m 'feat(m3): add int8 ONNX pocket model'`
3. Verify parity: re-run M2 benchmark on quantized model (need parity + KL check from export output)
4. Measure: cold download time, bundle bytes, peak browser memory, move latency (desktop + phone)
5. Run `python -m scripts.gen_golden_vectors` (needs engine) then open `docs/play/test_engine.html` to confirm all JS engine checks PASS
6. Push to GitHub Pages: enable Pages on `main` branch `docs/` folder in repo settings

**M3 exit gate:** static page plays a complete legal game offline, reports model hash, passes golden-vector suite, MCTS Hard mode responds in <1s.

See `SHIP_PLAN.md` M3 section for full exit gate.
