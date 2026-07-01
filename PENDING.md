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

1. Export `arena:21@hof` to ONNX with fixed input/output contract
2. Apply static int8 quantization with a representative UTTT position set
3. Verify PyTorch vs ONNX vs int8 policy/value parity; rerun M2 benchmark on quantized model
4. Port rules + MCTS to TypeScript/WASM; lock golden vectors against Python
5. Measure cold download, bundle bytes, peak memory, move latency (desktop + phone)
6. Publish static GitHub Pages build

See `SHIP_PLAN.md` M3 section for full exit gate.
