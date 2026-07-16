# RESULT_S4 -- multiprocess game actors: measurement + landing record

*Training box (RTX 3080 10 GiB, Ryzen 9 3900X / 24 logical cores), 2026-07-15.
Authored on this box in the `uttt-speed` worktree, measured on an idle box,
landed live at 8 actors. Speed track item S4 (STRENGTH_NEXT).*

---

## 1. What shipped

`scripts/game_actors.py` -- a persistent, spawn-based pool of self-play
worker PROCESSES. Wired into `scripts/expert_iter.py` behind `--actors N`
(default 0 = the sequential path, byte-for-byte unchanged). Live launcher
`start_goat.bat` now passes `--actors 8`.

- Each actor builds its own teacher replica and plays whole games end to end,
  its own batch-12 forwards, returning examples over a queue.
- Persistent across blocks: an actor pays ~300 MB CUDA context + a model
  build at startup, which would dwarf a 16-game block if respawned per block.
- **Distribution-preserving, NOT byte-identical.** The main process still
  draws every opponent-slice tag from its own RNG and hands tags to actors,
  so the opponent mix is sampled exactly as the sequential loop samples it.
  Only per-game stochasticity (Dirichlet noise, temperature sampling) moves
  to per-actor seeded streams. Proof is in the code + `test_game_actors.py`
  (`test_parent_draw_is_one_call_per_game`), not in a byte diff.

Commit: `1650d84` (speed-track, fast-forwarded into actor-critical-league).

## 2. Isolated A/B (idle box, gen-8 teacher, 24 games/config, live S1 mix)

`arena22` network (NOT small -- the live default), 200 sims, seed-fixed
opponent mix {self 11, heur 6, rnd 4, greg 3}. Generation only; the ~2s
train step is untouched by S4.

    config       time(24g)  games/hr   s/game   speedup
    sequential     111.4s      775      4.64     1.00x
    2 actors        67.2s     1285      2.80     1.66x
    4 actors        48.4s     1784      2.02     2.30x
    6 actors        46.0s     1879      1.92     2.42x
    8 actors        45.3s     1909      1.89     2.46x
    12 actors       39.8s     2173      1.66     2.80x

Diminishing hard after ~4-6 actors. Per-actor throughput at 12-way is 4.6x
worse than one alone -- they contend for something that is NOT cores.

## 3. Why it saturates: the GPU, not the CPU (the finding that redirects S4)

Under 12 actors, sampled live:

    CPU: ~65% of 24 cores (about 15 busy)   <- cores to spare
    GPU: 95% utilization                     <- pegged
    VRAM: 2.5 / 10.2 GiB                      <- room to spare

RESULT_S1 sec 5e concluded generation is "~9% GPU, 90% Python." That holds
for ONE process. It does not compose: N processes each fire tiny batch-12
forwards into N separate CUDA contexts, and a consumer GPU with no MPS
serializes them. The 95% is largely context-switch thrash, not useful FLOPs
-- RESULT_S1 5b already measured that a batch-12 forward costs the SAME
0.91 ms as batch-1, i.e. each kernel is far too small to fill the SM array.

**Consequence for the roadmap:** more processes cannot pass ~2.6-2.8x. The
remaining headroom is behind ONE context doing large batches -- the shared
eval server (RESULT_S1 5b: batch-256 = 78,641 pos/s vs 13,138 at batch-12,
6.0x more efficient per position). S4's "second cut" is therefore not
optional polish; it is where the rest of the multiple lives.

## 4. Live result (the real run, actor-critical-league)

Landed 2026-07-15, gen-8 rebuild, `--actors 8`. Generation time per 16-game
block, sequential window vs actor window of the SAME run:

    sequential (last 60 blocks pre-S4): median 84s
    actors=8   (1,079 blocks post-S4) : median 26s   (min 15, max 34)
    -> 3.21x median | games/hour 690 -> 2,215

The live 3.21x BEATS the isolated 2.46x at 8 actors. Reason: the A/B paid
pool startup + short-run tail effects on only 24 games; the live pool is
persistent and stays warm across thousands of blocks. GPU 52-95% (95% only
while a gate/promotion eval shares it), VRAM 5.7 / 10.2 GiB, 8 workers up,
no OOM. This is the honest steady-state number.

## 5. Did it damage the data? (the gate that actually matters)

S4 is distribution-preserving, so the test is: do the promotion panels keep
climbing, or did they regress when actors turned on? gen-8 segment, split at
the S4 restart (games_total ~93.9k):

    window                    n   head  winblock  random  gregory
    pre-S4  (sequential)      8   43.9    39.3     86.0    16.5
    post-S4 (8 actors)       18   50.0    46.1     90.0    21.3

Every panel is HIGHER post-S4. This is NOT a clean A/B -- the two windows sit
at different points on gen-8's rebuild curve (8 early checks right after the
promotion dip vs 18 later ones), so it confounds S4 with normal rebuild
progress. What it DOES establish is the only thing at risk: no collapse, no
regression, all four metrics moving up together in the shape a healthy
rebuild makes. gregory in particular keeps printing 18-22 (S1 trend intact).
The formal promotion of gen-9 -- entirely on actor-generated data -- will be
the clean confirmation.

## 6. Correctness gates (all green before landing)

- `scripts.test_expert_iter`: **11/11** (sequential path unchanged).
- `scripts.test_game_actors`: **5/5** -- a real 2-actor CPU pool plays a
  block and returns well-formed examples; the opponent mix draws at target
  rates; ONE parent draw per game; and the value_tanh reload guard (see 7).

## 7. The bug this could have shipped, and the guard

An actor holds its own model replica and reloads from `teacher.pt` at each
promotion. `load_state_dict` copies weights ONLY, not the runtime
`value_tanh` flag. Without a guard, an actor that started on a non-tanh gen-0
teacher and reloaded a promoted (always-tanh) teacher would feed unbounded
pre-tanh values into its MCTS and silently generate poisoned targets -- the
exact failure `_promote_teacher`'s docstring records (values in [-1.66, 3.27]
for ~12h on the first v2 run). `_load_weights` takes `value_tanh` from the
saved payload; `test_reload_adopts_value_tanh_from_payload` fails without it.

## 8. Status + next

- **LIVE at --actors 8.** Revert = `--actors 0` (byte-identical sequential).
- Honest speedup on this hardware: **~3.2x games/hour**, steady state.
- Ceiling of this cut is ~2.6-2.8x isolated / ~3.2x warm; the GPU-context
  wall (sec 3) caps it. Next build: the shared eval server (batch-256
  forwards from one context) to convert the idle CPU cores + the 6x
  per-position efficiency into the next multiple.
- S8 (per-sim hot path: C++ fill_planes, vectorized tensor build) is
  complementary -- it cuts each actor's Python cost -- but the GPU wall means
  it will not raise aggregate throughput until the eval server exists.
