# RESULT_S5 -- shared eval server + S8 hot path: measurement + landing record

*Training box (RTX 3080 10 GiB, Ryzen 9 3900X / 24 logical cores), 2026-07-16.
Authored on this box in the `uttt-speed` worktree, A/B'd on the paused box,
landed live on the gen-9 teacher. Speed-track items S5 (eval server) + S8 (per-
sim hot path), STRENGTH_NEXT. Follows RESULT_S4 (the first cut).*

---

## 1. What shipped

Two composable changes, one cutover:

- **S5 shared eval server** (`scripts/eval_server.py`). One `EvalServer` process
  owns the ONLY generation CUDA context and answers every actor's forward,
  batching all in-flight requests into a single `forward_both`. The N actors
  become PURE-CPU (no model, no CUDA context): they build planes and walk the
  tree, then ship each forward to the server over an mp.Queue and block for the
  reply. `EvalServerActorPool` has the same interface as the first cut's
  `GameActorPool` (`play_block`/`reload_weights`/`close`), so expert_iter selects
  it with `--eval_server` and the generation loop is unchanged.
- **S8 per-sim hot path** (`engine/cpp/uttt_engine.cpp` + `agents/agent_base.py`
  + `agents/mcts.py`). C++ `fill_planes(out)` writes the 7x9x9 input planes
  straight into a caller numpy buffer -- one pybind crossing per leaf, zero
  Python plane math. `board_to_tensor_from_gamestate` uses it when present;
  `wave_planes` fills a whole wave into one buffer. Byte-identical to the numpy
  build (parity-gated).

Distribution-preserving exactly as S4: the parent draws the opponent mix, actors
draw per-game noise. The value_tanh reload guard is now a SINGLE server-side site
(actors never touch weights). Commits: `204f7b3` (S5), `1889c01` (S8).

## 2. Why S5: the first cut's wall (RESULT_S4 sec 3, recap)

S4 topped out ~3.2x because each of 8 actors owned a CUDA context and fired
batch-12 forwards that a consumer 3080 (no MPS) serializes -- GPU pegged at 95%
on context-switch thrash while ~9 of 24 cores idled. One context doing large
batches is 6.0x more efficient per position (RESULT_S1 5b). S5 is exactly that:
collapse N contexts to 1, batch across all actors, free the cores.

## 3. Why S8: the actor's own CPU (profiled)

With the GPU forward stubbed to instant (so only the actor's residual Python
cost shows), cProfile of `collect_game` (40 games, 200 sims):

    cost                                   cumulative   share
    board_to_tensor_from_gamestate            77.6s      46%   <- dominant
    _expand_from_logits (mask/softmax/scatter)37.7s      23%
    PUCT tree select (_best_child/U/Q/max)   ~27s        16%
    rule_utl_valid_moves                       9.5s       6%

Plane build was the fat target, so S8 attacks it (not the tree). After S8, the
same profile is **167.0s -> 90.7s (1.84x actor CPU)**; `fill_planes` is 2.7us/
call vs the old ~146us (**54x**), and plane build fell from 46% to ~3.6%. The new
top cost is `_expand_from_logits` (ragged per-leaf softmax -> harder to
vectorize), left as a deferred follow-on.

## 4. A/B (paused box, gen-9 teacher, 32 games/config, 200 sims, arena22)

Live S1 mix; GENERATION only (the ~2s train step is untouched by S5/S8). The
dashboard held ~20% GPU throughout, so these are if anything conservative.

    config                 games/hr   s/game   speedup
    sequential (S8 off)        714      5.04     1.00x   <- anchor
    first-cut a12 +S8         2491      1.45     3.49x
    eval-server a8  +S8       9518      0.38    13.33x
    eval-server a12 +S8      10462      0.34    14.65x
    eval-server a16 +S8      11509      0.31    16.11x   <- knee
    eval-server a24 +S8      10875      0.33    15.23x

The eval server alone contributes 4.2x over the first cut at the SAME actor count
(14.65 / 3.49) -- the GPU-context-wall removal. S8 adds ~1.25x on top of the
first-cut path (3.49x vs S4's ~2.8x without it). Past 16 actors it regresses:
only 16 games to spread, and more actors just add scheduling overhead.

## 5. Live result (the real run, actor-critical-league)

Landed 2026-07-16 on the gen-9 teacher: `--eval_server --actors 16`.

    metric                    first cut (a8)     new stack (a16)
    generation / 16-game block  ~26s               ~5s
    block-to-block (gen+train)  ~27s               ~7s
    games/hour (end to end)     ~2,215             ~8,200
    VRAM                        5.7 GiB            3.1 GiB

Live generation is 5s / 16 games = **11,520 games/hr, matching the a16 bench
(11,509) exactly**. Block-to-block is ~7s because the fixed ~2s train step is now
~30% of a block -- so end-to-end is ~3.9x over the first cut, while generation
itself is the full 16x. GPU 75% (useful batched work, not thrash), VRAM 3.1 /
10.2 GiB with 7 GiB free. `[S5]` startup line present; ZERO tracebacks after the
restart (the only tracebacks are the old first-cut actors' KeyboardInterrupt on
the stop). 16 actors match the 16-game block: one parallel wave, no idle actors.

## 6. Correctness gates (all green before landing)

- `scripts.test_eval_server` (3): batched forward == per-row forward in eval()
  (the server's core faithfulness); a real 2-actor CPU pool plays a block and a
  reload round-trips; a dead server makes play_block RAISE (bounded), never hang.
- `scripts.test_hot_path` (2): fill_planes == the numpy build byte-for-byte over
  3000+ reachable positions; wave_planes == the per-leaf build. Skips cleanly on
  the pure-Python engine.
- Regressions: `test_game_actors` 5/5, `test_expert_iter` 11/11 (first-cut and
  sequential paths untouched). Re-run green in the training tree after the merge
  + C++ rebuild.

## 7. Status + next

- **LIVE at `--eval_server --actors 16`.** REVERT: drop `--eval_server` (first-cut
  context per actor) or `--actors 0` (sequential path, byte-identical).
- Honest end-to-end speedup vs the first cut: **~3.9x** (generation 16x; the train
  step is the new fixed cost). Vs the original sequential path: generation ~16x.
- **Easiest next lever: raise `--games_per_block`.** At 16 games the ~2s train
  step is ~30% of each block and actors do 1 game each (single wave, no per-actor
  pipelining). A larger block amortizes the train step AND deepens the pipeline;
  the a24-at-32-games point suggests headroom. One change at a time -- try it only
  after gen-10's promotion confirms the current stack is distribution-clean.
- Deferred: `_expand_from_logits` vectorization (the new #1 actor cost), and the
  shared-memory transport (mp.Queue is not the bottleneck at this rate).
- Watch gen-10's promotion (first gen fully generated on eval server + S8) as the
  clean distribution-safety confirmation, same discipline as S4/gen-9.
