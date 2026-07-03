# RESULT: throughput benchmark + perf A/B (home box, RTX 3080, 2026-07-02)

Home-box execution of the PENDING.md throughput-harness queue. All three steps ran:
parity gate, `bench_throughput` (full, 5 min/candidate, 3 repeats), `home_batch --phase perf`.
Raw reports are gitignored (local paths); everything decision-relevant is below.

Environment: Python 3.11.9, torch 2.7.1+cu128, CUDA 12.8, RTX 3080 10GB, C++ engine active.

---

## Verdicts (flag-hardening table)

| Flag | Verdict | Evidence |
|---|---|---|
| `--batch_opponents` | **BAKE DEFAULT ON** | parity PASS (80/80 byte-identical) + 1.09x A/B, 1.19x in bench at parallel=64 |
| `--wave_size` | **BAKE `_WAVE_SIZE = 64`** | 2.3 games/s vs 0.1 at wave=1 (~20x AZ self-play); monotonic 1<4<8<16<32<64 |
| `--compile` | **DEAD ON WINDOWS -- remove from gates** | `torch._inductor.exc.TritonMissing`; Triton does not exist on Windows. Not fixable here. |
| `--amp` | **BUG -- needs authoring-box fix before it can even be benchmarked** | see below |

## 1. Parity gate: PASS

`verify_opponent_batch_parity` (80 games, 27 batchable NN opponents, network=small, stage 4):
0 mismatches. Grouped batched argmax is byte-identical to the per-slot loop.

## 2. bench_throughput (full run) -- key rows

League suite (games/s mean of 3, stdev <= 0.2 throughout):

| candidate | games/s |
|---|--:|
| network=small | 9.1 |
| parallel=256 | 8.8 |
| parallel=128 | 8.0 |
| batch_opponents (at parallel=64) | 7.5 |
| parallel=64 baseline | 6.3 |
| network=medium / large | 6.4 / 6.6 |
| recompute (mb=0/32/64/128) | 5.9-6.0 |
| seq (parallel=0) | 0.6 |

Notes:
- batch_opponents is +19% over the parallel=64 baseline in-bench. Raw throughput of
  parallel=128/256 is higher still, but the earlier learning A/Bs showed big batches
  starve gradient updates (256 stalled) -- throughput alone does not promote them.
- recompute rows are consistently ~5% SLOWER than baseline here, consistent with the
  earlier "keep opt-in" call.

AlphaZero suite:

| candidate | games/s |
|---|--:|
| wave_size=64 | 2.3 |
| wave_size=32 | 1.6 |
| wave_size=16 / n_sims=100 / any network size | 1.1 |
| wave_size=8 | 0.7 |
| n_sims=200 | 0.6 |
| wave_size=4 | 0.4 |
| n_sims=400 | 0.3 |
| wave_size=1 | 0.1 |

Notes:
- wave_size is THE lever: ~20x at 64 vs 1, still not saturated at 64 on a 3080 --
  a wave_size=128 point may be worth adding to the matrix.
- Network size barely matters for AZ throughput (MCTS/Python dominates), so M4 can
  afford `--network large` nearly free if desired.

## 3. home_batch --phase perf (equal-budget A/B, single seed)

| config | secs | games/s | speedup | peak ELO | final WR | mean EV |
|---|--:|--:|--:|--:|--:|--:|
| baseline | 318.1 | 16.1 | 1.00x | 808.7 | 0.3955 | 0.0012 |
| batch_opponents | 290.7 | 17.6 | 1.09x | 808.7 | 0.3838 | 0.0015 |
| amp | crash | - | - | - | - | - |
| compile | crash | - | - | - | - | - |
| all | crash | - | - | - | - | - |

batch_opponents clears its gate: parity PASS + speedup > 1.05x, peak ELO identical.

**FLAG (per the report's own read-guidance):** final WR (0.3838 vs 0.3955) and min_loss
(0.376 vs 0.4169) are NOT digit-identical between baseline and batch_opponents, despite the
oracle claiming byte-identical games. Likely benign (CUDA kernel nondeterminism in the
LEARN phase -- two baseline runs would probably also differ; the parity script pins
deterministic algorithms, the A/B does not), but the authoring box should confirm that
reading, e.g. by rerunning baseline twice and observing the same spread.

## 4. Bugs for the authoring box

### --amp crashes at first backward (train_league)
```
File "agents/neural_net_agent_pg.py", line 427, in learn_from_trajectories
  self._scaler.scale(loss).backward()
RuntimeError: Found dtype Float but expected Half
```
Reproduced standalone (any tiny run with --amp). The loss graph mixes fp32 tensors into
the fp16 autocast backward -- classic cause: part of the loss (e.g. stored rollout
log-probs/values, or a term computed outside the `autocast()` context) is fp32 while the
forward under autocast produced fp16. Fix on the authoring box: ensure the entire loss
construction in `learn_from_trajectories` happens under the same autocast scope (or
explicitly float() the network outputs before combining). Until fixed, AMP cannot be
benchmarked at all.

### --compile is permanently unavailable on the training box
`TritonMissing` -- torch.compile requires Triton, which has no Windows build (already a
known project constraint in CLAUDE.md). Recommend: remove `--compile` from the gate
table for this hardware, or have train_league print a clear "[!] --compile requires
Triton (not available on Windows), ignoring" instead of crashing mid-run.

---

## Recommended M4 long-run config (this box)

```
set CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv\Scripts\python -m scripts.train_alphazero --network medium --value_tanh ^
  --n_sims 200 --wave_size 64 --games_per_iter 50
```
(fresh start or tanh-trained seed, NOT untanhed best.pt; league runs add
`--batch_opponents`.)
