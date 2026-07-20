# RESULT -- gen-19 oracle-champion certification (expert_iter_v2)

*Run on the RTX 3080 training box, 2026-07-20. The goat run was gracefully
STOPPED for the panel (clean Ctrl+C save, gen-19 never lost) and the GPU was
idle for the measurement. Raw panel output in `results/goat-gen19/`
(gitignored); authoritative numbers embedded below.*

## Candidate

The gen-19 teacher of the expert_iter_v2 run ("goat-train"). Lineage: seeded
from `arena:22@hof`, then nineteen promotions, each requiring >= 55% raw
head-to-head over 300 fixed-opening games plus external-panel ratchets. This
certification promotes it to oracle champion, replacing gen-5 (RESULT_M2_5.md).

| Field | Value |
|---|---|
| Manifest | `benchmarks/goat_certified.json` (now points at the gen-19 snapshot) |
| Snapshot | `models/expert_iter_v2/certified/candidate.pt` (gitignored) |
| SHA-256 (.pt, identity of record) | `671f67edd60a209a334275b2e55efd912f5a28afb3c6ebcd873add25b9489d72` |
| Architecture | arena22: conv=`[64,256,256,32,256,64,128]` fc=`[256,1024]`, tanh value head |
| Parameters / fp32 bytes | 6,766,386 / 27,072,859 |
| Ships as | `docs/models/champion.onnx` (ONNX SHA `99954b12093e2a65099f8f2af33aa12a7102660bb9162046fe9523e5be9d7a2d`) |

## Decision: PROMOTED to oracle champion

gen-19 clears the CHAMPIONS.md promotion rule on every instrument, with no
regression on any anchor.

### 1. Direct head-to-head vs the incumbent gen-5 (the decider)

300 games, raw argmax both sides, fixed openings, colors swapped, seed 9901
(the same fresh seed gen-5 was certified on):

**gen-19 score 0.913** vs gen-5 (bar is 0.55; gen-5's own bar over its
incumbent was 0.698). For continuity, gen-19 vs `arena:22@hof` = 0.960
(gen-5 scored 0.698 there).

### 2. M2 anchor panel (`scripts/benchmark_suite.py`)

Mean score vs the five frozen anchors (lottery, nn_big8, winblock, center,
first; 18 games each; oracle mode disabled -- retired as a cross-net criterion
in RESULT_M2_5.md):

| Mode | gen-19 | gen-5 (record) | delta |
|---|---:|---:|---:|
| raw | **0.955** | 0.700 | +0.255 |
| tactical | **0.978** | 0.800 | +0.178 |
| mcts_25 | **0.961** | 0.778 | +0.183 |
| mcts_100 | **1.000** | 0.856 | +0.144 |

Per-anchor raw rows: lottery 0.944, nn_big8 1.000, winblock 0.861, center
0.972, first 1.000. The historic winblock blind spot improved raw
0.361 -> 0.861 and is now clean at mcts_100 (1.000). gen-19 beats gen-5 in
all 20 panel cells.

### 3. GOLD provable-blunder rate

On the committed fixed suite `gold_endgame_suite.json` (336 gradable
positions, tactical mode, 100% oracle-proven):

| Candidate | Blunders | Rate |
|---|---:|---:|
| gen-19 | 10 / 336 | 2.98% |
| gen-5 | 12 / 336 | 3.57% |

gen-19 is slightly better than gen-5 and matches the older arena:22 incumbent
(2.98%). No regression.

## Ship

- `docs/models/champion.onnx` regenerated from the gen-19 snapshot via
  `scripts.export_onnx`; torch-vs-ONNX parity 40/40 policy top-1, max value
  error 3.8e-6, max logit error 5.3e-5, tanh baked into the graph.
- `docs/models/champion_config.json` updated (name/version/description).
- Play page: the opt-in "Champion" model picker now serves gen-19; Hard mode
  stays the certified 1-ply tactical argmax (a 27 MB net at 50 sims is too slow
  in WASM, and tactical certifies stronger anyway).
- `turn_based_games` (sibling repo) UTTT bot fetches `champion.onnx` cross-origin
  from this repo's Pages, so it adopts gen-19 automatically on deploy -- no
  separate change.

## Reproduce

```
set CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scripts.benchmark_suite ^
  --candidate benchmarks/goat_certified.json ^
  --anchors lottery,nn_big8,winblock,center,first ^
  --candidate-sims 0,25,100 --oracle-sims 0 ^
  --openings standard --out results/goat-gen19

python -m scripts.grade_agent --suite gold_endgame_suite.json ^
  --candidate benchmarks/goat_certified.json --tactical
```

Head-to-head: `scripts.expert_iter._play_fixed_match` protocol, 300 games,
seed 9901, both nets raw argmax via `benchmark_suite` loaders.
