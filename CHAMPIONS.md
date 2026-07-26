# Champions -- the current best player at each budget

This is the canonical, always-current record of the best model at each footprint
budget. It is a manifest, not a weight store: the actual weights live where they
ship (quantized ONNX in `docs/` for the pocket, Hugging Face for the oracle), and
are identified here by SHA-256. Update this file on every promotion; keep large
`.pt` binaries out of git history.

"Best" and "smallest" pull in opposite directions, so there are two standing
champions, not one:

- **Pocket champion** -- smallest model that is still strong; ships client-side.
- **Oracle champion** -- strongest model regardless of size; hosted, and the
  teacher for the pocket track.

## Promotion rule

A challenger replaces the incumbent only when it **beats the incumbent on the M2
independent panel** (`scripts/benchmark_suite.py`: raw + tactical + MCTS ladder
vs the frozen non-gene-pool anchors, and blunder rate on
`gold_endgame_suite.json`). Certify by panel, not by closed-loop Arena ELO --
run #4 (`RESULT_HOME_QUEUE.md`) showed `best.pt`'s ELO 4437 loses 0/40 to shallow
search, i.e. ELO does not measure real strength. On promotion: bump the row,
record the new SHA-256, and note which metric improved.

Amended 2026-07-11 (`RESULT_M2_5.md`): when panel modes disagree, a direct
300-game fixed-opening color-swapped raw head-to-head vs the incumbent is the
tie-breaker; the "holds own 400-sim oracle" score is reference-only, not a
cross-net criterion (it is self-referential -- a better value head makes the
candidate's own oracle stronger, so a better net can score lower on it).

## Pocket champion (current)

| Field | Value |
|---|---|
| Selector | `pocket:squeeze-gen22` (`models/pocket_candidate/manifest.json`) |
| Architecture | conv=`[56,56,56,56]` fc=`[256]` **`head_squeeze=2`**, tanh value head |
| Parameters | **172,389** (7.5x fewer than the superseded champion) |
| fp32 bytes | 697,117 (0.70 MB) -- no quantization needed to clear the 5 MB gate |
| SHA-256 (fp32 source) | `b028d3499eca8b1049c5cdbe0a6deed2f056851afad68fe0858ca778af09123b` |
| ONNX SHA-256 (`docs/models/model.onnx`) | `ce7e89943d82513f66922777797f327028ae6604d886aafa7624841890880483` |
| ONNX bytes | 696,567 (680 KB, was 5,035 KB -- a 7.4x smaller download) |
| Direct h2h vs prior champion | **0.9833** (300 games, raw, fixed openings, colour-swapped, seed 9901) |
| 300-game panel (raw) | random 0.983, winblock 0.905, gregory-d3 0.728, gregory-d4 0.653 |
| M2 panel | wins **15/15** cells vs the incumbent (raw/tactical/mcts_25 x 5 anchors) |
| Ships as | fp32 ONNX `docs/models/model.onnx` |

Promoted 2026-07-26, full certification in `RESULT_POCKET_SQUEEZE.md`. A
distilled expert-iteration student, not an arena self-play net: trained offline
on MCTS-200 targets from the frozen gen-22 teacher, using the `head_squeeze`
1x1-conv head from `RESULT_ARCH_AB.md`. That head change is the entire size win
-- the legacy heads flattened the whole conv stack into a Linear and held 83.6%
of the parameters.

Superseded: `arena:21@hof` (`06-26-26`, SHA `7498a31f...`, 1,287,314 params,
5.16 MB). It lost every cell of both panels, including **0.067 vs gregory-d3 and
0.158 vs winblock at 300 games** -- the historic winblock blind spot that the
oracle track closed at gen-19 was still shipping in the pocket slot.

Rejection note retained for the record: `arena:22` is ~6.8 MB int8 and misses
the 5 MB gate. That constraint no longer binds -- the current champion is 0.70
MB fp32, so strength, not size, is the live limit at this budget.

## Oracle champion (current)

| Field | Value |
|---|---|
| Selector | `benchmarks/goat_certified.json` (expert_iter_v2 gen-19, promoted 2026-07-20) |
| Architecture | conv=`[64,256,256,32,256,64,128]` fc=`[256,1024]`, tanh value head |
| Parameters | 6,766,386 |
| fp32 bytes | 27,072,859 (27 MB) |
| SHA-256 (fp32 source .pt) | `671f67edd60a209a334275b2e55efd912f5a28afb3c6ebcd873add25b9489d72` |
| ONNX SHA-256 (`docs/models/champion.onnx`) | `99954b12093e2a65099f8f2af33aa12a7102660bb9162046fe9523e5be9d7a2d` |
| Direct h2h vs prior champion (gen-5) | **0.913** (300 games, raw, fixed openings, color-swapped, seed 9901) |
| M2 aggregates | raw 0.955, tactical 0.978, mcts_25 0.961, **mcts_100 1.000** (gen-5 best-any-mode was 0.856) |
| GOLD suite blunder (tactical, fixed 336 positions) | 2.98% (gen-5 on same suite: 3.57%) |
| Ships as | fp32 ONNX `docs/models/champion.onnx` (browser opt-in) + `turn_based_games` UTTT solo bot |

gen-19 of the expert-iteration teacher lineage (MCTS-200 over its own tanh
net), seeded from `arena:22@hof` and promoted through fourteen further external-
panel gates past the gen-5 champion. It beats gen-5 in every panel cell and
closes the historic winblock blind spot (raw winblock 0.361 -> 0.861). Full
certification: `RESULT_GEN19_CERT.md`. Superseded: expert_iter_v2 gen-5 (SHA
`748e7732...`, h2h 0.698 vs its own incumbent -- see `RESULT_M2_5.md`).

## Provenance

Full M2 panel: `RESULT_M2.md`. Certified blunder rates: `RESULT_GRADING.md`,
`RESULT_HOME_QUEUE.md`. Size/latency prep: `RESULT_M3_PREP.md`. The fp32 arena
checkpoints themselves are gitignored generated artifacts (not committed); their
SHA-256 above is the identity of record.
