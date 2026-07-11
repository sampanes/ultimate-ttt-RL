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
| Selector | `arena:21@hof` (`06-26-26`) |
| Architecture | conv=`[32,128,32,32]` fc=`[128,256,512]` |
| Parameters | 1,287,314 |
| fp32 bytes | 5,156,981 (5.16 MB) |
| int8 estimate | ~1.29 MB (clears the M3 <= 5 MB gate) |
| SHA-256 (fp32 source) | `7498a31f3368f9c018713e346ee9dbfebf96de704860154ead4813c0bff1ca9d` |
| Move latency (raw) | ~0.77 ms CPU / ~1.08 ms CUDA (single position) |
| M2 aggregate (tactical) | 0.733 vs anchors; 0.333 vs its own 400-sim oracle |
| GOLD blunder (tactical vs center) | 7.75% |
| Ships as | quantized ONNX in `docs/` (M3) |

Rejection note for the strength net at this budget: `arena:22` is ~6.8 MB int8,
missing the 5 MB gate, so it cannot be the pocket champion despite being stronger.

## Oracle champion (current)

| Field | Value |
|---|---|
| Selector | `benchmarks/goat_certified.json` (expert_iter_v2 gen-5, promoted 2026-07-11) |
| Architecture | conv=`[64,256,256,32,256,64,128]` fc=`[256,1024]`, tanh value head |
| Parameters | 6,766,386 |
| fp32 bytes | 27,072,859 (27 MB) |
| SHA-256 (fp32 source .pt) | `748e77329aad34120cf0a050741cf151eb2e8afd5e1da700046e1daa6f4d3258` |
| Direct h2h vs prior champion | **0.698** (300 games, raw, fixed openings, color-swapped) |
| M2 aggregates | raw 0.700, tactical 0.800, mcts_25 0.778, **mcts_100 0.856** (prior best-any-mode was 0.844) |
| GOLD suite blunder (tactical, fixed 336 positions) | 3.57% (prior champion on same suite: 2.98% -- tie) |
| Ships as | fp32 ONNX `docs/models/champion.onnx` (browser opt-in) + `turn_based_games` UTTT solo bot |

This is the M4/M5 bounded-value, search-trained champion the previous row
anticipated: the expert-iteration teacher lineage (MCTS-200 over its own tanh
net) seeded from `arena:22@hof` and promoted five times on fixed external
panels. Search finally adds strength over the raw/tactical net (mcts_100
0.856 > tactical 0.800) instead of losing it (incumbent: 0.700 < 0.844).
Full certification: `RESULT_M2_5.md`. Superseded: `arena:22@hof` (SHA
`400374b1...`, M2 tactical 0.844 -- see `RESULT_M2.md`).

## Provenance

Full M2 panel: `RESULT_M2.md`. Certified blunder rates: `RESULT_GRADING.md`,
`RESULT_HOME_QUEUE.md`. Size/latency prep: `RESULT_M3_PREP.md`. The fp32 arena
checkpoints themselves are gitignored generated artifacts (not committed); their
SHA-256 above is the identity of record.
