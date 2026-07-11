# RESULT -- M5.5 oracle-champion certification (expert_iter_v2 gen-5)

*Run on the RTX 3080 training box, 2026-07-11, from commit `e330c56`. The goat
run was STOPPED for the panel (clean Ctrl+C save, resumed after; ~2.5h
downtime). Raw panel output in `results/goat-gen5/` (gitignored); authoritative
numbers embedded below.*

## Candidate

The gen-5 teacher of the M5 expert-iteration run ("goat-train"), promoted
2026-07-11 after 36,528 self-play games (lineage: seeded from the incumbent
`arena:22@hof`, five promotions, each requiring >= 55% raw head-to-head over
300 fixed-opening games plus external-panel ratchets).

| Field | Value |
|---|---|
| Manifest | `benchmarks/goat_certified.json` |
| Snapshot | `models/expert_iter_v2/certified/candidate.pt` (gitignored) |
| SHA-256 (.pt, identity of record) | `748e77329aad34120cf0a050741cf151eb2e8afd5e1da700046e1daa6f4d3258` |
| Architecture | arena22: conv=`[64,256,256,32,256,64,128]` fc=`[256,1024]`, tanh value head |
| Parameters / fp32 bytes | 6,766,386 / 27,072,859 |
| Goat-panel scores at promotion (300 games each) | head-to-head 58% vs gen-4, winblock 35%, random 82%, gregory(d3) 14% |

## Decision: PROMOTED to oracle champion

Three independent instruments, summarized: the candidate wins the direct
head-to-head against the incumbent decisively, wins 3 of 4 M2 anchor modes
(the fourth is a statistical tie), and ties on provable endgame blunders.

### 1. Direct head-to-head vs `arena:22@hof` (the decider)

300 games, raw argmax both sides, fixed openings, colors swapped, seed 9901
(fresh seed, never used in training):

**gen-5 score 0.698** (SE ~0.029; ~7 sigma above even).

### 2. M2 anchor panel (same instrument as `RESULT_M2.md`)

Mean score vs the five independent anchors (lottery, nn_big8, winblock,
center, first; 18 games each; oracle excluded from aggregate):

| Mode | gen-5 | incumbent | delta |
|---|---:|---:|---:|
| raw | **0.700** | 0.606 | +0.094 |
| tactical | 0.800 | **0.844** | -0.044 (tie; ~0.9 SE) |
| mcts_25 | **0.778** | 0.706 | +0.072 |
| mcts_100 | **0.856** | 0.700 | +0.156 |

Full rows: raw 0.806/0.639/0.361/0.861/0.833, tactical
0.917/0.806/0.500/0.917/0.861, mcts_25 0.833/0.722/0.472/0.917/0.944,
mcts_100 0.889/0.806/0.722/0.889/0.972 (anchor order: lottery, nn_big8,
winblock, center, first).

Notable: gen-5's best certified mode (mcts_100, 0.856) beats the incumbent's
best certified mode (tactical, 0.844). The historic winblock blind spot
improved raw 0.083 -> 0.361 and mcts_100 0.306 -> 0.722.

vs own 400-sim oracle (SELF-REFERENTIAL -- reference only, not a cross-net
criterion; see "Rule change" below): raw 0.083, tactical 0.194, mcts_25
0.111, mcts_100 0.222.

### 3. GOLD provable-blunder rate

On the committed fixed suite `gold_endgame_suite.json` (same 336 gradable
positions for both, tactical mode, 100% oracle-proven):

| Candidate | Blunders | Rate |
|---|---:|---:|
| gen-5 | 12 / 336 | 3.57% |
| incumbent | 10 / 336 | 2.98% |

A 2-position difference on a shared set: statistical tie. This is the first
time the fixed suite is used as the instrument of record (the incumbent had
never been scored on it); both numbers are now recorded.

Live-game instrument (positions from the candidate's own games vs center,
tactical, `RESULT_GRADING.md` method): gen-5 9.54% (37/388 from 1000 games)
vs incumbent 6.26% (54/862). NOT comparable across nets: gen-5 wins so fast
that only 388 games in 1000 reached a gradable endgame, and those survivors
are a skewed-hard subset. The fixed suite above removes this confound; the
live number is recorded for completeness only.

## Rule change (2026-07-11)

Two amendments to the `CHAMPIONS.md` promotion instrument, both motivated by
this certification:

1. **Direct head-to-head added** (300 fixed-opening color-swapped raw games)
   as the tie-breaking criterion when panel modes disagree.
2. **"Holds own 400-sim oracle" retired as a cross-net criterion.** It is
   self-referential: a net with a working (tanh) value head makes its own
   oracle far stronger, so a BETTER net can score lower. The incumbent's
   0.500 reflected its broken unbounded value head capping its own search
   (`RESULT_M2.md` finding 2). Evidence the gen-5 search works: search now
   ADDS strength over tactical (0.856 > 0.800) where the incumbent's search
   LOST strength (0.700 < 0.844). The M4 thesis -- fix the value head, unlock
   search -- is realized.

## Ship

- `docs/models/champion.onnx` (fp32, 26,438 KB; ONNX SHA-256
  `a6660aa572fee1d2fbf5fea77071ab6a131011719023e34c10aad5fabf5204c8`) +
  `champion_config.json`; torch-vs-ONNX parity 25/25 top-1, value err <2e-6,
  tanh baked into the graph.
- Play page: opt-in "Champion (26 MB)" model picker; pocket stays default.
  Champion Hard mode = certified tactical argmax (`_tacticalPool` in
  `agent.js` mirrors `engine/tactics.py tactical_filter`).
- `turn_based_games` (sibling repo): UTTT solo bot upgraded to the champion
  (cross-origin fetch from this repo's Pages; async brain behind the
  synchronous `computerMove` contract; win/block heuristic fallback until
  loaded/offline).

## Reproduce

```
set CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scripts.benchmark_suite ^
  --candidate benchmarks/goat_certified.json ^
  --anchors lottery,nn_big8,winblock,center,first ^
  --candidate-sims 0,25,100 --oracle-sims 400 ^
  --openings standard --out results/goat-gen5

python -m scripts.grade_agent --suite gold_endgame_suite.json ^
  --candidate benchmarks/goat_certified.json --tactical
```

Head-to-head: `scripts.expert_iter._play_fixed_match` protocol, 300 games,
seed 9901, both nets raw argmax via `benchmark_suite` loaders.
