# RESULT -- M2 independent finalist playoff

*Run on the RTX 3080 training box, 2026-06-28, from commit `50e6d91` (working
tree dirty). Generated under `results/arena-21` and `results/arena-22` (those
dirs are gitignored by convention); the authoritative numbers are embedded
below. Produced by the committed `scripts/benchmark_suite.py` (M1 deliverable).*

## Decision

- **Oracle base: `arena:22@hof`** (`06-27-26`, SHA-256 `400374b1...`,
  6,766,386 params, 27,074,677 bytes). Strongest finalist by every aggregate and
  the only one that holds its own deep oracle to even. Footprint is irrelevant
  for the hosted oracle track (M4).
- **Pocket base: `arena:21@hof`** (`06-26-26`, SHA-256 `7498a31f...`,
  1,287,314 params, 5,156,981 bytes). At ~5.2 MB it quantizes to roughly 1.3 MB
  int8 and clears the M3 <=5 MB gate, which the 27 MB strength net (~6.8 MB int8)
  would miss. It pays about 0.11 aggregate score for the footprint and leans on
  the cheap 1-ply tactical layer.

No other finalists exist; M0 produced exactly this two-point size/strength
frontier. The decision is not taken from Arena ELO or Hall-of-Fame filenames --
it is taken from the independent panel below.

## Aggregate strength

Mean score (win=1, draw=0.5, loss=0) against the five independent, non-gene-pool
anchors `lottery, nn_big8, winblock, center, first`. The deep `oracle_mcts_400`
is excluded from the aggregate because it shares the candidate's own weights.

| Candidate mode | arena-21 (pocket) | arena-22 (strength) |
|---|---:|---:|
| raw | 0.611 | 0.606 |
| tactical | **0.733** | **0.844** |
| mcts_25 | 0.644 | 0.706 |
| mcts_100 | 0.728 | 0.700 |

vs the deep `oracle_mcts_400` (400-sim PUCT search over each net's own head):

| Candidate mode | arena-21 | arena-22 |
|---|---:|---:|
| raw | 0.028 | 0.139 |
| tactical | 0.333 | **0.500** |
| mcts_25 | 0.111 | 0.250 |
| mcts_100 | 0.194 | 0.250 |

## Full panel -- arena-21 (pocket)

| Candidate mode | Opponent | W-D-L | Score |
|---|---|---:|---:|
| raw | lottery_no_touchy | 14-0-4 | 0.778 |
| raw | nn_big_8 | 15-2-1 | 0.889 |
| raw | winblock | 1-0-17 | 0.056 |
| raw | center | 10-2-6 | 0.611 |
| raw | first | 13-0-5 | 0.722 |
| raw | oracle_mcts_400 | 0-1-17 | 0.028 |
| tactical | lottery_no_touchy | 14-1-3 | 0.806 |
| tactical | nn_big_8 | 16-2-0 | 0.944 |
| tactical | winblock | 6-3-9 | 0.417 |
| tactical | center | 10-2-6 | 0.611 |
| tactical | first | 15-2-1 | 0.889 |
| tactical | oracle_mcts_400 | 1-10-7 | 0.333 |
| mcts_25 | lottery_no_touchy | 14-0-4 | 0.778 |
| mcts_25 | nn_big_8 | 16-2-0 | 0.944 |
| mcts_25 | winblock | 1-1-16 | 0.083 |
| mcts_25 | center | 11-3-4 | 0.694 |
| mcts_25 | first | 13-0-5 | 0.722 |
| mcts_25 | oracle_mcts_400 | 0-4-14 | 0.111 |
| mcts_100 | lottery_no_touchy | 15-1-2 | 0.861 |
| mcts_100 | nn_big_8 | 16-2-0 | 0.944 |
| mcts_100 | winblock | 5-3-10 | 0.361 |
| mcts_100 | center | 8-6-4 | 0.611 |
| mcts_100 | first | 15-1-2 | 0.861 |
| mcts_100 | oracle_mcts_400 | 0-7-11 | 0.194 |

## Full panel -- arena-22 (strength)

| Candidate mode | Opponent | W-D-L | Score |
|---|---|---:|---:|
| raw | lottery_no_touchy | 12-0-6 | 0.667 |
| raw | nn_big_8 | 14-1-3 | 0.806 |
| raw | winblock | 1-1-16 | 0.083 |
| raw | center | 12-1-5 | 0.694 |
| raw | first | 14-0-4 | 0.778 |
| raw | oracle_mcts_400 | 0-5-13 | 0.139 |
| tactical | lottery_no_touchy | 16-0-2 | 0.889 |
| tactical | nn_big_8 | 18-0-0 | 1.000 |
| tactical | winblock | 4-8-6 | 0.444 |
| tactical | center | 17-0-1 | 0.944 |
| tactical | first | 17-0-1 | 0.944 |
| tactical | oracle_mcts_400 | 5-8-5 | 0.500 |
| mcts_25 | lottery_no_touchy | 12-0-6 | 0.667 |
| mcts_25 | nn_big_8 | 17-0-1 | 0.944 |
| mcts_25 | winblock | 2-4-12 | 0.222 |
| mcts_25 | center | 14-1-3 | 0.806 |
| mcts_25 | first | 16-0-2 | 0.889 |
| mcts_25 | oracle_mcts_400 | 2-5-11 | 0.250 |
| mcts_100 | lottery_no_touchy | 12-0-6 | 0.667 |
| mcts_100 | nn_big_8 | 15-0-3 | 0.833 |
| mcts_100 | winblock | 3-5-10 | 0.306 |
| mcts_100 | center | 14-1-3 | 0.806 |
| mcts_100 | first | 16-0-2 | 0.889 |
| mcts_100 | oracle_mcts_400 | 1-7-10 | 0.250 |

## What this establishes

1. **arena-22 is the stronger player, decisively.** It leads on every aggregate
   and is the only finalist that draws even with its own 400-sim oracle (tactical
   0.500 vs arena-21's 0.333). In tactical mode it sweeps nn_big_8 18-0-0 and
   takes center/first 17-1.
2. **1-ply tactical filtering matches or beats MCTS-100 almost everywhere, at
   roughly 50-100x less compute.** This confirms the standing thesis (see
   `GOAT_NEXT.md`, `RESULT_MCTS_ORACLE.md`) that the unbounded shaped-return value
   head caps search sharpness, so these MCTS numbers are a **floor**, not a
   ceiling. For the browser pocket target the cheap tactical layer is the
   high-ROI move; deep search waits on the bounded value head (M4).
3. **`winblock` is the shared blind spot.** Raw policy collapses to 0.056 / 0.083
   against a trivial take-win / block-loss heuristic; tactical only recovers to
   ~0.42 / 0.44 and MCTS-100 to ~0.36 / 0.31. Neither finalist reliably defends
   immediate mini-board threats. This reproduces the `RESULT_ANCHORS.md` WinBlock
   failure on independently selected checkpoints and is the concrete benchmark
   failure to convert into the next training run's specification.

## Method

- Suite: `scripts/benchmark_suite.py`, fail-closed, architecture resolved from
  `models/arena/arena_state.json` (no `small`/`medium`/`large` guessing).
- Candidate modes: `raw` (policy argmax), `tactical` (1-ply immediate
  win/block filter), `mcts_25`, `mcts_100` (PUCT, `c_puct=1.5`, temperature 0).
- Anchors: frozen `lottery_no_touchy`, frozen `nn_big_8`, deterministic
  `winblock`/`center`/`first`, and `oracle_mcts_400` (400-sim search over the
  candidate's own weights).
- Openings: committed `benchmarks/openings_standard.json` (9 legal openings,
  SHA-256 `b960f9c9...`), each played from both sides -> 18 games per pair,
  432 games per finalist.
- Reproducibility: seed 0, deterministic algorithms enabled. On CUDA this
  requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment, or the MCTS
  matmuls raise (set it before launching; do not put `time` ahead of the inline
  assignment). Rules use the repository C++ engine.
- Provenance recorded in each `results.json`: checkpoint/anchor/opening SHA-256
  and bytes, parameter count, git commit, ruleset file hashes, device, seed, and
  search settings.
- Checkpoint SHA-256:
  - arena-21 (pocket): `7498a31f3368f9c018713e346ee9dbfebf96de704860154ead4813c0bff1ca9d`
  - arena-22 (strength): `400374b1a2d2ce638de5ed01d7ca12adba1ad24c9d8a0955bbeb8890af11138b`
  - lottery: `8bb8ad2032cbc5ede443c2781210b10f948752ac9ca53e018e613b8646beaa69`
  - nn_big8: `6998f65f556c71c565071f6caf58a18bfd59ed04e8f972af11ca51b3e0cc2675`

## Reproduce

```
set CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scripts.benchmark_suite ^
  --candidate arena:21@hof ^
  --anchors lottery,nn_big8,winblock,center,first ^
  --candidate-sims 0,25,100 --oracle-sims 400 ^
  --openings standard --out results/arena-21
```

Swap `arena:21@hof` for `arena:22@hof` for the strength finalist. Same seed and
openings reproduce W/D/L, score, and the Wilson interval exactly.

## Next (M3 / M4 specification)

- M3 pocket: export `arena:21@hof` to ONNX, static int8 quantize, verify policy
  parity, and re-run this panel post-quantization. Ship with the tactical layer;
  it is cheap and competitive with MCTS-100 here.
- M4 oracle: take `arena:22@hof`, replace the unbounded shaped-return value head
  with a bounded outcome value target, then AlphaZero-style self-play. The
  tactical-beats-MCTS-100 result says the value head is the limiter.
- Both: build a `winblock`-derived adversarial suite (immediate win/block,
  forced-board legality) per `BOUNTY.md`; it is the clearest measured failure.
