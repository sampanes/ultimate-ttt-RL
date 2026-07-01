# RESULT -- M3 prep: pocket finalist size + move latency

*Run on the RTX 3080 home box, 2026-06-30, commit `0409cde`. Produced by a
scratchpad harness that reuses the benchmark suite's architecture resolver
(`resolve_candidate` / `_build_base_candidate`, which read the arch from
`models/arena/arena_state.json`) plus a single-position latency probe. No repo
code was modified. These are the measured inputs M3's exit gate asks for
("static client runs the chosen quantized net plus search with measured size
and latency") -- captured up front so M3 starts with numbers in hand.*

## Candidate

`arena:21@hof` -- the M2 pocket base.

- arch: conv=`[32, 128, 32, 32]` fc=`[128, 256, 512]`
- parameters: **1,287,314**
- fp32 checkpoint: 5,156,981 bytes (**5.16 MB**)

## Size -- int8 estimate (the 5 MB gate)

| Quantity | Value |
|---|---:|
| weight params (conv+linear, -> int8 @ 1 byte) | 1,285,856 |
| other params (bias/norm, kept fp32 @ 4 bytes) | 1,458 |
| **int8 size estimate** | **~1.29 MB** |
| M3 gate | <= 5 MB |
| Verdict | **PASS** (4x headroom) |

This is a theoretical floor (weights to int8, small fp32 tails). The real ONNX
static-int8 file will add per-tensor scale/zero-point and graph overhead, but
4x headroom means the model byte gate is not the constraint -- the runtime/WASM
bundle is.

## Latency -- single-position inference (browser-style, one move at a time)

5,373 organically-reached candidate-to-move positions (200 games vs `center`,
seed 0). Raw policy, no search. 30-move warmup, then wall-clock per
`select_move` (which calls `.item()`, forcing device sync).

| Device | mean | median | p95 |
|---|---:|---:|---:|
| CPU  | 0.824 ms | 0.774 ms | 1.208 ms |
| CUDA | 1.242 ms | 1.080 ms | 1.840 ms |

**CPU is faster than CUDA here, and that is the expected, useful result.** At
batch-of-1 the GPU kernel-launch + host/device sync overhead dominates the tiny
matmuls; the GPU only wins at the batched throughput the browser never does.
The browser plays one move at a time, so the CPU number is the relevant one.

## Implication for M3

- The raw-policy forward is **sub-millisecond on CPU**. A WASM CPU inference
  path can be responsive on its own; WebGPU is a nice-to-have, not a
  requirement for a playable page. Build CPU/WASM first, treat WebGPU as
  optional acceleration.
- The per-move search budget, not the net forward, will set the move latency.
  At ~0.8 ms/forward on CPU, an N-simulation MCTS costs roughly `N x 0.8 ms`
  plus tree overhead -- e.g. ~40-80 ms for 50-100 sims, which is comfortably
  interactive. Pick the browser sim budget against the actual WASM forward time
  once ported, not against this native CPU number.
- The model byte gate (<= 5 MB) is met with 4x room. The 10 MB compressed
  bundle gate will be dominated by the inference runtime (ONNX Runtime Web /
  WASM), so that is where to measure next.

## Reproduce

The harness lives in scratch (it is glue, not repo code); the measured agent is
`arena:21@hof`. To re-derive size from committed tooling:

```
python -m scripts.benchmark_suite --candidate arena:21@hof --anchors center \
    --candidate-sims 0 --openings standard --out results/arena-21
```

(reports parameter count + checkpoint hash/bytes; the int8 split and latency
probe are the scratchpad add-on this report captured).
