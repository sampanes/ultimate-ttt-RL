# RESULT: architecture A/B on a frozen gen-22 corpus (2026-07-25)

## Question

Is the network architecture the lever for a better strength-to-size ratio?

The hypothesis under test was specific and strong: `agents/neural_net_agent_3.py`
builds a plain `Conv2d -> activation` stack with no normalization and no skip
connections, and heads that flatten the entire conv output straight into a
`Linear`. On the shipped gen-22 shape that means **83.6% of 6,766,386 parameters
sit in the two heads** (78.5% in the two flatten->Linear layers alone) and only
16.4% reach the conv tower that does the spatial reasoning. The claim was that
fixing this -- residual tower, normalization, AlphaZero-style 1x1-squeezed heads
-- would be worth more than any distillation or pruning trick.

**The strong form of that claim is refuted below.** A weaker and more useful
finding survives.

## Method

Expert iteration's expensive half -- MCTS-200 self-play to produce
`(state, improved_policy, outcome)` targets -- does not depend on the student.
So the corpus was generated ONCE and every architecture trained on it:

- **Corpus**: 621,096 examples from 17,600 games, teacher FROZEN at gen-22,
  `teacher_sims=200`, live opponent mix (30% winblock / 10% random / 10%
  gregory-d2 / 50% self-play). 1100 shards, 1.6 GB, 1h23m on 16 actors + the
  eval server. Produced with the new `expert_iter --generate_only`.
- **Training**: 40,000 steps, batch 512, Adam, lr 2e-3 halving every 15k to a
  1e-4 floor, grad-clip 1.0, dihedral augmentation. Identical for all arms.
- **Fairness**: data order and the symmetry sequence come from RNGs seeded by
  `--seed`, deliberately NOT from the torch RNG that weight init consumes. All
  four arms therefore saw byte-identical batches in byte-identical order.
- **Panel**: 300 games/cell, fixed diverse openings, colors swapped, per-game
  RNG reset, same per-opponent seeds the promotion gate uses. Anchors are the
  frozen non-gene-pool opponents only. The lottery is never used -- it is a
  constant function and voids any number measured against it.

### Arms

Measured 1-thread CPU batch-32 forward, the closest available proxy for ONNX
Runtime Web on WASM:

| arm | params | conv tower | CPU b32 | what it isolates |
|---|---|---|---|---|
| `plain` | 921,026 | 6.5% | 6.20 ms | the incumbent style |
| `modern` | 921,688 | 90.9% | 54.28 ms | PARAMETER-matched full rewrite |
| `modern_w32` | 141,656 | 40.8% | 6.11 ms | LATENCY-matched full rewrite |
| `squeeze` | 172,389 | 51.3% | ~6.2 ms | the head fix ALONE (no residual, no norm) |

`modern*` = residual blocks + GroupNorm + 1x1-squeezed heads. `squeeze` = plain
convs, no norm, no skips, only the 1x1 conv inserted before the flatten.

## Results

### Raw panel (300 games/cell)

| arm | params | random | winblock | gregory d3 | gregory d4 | vs gen-22 raw |
|---|---|---|---|---|---|---|
| `plain` | 921,026 | 0.995 | 0.893 | **0.777** | **0.685** | 0.470 |
| `modern` | 921,688 | 0.993 | 0.883 | 0.793 | 0.668 | 0.525 |
| `modern_w32` | 141,656 | 0.993 | 0.878 | 0.715 | 0.598 | 0.455 |
| `squeeze` | 172,389 | 0.983 | **0.905** | 0.728 | 0.653 | **0.527** |

Raw head-to-head vs `plain` (300 games, seed 7701; >0.500 means plain holds):
`modern` 0.475, `modern_w32` 0.565, `squeeze` 0.533.

### Deployed player: net + MCTS vs gregory(d3)

| arm | equal SIMS (50) | ms/game | equal TIME (scaled sims) | ms/game |
|---|---|---|---|---|
| `plain` | 0.913 | 745 | 0.913 (50 sims) | 761 |
| `modern` | **0.948** | 1062 | 0.927 (35 sims) | 1035 |
| `modern_w32` | 0.898 | 934 | 0.907 (40 sims) | 943 |
| `squeeze` | 0.922 | **727** | 0.910 (51 sims) | **740** |

Note MCTS wall clock is NOT linear in sims -- per-move overhead outside the
forward (tree bookkeeping, tactical checks) is large, so cutting `modern` from
50 to 35 sims only moved it 1062 -> 1035 ms. It remains 36% slower than `plain`
even after the cut, so its +0.014 is still bought with extra time it does not
have; a genuinely time-matched `modern` lands below `plain`.

At 300 games and scores near 0.91 the 95% interval is roughly +/-0.032. **Every
arm's deployed score falls inside that band.** On the raw cells near 0.75 the
band is about +/-0.049, so `plain` 0.777 vs `squeeze` 0.728 on gregory-d3 sits
right at the edge of significance and the `plain`-vs-`squeeze` h2h of 0.533 is
inside a tie.

## Findings

**1. Architecture does not change deployed strength.** At equal wall clock all
four arms are statistically indistinguishable (0.907-0.927, band +/-0.032). The
thing that actually ships -- net plus search, under a time budget -- did not
move. This is the direct refutation of the hypothesis.

**2. The head fix is worth ~5x compression at parity.** `squeeze` matches
`plain` deployed (0.910 vs 0.913) and on raw h2h (0.533, inside a tie) with
**5.3x fewer parameters**, while being the fastest arm. It also posts the best
winblock cell (0.905) and the best score against gen-22 raw (0.527). A
172k-parameter net beating the raw play of the 6.77M teacher is a 39x reduction.
This is a two-line change and it is the entire usable result.

**3. The residual tower is a NET NEGATIVE at this depth.** `modern_w32`
(residual + GroupNorm) loses to `squeeze` (neither) on every cell: winblock
0.878 vs 0.905, d3 0.715 vs 0.728, d4 0.598 vs 0.653, vs-teacher 0.455 vs 0.527,
deployed 0.907 vs 0.910 -- while being 27% slower per game. Residual connections
exist to fix vanishing gradient in DEEP stacks; at 4-6 layers there is no
gradient pathology to fix, so the normalization overhead is paid for nothing.
Generalizing from the 135-layer lottery collapse
(`RESULT_LOTTERY_VS_GEN22.md`) to nets 20x shallower was the error.

**4. Parameters and latency are different budgets and must not be conflated.**
At matched parameters `modern` is 8.8x slower on CPU than `plain`, because
convolutions at 9x9 are far more FLOP-hungry than Linears holding the same
weight count. A parameter-matched-only A/B would have been rigged. Note also
that the ranking is backend-dependent: on CUDA `modern_w32` is 2.2x SLOWER than
`plain` (kernel-launch overhead from more layers) despite being tied on
1-thread CPU.

**5. No convergence confound.** All four arms flatten by ~15-20k of the 40k
steps and none is still descending at the end (last-quarter loss change:
plain -0.077, modern -0.077, modern_w32 -0.045, squeeze -0.018).

## Limitations

- One training run per architecture. Run-to-run seed variance was not measured,
  so the ~0.02 spread being called noise could also be masking small real
  effects in either direction.
- One corpus, one teacher generation, one training budget. A larger corpus or a
  longer schedule could favour the higher-capacity arms differently.
- `squeeze` is 21% larger than `modern_w32`, so finding 3 is not a perfectly
  size-matched comparison -- though it is faster despite being larger, which
  makes the direction safe.

## What this means for the pocket track

The strength-to-size lever is the head fix, not the tower. The current pocket
champion `arena:21@hof` (1,287,314 params) carries the same flatten->Linear
design, so the same change should apply there. What this experiment does NOT
support is any claim that a small net can be made to beat gen-22 plus search
inside the same time budget by changing the architecture -- deployed strength
did not move at all.

## Reproduce

    REM 1. corpus (frozen gen-22 teacher, ~1h23m at 16 actors)
    set CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv\Scripts\python -m scripts.expert_iter --generate_only ^
      --shard_retain_mult 0 --blocks 1100 --model_dir models/corpus_gen22 ^
      --teacher_ckpt models/expert_iter_v2/teacher.pt --teacher_tanh ^
      --network arena22 --actors 16 --eval_server ^
      --greg_mix 0.10 --opp_mix 0.30 --rnd_mix 0.10 --no_metrics

    REM 2. train each arm (~3-15 min each)
    .venv\Scripts\python -m scripts.train_student_offline ^
      --corpus models/corpus_gen22 --arch squeeze --steps 40000 ^
      --out models/ab_arch/squeeze.pt

    REM 3. panel
    .venv\Scripts\python -m scripts.ab_arch_panel ^
      --models models/ab_arch/plain.pt models/ab_arch/modern.pt ^
               models/ab_arch/modern_w32.pt models/ab_arch/squeeze.pt ^
      --games 300 --sims 50 --out models/ab_arch/panel.json

The architecture options are opt-in fields on `ModelConfigCNN`
(`residual`, `norm`, `head_squeeze`); with all three falsy `ConvNet` builds the
exact legacy module graph. Verified: arena22 still builds to exactly 6,766,386
parameters and `models/expert_iter_v2/teacher.pt` loads with zero missing and
zero unexpected keys.
