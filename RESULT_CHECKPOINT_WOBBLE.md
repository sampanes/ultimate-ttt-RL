# RESULT: checkpoint wobble is real; averaging it away did not work (2026-07-26)

## Part 1: the teacher is a lucky checkpoint (free, from the metrics log)

Chasing why a student distilled from teacher+MCTS-200 measures *weaker* than its
teacher on winblock while beating it head to head, the answer turned out to be
selection, not strength.

- The gen-22 teacher scores 0.9267 vs winblock. Of the 127 student checkpoints
  measured during gen 22, **3 matched or beat it** -- the teacher sits at the
  **98th percentile of the student distribution**, not above it.
- Across all 22 generations, the promoting attempt averages the **76th
  percentile** of its own generation's attempts, with many generations at 100%.
  If promotion were unrelated to winblock this would be 50.
- The old gate required a high winblock score to promote, so every teacher in
  the lineage is by construction a peak of whatever wobble training was doing.
  Early on, when real improvement was fast, ordinary checkpoints cleared the
  bar (gen 0: 20th percentile). As improvement slowed the gate increasingly
  selected wobble peaks instead, until at gen 22 even the peak (0.9317) could
  not clear 0.9467 and the lineage stopped entirely.

**The wobble is fast and wide.** Span 0.080 (0.8517-0.9317), lag-1
autocorrelation **+0.120** at the ~28,000-step spacing between checks, trend
across all 127 attempts +0.005 (flat).

**It is not a learning-rate artifact.** 98% of gen-22 blocks ran at the 1e-4 LR
floor with the loss flat (2.0311 -> 2.0261 across the halves). The weights are
barely moving in loss terms while the panel swings 8 points, because the panel
is played with raw argmax: near-tied policy logits flip on tiny weight changes,
and one flipped move early in a game changes the whole game. **The panel score
is a chaotic function of the weights, not a smooth one.**

**Why this looked worth chasing:** picking a p90 checkpoint over a median one is
worth +0.028 on gregory-d3 and +0.037 head to head, against 0.020 for every
architecture in RESULT_ARCH_AB.md combined -- and unlike architecture it is free
at inference.

## Part 2: averaging the wobble away -- tested, did not work

**Prediction:** an averaged net should land at or above p90 on *all* panels at
once, where any single checkpoint is p90 on one and p50 on another.

**Method.** `squeeze` (172,389 params), 40,000 steps on the frozen gen-22
corpus, with a constant-LR tail from step 25,000 at 5e-4 -- the standard SWA
recipe, and necessary here because the normal schedule decays to 1e-4 and leaves
no wobble to average. Six snapshots every 3,000 steps, their mean (SWA), and an
EMA at decay 0.999. Panelled at 300 games/cell.

Weight-space check first: consecutive snapshots are 5.5-6.2 apart in L2, 7-8% of
the weight norm, and first->last (16.07) is well under the sum of the consecutive
steps (28.9) -- a wandering trajectory that partly cancels, which is the regime
where averaging is supposed to pay.

| checkpoint | winblock | gregory d3 | h2h vs final |
|---|---|---|---|
| snap40000 (plain final) | 0.885 | 0.778 | -- |
| **SWA** (mean of 6) | 0.877 | **0.783** | 0.533 loses |
| **EMA** (0.999) | 0.867 | 0.765 | 0.512 loses |
| snap25000 | 0.895 | 0.740 | 0.537 loses |
| snap28000 | 0.880 | 0.688 | 0.535 loses |
| snap31000 | 0.875 | 0.717 | 0.512 loses |
| snap34000 | 0.887 | 0.755 | 0.543 loses |
| snap37000 | 0.873 | 0.777 | 0.525 loses |
| squeeze.pt (A/B baseline, decayed LR) | **0.905** | 0.728 | **0.478 WINS** |

Snapshots alone: winblock mean 0.8825 (range 0.873-0.895), gregory-d3 mean
0.7425 (range 0.688-0.778).

**The prediction is refuted.** SWA is the best net on gregory-d3 -- 0.783, above
all six snapshots -- but it is *below median* on winblock and it loses the
head-to-head against the plain final at 0.533. By this repo's own tiebreak rule
(when panel cells disagree, a fixed-opening colour-swapped h2h decides), the
plain final wins and averaging loses. EMA is below median on both.

Two things worth recording about the experiment itself:

- **EMA was mis-specified.** Decay 0.999 is a ~1000-step window, shorter than
  the 3000-step snapshot spacing, so it sat 2.38 from the final snapshot and
  15.17 from the first. It was a lightly-smoothed final, not an average. 0.9999
  would have been the matched setting; its underperformance is not evidence
  against EMA in general.
- **The constant-LR tail cost more than averaging recovered.** The original
  decayed-LR baseline beats the constant-LR final head to head (0.478) and has
  the best winblock cell of anything tested (0.905). The manipulation needed to
  *create* measurable wobble made the net worse.

## What this means

The honest conclusion across two experiments now (architecture, and this one) is
a calibration result: **effects below roughly 0.05 on these panels are not
reliably attributable.** The panels only weakly agree with each other -- winblock
vs gregory-d3 correlate at **+0.266**, head-to-head vs gregory-d3 at +0.469 --
so a net that wins one cell routinely loses another, and with a single training
run as the unit of evidence there is no way to separate a real 0.02 from
trajectory luck. Resolving effects that size needs many replicate runs, which
costs more than the effects are worth.

What has produced large, reproducible effects, and where the effort belongs:

1. **`head_squeeze`** -- 5.3x fewer parameters at parity (172,389 matching
   921,026), measured, validated, and still unshipped on the pocket track. This
   is the actual "tiny" win and it is already in hand.
2. **The promotion gate fix** (RESULT_GATE_PLATEAU.md) -- 0% -> 39% promotion
   rate. The engine had been off for 39% of the project's lifetime.
3. **Generations**, historically the only lever with a large measured effect:
   gen-13 -> gen-15 was +13.5 points at fixed search.

Checkpoint selection remains genuinely worth +0.03, but the way to capture it is
to make the gate stop selecting on a saturated panel -- already done -- not to
average weights.

## Reproduce

    .venv\Scripts\python -m scripts.train_student_offline ^
      --corpus models/corpus_gen22 --arch squeeze --steps 40000 ^
      --swa_from 25000 --swa_lr 5e-4 --snapshot_every 3000 ^
      --ema_decay 0.999 --out models/ab_arch/swa_squeeze.pt

    .venv\Scripts\python -m scripts.ab_arch_panel ^
      --models models/ab_arch/swa_squeeze.snap40000.pt ^
               models/ab_arch/swa_squeeze.swa.pt ^
               models/ab_arch/swa_squeeze.ema.pt ^
      --opponents winblock,gregory_d3 --games 300
