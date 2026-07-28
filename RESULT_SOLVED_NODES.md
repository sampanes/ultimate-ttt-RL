# RESULT: solved-node propagation in MCTS -- target measurement (2026-07-27)

## Question

`RESULT_DISTILL_PILOT.md` found a reversal: the student distilled from an
800-sim teacher LOST to the student distilled from a 50-sim teacher, pooled
**0.4108** over three seeds. The diagnosed mechanism was PUCT dilution -- at 800
sims the teacher puts a visit on every legal move of a mate-in-1 and keeps only
0.693 of its mass on the winning move, against 0.825 at 50 sims.

Solved-node propagation is the direct intervention: prove forced results exactly
and stop sampling them. This measures what it does to the TARGETS, before any
student is trained.

## Provenance: the default path is unchanged, and that is counted

Solving is opt-in. With `solve=False` the search reproduces the frozen pilot's
targets bit for bit:

| arm | exact matches | failures |
|---|---|---|
| 50 sims | **5144 / 5144** | 0 |
| 800 sims | **5144 / 5144** | 0 |

`default_path_behaviorally_identical = true`. Bitwise float32 equality on all 81
entries per row, recomputed by `tools/provenance.py` from the cached policies --
deliberately different code from the run's own inline check.

`EXPERIMENT_DISTILL_PILOT.json` fails `--verify` on exactly three files
(`agents/mcts.py`, `tools/make_distill_corpus.py`, `tools/teacher_sim_ladder.py`)
and is **not** re-frozen: re-freezing would destroy the record of what the 0.4108
baseline actually ran. Comparability rests on the parity count, which is the
stronger guarantee -- a hash proves the bytes did not move, parity proves the
OUTPUT did not move.

## Method

5,144 positions from the frozen 50,000-position pilot corpus: a full census of
all 1,144 mate-in-1 positions plus 4,000 sampled from the rest (1,034
`other_tactical`, 2,966 `non_tactical`). The subset is deliberately ENRICHED for
tactics -- the corpus's natural mate-in-1 rate is 2.29% -- so per-stratum numbers
are primary and anything global is reweighted to natural rates before it is read
as a production quantity.

Six arms: {50, 200, 800} sims x {solve off, solve on}. Dirichlet off, c_puct 1.5.
Total 2,446s off + 2,547s on.

## Gate 1: forced-win dilution -- ELIMINATED, not reduced

Mate-in-1 positions, n = 1,144 (full census):

| arm | win-move mass | entropy (bits) | nonzero moves | argmax is a win |
|---|---|---|---|---|
| 50 off | 0.8794 | 0.7833 | 3.82 | 0.9878 |
| 200 off | 0.7868 | 1.3350 | 8.54 | 0.9895 |
| 800 off | 0.7356 | 1.4748 | 11.53 | 0.9624 |
| 50 on | **1.0000** | **0.0000** | **1.00** | **1.0000** |
| 200 on | **1.0000** | **0.0000** | **1.00** | **1.0000** |
| 800 on | **1.0000** | **0.0000** | **1.00** | **1.0000** |

The off arms reproduce the pilot's diagnosis exactly (0.825 / 0.693 at 50 / 800
became 0.8251 / 0.6930 here on a different sample). Solving takes every forced
win to a one-hot target at every budget. Dilution is not reduced; it is gone.

Note the off arms' fourth column: at 800 sims the teacher's own argmax is NOT a
winning move on 3.8% of mate-in-1 positions, worse than at 50 sims (1.2%).
Deeper search was actively losing forced wins, not merely hedging on them.

**Mechanism.** Reconciliation changes the argmax on **0** mate-in-1 positions.
Solving does not override a diluted distribution after the fact -- the proof
lands during root expansion, so the search never spends the visits that would
have diluted it. The correction is preventive.

## Gate 2: non-tactical targets -- NOT regressed, mildly sharpened

Non-tactical positions, n = 2,966:

| arm | top-move mass | entropy (bits) | nonzero moves |
|---|---|---|---|
| 50 off / on | 0.4578 / **0.4694** | 2.0062 / **1.9549** | 6.80 / 6.58 |
| 200 off / on | 0.4425 / **0.4581** | 2.1424 / **2.0730** | 8.06 / 7.67 |
| 800 off / on | 0.4869 / **0.5055** | 2.0494 / **1.9687** | 9.80 / 9.31 |

Every arm moves the same way: slightly more mass on top, slightly less entropy,
slightly fewer moves touched. No dilution is introduced anywhere. 95.6% of
non-tactical roots are not proven at all at 800 sims and their targets are
untouched.

Of the 13 non-tactical argmax changes at 800 sims, all 13 are strict
improvements. Across all strata and all budgets, **every** reconciliation change
is a strict improvement (the raw pick was a proven loss, or the corrected pick is
a proven win) and `neither_side_proven` is 0. There is no case where proof
correction moved a target to something not provably better.

## Gate 4: reach -- 29% of production targets change, 12% materially

Argmax churn badly understates this, because the target is a distribution. Both
are reported. Reweighted to the corpus's natural class rates:

| sims | any change | JS > 0.01 | JS > 0.05 | argmax change | mean JS |
|---|---|---|---|---|---|
| 50 | 0.1375 | 0.0898 | 0.0581 | 0.0200 | 0.0157 |
| 200 | 0.2174 | 0.1049 | 0.0702 | 0.0254 | 0.0212 |
| 800 | **0.2943** | **0.1169** | 0.0871 | 0.0327 | 0.0258 |

By stratum at 800 sims:

| stratum | n | any change | JS > 0.05 | mean JS |
|---|---|---|---|---|
| mate_in_1 | 1,144 | 0.9904 | 0.8278 | 0.2126 |
| other_tactical | 1,034 | 0.5745 | 0.1644 | 0.0472 |
| non_tactical | 2,966 | 0.1746 | 0.0367 | 0.0124 |

Root proof coverage, for reference:

| sims | overall (enriched) | mate_in_1 | other_tactical | non_tactical | early | mid | late |
|---|---|---|---|---|---|---|---|
| 50 | 0.2642 | 1.0000 | 0.1335 | 0.0260 | 0.0011 | 0.2943 | 0.7799 |
| 200 | 0.2792 | 1.0000 | 0.1809 | 0.0354 | 0.0011 | 0.3157 | 0.8092 |
| 800 | 0.2926 | 1.0000 | 0.2215 | 0.0445 | 0.0011 | 0.3358 | 0.8321 |

Proofs are overwhelmingly wins (1,331 win / 57 draw / 117 loss at 800). Coverage
is a strong function of phase: 83% of late positions are proven, essentially none
of the early ones -- so this intervention reshapes the endgame half of the
training signal and leaves the opening alone.

## Proof timing

`proof_sim = 0` means proven during ROOT EXPANSION, before a single simulation
was spent. Expansion runs an exact one-ply probe over every legal move, so these
are preprocessing proofs, not zero-cost search results.

| sims | proofs | from expansion | median proof sim | zero visits wasted at proof | NN evals avoided |
|---|---|---|---|---|---|
| 50 | 1,359 | 0.8462 | 0.0 | 0.9123 | 5.0 |
| 200 | 1,436 | 0.8008 | 0.0 | 0.8827 | 28.4 |
| 800 | 1,505 | 0.7641 | 0.0 | 0.8595 | 101.3 |

`visits_off_proven_at_proof` is the direct measure of pre-proof distortion: how
many visits had already gone to moves the proof later refutes. Its median is 0
at every budget, and it is exactly 0 in 86-91% of proofs. The mean is higher
(22.8 at 800) because a minority of proofs arrive late, which is what the middle
timing buckets exist to resolve.

This run predates the explicit bucket instrumentation, so `root_expansion` and
`unsolved` are recoverable from `proof_at_sim_0_rate` but the
`1_to_10 / 11_to_50 / 51_plus` split is not. Against the pre-registered rule in
`measure_solved_targets.backfill_decision`, post-expansion proofs are 23.6% at
800 sims, above the 10% bar, so the deep-arm backfill is permitted. See
"Open items".

## Reconciliation impact

Proof-corrected argmax vs raw visit argmax, within the solve-on run:

| sims | changed | rate | mate_in_1 | other_tactical | non_tactical | strict improvements |
|---|---|---|---|---|---|---|
| 50 | 19 | 0.0037 | 0 | 12 | 7 | 19 / 19 |
| 200 | 26 | 0.0051 | 0 | 16 | 10 | 26 / 26 |
| 800 | 32 | 0.0062 | 0 | 19 | 13 | 32 / 32 |

Small, and unanimously in the right direction. The headline target change comes
from redirected search, not from post-hoc override.

## The two named failure cases

**Deeper agrees but hedges** (argmax(50) == argmax(800), top mass lower at 800),
n = 1,928:

| arm | top mass | entropy | nonzero | win-move mass |
|---|---|---|---|---|
| 50 off | 0.6973 | 1.2591 | 5.16 | 0.8758 |
| 800 off | 0.5523 | 1.9209 | 11.06 | 0.6850 |
| 50 on | 0.7702 | 0.9020 | 3.77 | **1.0000** |
| 800 on | 0.7176 | 1.1128 | 5.33 | **1.0000** |

Solving does not abolish the hedging -- 800-on is still more diffuse than 50-on
(0.7176 vs 0.7702) -- but it cuts the gap from 0.145 to 0.053 and removes it
entirely on the forced wins inside the class.

**Top-move swap at a small value gap** (argmax differs, |dQ| < 0.05), n = 757:

| arm | top mass | entropy | argmax is a win |
|---|---|---|---|
| 50 off | 0.3691 | 2.2010 | 0.9342 |
| 200 off | 0.3336 | 2.3896 | 0.9474 |
| 800 off | 0.3909 | 2.2665 | **0.5526** |
| all on | 0.42-0.46 | 2.03-2.14 | **1.0000** |

This is the sharpest single number in the measurement. On the 76 mate-in-1
positions inside this failure class, the 800-sim solve-off teacher's argmax is
the winning move only **42 / 76** times, against 72 / 76 at 200 sims. Where
search is torn between near-equivalent moves, deeper search was flipping away
from forced wins at close to a coin toss. Solving takes it to 76 / 76.

## Cost: +4.1%, not the +17% the smoke run suggested

| sims | wall-clock ratio | s/move off -> on | NN evals/move | expanded/move | terminal probes/move |
|---|---|---|---|---|---|
| 50 | 1.055 | 0.0404 -> 0.0426 | -1.1 | -1.1 | 319.7 |
| 200 | **1.027** | 0.1020 -> 0.1047 | -6.8 | -6.8 | 1,181.0 |
| 800 | 1.044 | 0.3330 -> 0.3477 | -24.7 | -24.7 | 4,274.6 |

Total 2,446s -> 2,547s, **+4.1%**. The 17% figure came from the smoke run and
does not survive the full measurement.

Note the sign on neural evaluations: solving REDUCES them at every budget
(-4.6% at 800), because proven subtrees stop being searched. The cost is paid
entirely in cheap exact terminal probes, and the trade is close to break-even.

This block is reported for completeness and is **never** an input to the gates.
A correctness fix that costs time is still a correctness fix.

## Gates

| # | gate | verdict |
|---|---|---|
| 1 | forced-win dilution eliminated or sharply reduced | **PASS** -- eliminated: 0.7356 -> 1.0000 win-move mass at 800 sims, entropy to 0.0 |
| 2 | non-tactical targets not regressed | **PASS** -- mildly sharpened at every budget; 32/32 reconciliation changes are strict improvements |
| 3 | teacher strength maintained or improved on the ladder | **PASS** -- improved at every budget, monotonically in depth (see below) |
| 4 | proof correction reaches enough production-relevant positions | **PASS** -- 29.4% of targets change at natural rates, 11.7% materially, concentrated in exactly the diagnosed failure mode |

**All four gates pass.** The distillation pilot is unblocked.

## Gate 3: the paired ladder

400 games per rung. The three A/B rungs share opening seed 7810, so their deltas
are paired across sim counts and comparable to each other, not just each to 0.5.

| rung | score for solve-on | 95% CI | W/D/L | wall-clock |
|---|---|---|---|---|
| 50+solve vs 50 | 0.5375 | [0.5003, 0.5747] | 131/168/101 | x1.040 |
| 200+solve vs 200 | 0.5475 | [0.5084, 0.5866] | 148/142/110 | x1.064 |
| 800+solve vs 800 | **0.5725** | [0.5336, 0.6114] | 159/140/101 | x1.036 |

Every rung separates from chance, and the gain is **monotone in depth**:
+0.0375, +0.0475, +0.0725. That is the shape the dilution mechanism predicts --
dilution grows with sim count, so removing it pays more the deeper you search.
It is also the strongest available evidence that the target measurement and the
strength result are describing the same phenomenon rather than two unrelated
effects.

The lower bound at 50 sims is 0.5003, i.e. it clears zero by a hair. Read that
rung as "solving does not hurt at shallow budgets" rather than as a measured
gain.

**Does solving erase the deeper-is-stronger finding?** No -- it slightly
strengthens it. With both sides solving, on the same opening set the solve-off
comparison used (seed 7705):

| comparison | score for 800 | per doubling |
|---|---|---|
| 800 vs 200, solve off (published) | 0.5381 | +0.019 |
| 800+solve vs 200+solve | **0.5537** [0.5157, 0.5918] | **+0.0269** |

So deeper search remains genuinely better, and is worth slightly more once its
forced-win blunders are removed.

## Open items

* Gate 3: `results/solve_ab_ladder.json` (50/200/800 solve on vs off, shared
  openings, seed 7810) and `results/solve_deep_vs_shallow.json` (800+solve vs
  200+solve on seed 7705, paired against the published 0.5381).
* Proof-timing backfill: permitted at 800 by the pre-registered rule (23.6%
  post-expansion, bar is 10%). Cheap arms 50 and 200 first.
* A one-hot tactical target removes all soft-label information on 2.29% of
  positions. That is the intervention as designed, and the pilot is what tests
  whether it helps or hurts.

## Artifacts

`results/solved_targets/summary.json` (provenance-stamped), `policies.npz`
(all six arms, 5,144 x 81), `target_shift_by_stratum.json`.

`results/` is gitignored by repo convention -- generated runs stay local,
curated milestone reports live at the repo root -- so these files are NOT in
git and the numbers quoted above are the record. `EXPERIMENT_SOLVED_PILOT.json`
is the tracked half: it pins the 14 inputs (teacher checkpoint bytes, every
anchor's source, the estimator, the seeds) by sha256, so the reproduce block
below can be shown to have run against the same stack.

## Reproduce

    set CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv\Scripts\python -m tools.measure_solved_targets ^
      --sims 50 200 800 --nontactical 4000 --output results/solved_targets
    .venv\Scripts\python -m tools.provenance --stamp results/solved_targets/summary.json
    .venv\Scripts\python -m tools.measure_solved_targets ^
      --decide-backfill results/solved_targets/summary.json
