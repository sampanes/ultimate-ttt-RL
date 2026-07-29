# RESULT: solved-node distillation pilot -- NULL (2026-07-28)

Solving forced nodes exactly, which fixes the diagnosed defect completely,
recovers **0.97% of the distillation penalty**. The preregistered gate demanded
50%. It is not close, and it is not a power problem.

    pooled solved   0.4117   [0.3936, 0.4297]   2400 games
    published base  0.4108                      2400 games
    recovery        +0.0009  [-0.0172, +0.0189]
    threshold       +0.0446  (half of the 0.0892 penalty)

    MATERIALLY_SHRINKS = False

## What was being tested

`RESULT_DISTILL_PILOT.md` found that the student distilled from an 800-sim
teacher LOSES to the student distilled from a 50-sim teacher, pooled 0.4108.
The diagnosed mechanism was PUCT dilution: at 800 sims the teacher spends a
visit on every legal move of a mate-in-1, leaving only 0.693 of its policy mass
on the winning move against 0.825 at 50 sims. More search, blunter target.

Solved-node propagation is the direct intervention against exactly that.
`RESULT_SOLVED_NODES.md` established, before any student was trained, that it
does what it claims: win-move mass 0.7356 -> 1.0000 at 800 sims, entropy to
zero, 29.4% of production targets changed, and the teacher itself gets
*stronger* at every budget (0.5375 / 0.5475 / 0.5725 solve-on vs solve-off at
50 / 200 / 800). All four preregistered gates passed. This pilot asked the only
remaining question: does any of that reach the student?

It does not.

## The measurement

Three training seeds, two arms each (teacher at 50 sims vs 800 sims, both with
solving on), 800 head-to-head games per seed. Score is for the 800-sim
student, so 0.5 is parity and below 0.5 is the reversal persisting.

| seed | W | D | L | solved | baseline | delta |
|---|---|---|---|---|---|---|
| 11 | 278 | 112 | 410 | 0.4175 | 0.3331 | **+0.0844** |
| 22 | 197 | 124 | 479 | 0.3237 | 0.4650 | **-0.1413** |
| 33 | 325 | 140 | 335 | 0.4938 | 0.4344 | **+0.0594** |
| **pooled** | 800 | 376 | 1224 | **0.4117** | **0.4108** | **+0.0009** |

All four preregistered estimators agree, and every interval contains zero:

| role | estimator | delta | 95% CI |
|---|---|---|---|
| PRIMARY | pooled vs fixed 0.4108 | +0.0009 | [-0.0172, +0.0189] |
| CONFIRMATORY | paired by seed, game-level | +0.0008 | [-0.0238, +0.0255] |
| SENSITIVITY | two-sample, game-level | +0.0009 | [-0.0243, +0.0260] |
| GENERALIZATION WARNING (not a gate) | two-sample, seed-level | +0.0009 | [-0.0793, +0.0810] |

## This is a measured null, not an underpowered one

The distinction decides what the result is worth, so it was fixed in advance.
PRIMARY resolves 0.0180 against a threshold delta of +0.0446 -- the design had
roughly 2.5x the resolution it needed. A real half-penalty recovery would have
been seen. The estimator's own null calibration (baseline against itself,
+0.0000, gate False) and its at-threshold probe (all three gating reads exclude
zero) were both run before any solved student existed.

So the honest statement is not "we failed to detect an effect." It is: the
effect, if it exists at all, is at most about a fifth of what would have made
the intervention worth shipping.

## Pairing is licensed, not assumed

The published 0.4108 predates the change that records per-game outcome vectors,
so the old artifact carries seed totals only. The evaluation is deterministic --
match seed `9901 + seed`, students play with `sample_moves=0` -- so it was
replayed to recover the vectors. The replay reproduced the published counts
exactly:

    seed 11  193/147/460      seed 22  297/150/353      seed 33  261/173/366

Identical counts from an independent invocation is what licenses game-level
pairing rather than seed-level, and the estimator claims `GAME LEVEL` for all
three seeds on that basis. Had the replay drifted, the licence would have been
void and PRIMARY-vs-fixed-0.4108 would have been the only estimator left.

The corpora are paired at the byte level too: both arms hash to
`c0d695adca04e6ca`, identical to the original pilot's planes, with `z` identical
and `pi` different -- `SEEDS_SOLVED_PILOT_ARM_EQUALITY.json`. Same positions,
same value targets, different policy targets, nothing else.

## Read the seeds, but do not gate on them

Per-seed deltas are +0.0844, -0.1413, +0.0594: Q=60.386, I^2=0.967, spread
0.1700 (the baseline's own spread was 0.1319). One seed moved a seventh of a
point in the WRONG direction, larger in magnitude than any plausible treatment
effect.

This was anticipated and its role fixed before the data existed. Three highly
heterogeneous seeds cannot estimate how consistently anything generalizes
across training seeds, and the seed-level read is recorded as a warning, never
as a pass/fail. Its interval [-0.0793, +0.0810] is four times wider than
PRIMARY's, which is the whole reason it was demoted rather than used.

> The experiment can establish recovery for these preregistered matched
> training seeds. It cannot precisely estimate how consistently that recovery
> generalizes across arbitrary new training seeds.

The heterogeneity is consistent with the known checkpoint-wobble floor: a
single training run is one sample of a trajectory, and differences below about
0.05 cannot be separated from run-to-run luck.

## What this actually refutes

The dilution mechanism was real, measured, and is now fully repaired -- and the
student did not care. So the diagnosis was incomplete: **mate-in-1 policy
dilution is not what makes the 800-sim teacher a worse teacher.**

Three things in the prior work already pointed here, and this measurement
promotes them from suggestive to load-bearing:

* reconciliation changed **0** mate-in-1 argmaxes. The proof lands during root
  expansion, so the diluting visits were never spent. The fix is preventive; it
  never had a corrupted decision to correct.
* the churn study found distributions CONVERGE with more search while argmax
  churn stays flat -- search improvement is ~7x overstated by move-level
  disagreement.
* raising teacher sims already distilled worse across the board, all four
  anchors, at identical capacity.

The remaining candidates are about what the 800-sim distribution looks like
*away* from forced wins -- sharpness, temperature, the value head, or the
match between teacher and student capacity -- not about tactical correctness.
Solving fixed correctness and bought nothing.

## What it does not refute

Solving is still a strict improvement to the TEACHER: +0.0375 / +0.0475 /
+0.0725 at 50 / 200 / 800 sims, monotone in depth, at +4.1% wall clock with
*fewer* NN evaluations. That result stands on its own and is unaffected by this
null. Solving is worth keeping wherever the search itself is the product; it is
simply not the lever for distillation.

## Provenance

`EXPERIMENT_SOLVED_PILOT.json`, frozen 2026-07-28T08:12:35 at commit `0a2f45d`
-- before corpus generation started -- pins 14 inputs by sha256 including
`teacher.pt`, `mcts.py`, `make_distill_corpus.py` and the estimator itself.
`--verify` reports 14/14 unchanged.

`tools/pooled_estimator.py` is one of the frozen 14 and was **not** edited after
the result appeared. The evaluator and the estimator disagree about JSON shape;
that gap is bridged by `tools/adapt_eval_for_estimator.py`, which selects a
block and copies counts through, asserting each seed's outcome vector
reproduces its own W/D/L. A plumbing edit to a frozen estimator is exactly what
goalpost-moving looks like from the inside, so the data was brought to the
estimator instead.

`SEEDS_SOLVED_PILOT.json` locked seeds [11, 22, 33] and the full training
config before targets existed; `--check` still reproduces. The training driver
refused to proceed unless both arms of a seed reported identical init
fingerprints, and all three matched the lock file (-14.1126 / 7.5535 /
-13.2910).

## Cost

Corpus generation 9.1h (arm 50 at 2356s, x1.094 over solve-off; arm 800 at
28,290s, x1.113). Six students at ~4 min each. Evaluation ~150s per arm.

## Artifacts

`results/distill_pilot_solve/` (primary.json, flat.json, estimator.json),
`results/distill_pilot_baseline_replay/` (primary.json, outcomes.json),
`models/distill_pilot_solve/`. `results/` and `models/` are gitignored by repo
convention, so the numbers quoted here are the record;
`EXPERIMENT_SOLVED_PILOT.json` and `SEEDS_SOLVED_PILOT_ARM_EQUALITY.json` are
the tracked halves.

## Reproduce

    set CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv\Scripts\python -m tools.make_distill_corpus --corpus models/corpus_gen22 ^
      --checkpoint models/expert_iter_v2/teacher.pt --out models/distill_pilot_solve ^
      --positions 50000 --sims 50 800 --seed 20260727 --solve ^
      --expect-x-sha256 c0d695adca04e6ca1996474070ef77b89f026475020b67d7293471a432211d1e
    .venv\Scripts\python -m tools.lock_student_seeds ^
      --corpora models/distill_pilot_solve/sims50 models/distill_pilot_solve/sims800 ^
      --out SEEDS_SOLVED_PILOT_ARM_EQUALITY.json
    .venv\Scripts\python -m tools.run_distill_pilot --pilot models/distill_pilot_solve ^
      --arms 50 800 --seeds 11 22 33
    .venv\Scripts\python -m tools.eval_distill_pilot --pilot models/distill_pilot_solve ^
      --seeds 11 22 33 --arch squeeze --students-dir students ^
      --arm-a 800 --arm-b 50 --games 800 --primary ^
      --output results/distill_pilot_solve/primary.json
    .venv\Scripts\python -m tools.adapt_eval_for_estimator ^
      --eval results/distill_pilot_solve/primary.json ^
      --out results/distill_pilot_solve/flat.json
    .venv\Scripts\python -m tools.pooled_estimator ^
      --results results/distill_pilot_solve/flat.json ^
      --baseline-outcomes results/distill_pilot_baseline_replay/outcomes.json
