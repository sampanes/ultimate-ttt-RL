# RESULT: stronger teachers make WORSE distillation targets (2026-07-27)

## Question

`RESULT_TEACHER_SIM_LADDER.md` established that deeper MCTS is genuinely
stronger (+0.019 win rate per doubling of simulations). Does that strength
transfer through distillation into a small student?

Pre-registered in `EXPERIMENT_DISTILL_PILOT.json`, frozen at commit 7c2d4d1
before a single target was generated: 12 file hashes including `teacher.pt`
bytes and all four anchor implementations, plus the full corpus, student and
evaluation config. `--verify` still passes.

## Design

50,000 positions drawn uniformly from the gen-22 corpus (natural phase
distribution, NOT stratified -- a phase-balanced student learns a different
function from the one the pipeline produces). The frozen gen-22 teacher replayed
MCTS over those same positions at 50 and 800 simulations.

The arms are identical except the policy target:

| | |
|---|---|
| `x` | byte-identical, verified by re-hashing from disk |
| `z` | identical -- the ORIGINAL corpus game outcome, not either arm's root Q |
| `pi` | THE TREATMENT -- each arm's own full visit distribution |
| init weights | identical within a seed (`--init_seed`, fingerprint-checked) |
| batch order, symmetry draws, LR schedule, steps | identical |

Three seeds per arm, so training variance is measured rather than assumed.
Generation cost 7.66 h (50-sim arm 2,153 s, 800-sim arm 25,411 s).

## Result: the 800-sim arm is WORSE, and it is not non-transitivity

Student 172,389 params (`squeeze`), raw argmax, no search.

| | score for 800-arm | n | p |
|---|---|---|---|
| head to head vs 50-arm | **0.4108 [0.3932, 0.4284]** | 2,400 | <1e-4 |

| external anchor (300 games/cell, arm means) | 50-sim | 800-sim | delta |
|---|---|---|---|
| random | 0.9839 | 0.9728 | -0.0111 |
| winblock | 0.8261 | 0.8150 | -0.0111 |
| gregory_d3 | 0.6517 | 0.5544 | -0.0972 |
| gregory_d4 | 0.5200 | 0.4094 | **-0.1106** |

The 50-sim arm wins on ALL FOUR frozen anchors, so the head-to-head is not a
non-transitivity artifact -- a real risk in this repo, which is why the external
ladder was pre-registered. **The deficit scales with opponent strength**:
negligible against random, 11 points against gregory-d4. That is the signature
of a specific weakness only strong opponents can punish.

## Mechanism: more search DILUTES decisive positions

The two arms differed in two variables, not one. The second is a mechanical
consequence of PUCT: the exploration bonus scales with `sqrt(N_total)`, so more
simulations put more ABSOLUTE visits on moves already known to be lost.

| target top-move mass | 50-sim | 800-sim | nonzero moves (50 / 800 / legal) |
|---|---|---|---|
| mate-in-1 (n=1,144) | 0.8251 | **0.6930** | 3.82 / **11.53** / 11.53 |
| no mate-in-1 (n=48,856) | 0.4575 | 0.4859 | 6.79 / 9.65 / 9.66 |
| all | 0.4659 | 0.4906 | 6.73 / 9.70 / 9.70 |

On forced wins the 800-sim search puts a visit on EVERY legal move and smears
31% of its mass across provably-losing replies. Both teachers pick the same move
93% of the time there -- the deeper teacher is not wrong, it is **diluted**.

**More simulations sharpen ambiguous positions and soften decisive ones.** The
effect points in opposite directions by position type, which is why a single
global temperature cannot correct it (the solver hits its T=1.0 boundary: the
800 target is already sharper on average).

The students inherit exactly this. Agreement with the 800-sim target on
mate-in-1 positions: **50-arm 0.7864, 800-arm 0.6807**. The 800-arm student is
10.6 points worse at finding forced wins, and forced wins decide games.

## Control 1 -- fix the temperature: recovers ~40%, not more

Per-position temperature matching (`tools/sharpen_distill_corpus.py
--per-position`), solving T_i so each row's top-move mass equals the 50-sim
arm's. Median T 1.083, p10 0.51, p90 2.05 -- genuinely bidirectional. Match is
exact (mate-in-1 mass 0.8246 vs 0.8251 reference) and argmax is preserved 1.0000.

| | 800 raw | 800 temp-matched | 50-sim |
|---|---|---|---|
| h2h vs 50-arm | 0.4108 | **0.4450** | -- |
| gregory_d4 | 0.4094 | **0.4572** | 0.5200 |
| gregory_d3 | 0.5544 | 0.5806 | 0.6517 |

Recovers 38% of the head-to-head gap and 43% of the gregory-d4 gap -- two
independent measurements agreeing on the size of the temperature contribution.
**The majority of the deficit is not temperature.**

## Control 2 -- 5.3x the student: the gap does not move at all

`modern`, 921,688 params vs 172,389. Same three paired seeds, same schedule.

| | 172k student | 921k student |
|---|---|---|
| h2h 800-arm vs 50-arm | 0.4108 [0.3932, 0.4284] | **0.4202 [0.4028, 0.4376]** |
| across-seed spread | 0.132 | **0.044** |
| gregory_d4 delta | -0.1106 | **-0.1105** |

| gregory_d4 | 50-sim | 800-sim |
|---|---|---|
| 172k | 0.5200 | 0.4094 |
| 921k | 0.5744 | 0.4639 |

Capacity lifts BOTH arms by about +0.055 and leaves the gap between them
untouched to four decimal places. The across-seed spread also collapses from
0.132 to 0.044, so the larger model determines the result far better -- and it
determines the same result.

**Student capacity is not the binding constraint.** More teacher search produces
worse distillation targets regardless of student size.

## What this means

Three measured facts that only look contradictory:

1. Deeper search plays better (+0.019/doubling, `RESULT_TEACHER_SIM_LADDER.md`).
2. Deeper search changes ~15% of its moves per doubling, and the moves it swaps
   differ in value by only 0.013 (`RESULT_SEARCH_DISAGREEMENT.md`).
3. Deeper search's visit distribution is a worse thing to imitate.

Over the four doublings from 50 to 800, 33.8% of targets change while the
teacher gains roughly 7.6 points. The student must spend capacity fitting all
33.8% of that churn, most of which is near-equivalent reshuffling, and it comes
out net worse. Distillation imitates the DISTRIBUTION, not the STRENGTH, and
those two things come apart as search deepens.

**Raising teacher simulations is not the lever for a small, continually
improving model.** Two directions survive:

- **Search quality instead of quantity.** Symmetry folding, transposition
  merging, cross-move tree reuse, solved-node propagation -- all still
  unimplemented per `MCTS_STATUS.md`. These would produce better targets at
  equal or lower sim count rather than more diluted ones.
- **Target post-processing.** Per-position sharpening recovered ~40% of a
  deliberately induced gap. Whether it also improves the PRODUCTION 200-sim
  targets is untested and should not be assumed -- it is a cheap experiment
  (3 training runs plus an evaluation) and is the obvious next thing to try.

## Scope and honesty notes

- The pre-registered claim is the primary head-to-head and the frozen ladder.
  Everything from "Mechanism" onward is post-hoc inspection, run under the
  pre-registered decision rule for a reversal ("inspect whether higher-simulation
  policies are softer, noisier, or harder for the fixed student to fit").
- Three seeds is thin. At 172k the across-seed spread (0.132) exceeded the
  effect; the direction was consistent in all three seeds but the magnitude was
  not well determined. The 921k replication with spread 0.044 is what makes the
  conclusion solid.
- `z` was held identical across arms, so this measures the POLICY target only.
  Each arm's root Q is stored in `index.npz`; a value-target variant costs no
  regeneration.
- A 200-sim arm was deliberately NOT generated. The 800-vs-200 teacher gap
  (~4 points) sits under the ~0.05 panel floor.

## Artifacts

`models/distill_pilot/` (both corpora, `index.npz`, `manifest.json`),
`results/distill_pilot_eval.json`, `results/distill_pilot_eval_modern.json`,
`EXPERIMENT_DISTILL_PILOT.json`.

## Reproduce

    .venv\Scripts\python -m tools.make_distill_corpus --positions 50000 --sims 50 800
    .venv\Scripts\python -m tools.run_distill_pilot
    .venv\Scripts\python -m tools.eval_distill_pilot --primary --policy
    .venv\Scripts\python -m tools.eval_distill_pilot --ladder --ladder-games 300
    .venv\Scripts\python -m tools.sharpen_distill_corpus --arm 800 --match-arm 50 ^
      --per-position --suffix tmatch
