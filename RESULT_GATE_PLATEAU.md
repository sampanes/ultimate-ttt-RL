# RESULT: the gen-22 plateau was the promotion gate (2026-07-25)

## Question

Expert iteration had not promoted a teacher in **564,144 games** -- 39% of the
project's lifetime total of 1,439,792. Is the lineage out of room, or is
something in the loop broken?

## Finding 1: the loop is not out of fuel

`mcts_edge` is logged at every gate: search over the raw student, using the
student's own head. It is the entire teaching signal -- if the net could already
reproduce what its own search finds, distillation would have nothing to give.

| teacher_gen | 0 | 5 | 10 | 15 | 20 | 22 |
|---|---|---|---|---|---|---|
| mcts_edge | 0.877 | 0.825 | 0.822 | 0.813 | 0.803 | 0.811 |

Search still beats the raw net 81% of the time, down only 0.066 across 22
generations. There is plenty left to distill.

## Finding 2: the gate was not merely improbable, it was impossible

127 promotion attempts were logged at gen 22. Replaying `_promotion_decision`
against the live bars in `state.json`: **zero should have passed.**

| criterion | bar | failed | sole blocker |
|---|---|---|---|
| winblock | 0.9467 | **100%** | **52** |
| head_to_head | 0.5500 | 54.3% | 0 |
| gregory d3 | 0.7833 | 26.8% | 0 |
| random | 0.9650 | 0% | 0 |

**These panels are deterministic.** `_play_fixed_match` reseeds python and numpy
per game and promotion runs raw argmax (`sample_moves=0`), so a panel score is a
reproducible function of the weights. Measured three times against gen 22 it
returned 0.926667 every time, matching the stored `best_heur` to the bit. There
is no sampling noise here and nothing to get lucky on.

That makes the arithmetic stark. The winblock bar was `best_heur +
promote_margin`, and `best_heur` is **the teacher's own score**, 0.9267. So the
gate asked the student to beat the teacher by 0.02 on a fixed heuristic. Across
those 127 attempts the students averaged **0.9002** -- 0.0265 *worse* than the
teacher -- so they needed +0.047. The best student the run ever produced reached
0.9317, still short of 0.9467. Not unlikely: impossible, for every student
measured.

Meanwhile head_to_head averaged **0.545** and was above 0.500 on every single
attempt. The students really were beating the teacher.

## Root cause: a non-transitivity the gate could not express

The students are **better than the teacher head to head and slightly worse than
it against a fixed heuristic**. Both facts are deterministic and reproducible;
neither is noise. A gate that requires improvement on *both* is requiring
something the lineage does not produce.

Compounding it, winblock was carrying two incompatible jobs -- prove the student
improved, and guard against regression -- on a metric that is bounded, saturated
near 0.93, and where much of the residual against a 1-ply heuristic is
opening-determined rather than skill.

## The fix

**One criterion decides improvement, and it is head_to_head.** It is the only
panel that cannot saturate -- centred on 0.500 by construction at every
generation, however strong the lineage gets -- and it is the panel where the
students demonstrably *are* better. Everything else becomes a no-regression
guard whose job is to catch a student that beat the teacher by exploiting it
while getting worse in general.

- `--promote_margin` (absolute winblock improvement) is gone, replaced by
  `--winblock_tolerance` (default 0.03), a regression floor like the others.
- **Guard floors widen with expected wander**, to
  `max(tolerance, --noise_sigmas * _panel_sigma)`, default 2.5. This is *not* a
  sampling-noise correction -- there is no sampling noise. What moves between
  checks is the student's own weights as training continues; across the 127
  attempts its winblock score varied with sd 0.0154. The binomial expression is
  used as a scale-free proxy for that wander: it predicts 0.0150 at the same
  point, close enough to size a floor with, and it shrinks correctly as the
  panel grows.
- **Bars stay high-water marks.** All four use `max`. `best_heur` used to be
  assigned directly, which was safe *only* because the old rule required
  winblock to improve, so the new value was always the larger. With winblock
  demoted to a guard, a direct assignment would re-anchor the floor to each new
  teacher and slide it down by up to a tolerance per generation -- and since the
  students sit 0.0265 *below* the teacher on this panel, that slide is the
  expected case, not the unlucky one. head_to_head cannot prevent it: the whole
  reason it is the improvement criterion is that it is non-transitive with these
  heuristic panels.
- **No automatic rebase on resume.** The panels are deterministic, so
  re-measuring the same teacher reproduces the same numbers, and a rebase
  against a merely-current teacher can only lower a high-water mark.
  `--rebase_baselines` remains for the case where an anchor itself changed.
- **Failures are recorded.** `promote_failed` now goes into the metrics row.
  Attributing this deadlock required replaying the decision logic over 90k rows
  because only the scores were ever written down. The no-promotion log line also
  prints the floor actually applied instead of the raw bar.

## Verification: replay the 127 real attempts through the new gate

Floors implied by the real stored bars: winblock 0.8890, random 0.9650,
gregory-d3 0.7571.

| scenario | promote rate | h2h blocks | winblock blocks | gregory blocks |
|---|---|---|---|---|
| old gate | **0 / 127 (0%)** | 54% | 100% | 27% |
| new gate | **50 / 127 (39%)** | 54% | 20% | 3% |
| new gate, NULL student | 6 / 127 (5%) | 93% | 20% | 3% |

The NULL row replaces each real head-to-head score with a draw from a student
that is exactly the teacher's equal, holding the other panels fixed. It is the
check that the gate was loosened and not simply removed: real students promote
at 39%, a student with no edge at 5%, an **8.3x** separation.

Note winblock still blocks 20% of attempts. That is not a malfunction -- the
students genuinely dip below the floor sometimes, because they genuinely are
below the teacher on that panel. The guard is doing real work; it just is not
being asked to do the impossible.

## Also fixed: the deeper ruler had never run

`--gregory_hard_depth 4` landed in adfcc45 on 2026-07-21 and defaults to on, but
**0 of 89,984 metric rows contain it** -- the run had been stopped since before
that commit. Its first ever measurement, taken at this restart: the gen-22
teacher scores **0.638** against gregory d4. It is now armed as a guard. This
matters because winblock (0.927) and gregory-d3 (0.813) are both saturating
while d4 has real headroom.

## Limitations and residual risk

- **The students being weaker than the teacher on winblock is unexplained.**
  Distilling MCTS-200 targets from the teacher ought to produce a student that
  approaches teacher-plus-search, which should be stronger everywhere. It is
  stronger head to head and weaker on the heuristic panel. That is worth
  understanding on its own terms; this fix routes around it rather than
  explaining it.
- A 5% null-promotion rate is not zero. `--promote_thresh` 0.55 is a real
  requirement but a stalled lineage can still promote occasionally. Promotions
  driven by nothing are partly self-limiting (the high-water marks do not move,
  so the guards stay where they are) but a slow drift is not impossible.
  **The judge remains the fixed external panel, which the gate cannot see** --
  if gregory d3/d4 stop climbing across several generations, the promotions are
  not real regardless of what the gate reports.
- One state-file correction was applied by hand: the startup rebase (since
  removed) had lowered `best_random` from its 0.995 high-water mark to the
  current teacher's 0.9917. Restored to 0.995. One game in 300, on a guard that
  has never fired in 90k rows.

## Reproduce

    REM the diagnosis reads the live metrics log; no training required
    .venv\Scripts\python -m scripts.test_expert_iter

Bars, floors and blocking gate are now printed every promotion check and stored
in `promote_failed` in `loss_logs\metrics_log.jsonl`.
