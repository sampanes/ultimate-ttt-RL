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

## Finding 2: the gate had become unreachable

127 promotion attempts were logged at gen 22. Replaying `_promotion_decision`
against the live bars in `state.json`: **zero should have passed.**

| criterion | bar | failed | sole blocker |
|---|---|---|---|
| winblock | 0.9467 | **100%** | **52** |
| head_to_head | 0.5500 | 54.3% | 0 |
| gregory d3 | 0.7833 | 26.8% | 0 |
| random | 0.9650 | 0% | 0 |

The winblock bar was `best_heur + promote_margin` = 0.9267 + 0.02. Measured
across those same 127 fresh 300-game panels, the lineage's actual winblock
strength was **0.9002**, with an observed spread of 0.0154 against a binomial
prediction of 0.0173 -- the two agree, so the sequence is pure measurement noise
around a flat mean with no trend at all.

That puts the bar **2.69 sigma above the lineage's true strength**: p = 0.36%
per attempt, about 192 attempts for even odds, and that is winblock alone before
being ANDed with a head-to-head gate that fails 54% of the time.

Meanwhile head_to_head averaged **0.545** -- 1.6 sigma above parity. The student
was genuinely, consistently beating the teacher the whole time. It was being
asked to also win a lottery.

## Root cause: two compounding design errors

**1. A saturated metric was carrying the improvement requirement.** winblock was
doing two jobs -- prove the student improved, and guard against regression. It
cannot do the first. It is bounded, it had saturated near 0.90, and much of the
residual against a fixed 1-ply heuristic is opening-determined rather than skill.
Demanding +0.02 there is demanding progress on a ruler that has stopped
measuring; the requirement was 1.15x the measurement noise.

**2. Winner's curse in the bar.** `best_heur = promote_heur` assigned the bar
from *the single draw that promoted*. Under the old rule winblock was itself the
promotion criterion, so that draw was selected for being high -- 0.9267 against a
true 0.9002, a full 1.5 sigma of pure selection bias -- and then +0.02 more was
demanded on top of it.

The same creep had reached the other guards. `best_gregory` was a running `max`
over selected panels, at 0.8133 against a true 0.798, which is why a "tolerance"
gate was firing on 27% of honest measurements.

## The fix

**One criterion decides improvement, and it is head_to_head.** It is the only
panel that cannot saturate -- centred on 0.500 by construction at every
generation, however strong the lineage gets. Everything else becomes a
no-regression guard whose job is to catch a student that beat the teacher by
exploiting it specifically while getting worse in general.

- `--promote_margin` (absolute winblock improvement) is gone, replaced by
  `--winblock_tolerance` (default 0.03), a regression floor like the others.
- **Noise-aware tolerances.** Every guard floor widens to
  `max(tolerance, --noise_sigmas * binomial sigma)`, default 2.5 sigma. A flat
  0.03 against a bar near 0.90 fires on ~40% of honest panels; that is a coin
  toss wearing a lab coat, not a safety check.
- **Bars track the teacher in hand**, not the luckiest draw ever seen. All four
  are assigned from the promoting measurement; none is a running `max`. The
  ratchet is what let them climb above true strength and never come back.
- **Legacy bars are force-rebased.** `state.json` below `schema_version` 5 is
  winner's-cursed, so those bars are discarded on load and re-measured against
  the current teacher. `--rebase_baselines` forces it manually. This is
  load-bearing, not cosmetic (see the replay below). Going forward the curse is
  structurally gone: promotion is decided on the head_to_head panel, a different
  opponent with a different seed, so the winblock draw riding along with it is
  unbiased.
- **Failures are recorded.** `promote_failed` now goes into the metrics row.
  Attributing this deadlock required replaying the decision logic over 90k rows
  because only the scores were ever written down. The no-promotion log line also
  prints the floor actually applied instead of the raw bar.

## Verification: replay the 127 real attempts through the new gate

| scenario | promote rate | blocked by h2h | by winblock | by gregory |
|---|---|---|---|---|
| old gate | **0 / 127 (0%)** | 54% | 100% | 27% |
| new gate, legacy bars | 50 / 127 (39%) | 54% | 20% | 3% |
| new gate, rebased bars | **58 / 127 (46%)** | 54% | 2% | 2% |
| new gate, NULL student | 9 / 127 (7%) | 93% | 2% | 2% |

The NULL row replaces each real head-to-head score with a draw from a student
that is exactly the teacher's equal, holding the other panels fixed. It is the
check that the gate was loosened and not simply removed: real students promote
at 46%, a student with no edge at 7%, a **6.4x** separation. head_to_head does
essentially all the gatekeeping (54% block rate) and the guards are quiet at 2%,
which is what a catastrophe guard should look like.

Rebasing is worth 7 points and removes winblock's 20% false-fire rate.

## Also fixed: the deeper ruler had never run

`--gregory_hard_depth 4` landed in adfcc45 on 2026-07-21 and defaults to on, but
**0 of 89,984 metric rows contain it** -- the run has been stopped since before
that commit. It arms on the restart. This matters because winblock (0.90) and
gregory-d3 (0.798) are both saturating while d4 has real headroom: the offline
A/B students scored 0.598-0.685 against d4 versus 0.715-0.793 against d3.

## Limitations and residual risk

- A 7% null-promotion rate is not zero. `--promote_thresh` 0.55 on a 300-game
  panel is 1.73 sigma, so roughly a 4% one-tailed alpha by construction, and a
  fully stalled lineage would still promote on noise every few hours. Promotions
  driven by noise are self-limiting (the bars re-track the new teacher each
  time) but a slow random walk is not impossible. **The judge remains the fixed
  external panel, which the gate cannot see** -- if gregory d3/d4 stop climbing
  across several generations, the promotions are noise regardless of what the
  gate reports.
- The rebased bars are estimated from a single panel per anchor. They inherit
  that panel's noise; the 2.5-sigma widening is what absorbs it.
- The "true strength 0.9002" figure is the mean over 127 panels of a *student*
  that averages 0.545 against the teacher, so it slightly overstates the
  teacher. The direction is conservative for the winblock floor.

## Reproduce

    REM the diagnosis reads the live metrics log; no training required
    .venv\Scripts\python -m scripts.test_expert_iter

Bars, floors and blocking gate are now printed every promotion check and stored
in `promote_failed` in `loss_logs\metrics_log.jsonl`.
