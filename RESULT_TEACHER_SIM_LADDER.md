# RESULT: does more search make the teacher stronger? (2026-07-26)

## Question

`RESULT_SEARCH_DISAGREEMENT.md` measured that every doubling of MCTS simulations
changes the gen-22 teacher's chosen move at a flat ~14-16% rate out to 800 sims.
That is CHURN. It says nothing about whether the new move is better -- a search
reshuffling among near-equivalent moves produces an identical signature.

The controlled-distillation study (one fixed 172k student, teachers differing
only in sim count) only has a meaningful independent variable if the teachers
differ in STRENGTH, not merely in output. This measures that directly.

## Method

The teacher plays ITSELF at two simulation counts. Fixed openings, colors
swapped, raw argmax over visit counts, no Dirichlet noise, per-game reseed.
No referee, no third-party opponent, no averaged-Q estimate -- if the deeper
search plays better, it wins games.

`tools/teacher_sim_ladder.py`. The match loop mirrors the promotion gate's
`scripts.expert_iter._play_fixed_match` line for line and is parity-checked
against it (`--parity-check`, identical to 12 decimals); it additionally returns
per-game outcomes so the interval can use observed variance instead of assuming
binomial. Draws contribute exactly 0.5 with zero variance, so at a 30% draw rate
the honest interval is meaningfully tighter than the binomial one.

Two independent 400-game blocks with DIFFERENT opening sets (seeds 7705, 7805),
pooled. 3,538s + 3,516s.

## Result: deeper search IS stronger, by a little

| block | score for 800 sims | games | W/D/L |
|---|---|---|---|
| 1 (seed 7705) | 0.5437 | 400 | -- |
| 2 (seed 7805) | 0.5325 | 400 | 153/120/127 |
| **pooled** | **0.5381** | **800** | |

| interval | 95% CI | z | p | verdict |
|---|---|---|---|---|
| observed variance (sd 0.4176) | [0.5092, 0.5671] | 2.58 | 0.0098 | **separates** |
| conservative binomial | [0.5035, 0.5728] | 2.16 | 0.0310 | **separates** |

Separates from chance under both variance assumptions. The two blocks agree
closely (0.5437 vs 0.5325, difference 0.0112 against a standard error of the
difference of 0.0295), so this is a replicated effect, not one lucky opening set.

**800 sims beats 200 sims 0.538. That is +0.019 per doubling.**

Neither 400-game block separated on its own (0.5437 [0.4948, 0.5919] and 0.5325
[0.4916, 0.5734]). The effect is real and it is small enough that 400 games
cannot see it. Record this as the resolution scale for future sim-count work.

## Cross-check against an independent measurement

The gen-13 search curve (`uttt-gen13-search-curve`, vs gregory-d3) reads 0.757 at
200 sims and 0.778 at 400 -- **+0.021 per doubling**. This measurement, on a
different generation, against a different opponent (itself), by a different
method, gives **+0.019 per doubling**. Two independent estimates agree.

## The reconciled picture

Everything measured about deeper search now fits together:

| per doubling of simulations | magnitude |
|---|---|
| chosen move changes | ~15% |
| visit distribution (JS median) | converging, 0.0143 -> 0.0071 |
| root value (mean abs change) | converging, 0.0341 -> 0.0200 |
| **strength gained** | **+0.019** |
| value gap on the moves it swaps | 0.013 |

**Deeper search is genuinely better, and churn overstates the improvement by
roughly sevenfold.** Fifteen percent of moves change to buy two points of win
rate. The search is mostly reshuffling among near-equivalent moves while
occasionally finding a real improvement -- both halves of that sentence are now
measured rather than assumed.

This also explains, after the fact, why `tools/adjudicate_move_disagreement.py`
could not referee the individual disagreements: it was trying to resolve a 0.013
value difference using an independent referee that abstains 66% of the time.
See `RESULT_SEARCH_DISAGREEMENT.md` for that dead end and why refereeing single
moves is structurally unavailable here (it needs an oracle stronger than the
thing under test).

## What this means for the distillation study

The independent variable is REAL but SMALL, and it interacts badly with the
~0.05 panel resolution floor (`uttt-panel-resolution-floor`).

Extrapolating the gen-13 curve at +0.02/doubling gives approximate teacher
strengths of 0.630 (50 sims), 0.757 (200), ~0.798 (800) against gregory-d3:

| contrast | teacher gap | verdict |
|---|---|---|
| 800 vs 50 | ~17 points | comfortably above the floor, should resolve |
| 200 vs 50 | ~13 points | above the floor, should resolve |
| **800 vs 200** | **~4 points** | **below the floor before a student even trains** |

The stated key result `student_800 > student_200 > student_50` therefore splits
into an easy half and a near-impossible half. **Plan on the 800-vs-50 contrast
carrying the study and treat 800-vs-200 as unlikely to resolve** -- rather than
spending three full 200k-example corpora to discover that. Students also do not
inherit teacher strength one-for-one, so 4 points of teacher gap is an upper
bound on the student gap, not an estimate of it.

Nothing here argues against running the study. It argues for sizing the arms to
the contrast that can actually be measured.

## Artifacts

`results/teacher_sim_ladder.json` (block 1), `results/teacher_sim_ladder_block2.json`
(block 2, with per-game outcomes).

## Reproduce

    .venv\Scripts\python -m tools.teacher_sim_ladder ^
      --pairs 800:200 --games 400 --output results/teacher_sim_ladder.json
    .venv\Scripts\python -m tools.teacher_sim_ladder ^
      --pairs 800:200 --games 400 --seed-override 7805 ^
      --output results/teacher_sim_ladder_block2.json

A different `--seed-override` draws a different opening set, so blocks pool.
Reusing a seed replays identical games and adds nothing.
