# RESULT: the ruler ladder stops at d4 (2026-07-26)

## Question

The 564k-game plateau happened because a saturated panel was asked to detect
progress (`RESULT_GATE_PLATEAU.md`). winblock is now exhausted -- the gen-22
teacher scores 0.927 there -- and gregory-d3 is heading the same way at 0.813.
Before restarting the loop, how much runway is left in the honest ruler, and
can it be deepened?

## Cost, measured

Per-move cost on 40 sampled mid-game positions. `agents/gregory.py` estimates
d5 at ~1s/move; it is actually 20ms, a 50x error in the docstring.

| depth | ms/move | measured 300-game panel |
|---|---|---|
| d3 | 1.38 | 28s |
| d4 | 4.91 | 90s |
| d5 | 20.2 | 256s |
| d6 | 84.0 | not run (~10 min/panel projected) |

Per-move cost predicts panel cost to about 2.5x optimistic, so treat the
per-move number as a lower bound.

## Strength, measured -- the ladder flattens

300 games/cell, raw argmax, fixed openings, colours swapped, gate seeds.

| net | params | d3 | d4 | d5 |
|---|---|---|---|---|
| gen-22 teacher (oracle) | 6,766,386 | 0.813 | 0.638 | **0.637** |
| `pocket:squeeze-gen22` | 172,389 | 0.728 | 0.653 | **0.638** |

**Depth past 4 buys nothing.** d3 -> d4 is a large step (0.175 for the teacher),
d4 -> d5 is 0.001 while costing 2.8x more. Extra plies stop changing gregory's
decisions: UTTT's send-rule keeps the branching factor tiny for most of the
game, so d4 already reaches the useful horizon of a static heuristic, and
another ply gives the eval nothing new to exploit.

**So the ladder cannot be deepened by turning the depth dial.** When d4
eventually saturates the next ruler has to come from somewhere else -- a
different engine, or search over a frozen net from outside the lineage -- not
from gregory-d5/d6.

## Two consequences

**1. The current gate config is already correct, and this is not urgent.**
`--gregory_depth 3 --gregory_hard_depth 4` are the right two rungs. d4 sits at
0.638 for the teacher, i.e. **0.362 of headroom** -- a lot of runway before the
measurement problem returns. No flag change is needed. d5 should not be added:
it is 2.8x the cost for the same information.

Note the rungs are only regression floors under the fixed gate -- head-to-head
is what detects progress -- so gate saturation is not the risk. The risk is to
MONITORING: judging whether the loop is really improving needs an unsaturated
ruler, and that job is `scripts/ruler_ladder.py`, run per generation rather than
per promotion check.

**2. The pocket's capacity ceiling is not currently binding.** At d4 and d5 the
172,389-param pocket net is indistinguishable from the 6,766,386-param teacher
(0.653 vs 0.638, 0.638 vs 0.637 -- both inside the ~0.05 resolution floor from
`RESULT_CHECKPOINT_WOBBLE.md`). It trails only on d3 (0.728 vs 0.813), the rung
that is saturating.

That is the encouraging answer to the "small AND always improving" question: a
39x smaller net is already at parity against the strongest rulers available. The
open question is whether that holds as the teacher improves -- log both numbers
at every re-distill and watch the gap.

## Reproduce

    .venv\Scripts\python -m scripts.ruler_ladder --depths 3,4,5 --games 300
