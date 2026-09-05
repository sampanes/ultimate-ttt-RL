# ROADMAP

Written 2026-09-04. Supersedes `SHIP_PLAN.md`, `STRENGTH_ROADMAP.md`,
`STRENGTH_NEXT.md` and the open queues in `PENDING.md` as the statement of
direction. For what is currently true, read `CURRENT_STATE.md`.

```
CURRENT PRODUCT
pocket_filter
~1 second/move
frozen search engine

NEXT GOAL
make the learned model improve repeatedly while the engine stays fixed

NEXT EXPERIMENTAL PROGRAM
repair/restart continual training
-> anchored promotion gate
-> stronger teacher/self-play
-> train next student
-> evaluate complete 1-second agent
-> promote only genuine strength improvements
-> repeat for days/weeks
```

---

## Why this and not more engine work

The engine-optimization branch is closed. The remaining measurable
opportunities are individually smaller than the reproducibility floor of the
metric that would have to confirm them -- 26.9 ms of descent redundancy is
+3.3% against a throughput estimator that replicates to 1.7-6.1%. At that point
an A/B stops being evidence and becomes ceremony. The stopping rule is recorded
in `CURRENT_STATE.md` section 12.

Meanwhile the teacher lineage has been parked at generation 22 for **564,480
games** because of a promotion gate that could not be passed, and the teaching
signal says there is still plenty to distill (`mcts_edge` 0.811 at gen 22, down
only 0.066 across 22 generations). The gate was repaired on 2026-07-25 and
**has never been run**. Another 3% of search rate is worth much less than
finding out whether the machine can be left training for a week and come back
stronger.

---

## The engine does not move

`pocket_filter` is the chassis for the whole program. This is a scientific
requirement: if search keeps changing while networks train, a stronger
generation N+1 cannot be attributed. Learning may have improved, or the chassis
may have moved underneath it, and nothing separates those after the fact.

Enforced by the three drift layers in `tools/engine_registry.py`, with the
deployment baseline under the same strict source check as an anchor. Recovery
from drift is a tag checkout, not a repair.

---

## The cycle

**Two loops at different frequencies, measuring different things.** Keeping
them separate is the whole design, because the step between them -- distillation
-- carries a known, unexplained penalty and cannot be assumed transitive.

### Inner loop: the teacher (hours)

`scripts/expert_iter.py` via `start_goat.bat`. Self-play with MCTS-200 over the
current teacher, train the student, gate, promote, repeat.

The repaired gate, already in code and never run:

- **Head-to-head against the current teacher is the only improvement
  criterion.** It is the only panel that cannot saturate -- centred on 0.500 by
  construction at every generation, however strong the lineage gets.
- **Everything else is a no-regression guard** with a high-water floor:
  WinBlock, random, gregory d3, gregory d4. Their job is to catch a student
  that beat the teacher by exploiting it while getting worse in general.
- **Floors widen with expected wander**,
  `max(tolerance, noise_sigmas * _panel_sigma)` at 2.5 sigmas. This is not a
  sampling-noise correction. **These panels are deterministic** -- there is no
  sampling noise and nothing to get lucky on. It is a scale-free proxy for the
  student's own weights moving as training continues (sd 0.0154 measured across
  127 attempts).
- **Bars stay high-water marks.** With WinBlock demoted to a guard, a direct
  assignment would re-anchor the floor to each new teacher and slide it down
  every generation, because the students sit 0.0265 *below* the teacher there.
- **Failures are recorded** in the metrics row. Attributing the original
  deadlock required replaying 90k rows because only scores had been written
  down.

Replayed against the 127 real attempts: 39% promote, against 5% for a null
student with no edge. That 8.3x separation is a replay, not an outcome, and
turning it into an outcome is the first milestone.

### Outer loop: the product (days)

The inner gate measures the **raw argmax network**. The product is
**network + search under a 1,000 ms deadline**. Those are different agents, and
the distillation step between them is exactly where the known penalty lives, so
the product gets its own gate.

1. `expert_iter --generate_only` -> a corpus from the promoted teacher.
2. `scripts/train_student_offline.py` -> a 172k `squeeze` student.
3. `scripts/pocket_challenge.py` -> the raw-net screen: 300-game h2h against
   the incumbent pocket, plus the gene-pool-independent alpha-beta rulers.
4. `tools/arena_1s.py` -> **the deciding measurement**: the complete agent, the
   new pocket inside `pocket_filter`, at equal wall clock against the incumbent
   pocket and against `anchor_C`.

**Promote the shipped model only on step 4.** Steps 1-3 are screens.

---

## Milestones

| # | milestone | done when |
|---|---|---|
| 1 | the repaired gate promotes in production | teacher gen 23 exists, promoted by the new logic, with the gate line logged |
| 2 | the lineage moves | gen 24+ promoted, and `mcts_edge` still says there is signal left |
| 3 | a new pocket is distilled from a post-22 teacher | student trained, raw-net screen passed |
| 4 | the complete 1-second agent improves | new pocket beats the incumbent at equal clock, interval excluding 0.5 |
| 5 | it repeats unattended | two consecutive outer-loop promotions with no manual intervention |

Milestone 4 is the one that matters. Milestones 1-3 can all pass while the
product stands still.

---

## Decisions to make before the first long run

These are open on purpose. Each changes what gets measured, and picking one
silently is how a program ends up unable to explain its own results.

**1. What the outer loop is anchored to.** `anchor_C` (gen-22 at 2,000 ms) is
fixed, deterministic and ordered, which is what an anchor must be -- but every
rung shares the gen-22 gene pool, so it is not an independent opinion about
strength. As the lineage advances past gen 22 the anchor stays a fixed
reference and becomes a progressively older one. Either accept that and say so
on every result, or add a genuinely external opponent. `gregory(d4)` is
saturated at 1.0000 and cannot be deepened (d5 is 0.637 against d4's 0.638 at
2.8x the cost), so it is not the answer.

**2. Outer-loop cadence.** How many teacher generations per pocket refresh. Too
frequent and the outer gate spends all the wall clock; too rare and a
distillation regression hides for days.

**3. Sizing the deciding match.** The anchor ladder says one doubling of clock
is worth 0.59-0.69, so a generation's worth of improvement is a small effect.
Pre-register the game count from the expected effect size **before** the match,
not after seeing the interval.

**4. Whether the teacher lineage stays 6.77M.** The 172k pocket is at parity
with the 6.77M teacher on the d4 and d5 rulers, and architecture is not the
strength lever at equal wall clock. Training the teacher at 6.77M may be paying
for capacity that does not reach the product.

---

## Standing constraints

- **Judge only by fixed external panels and equal-clock matches.** Never by
  loss, never by generation count, never by closed-loop ELO -- `best.pt`
  reached ELO 4437 and lost 0 for 40 to shallow search over its own weights.
- **One change per run segment.** A segment with two new variables cannot be
  attributed.
- **Write the prediction down before the match**, including how it could be
  wrong. An interval computed after the fact is not a prediction.
- **Stop gracefully.** `stop_goat.bat` writes the `STOP` sentinel that
  `expert_iter` polls each block. Console Ctrl+C is swallowed by the
  `--eval_server` actors on Windows. Never `taskkill /F` -- it can kill the
  child mid state-save and corrupt the resume payload.
- **Never run two GPU measurements at once.** It corrupts timing studies
  silently: same JSON, no crash, 1.9x difference.
- **The shard store prunes per block.** It was append-only and reached 89 GB;
  only current-generation shards are ever read.

---

## What is not on this roadmap

Engine optimization, unless something clears the stopping rule in
`CURRENT_STATE.md` section 12. The named survivors -- descent `make_move`
redundancy, wave-loop Python, node creation, native `could_end` -- are recorded
there with their prices so that a future decision starts from measurements
instead of re-deriving them, not because any of them is queued.
