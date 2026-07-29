# RESULT -- target extraction: visit-versus-value characterization (ARCHIVED)

Status: **closed without a transformed arm.** The characterization below was the
deliverable; the program was redirected to playing strength under a fixed move
deadline before any transformation was preregistered or trained. Nothing here
was used to select a target form.

Inputs are the frozen 50- and 800-simulation pilot corpora behind the published
`0.4108` (`RESULT_DISTILL_PILOT.md`). No new games were generated.

---

## 1. What was extracted, and what makes it trustworthy

`tools/extract_child_q.py` replays each frozen position and records every
child's `N`, `W`, `Q`, prior and solved status, which the corpora do not store
(they keep only the visit policy and the root value).

Replaying only means something if it is the *same* search that produced the
frozen targets. The extractor recomputes the visit policy and compares it to the
corpus target, and a single mismatched row is fatal by default.

| arm | rows | drift | wall clock |
|---|---|---|---|
| 50 sims | 50,000 | **0 / 50,000** | 0.7 h |
| 800 sims | 50,000 | **0 / 50,000** | 6.8 h |

`tools/checkpoint_child_q.py` then gates each arm on four conditions before it
may be used: zero drift over the full sample (rechecked independently of the
extractor's own tally), unvisited-child prevalence reported rather than assumed
away, all nine per-child preservation checks, and a hash.

| arm | legal children unvisited | positions fully visited | sha256 (normalized) |
|---|---|---|---|
| 50 sims | 148,823 / 485,088 (**0.3068**) | 0.6743 | `e28b7f2f421d07be` |
| 800 sims | 278 / 485,088 (**0.0006**) | 0.9987 | `2baf3d790b248148` |

### Missing is preserved as missing

`MCTS.Node.Q()` is `W/N` with a `0.0` fallback, so the raw replay stores `0.0`
for a child that was never visited -- an imputed value wearing the costume of a
measurement. It is stripped: `child_q` is NaN wherever `N == 0`.

This was recoverable but not obvious, and the corpus says why. All 148,823
unvisited children report exactly `0.0`, and **716 genuinely visited children
also report exactly `0.0`**. Keying missingness off the value would have
mislabelled those 716. Missingness is keyed off `N`, always.

Censoring at 50 sims is severe and strongly non-uniform:

| stratum | share of legal children unvisited | positions fully visited |
|---|---|---|
| 1 legal move | 0.0000 | 1.0000 |
| 2-4 | 0.0113 | 0.9611 |
| 5-8 | 0.0596 | 0.7293 |
| **9+** | **0.5312** | 0.4089 |
| early | 0.3449 | 0.6349 |
| mid | 0.2722 | 0.6937 |
| late | 0.1555 | 0.7801 |
| teachers agree | 0.3750 | 0.6733 |
| **teachers disagree** | **0.1377** | 0.6761 |

At 9+ legal moves more than half the children are never visited. Usefully, the
positions where the two budgets disagree are the *least* censored, so the subset
the study cares about is the better-measured one.

Because censoring at 800 sims is essentially nil, **the 800 tree is the only
defensible cross-arm comparison surface**. Averaging a 50-sim Q against an
800-sim Q would average a censored quantity against a complete one and call the
result a value gap.

---

## 2. A sign error, and how the corpus caught it

The first run of the characterization compared visit counts against raw
`child_q` and reported Spearman **-0.74**, a **97%** argmax disagreement, and
94.7% of target mass placed off the best-valued move. Read literally: the search
almost never plays what its own value function prefers.

It was the sign. `child_q` is stored in the **child's** `to_play` frame, and the
child's mover is the opponent of the player choosing the move --
`MCTS._best_child` scores `-c.Q()` for exactly this reason. Comparing raw
`child_q` against the visit counts measures agreement with what would be good
*for the opponent*, so a correct search must come out near-perfectly inverted.

The tell was in the data, not in the code review. **Mate-in-1 was the worst
stratum in the broken table** (deficit 0.8079 at 50 sims, 0.9056 at 800). A
winning child is terminal at `-1` in its own frame, so visits piling onto
`Q = -1` while the maximum child Q sits near `+1` is the search *finding the
mate*, scored as its largest failure. A metric that ranks forced wins as the
worst behaviour on the board is measuring itself.

`characterize_visit_vs_q.py` now negates once at load into `child_v` (the
mover's frame), labels every column `V` rather than `Q`, and refuses to emit any
table unless the 800-sim visit argmax agrees with the mover-frame best value on
a majority of mate-in-1 positions. That check now reports **0.9038** and would
have caught this before a single number was read.

No re-extraction was needed. The stored artifact is faithful; the frame belongs
in analysis.

---

## 3. Part 1 -- visits versus value, within each arm

`V` is the child value in the mover's frame. "V deficit" is how much value the
most-visited move gives up against the best-valued visited child.

| arm | visit != V-best | Spearman(visits, V) | mass off V-best | V deficit |
|---|---|---|---|---|
| 50 sims | 0.4131 | 0.7423 | 0.6055 | 0.0352 |
| 800 sims | 0.4197 | 0.7000 | 0.5985 | 0.0307 |

The visit argmax disagrees with the value argmax in **about 42% of positions at
both budgets**, and 16x more search makes the agreement slightly *worse*
(Spearman 0.7423 -> 0.7000).

**This is not a defect.** Visits are deliberately the robust choice over Q: a
child with one lucky visit has the best Q and should not be played. The cost of
the disagreement is small and *shrinks* with search -- the most-visited move
gives up 0.031-0.035 of value -- so the disagreement is between near-equivalent
moves rather than between a good move and a bad one.

Selected strata (800 sims):

| stratum | n | visit != V-best | Spearman | V deficit |
|---|---|---|---|---|
| all | 50,000 | 0.4197 | 0.7000 | 0.0307 |
| early | 22,224 | 0.4656 | 0.7489 | 0.0296 |
| mid | 23,893 | 0.3939 | 0.6750 | 0.0341 |
| late | 3,883 | 0.3152 | 0.5660 | 0.0164 |
| 2-4 legal | 5,785 | 0.3058 | 0.5901 | 0.0191 |
| 9+ legal | 13,014 | 0.4548 | 0.7572 | 0.0355 |
| **mate-in-1** | 1,144 | **0.0962** | 0.7131 | **0.0012** |
| no tactic | 48,856 | 0.4272 | 0.6997 | 0.0314 |

Tactically decided positions are handled correctly (9.6% disagreement, deficit
0.0012). The 42% is concentrated in quiet positions, and it grows with branching
factor -- consistent with PUCT's `sqrt(N)` exploration term spreading visits
more widely as the budget grows.

---

## 4. Part 2 -- what changed from 50 to 800 sims

Evaluated in the 800 tree throughout.

| quantity | value |
|---|---|
| top move changed | 16,896 / 50,000 (**0.3379**) |
| 50's pick visited at 800 | 1.0000 (also 1.0000 on changed) |
| 800 visit argmax == 800 V argmax | 0.5803 (changed **0.3728**, unchanged 0.6863) |
| 800 top-two V gap, median | 0.0482 (changed **0.0339**, unchanged 0.0602) |
| V800(800 pick) - V800(50 pick) on changed | median **+0.0022**, mean +0.0271 |
| ... positive / negative | 0.5164 / 0.4804 |
| ... within the 0.013 reference | 0.1279 |
| JS(pi50, pi800), bits, median | 0.0554 (changed 0.0727, unchanged 0.0472) |

Two things stand out. The budgets diverge precisely where the top two moves are
close in value (gap 0.034 on changed against 0.060 on unchanged) -- expected,
and confirmatory. And 800's own visit argmax matches its own value argmax only
58% of the time, dropping to **37%** on exactly the positions where it overruled
the 50-sim target.

---

## 5. Part 3 -- the changed positions on the fixed 0.013 ruler

`0.013` is the swapped-move value gap already measured in
`RESULT_SEARCH_DISAGREEMENT.md`, fixed before this distribution was seen. The
bins are explanatory. They are not a claim that 0.013 is the right training
threshold, and no threshold was selected.

| bin | n | share | median advantage |
|---|---|---|---|
| negative -- 800's pick is worse | 8,116 | 0.4804 | -0.0574 |
| near-equivalent `[0, 0.013)` | 1,028 | **0.0608** | +0.0049 |
| meaningful `>= 0.013` | 7,752 | 0.4588 | +0.0836 |

Continuous distribution of the advantage: mean +0.0271, p10 -0.1043,
p50 **+0.0022**, p90 +0.1728.

**When 800 sims changes the top move, it is close to a coin flip whether the new
move is better by 800's own values.** The median advantage is +0.0022, a fifth
of the 0.013 reference.

And it is *not* near-equivalent reshuffling. Only 6% of changes land in the
near-equivalent bin; the distribution is bimodal, with 48% meaningfully worse and
46% meaningfully better, cancelling to almost nothing. Individually large
changes, collectively directionless.

---

## 6. What this says, and what it does not

This is a fifth pattern, not one of the four decision paths that were laid out
in advance:

- Visit/value argmax disagreement is high (42%) but **cheap** (0.03 of value)
  and does not shrink with search, so "visits are polluted by PUCT artifacts" is
  true in form and small in magnitude.
- The 50-to-800 changes are **large and directionless** rather than
  near-equivalent. The sign of the advantage is close to unpredictable at the
  moment the target is built.

The practical consequence is the important part: **no threshold on value gap can
separate the good changes from the bad ones**, because at the point the target is
constructed the sign of the change is nearly a coin flip. A gap-dependent
softening or a Q-derived target would be selecting on a quantity that does not
predict the thing it is meant to fix. That is consistent with the 800-sim
student's reversal and with the powered null in `RESULT_SOLVED_PILOT.md`.

It does **not** show that the visit policy is a bad distillation target in
general, that the value head is miscalibrated, or that any particular
transformation would fail -- none was trained. It also cannot speak to the
student reversal directly: that is a property of trained students, not of the
corpus.

**Stability was never measured across sim counts.** Decision path 4 asked for
Q stability at 200/400/800 on a shared subset if raw Q looked unstable. The
bimodal cancellation is exactly the signature that would have triggered it. That
measurement was not run, and it is the first thing to do if this line is ever
reopened.

---

## Artifacts

`results/` is gitignored (repo convention), so the numbers quoted here are the
record.

| artifact | sha256 |
|---|---|
| `results/child_q/sims50.norm.npz` | `e28b7f2f421d07be` |
| `results/child_q/sims800.norm.npz` | `2baf3d790b248148` |
| `results/child_q/characterization.json` | -- |

The normalized artifacts carry full per-child arrays (`legal_mask`,
`visited_mask`, `child_n`, `child_w`, `child_q`, `child_prior`, `child_solved`,
`root_value`, `pi_visits`), not summary statistics, so any transformation can be
constructed offline without repeating the 7.5 hour replay.

Cost: 7.5 h of replay, plus minutes of analysis.

## Reproduce

    python -m tools.extract_child_q --pilot models/distill_pilot --sims 50 800 \
        --out results/child_q
    python -m tools.checkpoint_child_q --arm results/child_q/sims50.npz --sims 50
    python -m tools.checkpoint_child_q --arm results/child_q/sims800.npz --sims 800
    python -m tools.characterize_visit_vs_q --child-q results/child_q \
        --pilot models/distill_pilot \
        --output results/child_q/characterization.json
