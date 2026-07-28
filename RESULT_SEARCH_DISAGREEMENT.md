# RESULT: does more search change the training target? (2026-07-26)

> **ERRATA 2026-07-27.** Two strata in this document -- the forced-target
> mini-board table, and any `mini_win_available` figure -- were computed with a
> mini-board indexing bug and are INVALID. Every global metric (churn, JS
> divergence, value delta, the disagreement subset, the phase and legal-move
> tables, the mate-in-1 validity check) is unaffected, as are the conclusions.
> Full account: `ERRATA_MINI_INDEX_BUG.md`.

## Question

The proposed controlled-distillation study -- one fixed 172k student, teachers
differing only in simulation count -- only has an independent variable if the
teacher's TARGETS actually differ across sim counts. This measures that
directly, on positions already in the gen-22 corpus, so it cost no new
self-play.

## Method

9,999 positions sampled from the gen-22 corpus, stratified equally across
early/mid/late (3,333 each, by occupied cells). MCTS replayed at 50/100/200/800
sims over the frozen gen-22 teacher, Dirichlet noise OFF so differences are
attributable to search depth rather than root exploration. Full visit
distributions saved, not just the selected move.

Positions are reconstructed from the stored (7,9,9) planes. Channel 4 collapses
a DRAWN mini-board onto the same -1.0 as an O-won one, so `mini_winners` is
recomputed from the board planes via the engine's own `rule_utl_check_mini_win`;
channel 3 (legal moves) then acts as a correctness check. **0 reconstruction
rejects in 9,999 positions.**

Cost: 6,643s total (50 sims 384s, 100 sims 672s, 200 sims 1,314s, 800 sims
4,273s). Batch size is bounded by search semantics, not VRAM -- `MCTS.search`
clamps `eff_wave = min(wave_size, n_sims // 16)`, giving wave 3 at 50 sims and
50 at 800. The 16-wave floor is load-bearing (mcts.py records 1 wave -> 0.00
strength), so a larger batch would have degraded the targets being measured.

## Headline: churn per doubling is FLAT, but the distribution CONVERGES

The 400-sim arm was added later so that every pair is a single doubling and no
normalisation is needed. Measured directly:

| pair | move-change (95% CI) | JS median (95% CI) | JS mean | mean abs dV |
|---|---|---|---|---|
| 50->100 | 0.142 [0.135, 0.149] | 0.0143 [0.0137, 0.0147] | 0.0302 | 0.0341 |
| 100->200 | 0.138 [0.131, 0.145] | 0.0083 [0.0081, 0.0086] | 0.0191 | 0.0304 |
| 200->400 | 0.141 [0.134, 0.148] | 0.0073 [0.0071, 0.0075] | 0.0150 | 0.0225 |
| 400->800 | 0.158 [0.152, 0.166] | 0.0071 [0.0069, 0.0073] | 0.0130 | 0.0200 |

**Two quantities move in opposite directions.** Argmax churn is flat -- and the
LAST doubling is the churniest of the four. But the visit distribution converges
monotonically (JS mean falls 2.3x) and so does the root value (1.7x).

That dissociation is the reshuffling signature. As search deepens it settles on
how much it likes each move and on how good the position is, while continuing to
swap which move is nominally on top at an undiminished rate -- what you see when
the top two moves are near-equivalent and the ordering between them keeps
re-flipping.

**Consistency check.** If the two doublings inside 200->800 flip independently,
predicted composite churn is `0.141 + 0.158 - 2(0.141)(0.158) = 0.254`. Measured
was 0.249. The flips are statistically independent; deeper search does not
revisit and confirm.

CAVEAT on the first row: at 50 sims across ~9 legal moves a large share of
visits are still the PUCT breadth requirement giving each child a look, so
low-sim distributions are systematically flatter and 50->100 JS is partly that
schedule burning off rather than genuine mind-changing. Visit-count quantisation
was checked and is ~an order of magnitude too small to matter (~0.0015 bits at
50 sims against a measured 0.0302).

SUPERSEDED: an earlier version of this table had only 50/100/200/800 and
normalised the two-doubling pair to 0.133 via `p = 1 - (1-r)^(1/k)`. That formula
ignores flip-backs and slightly UNDERSTATES the rate; the measured adjacent
doublings above are the ground truth.

## Where the disagreement lives

Move-change rate by stratum:

| pair | early | mid | late |
|---|---|---|---|
| 50->100 | 0.214 | 0.140 | 0.071 |
| 100->200 | 0.206 | 0.134 | 0.072 |
| 200->800 | 0.338 | 0.234 | 0.174 |

| pair | 1 legal | 2-4 | 5-8 | 9+ |
|---|---|---|---|---|
| 50->100 | 0.000 | 0.085 | 0.147 | 0.192 |
| 100->200 | 0.000 | 0.081 | 0.143 | 0.187 |
| 200->800 | 0.000 | 0.184 | 0.257 | 0.307 |

**INVALID -- do not cite this table.** See `ERRATA_MINI_INDEX_BUG.md`. The
forced mini was computed as `last_move % 9`, which is the COLUMN, not the local
cell; the labels are wrong on 11.88% of positions and the `drawn` column was
structurally unreachable. Retained only so the historical record is legible.

| forced target mini-board | open | won | drawn | none |
|---|---|---|---|---|
| 50->100 | 0.167 | 0.086 | 0.068 | 0.000 |
| 100->200 | 0.158 | 0.096 | 0.045 | 0.000 |
| 200->800 | 0.277 | 0.194 | 0.114 | 0.000 |

Search changes its mind most in the opening and in wide positions. It changes
its mind least in the endgame, in narrow positions, and never when there is only
one legal move. (The original text also claimed a forced-mini effect; that
clause rested on the invalid table above and is withdrawn.)

**Validity check.** On the 450 positions where a mate-in-1 exists, move-change
collapses to 0.011 / 0.016 / 0.082; on the rest it is 0.148 / 0.143 / 0.257.
Every simulation level finds the forced win. The measurement is tracking genuine
decision content, not noise.

## Decisions

Against the stated rule:

| flag | value |
|---|---|
| `distillation_50_100_supported` | **true** |
| `distillation_100_200_supported` | **true** |
| `drop_800_arm` | **false** -- targets differ materially; arm retained |
| `teacher_separation_measured` | **false** |

The disagreement subset (argmax differs OR JS >= 0.05) is **5,208 / 9,999 =
52.1%** of positions.

**Proceed with the distillation study, and do not drop the 800 arm.** The
independent variable is real and roughly constant per doubling.

## RESOLVED: the adjudication dead end, and the answer that replaced it

Two follow-ups ran the same day. Recorded here so the dead end is not rebuilt.

**1. Refereeing individual moves does not work.**
`tools/adjudicate_move_disagreement.py` took the 800 positions where 200 and 800
sims disagree and judged the two candidate moves three ways. It failed:

| signal | result |
|---|---|
| value delta (fresh equal-effort 1600-sim search per candidate) | deeper move better 0.366 [0.331, 0.400]; mean delta -0.0132 |
| 1600-sim root's own pick | 800-move 0.885, 200-move 0.034, neither 0.081 |
| gregory-d4 (independent) | abstains 0.662; restricted h2h 0.548, n=270, CI ~[0.489, 0.607] |

The independent referee is at chance. The two same-net signals CONTRADICT each
other -- the root search overwhelmingly picks the 800-move while independent
searches of the resulting positions call it slightly worse. The design is clean
on effort (each candidate gets its own fresh 1600-sim search, verified in code),
so the likely culprit is that `root.Q()` is a visit-weighted AVERAGE rather than
a minimax value and is depressed where search spends early visits on lines it
later refutes. Support is only partial: the anti-deep skew is worst early
(0.285) and gone late (0.415), which fits, but the legal-move buckets come out
non-monotonic (0.331 / 0.414 / 0.302), which does not.

**Do not try to fix this instrument.** Refereeing a single move requires an
oracle stronger than the thing under test, and none exists here. The magnitudes
also make it hopeless: the two candidate moves differ by ~0.013 on a [-1, 1]
value scale.

**2. Playing the match does work, and answers the question.**
`RESULT_TEACHER_SIM_LADDER.md`: the teacher plays itself, 800 sims vs 200 sims,
800 games over two independent opening sets. **0.5381, 95% CI [0.5092, 0.5671],
p = 0.0098** -- separates from chance under both observed-variance and
conservative-binomial assumptions. That is **+0.019 per doubling**, against the
gen-13 curve's independently measured +0.021.

So: **deeper search is genuinely stronger, and churn overstates the improvement
by roughly sevenfold** -- ~15% of moves change to buy ~2 points of win rate.
Both the reshuffling reading and the improvement reading were partly right.

Neither 400-game block separated alone; pooling 800 was required. That is the
resolution scale for sim-count work.

## The tension that governed interpretation (now resolved -- see above)

Targets differing is not targets improving, and the repo already contains a
reason to worry. The gen-13 search curve against gregory-d3 reads
`0.39 / 0.630 / 0.672 / 0.757 / 0.778` at 0/50/100/200/400 sims -- so above 200,
doubling the sims bought roughly **+0.02 of strength**.

Put beside this result, that implies search at high sim counts **changes ~13% of
its moves per doubling while gaining almost no strength**. The most likely
reading is that the extra simulations are reshuffling among near-equivalent
moves rather than finding better ones. If that holds at gen-22, the distillation
arms would imitate measurably different targets and still play the same.

This is exactly why the teacher-side ladder is a precondition, not a nicety.
**Before running the distillation study, measure the gen-22 teacher itself at
50/100/200/800 sims against the external ladder.** If teacher strength separates,
the study tests student capacity. If teacher strength is flat while targets
churn, the study has an independent variable that is nearly orthogonal to
quality, and the effort belongs in search QUALITY instead -- symmetry folding,
transposition merging, tree reuse, solved-node propagation, all still
unimplemented per `MCTS_STATUS.md`.

## Artifacts

`results/disagreement/` -- `summary.json`, `summary_analysis.json` (bootstrap
95% CIs over positions, all strata), `per_position.csv.gz`,
`disagreement_positions.csv.gz` (5,208 rows), `position_strata.csv.gz`,
`policies.npz` (full visit distributions), `diagnostic_top_js.json` (the 100
highest-JS positions with rendered board, legal moves, per-sim policy, selected
move and root value).

## Reproduce

    .venv\Scripts\python -m tools.analyze_search_disagreement ^
      --sample-size 10000 --sims 50 100 200 800 --output results/disagreement
    .venv\Scripts\python -m tools.summarize_search_disagreement ^
      --input results/disagreement
