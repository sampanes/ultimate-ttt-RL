# RESULT: does more search change the training target? (2026-07-26)

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

## Headline: churn per doubling is FLAT

The raw table invites a wrong conclusion, because the pairs do not span equal
ratios -- 50->100 and 100->200 are single doublings, 200->800 is two.

| pair | ratio | raw move-change (95% CI) | JS median (95% CI) | **per doubling** |
|---|---|---|---|---|
| 50->100 | 2x | 0.142 [0.135, 0.149] | 0.0143 [0.0137, 0.0147] | **0.142** |
| 100->200 | 2x | 0.138 [0.131, 0.145] | 0.0083 [0.0081, 0.0086] | **0.138** |
| 200->800 | 4x | 0.249 [0.240, 0.257] | 0.0238 [0.0233, 0.0243] | **0.133** |

Normalising by `p = 1 - (1 - r)^(1/k)` for k doublings: **0.142, 0.138, 0.133.**

**Every doubling of simulations changes the chosen move about 13-14% of the
time, essentially constantly from 50 sims to 800.** There is no collapse and no
diminishing return in target churn across this whole range. 200->800 looked like
the outlier only because it spans twice the ratio.

Mean absolute root-value change is likewise flat: 0.034 / 0.030 / 0.034.

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

| forced target mini-board | open | won | drawn | none |
|---|---|---|---|---|
| 50->100 | 0.167 | 0.086 | 0.068 | 0.000 |
| 100->200 | 0.158 | 0.096 | 0.045 | 0.000 |
| 200->800 | 0.277 | 0.194 | 0.114 | 0.000 |

Search changes its mind most in the opening, in wide positions, and when the
mover is confined to a live mini-board. It changes its mind least in the
endgame, in narrow positions, and never when there is only one legal move.

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

## The tension that governs interpretation

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
