# ERRATA: mini-board indexing bug in the disagreement summarizer (2026-07-27)

Two related indexing errors in `tools/summarize_search_disagreement.py` produced
wrong values for two strata. Every number those strata ever reported is invalid.
Nothing else is affected. This note is the permanent record; the numbers
themselves are disarmed in place rather than deleted.

## The bugs

Ultimate TTT stores the board as a flat row-major 9x9 array, so a move index
`m` decomposes as `r = m // 9`, `c = m % 9`. The mini-board containing it is
`(r//3)*3 + (c//3)`, and the mini a move FORCES the opponent into is the local
cell `(r%3)*3 + (c%3)`, which the engine exposes as `rule_utl_get_next_mini`.

| site | was | is | what the wrong expression actually returns |
|---|---|---|---|
| `tactical_flags` | `s.mini_winners[m // 9]` | `s.mini_winners[mini_of(m)]` | the board ROW |
| `forced_target_state` | `st.last_move % 9` | `rule_utl_get_next_mini(st.last_move)` | the COLUMN |

Only the first was reported. The second was found while fixing it, and is the
same class of error: a plausible-looking one-liner that coincides with the right
answer on part of the board. Row and mini agree on 27 of 81 squares -- the
leftmost mini of each band -- so both survive spot-checking and fail on the
other 54.

## Blast radius, measured

Over 2,904 random reachable positions, comparing the old and new expressions
directly:

| stratum | measured |
|---|---|
| `mini_win_available` | true rate **0.3705**, reported **0.1977**; mislabelled on **17.29%** (502/2904) |
| `forced_target` | mislabelled on **11.88%** (345/2904) |

The buggy `forced_target` distribution was `{open: 2533, won: 371}` against a
true `{open: 2593, won: 309, drawn: 2}`. It never emitted `drawn` at all --
`mini_winners[column]` cannot land on a drawn mini except by coincidence, so the
category was structurally unreachable rather than merely rare.

The `mini_win_available` rate is off by nearly a factor of two. These are not
numbers that can be caveated and cited; they are discarded.

## What is invalid

* every `mini_win_available` figure ever produced by this script
* every `forced_target` / `by_forced_target` figure ever produced by this script
* the `forced_target` and `mini_win_available` COLUMNS of
  `position_strata.csv.gz` (both copies)
* the `forced target mini-board` table in `RESULT_SEARCH_DISAGREEMENT.md`

## What remains valid

* `overall`, `by_phase`, `by_legal_bucket` -- no mini indexing anywhere in them
* `by_tactical.immediate_win_available` and `by_tactical.no_immediate_win` --
  derived from `s.winner == mover`, which never touches a mini index
* every global metric: churn per doubling, JS divergence, value deltas, the
  disagreement subset size, the independence check
* all of `RESULT_TEACHER_SIM_LADDER.md` and `RESULT_DISTILL_PILOT.md`

Per the owner's instruction, unrelated global metrics were NOT regenerated.

## Why it cannot reach search, targets, or the distillation findings

The bug lived exclusively in a post-hoc summarizer that reads policies which
were already computed and written to disk. Verified by audit:

* `tools/summarize_search_disagreement.py` is the only site of either
  expression. The two occurrences are lines 60 and 87 of the pre-fix file.
* `tools/analyze_search_disagreement.py` -- the pass that actually replayed
  MCTS and produced `policies.npz` -- computes **no strata at all**. Its only
  mini-board arithmetic recomputes `mini_winners` through the engine's own
  `rule_utl_check_mini_win` over `_MINI_INDICES`, and it cross-checks every
  reconstructed position against the stored legal-move plane (0 rejects in
  9,999).
* `agents/mcts.py`, `engine/rules.py`, `agents/agent_base.py`,
  `tools/make_distill_corpus.py` and `scripts/train_student_offline.py` were
  never touched by it. `agent_base.py`'s `// 9` uses are row extraction for
  plane scatter, which is what `// 9` correctly means.

So no training target, no search decision, no student weight and no head-to-head
result was ever computed through the bad index. The `0.4108` reversal and the
teacher ladder stand unchanged.

## Fix and guard

Fixed 2026-07-27. `mini_of()` now carries the derivation and a warning in its
docstring, and `forced_target_state` calls the engine helper instead of
recomputing.

`tools/test_summarize_search_disagreement.py` (5 tests) guards it. The tests are
differential, not merely confirmatory: they build an independent mini lookup by
membership scan of `_MINI_INDICES`, assert that exactly 54 of 81 squares
distinguish row from mini (and 54 distinguish column from local cell), and
assert that the OLD expressions disagree with the reference on at least 20 of
600 random positions. That last assertion is what stops the suite from quietly
degrading into a tautology if someone reintroduces the bug on both sides.

`tools/invalidate_bad_strata.py` disarms the historical artifacts. It stamps
JSON leaves in place with `_INVALID`, the reason, and `_invalid_original` so
nothing is erased, and drops a `.INVALID_COLUMNS.txt` sidecar next to each
gzipped CSV. It is idempotent, and `--check` exits non-zero if any affected
stratum is unstamped. Current state: **410 JSON leaves + 2 CSV sidecars**.

## Reproduce

    .venv\Scripts\python -m tools.test_summarize_search_disagreement
    .venv\Scripts\python -m tools.invalidate_bad_strata --check
