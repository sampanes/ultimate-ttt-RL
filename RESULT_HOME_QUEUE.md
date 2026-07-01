# RESULT -- home-box queue from PENDING.md (2026-06-30)

*Run on the RTX 3080 home box, 2026-06-30, commit `ff47bd7`. Clears the
torch/GPU items from `PENDING.md`'s "Home-box runs" section. #5 (AlphaZero
validation) runs separately -- see its own section at the bottom.*

## Summary

| # | Run | Result | Verdict |
|---|---|---|---|
| 1 | MCTS sign tests | 7/7 PASS | value-sign convention locked |
| 2 | Recompute parity + A/B | parity PASS; A/B not better | `--recompute` safe but keep opt-in |
| 3 | value_coef sweep | EVs within 0.004 (wash) | keep `--value_coef 0.5` |
| 4 | best.pt honest rating | 0/40 vs MCTS at all depths | closed-loop ELO 4437 is dead |
| 6 | GOLD suite + grade | 16.37% neutral blunder rate | first opponent-neutral number |
| 5 | AlphaZero validation | 20 iters, loss 6.54->1.66, tanh stable | loop HEALTHY -- GO for long run (2 bugs found) |

## 1. MCTS sign tests -- PASS

`python -m agents.test_mcts` -> 7/7 PASS (argmax, backup sign alternation,
clone non-mutation, immediate win, unvisited Q=0, U formula, terminal-value
perspective). The value-sign convention is locked before any search-training.

## 2. Recompute parity + A/B -- safe, but keep opt-in

Parity gate PASS twice (60 and 80 self-play games; worst abs delta 6e-08 vs
1e-03 tol). The collect-then-recompute learn path reproduces the trusted
in-graph loss terms exactly, so `--recompute` is numerically safe.

But the short single-seed A/B does NOT clear the "not worse" bar:

| config | secs | peak ELO | final WR | mean EV |
|---|---:|---:|---:|---:|
| baseline-ingraph | 198.6 | 808.7 | 0.400 | +0.0015 |
| recompute-bigbatch | 227.3 | 817.2 | 0.370 | -0.0539 |

recompute-bigbatch was slightly worse on EV and win-rate AND slower. Note the
big-batch row ran at `--parallel 256`, which is the config memory flags as
gradient-starving (256 stalls, 64 reaches stage 6) -- a likely confound.
**Recommendation: do NOT flip `--recompute` default ON on this evidence.** It is
cleared as safe; earning default-ON needs a longer/multi-seed A/B, ideally with
the big batch fed through `--minibatch_size` so updates are not starved.

## 3. value_coef sweep -- wash, keep 0.5

3 chunks x 1000 games, seed 0, `--parallel 64`, only `--value_coef` varies.

| value_coef | mean EV | final EV | value-MSE | final WR |
|---|---:|---:|---:|---:|
| 0.25 | 0.0041 | 0.0347 | 1.2266 | 0.381 |
| 0.5  | 0.0008 | -0.0048 | 1.2397 | 0.368 |
| 1.0  | 0.0017 | 0.0184 | 1.2453 | 0.392 |

All EVs are within ~0.004 of each other -- inside the script's own ~0.02 "wash"
threshold, and all near zero (the critic barely learns in a 3-chunk stage-0
window). **No reason to change the default; keep `--value_coef 0.5`.** A real
discriminating sweep needs runs long enough for EV to lift off zero.

## 4. best.pt honest absolute rating -- ELO 4437 is dead

`best.pt` (medium) vs deep-MCTS wrapped around its OWN net, 40 games/row:

| oracle sims | W/D/L | win% | s/game |
|---|---|---:|---:|
| 100 | 0/0/40 | 0.0% | 2.78 |
| 400 | 0/0/40 | 0.0% | 11.21 |
| 800 | 0/0/40 | 0.0% | 18.86 |
| 1600 | 0/0/40 | 0.0% | 33.19 |

**The raw net loses 100% to shallow search over the same weights -- crossover is
below 100 sims.** The closed-loop Arena ELO of 4437 is confirmed meaningless
(it measures self-play population rank, not strength). Search is the dominant
untapped lever; this is the empirical backing for the M4 search-training track.

## 6. GOLD endgame suite -- first opponent-neutral blunder rate

Built 375 pre-solved positions (`build_endgame_suite`, 1000 games, max_empty 15,
seed 0): 141 won / 195 drawn / 39 lost, 0 skipped (all solved within budget).
Graded `best.pt` on the 336 gradable (won+drawn) positions:

- **blunder rate: 16.37%** (55/336) -- 35 in won positions (wins thrown away),
  20 in drawn positions (draws pushed toward losses).

This is the opponent-neutral metric `RESULT_GRADING.md` flagged as missing: the
earlier 28% (vs `center`) was skewed by organically-reached positions against a
weak opponent. 16.37% on a curated, side-neutral solved set is the trustworthy
late-game blunder figure for `best.pt`.

Artifact: `suite.json` (375 entries) was generated at repo root but is NOT
committed -- it is regenerable from the command above (deterministic, seed 0).
Commit it as a standing regression fixture if you want a fixed GOLD set.

## 5. AlphaZero validation -- HEALTHY (loop runs; 2 authoring-box bugs found)

`train_alphazero --checkpoint best.pt --network medium --value_tanh --n_sims 200
--games_per_iter 50 --iters 20`. Completed all 20 iters, ~800s/iter (~4.4h).

The AZ loop is mechanically healthy:

| metric | iter 1 | iter 20 |
|---|---:|---:|
| total loss | 6.5398 | 1.6589 |
| policy loss | 6.0881 | 1.5467 |
| value loss | 0.4517 | 0.1122 |
| wr_vs_rand (eval'd every 5) | 42.5% | 55.0% (iter 16) |

- **Loss falls monotonically**, value loss 0.45 -> 0.11 with no NaN/explosion:
  the bounded `--value_tanh` head trains stably. This is the go/no-go the run
  was for -- **GO** for a long AZ run (mechanics are sound).
- **Caveat on the numbers, not the loop:** seeding from `best.pt` (trained
  WITHOUT tanh) while enabling `--value_tanh` is the exact combo the script's
  own docstring warns "produces garbage" -- the value scale starts wrong, which
  is why wr_vs_rand opens at 42% (below random) and only recovers to 55% by
  iter 16. The loop heals it, but a real run should start from a fresh model or
  a tanh-trained checkpoint so the value head is not fighting a scale mismatch.

### Two bugs found (authoring-box code -- fixed via shim to let this run tonight)

1. **`train_alphazero.py` calls `agent.save_model(path)` but `NeuralNetAgentPG`
   has no `save_model`** -- the method is `save(path, verbose=True)`. Crashes at
   the first checkpoint (line 332; also 343, 367). This run used a scratchpad
   monkeypatch aliasing `save_model -> save` (correct: `save` has the matching
   signature and writes the standard `{state_dict, elo}`). Fix: rename the three
   call sites to `agent.save(...)`.
2. **Non-ASCII in a print** -- `train_alphazero.py:354` uses a literal em-dash
   character (U+2014) as the wr_str fallback, which renders as a replacement
   glyph on the Windows console. Fix: use an ASCII fallback like `"n/a"` / `"--"`.
