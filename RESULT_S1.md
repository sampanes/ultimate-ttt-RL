# RESULT_S1 -- S1 segment start (T0 data package for the authoring box)

*Training box, 2026-07-13. PENDING.md "S0+S1+S2 runbook" steps 1-3 executed,
plus the paused-window diagnostics the authoring box cannot run itself (no
torch). The run was paused ~45 min with owner approval for clean-GPU
measurements, then restarted with the S1 flags. The 1-generation S1 judgment
section will be APPENDED to this file when that data exists.*

---

## 1. Runbook step 1 -- regression suites: PASS

- `scripts.test_expert_iter`: **11/11** (includes the new slice-layout and
  value-blend tests).
- `agents.test_mcts`: **11/11** (includes `test_root_q_sign_on_won_root`,
  the S2 trust gate -- so `--value_blend` stays eligible for the segment
  after S1).

## 2. Runbook step 2 -- baseline_vs_gregory: ENABLE

    Raw teacher gen 6 [arena22] vs gregory | 300 fixed color-swapped openings per depth | device=cpu
      vs gregory(d2): 0.138   (30s)
      vs gregory(d3): 0.138   (55s)
    VERDICT: d2 slice has headroom -> enable it

The d2 and d3 scores are IDENTICAL: every loss the raw net takes against
minimax is shallow enough that depth 2 already delivers it. Two readings:
(a) the d2 slice is maximally informative (huge headroom vs the 0.55
threshold); (b) d3's third ply currently adds nothing against this net, so
the d2-train / d3-rule split loses no signal. Sanity: the d3 number sits in
the promotion panel's 0.10-0.16 gen-7 band (same instrument, seed 8801).

## 3. Runbook step 3 -- segment live

- `start_goat.bat` now carries `--greg_mix 0.10 --opp_mix 0.30 --rnd_mix
  0.10` (self-play stays 0.50); committed `2b55cba`, this report and the
  restart follow it.
- Restart happened mid-gen-7 (58 promotion checks in, head at 47-54): the
  owner approved pausing for diagnostics rather than waiting for the gen-7
  promotion, so the S1 mix change lands mid-gen. Judge accordingly: the
  clean before/after boundary for the gregory slope is the RESTART, not a
  promotion.
- Prior session: 56h 46m continuous, stopped cleanly (state + resume.pt
  saved), monitor survived the bounce.
- **Restart verified live** (first new-schema metrics block, games_total
  69,152): `greg_games: 1` (16-game block at 0.10 mix -> 1-3 expected),
  `gen_secs: 75.9`, `train_secs: 2.2` (S0 fields streaming),
  `opponent_games: 8, rnd_games: 1` (rebalanced mix sampling correctly),
  and the block's gate read `mcts_edge 0.7` (healthy, tripwire is < 0.5).
  Total pause for diagnostics: ~40 min.

## 4. Gen-7 state at T0 (the "before" series)

58 promotion checks with teacher_gen=6 (first row = the check that promoted
gen-6; the rest are the gen-7 grind). games_total 69,072 at T0 (~18.7k this
gen -- already past gen-6's whole 14.1k rebuild).

    head    : 56 41 37 41 40 40 47 45 42 40 46 46 46 42 49 45 46 46 45 46 50 45 45 45 48 46 48 44 51 49 46 50 54 52 52 47 51 48 48 47 48 49 52 52 50 51 49 52 49 54 53 54 54 52 49 47 53 50
    winblock: 38 24 28 24 32 28 32 35 31 29 35 29 30 38 32 38 27 28 33 32 32 37 34 36 30 36 38 35 35 32 36 34 36 39 36 40 36 38 36 38 38 33 36 38 36 35 31 36 37 40 36 40 37 42 42 34 40 38
    random  : 87 81 80 82 82 84 81 84 84 84 84 84 79 87 83 81 84 84 84 87 83 82 88 89 84 88 86 86 85 86 86 89 88 86 85 84 87 83 84 87 88 83 88 86 86 88 87 89 87 87 86 85 90 90 85 88 87 86
    gregory : 14  7 12 12 14 11 10 11 10 14 10 11 11 16 11 12 14 12 14 12 12 16 12 14 10 13 14 13 15 13 16 16 16 14 13 16 17 15 14 14 15 13 16 13 16 16 15 17 17 17 19 16 14 16 15 22 15 19

Half-over-half means (first 29 vs last 29 checks): head 45.1 -> 50.4,
winblock 32.3 -> 36.9, gregory 12.3 -> 15.7, random ~84 -> ~87.

Two implications:
- **LR is AT the 1e-4 floor** (last check's logged lr = 0.0001) but every
  panel is still climbing -- so no LR intervention (same verdict logic as
  the gen-6 plateau). The per-gen LR reset means gen-8 starts back at 1e-3.
- **Gregory was already drifting up (+3.4 pts half-over-half) BEFORE S1.**
  S1's success bar must beat this pre-existing drift, not zero: call S1
  real if the gregory slope visibly steepens past ~+4 pts/gen post-restart.

## 5. GPU / timing diagnostics (S3 + S4 decision data)

All measured on this box (RTX 3080 10 GiB, torch 2.7.1+cu128, arena22 net,
6.77M params), teacher gen-6 weights, GPU otherwise idle unless noted.

### 5a. Live-run GPU picture (before the pause, training active)

22-30% GPU utilization, 4.1 / 10.2 GiB VRAM. The run leaves ~3/4 of the
GPU idle.

### 5b. Forward latency vs batch size (median of 30 after warmup)

    batch     1:    0.86 ms  ->      1,162 pos/s
    batch    12:    0.91 ms  ->     13,138 pos/s   <- today's generation batch
    batch    64:    1.34 ms  ->     47,663 pos/s
    batch   256:    3.26 ms  ->     78,641 pos/s   <- sweet spot, 6.0x today
    batch  1024:   29.18 ms  ->     35,096 pos/s   (degrades)
    batch  4096:  768.72 ms  ->      5,328 pos/s   (collapses; peak 9.46 GiB)

Batch 12 costs the SAME latency as batch 1 -- the current wave (200 sims //
16 = 12) rides for free, and the GPU-side ceiling for cross-game coalescing
is ~256-position forwards (about 20 concurrent games' waves).

### 5c. Where block time actually goes (last 500 blocks, console log)

    gen_secs   : mean 85.3   median 85.0   p90  99.0
    total_secs : mean 100.4  median 94.0   p90 127.0
    non-gen    : mean 15.1   median  2.0   (train + gate/promo evals)
    generation share of block time: 85%

Train (100 steps x batch 256) is a trivial ~2s/block; the non-gen mean is
inflated by periodic gate (~35s) and promotion (~90-120s) evals. Confirmed
exactly by the first S0-instrumented block post-restart: gen_secs 75.9,
train_secs 2.2.

### 5d. collect_game wall-clock (self-play, expert_iter params, idle GPU)

    n_sims 200: 7.77 s/game  (7.67, 7.24, 8.39)  ~51 examples/game
    n_sims  64: 3.13 s/game  (3.30, 3.01, 3.08)  ~51 examples/game

Measured cheap/full cost ratio 64-vs-200 sims: **0.40** (not the naive
64/200 = 0.32; fixed per-move overhead). Revised S3 arithmetic:
p_full 0.25 -> ~1.8x games/hour; p_full 0.5 -> ~1.4x. (Block-average
s/game is lower, ~5.3s, because opponent-slice games search only half
the moves.)

### 5e. THE S4 REDIRECT: generation is ~90% Python, not GPU

Per 200-sim self-play game: ~45 searched moves x 17 waves x 0.91 ms =
~0.7s of GPU in a 7.77s game (~9%). Combined with 5b/5c: coalescing
forward passes across games attacks only ~9% of generation time as
currently shaped. The dominant cost is per-sim Python -- tree ops,
GameState clones, tensor building. Cross-game batching only pays if the
Python side scales too: multiprocess game actors feeding a shared eval
server (N processes x tree work) rather than single-process coalescing.
Alternatively/complementarily: push clone+step+tensor into the C++ engine,
or vectorize board_to_tensor. The 6x GPU headroom (5b) is real but is the
CEILING of a second-order term until Python parallelism exists.

## 6. S5b -- champion(+search) vs gregory(d3), first-ever measurement

Same instrument as the baseline and the certified h2h protocol:
`_play_fixed_match`, fixed color-swapped openings, seed 8801. mcts_100 =
`MCTSAgent(raw, n_sims=100, c_puct=1.5, temperature=0.0)` (wave_size 1,
benchmark_suite's certified configuration). Raw baselines: gen-6 0.138
(section 2).

    certified gen-5 tactical : 0.283  (300 games, SE ~0.03)
    certified gen-5 mcts_100 : 0.342  (60 games, SE ~0.06)
    teacher   gen-6 tactical : 0.247  (300 games, SE ~0.03)
    teacher   gen-6 mcts_100 : 0.342  (60 games, SE ~0.06)

**Finding: gregory(d3) beats our BEST configuration, not just the raw
net.** The ladder is monotonic -- raw 0.138 -> tactical ~0.25-0.28 ->
mcts_100 0.342 -- but even the strongest certified mode loses roughly 2:1
to a depth-3 alpha-beta whose moves cost single-digit milliseconds.
Context: gregory was never in the M2 anchor panel, so "beats every
measured anchor" was true only while gregory went unmeasured.

Second finding: **generation progress does not transfer to this matchup.**
Gen-6 beat gen-5 head-to-head 56% and raised the winblock/random panels,
yet vs gregory it measures the same (tactical 0.247 vs 0.283, mcts_100
0.342 vs 0.342). The lineage is not learning anti-minimax play from the
current mix -- which is precisely the S1 thesis, now with the strongest
evidence yet.

Implications for the queue:
- S1's motive upgrades from "distribution gap" to "a millisecond-cheap
  alpha-beta currently outplays our deployed agent in every mode" -- the
  gap is the product, not just the proxy. The gregory(d3) panel is now the
  primary yardstick of real progress, as its flag help predicted.
- Search-over-bad-priors appears capped: +100 sims buys ~+0.06-0.10 over
  tactical here, vs the much larger jumps it buys on the M2 anchors. If
  the net's priors/value never consider the lines minimax plays, more
  sims help sublinearly. Fix the data (S1), then re-measure this ladder.
- S5a (browser MCTS "Brutal" mode): 100 sims is NOT sufficient against
  d3-level play; do not market it as such. Re-size after S1 lands.

## 7. What this box pastes back after ~1 gen on the new mix

Per the runbook step 5: the promote_gregory series before/after restart,
`greg_games` > 0 proof, the S0 gen/train split from the NEW metrics fields,
and the segment's promotion lines. **Delivered: see section 9.**

## 8. Index

- Baseline + verdict: section 2 (script: `scripts/baseline_vs_gregory.py`).
- Segment config commit: `2b55cba` (start_goat.bat flags + baseline note).
- Docs: `STRENGTH_NEXT.md` (S-queue reasoning), `PENDING.md` (runbook).
- Diagnostics ran from throwaway scripts against repo APIs only
  (`_play_fixed_match`, `collect_game`, `MCTSAgent`); protocols and seeds
  are fully specified above for codification if wanted.

---

# 9. S1 segment verdict -- PASS (with caveats)

*Appended 2026-07-15, the day gen-8 promoted. Gen-8 is the judgment gen: its
entire 19,424-game window was generated on the S1 mix.*

## 9a. The segment

    [**] INFERENCE PROMOTION: raw student beat raw teacher 55% | winblock 42%
         | random 90% | gregory 17% -> gen 7    (2026-07-14, games_total 71,488)
    [**] INFERENCE PROMOTION: raw student beat raw teacher 55% | winblock 50%
         | random 90% | gregory 18% -> gen 8    (2026-07-15, games_total 90,912)

Gen-8 segment: 62 promotion checks, 19,424 games, promoted at head 0.5517 /
winblock 0.4983 / random 0.8967 / gregory 0.1750, lr at the 1e-4 floor.
Weights preserved: `models/expert_iter_v2/snapshots/teacher_gen8.pt`
(SHA-256 `113a63c2...9747`, payload verified gen=8, value_tanh=True,
24 tensors / 6,766,386 params). NOT certified or shipped (standing hold).

## 9b. Mix proof -- the S1 slice ran at exactly its target

`greg_games` across 1,512 instrumented blocks: 2,431 gregory(d2) games,
mean **1.61 per 16-game block = 10.05%** against a `--greg_mix 0.10` target.
Distribution 0-7/block (490 blocks at 1, 423 at 2, 275 at 0, 225 at 3, ...)
-- i.e. per-block sampling, not a fixed quota. S0 split at the gen-8
promotion block: `gen_secs 74.4`, `train_secs 1.7` (generation is ~98% of a
non-eval block; matches section 5c).

## 9c. The ruler: gregory(d3) per rebuild segment

Segment N = the student's rebuild toward teacher gen N (rows measured against
teacher N-1, closed by the promotion row). Never trained against: training
uses d2, the ruler is d3.

    seg |  n | mean  | 1st half -> 2nd half | half-over-half | max
      3 | 18 |  6.53 |  5.90 ->  7.17       | +1.27          |  9.0
      4 | 21 |  8.09 |  6.73 ->  9.32       | +2.59          | 12.5
      5 | 33 | 10.00 |  9.03 -> 10.92       | +1.89          | 14.5
      6 | 44 | 11.62 | 10.46 -> 12.77       | +2.31          | 16.7
      7 | 66 | 14.20 | 12.66 -> 15.74       | +3.08          | 21.5   <- pre-S1
      8 | 62 | 18.29 | 16.15 -> 20.44       | +4.29          | 25.2   <- S1
      9 |  5 | 15.70 | (rebuild in progress)                 | 17.2

Section 4 set the bar: "call S1 real if the gregory slope visibly steepens
past ~+4 pts/gen post-restart." Gen-8 = **+4.29**. That clears it, but only
just, and the slope alone would be a thin result. The stronger evidence is
the level shift, below.

## 9d. Did gregory outrun GENERAL strength? (the confound that matters)

The null hypothesis: gen-8 just got stronger across the board and gregory
rode along. Test it against the panels that contain no minimax at all.
Gen-over-gen change in segment mean:

    step | gregory(d3) |  winblock |   random |    head
    3->4 |      +1.56  |    +3.35  |   +2.15  |   -1.59
    4->5 |      +1.91  |    +2.96  |   +1.82  |   -1.42
    5->6 |      +1.63  |    +3.74  |   +2.80  |   +0.74
    6->7 |      +2.58  |    +5.10  |   +2.03  |   -0.89
    7->8 |      +4.09  |    +4.17  |   +1.95  |   +0.00   <- S1

    same, relative to the previous segment mean:
    3->4 |     +23.9%  |   +16.4%  |   +2.8%  |   -3.1%
    4->5 |     +23.6%  |   +12.5%  |   +2.3%  |   -2.8%
    5->6 |     +16.3%  |   +14.0%  |   +3.5%  |   +1.5%
    6->7 |     +22.2%  |   +16.7%  |   +2.4%  |   -1.8%
    7->8 |     +28.8%  |   +11.7%  |   +2.3%  |    0.0%   <- S1

Read it this way:
- **Gregory posted its best-ever gain (+4.09 pts, 1.58x the previous best
  +2.58) in the same segment where winblock posted a BELOW-trend one**
  (+4.17 vs gen-7's +5.10). General strength did not surge in gen-8;
  gregory did.
- Relative-gain ratio gregory/winblock: 1.46, 1.89, 1.16, 1.33 historically
  -> **2.46 in gen-8**. First segment where the minimax ruler moved ~2.5x
  faster than general strength.
- `head` sits at ~48-50 every segment BY CONSTRUCTION (promotion fires when
  it crosses 55, so the segment mean cannot drift) -- it carries no signal
  here. `random` is saturating (87.5) and is a floor, not a yardstick.

## 9e. Instrument note: do NOT judge this from the promotion line

The promotion-boundary readings tell the OPPOSITE story from the segment
means, and the boundary is the biased one:

    boundary gregory: .080 .125 .138 .143 .168 .175   (gen-8 delta: +0.7 pts)
    boundary winblock: .247 .295 .350 .382 .422 .498  (gen-8 delta: +7.7 pts)

That looks like winblock surged and gregory stalled -- exactly backwards from
9d. Cause: **promotion requires winblock >= best + 0.02, so a promotion row is
by definition a row where winblock came in high.** The boundary winblock is a
selected sample, biased upward, and is not a valid gen-over-gen measurement.
Gregory has only a no-regression floor (never a "must be high" bar), so its
boundary value is unbiased -- but it is a single 300-game panel (SE ~2 pts),
which is why +0.7 there is noise, not a finding. **The 62-check segment mean
is the statistic; the promotion line is an announcement.** Same family of
error as the retired self-oracle criterion (CHAMPIONS.md).

## 9f. Verdict

**S1 PASS.** The gregory(d2) training slice is doing what it was bought to
do: the independent d3 ruler moved +4.09 pts of segment mean (best ever, 1.6x
the prior record) and steepened to +4.29 half-over-half (past the pre-set ~+4
bar), while the no-minimax panels moved at or below their own trend. Keep the
mix at `--greg_mix 0.10`.

Caveats, stated plainly:
- **n = 1 generation, uncontrolled.** There is no gen-8-without-S1 to compare
  against. The argument rests on gen-8 breaking a 5-segment trend line, not
  on a counterfactual.
- **Gregory's base is low** (14.20 vs winblock's 35.60), so relative-gain
  comparisons flatter it mechanically. This is why 9d leads with absolute pts
  (+4.09 vs its own +2.58 record) rather than the +28.8%.
- **We are still losing to gregory(d3) roughly 4:1** (0.175 = it wins ~82%).
  S1 bent the slope; it did not close the gap. Section 6's finding stands:
  a millisecond-cheap alpha-beta still outplays our best mode.
- Gen-9's rebuild opened at gregory 15.7 mean over 5 checks -- the normal
  post-promotion dip (the student restarts against a stronger teacher), not
  a regression.

Next per the runbook: S2 (`--value_blend 0.4`) is authored and sign-test
green; enable it at a promotion boundary keeping the S1 flags. Gen-8's
promotion (2026-07-15) is that boundary.
