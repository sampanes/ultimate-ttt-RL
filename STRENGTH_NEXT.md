# STRENGTH_NEXT -- long-horizon strength and robustness backlog

Machine-to-machine handoff, written 2026-07-13 on the training box (analysis
only; no training code was changed there). `PENDING.md` holds triggered
queues; this file holds the strength backlog WITH the reasoning, so the
authoring box does not have to re-derive it. Every code claim below was
verified against the live v2 run (gen-7 in progress, ~60k games) at the
commit this file lands on.

---

## The measured picture (why these items, why now)

- The promotion gates measure the RAW argmax net: no search, no tactics,
  draws count 0.5 (`scripts/expert_iter.py` `_play_fixed_match`).
- Raw-net panel scores at/near each promotion (from the metrics log):

      winblock:  0.12  0.20  0.26  0.23  0.35  0.38    (gens 1-6, ~+4 pts/gen)
      random:    0.77  0.73  0.80  0.79  0.83  0.87
      gregory:   (first measured gen 3)  0.07  0.11  0.11  0.12-0.14

  Gen-7 checks so far: winblock 0.27-0.38, random up to 0.89, gregory
  0.10-0.16.
- Net+search already beats every anchor it has been measured against
  (gen-5 mcts_100: lottery 0.889, nn_big8 0.806, winblock 0.722, center
  0.889, first 0.972; direct h2h 0.698 vs arena:22@hof -- `RESULT_M2_5.md`).
  The deployed agent is net+search; the raw-net gates are a proxy, not the
  product.
- gregory(d3) is the ONE panel opponent independent of the training gene
  pool ("the long-horizon honest ruler", its own flag help). The net has
  never trained against minimax-style play; that is why its gregory score
  crawls at +1-2 pts/gen while winblock moves at +4.
- Generation dominates block time and runs on tiny GPU batches: games are
  collected strictly sequentially (`expert_iter.py` generate loop), and the
  MCTS wave clamp (`agents/mcts.py`, `_MIN_WAVES`) means 200-sim searches
  do waves of only 200//16 = 12 leaf evals per forward pass. A 6.8M-param
  net at batch 12 leaves most of an RTX 3080 idle.

## Ground rules (binding, from the plateau lessons + design preferences)

1. Never train against the honest ruler. If a gregory depth joins the
   curriculum, that depth is burned as an instrument.
2. Diversify only against a MEASURED gap, never for variety's sake.
3. One change per run segment. Judge only by the fixed 300-game external
   panels. Expect a few checks of head/winblock wobble after any data-mix
   change and do not react to it.
4. Strongest-within-budget: no net scaling while the gates still move at
   the current size.

---

## S0. Log the generate/train time split  [tiny, land first]

`gen_secs` is computed and printed every block (`expert_iter.py`, the
"gen Xs / total Xs" line) but never logged. Add `gen_secs` and a
`train_secs` sibling to the `append_metrics(extra=...)` dict so the
dashboard and log show where block time actually goes. Non-behavioral,
zero risk. This is the decision data for S3/S4.

## S1. Gregory joins the training mix -- d2 in curriculum, d3 stays ruler

The headline item; decided 2026-07-13.

Motive: the one known distribution gap. The rnd_mix precedent proves the
fix class works (its flag help documents the measured random-panel
regression that opponent slice cured).

Change (mirrors the existing winblock/random branches in the generate
loop):
- New flag `--greg_mix`, default 0.0 (opt-in). Suggested first segment:
  0.10, donated equally from `--opp_mix` (0.35 -> 0.30) and `--rnd_mix`
  (0.15 -> 0.10) so pure self-play stays at 0.50. The winblock ratchet and
  random floor remain the regression guards for the donor slices.
- Opponent is `GregoryAgent(depth=2)`, constructed once at startup like
  heur/rnd. NOT depth 3 (rule 1: d3 is the ruler).
- Mini-tactics stay OFF in this slice (same reasoning as the random slice:
  the point is teacher-search targets on punished positions, not the
  WinBlock motif).
- Log a `greg_games` counter in extra alongside `opponent_games` /
  `rnd_games`.

Cost: d2 is ~1 ms/move (`agents/gregory.py` depth guide) x ~25 opponent
moves = ~0.03s per gregory game, on games that average ~6s wall-clock.
Zero measurable throughput change.

Pre-step (cheap, do before enabling): baseline the raw net vs d2 with one
300-game `_play_fixed_match` (d2 is ~1 ms/move; the match takes seconds,
CPU-only, can run on the training box without stopping the run). If the
raw net already scores >= 0.55 vs d2, the slice teaches little -- then
consider d3-in-mix and PROMOTE d4 TO RULER instead. d4 is ~10x d3 per
move, so a 300-game d4 panel (~400s) cannot run at the 30-min check
cadence; the workable variants are a 100-game d4 panel (~130s) or d4
measured only at promotion events. Decide with the baseline number in
hand; the d2-train / d3-rule split is the default because it burns
nothing.

Expectations (write these down so nobody panics):
- Success metric: the gregory(d3) panel slope, from +1-2 pts/gen to +4 or
  better. NOT promotion cadence -- promotions are gated on head-to-head
  and the winblock ratchet, which gregory games do not directly feed.
- head/winblock may wobble for a few checks after the mix change (rule 3).
- The gregory no-regress gate is already armed (best 0.143 >= arm 0.10),
  so a genuine regression still blocks promotion.

Note: this is unrelated to the old "Gregory curriculum integration" item
in `PENDING.md`, which wires gregory into the PG league_manager stage 5-6
pool. That item stands on its own.

## S2. Stop discarding the search's root value -- blended value targets

Today `collect_game` does `pi, _ = mcts.search(state)` and value targets
are pure game outcome z in {-1, 0, +1} (highest-variance estimator).
`MCTS.search` already returns the root node, whose visit-weighted value
(the same W/N the selection uses) is computed and thrown away.

Change: expose the root Q (from the root player's perspective) per
recorded position and train the value head on
`(1 - lam) * z + lam * q_root`, lam ~0.3-0.5 to start. Positions taken
via the tactics shortcut (win-in-1 one-hots, no search ran) keep pure z.
Simplest storage: precombine at example-build time so the shard schema
and buffer stay `(x, pi, z)`; alternatively store q as a 4th field and
version the shards.

SIGN WARNING: this tree stores W from the child's to-play perspective and
selection scores `-child.Q()` (see the M4 postmortem in `PENDING.md` --
the virtual-loss sign bug lived exactly here). Add a unit test in the
`agents/test_mcts.py` style asserting the exposed root Q sign on a
known-won root before trusting it.

Cost: zero compute (the number already exists). Risk: bootstrap bias --
q reflects the current net's misjudgments, so keep lam <= 0.5 toward z.
Judge over one gen: value_loss trend, the GOLD fixed suite
(`gold_endgame_suite.json`), and the h2h gate. Motive: the M4 thesis was
"fix the value head, unlock search," and it paid; lower-variance value
targets push the same lever again.

## S3. Playout-cap randomization  [experiment]

KataGo's generation trick: most moves only need to be PLAYED reasonably,
not labeled. Per net move, with p_full ~0.25-0.5 run the full 200 sims
and record the example; otherwise run a cheap 48-64-sim search, play the
move, record nothing.

Local supporting data: the wave clamp guarantees >= 16 waves per search,
and the 2026-07-04 sweep (in `agents/mcts.py` comments / M4 postmortem)
shows 16-wave searches hold a 0.80-0.925 edge over the raw net -- cheap
moves stay far above raw-policy quality, so game trajectories and z
integrity degrade little.

Arithmetic at p_full=0.25, cheap=56 sims: per-game cost ~0.46x -> ~2.2x
games/hour, but examples/hour ~0.55x (only full-search positions are
recorded). This trades example volume for game count and diversity
(decorrelated positions); in Go it nets out clearly positive, here it is
unproven. At p_full=0.5, cheap=64: cost ~0.66x -> ~1.5x games/hour,
examples/hour ~0.76x -- the gentler first experiment. Run exactly one gen
and judge by promotion wall-clock and panel slopes, not loss curves.
If S2 has landed, cheap-move positions can optionally record value-only
examples (KataGo records value everywhere, policy only on full searches).

## S4. Cross-game batched generation  [biggest lift, gate on S0 data]

Games are generated one at a time; within a search, leaf batches are
capped at n_sims//16 = 12 positions (see "measured picture"). Coalescing
leaf evals ACROSS N concurrent games into one forward pass is the classic
AlphaZero-engine speedup, and the repo already has the precedent pattern
(`ParallelGameRunner` + batched opponents in train_league; the wave engine
already dedups leaves within a search).

Expected 2-5x generation throughput; the 3080 is demonstrably underfilled.
Prereq: S0 confirming generation dominates block time (near-certain: 100
train steps at batch 256 are seconds, a 16-game block at 200 sims is not).
Risks: per-root Dirichlet/RNG correctness, memory, real refactor surface
in `collect_game`/`MCTS`. Do after S1/S2 land; do not combine with S3 in
the same segment.

## S5. Deploy-side strength (independent of training, anytime)

a) Browser "Brutal" mode: the certified best mode is mcts_100 (panel mean
   0.856 vs tactical's 0.800; vs winblock 0.722 vs 0.500), but the play
   page ships champion Hard = tactical argmax. `docs/play/agent.js`
   already has PUCT MCTS (pocket Hard = 50 sims since M3), so wiring the
   champion into search at ~100 sims is incremental frontend work; budget
   ort-web latency first (likely 1-3s/move, acceptable for an opt-in
   mode).

b) Missing measurement: champion(+search) vs gregory(d3) has NEVER been
   run -- the 0.14 gate number is the naked net. One CPU-only match on
   the training box (no run interrupt needed) closes the "beats
   everything measured" claim. Either add gregory as a benchmark_suite
   anchor or script `_play_fixed_match` with an MCTS-wrapped candidate.

## S6. Auto-snapshot the teacher at promotion  [ops, tiny]

`teacher.pt` is overwritten at each promotion; gen-6 was preserved only
by a manual copy to `models/expert_iter_v2/snapshots/teacher_gen6.pt`.
Have `_save_teacher` (or the promotion branch) also write
`snapshots/teacher_gen{N}.pt`. 26 MB/gen, gitignored, hundreds are fine
on disk; removes a manual step the training box currently performs by
hand at every promotion.

---

## Parked / do-not (recorded so they are not re-litigated)

- Train against gregory d3+: ruler corruption (rule 1). Only under the
  explicit d4-becomes-ruler swap in S1.
- Symmetry augmentation: ALREADY IN -- `train_on_examples` applies a
  random D4 symmetry per batch (`apply_dihedral_symmetry`,
  `scripts/train_alphazero.py`). A per-sample variant would add little.
- Bigger network: rule 4. Revisit only when winblock AND gregory slopes
  flatten across 2+ gens at the LR floor.
- torch.compile: no Triton on Windows; `enable_compile()` already degrades
  gracefully. Closed 2026-07-03 (`PENDING.md` flag table).
- Early resignation / adjudication: avg_len is ~51 plies and win-in-1
  shortcuts already skip search on forced moves; bounded savings,
  value-poisoning risk if resigns are wrong, calibration slice needed.
  Revisit only if S0 shows decided tails dominate generation time.
- Blind temperature / Dirichlet increases: no measured gap they close;
  dir_eps 0.10 + temperature_moves 10 already give opening variety, and
  touching them wobbles every panel (rule 3).
- Off-distribution position starts (branch self-play from buffer/random
  midgame states): rnd_mix already covers blunder-created positions and
  no current instrument shows a residual gap. The GOLD fixed suite is the
  detector; revisit if a future certification shows blunders concentrated
  off-distribution.

## Suggested order

1. S0 immediately (non-behavioral).
2. S1 pre-step (raw-vs-d2 baseline, seconds), then S1 for one full gen.
3. S2 for one gen, judged against S1's trajectory.
4. S3 or S4 next, chosen with S0's timing data; never both in one segment.
5. S5/S6 anytime -- they do not touch the run.

Context for the queue's pace: certification of gen-6+ is deliberately
deferred (owner decision 2026-07-12) until the compounded margin justifies
interrupting the run, so there is no external deadline; the panels decide.
