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

### Measured update, 2026-07-14 (training box; full tables in RESULT_S1.md sec 5)

The S0/S3/S4 decision data now exists; the speed items below were re-scoped
around it. Headlines:
- Generation is 85% of block wall-clock (train ~2s of a ~95s block).
- Within a 200-sim self-play game the GPU does ~0.7s of 7.77s (~9%). The
  per-sim cost is Python-side: MCTS tree bookkeeping, per-leaf tensor
  building, pybind attribute crossings. The game RULES are already C++
  (`engine/game.py` loads the pybind GameState: clone/make_move/
  valid_moves) -- do not re-solve those.
- Forward-latency sweep: batch 12 costs the same ~0.9 ms as batch 1;
  batch 256 is the sweet spot at 78.6k pos/s (6.0x today's 13.1k);
  1024 degrades, 4096 collapses.
- Cheap/full search cost ratio, 64-vs-200 sims: 0.40 measured (not the
  naive 0.32; fixed per-move overhead).
- Live run: 22-30% GPU, 4.1/10.2 GiB. Box has 24 logical cores (Ryzen 9
  3900X); the run occupies 1-2 of them.

Consequence: forward-batching alone attacks ~9% of generation; every real
multiplier below is Python-parallelism or fewer/cheaper sims. Owner
directive 2026-07-14: every speedup that does not cost agent skill is
FORMALLY REQUESTED -- statuses on S3/S4 and the new S7/S8 reflect that.

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

## S0. Log the generate/train time split  [DONE 2026-07-13, authoring box]

`gen_secs` is computed and printed every block (`expert_iter.py`, the
"gen Xs / total Xs" line) but never logged. Add `gen_secs` and a
`train_secs` sibling to the `append_metrics(extra=...)` dict so the
dashboard and log show where block time actually goes. Non-behavioral,
zero risk. This is the decision data for S3/S4.

**Status: LANDED.** Both fields stream to `loss_logs/metrics_log.jsonl`
automatically once the run restarts on this commit; no flags needed.

## S1. Gregory joins the training mix -- d2 in curriculum, d3 stays ruler

The headline item; decided 2026-07-13.

**Status: LIVE since 2026-07-13.** Baseline said ENABLE (raw gen-6 scored
0.138 vs d2 AND d3 -- identical, so the d2 slice loses no signal); flags
`--greg_mix 0.10 --opp_mix 0.30 --rnd_mix 0.10` are in `start_goat.bat`
(commit 2b55cba), `greg_games` confirmed streaming. T0 data package:
RESULT_S1.md. Gen-8 is the judgment gen; success bar = gregory(d3) slope
clearly beating the pre-S1 drift (~+3.4 pts half-over-half within gen-7).

Original authoring notes kept below for the record. `--greg_mix`
(default 0.0) + `--greg_mix_depth` (default 2, hard-errors if >= the ruler
depth) are in `expert_iter.py`, with a `greg_games` counter in metrics and a
slice-layout regression test. The pre-step baseline is
`scripts/baseline_vs_gregory.py` (CPU-only, safe while the run is live, and
it prints the enable/stop verdict itself).

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

**Status: AUTHORED 2026-07-13, opt-in, NOT yet enabled.** `--value_blend`
(default 0.0 = byte-identical, capped at 0.5 by argparse) on `expert_iter.py`
threads through `collect_game(value_blend=...)`; targets are precombined via
`_blend_value` so the shard schema stays `(x, pi, z)`. Tactics-shortcut
positions keep pure z; q_root is defensively clamped to [-1, 1] (untanhed
gen-0 teacher = the 2026-07-09 poisoning class). The sign warning below is
now a test: `agents/test_mcts.py::test_root_q_sign_on_won_root` (root Q
strongly positive on a won root, winning child exactly -1). Enable per the
PENDING.md runbook -- the segment AFTER S1 is judged, ideally right after a
promotion so the fresh window is uniformly blended.

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

## S3. Playout-cap randomization  [REQUESTED 2026-07-14 -- speed track, smallest diff]

**Status: formally requested; not yet authored** (verified 2026-07-14: no
p_full/playout code in the repo). Measured re-arithmetic (RESULT_S1.md 5d):
the cheap/full cost ratio at 64-vs-200 sims is 0.40, so the honest
projection is ~1.8x games/hour at p_full=0.25 and ~1.4x at p_full=0.5 --
the 2.2x below used the naive 0.32 ratio; keep the structure, use these
numbers. Skill-neutrality condition (binding, the reason this qualifies
for the owner's "no skill cost" list): policy examples recorded ONLY from
full 200-sim searches, so per-example quality is unchanged; cheap moves
still play through a >=16-wave search holding a 0.80+ edge over raw, so
trajectory/z quality degrades little. Behavioral (changes the games/
examples mix) -> one-gen segment per rule 3.

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

## S4. Parallel generation  [FIRST CUT DONE + LIVE 2026-07-15 -- see RESULT_S4.md]

**Status: first cut (independent actor processes) LANDED. Measured +3.2x
games/hour on the live run (median block generation 84s -> 26s), --actors 8
in start_goat.bat. commit 0d04bc0.** Full record: RESULT_S4.md.

Key correction to the estimate: the "3-5x" assumed cores were the limit. They
are not -- it saturates on the GPU. 12 actors peg the GPU at 95% while 9 of 24
cores sit idle, because N tiny batch-12 forwards serialize across N CUDA
contexts (no MPS on this consumer card). So the first cut's ceiling is
~2.6-2.8x isolated / ~3.2x warm, NOT 5x. The remaining multiple is behind the
SECOND CUT (shared eval server, below), now promoted to the top of the queue.

**The original design below -- single-process coalescing of leaf evals across
games -- was DEAD as a primary lever** (GPU only ~9% of generation time), which
is why the first cut used independent actors instead. The redirect, as built:

- N game-actor PROCESSES, each owning whole games end to end (tree ops,
  clones, tensor building -- the ~90% Python term). Multiprocessing, not
  threads (GIL).
- First cut, simplest: fully independent workers, each doing its own
  batch-12 forwards on the shared GPU (WDDM timeslices fine at these
  sizes; measured batch-12 latency equals batch-1). Workers return
  finished games' examples to the parent over a queue; shard writes,
  buffer, and the block train step stay in the parent, single-writer.
- Second cut, NOW CONFIRMED NEEDED (the first cut showed exactly the GPU
  contention it was gated on -- RESULT_S4.md sec 3): one shared eval server
  process coalescing leaf waves from all workers toward the measured
  batch-256 sweet spot (78.6k pos/s, 6x ceiling). THIS IS THE NEXT BUILD.
- Sizing (as landed): 8 actors is the knee. Isolated A/B gave 2.46x at 8 /
  2.80x at 12; live warm pool gives 3.2x at 8. VRAM 5.7/10.2 GiB at 8.

Skill-neutrality: distribution-preserving, NOT byte-identical -- each
worker needs its own seeded RNG stream (per-root Dirichlet stays per-game
and correct), so gate with a timing A/B plus panels-inside-noise rather
than a byte parity oracle; land at a promotion boundary to keep the
window clean. Teacher weights are read-only during generation (blocks
alternate generate/train), so workers can hold a frozen copy per block;
refresh on block start.

Original notes (kept for the batching-precedent pointers): the repo
already has the pattern precedent (`ParallelGameRunner` + batched
opponents in train_league; the wave engine already dedups leaves within a
search). Risks: per-root Dirichlet/RNG correctness, memory, real refactor
surface in `collect_game`/`MCTS`. Windows note: multiprocessing spawn +
CUDA in workers works but each worker pays a CUDA context (~300 MB VRAM);
at 6-10 workers that is ~2-3 GiB against the 6 GiB currently free --
budget it, or route evals through the shared server (cut two) which needs
only one context.

## S5. Deploy-side strength (independent of training, anytime)

a) Browser "Brutal" mode: the certified best mode is mcts_100 (panel mean
   0.856 vs tactical's 0.800; vs winblock 0.722 vs 0.500), but the play
   page ships champion Hard = tactical argmax. `docs/play/agent.js`
   already has PUCT MCTS (pocket Hard = 50 sims since M3), so wiring the
   champion into search at ~100 sims is incremental frontend work; budget
   ort-web latency first (likely 1-3s/move, acceptable for an opt-in
   mode).

b) [DONE 2026-07-13, RESULT_S1.md sec 6] champion(+search) vs gregory(d3)
   measured for the first time: gregory beats EVERY deployed config (best
   mode mcts_100 scores 0.342; gen-5 and gen-6 identical -- gen-over-gen
   gains do not transfer). The gregory(d3) panel is now the primary
   yardstick, and (a)'s "Brutal" mode must NOT be marketed as beating
   d3-level play at 100 sims; re-size (a) after S1 lands.

## S6. Auto-snapshot the teacher at promotion  [ops, tiny]

`teacher.pt` is overwritten at each promotion; gen-6 was preserved only
by a manual copy to `models/expert_iter_v2/snapshots/teacher_gen6.pt`.
Have `_save_teacher` (or the promotion branch) also write
`snapshots/teacher_gen{N}.pt`. 26 MB/gen, gitignored, hundreds are fine
on disk; removes a manual step the training box currently performs by
hand at every promotion (done again for gen-7).

## S7. MCTS tree reuse between moves  [NEW ASK 2026-07-14 -- speed track]

**Status: formally requested; does not exist** (verified: `MCTS.search`
builds a cold root every move; `collect_game` constructs one MCTS per game
but carries no tree across moves). The move just played is typically the
child holding the plurality of the previous search's 200 visits; that
subtree is discarded and recomputed. Keeping it warm-starts each search
~30-50% for near-zero code cost -- LC0/KataGo standard practice.

Change: MCTS keeps its last root; after the game advances with move `a`,
descend to `children[a]` and adopt it (with its N/W/children intact) as
the next search's root. Re-apply Dirichlet noise to the adopted root's
priors -- root noise is a per-search decoration, not stored tree state.
In pure self-play both sides are the same searcher, so consecutive
searches are exactly one ply apart and reuse fires on every move. In
opponent-slice games descend two plies (net move, then the opponent's
reply) when that child exists, else cold-start; restricting reuse to the
self-play slice is an acceptable first cut.

Skill notes: same net, same sim budget, strictly more accumulated search
per position -- skill-neutral-to-positive. One care point: an adopted
root arrives pre-visited, so fresh root noise has less influence;
mitigations are the re-noising above plus optionally cold-starting during
the `temperature_moves` opening plies to protect opening diversity.
Behavioral (pi targets sharpen slightly) -> one-gen segment per rule 3.
Expected ~1.3-1.5x effective sims per wall-clock second; multiplies with
S3 (cheap searches reuse too) and S4 (per worker).

## S8. Per-sim hot path: tensor build + pybind crossings  [NEW ASK 2026-07-14]

**Status: formally requested.** The rules are ALREADY C++ (`engine/game.py`
loads the pybind GameState; clone/make_move/valid_moves), so the residual
~90%-Python generation cost is MCTS tree bookkeeping,
`board_to_tensor_from_gamestate`, and pybind attribute crossings -- every
leaf eval reads `state.board` / `mini_winners` / `last_move` through
pybind (container conversion per access) and then builds the (7,9,9)
tensor in Python. Candidates, to be profile-ranked before building (the
training box will paste a cProfile of `collect_game` on request):

  a) C++ `fill_planes(out)` on GameState writing the 7x9x9 input planes
     straight into a caller-provided numpy buffer -- one crossing per
     leaf, zero Python plane math. Batch variant taking a wave of states.
  b) Vectorized wave tensor build: one numpy op for all ~12 leaves of a
     wave instead of 12 sequential per-leaf builds.
  c) Tree-in-C++ (node storage, PUCT select, backprop) -- biggest lift;
     only if profiling shows tree ops still dominate after (a)/(b).

Skill-neutrality: pure infra, byte-identical outputs, parity-gated like
batch_opponents (`verify_opponent_batch_parity` precedent) -- may land
mid-segment, no gen judgment needed. Multiplies with S4 (faster workers
x more workers) and raises S4's per-worker ceiling.

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

1. S0 immediately (non-behavioral).  [DONE 2026-07-13]
2. S1 for one full gen.  [LIVE 2026-07-13 -- gen-8 is the judgment gen]
3. S2 for one gen, judged against S1's trajectory.  [AUTHORED 2026-07-13 --
   opt-in `--value_blend`; enable at the first promotion after S1 is judged]
4. Speed track (ALL formally requested by the owner 2026-07-14 -- "every
   speedup that does not cost agent skill"):
   - Behavioral items, one segment each per rule 3: S3 (playout-cap,
     ~1.4-1.8x, smallest diff) then S7 (tree reuse, ~1.3-1.5x). Default
     slotting is the two segments after S2; pulling them ahead of S2 is
     the owner's call if wall-clock is the current pain -- either order
     is fine, just one behavioral change per segment.
   - Infra items, parity/A-B gated, no gen judgment: S8 (hot path,
     byte-identical) may land mid-segment; S4 (multiprocess actors,
     distribution-preserving) lands at a promotion boundary. Both can be
     AUTHORED anytime in parallel with the segments above.
   - Compounded honest estimate on this hardware: S3 x S7 x S4 ~= 5-10x
     games/hour; S8 raises S4's per-worker ceiling further. Measured
     anchors: cheap/full ratio 0.40, batch-256 = 6x GPU headroom, 24
     cores at 1-2 used.
5. S5a re-sized after S1 lands (100 sims is not "Brutal" vs d3); S6
   anytime -- they do not touch the run.

Context for the queue's pace: certification of gen-6+ is deliberately
deferred (owner decision 2026-07-12) until the compounded margin justifies
interrupting the run, so there is no external deadline; the panels decide.
