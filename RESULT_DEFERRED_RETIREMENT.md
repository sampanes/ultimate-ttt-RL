# RESULT -- deferred tree retirement: the reserve goes 95 ms -> 20 ms, and it is the search that sets it again (2026-08-15)

Task #46. `RESULT_NATIVE_SELECT.md` closed by promoting `release()` from a
cleanup item to a prerequisite: the reserve had gone 20 -> 35 -> 50 -> 95 ms
across four engines, every millisecond of it caller-side work outside the
search's own deadline, and it grows with every throughput win because what it
pays for is a walk over a tree that throughput makes bigger.

The brief was explicit that the first step was NOT to optimise the walk but to
find out whether it needs to be on the move path at all, and that memory decides
which architecture is available. So: characterise first
(`tools/profile_release.py`), then build.

`agents/mcts.py`, `tools/profile_release.py`, engine `pocket_defer`.

**PROMOTED 2026-08-15 at commit `0224669`: `pocket_defer` is the deployment
baseline.** See "What this does NOT license" at the end for the exact scope of
what the promotion claims -- it is one sentence, and it is about the stack, not
about any single change inside it.

## The answer

**Caller-side overhead p99 falls from 62.96 ms to 0.05 ms, the reserve goes back
to 20, and the engine completes 32.2% more search per full-clock move than
`pocket_graph` and 73.2% more than the deployed `pocket_r35`.** The gate passes
5/5 twice, and **the stack beats the deployed engine 0.5625 [0.5273, 0.5977]
over 240 paired games at equal wall clock** -- inside the band predicted from
throughput before the match was run.

Nothing was made faster. The O(nodes) walk still happens; it happens at the game
boundary, which no deadline covers.

| `tools.regress_engine`, 10 games vs `final` | `pocket_sel` (r95) | `pocket_defer` (r20) |
|---|---|---|
| **caller-side overhead p99** | 62.96 | **0.05** |
| caller-side overhead mean / max | 20.83 / 72.68 | 0.04 / 0.05 |
| latency p99 | 967.5 | 980.8 |
| latency max | 1007.1 | 981.7 |
| worst chunk p99 | 7.57 | 6.33 |
| tree reuse adoption | 0.955 | 0.960 = structural ceiling |
| inherited / new simulations | 0.502 | 0.515 |
| gate | 5/5 | **5/5** |

**The reserve means what it was designed to mean again.** It exists to absorb
chunk overrun -- a chunk is atomic, so its duration is the floor on how far a
search can overrun no matter how good the predictor is -- and for four engines
running it has instead been paying for tree destruction. It is 20 rather than 10
because the worst chunk still reaches 11.3 ms, so the number is set by the
search, not by housekeeping.

## #46a: characterising the walk before touching it

`tools/profile_release.py`, `pocket_sel` against `final`, 8 games, deployment
conditions (automatic cyclic GC off, collect at game boundaries).

| release, ms per move | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| | 20.39 | 20.34 | 47.98 | 53.27 | 61.63 |
| `_adopt` | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

Against a caller-side overhead p99 of 53.33 ms. **`release()` is not part of the
overhead, it is the overhead** -- 53.27 of 53.33.

    0.552 us per node, intercept -0.022 ms, R^2 0.9858, n=185

**There was nothing in it to optimise.** A fit that tight means no fixed cost to
attack and no per-node work worth shaving, which also settles the last item on
the brief's list: porting the walk to C++ would have recovered a fraction of
work that does not need to happen on the move path at all.

| what a move retires | |
|---|---|
| retired nodes / move | 38,513 mean, 90,264 p95, 137,467 max |
| of which expanded (these carry the mirror) | 4,967 -- 12.9% |
| retained after adopt | 18,230 -- 32.1% of the tree that existed |
| live tree after the search | 55,752 nodes |

### The number the design turned on

Per-game retired totals, eight games: 688k / 790k / 812k / 816k / 898k / 943k /
1,027k / 1,148k nodes. At **430.1 bytes a node** that is 282 to 471 MB, and the
worst game plus its live tree is **532 MB -- 1.4% of this box's 31.9 GB and 2.6%
of what was free.**

So the simplest architecture was available, which is what got built: detach on
the move path, destroy at the game boundary.

### How bytes-per-node was measured, and two ways of getting it wrong

Both were hit and both returned a clean, plausible, wrong answer.

**A build-then-free-then-build loop reports 0.0 bytes per node.** The second tree
is served out of the arenas the first one left behind. The fix is to build trees
CUMULATIVELY and hold them, which is also precisely the regime a deferred queue
puts the process in.

**A process that has just played a match reports 0.0 as well**, for the same
reason at a larger scale -- it is sitting on hundreds of megabytes of freed tree.
The measurement runs in a fresh subprocess.

**And `GetCurrentProcess` left at ctypes' default `c_int` restype reports 0.**
The pseudo-handle 0xFFFFFFFFFFFFFFFF truncates to a 32-bit -1, the call fails,
and the tool would have read "the tree costs nothing".

With those fixed: **430.1 bytes/node** from the commit charge, against **397.0**
from a `sys.getsizeof` structural walk -- 8% apart, which is what the GC header,
the 16-byte size-class rounding and the C++ vectors account for. Per-tree deltas
are 423 / 422 / 416 / 428, so the arena quantisation that produced 336-534 on
smaller trees is gone.

Trees are built by the engine's OWN `_build_children_mirrored`, at the branching
factor the game arm measured, on the CPU. A process-RSS delta taken around a real
search would also capture whatever the CUDA caching allocator did that second,
which is not a tree.

## The design, and the two things that make it small

**The queue is a list because retired roots nest.** Retired root k's subtree
CONTAINS retired root k+1 -- k+1 is the survivor that was adopted out of k -- so
`release(oldest, keep=live_root)` destroys the whole queue in one walk and stops
at the live tree. Later entries then walk an already-emptied structure for
nothing. Adoption misses start a new chain, which is the only reason it is a list
and not a single slot.

**The watermark counts on the way IN.** `MCTS.stat_nodes_created += len(valid)`,
once per expanded node, about 4,700 adds a move. Counting on the way out would
mean a per-node increment inside `release()` -- and `release()` is shared with
`final`, `pocket_graph` and every anchor rung, so that would have slowed the
incumbent arm of this change's own A/B. `release()` itself is untouched.

`alive_bound()` is therefore an over-estimate: it counts the live tree as well as
the queue. That is the right quantity for a MEMORY bound.

`DEFAULT_WATERMARK` is 3,000,000 nodes, about 1.2 GB and 2.6x the worst game
observed. It is the ONLY thing that can put the walk back on a move path, and
when it does it degrades to exactly the behaviour it replaced -- a pathological
game should get slow, not novel. `stats()["forced_drains"]` is non-zero if it
ever fires. It did not fire in any run reported here.

### Measured deferral bookkeeping, from the gate

    drains 9, forced 0, 525 ms per boundary drain, 865,861 nodes held (355 MB)

#46a predicted 441 ms of release per game and a 365 MB mean. Both land.

## Parity is the gate, for the third time

Deferral cannot change what is computed: the same nodes are built, the same
statistics kept, and only the moment of destruction moves. That is a claim to be
proved, not argued, so it is gated the same way native selection was --
`test_deferred_policies_are_bit_identical_across_rerooting` runs whole searches
at a fixed simulation count with deferral on and off and requires **identical
visit policies across six plies of re-rooting**, plus identical adoption counts
and inherited simulations.

Around it: that the queued trees really are still alive (or #46a's memory
projection describes nothing), that they really do die by refcount at the
boundary with the collector off (or it leaks), that the non-deferred arm still
frees on the move path (or the first test passes for the wrong reason), that the
watermark fires and the search survives it, and that `stat_nodes_created` matches
an independent walk of the tree that actually exists.

`tools/test_profile_release.py` additionally guards `counting_release`, which is
a hand copy of production's walk carrying one extra tally -- taken because the
count has to come from inside the walk, and a copy of production code drifts
silently. It is compared field by field against the real function on real
mirrored trees, and **three tests plant defects it must catch**, including
re-introducing #45a's `kids` bug.

## Throughput: the honest number, and the one not to quote

**Do not quote nn-evals per move.** It is contaminated by the early-stop mix: a
move that returns on a proven root costs almost nothing and drags the per-move
mean down, so two runs with different proof structure are not comparable even at
identical speed. Between these two runs the early-stop share moved 0.213 ->
0.107, which mechanically raises nn/move with no change in speed. The same trap
moved #45a's numbers in the other direction, and `pocket_r35` has measured 3,047
and 2,489 nn/move on two separate gate runs of the SAME engine.

**Quote nn-evals per SECOND times the search deadline.** An early stop adds
almost nothing to either the numerator or the denominator of nn/second, so the
composition cancels; multiplying by the deadline gives what a full-clock move
completes, which is what strength depends on.

| | nn/second | deadline | **nn per full-clock move** | run-to-run spread |
|---|---|---|---|---|
| `pocket_graph` | 5,095 / 5,008 | 950 ms | **4,798.7** | 1.7% |
| `pocket_sel` | 6,328 / 6,623 | 905 ms | **5,860.4** | 4.5% |
| `pocket_defer` | 6,279 / 6,672 | 980 ms | **6,346.0** | 6.1% |

    pocket_sel   vs pocket_graph   +22.1%
    pocket_defer vs pocket_graph   +32.2%
    pocket_defer vs pocket_sel      +8.3%

**The +8.3% is exactly the arithmetic entitlement** -- the search deadline moved
905 -> 980 ms, which is +8.29% -- and nn/second is flat at -0.8%. Two
independent things agree: deferral does not make the search faster, it buys the
search time, and the estimator recovers the mechanism to a tenth of a point.

Note that this estimator says `pocket_sel` was worth +22.1% over `pocket_graph`
where `RESULT_NATIVE_SELECT.md` reported +12.2% from nn/move. The difference is
composition, in the direction that understated it: that run had an early-stop
share of 0.213 against `pocket_graph`'s 0.155. The +12.2% is not withdrawn -- it
is what nn/move said -- but nn/full is the better estimator and is what #46
onward uses.

## Sizing the strength match from throughput, not from a sizing match

The brief for #46 is explicit: size any confirmation from `measured deployment
throughput gain -> clock-ladder predicted effect -> observed draw/variance
rate`, and never from another sizing run. #45a's 120-game 0.5708 is spent.

The ladder (`uttt-anchor-ladder-ordered`) says one doubling of the clock is
worth 0.59 to 0.69, so a throughput ratio converts directly into an expected
score. The correction that has to be applied is the one #45a measured: two
engines sharing a NETWORK predict each other's replies, both inherit far more of
their own tree, and the in-match throughput edge came out at +5.2% where the
gate said +12.2% -- a factor of 0.43.

| candidate vs | nn/full | gain | in-match gain | predicted score | games to resolve |
|---|---|---|---|---|---|
| `pocket_sel` | 5,860.4 | +8.3% | +3.6% | 0.5046 to 0.5097 | 2,700 to 12,000 |
| `pocket_graph` | 4,798.7 | +32.2% | +13.7% | 0.5167 to 0.5353 | 207 to 922 |
| **`pocket_r35`** (deployed) | **3,664.1** | **+73.2%** | **+31.2%** | **0.5352 to 0.5744** | **47 to 208** |

**The match to run is against the deployed engine.** Isolating deferral alone is
not fundable at any price and would not decide anything; against `pocket_graph`
it is marginal and `pocket_graph` is not deployed either, so winning would
settle nothing. Against `pocket_r35` the effect is the largest and the cost is
the lowest, and it is the question the branch exists to answer.

That bundles three changes, which is legitimate here for a specific reason: each
one is separately proven not to change what is computed -- the graph wave
returns bit-identical priors and leaf values, native selection returns the same
child index on every selection, deferred retirement returns identical visit
policies. The bundle is therefore a pure throughput change, and equal-clock win
rate is the only thing left to measure about it.

**Pre-registered before the match**: 240 paired games, seed 7300
(`release_ab`, held out), predicted 0.5352 to 0.5744, sized to resolve the LOW
end of that band. #45a's one calibration point landed just above the top of its
compressed band, so the low end is the conservative sizing target rather than
the expected outcome.

## The match: 0.5625 [0.5273, 0.5977], and the prediction was right

Paired openings, colours swapped, equal wall clock, both arms meeting the
frozen latency requirement.

| | W/D/L | n | score | 95% CI | |
|---|---|---|---|---|---|
| `pocket_defer` vs `pocket_r35` | 54/162/24 | 240 | **0.5625** | [0.5273, 0.5977] | **excludes 0.5** |

Predicted before the match: 0.5352 to 0.5744. **Observed 0.5625, inside the
band.** This is the first equal-clock strength result on this branch whose size
was set in advance from throughput rather than read off a sizing run, and the
prediction held.

**The ladder is now calibrated three times.** In-match throughput came out at
+41.9% (nn/second times the search deadline, 4,334.4 against 3,055.0), which is
a mirror-network compression factor of 0.572 on the gate's +73.2% -- less
compression than #45a's 0.426, so the pre-registered band was pessimistic.
Feeding the OBSERVED in-match gain through the ladder gives 0.5454 to 0.5959,
and 0.5625 sits in the middle of it.

**Latency: the candidate is tighter than the engine it is challenging.**

| | p50 | p99 | max | over budget |
|---|---|---|---|---|
| `pocket_defer` | 978.9 | 980.2 | 981.9 | 0 of 5,562 |
| `pocket_r35` | 970.5 | 986.1 | 1003.4 | 1 of 5,553 |

Caller-side overhead p99 0.05 ms against 22.38.

**240 games was three times more than this needed.** At the observed spread (SD
0.2781, 67.5% draws) the effect was resolvable in 76 games. The over-sizing came
from using #45a's compression factor, which is the right way to be wrong.

### What the match does and does not separate

It compares the CANDIDATE STACK against the DEPLOYED ENGINE, so what it
establishes is that the stack is stronger at equal wall clock. It does not
attribute the gain among the three changes.

In particular, `pocket_graph` already beat `pocket_r35` at 0.5458 [0.5126,
0.5791] (#42). The increment to 0.5625 is +0.0167 with a standard error on the
difference of about 0.025, so **native selection and deferred retirement are
not separately demonstrated on top of the graph** -- the point estimate moved in
the direction and roughly the magnitude the ladder predicts for their combined
throughput, and that is all that can be said. Isolating deferral alone would
need thousands of games, which is why it was not attempted.

## What this does NOT license

**PROMOTED 2026-08-15, recorded at commit `0224669`.** The owner took the
decision after this document was written; `tools/engine_registry.DEPLOYED` is
now `pocket_defer`, `pocket_r35` moves to `SUPERSEDED` and stays buildable as
the historical comparator, and the promotion is logged in
`engine_registry.PROMOTIONS` with the pre-registered band next to the result.

The supported claim is exactly one sentence, and it is narrower than the set of
changes that shipped:

> `pocket_defer` as a complete 1-second agent is stronger than `pocket_r35`
> at equal wall clock.

**Promotion does not retroactively license the attribution above.** Native
selection and deferred retirement are still not separately demonstrated, and
nothing downstream should say they are. What the promotion says is that the
stack ships; what the section above says is why nobody can yet name which part
of it earned the last 0.017.

The one thing being the baseline changes, besides which engine other work is
measured against: `pocket_defer` joins `STRICT_SOURCE_ROLES`, so a source edit
underneath it is a hard failure rather than a warning. It is the B side of
every future A/B, and a B side that moves quietly is worse than no baseline at
all. It deliberately does NOT join `ANCHOR_ROLES` -- an anchor may never be
overridden, and the next question is an ablation off this very engine
(`engine:pocket_defer+solve=0`).

**A mirror-network match will show much less than +32.2%.** Two engines sharing
a network predict each other's replies and both inherit far more of their own
tree, which compresses the difference: #45a measured +5.2% in-match where its
gate said +12.2%. Applying that factor here gives about +13.7% in-match.

**The boundary is not free, it is untimed.** A drain costs 525 ms and the
end-of-game collect runs while the queue is still full, which costs 246 ms
instead of 73 -- 173 ms per boundary, 711 ms of boundary work per game either
way. In deployment that lands at game end. `play_match` collects before
`new_game()` drains, and reordering it was NOT done: it means touching the shared
match protocol every published result uses, which is a large blast radius for
173 ms of untimed work.

**This change is structurally unmeasurable on fixed positions.** It only acts on
the re-rooting path, and a fixed-position run searches from a bare root, so
`release` is never called. Game mode is the only workload that can see it, which
is why the composition confound had to be handled rather than avoided.

## Reproduce

    python -m tools.test_profile_release
    python -m agents.test_mcts_timed
    python -m tools.profile_release --engine pocket_sel --games 8 --tag release46
    python -m tools.regress_engine --engine pocket_defer --games 10 \
        --tag defer-candidate
    python -m tools.regress_engine --engine pocket_sel --games 10 \
        --tag sel-recheck
    python -m tools.arena_1s --mode h2h --player-a engine:pocket_defer \
        --player-b engine:pocket_r35 --games 240 --seed 7300 --tag defer-h2h
