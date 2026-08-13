# RESULT -- native PUCT selection: 5.5x per call, +12.2% network evaluations, and a reserve that had to grow (2026-08-12)

Task #45a. #44 measured `_best_child` at 193.9 ms/move in deployment -- 22.2%
of the move, the largest host term by 64% -- and showed it grew 44% purely
because there were more calls, not slower ones. That made selection a
SELECTION-ENGINE project rather than a tree-port project: build one native
primitive over a mirrored child array, keep Python's `MCTSNode` authoritative,
and prove exact parity before measuring anything.

`engine/cpp/uttt_select.cpp`, `agents/native_select.py`,
`tools/select_parity.py`, engine `pocket_sel`.

## The answer

**`_best_child` costs 0.712 us/call instead of 3.947 -- 5.5x -- and the engine
that can actually be deployed completes 12.2% more network evaluations in the
same second.** Parity is exact: the native and Python selectors returned the
same child on all 7,695,074 selections tested, including 4.6M inside real games
with tree reuse.

Getting to "can actually be deployed" cost a third of the raw gain. At the
50 ms reserve it inherited, `pocket_sel` **fails the frozen latency
requirement** -- p99 1027.9 ms -- and needs 95. That is the third time this one
mechanism has forced a reserve up and the first time it has been expensive
enough to deserve a section of its own.

| clean arm, 40 identical fixed positions | `pocket_graph` | `pocket_sel` | change |
|---|---|---|---|
| wall ms/move | 878.2 | 878.6 | +0.05% |
| **network evaluations/move** | 4,345.4 | 4,695.1 | **+8.05%** |
| simulations/move | 6,180.4 | 7,223.2 | +16.87% |
| proven-subtree descents/move | 1,835.0 | 2,528.1 | +37.77% |

| deployment, 10 games vs `final`, reuse on | `pocket_graph` | `pocket_sel` | change |
|---|---|---|---|
| **network evaluations/move**, shipping reserve | 4,168.7 | 4,675.5 | **+12.2%** |
| network evaluations/move, both at reserve 50 | 4,168.7 | 4,903.4 | +17.6% |
| simulations/move, shipping reserve | 5,762.7 | 6,108.0 | +6.0% |

The shipping row is the one that counts: `pocket_sel` needs a 95 ms reserve to
meet the frozen latency requirement and `pocket_graph` does not, so **+12.2% is
what the engine that can actually be deployed delivers.** The reserve costs
about a third of the raw gain and the section below explains why it had to.

**Fixed positions understate this change by half, and that was predictable.**
#44 measured selection at 130.7 ms/move on fixed positions against 193.9 in
deployment, because tree reuse makes deployment trees deeper and selection
scales with descents. A change to the selection path therefore has to be quoted
in deployment; the fixed-position number is the controlled one, not the
representative one.

**Within either workload, quote network evaluations.** The two units disagree
by 2x for the reason #44 identified and this run makes vivid: the extra search
is disproportionately free descents into already-proven subtrees, which cost
almost nothing and still count as simulations. On fixed positions proven
descents rose 37.8% against 8.1% for the work that consults the network.

## Where the time went

From the in-situ arm -- one run per engine with ONLY `_best_child` wrapped, so
the instrument costs ~10% of the search instead of ~30%:

| | calls/move | us/call | ms/move | share of move |
|---|---|---|---|---|
| `pocket_graph` | 34,847 | 3.947 | 153.94 | 17.5% |
| `pocket_sel` | 38,478 | **0.712** | **30.80** | 3.5% |

`pocket_sel` makes 10.4% MORE selection calls because it searches more, and
still spends 123.14 ms/move less doing them.

**The mirror is not free, and it takes back half.** Measured as the change in
per-call cost of the three operations that maintain it, times the calls the
candidate actually makes:

| operation | us/call change | calls/move | ms/move |
|---|---|---|---|
| node creation (`_expand_children`) | +46.564 | 649.3 | 30.23 |
| wave loop (virtual loss apply + undo) | +24.619 | 893.2 | 21.99 |
| backup | +1.190 | 7,145.4 | 8.50 |
| **mirror maintenance** | | | **60.73** |

    selection saved       123.14 ms/move
    mirror maintenance    -60.73 ms/move
    ------------------------------------
    NET                    62.41 ms/move    7.1% of the move

The promotion inequality from the brief -- `selection_saved_ms >
pybind_call_cost + mirror_update_cost` -- holds by a factor of 2.0. It is not
the factor of 5.5 the per-call number suggests, and the difference is the
mirror.

**Two instruments agree.** The in-situ slope says one millisecond off the
selection path buys 4.9 network evaluations a move, so 62.41 ms should buy 306.
The clean arm measured 350. Those are independent measurements of a derived
quantity and they agree to 14%.

## The ceiling: respected where it applies, and the deployment number needs an account

The brief set a sanity bound -- a free selector was measured at +15.5% network
evaluations, and anything above it should be assumed to be instrumentation or
semantic drift until proven otherwise. The fixed-position result is
comfortably under: this run re-measured the ceiling on its own positions at
**+17.5%** and the implementation delivered **+8.05%, 46% of it**.

The deployment result is +12.2% at the shipping reserve and +17.6% at reserve
50, so the raw one is at or above the bound as stated and needs an explanation
rather than a shrug. Three things, in order of how much weight they carry:

1. **The +15.5% was explicitly a FIXED-POSITION ceiling.**
   `RESULT_SELECTION_PROFILE` says so in as many words: the slope was measured
   from a bare root, deployment selection costs 193.9 ms against the 140.0 the
   slope was applied to, "so the deployment ceiling is plausibly larger -- but
   that was not measured and is not claimed". Scaled by that ratio it would be
   about +21%. The bound was never a deployment bound.
2. **A bare-root slope structurally cannot see tree reuse compounding.** More
   search this move leaves a bigger tree for the next one: inherited
   simulations rose 24.4%. Each search starts further ahead, and that gain
   accumulates across a game in a way no fixed-position measurement can
   express.
3. **A 10-game sample cannot support three digits.** The two arms play
   different games the moment they choose differently, so they are not
   measured on the same positions -- 243 moves against 233. Game-mode
   throughput on this box replicates only to 16-21%
   (`uttt-concurrent-runs-corrupt-timings`, and #44's own replication check).

So: **+8.05% is the reproducible number and +12.2% is the deployment
indication with a wide error bar.** How wide: the same engine at reserve 50 and
at reserve 95 reports 8,198 and 6,108 simulations a move, a 25% gap that a 4.7%
cut in thinking time cannot produce. Two mechanisms, both compositional rather
than noise, and both invisible in a single number -- the arms played different
games, and the share of moves that ended early on a proven root went from 0.128
to 0.213, which mechanically lowers simulations per move because those moves
stop almost immediately. Network evaluations, being the unit that is not
inflated by free descents, moved far less: 4,903 to 4,675.

What is NOT a candidate explanation is semantic drift: parity makes the two
engines the same player at a fixed simulation count, checked over 7,695,074
selections including 4.6M in games.

The candidate's OWN in-situ slope is the more useful forward-looking number:
with selection down to 30.8 ms/move on fixed positions, making it free from
here would be worth a further **+3.5%**. Selection is no longer where the
money is.

## The latency gate failed, and `release()` is now the binding constraint

At the reserve it inherited from `pocket_graph`, `pocket_sel` misses the frozen
requirement:

| `tools.regress_engine`, 10 games vs `final` | `pocket_graph` (r50) | `pocket_sel` (r50) | `pocket_sel` (r95) |
|---|---|---|---|
| latency p99 | 991.6 **PASS** | **1027.9 FAIL** | 967.5 **PASS** |
| latency max | 1002.9 | 1043.9 | 1007.1 |
| caller-side overhead p99 | 42.77 (of 50) | **78.99 (of 50)** | 62.96 (of 95) |
| worst chunk p99 | 8.3 | 9.3 | 7.6 |
| tree reuse adoption | 0.9571 = ceiling | 0.9588 = ceiling | 0.9548 = ceiling |
| inherited / new simulations | 0.538 | 0.470 | 0.502 |
| gate | 5/5 | **3/5** | **5/5** |
| network evaluations/move | 4,168.7 | 4,903.4 | 4,675.5 |

**No proof or reuse regression.** Adoption sits exactly on its structural
ceiling in all three runs, and inheritance stays around half. Proofs land more
often, not less: moves returning early on a proven root went 0.155 -> 0.213.

**The search never missed its own deadline.** Every millisecond of the overrun
is caller-side work -- re-rooting and `release()` walking the discarded tree --
which is inside the move the requirement is written against and outside the
interval the search times itself over. This is the same mechanism that took
`pocket` to a 20 ms reserve, `pocket_r35` to 35 and `pocket_graph` to 50. It is
sharper here only because the throughput win is bigger: **42.3% more
simulations a move bought 84.7% more caller-side overhead.**

**About 18 of those 79 ms are the mirror, not the tree.** Scaling
`pocket_graph`'s 42.77 ms by the ratio of simulations predicts 60.8; the
measured value is 79.0. The excess is four extra Python objects per expanded
node -- a `ChildArray` and three numpy columns -- that `release()` has to drop.
That is a real cost of this design and it is stated here rather than absorbed
into a reserve.

The reserve is now **95 ms**, sized as the tool prescribes (overhead p99 78.99
+ worst-chunk p99 9.3 = 88.3), and the re-run passes 5/5 with 32 ms of margin.
It costs 4.7% of thinking time on paper and about a third of the throughput
gain in practice -- 4,903 network evaluations a move became 4,675 -- which is
more than the arithmetic predicts and is within the game-mode noise band.

**This promotes `release()` from a cleanup item to a prerequisite.** The brief
set it aside as "not a strength optimization ... latency-tail management and
reserve size", which is exactly right and is exactly why it now binds: the
reserve has gone 20 -> 35 -> 50 -> 95 ms, it is 9.5% of the budget, and it
grows with every throughput win because it is paying for a walk over a tree
that throughput makes bigger. The next optimisation on this path buys less than
it looks like it should until that walk is off the critical path.

## Strength: 0.5708 [0.5285, 0.6131] at equal wall clock, n=120

Paired openings on the held-out `select_ab` namespace (seed 7000), `pocket_sel`
against `pocket_graph`, both meeting the frozen latency requirement.

    score for A  0.5708 [0.5285, 0.6131]   W23 / D91 / L6   n=120

The interval excludes parity. Three things about it deserve stating before it
is believed.

**The point estimate is 3.5x the pre-registered expectation, and the
pre-registration was the weaker number.** #44 projected +0.01 to +0.02, derived
from a single prior data point -- the CUDA graph's +29% network evaluations
buying +0.0458 -- which itself carried a +/-0.033 interval. Two noisy estimates
disagreeing is not a contradiction, and the honest reading is that the
search-to-strength exchange rate is not pinned down well enough by one prior
match to have predicted this one. The brief's instruction not to preregister a
required win rate from that projection was correct.

**The narrow interval is real and comes from the draws.** 91 of 120 games were
drawn, so the outcome variance is low (SD 0.235) and 120 games resolve more
than they usually would. Restricted to decisive games it is 23-6, which is
p ~= 0.002 against a coin.

**In THIS match the throughput edge was only +5.1%** (3,553.6 against 3,382.3
network evaluations a move), not the +12.2% the regression gate measured
against `final`. Both engines here share a network, so tree-reuse inheritance
is inflated in the way `uttt-mirror-gate-hides-reroot-cost` describes --
5,801 and 5,097 inherited simulations a move against ~3,000 in the gate. So
this match produced MORE strength per unit of throughput than the deployment
figure would suggest, which is another reason to treat the exchange rate as
unmeasured rather than to reason forward from it.

`pocket_sel` also used less wall clock per move while doing more work: p50
917.0 ms against 957.5, mean 806.4 against 849.8, both PASS.

A 240-game confirmation is running on a fresh held-out namespace (seed 7100).
It is not a formality: this 120-game match is the run that SIZED the effect,
and a verdict read off it would be one sample used twice. Extending the same
seed would not help either -- openings are taken in order, so the first 120 of
a 240-game run on seed 7000 are the games already played.

## Parity, and why the oracle is built the way it is

**7,695,074 selections, zero disagreements.**

| | cases | mismatches |
|---|---|---|
| named fixtures, one per branch claim | 26 | 0 |
| exhaustive sweep, 2-3 children | 2,270,592 | 0 |
| tie-heavy fuzz | 400,000 | 0 |
| shadow, 12 fixed positions | 400,471 | 0 |
| shadow, 4 games vs `final` (reuse + adoption) | 4,623,985 | 0 |

**SHADOW MODE IS THE PRIMARY ORACLE AND REPLAY IS NOT.** A replay harness
reconstructs the native input from Python's authoritative node state and then
checks the scoring -- so it validates the arithmetic and nothing else. It
cannot see a mirror that has drifted, because it rebuilds the mirror from the
truth every time. Drift is the entire risk of this design: five columns updated
by hand at eight call sites, one of which is virtual loss, which applies and
undoes `W += 1.0` at magnitudes where float addition does not round-trip (a
child at W = 0.03 comes back as 0.030000000000000027). Shadow mode reads the
mirror THE ENGINE MAINTAINED, compares the chosen index, and additionally
compares all five columns against the nodes bit for bit -- per call, and once
per search over the whole tree, because a node whose mirror drifted and which
then stopped being visited is invisible to per-call sampling and "stopped being
visited" is exactly what a wrong score causes.

Shadow returns the PYTHON answer on every call, so a disagreement would not
fork the tree and make every later comparison unattributable.

**The oracle is tested for its ability to fail.** Four tests plant defects and
require detection: a native selector that always answers with the last child; a
one-ULP `W` drift that does NOT change the argmax (caught only by the column
check -- index parity would call it clean); a reordered `kids` list; a `solved`
value written to the node but not the mirror. Without these, seven million
green results would be evidence of nothing.

### Tie semantics, stated rather than discovered

`max()` keeps the FIRST maximal element and walks `node.children.values()`,
which is dict insertion order, which is `rule_utl_valid_moves` order --
mini-major, and NOT ascending board index on a send-anywhere position. The
mirror stores that order and scans forward with a strict `>`. A native side
that sorted by move, or that walked an 81-cell mask, would break ties
differently on the 11-in-400 positions whose legal moves come back unsorted;
the test generates such positions and fails if it cannot find any.

The proven-win branch is the exception: it carries an explicit `(N, -move)`
key, so there the tie-break is lowest move index regardless of order. Only the
PUCT branch depends on insertion order.

Arithmetic is reproduced left to right -- `((c_puct * prior) * sqrt(parent_N))
/ (1 + N)` and `(-q) + u` -- with `sqrt` hoisted, which is exact because
`parent_N` does not vary across the scan. The module is built `/fp:precise`.
Nothing in the expression is an add fed directly by a multiply, so there is no
FMA contraction to worry about, but the flag is stated rather than reasoned
about.

## What the boundary actually costs on this box

The design turned on one number, and the obvious way to get it was wrong.
`uttt_engine.is_valid_move(int)` measures 1.14 us, which would have made a
per-descent native call barely worth making. That call does real work. Priced
against no-ops in the module that is actually hot:

    free noop0()                      0.187 us
    free noop1(int)                   0.207 us
    bound method m0()                 0.259 us
    bound method m1(int)              0.282 us     <- the shape of best()
    obj.m1(int) including the lookup  0.303 us

Mirror write-through is 0.18 us per edge for both columns, against 0.063 us for
the two `__slots__` stores it shadows. Note that `arr[i] = v` costs 0.09 us but
`arr[i] += 1` costs 0.25 -- the mirror writes Python's post-update value rather
than incrementing, which is both cheaper and the reason it cannot drift into a
second answer.

## Two things this cost that were not on the list

**`node.kids` is a second list of strong references to the children
`release()` drops.** Leaving it would keep the parent/child cycle intact and
hand the discarded tree back to the cyclic collector -- worst chunk 3.1 ms to
25.2, p99 980 ms to 1037, which is a promotion-gate failure found a day later
on a match. Nothing functional would have noticed. `release()` now clears the
mirror slots under a `sel is not None` test, and a `__del__`-tracked test
requires a mirrored tree to die by refcount with the collector off.

**The incumbent path is not quite free.** Two costs are unconditional: one
`self.sel = None` store per node created, and one `sel is not None` load per
selection. Together about 0.1% of a move. Every other mirror site is an `if
self._mirror:` branch that duplicates its loop rather than adding a test inside
it, specifically so the arm being compared against does not get slower.

## Where the remaining mirror cost is

Measured, not guessed. Per expanded node the mirror costs 6.44 us, decomposed
on a bench:

    ChildArray constructor              2.90 us   (3 numpy allocations ~0.88)
    three column views (.N .W .S)       0.79 us
    cidx stores plus two Python lists   0.72 us

Identified and NOT implemented:

    fold the three column views into one call     ~2.5 ms/move
    drop the two pybind list->vector conversions  ~7.0 ms/move
    apply virtual loss inside the native call     ~11.0 ms/move

About 20 ms/move, roughly +2.6% network evaluations. That does not change the
routing bucket this result falls into, which is why the effect-size match was
run on the implementation as measured rather than after another optimisation
pass.

**A fourth item is worth more than all three and is not about throughput.**
The mirror costs four Python objects per expanded node, and that is what adds
18 ms to the caller-side overhead p99 and therefore 18 ms to the reserve.
Collapsing the three numpy columns into one array would take it to two objects
per node. There is also a structural simplification available: if `best()`
returned the MOVE instead of the array index, `node.kids` would not need to
exist at all -- the lookup becomes `node.children[mv]`, about 0.02 us dearer
per call, and in exchange the parallel strong-reference list disappears, one
allocation per expansion goes away, and `release()` no longer has to clear
anything to break the cycle. Both are mirror-maintenance work, both are inside
#45a's scope, and both were left alone so the numbers above describe the
implementation the match was actually run on.

## What this does NOT license

**Throughput did not predict the strength, in either direction.** It has failed
before on this code (`RESULT_MODEL_SIZE`), and here it undershot by 3.5x. The
match is the only thing that can promote, and the 240-game confirmation is the
only thing that can promote it on evidence that was not also used to size it.

**The host/device split moved and the fixed-position numbers are not the
deployment ones.** On these positions the device path is 54.3% of a move for
`pocket_graph` and 59.2% for `pocket_sel` -- host work fell from 401.1 to 358.5
ms. #44's "59.2% host" was measured in GAME mode, which is a different
workload; do not read the two against each other.

**One sample.** Fixed-position figures replicate to a few percent; game-mode
figures do not, because CUDA reductions are not bit-reproducible and one
differing move diverges the game.

## Reproduce

    cmake --build engine/cpp/build --config Release
    python -m tools.test_select_parity
    python -m tools.select_parity --mode all --positions 12 --games 4
    python -m tools.profile_selection --mode fixed \
        --arms pocket_graph,pocket_sel --positions 40 --position-games 3 \
        --tag select45
    python -m tools.regress_engine --engine pocket_sel --games 10 \
        --tag sel-candidate
    python -m tools.arena_1s --mode h2h --player-a engine:pocket_sel \
        --player-b engine:pocket_graph --games 120 --seed 7000 --tag sel-h2h
    python -m tools.arena_1s --mode h2h --player-a engine:pocket_sel \
        --player-b engine:pocket_graph --games 240 --seed 7100 \
        --tag sel-h2h-confirm
