# RESULT -- the probes are now the largest host item, and selection is not (2026-08-15)

Task #47. `RESULT_SELECTION_PROFILE.md` closed with two numbers that set the
agenda for months: `_best_child` at 193.85 ms/move and 22.2% of a move, and
terminal probes at 118.8 ms/move inclusive with "roughly 42 ms of Python loop"
recoverable. Native selection and deferred retirement have both landed since,
`pocket_defer` is the deployment baseline, and it runs 80.5% more network
evaluations in the same second. Every per-move figure in that sentence was
measured on a different engine.

The brief for #47 was explicit that re-pricing the cost is only half a decision,
and it is the half that cannot answer the question: a native implementation that
makes 35,000 terminal checks cheap is worth building only if those checks are
doing enough proving to justify keeping the eager one-ply scan at all. So this
measures benefit density alongside cost, and then ablates the whole subsystem.

`tools/profile_selection.py` (cost), `tools/probe_ablation.py` (work units,
benefit, ablation).

## The answer

**`_best_child` fell from 193.85 ms/move to 39.64, from 22.2% of a move to
4.7%. The terminal probes rose from 118.8 ms/move inclusive to 142.53, and are
now the single largest host line item at 16.7%.** The ranking that has driven
this branch since #44 has inverted, and it inverted because the thing at the top
was fixed.

**And the right move is neither of the two that were on the table.** The probes
cost 18.9% of the engine's search and 99.26% of them find nothing, but they are
not worthless: at equal simulations they change the move on 3.3% of positions,
and in 2 of 120 they play a PROVEN WIN the probe-less search does not find.
Deleting them is a strength bet that would take 700-1,200 games to settle.
Porting them to C++ is a large piece of work for a bounded win.

The third option is strictly better than both, and it needs no strength match
at all:

> **Skip the probe loop at any node where the engine's own rules say no legal
> move can end the game.** A necessary condition read off `check_ultimate_win`
> keeps 17.5% of the per-child work, with **zero false negatives over 526,097
> probe roots** -- so every proof the engine makes today, it still makes.

That recovers most of a 126.9 ms/move item at EXACT behavioural parity, and it
also removes the case for a native fused probe: after filtering, the whole probe
path is about 22 ms/move.

    #44, pocket_graph          #47, pocket_defer
      _best_child   193.85       terminal probes  142.53   16.7%
      make_move     118.14       wave loop        135.78   15.9%
      terminal pr.  118.80(*)    descent make_mv   78.59    9.2%
      wave loop      73.31       node creation     72.03    8.5%
                                 _best_child       39.64    4.7%

    (*) inclusive of the clone and make_move it causes, as here.

## The re-profile: what a move costs on the deployed engine

`tools/profile_selection.py --mode all --arms pocket_r35,pocket_defer`, both
arms on the SAME fixed positions and the same instrument, against `final` as a
cross-network opponent, tree reuse on, solve on, deferred GC. `calls/move` comes
from the counting arm and `us/call` from the timed arm, so no row is scaled by a
simulation-rate ratio.

`pocket_defer` in deployment: 852.0 ms/move of clean wall, 8,572.6 sims/move,
5,637.4 network evaluations a move.

| operation | calls/move | us/call | ms/move | share |
|---|---|---|---|---|
| `state.make_move` (both callers) | 113,270 | 1.232 | 139.52 | 16.4% |
| `wave loop` | 1,246 | 108.94 | 135.78 | 15.9% |
| `node creation` | 813 | 88.61 | 72.03 | 8.5% |
| **`terminal probes` (own loop)** | 5,637 | 9.173 | **51.71** | **6.1%** |
| **`_best_child`** | 69,074 | 0.574 | **39.64** | **4.7%** |
| `state clone` | 54,167 | 0.716 | 38.78 | 4.6% |
| `backup` | 9,971 | 2.176 | 21.70 | 2.5% |
| `legal moves` | 5,637 | 2.219 | 12.51 | 1.5% |
| proof: backward induction | 5,878 | 0.409 | 2.41 | 0.3% |
| proof: propagation | 189 | 0.475 | 0.09 | 0.0% |
| `tree reuse: adopt` | 1.2 | 6.85 | 0.01 | 0.0% |
| **`tree release`** | -- | -- | **absent** | **0.0%** |
| device: graph replay | 639 | 344.59 | 220.29 | 25.9% |
| device: network forward (k != 8) | 174 | 619.45 | 107.62 | 12.6% |
| device: eager wave (k != 8) | 174 | 261.11 | 45.35 | 5.3% |
| device: graphed wave glue | 639 | 18.94 | 12.11 | 1.4% |
| device: plane build + H2D (k != 8) | 174 | 67.07 | 11.65 | 1.4% |
| everything else | - | - | -59.21 | -6.9% |

    terminal probes, INCLUSIVE of the clone and make_move they cause  142.53 ms
    whole device path                                                 397.01 ms
    _adopt + release, outside the search's own deadline                  0.01 ms

**The residual is -6.9% and it is a real limitation, not a rounding note.** The
rows multiply one run's call counts by another run's per-call costs, and a
residual this size means the two runs disagree about how much work a move
contains by that much. Shares are good to about a point; the ORDERING and the
ratios between rows are what this table supports.

### What moved, against the superseded baseline on identical positions

`pocket_r35` on the same positions: 843.7 ms/move, 5,123.3 sims, 3,123.2 nn.

| operation | `pocket_r35` | `pocket_defer` | change |
|---|---|---|---|
| `_best_child` | 170.13 (20.2%) | **39.64 (4.7%)** | **-76.7%** |
| terminal probes, inclusive | 87.96 | **142.53** | **+62.0%** |
| terminal probes, own loop | 31.56 | 51.71 | +63.8% |
| `state.make_move`, all | 92.54 | 139.52 | +50.8% |
| `state clone` | 25.88 | 38.78 | +49.8% |
| `node creation` | 24.55 | 72.03 | +193.4% |
| `wave loop` | 78.26 | 135.78 | +73.5% |
| `backup` | 5.54 | 21.70 | +291.7% |
| `tree release` | 7.59 | absent | -100% |
| work outside the search's deadline | 7.60 | **0.01** | **-99.9%** |
| device path | 521.36 (61.8%) | 397.01 (46.6%) | -23.9% |
| everything not the device path | 445.99 | 514.17 | +15.3% |
| network evaluations a move | 3,123.2 | 5,637.4 | **+80.5%** |

Three things to read off this and one not to.

**Native selection did what it was built to do, and more than the gate
predicted.** `us/call` for `_best_child` went 4.296 -> 0.574, a factor of 7.5,
and the row went from the largest host term to fifth. This is the first
measurement of that change on the deployed engine rather than on a candidate.

**Deferred retirement is visible by ABSENCE.** `tree release` does not appear in
the `pocket_defer` table at all -- not because it became free, but because it no
longer runs inside the gated move. The one number that survives is the roll-up:
7.60 ms/move of work outside the search's own deadline becomes 0.01. That is
what took the reserve from 95 ms to 20.

**Every count term grew, because the engine does 80.5% more search.** `backup`
+292% and `node creation` +193% are not regressions; they are the same work per
unit, more units. `us/call` for `state.make_move` actually FELL, 1.431 ->
1.232.

**What not to read off it:** the device path did not get faster. Its share fell
from 61.8% to 46.6% because the host got faster and because the graph replaced
per-kernel dispatch, and `pocket_r35` here runs the EAGER wave (graph off) --
it is the superseded baseline, not a controlled device comparison.

## Probe cost, priced properly

    5,637 probe roots a move
    x 8.66 legal children each
    = 48,834 children probed a move

Each probed child costs one `clone`, one `make_move`, and -- measured, not read
off the source -- **four transitions into C++**:

| # | crossing | why |
|---|---|---|
| 1 | `GameState(self)` | the copy constructor behind `clone()` |
| 2 | `_CppGameState.make_move` | the move itself |
| 3 | `self._raw_winner()` | inside `make_move`, to build its `(ok, winner)` return tuple |
| 4 | `self._raw_winner()` | the loop's own `if probe.winner is not None` |
| 5 | `self._raw_winner()` | ONLY on a hit, inside `_terminal_value` |

**Crossing 3 is pure waste.** `GameState.make_move` computes `(True,
self.winner)`; `_mark_terminal_children` discards the return value and then
immediately reads `probe.winner` again. That is one redundant boundary crossing
per probed child, about 48,800 a move, available today with no native work at
all. `tools/test_probe_ablation.py` asserts the count is exactly `4 x children +
hits` so that if the redundancy is ever removed, the test fails and this
paragraph gets corrected rather than quietly going stale.

## Probe work units, on 120 fixed positions

    4,113.5 probe roots a move
    x   8.19 legal children each
    =  33,699 children probed a move, 135,196 pybind crossings

| | |
|---|---|
| per probed child, inclusive | **3.766 us** |
| per probed child, the Python loop alone | 1.471 us |
| pybind crossings per probed child | **4.01** |
| probes per move, inclusive | **126.9 ms** |
| of which the Python loop | 49.6 ms |
| instrument subtracted from those figures | 54.5 ms |
| raw, before pricing (do not quote) | 162.0 ms |

**The instrument correction is 34% of the raw reading and it is not optional.**
Wrapping `clone` and `make_move` puts ~57,000 wrappers a move INSIDE the
interval the probe is timed over. The first run of this tool got the unit wrong
and reported **-1.905 us per probed child**; `price()` now refuses to return a
correction larger than the measurement it corrects, which is the check that
would have caught it before it printed.

## Benefit density: 99.26% of probes find nothing

| | |
|---|---|
| probed children that ARE terminal | **0.0074** |
| probe roots finding any terminal child | 0.0491 |
| expansions that produce a node proof | 0.0401 |
| searches whose ROOT is proven | 0.0667 (8 of 120) |
| proof propagations per move | 164.9 |
| levels climbed per propagation | **0.23** |

The last row is the one that surprised me: three quarters of proof
propagations climb zero levels. `_propagate_solved` is called, checks its
parent, and returns.

**All 8 proven roots were proven DRAWS, and the probe-less engine chose the same
move in all 8.** Under a clock, on these positions, not one root proof changed a
decision.

## The ablation: probes cost 18.9% of the search

`pocket_defer` against `pocket_defer+solve=0`, same 120 fixed positions, same
clock. `solve=0` removes the whole subsystem -- the one-ply probes, the marking
of terminals met during descent, propagation, the proven-win/refuted-loss short
circuit (`ChildArray` is constructed with the flag, so the native selector drops
it too), the early return, and the one-hot root correction.

| | ON | OFF | OFF vs ON |
|---|---|---|---|
| **nn/second x deadline** | 4,750.5 | 5,650.3 | **+18.9%** |
| nn per second | 4,847.5 | 5,765.6 | +18.9% |
| nn per move | 4,465.5 | 5,646.3 | +26.4% |
| simulations per move | 8,104.5 | 14,298.2 | +76.4% |
| children probed per move | 36,630.7 | 0 | -100% |
| search p99 ms | 980.0 | 980.4 | +0.0% |
| proven roots | 8 | 0 | -100% |
| early returns | 7 | 0 | -100% |

**Two independent instruments agree.** The profiler says the probes are 126.9 ms
of a 921.2 ms search, 13.8%, which predicts +16.0% more work if removed. The
ablation measures +18.9%. The gap is the rest of the solve subsystem, which the
profiler charges to other rows.

**Do not quote the simulations row.** +76.4% is the early-stop composition
effect `uttt-warmup-in-numerator-not-denominator` warns about: a descent into a
proven subtree is a simulation that costs almost nothing. Network evaluations
per second is the unit.

## Move disagreement, and the confound in it

| | | |
|---|---|---|
| ON vs ON, equal clock (the noise floor) | 0 / 120 | **0.0000** |
| ON vs OFF, equal clock | 12 / 120 | 0.1000 |
| ON vs OFF, **equal simulations** | 4 / 120 | **0.0333** |

**The floor is a real measurement, not a tautology.** The two ON runs are
wall-clock bound and produced different simulation counts on 104 of 120
positions -- mean 68.7 apart, up to 552, 0.91% relative -- and still chose the
same move 120 times out of 120. The argmax is robust to run-to-run wobble, so
the whole of the ON/OFF difference is attributable to the feature.

**The equal-CLOCK row is confounded and the equal-SIMS row is the answer.** The
OFF arm does not merely lack probes, it also gets 18.9% more search, so a
changed move might be either cause. Removing the clock (`ms=0`, 800 simulations,
identical on 120/120 positions) isolates the feature: **3.3%**. So roughly two
thirds of the equal-clock disagreement was the extra search, not the probes.

800 simulations is about a tenth of the tree the engine builds in a second,
which FAVOURS the probes -- less search means more room for a one-ply proof to
matter -- so 3.3% is an upper bound on the deployment figure.

### The 3.3% is not noise, and two of the four matter

    late   ON mv  7  root_solved None  | OFF mv 24
    late   ON mv 17  root_solved None  | OFF mv 24
    mid    ON mv 10  root_solved 1     | OFF mv 20
    late   ON mv 23  root_solved 1     | OFF mv 13

**In two of 120 positions the probes proved a WIN and the probe-less search
played something else.** A disagreement generally has no sign, but this kind
does: a proven win is proven. Of 11 proven roots in the fixed-simulation ON arm,
the OFF arm matched the move in 9 -- so 82% of proofs were redundant and 18%
were not.

That is the whole case against deleting the probes, and it is a small case, but
it is not zero and it is on the side that cannot be recovered by more search.

## What decides it: a necessary condition with zero false negatives

`GameState.make_move` only calls `check_ultimate_win` when a move newly DECIDES
a mini-board, and that function returns a result in exactly two cases. So a
terminal child requires:

  * **(a)** the mover already owns two mini-boards of some macro triple whose
    third is still undecided -- otherwise no line can complete; or
  * **(b)** only one mini-board is still undecided -- otherwise deciding one
    leaves another undecided and the board cannot be full.

If neither holds, **no legal move from that position ends the game** and the
entire probe loop is dead work. `could_end()` in `tools/probe_ablation.py` is
that test, and it costs one pass over nine mini-board states plus at most eight
triple comparisons -- per NODE, against 8.19 children each costing 3.766 us.

Measured against the real probe on real searches (`--mode filter`):

| | | |
|---|---|---|
| probe roots seen | 526,097 | |
| roots the filter keeps | 99,334 | 0.1888 |
| **children the filter probes** | 742,063 | **0.1752** |
| roots where the probe found something | 26,340 | 0.0501 |
| **FALSE NEGATIVES** | **0** | must be zero |

**Zero false negatives over half a million probe roots.** Not "few" -- zero, as
the derivation requires. The condition is necessary, so a filtered probe writes
the same `solved` bits as an unfiltered one on every node, and there is nothing
for a strength match to resolve.

It is not efficient in the other direction -- it keeps 18.9% of roots and only
5.0% find anything, a precision of 26.5% -- but precision is not the gate here.
Recall is, and recall is 1.

## What follows

**Build the selective probe. Do not port the probe loop to C++, and do not
delete it.**

* **Selective.** 82.5% of the per-child work goes away at exact parity. The
  probe path falls from 126.9 ms/move to about 22, plus roughly 4-9 ms for the
  filter itself. Scaling the ablation's calibration by the share removed
  predicts **+11% to +13% network evaluations a second**, to be confirmed by
  the throughput gate rather than assumed here.
* **NOT a native fused probe.** After filtering, the whole probe path is ~22
  ms/move -- about 2.4% of search. A `for each legal child: clone, make_move,
  classify, write` primitive in C++ is a large change with a real parity
  surface, and it would be competing for that 2.4%. The filter takes the same
  work off the table first, and it is a dozen lines of Python.
* **NOT deletion.** +18.9% is the largest number here, but it buys giving up a
  proven win in ~1.7% of positions, and settling whether that trade is
  favourable needs 700-1,200 paired games by the ladder. The filter makes the
  question moot.
* **One free crossing, separately.** `GameState.make_move` builds a
  `(True, self.winner)` tuple that `_mark_terminal_children` discards before
  re-reading `probe.winner`. That is one of the four crossings per probed child,
  removable with no design work. Worth folding into the same change.

### Ordering, unchanged from the graph and the release work

    exact parity vs the current probe, on random reachable states and
        exhaustive late-game states
      -> fixed-position throughput
      -> deployment latency gate
      -> ladder-derived pre-registered band, IF a strength match is still wanted
      -> held-out equal-clock confirmation

The second line of that list is where this one should stop. A change with a
proven-identical output does not need an equal-clock strength match: what it
buys is throughput, and throughput is what the gate measures. The ladder
prediction is worth recording anyway, because a promoted engine that does not
move as predicted is a signal that something else changed.

## What this does NOT license

**The 3.3% is measured at 800 simulations, not at the deployment budget.** It
should be an upper bound at 8,000, and the direction of the bias is argued from
first principles rather than measured. If the selective probe ever gets compared
against deletion for real, this is the number that has to be re-measured at the
deployment tree size.

**The benefit numbers are from 120 positions from 6 games.** Eight proven roots
and two decisive proofs are small counts; the ratios around them (0.0667,
0.0167) carry standard errors of about 0.023 and 0.012. What they support is
"proofs are rare and mostly redundant", not a precise rate.

**Nothing here says the probes are worthless at other budgets.** At the teacher's
800-simulation distillation budget the same subsystem is doing target
construction, not deployment search, and
`uttt-solving-fixes-targets-not-students` already measured that separately.

## Reproduce

    python -m tools.profile_selection --mode all --arms pocket_r35,pocket_defer \
        --tag defer_profile
    python -m tools.probe_ablation --mode all --positions 120 \
        --position-games 6 --tag probe47
    python -m tools.probe_ablation --mode filter --positions 120 \
        --position-games 6 --tag probe47_filter
    python -m tools.probe_ablation --mode fixedsims --positions 120 \
        --position-games 6 --tag probe47_fs
