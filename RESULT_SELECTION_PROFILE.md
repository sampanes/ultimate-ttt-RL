# RESULT -- after the graph, the host is 59.2% of a move and selection is its largest term (2026-08-10)

> ## SUPERSEDED 2026-08-27 -- see `CURRENT_STATE.md` section 4
>
> **The headline is obsolete: `_best_child` is no longer the largest host term
> and is no longer worth attacking.** Native selection (`RESULT_NATIVE_SELECT.md`)
> made the call 5.5x cheaper, and the selective terminal probe removed the item
> that replaced it at the top. Re-measured on `pocket_filter`, `_best_child`
> costs **32.2 ms/move, 3.5% of the move** -- against the 193.9 ms and 22.2%
> below.
>
> The question this document opened is also closed. A dedicated run prices a
> **free** selection primitive -- infinitely fast, zero cost -- at **+4.0%
> network evaluations**. There is no remaining headroom here.
>
> Two methodological points below still stand and generalize: the growth was
> **more calls, not slower ones** (a distinction worth checking before
> optimizing anything), and the `device:` rows are host-observed launch/sync
> intervals rather than a GPU budget.

Task #44. `RESULT_TREE_PROFILE.md` priced `_best_child` at 108 ms/move on
`pocket_r35`, before the CUDA graph. `pocket_graph` runs ~30% more network
evaluations in the same second, so that number had to be re-measured on the
engine the native-selection decision would actually be made about.

`tools/profile_selection.py`, `RESULT_GRAPH_WAVE.md` for the engine.

## The answer

**`_best_child` costs 193.9 ms/move in deployment, 22.2% of the move, and it is
the largest host term by 64%.** It grew 44.1% when the graph landed, and the
growth is entirely MORE CALLS -- each call got 3% cheaper, not slower.

| | sims/move | descents/sim | us/descent | ms/move |
|---|---|---|---|---|
| `pocket_r35`, fixed positions | 7,244 | 3.586 | 4.192 | 108.9 |
| `pocket_graph`, fixed positions | 8,172 | 4.114 | 3.889 | **130.7** |
| `pocket_r35`, deployment | 6,034 | 5.329 | 4.184 | 134.5 |
| `pocket_graph`, deployment | 7,345 | 6.509 | 4.055 | **193.9** |

Deployment, `pocket_r35` -> `pocket_graph`: **+21.7% simulations, +22.1%
descents per simulation, -3.1% per descent**. Both of the growth terms are
count terms. A native primitive attacks the third one, which did not move.

## The whole move, `pocket_graph` at 1,000 ms in deployment

Against `final`, tree reuse on, solve on, 50 ms reserve, deferred GC. 872.1
ms/move of clean wall.

| operation | calls/move | us/call | ms/move | share |
|---|---|---|---|---|
| **`_best_child`** | 47,805 | 4.055 | **193.85** | **22.2%** |
| `state.make_move` | 81,994 | 1.441 | 118.14 | 13.5% |
| `wave loop` | 762 | 96.17 | 73.31 | 8.4% |
| `terminal probes` (own loop) | 4,427 | 9.554 | 42.30 | 4.8% |
| `state clone` | 40,287 | 0.806 | 32.46 | 3.7% |
| `node creation` | 627 | 48.39 | 30.33 | 3.5% |
| `legal moves` | 4,427 | 2.463 | 10.90 | 1.3% |
| `tree release` | 1.0 | 8,444.8 | 8.46 | 1.0% |
| `backup` | 6,098 | 0.781 | 4.76 | 0.5% |
| proof: backward induction | 4,666 | 0.388 | 1.81 | 0.2% |
| `tree reuse: adopt` | 1.0 | 7.79 | 0.01 | 0.0% |
| proof: propagation | 193 | 0.360 | 0.07 | 0.0% |
| device: graph replay | 512 | 875.96 | 200.87 | 23.0% |
| device: network forward (k != 8) | 115 | 746.59 | 86.06 | 9.9% |
| device: eager wave (k != 8) | 115 | 304.47 | 35.08 | 4.0% |
| device: graphed wave glue | 512 | 21.40 | 10.94 | 1.3% |
| device: plane build + H2D (k != 8) | 115 | 87.29 | 10.06 | 1.2% |
| everything else | - | - | 12.68 | 1.5% |

    host (everything not the device path)   516.40 ms   59.2%
    device path                             343.01 ms   39.3%
    everything else                          12.68 ms    1.5%

**The engine is now host-bound.** Before the graph it was the other way round:
`RESULT_TREE_PROFILE` had the device path at 35% and the tree at 23%. Removing
~950 us/wave of dispatch did not make the remaining work smaller, it made it the
majority -- which `RESULT_GRAPH_WAVE` predicted in one line and this measures.

## Two things the task brief had wrong

**Terminal probes are not a ~15 ms afterthought. They cost 118.8 ms/move**
inclusive of the clone and `make_move` they drive: 42.3 ms of Python loop plus
35,618 `make_move` crossings and 26,628 clones, every move, for the 1-ply scan
at each expansion. `make_move` costs 1.441 us at the pybind boundary against
maybe 0.1-0.2 us of C++ inside it, so most of that is crossing overhead and most
of it is recoverable by a native probe. This does not change the ORDER --
selection is still larger and still first -- but "probes are only worth 15 ms"
is off by roughly 5x and should not be the reason level three is skipped.

**`_adopt` is free; `release` is the whole re-rooting cost.** Measured
separately for the first time: `_adopt` 7.8 us/move, `release` **8,444.8 us**,
one call each. The 50 ms reserve is sized against an 8.5 ms tree walk plus the
worst chunk, not against anything `_adopt` does.

## What #45 can actually expect

The in-situ arm measures a SLOPE rather than inferring a ceiling. One run per
engine with only `_best_child` wrapped adds a known cost to exactly the code
path a native primitive would replace, and the deadline-bound search gives back
a known amount of work:

    pocket_graph   +1.607 us per descent cost 50.55 ms/move and 11.1% of sims
    pocket_r35     +1.304 us per descent cost 32.24 ms/move and 11.8% of sims

Inverting, on fixed positions:

| | ms/move of selection | if selection were FREE |
|---|---|---|
| `pocket_r35` | 110.4 | +12.1% nn-evals (+40.6% sims) |
| `pocket_graph` | 140.0 | **+15.5% nn-evals (+30.8% sims)** |

**The two units disagree by 2x and neither is wrong.** A faster descent buys
proportionally more descents into already-proven subtrees, which are nearly free
and still counted as simulations. Network evaluations is the conservative
reading and the one this project settled on
(`uttt-warmup-in-numerator-not-denominator`); simulations is the optimistic one.

For scale: **the CUDA graph's +29% nn-evals bought +0.0458 win rate.** A
PERFECT selection port at +15.5% is about half that lever, and no port is
perfect -- a native primitive still reads the mirrored arrays and crosses the
boundary once per descent. So the pre-registered expectation for #45 is a
strength effect around +0.01 to +0.02, which 240 games cannot resolve (the
graph's own 240-game interval was +/-0.033). Size the confirmation from the
throughput result, not before it.

## How the numbers were made, and why it took four arms

`ms/move = calls/move x us/call`, and the two factors want opposite instruments.

    clean      no instrumentation. The wall every share is taken of.
    timed      all 16 targets wrapped. Supplies us/call. Loses 13.7-23.5% of
               the search, so its OWN call counts are not used.
    counting   the same targets, an increment and no clock. Loses 2.9-5.8%.
               Supplies calls/move.
    in-situ    only `_best_child`. Prices that wrapper where it is used, and
               gives the slope above.

Every row is `counting calls x timed us/call`. Nothing is scaled by a
simulation-rate ratio. **The tables reconcile to the clean arm's wall time
within 0.7% and 1.5%** -- and that is a real check, not an identity, because the
two factors come from different runs.

### Three instrument errors, each caught by a consistency check

**The wrapper price was charged twice.** `_best_child` came out at **-1.113
us/call**. A wrapper's clock starts after its frame is built and stops before
its accumulators are written, so most of its cost is never inside the interval
it measures and cannot be subtracted from that row. `calibrate()` now returns
both halves -- ~0.34 us inside, charged to the row; ~0.62 us outside, charged to
the CALLER once per nested call -- and the unattributable remainder comes off as
a proportional deflation.

**The work unit was simulations.** Corrected to network evaluations, per
`uttt-warmup-in-numerator-not-denominator`. It matters here: on identical
positions the two engines differ by **+12.8% in simulations and +33.5% in
network evaluations**, and the instrument's own cost reads 13.9% by one unit and
29.8% by the other. Both are printed and a disagreement is flagged.

**The counting arm produced no rows at all.** `summarize` iterated the timer's
`total` dict, which a counting timer never writes, so its `ops` came back empty
and the code silently fell back to the timed arm's call counts -- reinstating
the exact extrapolation the counting arm exists to remove. Found by the residual
check, which read 22.3% of the move instead of 6.8%.

## What this does NOT license

**The deployment table is one sample.** Two independent full runs of this tool
agree to 0.3-5.4% on fixed positions but only to **16-21% on simulations per
move in game mode**, because CUDA reductions are not bit-reproducible run to
run, one differing move diverges the game, and the two runs then measure
different positions. Fixed-position numbers are reproducible; game numbers are
not, and no game-mode figure here should be quoted to three digits.

**No strength claim.** This is a cost profile. The only thing it licenses is
choosing what to build next.

**The slope is one arm of one run**, not an interval, and it was measured on
fixed positions. Deployment selection costs 193.9 ms against the 140.0 the
slope was applied to, so the deployment ceiling is plausibly larger -- but that
was not measured and is not claimed.

**The eager path has not gone away.** 115 of 627 waves a move are not k=8 even
in deployment, and they cost 131 ms of the 343 ms device path. The graph covers
82% of waves here, against the 90.7% measured from a bare root in #40.

## Reproduce

    python -m tools.test_profile_selection
    python -m tools.profile_selection --mode all --positions 40 \
        --position-games 3 --games 6 --tag select
    python -m tools.profile_selection --rerender results/profile_selection/select.json
