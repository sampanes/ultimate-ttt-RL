# #49 -- what the residual probe path is made of

`pocket_filter` shipped and the probe loop stopped being the largest host item.
This is the re-profile of what it left behind, the price of the one redundancy
still in it, and the first-class measurement of allocation and object churn.

Tools: `tools/probe_cost.py` (new), `tools/test_probe_cost.py` (40 tests).
Data: `results/probe_cost/probe49.json`. Engine: `engine:pocket_filter`, the
deployed baseline, git `08f9d9e`.

---

## The headline is a correction, not a saving

**The wrapper-priced probe cost that #47 and #48 published is 25% too high.**
Measured directly, not inferred: run the real `_mark_terminal_children` over a
fixed corpus with no instrument, run it again with exactly the wrappers
`tools/probe_ablation` installs, then hand the wrapped result to that module's
own `price()` and ask whether it recovers the bare number.

| | us/child |
|---|---|
| bare, no instrument | 3.070 |
| raw, wrapped | 5.955 |
| wrapper subtracted by `price()` | 2.119 |
| **priced result** | **3.836** |
| **priced / bare** | **1.250** |

`AttributedTimer.calibrate` prices a wrapper at 0.982 us/call. The price that
would make the correction land on the bare number is **1.338 us/call**, 36%
higher. The calibration times a wrapper around a trivial Python no-op in a hot
loop; the wrappers that matter here sit on `GameState.clone` and
`GameState.make_move` -- pybind bound methods reached through a Python subclass
-- about 13,000 times a move. They cost more there than around a no-op, so
`price()` under-subtracts and every `probe_ms_per_move` figure this repo has
published is inflated by the difference.

Corrected figures:

| published | corrected |
|---|---|
| #47 probes 126.9 ms/move (13.8% of a move) | ~101.5 ms (11.0%) |
| #48 legacy probes 154.6 ms/move | ~123.7 ms |
| #48 selective probes 46.6 ms/move | ~37.2 ms |

**No decision changes.** Every promotion in this series was gated on
*uninstrumented* arms -- the #48c throughput gate, the latency gate, and the
fixed-simulation identity proof all ran clean engines. What moves is three
descriptive cost figures and the shares derived from them.

### And the correction makes the instrument look worse, not better

Re-deriving #48c's prediction from the corrected probe share: 0.134 of the
search, 85.2% of the per-child work removed, so `1/(1-0.114) - 1` = **+12.9%**
predicted against **+20.6%** measured. The ratio goes from 1.24 to **1.60**.

So the instrument **overstates the time and understates the value**, and both
at once. That is only coherent if the probe loop costs the search more than the
interval it occupies -- which is the standing hypothesis. This run tried to
reproduce that mechanism in isolation and could not (below).

---

## The split

In-situ envelope and call volume measured on this run's own positions, not
imported. Three runs of the same measurement gave probe totals of 39.50, 40.73
and 52.24 ms/move; the position sample moves it +/-14%, so read the *shares*,
not the third decimal place.

| | ms/move | share |
|---|---|---|
| predicate (`could_end`, every root) | 7.10 | 17.4% |
| clone (alloc + copy ctor) | 4.54 | 11.1% |
| `make_move` native execution | 3.47 | 8.5% |
| tuple construction + redundant crossing | 2.47 | 6.1% |
| pybind call overhead (2 crossings/child) | 2.46 | 6.0% |
| solved-status writes / propagation | 0.56 | 1.4% |
| `probe.winner` readback | 0.40 | 1.0% |
| Python child iteration | 0.23 | 0.6% |
| skipped-root method entry (4,557 roots) | 0.14 | 0.3% |
| **ladder subtotal** | **21.37** | |
| residual (see below) | 19.36 | 47.5% |
| **in-situ, wrapper-priced** | **40.73** | |

Steps below ~0.05 us/child are at the noise floor. One run returned
`solved-status writes` at -0.23 ms; read that row as "about zero", not as a
negative cost.

### The ladder is validated, not asserted

Each rung is the rung below it plus exactly one operation, so a step is a
difference and needs no wrapper. The top rung is checked against the real bound
method timed the same way:

| corpus | replica + predicate | real method | delta |
|---|---|---|---|
| admitted roots (filter on) | 3.047 | 3.215 | **-5.2%** |
| all roots (filter off) | 2.934 | 2.946 | **-0.4%** |

Two earlier versions of this comparison were wrong in opposite directions and
both looked like instrument failures. Omitting the predicate when production
runs it read as the replica being 13.5% too cheap; adding it when production
does *not* run it read as 7.7% too expensive; and timing the filtered engine
against a corpus half of which it skips read as 76%. All three are now handled
explicitly and `tools/test_probe_cost.py` fails if any regresses.

### The residual is bounded but not explained

Of the 19.36 ms, the instrument's own 25% overstatement accounts for ~8.1 ms.
Another ~1.6 us/child is definitional: `us_per_probed_child` divides
skipped-root work by probed children, which the ladder's admitted-only corpus
does not carry. After both, roughly **10 ms -- a third of the path -- is cost a
faithful replay of the same code cannot reproduce.**

Two candidate mechanisms were tested directly and neither survives:

| hypothesis | test | result |
|---|---|---|
| live-heap / allocator pressure | pin a real 69,639-node tree, rerun the identical loop | **+1.9%** |
| first-touch locality | working set 261 kB -> 31.8 MB (118x) | **+4.0%** |

The loop costs ~2.8-2.9 us/child under every condition constructible in
isolation. Production's wrapper-priced figure is 8.34. **Do not quote a summed
wrapper total as an optimisation ceiling on this path**, and do not attribute
the gap to the allocator -- that was the obvious answer and it is measured at
roughly two percent.

---

## #49b: the redundant winner crossing

`make_move()` builds `(True, self.winner)`, the probe discards the tuple, then
reads `probe.winner` again. Priced two ways from independent data:

| | us/child |
|---|---|
| measured (rung 4 minus rung 3) | 0.5062 |
| structural (primitives summed) | 0.5817 |
| agreement | 0.87 |

The structural model is the shipped body minus the candidate body, spelled out:
a Python method frame (30.1) + the native `make_move` crossing (303.8) + two
`GameState.winner` reads (381.0 each) + a tuple (34.1), against the candidate's
crossing (303.8) + `_raw_winner` (244.4). An earlier version priced the two
winner reads as bare property gets (57.6 ns) and landed 54% under the
measurement -- `GameState.winner` is not a bare descriptor, it runs
`_raw_winner()` and translates the sentinel.

**At 4,884 probed children a move: 2.47 ms/move. Band: probably archive unless
trivial. ARCHIVED.**

Also archived, both measured rather than assumed:

- **a fused native `probe_make_move`: +1.48 ms/move** on top of the candidate
  path -- below the 2 ms floor. This is the third independent measurement
  saying not to build it.
- **inlining `_terminal_value`: -0.44 ms/move**, i.e. nothing. It only fires on
  a hit, and hits are ~3-5% of probed children.

### The probe was the wrong call site

The descent does the same thing, character for character:

```
state.make_move(node.move)      # builds (True, self.winner)
...                             # Python discards the tuple
if state.winner is not None:    # and reads the winner again
```

| | per move |
|---|---|
| `make_move` from probes | 4,884 |
| `make_move` in the descent | **53,048** |
| ratio | **10.9 : 1** |
| winner reads per descent `make_move` | **2.00** |

That last row is the structural check: it must be ~2.0 or the second read is
not happening and the estimate is void. It is 2.00.

The same 0.506 us, at the descent's call count, is **26.85 ms/move**; both
sites together are **29.33 ms/move, 3.29% of the search** -- the "real
optimisation candidate" band. This figure comes from the bare ladder, not from
the wrapper, so the 25% correction above does not apply to it.

Nothing has been implemented. This is a candidate for the re-ranking, not a
widening of #49b, and it is exactly the term the re-ranking brief asked about
(`state.make_move during normal traversal`).

---

## #49c: allocation and object churn, measured

Per move, on the deployed engine:

| | |
|---|---|
| GameState clones created | 12,605 |
| MCTSNode objects created | 40,610 |
| expansions | 5,378 |
| children probed | 5,045 |
| peak live pymalloc blocks | 164,984 |
| blocks still held after release | **290** |
| live blocks per node created | 4.06 |

`sys.getallocatedblocks()` is one C call, so it can be read on the move path
without perturbing it. It is a NET measure -- live blocks, never blocks
created -- so gross creation is counted directly for the two types that
dominate. A total Python allocation count would mean `tracemalloc` and a ~10x
slowdown, which would measure a search that does not run.

Two readings worth keeping:

- **Retirement is clean.** 290 blocks held after `release()` against 164,984
  live at peak. Deferred retirement is not leaking.
- **Cloning alone is ~10.6 ms/move** (12,605 x 845 ns), which is more than the
  entire redundant-crossing question and about a quarter of the whole probe
  path. The clone is 724.6 ns of pybind copy constructor plus ~120 ns of Python
  method and `hasattr` around it.

But allocation *pressure* is not the lever the #47/#48 discrepancy suggested.
Pinning 70,000 live nodes moves the loop 1.9%. Whatever costs the search more
than the probe interval, it is not the size of the live heap.

---

## #49d: the re-ranking

`tools/profile_selection --mode all --arms pocket_filter`, 40 fixed positions
plus 6 games against `final`. Calls/move from the counting arm, us/call from the
timed arm, so no row is scaled by a simulation-rate ratio.

| operation | fixed, ms/move | share | game, ms/move | share |
|---|---:|---:|---:|---:|
| device: graph replay | 412.75 | 45.5% | 263.53 | 32.3% |
| wave loop (own Python) | 90.36 | 10.0% | 103.73 | 12.7% |
| device: network forward | 76.83 | 8.5% | 94.91 | 11.6% |
| **node creation** | **76.69** | **8.5%** | **70.66** | **8.7%** |
| **state.make_move** | **72.56** | **8.0%** | **92.87** | **11.4%** |
| device: eager wave | 52.53 | 5.8% | 52.51 | 6.4% |
| _best_child | 32.19 | 3.5% | 38.59 | 4.7% |
| terminal probes (own loop) | 23.46 | 2.6% | 22.84 | 2.8% |
| device: graphed wave | 14.32 | 1.6% | 13.26 | 1.6% |
| backup | 14.01 | 1.5% | 16.32 | 2.0% |
| device: plane build + H2D | 13.90 | 1.5% | 15.25 | 1.9% |
| legal moves | 13.37 | 1.5% | 13.27 | 1.6% |
| state clone | 11.50 | 1.3% | 15.22 | 1.9% |
| proofs (induction + propagation) | 0.62 | 0.1% | 0.98 | 0.1% |
| **wall** | **906.67** | | **816.18** | |

Answering the six terms the brief asked about, in order:

1. **graph/device path -- yes, but read it correctly.** 570.3 ms/move fixed,
   439.5 game. These are host-observed intervals *inside* device-facing calls,
   not a GPU-compute budget: CUPTI previously put actual device busy at 7.8% of
   a move (see `uttt-wave-is-dispatch-bound`). Most of `graph replay` is the
   host in a launch-and-sync, and #48 proved the host is not merely waiting --
   removing probe work bought +20.6% more search, which a GPU-bound engine
   could not have given back.
2. **`_best_child` -- no longer major.** 32-39 ms, 3.5-4.7%. The dedicated
   solo-wrapped run puts it at 35.89 ms and prices the ceiling for a *free*
   selection primitive at **+4.0% network evaluations**. That closes the
   remaining native-selection headroom question.
3. **remaining probe path -- confirmed out of the top tier.** 23.46 ms own loop,
   35.83 inclusive of the clone and make_move it causes, 2.6-3.9%.
4. **`state.make_move` during normal traversal -- YES, and it is the big one.**
   72.56 ms fixed / 92.87 game, of which the descent is **65.05 / 85.84** and
   the probes only 7.51 / 7.02. This is where the #49b redundancy actually
   lives.
5. **node creation / object churn -- YES.** 76.69 / 70.66 ms at 94.6 us per
   `_expand_children` call, roughly **1.9 us per MCTSNode** across 40,610 nodes
   a move, mirror construction included.
6. **wave-loop Python -- YES.** 90.36 / 103.73 ms *exclusive* of the device
   calls, `_best_child` and `make_move` it makes. This is virtual-loss
   bookkeeping, path lists, dedup and pending lists, and nothing has ever
   targeted it.

### Two cross-checks worth keeping

**The probe path now agrees across two independent tools.** `probe_ablation`
says 40.73 ms inclusive; corrected for the 25% over-price that is ~32.6.
`profile_selection`, which prices its wrappers in situ and applies a deflation,
says **35.83** inclusive. Within 10% of each other, from different
instruments.

**And the wrapper under-price replicates on a different code path.** This
profile measures one `_best_child` wrapper in situ at **1.579 us/call** against
the tight-loop calibration -- a **1.6x** under-price, matching the 1.36x found
on the probe wrappers. `profile_selection` already knew this and corrects for
it; `probe_ablation.price()` does not, which is precisely why the probe figures
were the inflated ones.

## What follows

The probe path is finished as an optimisation target. It is 23.5 ms of its own
loop in a 907 ms move, and the two things still inside it are both small:

- **`could_end`, 7.10 ms/move**, now the largest single item in the path it
  created. 1,814 ns per root, of which only ~660 ns is crossings
  (`mini_winners` 364, `player` 294) -- **~1,150 ns is its own interpreted
  body**. A native predicate would be one crossing (~250 ns), and
  `agents/test_probe_filter.py` already enumerates all 391,550 macro
  configurations, so parity would be exhaustive rather than statistical. There
  is also a free plumbing saving: `_expand_wave` materialises `s.mini_winners`
  for `rule_utl_valid_moves` and throws it away three lines before `could_end`
  materialises it again.
- **the probe's share of the redundant crossing, 2.47 ms/move** -- archived.

**The ranked candidates are elsewhere, and the re-profile says where.**

| candidate | ms/move | basis |
|---|---:|---|
| descent `make_move` redundancy | **26.9** | bare ladder x counted calls |
| wave-loop Python | 90.4 | profiled, nothing targeted yet |
| node creation (~1.9 us/node) | 76.7 | profiled |
| native `could_end` | ~9 | primitive floor x roots |
| free `_best_child` primitive | 35.9 | ceiling, +4.0% nn-evals |

The descent redundancy is the one with a measured price, a known shape, an
exhaustive parity story, and a change small enough to state in a sentence: a
hot-path-only `probe_make_move(mv) -> winner_code` on `GameState`, used by both
call sites, with `make_move` untouched.

**But the resolution problem is now the binding constraint.** The
composition-robust throughput estimator replicates to 1.7-6.1% run to run. 2.5
ms is +0.28%; 26.9 ms is +3.3%; even deleting the entire probe path would be
+4.6%. The decision rule's verification step -- an uninstrumented fixed-position
A/B -- cannot separate most of these from noise without a repeat count nobody
has budgeted. Anything further has to be justified the way `pocket_filter` was:
**proven identity plus a measured rate**, never a strength match, and never a
single A/B on an effect smaller than its own spread.
