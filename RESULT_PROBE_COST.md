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

## What follows

1. **`could_end` is now the largest single item in the path it created**, at
   7.10 ms/move (17.4%). It costs 1,814 ns per root, of which only ~660 ns is
   crossings (`mini_winners` 364, `player` 294) -- **~1,150 ns is its own
   interpreted body**. A native predicate would be one crossing (~250 ns), and
   `agents/test_probe_filter.py` already enumerates all 391,550 macro
   configurations, so parity would be exhaustive rather than statistical.
   There is also a free plumbing saving: `_expand_wave` materialises
   `s.mini_winners` for `rule_utl_valid_moves` and throws it away three lines
   before `could_end` materialises it again.
2. **The descent crossing (26.85 ms/move)** as above.
3. Neither is worth acting on before the full host/device re-ranking, because
   the whole probe path is now 40.7 ms of an ~890 ms move -- **4.6%** -- and
   deleting all of it would be at the edge of what an uninstrumented A/B can
   resolve at all.

**The resolution problem is the real constraint now.** The composition-robust
throughput estimator replicates to 1.7-6.1% run to run. A 2.5 ms saving is
+0.28%; even 29 ms is +3.3%. The decision rule's own verification step -- an
uninstrumented fixed-position A/B -- cannot see effects this small without a
number of repeats nobody has budgeted. Any further probe-path work has to be
justified by identity plus a *predicted* rate, the way `pocket_filter` was, or
not at all.
