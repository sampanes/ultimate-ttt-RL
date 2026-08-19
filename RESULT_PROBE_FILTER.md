# RESULT -- prove when probing cannot possibly matter, and don't do it (2026-08-18)

`#48`. `RESULT_PROBE_ABLATION.md` closed #47 with a choice between three
options and picked the one that was not on the original list: not "keep the
expensive probes" and not "delete the useful probes", but **skip the probes at
nodes where the rule says they cannot find anything**. This is that change,
built and gated.

It is the fourth optimisation in this series and the first that is not a trade.
The CUDA-graph wave bought throughput with a device path that can silently fall
back. Native selection bought it with a mirror that has to be kept in step with
the tree. Deferred retirement bought it with 471 MB of held memory. This one
removes work that is provably incapable of producing a result, so there is
nothing on the other side of the ledger -- and nothing for a strength match to
resolve.

## The predicate

`GameState.make_move` calls `check_ultimate_win` only when a move newly DECIDES
a mini-board, and that function returns a result in exactly two cases: a macro
line of three mini-boards owned by one player, or all nine decided (a draw). So
a child of a node can be terminal only if the move decides a mini-board AND

* **(a)** the mover already owns two mini-boards of some macro triple whose
  third is still undecided, or
* **(b)** only one mini-board is still undecided.

If neither holds, no legal move from that node ends the game.

```python
def could_end(mini_winners, mover):
    undecided = 0
    for m in mini_winners:
        if m == EMPTY:
            undecided += 1
    if undecided <= 1:
        return True
    for a, b, c in _MACRO_TRIPLES:
        x, y, z = mini_winners[a], mini_winners[b], mini_winners[c]
        owned = (x == mover) + (y == mover) + (z == mover)
        if owned == 2 and (x == EMPTY or y == EMPTY or z == EMPTY):
            return True
    return False
```

Three details are load-bearing and none of them is decoration.

`_MACRO_TRIPLES` is retupled from `engine.constants.WIN_PATTERNS`, the same
constant `check_ultimate_win` iterates, so the predicate cannot drift away from
the rule it was derived from.

Condition (a) tests ownership **by the mover**, not merely non-emptiness,
because a DRAW mini-board is decided and can never complete a line. Treating
"not empty" as "maybe mine" would admit triples that can never complete -- which
would still be correct, just weaker, and would have quietly halved the saving.

Condition (b) needs no separate fullness test, because a mini-board with
`mini_winners[m] == EMPTY` always has an empty cell: it is marked the moment its
last cell is filled, DRAW if no line formed. "Undecided" and "still playable"
are the same set.

## Why skipping is exact rather than a heuristic cut

A probe loop that marks nothing leaves every child with `solved is None`. Then
`_solve_from_children` over children that are all unsolved with none refuted
returns `None` -- the same `None` it would have returned after the loop ran, so
`_mark_solved` is not called and no propagation happens. The children are always
fresh at this point: both call sites (`_expand_children` and
`_expand_from_logits`) invoke the probe immediately after building them, so
there is no earlier proof to interact with.

Returning early therefore skips dead work and nothing else. There is no
approximation being traded for speed, which is what makes the rest of this
document short.

## The predicate is EXACT at the macro level, which was a finding

The design only needed a necessary condition. The sweep says it is more than
that.

Over all 4^9 macro configurations times both movers, minus the ones already
decided -- **391,550 cases, no sampling**:

| | count | rate |
|---|---:|---:|
| can really end this ply | 177,566 | 0.4535 |
| `could_end` admits | 177,566 | 0.4535 |
| **slack** | **0** | |

Zero. The two conditions are not a convenient over-approximation of
`check_ultimate_win`; they are that rule projected onto `mini_winners`.

All the remaining looseness is at the **position** level, where "some undecided
mini could become the mover's or a draw" does not imply that any single legal
move achieves it:

| corpus | n | `could_end` admits | really has a terminal child |
|---|---:|---:|---:|
| random reachable | 1,500 | 0.0147 | 0.0013 |
| late random (45+ plies) | 600 | 0.4483 | 0.0800 |
| exhaustive endgame closure | 12,172 | 0.2156 | 0.0814 |

None of that is what licenses the skip. **Necessity is**, and necessity is
checked separately and against the engine rather than against another
derivation -- `tools/probe_ablation --mode filter` ran the real probe at every
root of real searches and compared: **zero false negatives over 526,097 probe
roots**.

The oracle in the test suite is deliberately **re-derived rather than
imported**. An oracle that called the shipped `check_ultimate_win` would prove
only that the code equals itself, and a shared misreading of the rule would be
invisible because both sides would be wrong together. That paid for itself
immediately: it rejected a hand-built "two undecided, no threat" fixture that
had X two-of-a-line on the diagonal.

## Parity: zero drift, not statistical equivalence

Every earlier optimisation in this series was allowed to change something and
then argue the change was harmless. The graph wave computes the same numbers in
a different order; native selection returns the same index by a different route;
deferred retirement destroys the same nodes at a different moment. This one is
allowed to change **nothing**, and that is what buys it the promotion exemption:
a search proven identical at a fixed simulation count, which simply runs more of
itself under a clock, cannot be worse.

`agents/test_probe_filter.py` requires, at fixed simulations, over a corpus of
openings and late positions:

* bit-identical visit policies (`np.array_equal`, not a tolerance);
* identical tree SHAPE, keyed by move path so a structural difference shows up
  as a missing key rather than silently shifting every later comparison;
* identical `N`, `W`, `solved`, `is_terminal` and `terminal_value` on **every
  node** -- every legacy proof reproduced AND no proof that legacy did not make;
* both call sites (`wave_size=8` and the serial `wave_size=1` path expert
  iteration still uses);
* the mirror on and off -- the whole class runs twice, because the filter's
  early return leaves `node.selS` untouched and "the legacy loop also leaves it
  untouched when it marks nothing" is an argument, not a check.

**Both halves guard against vacuity**, because the expensive mistake in this
project has been a fixture with nothing to find. A corpus where the filter never
skips, or where no node is ever proved, would pass every assertion above while
testing nothing. So each is asserted to have happened -- and the serial-path
test failed its own guard on the first draft and had to be widened from one
position to the whole corpus.

What the corpus exercised:

| | |
|---|---:|
| positions | 24 |
| probe roots | 2,973 |
| roots skipped | 2,418 (0.8133) |
| children probed, legacy -> selective | 24,477 -> 4,460 (0.1822 kept) |
| nodes proved | 392 |

The 18.22% kept lines up with the 17.52% measured over 526,097 production probe
roots in #47.

There is also an end-to-end check that does not trust the predicate at all: for
every node in a legacy search's tree that the filter *would* have skipped, run
the probe loop's own definition independently and require it finds nothing.

## The throughput gate

120 fixed positions from 6 games, bare root, `pocket_defer` against
`pocket_filter` -- **the same instrument, the same `calibrate()`, the same
positions, arms run back to back on an idle box.** A gate whose two halves ran
under different instruments would be measuring the instruments.

| per move | legacy probes | selective | change |
|---|---:|---:|---:|
| probe roots considered | 4,869.2 | 5,868.1 | +20.5% |
| probe roots actually scanned | 4,869.2 | 788.3 | **-83.8%** |
| legal children probed | 39,169.4 | 5,806.2 | **-85.2%** |
| clone calls | 46,112.6 | 13,790.3 | -70.1% |
| make_move calls | 82,859.0 | 58,886.7 | -28.9% |
| probe ms/move | 154.6 | 46.6 | **-69.9%** |
| **nn/second x deadline** | 5,386.3 | 6,497.7 | **+20.6%** |
| simulations/move | 7,282.3 | 8,567.8 | +17.7% |
| search p99 ms | 980.6 | 980.4 | -0.0% |
| proven roots | 8 | 8 | 0 |

"Probe roots considered" going UP is not a regression -- it is the gain. Roots
equal expansions, and the selective arm does 20.7% more of them because it
finished 20.6% more search. Per unit of search the filter keeps **12.3%** of the
per-child work, better than the 17.5% #47 predicted, because the roots that
survive the filter have slightly fewer legal children each (7.37 against 8.04).

The predicate costs **1.711 us per root, 10.0 ms/move** -- about a fifth of the
46.6 ms that remains, and it is included in that figure rather than netted out
of it.

**Probe precision went up 7.1x**, which is the shape of the change: terminal
children per probed child 0.0041 -> 0.0293. The filter does not find more. It
finds the same eight proven roots with a seventh of the work, and
`searches with a proven root` is 0.0667 on both arms.

### It saved more than the instrument said it should -- and #47 did too

Predicted **+16.7%**, measured **+20.6%**, a ratio of 1.24. On the stricter model
that prices the remaining 46.6 ms rather than assuming the per-child ratio, the
prediction is +13.3% and the ratio is 1.55.

This is not new. #47 predicted +16.0% from the probe's profiled share and the
clean ablation measured +18.9%, a ratio of 1.18. Two independent runs, same
direction, same rough magnitude.

**A per-call clock can only see time inside the call.** What it cannot see is
what 33,000 `GameState` allocations and frees a move do to the allocator and the
cache *outside* the probe -- work that shows up as the rest of the search being
slower, attributed to the rest of the search. So the wrapper-derived probe cost
is a **lower bound**, and the honest reading is that probes cost 20-25% more
than the profile says.

Nothing here rests on the underestimate: the gate is decided by the two
uninstrumented `clean` arms. But it should stop being a surprise the third time,
and it is a reason to distrust any future "this is only N ms" argument built
from wrapper timings alone on an allocation-heavy path.

## The deployment latency gate

`tools/regress_engine --engine pocket_filter`, 10 games + 2 warmup against
`engine:final` (a different network -- never a mirror, which understates
re-rooting cost).

| | `pocket_defer` | `pocket_filter` |
|---|---:|---:|
| latency p99 | 980.0 | **980.1** |
| latency max | 980.2 | **981.4** |
| moves over budget | 0 | **0** |
| caller-side overhead p99 | 0.05 | **0.05** |
| worst chunk p99 | 7.2 | 6.9 |
| tree reuse adoption | 0.9569 | 0.9573 (= ceiling) |
| nn-evals/second | 6,671.8 | **7,725.7** |
| forced drains | 0 | 0 |

**5/5 checks passed**, and the reserve prediction held exactly. The registry
comment predicted that 20 ms would still be enough -- every earlier rise in this
series came from `release()` walking a bigger tree outside the search's
deadline, and #46 removed that walk -- and the measurement agrees: caller-side
overhead p99 **0.05 ms against 20 ms of reserve**, +19.95 ms of margin. This is
the first candidate in the series that got faster without needing to buy back
latency.

The +15.8% in nn/second here is smaller than the +20.6% on fixed positions, and
that is expected rather than contradictory: real games re-root into an adopted
subtree, so a smaller fraction of each move is fresh expansion, and expansion is
where probes happen. The fixed-position number is the controlled one; this is
corroboration under deployment conditions.

## Promoted, without a strength match

The five conditions set for promotion without games were:

1. fixed-simulation search bit-identical -- **yes**, on every node;
2. every legacy proof reproduced -- **yes**;
3. no new proof that legacy did not produce -- **yes**, the same test asserts
   equality in both directions;
4. deployment latency passes -- **yes**, 5/5;
5. effective search rate rises materially -- **yes**, +20.6%.

`DEPLOYED` is now `pocket_filter`; `pocket_defer` joins `SUPERSEDED` and stays
buildable, because it is the B side of the gate above and a promotion whose
predecessor stops building is a result that stops being checkable.

**The ladder-derived band is recorded and not spent.** +20.6% is 0.271 doublings
of clock, worth 0.5244-0.5514 head to head at fixed positions, or **0.5145 to
0.5306** in a mirror match after the compression measured in #46d -- about **585
games** to resolve the midpoint. `PROMOTIONS` stores that under `expected`, with
no `score`, no `ci` and no `games` field, and a test enforces their absence: an
expected band sitting in a field named like an observation is exactly how a
prediction gets quoted later as a result.

## What this does NOT claim

**It is not faster because it is smarter about search.** It is the same search.
Every number in the gate section is throughput, and the only reason throughput
converts to strength here is that the clock spends it on more of the identical
tree.

**The exactness at the macro level is not a correctness argument.** If a future
edit loosens `could_end` into a genuine over-approximation, the skip is still
sound as long as it stays necessary. The exactness assertion exists so that such
an edit has to be written down, not because anything depends on it.

**Nothing here revisits whether probes are worth their cost at other budgets.**
At the teacher's 800-simulation distillation budget the same subsystem is doing
target construction rather than deployment search, and
`uttt-solving-fixes-targets-not-students` measured that separately.

**It does not widen the previous promotion's scope.** `pocket_defer` beat
`pocket_r35` as a STACK, 0.5625 [0.5273, 0.5977]; that match never separated the
graph wave, native selection and deferred retirement from each other, and
nothing here revisits it.

**The five move disagreements between the two clean arms are not a parity
failure.** Both arms are wall-clock bound and the selective one completed 17.7%
more simulations -- mean |delta sims| 1,372.8 -- so it is a deeper search choosing
differently, which is the intended effect of the change and has no sign. Parity
is the fixed-simulation claim, and there it is 0 disagreements by construction.

## What follows

**Do not build the fused native probe.** The residual probe path is 46.6 ms of a
~980 ms move, about 4.8%, and 10.0 ms of that is the predicate rather than the
loop -- so a C++ `clone / make_move / classify / write` primitive would be
competing for roughly 36 ms, with a real parity surface, on a path that now runs
5,806 times a move instead of 39,169. #47's conclusion survives its own gate.

**Price the redundant `winner` crossing separately, and only now.** `make_move`
builds a `(True, self.winner)` tuple the probe loop discards before re-reading
`probe.winner` -- one of the four measured crossings per probed child. It had to
be measured after this change rather than folded into it, because the filter
removed 85% of the calls whose crossing was being priced. At 5,806 children a
move it may be too small to matter, which is the answer that measurement is for.

## Reproduce

    python -m agents.test_probe_filter
    python -m tools.probe_ablation --mode gate --positions 120 \
        --position-games 6 --tag probe48_gate
    python -m tools.regress_engine --engine pocket_filter --tag filter_gate
