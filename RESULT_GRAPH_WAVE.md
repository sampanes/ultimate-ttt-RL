# RESULT -- the CUDA-graph wave wins at equal clock: 0.5458 [0.5126, 0.5791] (2026-08-09)

Task #41. The expansion wave's device sequence is replayed from a captured CUDA
graph instead of issued kernel by kernel. Two stages: throughput first, then
strength, and only the second could promote.

`agents/graph_wave.py`, `engine:pocket_graph` (fingerprint `df93af350ef9f906`).
Predecessor: `RESULT_KERNEL_TRACE.md`.

## Stage 2 -- strength, which is the only thing that promotes

    pocket_graph vs pocket_r35    0.5458 [0.5126, 0.5791]
    W45 / D172 / L23              n=240, paired openings, sides swapped
    1000 ms per move both sides   seed 6500 (held-out `confirm` namespace)
                                  157.7 min

The interval excludes 0.5. The seed namespace was held out: nothing in the
design or the reserve was chosen against these openings.

**The candidate is handicapped in this comparison and still wins.** It differs
from the incumbent in TWO declared keys, not one -- the graph flag and the
50 ms reserve the graph forces (below). The larger reserve costs it 1.5% of
thinking time, so the graph's own effect is at least the measured 0.0458 and
the confound runs against the result rather than for it.

## Stage 1 -- throughput, against a pre-registered prediction

`RESULT_KERNEL_TRACE.md` predicted the device sequence was worth 279.3 ms of a
~913 ms search (195.5 after a 30% discount), so removing it should buy 27.2% to
44.1% more waves. Registered before this was built.

| | predicted | measured |
|---|---|---|
| identical positions, bare root | +27.2% .. +44.1% | **+31.5%** |
| real games, tree reuse on | | **+29.6%** |

Both inside the band. The bench saving converts into real search rather than
being absorbed by the search's own bookkeeping, which was the open question.

`tools/regress_engine`, 10 games vs `final`:

| | `pocket_r35` | `pocket_graph` |
|---|---|---|
| latency p99 | 986.2 | 980.2 |
| latency max | 991.4 | 987.3 |
| overhead p99 (caller-side tree work) | 22.55 | 30.87 |
| worst chunk p99 | 24.71 | 7.60 |
| nn-evals / move | 3047.3 | 3949.9 |
| nn-evals / sec | 3509.4 | 5007.6 |
| tree reuse | 0.9524 | 0.9558 |
| **gate** | **5/5** | **5/5** |

The candidate runs FEWER simulations per move (5236 vs 5411) and MORE network
evaluations (3950 vs 3047): 75% of its simulations reach the net against 56%.
Only the second tracks the speedup, and quoting sims/move here would have shown
a regression.

## What is captured

    forward -> masked_fill -> softmax -> D2H probs -> D2H values

Both pulls are inside the capture, as asynchronous copies into pinned host
buffers. That is where a second win comes from: the eager wave performs **four
`cudaStreamSynchronize` calls that the search never asks for** -- PyTorch emits
one behind every pageable memcpy -- and staging through pinned memory with the
pulls captured leaves one event wait.

Not captured: building the planes and the legal mask, and constructing
children. All host work. The graph is a device-side replay, not a rewrite of
the search.

**One shape.** Live play spends 90.7% of waves and 95.4% of network evaluations
at k=8, so a single capture covers the workload. Every other size keeps the
eager path, which is also the correctness oracle.

## Correctness, established before any game was played

| gate | result |
|---|---|
| bit-identical priors and leaf values vs eager | 320 distinct production waves, max diff **0.0** |
| fixtures actually vary | asserted, or the parity test is vacuous |
| stale-buffer check | 40 distinct inputs -> 40 distinct outputs |
| `run()` returns copies | pinned outputs are overwritten every replay |
| k != 8 falls back and matches | k in {1, 2, 5, 7} |
| capture failure degrades | runs eagerly; an arm that ASKED for the graph refuses to start |
| full search at fixed sims | identical visit policy and counters |

`_expand_wave` is split into the device half plus a shared `_expand_children`
used by both paths. Sharing it is the point: two copies of that loop would make
the parity test compare two copies of the same code instead of the thing that
actually differs.

## Why the reserve had to grow

50 ms, up from 35. Same cause as `pocket` -> `pocket_r35`: all p99 risk is
`_adopt` and `release()` of the discarded tree, which sit inside the move the
requirement is written against but outside the interval the search times
itself over. More search means a bigger tree to walk, so **the overhead grows
with the speedup that caused it** -- overhead p99 22.55 -> 30.87 ms. The
reserve is the configuration fix; moving that walk off the critical path is
tree-core work.

Worst chunk p99 went the other way, 24.71 -> 7.60 ms, because a faster wave
makes each atomic chunk shorter.

## Registry

`graph` is now pinned in `_FINAL`, because the registry's job is that a changed
DEFAULT trips the guard, not only a changed flag. That moved all twelve
fingerprints. **Nothing was re-measured**, so the claim that no engine moved is
checkable rather than asserted: `PRE_GRAPH_FINGERPRINTS` holds the old values
and a test requires that stripping the one new key reproduces every one of
them. Verified for all twelve.

`resolved_config` keys off the REQUESTED flag, not the captured object.
`--freeze` runs on CPU where capture cannot succeed, and keying off the object
would have frozen the graph engine as `graph_wave=False` -- fingerprint-
identical to the incumbent. Freeze this engine with `--device cuda`.

`agents/mcts.py` was re-hashed. The eager path's behaviour is unchanged and
that is checked, not claimed: the frozen replicas in
`tools/test_profile_expand.py` and `tools/test_profile_kernels.py` were written
statement-for-statement against the PRE-change `_expand_wave` and still require
bit-identical output. Anchors were not re-measured because they do not play
differently.

## Two harness errors, both caught by the control arm

**A 2000 ms opponent failed the incumbent.** The first version of
`tools/graph_ab` defaulted to `anchor_C`. That opponent builds a far larger
tree, and under deferred GC the collections it produces pushed worst-chunk p99
to 90 ms and failed BOTH arms -- including the frozen incumbent, which the real
gate passes at 986.2 ms. A control arm that fails is a broken harness, not a
finding, and the "the graph fails latency" reading from that run is void. The
match half now reads `tools/regress_engine`'s own output instead of
re-implementing the match, so there is one latency number and it is the one the
requirement is written against.

**A metric divided by itself.** The stage 1 verdict computed "what fraction of
the wave speedup arrived as extra search" by dividing the nn gain by the change
in a `wave_ms` defined as `search_ms / nn`. Those are the same measurement and
their ratio is ~1 by construction. It checks the pre-registered prediction now.

`tools/runlock.py` gives `profile_kernels`, `graph_ab` and `regress_engine` a
single-instance lock. A p99 is exactly the statistic a stray process ruins.

## #43 -- model size is still not a lever, and the graph did not change that

The wave went from about 18% device-bound to about 61% (CUPTI device busy
~244 us against a wave that fell from 1349 to 398 us), which is the trigger for
re-testing `RESULT_MODEL_SIZE`: its conclusion was measured on an engine
spending most of a wave issuing tiny CUDA operations, and that condition has
now changed.

Four arms, one set of 40 real positions, full deadline each, bare root:

| engine | params | nn-evals/move | sims/move |
|---|---|---|---|
| `pocket_r35` | 172,389 | 2627.8 | 2632.2 |
| `pocket_graph` | 172,389 | 3525.3 | 3531.6 |
| `final` | 6,766,386 | 2669.0 | 2675.6 |
| `final+graph=1` | 6,766,386 | 3577.6 | 3588.4 |

    172k / 6.77M search rate, graph OFF    0.985x
    172k / 6.77M search rate, graph ON     0.985x

**Unchanged to three digits.** A 39x parameter cut buys no additional search
either way, and the graph lifts both networks by the same amount (+34.2% and
+34.0%). That the gain is identical across two different architectures is
itself the point: what the graph removes is dispatch, and dispatch does not
care how many parameters the kernels are multiplying.

So the answer to #43 is no. Removing ~950 us/wave of CPU dispatch does not make
network size relevant again, and `RESULT_MODEL_SIZE`'s deployment conclusion --
prefer the small net, it is not paying for its size -- survives the change that
was supposed to threaten it.

**This does not refute the published 1.24x.** That figure is simulations/move
measured in real play with tree reuse, where the two engines diverge into
different positions; this is network evaluations per move on identical
positions from a bare root, where nearly every simulation expands a leaf
(2627.8 nn against 2632.2 sims). Both can be true. What carries is the
comparison this table was built for -- the same measurement on both arms,
before and after -- and that ratio did not move. `RESULT_MODEL_SIZE` already
notes its own isolated benches gave 1.13x and warns that 276 moves cannot rank
throughput.

## What this does NOT license

**The latency numbers in the stage 2 match are not a deployment measurement.**
Both arms run the same network, so it is effectively a mirror: inherited
simulations run 5020 and 4000 per move against 2818 and 2142 in the cross-
network gate. Self-play inflates tree-reuse inheritance and understates
re-rooting cost. The deployment latency verdict is the `regress_engine` run
against `final`, and only that.

**No claim about other budgets.** Measured at 1000 ms only. The wave is a fixed
per-call cost, so the share it represents falls as the budget grows.

**The gain is not additive with the native-tree work.** Removing ~950 us/wave
moves the bottleneck; the ~195 ms/move of tree bookkeeping that `_best_child`
targets is now a larger share of what remains, not a smaller one.

## Reproduce

    python -m agents.test_graph_wave
    python -m tools.regress_engine --engine pocket_r35   --games 10 --tag graph-control
    python -m tools.regress_engine --engine pocket_graph --games 10 --tag graph-candidate
    python -m tools.graph_ab --positions 50 --position-games 3
    python -m tools.arena_1s --mode h2h --player-a engine:pocket_graph \
        --player-b engine:pocket_r35 --games 240 --seed 6500 --tag graph-h2h
