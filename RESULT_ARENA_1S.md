# RESULT -- the 1,000 ms arena: search-engine work, measured by strength

> **Superseded as the live reference by `ENGINE_BASELINE.md`.** This file is the
> record of how the engine was built and what was rejected along the way; the
> baseline it produced is now frozen at tag `arena-1s-baseline`, with named,
> fingerprint-verified configurations in `tools/engine_registry.py`. Reproduce
> anything here with `engine:final` / `engine:original`, not with the ad-hoc
> spec strings in the Reproduce section below -- those still inherit code
> defaults and are no longer the authority on what was run.

The product is the complete network-plus-search agent under a move deadline, not
the raw network. This is the first program judged that way: every change below
was promoted or rejected on **win rate at an identical wall clock**, and no
change was promoted for making a cost metric look better.

The network was not touched. Same gen-22 checkpoint (6,766,386 params)
throughout. Everything here is search engine.

---

## Frozen operational requirement

Set in `tools/arena_1s.py:REQUIREMENT` before any candidate was benchmarked:

    p99 move latency  <=  1000 ms      the requirement
    max move latency  <=  1250 ms      reported, and a hard reject

The strict p99 because a mean cannot see a player that averages 900 ms and
occasionally takes 1.8 s; the absolute cap because a p99 over a few thousand
moves still permits a handful of arbitrarily long ones.

Reference hardware is the CUDA box. The browser port is a separate measurement
on different silicon and is out of scope here.

---

## Headline

| comparison | score for A | 95% CI | W/D/L | n |
|---|---|---|---|---|
| cross-move tree reuse vs rebuild every move | **0.6375** | [0.6000, 0.6750] | 84/138/18 | 240 |
| batched wave expansion vs per-leaf | **0.6167** | [0.5790, 0.6544] | 77/142/21 | 240 |
| **final engine vs original engine** | **0.7229** | [0.6884, 0.7575] | **113/121/6** | 240 |

All at 1,000 ms per move on both sides, paired openings, colours swapped,
temperature 0. `final` = tree reuse + batched expansion; `original` = rebuild
the tree every move, per-leaf expansion. Six losses in 240 games.

The combined number is measured, not inferred. Two separate one-change wins do
not multiply, and they could in principle overlap or interfere, so the whole
delta was played out in a single match rather than composed from the parts.

Cost and compliance, same two configurations:

| | original | final |
|---|---|---|
| network evaluations / s | 1,382 | 2,997 |
| network evaluations / move | 1,177 | 2,461 |
| root visits / move | 2,186 | 3,877 + 3,150 inherited |
| p99 move latency | 1000.6 ms (**FAIL**) | 998.7 ms (**PASS**) |
| worst chunk p99 | 68.6 ms | 5.0 ms |
| CUDA peak | 89 MB | 89 MB |

---

## 1. Cross-move tree reuse

`TreeReuseSearcher` re-roots at the position actually reached instead of
starting from an empty tree.

Adoption is **proven, never assumed**: exactly one of our marks and one of
theirs changed, both moves exist as nodes, the survivor is already expanded, and
its `to_play` matches. Anything else is a miss with a named reason and a fresh
tree. A wrongly adopted subtree would score a position other than the one on the
board -- silent, and the worst kind of strength bug. Detaching the survivor is
load-bearing: `_backup` walks to the root, so an attached one would push every
new simulation into the discarded tree.

Measured: adoption **0.957** (misses are one first-move per game, plus 2 in
5,593 where the opponent replied into a node the search never expanded), and
**3,150 inherited simulations per move** against 3,877 new ones -- the tree
arrives carrying more search than the move itself adds.

It wins while spending **less** time per move and **fewer** network evaluations
than the arm it beats. It is not buying strength with compute.

---

## 2. The GPU was never starved

The premise going in was that small per-tree batches were leaving the GPU idle
and that an inference server collecting leaves across trees was the answer. A
wave-size sweep at 1,000 ms refuted it:

| wave | nn-evals/s | sims/move |
|---|---|---|
| 1 | 686 | 1,016 |
| 8 | 1,399 | 2,269 |
| 32 | 1,504 | 1,328 |
| 64 | 1,525 | 2,631 |

Batching 1 -> 8 doubles throughput; **8 -> 64 buys 9%**, at a CUDA peak of
89-107 MB. An inference server would have solved a problem that does not exist.

The cost was *around* the forward pass, not in it. `_expand_from_logits` masked
and softmaxed on the device and pulled one row back, so every leaf paid a
host-to-device tensor build plus a device-to-host sync -- **2K round trips per
wave of K, after the single batched forward**. `_expand_wave` masks the whole
`(K, 81)` block at once, leaving exactly two transfers:

    nn-evals/s   1,382 -> 3,033
    sims/move    2,172 -> 4,792

Throughput is not a promotion criterion, so it was then played: **0.6167** at
equal wall clock, same binary, one flag apart.

Gated to timed mode by default. The wave path generated the frozen distillation
corpora, and softmaxing a `(K, 81)` block reduces in a different order than 81
separate vectors, so priors differ in the last bits -- irrelevant under a clock,
but under a fixed simulation count it would make `tools/extract_child_q` report
drift against artifacts already hashed. Parity is tested: identical priors to
1e-6 and identical visit counts across wave 2/8/16.

---

## 3. The latency tail was the garbage collector

The first baseline failed the frozen requirement at p99 1000.6 ms. Predictive
admission was not the problem: a chunk is atomic, and the worst chunk was 90 ms
against a ~6 ms typical, so something was stalling mid-chunk that no predictor
can foresee.

Isolated by experiment rather than guessed:

| configuration | worst chunk mean / p99 / max | move p99 | verdict |
|---|---|---|---|
| GC auto (CPython default) | 25.2 / 87.6 / 90.6 | 1037 ms | FAIL |
| + cycle-breaking, GC auto | 23.5 / 52.7 / 53.2 | 1016 ms | FAIL |
| + `--gc deferred` | **3.1 / 4.7 / 5.6** | **989 ms** | **PASS** |

Two distinct problems, which is why the first fix only half-worked.

**Cyclic garbage.** Every `MCTSNode` points at its parent and every parent at
its children, so a discarded tree is precisely the garbage only the cyclic
collector can reclaim -- and at ~2,700 expansions per move there is a great deal
of it. `TreeReuseSearcher.release()` breaks those cycles so the tree dies by
refcount at a moment we choose: one O(nodes) walk instead of an unpredictable
heap scan. It runs after adoption and after node counting but **before the clock
starts**.

**Gen-2 scans.** Breaking cycles does not stop CPython walking every tracked
object looking for cycles, and that pause scales with the ~20k-node **live**
tree whether or not any of it is collectable. That is the residual 52.7 ms.

`--gc deferred` turns automatic collection off during play and collects once per
**game** boundary. This is safe *only because* `release()` breaks the cycles:
refcounting reclaims the trees, and the explicit collect is insurance against
third-party cycles rather than the mechanism keeping memory bounded. Between
games and not between moves, because both players share one process here and a
collect between moves would land inside the other player's budget.

Verified both halves: with cyclic GC disabled a released tree is still reclaimed
(live `MCTSNode` instances drop >90%), and the retained subtree survives release
with its children and inherited visits intact.

The honest cost is visible in every report: `release()` shows up as ~6.5 ms of
non-search overhead per move, charged inside `move_ms` rather than hidden
outside the deadline.

---

## 4. A proven root should stop searching

Under a clock, a proven root returns immediately. Its remaining simulations skip
the network entirely, so the loop would spend the whole deadline on descents
that cannot change the answer -- measured at **1,193 sims/move of which only 48
did any work**. This is the clearest case in the whole program of why simulation
count is not a measurement.

Deadline mode only: under a fixed count the visit counts *are* the product, and
the frozen pilot corpora came from that path.

Two bugs surfaced here, both reachable only once a root can be proven before the
counts are populated. An all-zero visit policy argmaxes to cell 0, illegal in
almost any position -- the arena caught it as `returned illegal move 0` -- so
`pi` now falls back to the priors; and the early exit is gated behind `min_sims`
so a proven loss or draw has visits to order.

---

## 5. Transposition and symmetry: measured, not built

60 real 1,000 ms searches, every expanded node's state reconstructed and hashed.

    2,947 expanded nodes per search
    transposition ceiling   0.0313
    symmetry adds           0.0000
    combined                0.0313

    phase    searches  expanded   transp   +symm  combined
    early          20     3,538   0.0191  0.0000    0.0191
    mid            30     3,160   0.0357  0.0000    0.0357
    late           10     1,126   0.0702  0.0000    0.0702

A **perfect, zero-overhead** transposition table would avoid 3.1% of network
evaluations. Against a DAG's parent-independent backup bookkeeping that is not
worth building. Late game has the highest rate (7.0%) and the smallest trees
(1,126 nodes against 3,538 early), so it is also the smallest absolute saving.

Symmetry adds **exactly zero**, and the reason is structural rather than a small
effect: every node in one search tree descends from a single root, so two lines
can only be symmetric to each other if the root itself is symmetric.

**Scope, stated precisely.** This measures duplication *within* a search, which
is what a transposition table or an in-search evaluation cache would exploit. A
persistent cache shared across moves or across games is a different question
this does not answer. Given the within-tree figure is 0.0000 for symmetry, it is
not worth funding the follow-up.

The symmetry group is verified rather than assumed. An incorrect group would not
crash; it would merge positions that are not equivalent and **inflate** the
ceiling, arguing for a cache that cannot exist. `tools/test_search_reuse.py`
checks distinctness, mini-structure preservation, canonical invariance and
canonical separation, and asks the engine itself whether each of the eight
transforms really is an automorphism of the legal-move rule.

---

## 6. The external anchor is saturated

    final vs gregory(d4), 1000 ms, 60 games:  1.0000  (60/0/0)

gregory(d4) no longer resolves anything about this player and cannot serve as a
ruler for further search work. It was the one anchor independent of the training
gene pool, so **a harder external opponent is now a prerequisite** for the next
round of search changes -- otherwise every future variant will be compared only
against other variants of itself, which is how a ladder quietly detaches from
absolute strength.

---

## What was NOT done, and why

- **No inference server / concurrent leaf collection across trees.** The wave
  sweep says the GPU is not the constraint. Building it would have been work
  aimed at a measured non-problem.
- **No transposition DAG, no symmetry folding.** 3.1% and 0.0% ceilings.
- **No `c_puct` sweep.** It sits behind reuse and throughput in the ordering and
  has not been reached. Note that the operating point moved a long way -- from
  ~2,200 to ~7,000 root visits per move -- so the previous null at 50 sims
  ([[uttt-cpuct-null-gen15-gain]]) does not transfer and the sweep is now open
  again.
- **No small-net vs large-net comparison at 1,000 ms.** This is the remaining
  item from the original plan and the one most likely to pay: the question is
  whether 172k + more search beats 6.77M + less search under the deadline, and
  it has never been asked under a clock.

---

## Artifacts

`results/` is gitignored, so the numbers here are the record.

    results/arena_1s/h2h_final_vs_original.json    the headline
    results/arena_1s/h2h_reuse.json                tree reuse
    results/arena_1s/h2h_bexp.json                 batched expansion
    results/arena_1s/anchor_final.json             gregory d4
    results/arena_1s/search_reuse.json             transposition / symmetry
    results/arena_1s/*_moves.npz                   per-move records

Per-move records carry latency, simulations, network evaluations, nodes
expanded, nodes reused, transposition hits, chosen move and root policy, so a
match can be replayed exactly by pinning the simulation counts.

## Reproduce

    python -m tools.arena_1s --mode h2h --games 240 --warmup-games 2 \
        --player-a "ms=1000,wave=8,solve=1,cpuct=1.5,reuse=1,bexp=1" \
        --player-b "ms=1000,wave=8,solve=1,cpuct=1.5,reuse=0,bexp=0" \
        --gc deferred --tag h2h_final_vs_original
    python -m tools.measure_search_reuse --positions 60 --ms 1000 --wave 8
