# ENGINE BASELINE -- frozen 2026-07-30, tag `arena-1s-baseline`

The 1,000 ms network-plus-search agent is now the production baseline, not an
experimental option. This file is the freeze: what it is, what it was measured
on, and what must not move underneath it.

Everything here is enforced by code rather than by convention. The authority is
`tools/engine_registry.py`; this document is its readable form. If the two ever
disagree, the module is right and this file is stale.

    python -m tools.engine_registry        # verify all 8 frozen engines
    python -m tools.regress_engine         # re-measure the latency/reuse gate
    python -m tools.test_engine_registry   # prove the guards still bite

---

## 1. The latency policy

    p99 move latency  <=  1000 ms     the requirement
    max move latency  <=  1250 ms     reported, and a hard reject

Frozen 2026-07-28, before any candidate was benchmarked, and unchanged since.
A mean is the wrong statistic for a deadline: a player averaging 900 ms that
occasionally takes 1.8 s has not met the objective, and the mean cannot see it.
The absolute cap rides alongside because a p99 over a few thousand moves still
permits a handful of arbitrarily long ones.

Timing is an OUTER measurement around the whole move -- re-rooting, tree
release, instrumentation and the argmax included -- not the interval the search
times itself over. Reporting the inner number against a deadline would quietly
exclude work the deadline covers.

**Ladder rungs above 1,000 ms are exempt by design.** `anchor_C` and `anchor_D`
exist to be hard fixed opponents, not to ship. Their latency is still reported,
because an exempt rung can still reveal a stall and the p99 is how you would see
it, but it is not judged.

---

## 2. The promoted engine

`engine:final` -- gen-22 network, wall-clock MCTS, cross-move tree reuse,
batched wave expansion, solved-node propagation, `--gc deferred`.

| flag | value | why |
|---|---|---|
| `ms` | 1000 | the deployment budget |
| `reserve` | 20 | 2% of budget, held back so the last chunk cannot overrun |
| `wave` | 8 | 8 -> 64 buys 9%; the GPU is not the constraint |
| `cpuct` | 1.5 | inherited, NOT yet tuned under a clock |
| `solve` | 1 | solved-node propagation, incl. the proven-root early exit |
| `reuse` | 1 | cross-move tree reuse |
| `bexp` | 1 | batched wave expansion |
| `maxsims` | 200000 | a ceiling, not a target; never binds at 1 s |

Implicit parameters that no flag pins -- `min_sims`, `deadline_margin`, the
virtual-loss magnitude, the `_MIN_WAVES` floor, `add_dirichlet` -- are covered
by the fingerprint (section 5), because a change to any of them changes the
engine without changing a single character of the command line.

### Measured, 240 games, both sides at 1,000 ms

| | original | final |
|---|---|---|
| score | -- | **0.7229** [0.6884, 0.7575], W113/D121/L6 |
| network evaluations / s | 1,382 | 2,997 |
| network evaluations / move | 1,177 | 2,461 |
| root visits / move | 2,186 | 3,877 + 3,150 inherited |
| tree reuse adoption | -- | 0.9569 |
| p99 move latency | 1000.6 (FAIL) | 998.7 (PASS) |
| worst chunk p99 | 68.6 ms | 5.0 ms |
| CUDA peak | 89 MB | 89 MB |

---

## 3. Named configurations

"Original" must never become whatever the defaults happen to be. Both sides of
the headline are pinned entries in the registry, and every option is explicit --
none is inherited.

| name | role | budget | params | reuse | bexp |
|---|---|---|---|---|---|
| `original` | candidate | 1000 ms | 6,766,386 | 0 | 0 |
| `final` | candidate | 1000 ms | 6,766,386 | 1 | 1 |
| `anchor_A` | anchor | 250 ms | 6,766,386 | 1 | 1 |
| `anchor_B` | anchor | 500 ms | 6,766,386 | 1 | 1 |
| `anchor_C` | anchor | 2000 ms | 6,766,386 | 1 | 1 |
| `anchor_D` | anchor | 4000 ms | 6,766,386 | 1 | 1 |
| `pocket` | candidate | 1000 ms | 172,389 | 1 | 1 |
| `midsize` | candidate | 1000 ms | 921,026 | 1 | 1 |
| `gen22_raw` | network only | no search | 6,766,386 | -- | -- |
| `pocket_raw` | network only | no search | 172,389 | -- | -- |
| `midsize_raw` | network only | no search | 921,026 | -- | -- |

The `_raw` arms are `sims=0`: masked policy argmax, no tree. They exist because
a model-size result is otherwise uninterpretable -- if the small net wins at
1,000 ms you cannot tell whether the network is better or whether it merely
bought more search, and those two answers imply opposite next moves.

`sims=0` is a separate code path rather than a one-simulation search, and that
is not fastidiousness: at the root every child has `N=0`, so `sqrt(N_parent)=0`
kills the PUCT exploration term for all of them, the scores tie, and the pick
falls out of dict order. A 1-sim search agrees with the policy argmax on
**0.197** of positions -- it would have been a plausible-looking way to measure
the wrong thing.

`original` and `final` differ in **exactly** `reuse` and `bexp` -- asserted by a
test, because the 0.7229 attribution is false the moment a third difference
appears. Ladder rungs differ from `final` in **exactly** the budget and its
proportional reserve -- also asserted, because otherwise an ordering result is
not about time at all.

    --player-a "engine:final"      resolved from the registry, then verified
    --player-a "ms=1000,reuse=1"   ad-hoc, and reported as NOT registry-frozen

---

## 4. Checkpoints

| path | sha256 (first 16) | arch | params |
|---|---|---|---|
| `models/expert_iter_v2/teacher.pt` | `cfef6febd4a43036` | arena22 | 6,766,386 |
| `models/pocket_candidate/squeeze_pocket.pt` | `b028d3499eca8b10` | squeeze | 172,389 |
| `models/ab_arch/plain.pt` | `02a90b8364885ab5` | plain | 921,026 |

Full digests are in `tools/engine_registry.py:CHECKPOINTS` and are verified on
every build. A retrained file at the same path is a hard failure.

---

## 5. What is checked, and when

Three layers, each catching drift the others cannot see.

1. **Resolved-config fingerprint.** A sha256 over every parameter that can
   change how a player plays, including the code defaults no spec mentions.
   Hard failure, always.
2. **Checkpoint sha256.** Catches a retrained or overwritten `.pt`. Hard
   failure, always.
3. **Engine source sha256** over `agents/mcts.py`, `agents/agent_base.py`,
   `agents/neural_net_agent_3.py`, `engine/rules.py`, `engine/game.py` -- the
   files whose bytes decide how a search plays. No config fingerprint can see an
   edit to the search itself.
   **Hard failure for an anchor, warning for a candidate**, because a candidate
   is *expected* to change the search and an anchor is precisely the thing that
   must not.

`tools/arena_1s.py` is deliberately outside layer 3: it decides how play is
measured and recorded, not how it is played, and a regression there surfaces in
the gate instead.

Layer 3 cannot prevent drift, only make it impossible to miss. The recovery path
is this tag:

    git worktree add ../uttt-anchor arena-1s-baseline

A candidate must never alter the anchor implementation it is evaluated against.
If the anchor's sources have moved, re-run the anchor from the tag -- do not
reach for `--allow-anchor-drift`, which exists to make the override visible in
the logs, not to make it acceptable.

---

## 6. Openings and seeds

Openings come from `scripts.expert_iter._eval_openings(n, seed)`; colours are
swapped within each opening; the per-game RNG is reseeded to
`seed + opening_idx*2 + side`. Students play at temperature 0. A seed therefore
*is* an opening set.

| namespace | seed | used for |
|---|---|---|
| headline | 6100 | original vs final; the published 0.7229 |
| ladder | 6200 | ordering between adjacent rungs |
| anchor | 6300 | a candidate measured against a frozen anchor |
| tune | 6400 | elimination rounds of a parameter sweep |
| confirm | 6500 | held out: the powered final between survivors |
| gregory_d4 | 6144 | the retired low anchor |

Separate namespaces because the same fixed openings are reused across many
experiments. A sweep that eliminates on the same positions it is later confirmed
on will overfit them -- cheap to avoid, invisible if not.

The only nondeterminism left is machine timing jitter deciding how many
simulations fit in the budget. That is inherent to the thing being measured, not
a defect, and it is why `simulations_completed` is recorded per move: a match
can be replayed exactly by pinning those counts.

---

## 7. Reference environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3080, 10 GB, capability 8.6 |
| driver | 591.74 |
| CPU | AMD64 family 23 model 113, 24 threads |
| OS | Windows 10 19045 |
| python | 3.11.9 |
| torch | 2.7.1+cu128 (CUDA 12.8, cuDNN 90701) |
| TF32 matmul | enabled |
| required env | `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

Compared on every run, never enforced. A hardware or driver change does not
invalidate a frozen *configuration* -- it invalidates a frozen *latency*, which
is exactly why the requirement is re-measured by the regression gate rather than
trusted from this table. `tools.regress_engine` skips its throughput check off
the reference box and says so.

---

## 8. The regression gate

    python -m tools.regress_engine        # ~10 min, non-zero exit on failure

| check | gate | frozen |
|---|---|---|
| latency p99 | <= 1000 ms | 998.7 |
| latency max | <= 1250 ms | -- |
| tree reuse adoption | >= ceiling - 0.01 | 0.9569 |
| inherited / new simulations | >= 0.50 | 0.81 |
| network evaluations / move | >= 70% of frozen | 2,461 |

Two of these are shaped against specific failures that a naive version would
miss:

- **Adoption is gated against its own structural ceiling**, `1 - games/moves`,
  not a flat number. The first move of every game has no prior tree, so a fixed
  floor would mean different things at different game counts.
- **Inherited simulations are gated separately.** A bug that re-rooted correctly
  but dropped the subtree's statistics would hold adoption at 0.957 and inherit
  nothing; the adoption check alone would pass it.
- **Throughput is gated only on the reference box.** Under a deadline, latency
  is pinned by construction -- an engine that became three times slower would
  still show p99 ~= 1000 and clear every latency check while playing far weaker.
  Simulations per move is the statistic that actually degrades, and it is also
  machine-dependent, so it is enforced here and explicitly skipped elsewhere.

Playing strength is deliberately **not** gated. A strength regression needs a
match against a fixed opponent, which is hours; that is the anchor ladder's job.
This gate catches the cheap common failure -- a broken deadline or silently
disabled reuse -- in about ten minutes.

---

## 9. Known-open at freeze time

- **`c_puct` is untuned under a clock.** 1.5 is inherited from a fixed-simulation
  regime. The operating point moved from ~2,200 to ~7,000 root visits per move,
  so the old null at 50 sims does not transfer.
- **`gregory(d4)` is saturated** at 1.0000 (60/0/0) and is retired as a primary
  discriminator. It stays in the graph as a low anchor.
- **The ladder shares the training gene pool.** Every rung is the same gen-22
  network. It is fixed and deterministic relative to future candidates, which is
  what an anchor has to be, but it is not an independent opinion about strength.
  A genuinely external opponent remains wanted.
