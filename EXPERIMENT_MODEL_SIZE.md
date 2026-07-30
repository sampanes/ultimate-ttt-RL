# PRE-REGISTRATION: 172k vs 6.77M under the 1,000 ms deadline

Written and committed **before the first game was played**, because the prior
evidence is strong enough that a result in either direction could be
rationalised after the fact. Committing the prediction first makes that
impossible to do quietly.

## The question

At a fixed 1,000 ms move budget, is the deployment agent stronger with the
6,766,386-parameter gen-22 network or the 172,389-parameter squeeze network?

This has never been asked under a clock. Every previous model-size comparison in
this project was at a fixed simulation count, which cannot see the trade at all:
a network that is stronger per simulation can be weaker per second, and choosing
on per-simulation strength is the single most likely way to pick the wrong model
for deployment.

## What is already known

The two 172k checkpoints on disk -- `models/ab_arch/squeeze.pt` and
`models/pocket_candidate/squeeze_pocket.pt` -- were verified to hold
**identical weights** (the pocket file is a raw state-dict export of the other),
so the results below apply to the arm being run without re-derivation.

From `RESULT_ARCH_AB.md`, 300 games, raw argmax, colours swapped:

| net | params | vs gen-22 raw | winblock | gregory d3 | gregory d4 |
|---|---|---|---|---|---|
| `squeeze` | 172,389 | **0.527** | 0.905 | 0.728 | 0.653 |

So the small network is **already at parity or slightly ahead of the large one
as a raw network**, at 39x fewer parameters. It is a distilled student of that
very teacher, which is how it can be.

From the anchor ladder measured today (same engine, same net, budget the only
variable), one doubling of thinking time is worth:

    250 -> 500 ms    0.6875
    500 -> 1000 ms   0.6583
    1000 -> 2000 ms  0.6250

## Prediction, before the run

**The 172k agent wins at 1,000 ms.** Point estimate 0.55-0.70.

The reasoning: it starts from raw-network parity (0.527) and then adds search
that the large network cannot afford in the same budget. Even one extra doubling
of effective search is worth ~0.63 by the ladder measured above, and the
throughput ratio should exceed that.

## Where this prediction could be wrong, stated in advance

1. **Throughput may not scale with parameter count.** Params are not latency:
   an 8.8x CPU gap at matched params has already been measured to *flip its
   ranking on CUDA*, and much of a simulation's cost here is Python tree
   bookkeeping that both networks pay identically. If the small net buys only
   1.2x, there is no doubling to collect and the ladder arithmetic above does
   not apply. **Steps 1-2 measure this before the match, and the prediction
   should be read as conditional on that ratio.**

2. **Raw-argmax parity is a POLICY claim; MCTS also leans on the value head.**
   A distilled student can match its teacher's policy argmax while carrying a
   worse-calibrated value head, and value quality matters more under search than
   it does at argmax. Nothing measured so far constrains this.

3. **0.527 over 300 games is parity, not an edge.** The interval is roughly
   +/-0.04, so "slightly ahead" is not established; "not behind" is.

## Decision rule, fixed in advance

Promotion is decided by **step 5 alone** -- `pocket` vs `final`, both at
1,000 ms, 240 games, paired openings, colours swapped -- with step 6 (`pocket`
vs `anchor_C`) confirming the winner keeps its place on the ladder.

Steps 1-4 explain a result. They never justify one. In particular: **a model is
not promoted for achieving more simulations.**

If the 172k agent loses despite materially more search, the next step is exactly
one intermediate size (`midsize`, 921,026 params, already registered) -- not a
broad architecture sweep. `RESULT_ARCH_AB.md` already returned a null on
architecture at equal wall clock, and re-running that is not a use of the box.

If it wins, it changes the production architecture and the future training
target, and the browser deployment inherits a 0.70 MB model that is also
stronger.

## Arms

    engine:pocket        172,389 params, squeeze,  1000 ms, final engine
    engine:final       6,766,386 params, arena22,  1000 ms, final engine
    engine:pocket_raw    172,389 params, no search
    engine:gen22_raw   6,766,386 params, no search
    engine:anchor_C    6,766,386 params, 2000 ms, the frozen primary anchor

All resolved from `tools/engine_registry.py` and fingerprint-verified. Seeds:
6300 (`anchor` namespace) throughout, so no opening set is shared with the
ladder that produced the anchor.
