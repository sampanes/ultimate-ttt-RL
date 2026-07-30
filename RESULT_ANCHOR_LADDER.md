# RESULT -- the time-budget anchor ladder (2026-07-30)

`gregory(d4)` saturated at 1.0000 (60/0/0) and stopped resolving anything about
this player. Every search variant measured after that point would have been
compared only against other variants of itself, which is how a ladder quietly
detaches from absolute strength while all of its numbers keep looking healthy.

This is the replacement ruler: the frozen final engine at other budgets. Same
network, same search, **budget is the only variable** -- asserted by a test, not
by intention, because if the rungs differed in anything else an ordering result
would not be about time at all.

    anchor_A   250 ms      anchor_C  2000 ms   evaluation-only, latency-exempt
    anchor_B   500 ms      anchor_D  4000 ms   evaluation-only, latency-exempt
    final     1000 ms      <- the deployment agent

---

## The ladder is ordered

Every adjacent pair measured. 120 games each (100 at the top), paired openings,
colours swapped, temperature 0, one shared opening set (seed 6200).

| doubling | score for the longer budget | 95% CI | W/D/L | n | verdict |
|---|---|---|---|---|---|
| 250 -> 500 ms | **0.6875** | [0.6369, 0.7381] | 50/65/5 | 120 | ORDERED |
| 500 -> 1000 ms | **0.6583** | [0.6052, 0.7115] | 46/66/8 | 120 | ORDERED |
| 1000 -> 2000 ms | **0.6250** | [0.5768, 0.6732] | 36/78/6 | 120 | ORDERED |
| 2000 -> 4000 ms | **0.5900** | [0.5473, 0.6327] | 20/78/2 | 100 | ORDERED |

No inversion anywhere, and every interval clears 0.5.

This was worth measuring rather than assuming. "More search is stronger" is
exactly the kind of proposition this project has already been burned by: an
800-simulation teacher produces *worse* distillation targets than a 50-simulation
one, which was equally obvious in the other direction beforehand. The ladder is
the ruler for everything that follows, so an unmeasured ordering would have
propagated into every later result silently.

**One doubling of thinking time is worth 0.59 to 0.69, decaying smoothly.** The
decay is the interesting part: 0.6875 -> 0.6583 -> 0.6250 -> 0.5900, roughly
-0.03 per doubling with no sign of a cliff. Extrapolating, the curve reaches
parity somewhere past a 100x budget -- so buying strength with raw clock keeps
working, it just gets expensive fast, and the draw rate rises to meet it (78 of
100 games at the top rung).

---

## What the clock actually buys

| budget | new sims | inherited | nn-evals | nn-evals/s | nn-evals / sim | p99 ms |
|---|---|---|---|---|---|---|
| 250 ms | 975 | 557 | 717 | 3,329 | 0.735 | 248.6 |
| 500 ms | 1,917 | 1,470 | 1,311 | 3,151 | 0.684 | 498.1 |
| 1000 ms | 3,896 | 3,444 | 2,475 | 2,997 | 0.635 | 998.4 |
| 2000 ms | 8,247 | 7,946 | 4,700 | 2,801 | 0.570 | 2000.2 |
| 4000 ms | 17,454 | 18,762 | 8,616 | 2,503 | 0.494 | 4003.2 |

Three things fall out of this table that were not the point of the run:

**1. Simulations scale slightly SUPER-linearly with budget** (1.97x, 2.03x,
2.12x, 2.12x per doubling) while network evaluations scale sub-linearly (1.83x,
1.89x, 1.90x, 1.83x). Those are not in tension: the ratio of evaluations to
simulations falls from 0.735 to 0.494, so at 4,000 ms **more than half of all
simulations never touch the network**. They terminate at already-expanded,
terminal, or *solved* nodes. Solved-node propagation is doing progressively more
of the work as the tree deepens, which is the behaviour it was added for.

**2. Network throughput DEGRADES with budget**, 3,329 -> 2,503 nn-evals/s, about
25% across the range. Bigger trees mean longer selection paths and more Python
bookkeeping per evaluation. Worth knowing before anyone reads a fixed-budget
throughput figure as a constant of the engine: it is a function of how long the
engine has been thinking.

**3. Inherited simulations overtake new ones at 4,000 ms** (18,762 vs 17,454).
The tree arrives carrying more search than the move itself adds.

Latency is honoured at every rung -- 248.6, 498.1, 998.4, 2000.2, 4003.2 ms
against nominal budgets of 250/500/1000/2000/4000. The exempt rungs overshoot by
0.2 ms and 3.2 ms, which is the reserve doing its job at budgets it was never
tuned for.

---

## The primary anchor

    final (1000 ms) vs anchor_C (2000 ms):   0.3750   [0.3268, 0.4232]
    final (1000 ms) vs anchor_B  (500 ms):   0.6583   [0.6052, 0.7115]

**`anchor_C` at 2,000 ms is the primary anchor.** It is the hardest rung that
still puts the deployment agent inside the 25-75% target band, which is the
selection rule that matters: an anchor is chosen for *headroom*, not for
proximity to 0.5. gregory(d4) died by saturating, and picking a rung the current
agent is already level with would repeat that within one improvement.

At 0.3750, a candidate has room to improve by more than 0.37 before `anchor_C`
stops discriminating. `anchor_D` (4,000 ms) is measured, ordered, and held in
reserve for when that happens -- validated *before* it is needed rather than
discovered to be broken at the moment it is first reached for.

`gregory(d4)` stays in the graph as a low anchor for absolute context. It is no
longer a discriminator.

---

## What this ladder is not

**It shares the training gene pool.** Every rung is the same gen-22 network, so
the ladder is a fixed, deterministic, reproducible reference -- which is what an
anchor must be -- but it is *not* an independent opinion about strength. A
candidate that improves by exploiting something gen-22 systematically
misunderstands could beat the whole ladder without being better in any absolute
sense. A genuinely external opponent remains wanted; nothing here supplies one.

**Anchors are frozen and must not be altered by the thing they measure.**
`engine:anchor_C+cpuct=2.0` is a hard error, and building an anchor whose engine
source bytes have drifted is a hard error too. If the sources move, re-run the
anchor from the tag:

    git worktree add ../uttt-anchor arena-1s-baseline

---

## Reproduce

    python -m tools.arena_1s --mode h2h --games 120 --warmup-games 2 \
        --player-a "engine:anchor_C" --player-b "engine:final" \
        --seed 6200 --gc deferred --tag ladder_C2000_final
    python -m tools.ladder_report

Total 7.1 hours on the reference box. `results/` is gitignored, so this file and
`tools/ladder_report.py` are the record.
