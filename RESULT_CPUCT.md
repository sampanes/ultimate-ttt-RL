# RESULT -- c_puct under the 1,000 ms deadline is a NULL (2026-07-31)

**Keep `c_puct = 1.5`.** Measured on `pocket_r35`, the promoted deployment
engine, at its real operating point of ~4,700 root visits per move. The
incumbent versus the best challenger over 240 head-to-head games on held-out
openings: **0.5167 [0.4792, 0.5541]**. The interval contains 0.5.

This was worth asking rather than assuming, because the previous c_puct null
was measured at 50 simulations and the operating point has since moved by two
orders of magnitude. The exploration term is scaled by `sqrt(N_parent)` and
divided by `1 + N_child`, so the balance it strikes at 2,200 visits is not
mechanically the balance it strikes at 4,700. It turns out to be the same
answer anyway.

---

## Round 1: elimination, 48 games per value vs the frozen anchor

Seed 6400 (`tune`), every arm on the same openings against `engine:anchor_C`.

| c_puct | score | 95% CI | W/D/L | latency |
|---|---|---|---|---|
| 0.5 | 0.4896 | [0.3711, 0.6081] | 16/15/17 | PASS p99 989.2 |
| **1.0** | **0.5521** | [0.4275, 0.6767] | 21/11/16 | PASS p99 988.0 |
| 1.5 (incumbent) | 0.4792 | [0.3590, 0.5993] | 16/14/18 | PASS p99 985.5 |
| 2.0 | 0.4688 | [0.3660, 0.5715] | 11/23/14 | PASS p99 985.9 |
| 3.0 | 0.4167 | [0.3024, 0.5309] | 12/16/20 | PASS p99 985.6 |

**Nothing was eliminated.** The elimination bar was fixed before the arms were
read: a value had to trail the best by more than 0.167, which is the gap two
independent 48-game scores need before they separate. The full spread is 0.135.

The design is paired, not independent -- every arm faced the same anchor on the
same openings -- so the correct comparison is paired, and the pre-committed bar
was derived from the wrong variance. Recomputed properly it changes nothing:

| pair | paired diff | paired SE | t | correlation |
|---|---|---|---|---|
| 1.0 - 3.0 | 0.1354 | 0.0755 | 1.79 | 0.234 |
| 1.0 - 2.0 | 0.0833 | 0.0688 | 1.21 | 0.310 |
| 1.0 - 1.5 | 0.0729 | 0.0729 | 1.00 | 0.319 |
| 1.5 - 2.0 | 0.0104 | 0.0605 | 0.17 | 0.444 |

The largest statistic across ten comparisons is 1.79. Arm-to-arm correlation is
only 0.23-0.44, so **pairing buys much less here than the design suggests** --
the paired SE is barely below the unpaired one. Two configurations of the same
engine, on the same opening, against the same opponent, still produce largely
independent games. That is worth knowing before designing the next sweep.

---

## Round 2: the powered head-to-head, and it flips sign

Seed 6500 (`confirm`), openings round 1 never touched, 240 games, played
directly rather than each against a third party.

    engine:pocket_r35  (c_puct 1.5)  vs  engine:pocket_r35+cpuct=1.0
    score for the incumbent  0.5167 [0.4792, 0.5541]   W46/D156/L38

**Round 1 put 1.0 ahead of 1.5 by +0.073. Round 2 puts 1.5 ahead of 1.0 by
+0.017.** The sign reversed. That is exactly what regression to the mean looks
like when the first round was noise, and it is the clearest possible
demonstration that the elimination round should not have been read as a
ranking -- which is why it wasn't.

It is also why round 2 was the incumbent against the challenger rather than the
best against the worst. A match between two non-incumbent values could not have
answered the only operational question here: does the deployed value change?
Under a null, the two extremes are just the luckiest and unluckiest draws.

The interval is tighter than the pre-run estimate of +/-0.045 because 156 of
240 games were drawn. Two near-identical configurations of the same engine draw
65% of the time; against `anchor_C` in round 1 the draw rate was 29%.

---

## Decision

**c_puct stays at 1.5.** No value tested beats it at a resolution worth acting
on, and the default when a sweep returns a null is to change nothing.

No dynamic schedule was tried, per the stated ordering: the best fixed value
comes first, and there is no evidence here that the fixed value is the binding
constraint.

An effect too small for 240 games to resolve is below +/-0.037. Buying it would
cost hours per additional halving of the interval, against a tree-core program
that the model-size result says is worth roughly three quarters of every
simulation. This is not where the box should go.

---

## A free result

**All seven arms passed the latency requirement**, p99 983.2 to 989.2 ms, zero
moves over budget in round 2. Five independent 48-game samples plus two
240-game samples, across a 6x range of c_puct.

The reserve fix from `RESULT_MODEL_SIZE.md` was sized on one configuration; it
holds across the whole sweep. `pocket_r35` has now played 480 games against a
2,000 ms opponent and 480 against itself without a single move over 1,000 ms --
a stronger deployment signal than the sweep it came from.

## Reproduce

    bash run_cpuct.sh pocket_r35 anchor_C          # round 1, ~3.9 h
    bash run_cpuct_final.sh pocket_r35 1.5 1.0     # round 2, ~2.6 h
