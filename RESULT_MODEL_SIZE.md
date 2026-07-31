# RESULT -- 172k vs 6.77M under the 1,000 ms deadline (2026-07-30)

**The 172,389-parameter network wins at equal wall clock: 0.5854 [0.5360,
0.6349] over 240 games.** The interval excludes 0.5. It is 39x smaller than the
6,766,386-parameter gen-22 network it beat.

Pre-registered in `EXPERIMENT_MODEL_SIZE.md` before the first game, including
the prediction, the decision rule, and three ways the prediction could be
wrong. One of those three fired. Read this against that file, not instead of it.

    engine:pocket    172,389 params, squeeze,  1000 ms, final engine
    engine:final   6,766,386 params, arena22,  1000 ms, final engine
    seeds 6300 (`anchor` namespace) throughout, paired openings, colours swapped

---

## The decision match

| | score for pocket | 95% CI | W/D/L | n |
|---|---|---|---|---|
| **pocket vs final**, both 1,000 ms | **0.5854** | [0.5360, 0.6349] | 97/87/56 | 240 |
| pocket vs anchor_C (2,000 ms) | 0.5458 | [0.4763, 0.6154] | 42/47/31 | 120 |

For comparison, `final` scores **0.3750** [0.3268, 0.4232] against the same
anchor. The small network at 1,000 ms is at parity with the gen-22 engine given
twice the clock.

**Promotion follows from the first row alone**, as fixed in advance. The second
row confirms the winner keeps its place on the ladder rather than justifying
anything.

---

## The prediction was right about the outcome and wrong about the mechanism

The pre-registration predicted 0.55-0.70, and 0.5854 is inside it. That is not
a vindication, because the stated reasoning was "raw-network parity plus search
the large network cannot afford", and the second half of that did not happen.

Failure mode 1, quoted from the pre-registration:

> Params are not latency [...] If the small net buys only 1.2x, there is no
> doubling to collect and the ladder arithmetic above does not apply.

Measured in the decision match itself, over 5,517 and 5,501 moves:

| | pocket 172k | final 6.77M | ratio |
|---|---|---|---|
| simulations / move | 4,193.1 | 3,389.4 | **1.24x** |
| network evals / move | 2,882.3 | 2,466.6 | 1.17x |
| simulations / s | 5,191.1 | 4,149.6 | 1.25x |
| inherited sims / move | 1,945.8 | 1,453.4 | 1.34x |
| early stop rate | 0.192 | 0.183 | -- |
| tree reuse | 0.956 | 0.955 | -- |

**1.24x, not 2x.** By the anchor ladder a doubling of search is worth 0.125
above parity, so 1.24x (0.31 of a doubling) is worth about +0.039 -- less than
half the +0.0854 observed. Removing 97.5% of the parameters removes about a
quarter of the per-simulation cost, because the rest is not the network.

(The isolated 12-game benches gave 1.13x and even showed *fewer* network
evaluations for the small net. The 5,517-move match is the number to trust;
276 moves is not enough to rank throughput.)

Failure mode 2 did **not** fire:

> Raw-argmax parity is a POLICY claim; MCTS also leans on the value head. A
> distilled student can match its teacher's policy argmax while carrying a
> worse-calibrated value head.

If the student's value head were worse under search, the complete agent would
have underperformed its raw parity. It did the opposite.

---

## The networks alone

Search was removed entirely (`sims=0`, masked policy argmax -- not a 1-sim
search, which is a different thing and agrees with the policy argmax only 0.197
of the time).

| arm | score | 95% CI | W/D/L | n |
|---|---|---|---|---|
| pocket_raw vs gen22_raw | 0.5175 | [0.4605, 0.5745] | 71/65/64 | 200 |
| pocket_raw vs gregory(d4) | 0.6700 | [0.5922, 0.7478] | 54/26/20 | 100 |
| gen22_raw vs gregory(d4) | 0.6250 | [0.5470, 0.7030] | 47/31/22 | 100 |
| pocket_raw vs anchor_A (250 ms) | 0.0833 | [0.0461, 0.1206] | 2/16/102 | 120 |
| gen22_raw vs anchor_A (250 ms) | 0.0833 | [0.0461, 0.1206] | 2/16/102 | 120 |

The head-to-head interval includes 0.5: **parity, not an edge.** The two
identical 0.0833 scores are not a configuration bug -- the checkpoints,
fingerprints and per-game outcomes all differ (26 of 120 games have different
results; pocket won games 47 and 106, gen22 won 76 and 105). It is a ceiling
effect: a raw network is crushed by even a 250 ms search, and that anchor
cannot resolve two raw networks. gregory(d4) can, and mildly favours pocket.

So neither factor alone accounts for the win. Roughly +0.039 is attributable to
the extra search, the raw network is at worst equal, and the remainder is
inside the resolution of this arithmetic. What can be said cleanly is the thing
that matters for deployment: **at a fixed 1,000 ms the small network is
stronger, and it is not stronger merely because it searched more.**

---

## It does not sit where transitivity says it should

Composing the two measured results through Elo predicts pocket vs anchor_C at
0.4585. Observed 0.5458 [0.4763, 0.6154] -- the prediction falls below the
interval.

Two candidates, not separated here. The opening sets differ (`final` vs
anchor_C was measured on seed 6200, this on 6300). And this project has already
recorded real non-transitivity of exactly this shape: distilled students score
*below* their teacher on a fixed heuristic panel while beating it head to head
0.545 every time. `pocket` is a distilled student of the network inside
anchor_C, so the same effect appearing here is unsurprising.

Either way it is a reminder that the ladder shares the gen-22 gene pool and is
a fixed reference, not an independent opinion about strength.

**Anchor headroom.** `anchor_C` puts the new deployment agent at 0.5458 rather
than the 0.3750 it gave `final`, so it still discriminates but with roughly
half the room. It is adequate for the c_puct sweep. `anchor_D` (4,000 ms) is
already validated and in reserve for when it is not.

---

## Latency: pocket does not meet the frozen requirement as configured

| | move p99 | move max | search p99 | overhead p99 | verdict |
|---|---|---|---|---|---|
| pocket (step 5) | 1002.34 | 1014.7 | 980.03 | 23.78 | **FAIL** |
| final (step 5) | 997.75 | 1010.0 | 979.97 | 19.17 | PASS |
| pocket (step 6) | 1001.5 | 1580.3 | -- | -- | **FAIL** |

106 of 5,517 moves over budget in the decision match. **The search is
blameless**: across those 5,517 moves it never exceeded 981.1 ms against its own
980 ms deadline, and on every one of the 106 failures the worst chunk was
around 3.3 ms. All of the overshoot is `move_ms - search_ms` -- re-rooting and
`release()` of the discarded tree, inside the move the requirement is written
against, outside the interval the search times itself over.

`pocket` pays more of it for the same reason it is faster: 4,193 simulations
per move builds a 24% bigger tree to walk. The reserve was sized for chunk
overrun (~6 ms) and never for this (~24 ms), which means `final` has been
passing on a 0.8 ms margin all along.

The fix is configuration, not code: `pocket_r35` (fingerprint
`d9769168cae6af7c`) is `pocket` with the reserve raised 20 -> 35 ms, covering
the measured overhead p99 plus chunk p99 with room, at 1.5% of thinking time --
worth about 0.002 by the ladder against an 0.0854 edge. `pocket` is left
frozen exactly as measured, failure included, because it is the arm that scored
0.5854.

### A contamination that must be stated

Both matches ran while this session executed CPU-only test suites and
checkpoint loads on the same machine. Step 5's 5,517-move sample shows no chunk
above 27.1 ms. Step 6 shows three above 50 ms -- 1,225.8, 742.2 and 97.0 --
**all within the first 17% of the run**, which is exactly the window that
concurrent work occupied. The 1,580 ms move and the 1,225 ms chunk are
therefore not attributed to the engine.

### The clean-box re-measurement, and the opponent effect it exposed

Re-measured on an idle box, 40 games per arm, varying one thing at a time:

| run | opponent | box | inherited/move | overhead p99 | move p99 | verdict |
|---|---|---|---|---|---|---|
| pocket | itself | idle | 3,707.0 | 18.77 | 996.9 | PASS, +1.23 ms |
| pocket | final | idle | 2,223.6 | 23.06 | 1000.7 | **FAIL** |
| pocket | final | contended | 1,945.8 | 23.78 | 1002.34 | **FAIL** |
| pocket_r35 | itself | idle | 4,044.9 | 18.28 | 982.0 | PASS, +16.72 ms |
| **pocket_r35** | **final** | **idle** | **2,321.9** | **24.12** | **988.2** | **PASS, +10.88 ms** |

**The opponent accounts for +4.29 ms of overhead p99; contention for +0.72 ms.**
So the contamination I flagged was real but nearly irrelevant to the p99 -- it
does still explain step 6's 1,225 ms chunk, since the idle runs cap out at
7.6 ms. The failure is a property of the engine, and it reproduces.

The mechanism is inheritance. A search predicts its own replies far better than
a different network's, so it keeps a much larger subtree and `release()` walks
correspondingly fewer nodes. Measured across all four runs, overhead tracks
`total tree - kept subtree` rather than tree size alone.

**This is a defect in the regression gate, not just a fact about pocket.** The
gate as first written played a MIRROR, and a mirror is the single most
favourable opponent in existence for tree reuse. It passed `pocket` at 996.9 ms
-- an engine that fails at 1000.7 ms against a real opponent. The same effect
distorts the frozen baseline: `final` inherits 3,149.5 simulations per move
against `original` (same network) and 1,551.9 against `pocket`.

Fixed: the gate now derives its opponent from the CHECKPOINT and refuses a
mirror unless explicitly asked. Deriving it from the engine name would have
left `original` and every anchor rung facing a same-network opponent, which is
the same defect wearing a different hat. The inherited-simulation floor moves
0.50 -> 0.35, because 0.50 was calibrated on the same-network case (0.81) and
fails the cross-network one (0.48) for an engine that is behaving correctly.

**`pocket_r35` passes against the real opponent with room: p99 988.2, max
992.3, zero moves over budget, 10.9 ms of reserve margin.** That is the
deployment configuration.

---

## What follows

* **`pocket_r35` is the deployment engine** -- the 172k network at 1,000 ms
  with a 35 ms reserve. Stronger than the 6.77M incumbent at equal wall clock
  (0.5854) and inside the frozen requirement against a real opponent (p99
  988.2). Its artifact is 0.70 MB against 27 MB, so the browser deployment
  inherits a model that is both 39x smaller and stronger -- the
  "Hard"/"Impossible" ladder currently runs the 6.77M champion.
* **No intermediate size.** The pre-registration reserved `midsize` (921,026
  params) for the case where the small net LOST despite more search. It won, so
  that arm is not run.
* **c_puct is tuned on `pocket_r35`**, not on `final` and not on both.
* **The tree is the next engineering target, and this result is the third
  independent argument for it.** Stripping 97.5% of the parameters bought 1.24x
  search, so roughly three quarters of a simulation is tree bookkeeping; and
  the deployment latency failure is itself a tree walk, on a path the search
  cannot even see.

## Reproduce

    bash run_modelsize.sh anchor_C          # all six steps, ~5.5 h
    python -m tools.regress_engine --engine pocket_r35 --games 40
    python -m tools.arena_1s --mode h2h --games 40 --warmup-games 2 \
        --player-a "engine:pocket_r35" --player-b "engine:final" \
        --seed 6300 --gc deferred --tag lat_pocket_r35_vs_final

`results/` is gitignored, so this file is the record.
