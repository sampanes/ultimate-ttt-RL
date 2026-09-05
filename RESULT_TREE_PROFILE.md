# RESULT -- where the 1,000 ms goes (2026-07-31, corrected 2026-08-07)

> ## RANKING SUPERSEDED 2026-08-27 -- see `CURRENT_STATE.md` section 4
>
> **Every number in this document was measured on `pocket_r35`, an engine that
> has since been superseded twice. Do not quote the ranking.** Four
> optimizations landed after it (CUDA-graph wave, native selection, deferred
> retirement, selective probes), each of which moved the composition of a move.
> The current ranking is wave-loop Python 90.4 ms, node creation 76.7,
> `state.make_move` 72.6, `_best_child` 32.2, terminal probes 23.5.
>
> Two conclusions here inverted specifically:
>
> - **"The network forward is the largest single cost at about 35%"** is no
>   longer true, and the device figure it rests on was itself an artifact --
>   see the next block, and section 5 of `CURRENT_STATE.md`.
> - **`_best_child` at 108 ms/move** rose to 193.9 on `pocket_graph` and then
>   fell to 32.2 once native selection landed. A *free* selection primitive is
>   now worth only +4.0% network evaluations.
>
> The warmup correction below still stands and is still worth reading -- it is
> a general instrument failure, not a detail of this profile.

> **CORRECTED 2026-08-07. Every figure below changed.** The first version of
> this profile counted the warmup games. `play_match(warmup=2)` discards them
> from the PLAYERS -- records, policies, the cumulative MCTS counters and the
> reuse tallies are all cleared by `reset_counters()` -- but every instrument
> in `tools/profile_tree.py` accumulates from outside the players and kept
> counting them, while the denominator (moves, sims) and the total they are
> expressed as a share of (`search_ms`) did not. Two games in fourteen is
> 16.7%. The tree share was overstated by about four points and the residual
> understated by nine. Fixed by a gate driven by `player.recording`; the test
> `TestWarmupGate` asserts the wrapped call count equals the recorded move
> count exactly. The retracted numbers are kept at the foot of this file.
>
> The load-bearing CAVEAT in the original -- that the device bucket was an
> UPPER BOUND until CUDA events separated compute from transfer -- has since
> been resolved. See `RESULT_EXPAND_CUDA.md`: it is real, it is mostly the GPU
> genuinely working, and it is not recoverable by restructuring transfers.

Profiled on `engine:pocket_r35`, the promoted deployment engine, in its shipping
configuration: reuse on, batched expansion, solved propagation, deferred GC,
1,000 ms. 595 moves of self-play, 4,929 simulations per move.

**The tree is 23% of a move. The network forward is the largest single cost at
about 35%. Host/device traffic and expansion is 35%.**

| category | ms/move | share |
|---|---|---|
| residual: network forward + `make_move` | 351.3 | 41.7% |
| device traffic + expansion (MIXED) | 296.0 | 35.1% |
| **pure-Python tree bookkeeping** | **194.9** | **23.1%** |
| total search | 842.2 | 100% |

Per operation, exclusive of nested operations:

| operation | ms/move | us/sim | category |
|---|---|---|---|
| expansion (`_expand_wave`) | 254.7 | 51.67 | mixed |
| child scoring / best-child | 108.0 | 21.90 | tree |
| proof: terminal probes | 51.2 | 10.38 | tree |
| plane build + H2D | 41.3 | 8.38 | mixed |
| selection: state clone | 21.3 | 4.32 | tree |
| legal-child iteration | 6.3 | 1.27 | tree |
| tree release | 5.7 | 1.16 | tree |
| backup traversal | 2.0 | 0.40 | tree |
| proof: backward induction | 0.4 | 0.08 | tree |
| proof: propagation | 0.0 | 0.01 | tree |
| tree reuse: adopt | 0.0 | 0.00 | tree |

`expansion` and `plane build` are deliberately NOT summed into the tree.
`_expand_wave` is dominated by a device round trip -- a mask H2D, a softmax,
and two D2H pulls -- with only the node-allocation loop inside it being Python.
`wave_planes` is a C++ fill plus an H2D copy. Adding them to a "tree total"
would attribute ~300 ms of CUDA traffic to a port that cannot touch it, which
is the exact mistake this profile exists to prevent. An earlier draft of this
file made it. `RESULT_EXPAND_CUDA.md` has since decomposed that bucket
directly, and confirms the split: of `_expand_wave`'s cost only 37.2 ms/move is
child construction, the rest is device calls.

Backing out `state.make_move` leaves the network forward at roughly **298 ms,
35%** -- measured independently at 297.9 ms/move by the CUDA study, against
351.3 - 53.4 here. That is the largest single line item in a move.

---

## The first instrument was wrong, and the way it was validated was wrong

The first version of this profile was produced by a stack sampler and said the
tree was **0.0%** -- `best_child` 0.4 ms, backup 0.0 ms, proof propagation
absent entirely -- with plane build at 30.5% and the network at 34%.

Every one of those numbers was an artifact. An in-process sampler can only take
a sample while it holds the GIL, so it samples freely whenever the main thread
has released it (CUDA syncs, pybind crossings) and must wait out the switch
interval during pure Python. Measured against a workload constructed to be an
exact 50/50 split between a pure-Python spin and a GIL-releasing block:

    ground truth 50 / 50        sampler reported 13.5 / 86.4

A **6.4x bias toward C**, pointing directly at the answer, because every tree
operation is pure Python and everything it competes with releases the GIL.

The reconciliation pass caught it -- counts x isolated cost disagreed with the
sampler by 353x on `best_child` and 221x on backup -- which is the entire
reason that pass exists and the argument for never trusting one instrument.

**The deeper failure was the validation, not the sampler.** The sampler had a
ground-truth test: a workload with a known 75/25 split, recovered correctly. But
both arms of it were pure Python, so it was structurally incapable of detecting
a Python-versus-C bias. A calibration workload has to span the axis the
instrument is used across. That test now lives in the suite asserting the bias
*still exists*, so the sampler cannot be trusted by accident.

The sampler is kept. Its ranking within a single regime is sound and its
line-level detail inside `_expand_wave` is what identified the device round trip
as that function's dominant cost. It must never compare Python against C.

The replacement is `--mode wrap`: exclusive wall time per operation, no GIL
bias, overhead priced by calibration (0.724 us/call over 37.9M calls = 46.1
ms/move) and subtracted before reporting.

---

## The warmup was in the numerator and not in the denominator

Found 2026-08-07, after this file was published. `play_match(warmup=N)` plays N
games for real and then throws them away -- but only from the players. Look at
what it clears:

    if not is_warmup and p.recording is False:
        p.reset_counters()          # records, policies, MCTS counters, reuse

Every instrument in `tools/profile_tree.py` -- the sampler, the exclusive
timer, the operation counters -- accumulates from OUTSIDE the players, and none
of them was told. So each wrapped operation carried 14 games of cost and was
divided by 12 games of moves, while `search_ms`, the total those costs are
expressed as a share of, was summed over the 12. The priced wrapper overhead
subtracted from it had the same problem in the other direction.

The fix is a shared context whose gate follows `player.recording`, which is the
same flag `play_match` uses to decide whether a move counts -- so the
instruments and the denominator cannot disagree about which moves are in the
measurement. `TestWarmupGate` asserts equality, not a tolerance: the warmup
must reach the instrument (proving the gate is doing something) and contribute
exactly zero to the total.

---

## What this means for the native-tree program

The case for the port was built on an inference: a 39x parameter cut bought only
1.24x more search, so ">=87% of a simulation is not the network". **That
inference was wrong, and `RESULT_EXPAND_CUDA.md` explains why:** the forward
pass costs what it costs because of how many kernels it launches, not how many
parameters they touch, so cutting parameters 39x left its wall time almost
unchanged. The network is the largest single line item in a move, at ~35%.

**The tree is worth 23%, and 55% of that is two operations:** `_best_child`
(108.0 ms) and the solved-node terminal probes (51.2 ms). A perfect port of
those two -- to zero cost, which no port achieves -- buys 159 ms, or 19% more
thinking time. By the anchor ladder, 19% more clock is worth about +0.03.
Real-world, a good port might return half of that.

**The device bucket is NOT the cheaper target after all.** That was this file's
recommendation and the measurement overturned it. 296 ms/move of expansion is
dominated by per-call dispatch cost, not by bytes: a 6.25x change in batch size
moves the forward pass by 8% and the transfers by nothing. Restructuring
transfers cannot recover it; only issuing fewer device operations can, and the
candidate that matches the diagnosis is CUDA graphs. See `RESULT_EXPAND_CUDA.md`
for the full decomposition and the two clear targets.

### Recommended order

1. ~~Measure sync versus transfer with CUDA events.~~ **DONE**, 2026-08-07 --
   `RESULT_EXPAND_CUDA.md`. Compute-bound, 69% of GPU busy time. Transfers are
   not the lever.
2. Count the kernels in one forward pass and prototype a CUDA-graph capture of
   the forward plus the mask and softmax. Cheap, and it is the intervention the
   diagnosis points at. Also price the mask upload, which costs 111 ms/move of
   CPU for 648 bytes.
3. Then the tree port, scoped to `_best_child` plus the terminal probes only --
   not "selection plus backup and compact node storage". Backup is 2.0 ms, 0.2%,
   and porting it would be motivated by intuition rather than measurement.

No strength claim is projected from any of these speedups. The model-size
experiment is the standing reminder: 1.24x more search was worth +0.039, and
the arithmetic that predicted otherwise was wrong on its own stated terms --
and we now know exactly which term was wrong.

## Reproduce

    python -m tools.profile_tree --mode wrap --engine pocket_r35 --games 12
    python -m tools.profile_tree --mode all  --engine pocket_r35 --games 12

---

## Retracted figures, kept for provenance

Published 2026-07-31, warmup-contaminated, superseded by the table at the top:

    device traffic + expansion   328.4 ms/move  40.5%
    network forward + make_move  265.1 ms/move  32.7%
    pure-Python tree             218.1 ms/move  26.9%
    total search                 811.6 ms/move

    expansion 283.8, best_child 119.0, terminal probes 58.7, plane build 44.6,
    clone 24.0, legal-child 7.0, release 6.6, backup 2.3

The RANKING was unaffected -- every operation was inflated by the same warmup
fraction -- so "port `_best_child` plus the probes, not backup" stood. The
SHARES were not, and the residual was understated by nine points, which is what
made the network look cheaper than the tree.
