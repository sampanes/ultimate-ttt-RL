# RESULT -- 36 kernels a wave, 26 us to launch each, and the GPU idle for 79% of its own "busy" time (2026-08-09)

Task #40. Measurement only: nothing in `agents/mcts.py` changed, no candidate
was built, and no strength claim is made anywhere below.

`tools/profile_kernels.py`, `results/profile_kernels/kernels.json`,
`engine:pocket_r35` (fingerprint `d9769168cae6af7c`, 172,389 params, 1000 ms),
seed 6800. Six games of live play for the wave-size histogram, then 600 replays
of real production waves per size for the trace and the bench.

## Headline

| | k=1 | k=4 | k=8 |
|---|---|---|---|
| kernels per wave | 34.0 | 36.0 | 36.0 |
| mean kernel | 5.68 us | 7.13 us | 6.44 us |
| median kernel | 3.65 us | 3.87 us | 3.68 us |
| under 10 us, by count | 87% | 82% | 83% |
| under 10 us, by GPU time | 60% | 49% | 52% |
| kernel time, CUPTI | 193.1 us | 256.8 us | 231.8 us |
| memcpy time, CUPTI | 12.1 us | 11.4 us | 11.9 us |
| **device busy, CUPTI** | **205.2 us** | **268.2 us** | **243.7 us** |
| stream-elapsed, CUDA events | 1046.9 us | 1175.9 us | 1181.3 us |
| CPU wall for the wave | 1414.7 us | 1527.2 us | 1552.1 us |
| explicit synchronizations | 4.00 | 4.00 | 4.00 |

**36 kernels, not the ~20 that `RESULT_EXPAND_CUDA.md` estimated.** Split by
the statement that launches them: **33 in the network forward, 3 in masking
and softmax.** The count does not move with wave size -- k=1 and k=8 issue the
same work -- which is the dispatch-bound diagnosis restated at the kernel
level.

At k=8 the forward costs **849.7 us of CPU to issue 33 kernels: 25.8 us per
launch**, against kernels whose median device time is 3.68 us. The launch is
seven times the work.

## The GPU was never busy

The previous study reported "GPU busy 482.5 ms/move, of which 69% is compute"
from CUDA events. That number is stream-elapsed time -- from when the opening
event is processed to when the closing one is -- so it charges the GPU for
every microsecond it spent waiting for the next batch of commands. CUPTI times
the kernels themselves.

    device busy (CUPTI)          243.7 us/wave
    stream-elapsed (events)     1181.3 us/wave
    -> 79% of the "GPU time" is the device idle, waiting to be given work

At 293.7 waves/move that is **71.6 ms/move of real device work inside a ~913 ms
move: 7.8%**. The engine is not compute-bound and it never was; the earlier
figure was measuring the consequence of slow dispatch and reading it as
compute. The conclusion that survives is the one that matters -- the wave is
dispatch-bound -- and it survives more strongly than before.

The transfer claim collapses the same way. Measured DMA, per wave at k=8:

| copy | n | bytes | device time |
|---|---|---|---|
| HtoD (Pageable -> Device) | 2 | 18,792 | 5.11 us |
| DtoH (Device -> Pageable) | 2 | 2,624 | 3.78 us |
| DtoD (Device -> Device) | 1 | 2,592 | 3.02 us |

That is 5.5 MB up and 0.8 MB down per move, about 7 MB/s, roughly 0.06% of the
bus. The previous study's "H2D 81.4 + D2H 67.8 ms/move" was 149 ms/move of
stream-elapsed time around 3.4 ms/move of actual copying. **Do not optimize
transfers** remains correct, now by two orders of magnitude rather than one
argument.

The DtoD copy is new. Nothing in the source asks for it; it is the contiguity
copy behind `masked_fill` or `softmax`, and it costs more device time than
either real transfer.

## Every synchronization in the wave is accidental

Four per wave, and the search code requests none of them. Each is a
`cudaStreamSynchronize` that PyTorch emits behind a pageable memcpy:

| segment | kernels | copies | bytes | syncs |
|---|---|---|---|---|
| H2D: planes | 0 | 1 | 18,144 | 1 |
| network forward | 33 | 0 | 0 | 0 |
| H2D: mask | 0 | 1 | 648 | 1 |
| device: mask + softmax | 3 | 1 (DtoD) | 2,592 | 0 |
| D2H: probs | 0 | 1 | 2,592 | 1 |
| D2H: values | 0 | 1 | 32 | 1 |

## The mask path: it is induced synchronization, and the dtype theory was wrong

`RESULT_EXPAND_CUDA.md` guessed the bool dtype was missing a fast path. It is
not. Every arm below at k=8, 300 reps, stream drained before each call except
where marked:

| arm | us |
|---|---|
| host: `rule_utl_valid_moves` x k | 27.9 |
| host: `np.zeros((k,81), bool)` | 0.6 |
| host: zeros + python fill loop | 19.7 |
| host: `torch.from_numpy` (no copy) | 2.2 |
| upload: pageable bool `.to()` [drained] | 74.7 |
| upload: pageable **uint8** `.to()` [drained] | 70.4 |
| upload: pinned bool `.to()` [drained] | 46.1 |
| upload: pinned `.to(non_blocking)` [drained] | 16.2 |
| upload: `copy_` into a device buffer [drained] | 7.9 |
| build on device from a pinned index [drained] | 71.8 |
| upload: pageable bool `.to()` **[FORWARD IN FLIGHT]** | **207.9** |
| upload: pinned `.to(non_blocking)` [FORWARD IN FLIGHT] | 24.3 |
| upload: `copy_` into a device buffer [FORWARD IN FLIGHT] | 15.1 |

uint8 costs the same as bool, so dtype is not the mechanism. Construction is
19.7 us and allocation is 0.6. The statement costs 74.7 us drained and 207.9 us
with a forward queued, and **the 133.2 us difference is synchronization induced
by producing the mask**: the pageable H2D emits `cudaMemcpyAsync` and then
`cudaStreamSynchronize`, which waits on the entire forward pass launched three
statements earlier. Cross-check, measured independently: the forward's own tail
after its launch returns is 173.2 us (ratio 0.77 -- the same wait, two ways).

**This is not 133 us of recoverable wall clock.** The GPU is working for all of
it. Removing the sync moves the wait to the next blocking point; it pays only
if the host has real work to do meanwhile, or if the dispatch that made the GPU
slow goes away too. Labelling it as recoverable would be the same error as
labelling `.cpu()` wall time as transfer time, which this project has already
made once.

## Wave sizes: one graph covers the workload

86,358 waves over 294 moves of live play, 293.7 waves/move, mean k 7.60:

| k | waves | % of waves | % of leaves |
|---|---|---|---|
| 1 | 1,597 | 1.8% | 0.2% |
| 2 | 1,312 | 1.5% | 0.4% |
| 3 | 1,109 | 1.3% | 0.5% |
| 4 | 1,039 | 1.2% | 0.6% |
| 5 | 894 | 1.0% | 0.7% |
| 6 | 851 | 1.0% | 0.8% |
| 7 | 1,258 | 1.5% | 1.3% |
| **8** | **78,298** | **90.7%** | **95.4%** |

Median and p90 are both 8; the requested five measurement points collapse to
three. A single captured graph at k=8 covers 95.4% of all network evaluations,
so the graph-cache question answers itself -- and since the forward's cost is
nearly flat in k, padding short waves up to 8 is a live option that would need
no cache at all. Neither is decided here.

## CUDA graph: it captures, and the replay is measured

Every blocker from the brief, checked:

| blocker | finding |
|---|---|
| dynamic tensor shapes | real, but 90.7% of waves are one shape |
| allocations inside the region | conv/softmax outputs come from the graph's private pool at capture and are reused; legal, but the outputs are the SAME tensors every replay, so the consumer must copy out before the next one |
| CPU-dependent branches | `forward_both` branches on `x.dim()`, `value_tanh` and `policy.shape[0] == 1`; all fixed once the shape is, so all resolved at capture |
| changing memory addresses | inputs must be copied into static device buffers -- the wave cannot hand the graph a fresh tensor |
| unsupported ops | none: conv, linear, activation, `masked_fill`, softmax |
| hidden synchronizations | a pageable H2D or a `.cpu()` inside the region aborts capture; both are outside by construction |

Both variants captured. Numerics were verified on **40 distinct production
waves**, not one: max absolute difference from eager is **0.0** for both probs
and values, and all 40 outputs are distinct from each other. That second check
is the one that matters -- a graph captured around a stale buffer would replay
its captured input forever and still match perfectly on the wave it came from.

    region only        eager launch  901.5 us    graph replay   13.7 us
                       + drain       980.3 us    + drain       267.4 us
    with D2H captured into pinned host buffers   233.3 us end to end

**The whole wave, game states in and host arrays out**, same states, both
drained, profiler off:

    eager    1349.3 us
    graphed   398.5 us
    saved     950.9 us/wave  (70.5%)

At 293.7 waves/move that is **279.3 ms/move**, or **195.5 ms/move** after a 30%
efficiency discount for what a bench replaying production waves in a loop does
not carry: the engine's allocator pressure, its varying shapes, and the buffer
plumbing a real integration needs.

This is a bench. `agents/mcts.py` is untouched and none of it is wired into the
engine.

## The decision rule, evaluated

The rule was: **>=150 ms/move of graphable overhead and capturable -> graph
first, #36 stays blocked.** Measured 195.5 ms/move discounted, capture clean.
**Graph-first. #36 remains blocked.**

One thing had to be corrected to reach that number honestly. The obvious
reading -- sum the CPU column inside the device region -- gives 326.0 ms/move,
and it is wrong. Most of it is the host blocked on a pageable H2D while the GPU
genuinely works, and removing a block does not create wall clock. Scored that
way the first draft returned 461 ms/move on a wave that costs 1.35 ms in total.
The verdict turns on the end-to-end bench instead, because that is the only
measurement that nets the dispatch removed against the GPU time that remains.

For the record, the arithmetic the rule was written against:

| | measured | note |
|---|---|---|
| CPU inside forward + mask upload + softmax | 326.0 ms/move | NOT the recoverable figure |
| stream-elapsed in the same region | 236.9 ms/move | mostly idle device |
| real device work, whole wave | 71.6 ms/move | 7.8% of a move |
| **end-to-end saving, measured** | **279.3 ms/move** | 195.5 discounted |

## What this does NOT license

No strength claim. The model-size result is the standing reminder that
per-operation arithmetic does not survive contact with games, and this study
has just shown the previous one measuring the wrong device quantity. A 70%
faster wave is a throughput claim about a bench, and throughput has already
failed to predict strength once on this code. #41 is judged by win rate at
equal wall clock, by itself, or not at all.

Nor does it license bundling. Whatever lands first moves the bottleneck: remove
950 us/wave and the wave's remaining cost is host work the graph never touched
-- the ~195 ms/move of tree bookkeeping, the ~53 ms of `state.make_move`, the
child construction and the terminal probes. `_best_child` is still a real ~97 ms
target and it is still behind this one.

## Instrument notes

**Two overlapping runs corrupted the first result, silently.** A launcher's
child outlived the shell that was believed to have killed it, so two instances
of this tool ran concurrently and both wrote `kernels.json`. Neither crashed.
The k=8 kernel time differed by 1.9x between them because they were sharing the
GPU. Both were discarded and the study re-run alone. `acquire_lock()` now
refuses a concurrent start; a timing study that can be invalidated by a stray
process needs a lock, not a convention.

**Counts come from the annotated pass, timings never do.** Attributing kernels
to statements needs `record_function` brackets, and those are profiler events
themselves: the annotated pass reports 287.2 us of device time for the forward
against 231.8 us unannotated. Only counts are read off it. Relatedly, kernel
durations on a consumer card are not invariant -- they depend on the clock
state, which depends on how densely work arrives -- so a kernel time measured
under one dispatch rate does not transfer to another.

**CUPTI inflates every launch,** so no launch cost is read off the profiler.
The us-per-launch figure is the unprofiled segment time divided by the profiled
kernel count, and is stated that way wherever it appears.

**The replay is required to be the frozen sequence.** `agents/mcts.py` is
hash-gated, so `wave_sequence` is a copy, and `tools/test_profile_kernels.py`
requires it to produce bit-identical priors and leaf values against the frozen
`wave_planes` + `forward_both` + `_expand_wave` -- including the k=1 squeeze
branch, which is the one most likely to be got wrong. 27 tests.

**The box was not idle.** GPU load before the run: 31.6% mean, 38% peak, from
the desktop. Recorded in the result. The load-bearing comparisons here are
relative (eager against graphed, drained against in-flight) and both arms
carried it.

## Reproduce

    python -m tools.profile_kernels --mode all --engine pocket_r35 --games 6 --reps 600
    python -m tools.profile_kernels --rerender results/profile_kernels/kernels.json
    python -m tools.test_profile_kernels
