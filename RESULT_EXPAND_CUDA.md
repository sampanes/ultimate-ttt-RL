# RESULT -- the expansion wave is dispatch-bound (2026-08-07)

`engine:pocket_r35`, the deployment engine, in its shipping configuration:
reuse on, batched expansion, solved propagation, deferred GC, 1,000 ms. Two
independent instruments over 12 games each; seed namespace `expand` (6700).
No score is read off these runs.

**The 328 ms device bucket is real, almost none of it is bytes, and it is
per-CALL cost that does not care how much data moves or how big the network
is.** The upper bound stated in `RESULT_TREE_PROFILE.md` is resolved:
restructuring the transfers cannot recover it.

> ### CORRECTED 2026-08-09 by `RESULT_KERNEL_TRACE.md` -- read that first
>
> The headline conclusion stands and is stronger: the wave is dispatch-bound.
> Three sub-claims below are wrong, and one guess was refuted.
>
> **"Mostly the GPU genuinely working" is wrong, and so is every GPU column in
> this document.** A CUDA event pair spans from when the opening event is
> processed to when the closing one is, so it charges the GPU for the time it
> spends waiting to be given work. CUPTI times the kernels. Measured at k=8:
> **243.7 us/wave of real device busy inside 1,181.3 us of stream-elapsed
> time -- 79% of the "GPU time" is an idle device.** Real device work is
> ~71.6 ms/move, 7.8% of a move, not 482.5 ms and 53%.
>
> **The transfer rows overstate by ~40x.** Actual DMA is 5.11 us/wave up and
> 3.78 us/wave down (18,792 and 2,624 bytes), about 3.4 ms/move against the
> 149.2 ms/move claimed here. "Do not optimize transfers" was the right call
> for a weaker reason than the one given.
>
> **"Roughly twenty kernels" is 36** -- 33 in the forward, 3 in mask+softmax --
> and there is a fifth copy nobody counted, a 2,592-byte device-to-device
> contiguity copy that costs more device time than either real transfer.
>
> **The mask upload is not a bool-dtype slow path.** uint8 measures the same.
> It is synchronization induced by a pageable H2D: `cudaMemcpyAsync` followed
> by `cudaStreamSynchronize`, waiting on the forward launched three statements
> earlier. Nor is the 86 ms/move of "host overhead" simply recoverable -- the
> GPU is busy for it.
>
> What survives unchanged: batch-size invariance, the per-call diagnosis, the
> reconciliation with the model-size result, and the ordering advice that put
> the device path ahead of the native tree. Ceiling B was projected here at
> 200-230 ms/move and measured there at 195.5.

## Where the wave goes

Scaled to a move by the UNTOUCHED build's 2,559 network evaluations per move.

| quantity | ms/move |
|---|---|
| GPU compute (forward + softmax) | 333.3 |
| GPU transfer H2D | 81.4 |
| GPU transfer D2H | 67.8 |
| **GPU busy, total** | **482.5** |
| GPU span, first record to last | 686.0 |
| GPU idle inside the wave | 203.5 |
| CPU in the wave, total | 702.6 |
| CPU host work (no device involved) | 146.8 |
| CPU inside the two `.cpu()` calls | 92.1 |
| &nbsp;&nbsp;of which waiting | 48.6 |
| &nbsp;&nbsp;of which copying | 45.3 |

GPU compute is **69%** of GPU busy time. The two `.cpu()` calls are 52%
waiting and 48% copying -- so `.cpu()` is indeed largely the point where the
forward pass becomes visible, and its wall time is not transfer time.

The two instruments agree to **1.02**: the events build measures the two
`.cpu()` calls as one blocked interval (92.1 ms/move); the sync build drains
the stream first and splits the same interval into a wait plus two drained
copies (48.6 + 45.3 = 93.9 ms/move). They are the same physical time reached
two different ways.

Per segment, from the events build:

| segment | kind | cpu us/leaf | gpu us/leaf | cpu ms/mv | gpu ms/mv |
|---|---|---|---|---|---|
| host: plane fill | host | 4.37 | -- | 11.2 | -- |
| H2D: planes | copy | 12.25 | 22.03 | 31.3 | 56.4 |
| network forward | gpu | 116.43 | 120.91 | 297.9 | 309.3 |
| host: legal masks | host | 10.21 | -- | 26.1 | -- |
| H2D: mask | copy | 43.50 | 9.79 | 111.3 | 25.0 |
| device: mask + softmax | gpu | 9.06 | 9.36 | 23.2 | 24.0 |
| D2H: probs | copy | 26.79 | 18.03 | 68.5 | 46.1 |
| D2H: values | copy | 9.21 | 8.46 | 23.6 | 21.6 |
| host: child construction | host | 14.54 | -- | 37.2 | -- |
| host: make_move / probes | host | 28.25 | -- | 72.3 | -- |
| **total in the wave** | | **274.63** | **188.58** | **702.6** | **482.5** |

---

## The decisive measurement: device cost is per WAVE, not per leaf

The search hands the network wildly different batch sizes depending on the game
stage -- 8 in the opening, about 1.3 in the endgame, because late positions have
few legal moves and most descents land in already-proven subtrees. That is a
natural 6.25x experiment in batch size, already inside the data:

| GPU us per wave | early (k=8.00) | mid (k=6.25) | late (k=1.28) |
|---|---|---|---|
| H2D: planes | 136.3 | 142.1 | 167.9 |
| network forward | 777.7 | 791.6 | 711.8 |
| H2D: mask | 62.9 | 63.5 | 61.9 |
| device: mask + softmax | 61.9 | 61.5 | 46.8 |
| D2H: probs | 116.1 | 116.6 | 115.5 |
| D2H: values | 54.3 | 54.7 | 54.6 |

**A 6.25x change in batch size moves the forward pass by 8%, and moves the
transfers by nothing at all.** The CPU side says the same: launching the
forward costs 752.6, 752.6 and 746.0 us per wave across the three stages.

So the device time is a fixed toll per host/device operation. It is not
arithmetic -- a 56-channel 4-layer convolution over 9x9 at batch 8 is a couple
of megaflops, microseconds of real work on a 3080. It is not bytes -- the whole
plane buffer is 18 KB and the mask is 648. It is the cost of issuing the calls,
paid on both sides of the boundary, and on Windows/WDDM that toll is large.

### This is why the 39x parameter cut bought only 1.24x more search

`RESULT_MODEL_SIZE.md` recorded that a 39x smaller network bought 1.24x more
simulations, and the conclusion drawn was that ">=87% of a simulation is not
the network". The premise was measuring the wrong axis. Cutting parameters does
not cut the number of kernels launched, and the number of kernels is what the
forward pass costs here. The network's *wall time* barely moved because nothing
that sets its wall time changed. Both results are now consistent, and neither
supports "the network is cheap".

---

## Cross-check against the tree profile

`RESULT_TREE_PROFILE.md` measures the same engine with a completely different
instrument -- exclusive wrapper timing around whole functions, no CUDA events,
no replicas, a different run on different games. Where the two overlap they
should agree, and they do:

| quantity | tree profile | CUDA study |
|---|---|---|
| `wave_planes` (fill + H2D) | 41.3 | 42.5 |
| network forward | 351.3 *(incl. `make_move`)* | 297.9 |
| `_expand_wave`, exclusive | 254.7 | 283.7 |
| terminal probes | 51.2 | 72.3 |

Summed, the two instruments account for **95.6%** of the 912.9 ms/move of
search the untouched build measured. `wave_planes` agrees to 3%, and the
forward's residual implies `state.make_move` at 53.4 ms/move, which is what a
C++ engine doing ~35,000 descent steps should cost.

The two loose rows are expansion (11%) and the terminal probes (41%). Both are
proportional to how many nodes get expanded and how branchy the positions are,
both runs played different games, and the CUDA column is additionally scaled by
a reference leaves-per-move taken from a third arm. The probes row is the one to
distrust if it ever matters; it does not matter here, because nothing in the
conclusion turns on it.

---

## What follows, against the pre-registered decision tree

* **"If GPU event time accounts for most of the 328 ms, the block is
  compute-bound. Do not optimize transfers first."** -- FIRES. 69% of GPU busy
  time is the forward plus the softmax. Transfer volume is not the lever, and
  pinned host memory (candidate 5) is not indicated: the copies are 149 ms/move
  of GPU time for 19 KB per wave, so they are latency, not bandwidth.
* **"If synchronization wait is large but GPU compute is modest"** -- does not
  fire. The wait is 48.6 ms/move, 5% of the move.
* **"If H2D or D2H transfer itself is large, reduce round trips"** -- fires in
  the COUNT sense, not the volume sense. Removing a call is worth its whole
  fixed toll no matter how few bytes it carried.
* **"If CPU child construction after the transfer is large"** -- does not fire.
  37.2 ms/move.
* **"If many small batches dominate"** -- 70.6% of waves are the full k=8 and
  11.5% are k=1. The small ones are the endgame, where the tree is mostly
  proven. This does NOT revive the inference server: batching more leaves per
  call cannot help a cost that is independent of how many leaves the call
  carries. It is an argument for fewer CALLS, which is a different change.

### The one clearly addressable item

**The mask upload costs 111.3 ms/move of CPU and 25.0 ms/move of GPU, for 648
bytes.** It is the only segment where the CPU cost is more than four times the
device cost -- 43.50 us/leaf against 9.79 -- which is 86 ms/move of pure
host-side overhead around a copy that moves nothing. Contrast the plane upload:
28x more data, and its CPU cost (12.25 us/leaf) is *below* its device cost, as
an asynchronous launch should be. Something about this particular call, most
likely the bool dtype, is not taking a fast path.

Candidates 1 and 3 from the brief both attack it -- keep the mask on device,
or fuse the masking into the softmax so no mask crosses at all. Neither is
claimed to work here. The measurement says only that 111 ms/move is spent on it
and that ~86 ms/move of that is not the transfer.

### The other candidate the data points at

Every device segment being a fixed toll per call means the total toll scales
with the NUMBER of operations issued per wave, and there are six CPU-visible
ones plus roughly twenty kernels inside the forward. Collapsing the forward
into a single replayable unit is the intervention that matches the diagnosis:
**CUDA graphs**, which capture a fixed sequence once and replay it with one
dispatch. It needs no Triton, so the standing "no `torch.compile` on Windows"
constraint does not apply. Kernel count has NOT been measured yet; that is the
cheap next step and it should precede any implementation.

---

## Ceiling comparison

Milliseconds only. These are not converted to win rate anywhere, and they are
projections, not measurements -- the whole reason the model-size experiment is
the standing reminder is that arithmetic like this was wrong before.

Counts, from `--mode count`: 35,101 `_best_child` calls per move over a mean
6.6 children, 17,105 terminal probes, 2,515 expansions, ~396 waves.

### A. Native `_best_child` + terminal probes -- about 100 ms/move

| | measured | recoverable | why not more |
|---|---|---|---|
| `_best_child` | 108.0 | ~97 | 3.08 us/call today. A native loop over 6.6 floats is ~0.1 us plus one crossing, but ONLY if the child data is in compact native arrays -- reading `N`, `W`, `prior`, `solved` back through the Python C-API would not beat the current code. |
| terminal probes | 51.2 | ~15 | 2.99 us/probe in situ, of which the microbench says ~2.12 us is already `clone` + `make_move` in C++. **A port keeps that.** Only the ~0.87 us of Python glue and two pybind crossings per probe are on the table. |
| **total** | **159.2** | **~110** | ~13% more clock |

The probe row is the correction to the obvious reading. The probes look like the
second-biggest tree cost, but 71% of that cost is already native and a C++ port
inherits it. Anyone planning around "51 ms of probes" would be planning around
15.

Against this, the compact child arrays have a write side: 26,606 nodes created
per move and every backup updating `N` and `W`. Backup is 2.0 ms/move today and
mirroring into a native array will not make it cheaper.

### B. Device path -- about 200-230 ms/move, and a smaller change

| | measured | recoverable | why not more |
|---|---|---|---|
| forward launch (CPU) | 297.9 | ~150 | 752 us/wave of CPU to issue one forward. A graph replay is one dispatch. Assumes 50% efficiency and a kernel count worth collapsing -- **unmeasured**. |
| mask upload (CPU) | 111.3 | ~67 | 86 ms/move of it is host overhead, not transfer. Assumes 60% recovery from keeping the mask on device or fusing it into the softmax. |
| softmax launch (CPU) | 23.2 | ~14 | folded into the same graph |
| **total** | **432.4** | **~230** | ~27% more clock |

**With one hard cap.** CPU-in-wave is 702.6 ms/move against 482.5 of GPU busy
time, so the CPU is the bottleneck today. Remove 230 ms of CPU and the two are
level, and the wave becomes GPU-bound -- further CPU savings buy nothing unless
the graph also cuts GPU-side dispatch. It should, for the same reason it cuts
the CPU side, but by how much is not measured. So B is ~200-230 and not more,
and its own ceiling is the first thing the prototype will reveal.

### The choice

B is roughly twice A for a much smaller change, and A's ceiling has just been
cut by a third by the probe correction. **Do the device path first**, and gate
it on the kernel count, which is one profiler call.

Both are capped by the same arithmetic: a move is ~913 ms, and neither path
touches the ~195 ms of tree bookkeeping, the ~53 ms of `state.make_move`, or
whatever the GPU genuinely needs. They are not additive with each other in any
simple way, because whichever lands first moves the bottleneck.

## What this does NOT license

No strength claim is projected from any of these numbers. The model-size
experiment is the standing reminder that per-operation arithmetic does not
survive contact with games -- and it is the same reasoning, on the same code,
that this study just showed was measuring the wrong axis. Any change here is
judged by win rate at equal wall clock or not at all.

---

## Instrument notes

Two failures during bring-up, both caught by a sanity check rather than by
reading the code. Both are now permanent gates.

**1. Events recorded on an idle stream are not timestamped when you record
them.** Under WDDM the driver batches submissions, so an end event after a
blocking `.cpu()` sits unsubmitted until something else flushes the queue --
the next wave. Its interval silently absorbed child construction, the terminal
probes and the next wave's entire selection descent: `D2H: values` came back at
342.8 ms/move for an 8-float copy, and the six device segments summed to 901.9
ms of a 904.4 ms move. That is impossible on its face, because the wave is
strictly serial and ~200 ms/move of pure-Python tree work runs with nothing
queued. A non-blocking `query()` forces the submission. `gpu_credible()` now
refuses any report whose device total exceeds 85% of the wave.

**2. The instrument cost what it measured.** Twelve event records plus six
queries came to roughly 0.6 ms per wave against a device section of roughly
0.6 ms. Events are therefore SAMPLED, one wave in 100, and the sampled and
clean waves are disjoint populations: clean waves supply the CPU column,
sampled waves the device column, because a sampled wave's CPU timing is
inflated by the very instrument being priced.

**3. Simulations per move is not a perturbation metric.** Reading it as one
would have reported a 25% instrument cost that does not exist:

    build        moves  sims/mv  leaf/mv   us/leaf      free
    untouched      157     5876     2559     356.8     56.5%
    off            156     4358     2502     342.8     42.6%
    events         149     4305     2501     342.8     41.9%
    sync           161     3841     2485     337.6     35.3%

Under a wall clock a simulation is not a fixed unit of work. A descent into an
already-expanded or proven subtree costs no evaluation at all and is nearly
free, and the share of such descents is a property of the POSITIONS: it ran
from 35% to 57% across four arms that began from identical openings and then
diverged, which moves sims/move by a quarter on its own. Wall time per network
evaluation has no such problem, and it is flat -- 337.6 to 356.8, a 5.7% range
that is the noise floor of the ladder. The instrument is inside it.

`agents/mcts.py` is frozen and hash-gated, so `wave_planes` and `_expand_wave`
are replicated in the profiler with timing boundaries between statements.
`tools/test_profile_expand.py` requires the replicas to produce bit-identical
children, priors, leaf values, expansion counts and probe counts against the
frozen originals, and to be callable through the patch on a real instance --
the first version stored a bound method on the class, which is not re-bound on
access, so every argument shifted by one and it only surfaced nine minutes into
a match.

## Reproduce

    python -m tools.profile_expand --mode all --engine pocket_r35 --games 12
    python -m tools.profile_expand --replay results/profile_expand/expand.json
