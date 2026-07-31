# RESULT -- where the 1,000 ms goes (2026-07-31)

Profiled on `engine:pocket_r35`, the promoted deployment engine, in its shipping
configuration: reuse on, batched expansion, solved propagation, deferred GC,
1,000 ms. 627 moves of self-play, 4,765 simulations per move.

**The tree is 27% of a move. Host/device traffic around the forward pass is up
to 40%. Neither the network weights nor the tree is the largest single cost.**

| category | ms/move | share |
|---|---|---|
| device traffic + expansion (MIXED) | 328.4 | 40.5% |
| residual: network forward + `make_move` | 265.1 | 32.7% |
| **pure-Python tree bookkeeping** | **218.1** | **26.9%** |
| total search | 811.6 | 100% |

Per operation, exclusive of nested operations:

| operation | ms/move | us/sim | category |
|---|---|---|---|
| expansion (`_expand_wave`) | 283.8 | 59.56 | mixed |
| child scoring / best-child | 119.0 | 24.99 | tree |
| proof: terminal probes | 58.7 | 12.33 | tree |
| plane build + H2D | 44.6 | 9.36 | mixed |
| selection: state clone | 24.0 | 5.04 | tree |
| legal-child iteration | 7.0 | 1.47 | tree |
| tree release | 6.6 | 1.38 | tree |
| backup traversal | 2.3 | 0.49 | tree |
| proof: backward induction | 0.4 | 0.08 | tree |
| proof: propagation | 0.0 | 0.01 | tree |
| tree reuse: adopt | 0.0 | 0.00 | tree |

`expansion` and `plane build` are deliberately NOT summed into the tree.
`_expand_wave` is dominated by a device round trip -- a mask H2D, a softmax,
and two D2H pulls -- with only the node-allocation loop inside it being Python
(priced separately at ~11 ms by counts x isolated cost). `wave_planes` is a C++
fill plus an H2D copy. Adding them to a "tree total" would attribute 328 ms of
CUDA traffic to a port that cannot touch it, which is the exact mistake this
profile exists to prevent. An earlier draft of this file made it.

Backing out `state.make_move` (35,101 descent steps per move at ~1.33 us, C++,
unwrapped) leaves the network forward at roughly **216 ms, 26.6%**.

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
bias, overhead priced by calibration (0.716 us/call over 43.6M calls = 49.8
ms/move) and subtracted before reporting.

---

## What this means for the native-tree program

The case for the port was built on an inference: a 39x parameter cut bought only
1.24x more search, so ">=87% of a simulation is not the network". That inference
was too strong. The network forward is ~27% of a move, and what the parameter
cut did not touch was not all tree -- most of it is device traffic.

**The tree is worth 27%, and 55% of that is two operations:** `_best_child`
(119.0 ms) and the solved-node terminal probes (58.7 ms). A perfect port of
those two -- to zero cost, which no port achieves -- buys 178 ms, or 22% more
thinking time. By the anchor ladder, 22% more clock is worth about +0.035.
Real-world, a good port might return half of that.

**Device traffic is the bigger target and much cheaper to attack.** 328 ms sits
in four transfers per wave at ~650 waves per move, and reducing round trips
around the forward pass is the same lever that already took nn-evals/s from
1,382 to 3,033 when `_expand_wave` replaced per-leaf expansion. It is also
outside everything ruled out: it is not an inference server, not a transposition
DAG, not symmetry folding.

**CAVEAT, load-bearing.** Part of the 328 ms is the GPU genuinely computing,
awaited at the first `.cpu()`. Separating compute from transfer needs CUDA
events. Until that is measured, **40.5% is an upper bound on what restructuring
the transfers could recover**, and it would be exactly the error this profile
just corrected to treat it as a target without checking.

### Recommended order

1. **Measure sync versus transfer** with CUDA events inside `_expand_wave`.
   Cheap, and it decides whether the 328 ms is addressable at all.
2. If it is largely transfer: restructure the wave to fewer round trips. Larger
   expected return than the tree, far smaller change.
3. Then the tree port, scoped to `_best_child` plus the terminal probes only --
   not "selection plus backup and compact node storage". Backup is 2.3 ms, 0.3%,
   and porting it would be motivated by intuition rather than measurement.

No strength claim is projected from any of these speedups. The model-size
experiment is the standing reminder: 1.24x more search was worth +0.039, and
the arithmetic that predicted otherwise was wrong on its own stated terms.

## Reproduce

    python -m tools.profile_tree --mode wrap --engine pocket_r35 --games 12
    python -m tools.profile_tree --mode all  --engine pocket_r35 --games 12
