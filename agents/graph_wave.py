"""CUDA-graph replay of the expansion wave's device sequence.

WHY. `RESULT_KERNEL_TRACE.md` measured the wave issuing 36 kernels -- 33 in the
network forward, 3 in masking and softmax -- at about 26 us of CPU per launch
against kernels whose median device time is 3.68 us. The GPU does 7.8% of a
move's real work; the rest of the wave is the host issuing tiny operations one
at a time and then blocking on them. A CUDA graph captures that fixed sequence
once and replays it with a single dispatch.

WHAT IS CAPTURED. Everything between the planes arriving on the device and the
results arriving back in host memory:

    forward -> masked_fill -> softmax -> D2H probs -> D2H values

The two pulls are inside the graph as captured asynchronous copies into pinned
host buffers. That is deliberate and it is where a second win comes from: the
eager wave performs FOUR `cudaStreamSynchronize` calls, none of which the
search asks for -- PyTorch emits one behind every pageable memcpy. Staging
through pinned memory and folding the pulls into the capture leaves exactly one
event wait per wave.

WHAT IS NOT CAPTURED. Building the planes and the legal mask (host work), and
constructing children (host work). The graph is a device-side replay, not a
rewrite of the search.

ONE SHAPE. A graph is captured for one wave size. Live play spends 90.7% of its
waves and 95.4% of its network evaluations at k=8, so a single capture covers
the workload and every other size keeps the eager path -- which also remains
the correctness oracle. Nothing here changes what the search decides; it
changes how long a wave takes, and under a clock that buys more waves.

THE OUTPUT BUFFERS ARE REUSED. A replay overwrites the same pinned arrays every
time, so `run` returns copies. That costs ~2 us against ~950 us saved, and it
removes an aliasing bug that would surface as a wrong prior in one wave out of
thousands.
"""

import numpy as np
import torch
import torch.nn.functional as F

from agents.agent_base import (_has_fill_planes,
                               board_to_tensor_from_gamestate)


def fill_planes_into(buf, states):
    """`agent_base.wave_planes`'s fill, branch for branch, into a caller buffer.

    Kept here rather than reusing `wave_planes` because that function allocates
    its own array and ends with a pageable `.to(device)` -- the two things this
    path exists to avoid. `tests` require this to agree with it byte for byte.
    """
    if len(states) and _has_fill_planes(states[0]):
        for i, s in enumerate(states):
            s.fill_planes(buf[i])
    else:
        for i, s in enumerate(states):
            buf[i] = board_to_tensor_from_gamestate(s).numpy()


class GraphedWave:
    """A captured device sequence for one wave size, or a dud that says why.

    Construction never raises. A box without CUDA, a driver that refuses
    capture, or a model with an op that cannot be captured all produce an
    instance with `ok == False` and a `reason`, and the caller keeps running
    the eager path. A performance optimisation must not be able to take the
    engine down.
    """

    def __init__(self, model, device, k=8, warmup=5):
        self.model = model
        self.device = device
        self.k = int(k)
        self.ok = False
        self.reason = None
        self.replays = 0
        try:
            self._build(warmup)
            self.ok = True
        except Exception as exc:                       # pragma: no cover
            self.reason = "%s: %s" % (type(exc).__name__, exc)

    # ------------------------------------------------------------------

    def _region(self):
        """Exactly the ops `_expand_wave` performs on the device, in order."""
        logits, values = self.model.forward_both(self.static_x)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            values = values.unsqueeze(0)
        probs = F.softmax(logits.masked_fill(~self.static_m, float("-inf")),
                          dim=1)
        return probs, values.reshape(-1)

    def _build(self, warmup):
        k = self.k
        if not (isinstance(self.device, str) and self.device.startswith("cuda")
                or getattr(self.device, "type", None) == "cuda"):
            raise RuntimeError("graph capture needs a CUDA device, got %r"
                               % (self.device,))
        dev = self.device

        # Device-side inputs the graph reads every replay. Their ADDRESSES are
        # baked into the capture, so the wave copies into them; it can never
        # hand the graph a fresh tensor.
        self.static_x = torch.zeros((k, 7, 9, 9), dtype=torch.float32,
                                    device=dev)
        self.static_m = torch.zeros((k, 81), dtype=torch.bool, device=dev)

        # Pinned host staging. Pageable memory is the whole problem: a pageable
        # H2D emits cudaMemcpyAsync AND cudaStreamSynchronize, and that sync
        # waits on whatever is already queued -- measured at 74.7 us drained
        # against 207.9 us with a forward in flight, for 648 bytes.
        self.pin_x = torch.empty((k, 7, 9, 9), dtype=torch.float32,
                                 pin_memory=True)
        self.pin_m = torch.empty((k, 81), dtype=torch.bool, pin_memory=True)
        self.host_probs = torch.empty((k, 81), dtype=torch.float32,
                                      pin_memory=True)
        self.host_values = torch.empty((k,), dtype=torch.float32,
                                       pin_memory=True)
        self._vx = self.pin_x.numpy()
        self._vm = self.pin_m.numpy()
        self._hp = self.host_probs.numpy()
        self._hv = self.host_values.numpy()

        # Capture requires the workload to have run on a side stream first --
        # cudnn picks algorithms and the allocator settles on the first calls,
        # and neither may happen inside the capture.
        with torch.no_grad():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    probs, values = self._region()
                    self.host_probs.copy_(probs, non_blocking=True)
                    self.host_values.copy_(values, non_blocking=True)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                probs, values = self._region()
                self.host_probs.copy_(probs, non_blocking=True)
                self.host_values.copy_(values, non_blocking=True)
        self.done = torch.cuda.Event()

    # ------------------------------------------------------------------

    def accepts(self, k):
        return self.ok and k == self.k

    def run(self, states, valids):
        """One wave: states and legal moves in, probabilities and values out.

        Returns copies, not views -- see the module docstring.
        """
        fill_planes_into(self._vx, states)
        self.static_x.copy_(self.pin_x, non_blocking=True)
        self._vm[:] = False
        for i, v in enumerate(valids):
            self._vm[i, v] = True
        self.static_m.copy_(self.pin_m, non_blocking=True)
        self.graph.replay()
        self.done.record()
        self.done.synchronize()
        self.replays += 1
        return self._hp.copy(), self._hv.copy()
