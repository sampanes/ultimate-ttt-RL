"""Tests for the kernel/launch trace.

THE LOAD-BEARING TEST IS PARITY, for the same reason as in
`tools/test_profile_expand.py`: `agents/mcts.py` is frozen and hash-gated, so
this study cannot instrument the wave in place. It replays the device sequence
from its own copy, and a copy that has drifted profiles code nobody runs -- and
the drift is invisible, because the numbers still look like numbers. So the
replay is required to produce bit-identical priors and leaf values against the
frozen `wave_planes` + `forward_both` + `_expand_wave`.

The second thing worth testing is the warmup gate. `play_match(warmup=N)`
discards the warmup from the PLAYERS only, and an accumulator living outside
them keeps counting -- which is exactly how the first tree profile came to be
wrong by four points. The collector here is such an accumulator.

    python -m tools.test_profile_kernels
"""

import collections
import unittest

import numpy as np
import torch

from agents import agent_base
from agents import mcts as mcts_mod
from agents.mcts import MCTS, MCTSNode
from tools import engine_registry
from tools import profile_kernels as pk
from tools.profile_tree import _sample_states


class _TinyNet(torch.nn.Module):
    """Deterministic stand-in with the real forward_both contract, including
    the batch-1 squeeze -- which is the branch most likely to be got wrong."""

    def __init__(self, seed=7):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(81, 7 * 81, generator=g) * 0.1)
        self.v = torch.nn.Parameter(torch.randn(1, 7 * 81, generator=g) * 0.1)

    def forward_both(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        flat = x.reshape(x.shape[0], -1)
        policy = flat @ self.w.t()
        value = torch.tanh(flat @ self.v.t()).squeeze(-1)
        if policy.shape[0] == 1:
            return policy.squeeze(0), value.squeeze(0)
        return policy, value


def _fixture(k=6, seed=3):
    return _sample_states(k, seed)[:k]


class TestReplayParity(unittest.TestCase):
    """`wave_sequence` must be the frozen device sequence, statement for
    statement."""

    @torch.no_grad()
    def _reference(self, states, model):
        """What the engine actually does: mcts.wave_planes -> forward_both ->
        the head of _expand_wave, read back off the children it builds.

        Under no_grad because `MCTS.search` is, and `_expand_wave` calls
        `.numpy()` on the result -- which raises on a grad-tracking tensor.
        """
        m = MCTS(model=model, device="cpu", n_sims=1, c_puct=1.5,
                 wave_size=8, solve=True)
        xs = mcts_mod.wave_planes(states, "cpu")
        logits_b, values_b = model.forward_both(xs)
        if logits_b.dim() == 1:
            logits_b = logits_b.unsqueeze(0)
            values_b = values_b.unsqueeze(0)
        nodes = [MCTSNode(to_play=s.player) for s in states]
        to_eval = [(i, nodes[i], states[i].clone()) for i in range(len(states))]
        out = MCTS.__dict__["_expand_wave"](m, to_eval, logits_b, values_b)
        priors = []
        for n in nodes:
            priors.append(sorted((mv, c.prior) for mv, c in n.children.items()))
        return priors, [out[id(n)] for n in nodes]

    @torch.no_grad()
    def _replay(self, states, model):
        probs, values = pk.wave_sequence(model, states, "cpu")
        valids = [
            mcts_mod.rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
            for s in states]
        priors = [sorted((mv, float(probs[i][mv])) for mv in valids[i])
                  for i in range(len(states))]
        return priors, [float(v) for v in values]

    def test_priors_and_values_are_bit_identical(self):
        model = _TinyNet()
        for k, seed in ((6, 3), (8, 11), (3, 29)):
            with self.subTest(k=k, seed=seed):
                states = _fixture(k, seed)
                ref_p, ref_v = self._reference(states, model)
                got_p, got_v = self._replay(states, model)
                self.assertTrue(any(p for p in ref_p),
                                "fixture built no children -- vacuous test")
                self.assertEqual(ref_p, got_p)
                self.assertEqual(ref_v, got_v)

    def test_batch_of_one_takes_the_squeeze_branch(self):
        """k=1 is 11.5% of production waves and is the only size where
        forward_both returns (81,) instead of (1, 81)."""
        model = _TinyNet()
        states = _fixture(1, 5)
        with torch.no_grad():
            xs = mcts_mod.wave_planes(states, "cpu")
            logits, _v = model.forward_both(xs)
        self.assertEqual(logits.dim(), 1, "fixture did not exercise the "
                                          "squeeze branch")
        ref_p, ref_v = self._reference(states, model)
        got_p, got_v = self._replay(states, model)
        self.assertEqual(ref_p, got_p)
        self.assertEqual(ref_v, got_v)

    def test_planes_match_the_frozen_builder(self):
        states = _fixture(5, 5)
        ref = mcts_mod.wave_planes(states, "cpu").numpy()
        buf = np.empty((5, 7, 9, 9), dtype=np.float32)
        pk._fill(buf, states)
        np.testing.assert_array_equal(ref, buf)

    def test_fill_handles_the_python_engine_branch(self):
        """Both branches of agent_base.wave_planes exist upstream, so both must
        exist in the replica -- a fast-path-only copy would profile something
        else on a box without the C++ extension built."""
        states = _fixture(4, 17)
        buf_fast = np.empty((4, 7, 9, 9), dtype=np.float32)
        pk._fill(buf_fast, states)
        saved = agent_base._FILL_PLANES
        agent_base._FILL_PLANES = False
        try:
            buf_slow = np.empty((4, 7, 9, 9), dtype=np.float32)
            pk._fill(buf_slow, states)
        finally:
            agent_base._FILL_PLANES = saved
        np.testing.assert_array_equal(buf_fast, buf_slow)


class TestSegments(unittest.TestCase):
    """The two CUDA studies must name the same things the same way, or the
    cross-check between them compares different segments."""

    def test_segment_names_match_the_expansion_study(self):
        from tools import profile_expand as pe
        host_only = {"first host synchronization",
                     "host: child construction", "host: make_move / probes"}
        expected = [s for s, _k in pe.SEGMENTS if s not in host_only]
        self.assertEqual(expected, pk.SEGMENTS)

    def test_gpu_segments_are_a_subset_of_segments(self):
        self.assertTrue(set(pk.GPU_SEGMENTS) <= set(pk.SEGMENTS))

    def test_production_reference_covers_every_segment(self):
        ref = pk.PRODUCTION_REFERENCE
        self.assertEqual(set(ref["cpu_us_per_wave"]), set(pk.SEGMENTS))
        self.assertEqual(set(ref["gpu_us_per_wave"]), set(pk.GPU_SEGMENTS))


class TestQuantile(unittest.TestCase):

    def test_median_and_p90_of_a_flat_distribution(self):
        h = collections.Counter({1: 10, 2: 10, 3: 10, 4: 10, 5: 10})
        self.assertEqual(pk.quantile_k(h, 0.5), 3)
        self.assertEqual(pk.quantile_k(h, 0.9), 5)

    def test_the_production_shape_collapses_median_and_p90(self):
        """The real histogram is bimodal with 70% of its mass at k=8, so the
        requested median and p90 are the SAME size. The report has to say so
        rather than silently measuring one size twice."""
        h = collections.Counter({1: 22047, 2: 10455, 3: 6636, 4: 5421,
                                 5: 4515, 6: 3611, 7: 3886, 8: 135877})
        self.assertEqual(pk.quantile_k(h, 0.5), 8)
        self.assertEqual(pk.quantile_k(h, 0.9), 8)

    def test_empty_histogram_returns_none(self):
        self.assertIsNone(pk.quantile_k(collections.Counter(), 0.5))


class _StubPlayer:
    """Enough of TimedPlayer for the gate test: `recording` is the flag
    play_match itself uses to decide what counts."""

    def __init__(self):
        self.recording = False
        self.mcts = None

    def move(self, state, move_num):
        return 0


class TestWaveCollector(unittest.TestCase):

    def setUp(self):
        self.ctx = {"phase": "mid", "on": True}
        self.col = pk.WaveCollector(self.ctx, cap=3, stride=2)
        self.calls = []

        def orig(states, device):
            self.calls.append((len(states), device))
            return "planes"

        self.col._orig = orig

    def _states(self, k):
        return _fixture(k, 3)

    def test_passes_through_and_returns_the_original_result(self):
        got = self.col.wave_planes(self._states(4), "cpu")
        self.assertEqual(got, "planes")
        self.assertEqual(self.calls, [(4, "cpu")])

    def test_counts_every_wave_but_clones_only_on_the_stride(self):
        for _ in range(6):
            self.col.wave_planes(self._states(4), "cpu")
        self.assertEqual(self.col.hist[4], 6)
        self.assertEqual(len(self.col.pool[4]), 3)

    def test_pool_is_capped(self):
        col = pk.WaveCollector(self.ctx, cap=2, stride=1)
        col._orig = lambda s, d: None
        for _ in range(20):
            col.wave_planes(self._states(2), "cpu")
        self.assertEqual(col.hist[2], 20)
        self.assertEqual(len(col.pool[2]), 2)

    def test_cloned_states_are_independent_of_the_live_search(self):
        col = pk.WaveCollector(self.ctx, cap=1, stride=1)
        col._orig = lambda s, d: None
        live = self._states(2)
        col.wave_planes(live, "cpu")
        saved = col.pool[2][0]
        self.assertIsNot(saved[0], live[0])
        before = list(saved[0].board)
        live[0].make_move(
            mcts_mod.rule_utl_valid_moves(live[0].board, live[0].last_move,
                                          live[0].mini_winners)[0])
        self.assertEqual(before, list(saved[0].board))

    def test_warmup_is_not_counted(self):
        """THE GATE. Warmup waves must reach the collector -- the search really
        runs them -- and contribute exactly zero. Not a tolerance: zero."""
        self.ctx["on"] = False
        for _ in range(5):
            self.col.wave_planes(self._states(4), "cpu")
        self.assertEqual(len(self.calls), 5, "warmup never reached the "
                                             "collector -- vacuous gate test")
        self.assertEqual(sum(self.col.hist.values()), 0)
        self.assertEqual(self.col.pool, {})

    def test_install_and_remove_restore_the_frozen_function(self):
        raw = mcts_mod.wave_planes
        col = pk.WaveCollector(self.ctx)
        col.install()
        self.assertIsNot(mcts_mod.wave_planes, raw)
        col.remove()
        self.assertIs(mcts_mod.wave_planes, raw)

    def test_remove_is_idempotent(self):
        raw = mcts_mod.wave_planes
        col = pk.WaveCollector(self.ctx)
        col.install()
        col.remove()
        col.remove()
        self.assertIs(mcts_mod.wave_planes, raw)


class TestDemangle(unittest.TestCase):

    def test_recognises_the_kernels_this_model_runs(self):
        self.assertEqual(pk._demangle(
            "_Z23implicit_convolve_sgemmIffLi1024E"), "conv:implicit_gemm")
        self.assertEqual(pk._demangle(
            "_ZN43_GLOBAL__N_SoftMax_cu_softmax_warp_forward"), "softmax")
        self.assertEqual(pk._demangle(
            "_ZN2at6native29vectorized_elementwise_kernelILi4E"),
            "elementwise")

    def test_unknown_kernels_are_truncated_not_dropped(self):
        got = pk._demangle("_ZN9some_kernel_nobody_has_seen_before" * 3)
        self.assertTrue(got)
        self.assertLessEqual(len(got), 48)


class TestReporting(unittest.TestCase):
    """Format-crash smoke. A report that raises after a twenty-minute
    measurement loses the measurement."""

    def _hist(self):
        return {"games": 4, "moves": 200, "waves": 1000, "leaves": 6000,
                "mean_k": 6.0, "median_k": 8, "p90_k": 8,
                "waves_per_move": 5.0, "leaves_per_move": 30.0,
                "histogram": {"1": 300, "8": 700},
                "share_of_leaves": {"1": 0.05, "8": 0.95},
                "by_phase": {"mid": {"8": 700}}}

    def _trace(self):
        return {"reps": 10, "kernels_total": 200, "kernels_per_wave": 20.0,
                "kernel_us_mean": 4.0, "kernel_us_median": 3.0,
                "kernel_us_max": 12.0, "small_kernel_count_share": 0.9,
                "small_kernel_time_share": 0.7,
                "gpu_kernel_us_per_wave": 80.0,
                "gpu_memcpy_us_per_wave": 20.0,
                "gpu_busy_us_per_wave": 100.0,
                "device_gap_us_median": 5.0,
                "device_gap_us_total_per_wave": 100.0,
                "profiled_launch_cadence_us_median": 30.0,
                "profiled_launch_us_mean": 25.0, "launches_per_wave": 20.0,
                "explicit_syncs_per_wave": 3.0,
                "sync_kinds": {"cudaStreamSynchronize": 30},
                "sync_us_per_wave_profiled": 40.0,
                "copies_per_wave": {"H2D": {"n": 2.0, "bytes": 18792.0,
                                            "device_us": 3.0}},
                "kernels_by_kind": {"conv:implicit_gemm":
                                    {"n_per_wave": 6.0, "us_per_wave": 40.0}}}

    def _bench(self):
        return {"reps": 10, "k": 8,
                "cpu_us_per_wave": {s: 50.0 for s in pk.SEGMENTS},
                "gpu_us_per_wave": {s: 40.0 for s in pk.GPU_SEGMENTS}}

    def _mask(self):
        return {"k": 8, "reps": 100,
                "arms": collections.OrderedDict([
                    ("host: np.zeros((k,81), bool)", 0.6),
                    ("host: zeros + python fill loop", 19.6),
                    ("upload: pageable bool .to() [drained]", 70.3),
                    ("upload: pageable uint8 .to() [drained]", 66.9),
                    ("upload: pageable bool .to() [FORWARD IN FLIGHT]", 353.1),
                ]),
                "forward_launch_us": 700.0,
                "forward_launch_plus_drain_us": 800.0,
                "forward_tail_us": 100.0}

    def _capture(self, wave_saving_us=800.0):
        return {"k": 8, "reps": 100,
                "blockers_checked": [{"blocker": "b", "finding": "f"}],
                "captured": True, "capture_error": None,
                "max_abs_probs_diff": 0.0, "max_abs_values_diff": 0.0,
                "replay_launch_us": 30.0, "replay_plus_drain_us": 500.0,
                "eager_launch_us": 700.0, "eager_plus_drain_us": 800.0,
                "captured_with_d2h": True, "capture_with_d2h_error": None,
                "max_abs_probs_diff_d2h": 0.0, "max_abs_values_diff_d2h": 0.0,
                "replay_d2h_end_to_end_us": 520.0,
                "wave_eager_us": 1700.0,
                "wave_graphed_us": 1700.0 - wave_saving_us,
                "wave_saving_us": wave_saving_us,
                "wave_max_abs_probs_diff": 0.0,
                "wave_max_abs_values_diff": 0.0,
                "wave_checks": 12, "wave_distinct_outputs": 12}

    def _segmented(self):
        return {"reps": 10, "unattributed_runtime_calls": 3,
                "per_segment": {
                    "network forward": {
                        "kernels_per_wave": 33.0, "memcpys_per_wave": 0.0,
                        "bytes_per_wave": 0.0, "syncs_per_wave": 0.0,
                        "device_us_per_wave": 300.0,
                        "kernel_names": {"conv:cudnn": 9.0}},
                    "device: mask + softmax": {
                        "kernels_per_wave": 2.0, "memcpys_per_wave": 0.0,
                        "bytes_per_wave": 0.0, "syncs_per_wave": 0.0,
                        "device_us_per_wave": 8.0,
                        "kernel_names": {"softmax": 1.0}},
                }}

    def test_every_report_renders(self):
        traces = {"8": self._trace()}
        benches = {"8": self._bench()}
        pk.report_hist(self._hist())
        pk.report_trace(traces)
        pk.report_segmented(self._segmented())
        pk.report_bench(benches, traces)
        pk.report_bench_vs_production(benches, pk.PRODUCTION_REFERENCE)
        pk.report_mask(self._mask())
        pk.report_capture(self._capture())

    def test_capture_failure_renders_without_the_success_fields(self):
        pk.report_capture({"k": 8, "blockers_checked": [], "captured": False,
                           "capture_error": "boom"})

    def _decide(self, waves_per_move, capture=None):
        # The rate must come from THIS run's histogram, not the stale
        # reference -- that substitution was worth 31% on the headline.
        hist = self._hist()
        hist["waves_per_move"] = waves_per_move
        return pk.decide(hist, {"8": self._trace()}, {"8": self._bench()},
                         self._mask(),
                         self._capture() if capture is None else capture,
                         pk.PRODUCTION_REFERENCE)

    def test_the_rate_comes_from_this_run_not_the_reference(self):
        ref = dict(pk.PRODUCTION_REFERENCE)
        ref["waves_per_move"] = 9999.0
        hist = self._hist()
        hist["waves_per_move"] = 400.0
        out = pk.decide(hist, {"8": self._trace()}, {"8": self._bench()},
                        self._mask(), self._capture(), ref)
        self.assertAlmostEqual(out["waves_per_move_reference"], 400.0)

    def test_decision_thresholds(self):
        # 800 us/wave saved, discounted 30%: 0.56 ms per wave/move.
        self.assertIn("graph-first", self._decide(400.0)["verdict"])
        self.assertIn("comparable", self._decide(150.0)["verdict"])
        self.assertIn("abandon", self._decide(50.0)["verdict"])

    def test_the_verdict_uses_the_measured_saving_not_the_cpu_column(self):
        """The CPU column inside the region is dominated by the host blocked
        on a pageable H2D while the GPU works. Scoring the verdict off it read
        461 ms/move on a wave that costs 1.7 ms in total."""
        out = self._decide(400.0, capture=self._capture(wave_saving_us=10.0))
        self.assertIn("abandon", out["verdict"])
        self.assertGreater(out["cpu_in_region_ms_per_move"], 50.0)

    def test_a_failed_capture_cannot_produce_a_graph_first_verdict(self):
        out = self._decide(1500.0, capture={"captured": False,
                                            "capture_error": "boom"})
        self.assertIn("abandon", out["verdict"])


class TestSeeds(unittest.TestCase):

    def test_kernels_has_its_own_seed_namespace(self):
        self.assertEqual(engine_registry.SEEDS["kernels"], 6800)
        vals = list(engine_registry.SEEDS.values())
        self.assertEqual(len(vals), len(set(vals)), "seed namespaces collide")

    def test_the_tool_uses_it(self):
        self.assertEqual(pk.KERNEL_SEED, engine_registry.SEEDS["kernels"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
