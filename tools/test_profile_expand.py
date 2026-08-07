"""Tests for the expansion-wave CUDA study.

THE LOAD-BEARING TEST IS PARITY. `agents/mcts.py` is frozen and hash-gated, so
`tools/profile_expand.py` cannot instrument `_expand_wave` in place -- it
replicates the function with timing boundaries between the statements. A
replica that has drifted from the original profiles code nobody runs, and the
drift would be invisible: the numbers would still look like numbers. So the
replicas are run against the frozen originals on identical inputs and required
to produce bit-identical children, priors and leaf values.

    python -m tools.test_profile_expand
"""

import types
import unittest

import numpy as np
import torch

from agents import agent_base
from agents import mcts as mcts_mod
from agents.mcts import MCTS, MCTSNode
from tools import engine_registry
from tools import profile_expand as pe
from tools.profile_tree import _sample_states


def _fixture(k=6, seed=3):
    """K reachable positions, fresh unexpanded nodes, deterministic logits."""
    states = _sample_states(k, seed)[:k]
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(k, 81, generator=g)
    values = torch.randn(k, generator=g)
    return states, logits, values


def _to_eval(states):
    nodes = [MCTSNode(to_play=s.player) for s in states]
    return nodes, [(i, nodes[i], states[i].clone()) for i in range(len(states))]


def _snapshot(nodes, out):
    """Everything `_expand_wave` is contractually responsible for."""
    return [
        {
            "value": out[id(n)],
            "children": sorted(
                (mv, c.prior.hex() if isinstance(c.prior, float)
                 else c.prior, c.to_play, c.solved, c.is_terminal,
                 c.terminal_value)
                for mv, c in n.children.items()),
        }
        for n in nodes
    ]


class TestReplicaParity(unittest.TestCase):
    """The replicas must be the frozen functions, statement for statement."""

    def _mcts(self, solve=True):
        return MCTS(model=None, device="cpu", n_sims=1, c_puct=1.5,
                    wave_size=8, solve=solve)

    def _run_both(self, solve=True, k=6, seed=3):
        states, logits, values = _fixture(k, seed)

        m_ref = self._mcts(solve)
        nodes_ref, te_ref = _to_eval(states)
        out_ref = MCTS.__dict__["_expand_wave"](m_ref, te_ref, logits.clone(),
                                                values.clone())

        probe = pe.ExpandProbe("off")
        probe._orig_expand_wave = MCTS.__dict__["_expand_wave"]
        probe._begin("mid", k)
        m_got = self._mcts(solve)
        nodes_got, te_got = _to_eval(states)
        out_got = probe.expand_wave(m_got, te_got, logits.clone(),
                                    values.clone())
        return (_snapshot(nodes_ref, out_ref), _snapshot(nodes_got, out_got),
                m_ref, m_got, probe)

    def test_expand_wave_replica_is_bit_identical(self):
        ref, got, m_ref, m_got, _p = self._run_both(solve=True)
        self.assertEqual(len(ref), 6)
        self.assertTrue(any(r["children"] for r in ref), "fixture expanded no "
                        "children -- the test would pass vacuously")
        self.assertEqual(ref, got)
        self.assertEqual(m_ref.stat_expansions, m_got.stat_expansions)
        self.assertEqual(m_ref.stat_probes, m_got.stat_probes)

    def test_expand_wave_replica_matches_with_solve_off(self):
        ref, got, m_ref, m_got, _p = self._run_both(solve=False)
        self.assertEqual(ref, got)
        self.assertEqual(m_ref.stat_probes, 0)
        self.assertEqual(m_got.stat_probes, 0)

    def test_expand_wave_replica_matches_across_several_fixtures(self):
        for seed in (11, 29, 47):
            with self.subTest(seed=seed):
                ref, got, _a, _b, _p = self._run_both(k=8, seed=seed)
                self.assertEqual(ref, got)

    def test_wave_planes_replica_is_bit_identical(self):
        states, _l, _v = _fixture(5, seed=5)
        ref = agent_base.wave_planes(states, "cpu")
        probe = pe.ExpandProbe("off")
        probe.ctx["on"] = True
        probe.ctx["phase"] = "mid"
        got = probe.wave_planes(states, "cpu")
        self.assertEqual(ref.shape, got.shape)
        self.assertEqual(ref.dtype, got.dtype)
        self.assertTrue(torch.equal(ref, got))

    def test_wave_planes_passes_through_when_the_gate_is_shut(self):
        """A warmup wave must not open a slot the expansion will then fill."""
        states, _l, _v = _fixture(4, seed=9)
        probe = pe.ExpandProbe("off")
        self.assertFalse(probe.ctx["on"])
        got = probe.wave_planes(states, "cpu")
        self.assertIsNone(probe.slot)
        self.assertTrue(torch.equal(agent_base.wave_planes(states, "cpu"), got))


class TestSegmentBookkeeping(unittest.TestCase):
    """A mistyped segment key would silently vanish from the report."""

    def test_every_segment_the_replica_writes_is_declared(self):
        states, logits, values = _fixture(5, seed=13)
        probe = pe.ExpandProbe("off")
        probe.ctx["on"] = True
        probe.ctx["phase"] = "mid"
        probe.wave_planes(states, "cpu")
        m = MCTS(model=None, device="cpu", n_sims=1, wave_size=8, solve=True)
        _nodes, te = _to_eval(states)
        probe.expand_wave(m, te, logits, values)
        probe.flush()
        written = {seg for (_ph, seg) in list(probe.cpu) + list(probe.cpu_s)}
        self.assertTrue(written, "nothing was recorded")
        self.assertTrue(written <= set(pe.SEG_ORDER),
                        "undeclared segments: %s" % (written - set(pe.SEG_ORDER)))
        # The two that decide the study must actually be populated.
        for seg in ("host: plane fill", "D2H: probs",
                    "host: child construction"):
            self.assertIn(seg, written)

    def test_segment_order_is_well_formed(self):
        self.assertEqual(len(pe.SEG_ORDER), len(set(pe.SEG_ORDER)))
        self.assertTrue(set(pe.GPU_SEGMENTS) <= set(pe.SEG_ORDER))
        for seg, kind in pe.SEGMENTS:
            self.assertIn(kind, ("host", "copy", "gpu", "drain"))

    def test_the_ring_harvests_every_wave(self):
        probe = pe.ExpandProbe("off", sample_every=20, ring=2)
        for i in range(5):
            slot = probe._begin("mid", 3 + i)
            slot.cpu["host: plane fill"] = 0.001
        probe.flush()
        # Wave 0 is a sampled wave, so it lands in the other population.
        self.assertEqual(probe.waves["mid"] + probe.waves_s["mid"], 5)
        self.assertEqual(probe.leaves["mid"] + probe.leaves_s["mid"],
                         sum(3 + i for i in range(5)))
        self.assertAlmostEqual(
            probe.cpu[("mid", "host: plane fill")]
            + probe.cpu_s[("mid", "host: plane fill")], 0.005, places=9)

    def test_sampled_and_clean_waves_are_disjoint_and_exact(self):
        """The CPU column comes from the clean population and the device column
        from the sampled one; a wave counted in both would be double-counted."""
        probe = pe.ExpandProbe("off", sample_every=4, ring=3)
        for _i in range(20):
            probe._begin("mid", 8)
        probe.flush()
        self.assertEqual(probe.waves_s["mid"], 5)      # waves 0, 4, 8, 12, 16
        self.assertEqual(probe.waves["mid"], 15)

    def test_sampling_every_wave_leaves_the_clean_population_empty(self):
        probe = pe.ExpandProbe("off", sample_every=1)
        for _i in range(6):
            probe._begin("mid", 8)
        probe.flush()
        self.assertEqual(probe.waves["mid"], 0)
        self.assertEqual(probe.waves_s["mid"], 6)


class TestSynchronization(unittest.TestCase):
    """The whole design rests on which build synchronizes and which does not."""

    def _count_syncs(self, mode):
        calls = {"n": 0}
        orig = torch.cuda.synchronize

        def fake(*a, **kw):
            calls["n"] += 1
        torch.cuda.synchronize = fake
        try:
            states, logits, values = _fixture(4, seed=21)
            probe = pe.ExpandProbe(mode)
            probe._orig_expand_wave = MCTS.__dict__["_expand_wave"]
            probe._begin("mid", 4)
            m = MCTS(model=None, device="cpu", n_sims=1, wave_size=8,
                     solve=True)
            _nodes, te = _to_eval(states)
            probe.expand_wave(m, te, logits, values)
        finally:
            torch.cuda.synchronize = orig
        return calls["n"]

    def test_the_sync_build_drains_once_per_wave(self):
        self.assertEqual(self._count_syncs("sync"), 1)

    def test_the_events_build_never_drains(self):
        # If this ever fires, the "no added synchronization" claim in the
        # writeup is false and the events figures are perturbed.
        self.assertEqual(self._count_syncs("events"), 0)
        self.assertEqual(self._count_syncs("off"), 0)

    def test_the_sync_build_records_the_drain_segment(self):
        orig = torch.cuda.synchronize
        torch.cuda.synchronize = lambda *a, **kw: None
        try:
            states, logits, values = _fixture(4, seed=23)
            probe = pe.ExpandProbe("sync")
            probe._orig_expand_wave = MCTS.__dict__["_expand_wave"]
            probe._begin("mid", 4)
            m = MCTS(model=None, device="cpu", n_sims=1, wave_size=8,
                     solve=True)
            _nodes, te = _to_eval(states)
            probe.expand_wave(m, te, logits, values)
            probe.flush()
        finally:
            torch.cuda.synchronize = orig
        # The drain only happens on sampled waves, so it is in that population.
        self.assertIn(("mid", "first host synchronization"), probe.cpu_s)
        self.assertNotIn(("mid", "first host synchronization"), probe.cpu)


class TestGpuCredibility(unittest.TestCase):
    """The gate that caught the deferred-timestamp bug on the first run."""

    def _res(self, busy, wave=900.0):
        return {"gpu_us_per_wave": {"network forward": busy},
                "cpu_us_per_wave": {"network forward": wave}}

    def test_a_device_total_near_the_whole_wave_is_rejected(self):
        # What the first run actually produced: 901.9 of 904.4 ms/move.
        self.assertFalse(pe.gpu_credible(self._res(901.9, 904.4)))

    def test_a_plausible_device_total_passes(self):
        self.assertTrue(pe.gpu_credible(self._res(300.0, 900.0)))

    def test_the_ceiling_is_below_one(self):
        # A ceiling of 1.0 would accept the exact failure it exists to catch.
        self.assertLess(pe.GPU_BUSY_CEILING, 1.0)


class TestWaveSize(unittest.TestCase):
    """The reference wave rate is derived, not measured, so its constant has to
    track the registry."""

    def test_wave_size_matches_the_frozen_deployment_engine(self):
        cfg = engine_registry.ENGINES["pocket_r35"]
        self.assertEqual(int(cfg["wave"]), pe.WAVE_SIZE)

    def test_every_timed_engine_shares_that_wave_size(self):
        # waves == sims / WAVE_SIZE only holds if the engine under study uses
        # it. If a rung ever differs, this study must take wave from the player.
        for name, cfg in engine_registry.ENGINES.items():
            if "ms" in cfg and "wave" in cfg:
                self.assertEqual(int(cfg["wave"]), pe.WAVE_SIZE, name)


class _FakeModel:
    def forward_both(self, x):
        return x, x


class _FakePlayer:
    def __init__(self, model, bexp=True):
        self.name = "fake"
        self.model = model
        self.recording = True
        self.mcts = types.SimpleNamespace(batched_expand=bexp)

    def move(self, state, move_num):
        return 0


class TestInstall(unittest.TestCase):

    def test_install_then_remove_restores_the_frozen_functions(self):
        before = (mcts_mod.wave_planes, MCTS.__dict__["_expand_wave"],
                  _FakeModel.__dict__["forward_both"])
        probe = pe.ExpandProbe("off")
        probe.install([_FakePlayer(_FakeModel())])
        self.assertIsNot(mcts_mod.wave_planes, before[0])
        self.assertIsNot(MCTS.__dict__["_expand_wave"], before[1])
        probe.remove()
        self.assertIs(mcts_mod.wave_planes, before[0])
        self.assertIs(MCTS.__dict__["_expand_wave"], before[1])
        self.assertIs(_FakeModel.__dict__["forward_both"], before[2])

    def test_the_patched_expand_wave_is_callable_through_an_instance(self):
        """Identity checks are not enough. A bound method stored on a class is
        not re-bound on access, so the patch has to be a plain function or the
        MCTS instance is swallowed and every argument shifts by one -- which is
        exactly what the first version did, and it only surfaced nine minutes
        into a match."""
        probe = pe.ExpandProbe("off")
        probe.install([_FakePlayer(_FakeModel())])
        try:
            states, logits, values = _fixture(4, seed=31)
            probe.ctx["on"] = True
            probe.ctx["phase"] = "mid"
            probe.wave_planes(states, "cpu")
            m = MCTS(model=None, device="cpu", n_sims=1, wave_size=8,
                     solve=True)
            nodes, te = _to_eval(states)
            out = m._expand_wave(te, logits, values)     # through the patch
        finally:
            probe.remove()
        self.assertEqual(len(out), len(nodes))
        self.assertEqual(m.stat_expansions, len(nodes))

    def test_a_per_leaf_engine_is_refused(self):
        probe = pe.ExpandProbe("off")
        with self.assertRaises(SystemExit):
            probe.install([_FakePlayer(_FakeModel(), bexp=False)])
        probe.remove()

    def test_the_root_expansion_is_not_counted_as_a_wave(self):
        """`_expand` runs one forward outside any wave. Folding it in would
        corrupt every per-leaf figure."""
        probe = pe.ExpandProbe("off")
        seen = {"n": 0}

        def orig(model, x):
            seen["n"] += 1
            return x

        self.assertIsNone(probe.slot)
        probe.forward_both(orig, None, torch.zeros(1))
        self.assertEqual(seen["n"], 1)
        self.assertEqual(probe.waves, {})
        self.assertEqual(dict(probe.cpu), {})


def _fake_res(mode):
    """A payload shaped exactly like run_instrumented's return value."""
    segs = pe.SEG_ORDER
    res = {
        "mode": mode, "sample_every": 100, "moves": 600, "sims": 3000000,
        "nn_evals": 1200000, "sims_per_move": 5000.0, "nn_per_move": 2000.0,
        "search_ms_per_move": 960.0, "fingerprint": "x", "params": 172389,
        "budget_ms": 1000.0,
        "waves": 200000, "clean_waves": 198000, "sampled_waves": 2000,
        "leaves": 1200000, "mean_k": 6.1, "mean_k_sampled": 6.1,
        "own_waves_per_move": 333.0, "own_leaves_per_move": 2000.0,
        "evaluating_wave_fraction": 0.53,
        "cpu_us_per_leaf": {s: 3.0 for s in segs
                            if s != "first host synchronization"},
        "cpu_us_per_wave": {s: 18.0 for s in segs
                            if s != "first host synchronization"},
        "sampled_cpu_us_per_leaf": {}, "sampled_cpu_us_per_wave": {},
        "gpu_us_per_leaf": {s: 2.0 for s in pe.GPU_SEGMENTS},
        "gpu_us_per_wave": {s: 12.0 for s in pe.GPU_SEGMENTS},
        "gpu_span_us_per_leaf": 14.0, "gpu_span_us_per_wave": 84.0,
        "k_histogram": {"1": 100, "8": 1900},
        "by_phase": {ph: {"waves": 100, "sampled_waves": 2, "leaves": 610,
                          "mean_k": 6.1,
                          "cpu_us_per_wave": {s: 18.0 for s in segs},
                          "gpu_us_per_wave": {s: 12.0
                                              for s in pe.GPU_SEGMENTS}}
                     for ph in ("early", "mid", "late")},
    }
    if mode == "sync":
        for d, v in (("sampled_cpu_us_per_leaf", 3.0),
                     ("sampled_cpu_us_per_wave", 18.0)):
            res[d] = {"first host synchronization": v,
                      "D2H: probs": v, "D2H: values": v}
    return res


class TestReporting(unittest.TestCase):
    """A format specifier must never be able to destroy a finished run.

    It already happened once in this program: a single None line number threw
    away nine minutes of collected data at the serialization step. Here the
    report runs after forty minutes of GPU time, so it is exercised in the
    suite rather than for the first time at the end.
    """

    def test_report_runs_on_both_builds(self):
        for mode in ("events", "sync"):
            with self.subTest(mode=mode):
                pe.report(_fake_res(mode), ref_leaves=2000.0)

    def test_decide_runs_and_returns_every_row(self):
        out = pe.decide(_fake_res("events"), _fake_res("sync"),
                        ref_leaves=2000.0)
        self.assertIn("GPU busy, total", out)
        self.assertIn("  of which waiting [sync build]", out)
        for v in out.values():
            self.assertIsInstance(v, float)

    def test_perturb_ladder_reports_the_per_leaf_rate(self):
        """The ladder must be read per leaf. Reading sims/move as the
        perturbation reported a 25% instrument cost that did not exist."""
        arms = {"untouched": {"moves": 157, "sims_per_move": 5876.0,
                              "nn_per_move": 2559.0,
                              "search_ms_per_move": 912.9},
                "off": {"moves": 156, "sims_per_move": 4358.0,
                        "nn_per_move": 2502.0, "search_ms_per_move": 857.5}}
        pe.report_perturb(arms)

    def test_report_survives_a_build_with_no_device_column(self):
        res = _fake_res("sync")
        res["gpu_us_per_leaf"] = {}
        res["gpu_us_per_wave"] = {}
        res["gpu_span_us_per_leaf"] = 0.0
        pe.report(res, ref_leaves=2000.0)


class TestSeeds(unittest.TestCase):

    def test_expand_seed_is_its_own_namespace(self):
        seeds = engine_registry.SEEDS
        self.assertIn("expand", seeds)
        self.assertEqual(len(set(seeds.values())), len(seeds))
        self.assertEqual(pe.EXPAND_SEED, seeds["expand"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
