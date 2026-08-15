"""Tests for the CUDA-graph wave path.

THE GRAPH IS A PERFORMANCE CHANGE THAT MUST CHANGE NOTHING ELSE. So the tests
that matter are not "does it run" -- they are:

  1. bit-identical priors and leaf values against the eager path, over many
     DIFFERENT production positions rather than one repeated input;
  2. an explicit stale-buffer check: a graph captured around a buffer the wave
     no longer writes into would replay its captured input forever and still
     match perfectly on the position it was captured from. Distinct inputs must
     produce distinct outputs;
  3. k != the captured size falls back to eager and is still correct;
  4. a capture failure degrades to eager rather than taking the engine down,
     and an arm that ASKED for the graph refuses to run instead.

CUDA tests skip cleanly on a box without a GPU; the fallback and plumbing tests
run everywhere.

    python -m agents.test_graph_wave
"""

import unittest

import numpy as np
import torch

from agents import mcts as mcts_mod
from agents.graph_wave import GraphedWave, fill_planes_into
from agents.mcts import MCTS, MCTSNode
from engine.rules import rule_utl_valid_moves
from tools.arena_1s import load_net
from tools.profile_tree import _sample_states

HAVE_CUDA = torch.cuda.is_available()
CKPT = "models/pocket_candidate/squeeze_pocket.pt"
ARCH = "squeeze"
WAVE = 8


def _states(n, seed):
    return _sample_states(n, seed)


def _to_eval(states):
    nodes = [MCTSNode(to_play=s.player) for s in states]
    return nodes, [(i, nodes[i], states[i].clone())
                   for i in range(len(states))]


def _snapshot(nodes, out):
    """Everything the expansion is contractually responsible for."""
    return [
        {"value": out[id(n)].hex(),
         "children": sorted((mv, c.prior.hex(), c.to_play, c.solved,
                             c.is_terminal, c.terminal_value)
                            for mv, c in n.children.items())}
        for n in nodes
    ]


class TestFillParity(unittest.TestCase):
    """The local fill must be `agent_base.wave_planes`, byte for byte."""

    def test_matches_wave_planes(self):
        for k, seed in ((8, 3), (5, 19), (1, 41)):
            with self.subTest(k=k, seed=seed):
                st = _states(k, seed)
                ref = mcts_mod.wave_planes(st, "cpu").numpy()
                buf = np.empty((k, 7, 9, 9), dtype=np.float32)
                fill_planes_into(buf, st)
                np.testing.assert_array_equal(ref, buf)


@unittest.skipUnless(HAVE_CUDA, "needs CUDA")
class TestGraphParity(unittest.TestCase):
    """Bit-identical expansion, over many distinct positions."""

    @classmethod
    def setUpClass(cls):
        cls.model, _info = load_net(CKPT, ARCH, "cuda")

    def _mcts(self, graph):
        m = MCTS(self.model, "cuda", n_sims=1, c_puct=1.5, wave_size=WAVE,
                 solve=True, batched_expand=True, graph_wave=graph)
        if graph:
            self.assertIsNotNone(m.graph_wave,
                                 "capture failed: %s"
                                 % getattr(m, "graph_wave_reason", "?"))
        return m

    def _run(self, m, states):
        nodes, te = _to_eval(states)
        with torch.no_grad():
            if m.graph_wave is not None and m.graph_wave.accepts(len(te)):
                out = m._expand_wave_graphed(te)
            else:
                xs = mcts_mod.wave_planes([s for _, _, s in te], "cuda")
                lg, vl = m.model.forward_both(xs)
                if lg.dim() == 1:
                    lg = lg.unsqueeze(0)
                    vl = vl.unsqueeze(0)
                out = m._expand_wave(te, lg, vl)
        return _snapshot(nodes, out)

    def test_bit_identical_over_many_distinct_waves(self):
        """Several hundred production waves, all different, not one repeated.

        A graph that ignored its input buffer would pass a single-wave test.
        """
        eager, graphed = self._mcts(False), self._mcts(True)
        n_waves, seen = 0, set()
        for seed in range(200, 520):
            states = _states(WAVE, seed)
            ref = self._run(eager, states)
            got = self._run(graphed, states)
            self.assertEqual(ref, got, "wave from seed %d differs" % seed)
            seen.add(repr(ref))
            n_waves += 1
        self.assertGreaterEqual(n_waves, 300)
        self.assertGreater(len(seen), n_waves // 2,
                           "the fixtures barely vary -- a stale-buffer graph "
                           "could pass this")

    def test_distinct_inputs_give_distinct_outputs(self):
        """THE STALE-BUFFER CHECK, stated directly."""
        graphed = self._mcts(True)
        outs = []
        for seed in range(300, 340):
            states = _states(WAVE, seed)
            valids = [rule_utl_valid_moves(s.board, s.last_move,
                                           s.mini_winners) for s in states]
            with torch.no_grad():
                probs, values = graphed.graph_wave.run(states, valids)
            outs.append(probs.tobytes())
        self.assertEqual(len(set(outs)), len(outs),
                         "replays returned repeated output for different "
                         "inputs -- the graph is reading a stale buffer")

    def test_run_returns_copies_not_views(self):
        """The pinned outputs are overwritten every replay."""
        graphed = self._mcts(True)
        gw = graphed.graph_wave
        a = _states(WAVE, 401)
        b = _states(WAVE, 402)
        va = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
              for s in a]
        vb = [rule_utl_valid_moves(s.board, s.last_move, s.mini_winners)
              for s in b]
        with torch.no_grad():
            pa, _ = gw.run(a, va)
            first = pa.copy()
            gw.run(b, vb)
        np.testing.assert_array_equal(pa, first)

    def test_non_captured_wave_size_falls_back_and_is_correct(self):
        """k != 8 is 9.3% of waves. It must take the eager path and match."""
        eager, graphed = self._mcts(False), self._mcts(True)
        for k in (1, 2, 5, 7):
            with self.subTest(k=k):
                self.assertFalse(graphed.graph_wave.accepts(k))
                states = _states(k, 500 + k)
                self.assertEqual(self._run(eager, states),
                                 self._run(graphed, states))

    def test_counters_match_the_eager_path(self):
        eager, graphed = self._mcts(False), self._mcts(True)
        states = _states(WAVE, 601)
        self._run(eager, states)
        self._run(graphed, states)
        self.assertEqual(eager.stat_nn_evals, graphed.stat_nn_evals)
        self.assertEqual(eager.stat_nn_batches, graphed.stat_nn_batches)
        self.assertEqual(eager.stat_expansions, graphed.stat_expansions)
        self.assertEqual(eager.stat_probes, graphed.stat_probes)

    def test_a_full_search_agrees_move_for_move(self):
        """End to end at a fixed simulation count, where the two paths must
        make the same decisions -- under a clock they would not, because the
        graph arm gets more simulations, which is the entire point."""
        for seed in (11, 23, 37):
            with self.subTest(seed=seed):
                state = _states(1, seed)[0]
                a = MCTS(self.model, "cuda", n_sims=64, c_puct=1.5,
                         wave_size=WAVE, solve=True, batched_expand=True)
                b = MCTS(self.model, "cuda", n_sims=64, c_puct=1.5,
                         wave_size=WAVE, solve=True, batched_expand=True,
                         graph_wave=True)
                self.assertIsNotNone(b.graph_wave)
                with torch.no_grad():
                    pi_a, _ra = a.search(state.clone())
                    pi_b, _rb = b.search(state.clone())
                np.testing.assert_array_equal(pi_a, pi_b)
                self.assertEqual(a.stat_nn_evals, b.stat_nn_evals)
                self.assertEqual(a.stat_expansions, b.stat_expansions)


class TestDegradation(unittest.TestCase):
    """A performance optimisation must not be able to take the engine down."""

    def test_capture_failure_leaves_a_dud_not_an_exception(self):
        gw = GraphedWave(model=None, device="cpu", k=8)
        self.assertFalse(gw.ok)
        self.assertTrue(gw.reason)
        self.assertFalse(gw.accepts(8))

    def test_mcts_falls_back_to_eager_when_capture_fails(self):
        m = MCTS(model=None, device="cpu", n_sims=1, wave_size=8,
                 batched_expand=True, graph_wave=True)
        self.assertIsNone(m.graph_wave)

    def test_graph_is_off_unless_asked(self):
        m = MCTS(model=None, device="cpu", n_sims=1, wave_size=8,
                 batched_expand=True)
        self.assertIsNone(m.graph_wave)

    def test_graph_requires_batched_expansion(self):
        """The per-leaf path has no wave to replay."""
        m = MCTS(model=None, device="cpu", n_sims=1, wave_size=8,
                 batched_expand=False, graph_wave=True)
        self.assertIsNone(m.graph_wave)


class TestRegistryIsUntouched(unittest.TestCase):
    """The incumbent must not move because this landed.

    The registry-side assertions -- that stripping `graph_wave` reproduces
    every pre-2026-08-09 fingerprint, and that `graph=1` is a declarable
    override -- live in tools/test_engine_registry.py::TestGraphRefreeze, next
    to the values they guard. What belongs here is the one thing about the
    ENGINE: every frozen spec pins the graph off.
    """

    def test_every_frozen_SEARCH_engine_pins_the_graph_flag(self):
        """Pinned, not necessarily off -- `pocket_graph` is a declared
        candidate. What must not exist is a search engine that leaves the flag
        to a code default, because then a changed default moves it silently."""
        from tools import engine_registry
        searched = [n for n in engine_registry.ENGINES
                    if not engine_registry.is_raw(n)]
        self.assertTrue(searched)
        for name in searched:
            with self.subTest(engine=name):
                spec = engine_registry.spec_of(name)
                self.assertTrue("graph=0" in spec or "graph=1" in spec,
                                "%s does not pin the graph flag" % name)

    def test_only_declared_candidates_turn_it_on(self):
        """The graph is still not promoted. `pocket_sel` (#45a) and
        `pocket_defer` (#46) have it on because each is built ON the candidate
        before it -- one declared difference per step -- so they inherit the
        flag rather than granting it. Nothing the ladder or the incumbent
        depends on may have it."""
        from tools import engine_registry as reg
        on = {n for n in reg.ENGINES if reg.ENGINES[n].get("graph") == "1"}
        self.assertEqual(on, {"pocket_graph", "pocket_sel", "pocket_defer"})
        self.assertFalse(on & reg.ANCHOR_ROLES)
        for name in ("final", "pocket_r35", "original", "pocket", "midsize"):
            self.assertNotIn(name, on)

    def test_the_raw_arms_have_no_wave_to_graph(self):
        """sims=0 is the network alone -- no tree, no wave, nothing to
        capture. Pinning a wave flag on it would be noise."""
        from tools import engine_registry
        for name in engine_registry.ENGINES:
            if engine_registry.is_raw(name):
                self.assertNotIn("graph", engine_registry.ENGINES[name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
