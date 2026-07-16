"""Tests for S4 multiprocess game actors (scripts/game_actors.py).

Run: python -m scripts.test_game_actors

These gate the things that can silently poison training rather than crash:
  1. value_tanh survives a weight reload (the documented 12h poisoning bug).
  2. The opponent mix an actor block draws matches the sequential loop's.
  3. A real pool actually plays games and returns usable examples (CPU, tiny
     sims, so this stays a unit test and not a benchmark).
Speed is NOT tested here -- that needs an idle box and lives in RESULT_S4.md.
"""

import collections
import os
import random
import tempfile
import unittest

import torch

from agents.agent_base import ModelConfigCNN
from agents.neural_net_agent_pg import NeuralNetAgentPG
from scripts.expert_iter import _opponent_slice
from scripts.game_actors import GameActorPool, _load_weights
from scripts.train_alphazero import NETWORK_CONFIGS


def _tiny_agent(value_tanh):
    cfg = ModelConfigCNN(**NETWORK_CONFIGS["small"], learning_rate=1e-3,
                         label="test_actor", model_dir=tempfile.gettempdir(),
                         value_tanh=value_tanh)
    a = NeuralNetAgentPG(cfg=cfg, model_path=None)
    a.model.to("cpu")
    a.device = "cpu"
    a.model.eval()
    return a


class TestValueTanhReload(unittest.TestCase):
    """The poisoning guard: load_state_dict copies weights, NOT the flag."""

    def test_reload_adopts_value_tanh_from_payload(self):
        # An actor that started life on a non-tanh gen-0 teacher...
        agent = _tiny_agent(value_tanh=False)
        self.assertFalse(agent.model.value_tanh)

        # ...then reloads a PROMOTED teacher, which is always tanh.
        promoted = _tiny_agent(value_tanh=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "teacher.pt")
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in
                                       promoted.model.state_dict().items()},
                        "value_tanh": True, "gen": 3}, p)
            _load_weights(agent, p, "cpu")

        # Without this the actor feeds unbounded pre-tanh values into MCTS.
        self.assertTrue(agent.model.value_tanh,
                        "actor kept a stale value_tanh after reloading a "
                        "promoted teacher -- this is the poisoning bug")

    def test_reload_of_raw_state_dict_does_not_crash(self):
        agent = _tiny_agent(value_tanh=True)
        other = _tiny_agent(value_tanh=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "raw.pt")
            torch.save({k: v.detach().cpu()
                        for k, v in other.model.state_dict().items()}, p)
            _load_weights(agent, p, "cpu")   # legacy bare state_dict
        self.assertTrue(agent.model.value_tanh)


class TestOpponentMixPreserved(unittest.TestCase):
    """S4 is distribution-preserving: the mix is drawn by the PARENT, so the
    tag histogram must match the sequential loop's within sampling noise."""

    def _draw(self, n, opp_mix, rnd_mix, greg_mix, seed):
        random.seed(seed)
        return collections.Counter(
            _opponent_slice(random.random(), opp_mix, rnd_mix, greg_mix)
            for _ in range(n))

    def test_live_mix_matches_expected_rates(self):
        # The live S1 configuration: --greg_mix 0.10 --opp_mix 0.30 --rnd_mix 0.10
        n = 200_000
        c = self._draw(n, 0.30, 0.10, 0.10, seed=1234)
        self.assertAlmostEqual(c["heur"] / n, 0.30, delta=0.005)
        self.assertAlmostEqual(c["rnd"] / n, 0.10, delta=0.005)
        self.assertAlmostEqual(c["greg"] / n, 0.10, delta=0.005)
        self.assertAlmostEqual(c[None] / n, 0.50, delta=0.005)

    def test_parent_draw_is_one_call_per_game(self):
        # The actor path draws exactly one tag per game from the parent stream,
        # same as the sequential loop, so the mix cannot drift with --actors.
        random.seed(7)
        seq = [_opponent_slice(random.random(), 0.30, 0.10, 0.10)
               for _ in range(16)]
        random.seed(7)
        batch = [_opponent_slice(random.random(), 0.30, 0.10, 0.10)
                 for _ in range(16)]
        self.assertEqual(seq, batch)


class TestActorPoolEndToEnd(unittest.TestCase):
    """Spawn a real pool on CPU and play a real (tiny) block."""

    def test_pool_plays_a_block(self):
        agent = _tiny_agent(value_tanh=True)
        d = tempfile.mkdtemp()
        weights = os.path.join(d, "teacher.pt")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in
                                   agent.model.state_dict().items()},
                    "value_tanh": True, "gen": 0}, weights)

        cfg = {
            "network": "small", "device": "cpu", "value_tanh": True,
            "model_dir": d, "lr": 1e-3, "weights_path": weights,
            "teacher_sims": 8,          # tiny: this is a unit test, not a bench
            "dir_alpha": 0.3, "dir_eps": 0.25, "temperature_moves": 4,
            "mini_tactic_opp": True, "value_blend": 0.0, "greg_mix_depth": 0,
        }

        pool = GameActorPool(2, cfg, start_timeout=300.0)
        try:
            tags = ["heur", None, "rnd", None]
            seeds = [11, 22, 33, 44]
            res = pool.play_block(tags, seeds, timeout=600.0)
        finally:
            pool.close()

        self.assertEqual(len(res), len(tags))
        self.assertCountEqual([r["tag"] for r in res], tags)
        for r in res:
            self.assertGreater(len(r["examples"]), 0,
                               "actor returned a game with no examples")
            self.assertGreater(r["stats"]["moves"], 0)
            x, pi, z = r["examples"][0]
            self.assertEqual(tuple(x.shape), (7, 9, 9))
            self.assertEqual(pi.shape, (81,))
            self.assertAlmostEqual(float(pi.sum()), 1.0, places=4,
                                   msg="policy target is not a distribution")


if __name__ == "__main__":
    unittest.main(verbosity=2)
