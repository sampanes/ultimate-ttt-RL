"""Tests for S5 shared eval server (scripts/eval_server.py).

Run: python -m scripts.test_eval_server

Gates the things that can silently break generation rather than crash loudly:
  1. Batching is numerically faithful -- a row inside a batched forward equals
     that row forwarded alone (the eval server's central correctness claim). If
     this ever fails, every actor's MCTS gets subtly wrong priors/values.
  2. A real eval-server pool actually plays games and returns usable examples
     (CPU, tiny sims, so this stays a unit test and not a benchmark), and a
     weight reload round-trips through the server.
  3. A dead server makes play_block RAISE within a bounded time, never hang.
Speed is NOT tested here -- that needs an idle box and lives in RESULT_S5.md.
The value_tanh reload guard itself is _load_weights, covered by
test_game_actors; the server reuses that exact function.
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from agents.agent_base import ModelConfigCNN, board_to_tensor_from_gamestate
from agents.neural_net_agent_pg import NeuralNetAgentPG
from engine.game import GameState
from scripts.eval_server import EvalServerActorPool
from scripts.train_alphazero import NETWORK_CONFIGS


def _tiny_agent(value_tanh):
    cfg = ModelConfigCNN(**NETWORK_CONFIGS["small"], learning_rate=1e-3,
                         label="test_eval_server", model_dir=tempfile.gettempdir(),
                         value_tanh=value_tanh)
    a = NeuralNetAgentPG(cfg=cfg, model_path=None)
    a.model.to("cpu")
    a.device = "cpu"
    a.model.eval()
    return a


def _random_position_tensors(n, seed=0):
    """Build (n, 7, 9, 9) planes from n distinct reachable positions."""
    import random
    rng = random.Random(seed)
    tensors = []
    while len(tensors) < n:
        s = GameState()
        depth = rng.randint(0, 30)
        for _ in range(depth):
            vm = s.valid_moves()
            if not vm or s.is_over():
                break
            s.make_move(rng.choice(vm))
        if s.is_over():
            continue
        tensors.append(board_to_tensor_from_gamestate(s))
    return torch.stack(tensors)


def _save_weights(agent, path, gen=0, value_tanh=True):
    torch.save({"state_dict": {k: v.detach().cpu()
                               for k, v in agent.model.state_dict().items()},
                "value_tanh": value_tanh, "gen": gen}, path)


def _tiny_cfg(weights, d, value_tanh=True):
    return {
        "network": "small", "device": "cpu", "value_tanh": value_tanh,
        "model_dir": d, "lr": 1e-3, "weights_path": weights,
        "teacher_sims": 8,          # tiny: unit test, not a bench
        "dir_alpha": 0.3, "dir_eps": 0.25, "temperature_moves": 4,
        "mini_tactic_opp": True, "value_blend": 0.0, "greg_mix_depth": 0,
    }


class TestBatchedForwardParity(unittest.TestCase):
    """The server batches many actors' leaves into one forward. In eval() that
    must equal forwarding each leaf alone, or every actor gets wrong targets."""

    def test_batched_equals_per_row(self):
        for value_tanh in (False, True):
            agent = _tiny_agent(value_tanh)
            xs = _random_position_tensors(24, seed=7)
            with torch.no_grad():
                logits_b, values_b = agent.model.forward_both(xs)     # (24,81),(24,)
                for i in range(xs.shape[0]):
                    lo, va = agent.model.forward_both(xs[i])           # (81,), scalar
                    self.assertTrue(
                        torch.allclose(logits_b[i], lo, atol=1e-4, rtol=1e-4),
                        f"row {i} logits differ in a batch (value_tanh={value_tanh})")
                    self.assertTrue(
                        torch.allclose(values_b[i], va, atol=1e-4, rtol=1e-4),
                        f"row {i} value differs in a batch (value_tanh={value_tanh})")


class TestEvalServerPoolEndToEnd(unittest.TestCase):
    """Spawn a real server + 2 CPU actors and play a real (tiny) block."""

    def test_pool_plays_a_block(self):
        agent = _tiny_agent(value_tanh=True)
        d = tempfile.mkdtemp()
        weights = os.path.join(d, "teacher.pt")
        _save_weights(agent, weights, gen=0, value_tanh=True)

        pool = EvalServerActorPool(2, _tiny_cfg(weights, d), start_timeout=300.0)
        try:
            tags = ["heur", None, "rnd", None]
            seeds = [11, 22, 33, 44]
            res = pool.play_block(tags, seeds, timeout=600.0)

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

            # A promotion reload must round-trip through the server (its own
            # weights swap), and the pool must keep playing afterward.
            promoted = _tiny_agent(value_tanh=True)
            p2 = os.path.join(d, "teacher_gen1.pt")
            _save_weights(promoted, p2, gen=1, value_tanh=True)
            pool.reload_weights(p2)
            res2 = pool.play_block(["heur", None], [55, 66], timeout=600.0)
            self.assertEqual(len(res2), 2)
        finally:
            pool.close()


class TestServerDeathRaises(unittest.TestCase):
    """If the one GPU owner dies, play_block must raise (bounded), not hang."""

    def test_dead_server_raises(self):
        agent = _tiny_agent(value_tanh=True)
        d = tempfile.mkdtemp()
        weights = os.path.join(d, "teacher.pt")
        _save_weights(agent, weights, gen=0, value_tanh=True)

        pool = EvalServerActorPool(2, _tiny_cfg(weights, d), start_timeout=300.0)
        try:
            pool._server.terminate()     # simulate the GPU owner crashing
            pool._server.join(timeout=10.0)
            with self.assertRaises(RuntimeError):
                pool.play_block(["heur", None, "rnd", None],
                                [1, 2, 3, 4], timeout=60.0)
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
