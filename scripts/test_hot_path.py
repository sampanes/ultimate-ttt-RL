"""Tests for the S8 per-sim hot path (C++ fill_planes + vectorized wave build).

Run: python -m scripts.test_hot_path

S8 is a SPEED change that must be byte-identical to the pure-numpy plane build
it replaces -- any drift silently changes every MCTS prior/value and poisons
training. These gate that:
  1. fill_planes == the pure-numpy board_to_tensor, byte-for-byte, over many
     reachable positions.
  2. wave_planes (the batched buffer fill used in the MCTS wave) == stacking the
     per-leaf numpy build.
Both SKIP cleanly when the pure-Python engine is active (no fill_planes), so the
suite passes on any machine; the parity that matters runs where the C++ engine
is built (the training/authoring box).
"""

import random
import unittest

import numpy as np
import torch

import agents.agent_base as ab
from agents.agent_base import board_to_tensor_from_gamestate, wave_planes
from engine.game import GameState


def _has_fill_planes():
    return hasattr(GameState(), "fill_planes")


def _reachable(rng, max_depth=40):
    s = GameState()
    for _ in range(rng.randint(0, max_depth)):
        vm = s.valid_moves()
        if not vm or s.is_over():
            break
        s.make_move(rng.choice(vm))
    return s


def _numpy_reference(state):
    """board_to_tensor via the pure-numpy path, bypassing the C++ fast path."""
    saved = ab._FILL_PLANES
    ab._FILL_PLANES = False        # force the fallback for the reference
    try:
        return board_to_tensor_from_gamestate(state).numpy().copy()
    finally:
        ab._FILL_PLANES = saved


@unittest.skipUnless(_has_fill_planes(),
                     "pure-Python engine: no fill_planes to check")
class TestFillPlanesParity(unittest.TestCase):

    def test_fill_planes_byte_identical(self):
        rng = random.Random(20260716)
        checked = 0
        for _ in range(3000):
            s = _reachable(rng)
            if s.is_over():
                continue
            ref = _numpy_reference(s)
            buf = np.empty((7, 9, 9), dtype=np.float32)
            s.fill_planes(buf)
            self.assertTrue(np.array_equal(ref, buf),
                            f"fill_planes drifted from the numpy build:\n"
                            f"{np.argwhere(ref != buf)[:5]}")
            checked += 1
        self.assertGreater(checked, 500, "not enough non-terminal positions")


@unittest.skipUnless(_has_fill_planes(),
                     "pure-Python engine: no fill_planes to check")
class TestWavePlanesParity(unittest.TestCase):

    def test_wave_equals_per_leaf_numpy(self):
        rng = random.Random(4242)
        states = [_reachable(rng) for _ in range(24)]
        states = [s for s in states if not s.is_over()]
        ref = np.stack([_numpy_reference(s) for s in states])
        got = wave_planes(states, "cpu").numpy()
        self.assertEqual(got.shape, ref.shape)
        self.assertTrue(np.array_equal(ref, got),
                        "wave_planes drifted from the per-leaf numpy build")


if __name__ == "__main__":
    unittest.main(verbosity=2)
