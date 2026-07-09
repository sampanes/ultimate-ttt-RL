"""Regression tests for the expert-iteration v2 safety gates."""

import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from scripts.expert_iter import ShardStore, _decayed_lr, _promotion_decision
from scripts.train_alphazero import (
    apply_dihedral_symmetry,
    _mini_tactical_target,
    policy_value_loss,
)
from engine.constants import EMPTY, X, O
from engine.game import _PyGameState


class _FixedModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward_both(self, xs):
        batch = xs.shape[0]
        return self.logits.unsqueeze(0).expand(batch, -1), torch.zeros(batch)


class ExpertIterationTests(unittest.TestCase):
    def test_policy_loss_ignores_logits_on_inference_masked_moves(self):
        xs = torch.zeros((1, 7, 9, 9))
        xs[0, 3, 0, 0] = 1.0
        xs[0, 3, 0, 1] = 1.0
        pis = torch.zeros((1, 81))
        pis[0, 0] = 0.75
        pis[0, 1] = 0.25
        zs = torch.zeros(1)

        logits_a = [0.0] * 81
        logits_b = list(logits_a)
        logits_b[80] = 10_000.0
        loss_a = policy_value_loss(
            _FixedModel(logits_a), xs, pis, zs, 1.0)[1]
        loss_b = policy_value_loss(
            _FixedModel(logits_b), xs, pis, zs, 1.0)[1]
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=6)

    def test_policy_loss_rejects_illegal_target_mass(self):
        xs = torch.zeros((1, 7, 9, 9))
        xs[0, 3, 0, 0] = 1.0
        pis = torch.zeros((1, 81))
        pis[0, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "illegal move"):
            policy_value_loss(
                _FixedModel([0.0] * 81), xs, pis, torch.zeros(1), 1.0)

    def test_mini_tactical_target_is_opt_in_winblock_signal(self):
        board = [EMPTY] * 81
        board[0] = O
        board[1] = O
        state = _PyGameState(board=board, player=X, last_move=None,
                             mini_winners=[EMPTY] * 9, winner=None)
        valid = list(range(81))

        pi, move, kind = _mini_tactical_target(state, valid)

        self.assertEqual(kind, "block")
        self.assertEqual(move, 2)
        self.assertEqual(float(pi.sum()), 1.0)
        self.assertEqual(float(pi[2]), 1.0)

    def test_symmetry_moves_state_and_policy_together(self):
        xs = torch.zeros((1, 7, 9, 9))
        pis = torch.zeros((1, 81))
        xs[0, 0, 1, 2] = 1.0
        xs[0, 3, 1, 2] = 1.0
        pis[0, 1 * 9 + 2] = 1.0

        for symmetry in range(8):
            tx, tp = apply_dihedral_symmetry(xs, pis, symmetry)
            state_pos = torch.nonzero(tx[0, 0], as_tuple=False)
            legal_pos = torch.nonzero(tx[0, 3], as_tuple=False)
            policy_pos = torch.nonzero(
                tp.reshape(1, 9, 9)[0], as_tuple=False)
            self.assertTrue(torch.equal(state_pos, legal_pos))
            self.assertTrue(torch.equal(state_pos, policy_pos))
            self.assertEqual(float(tp.sum()), 1.0)

    def test_resume_window_excludes_old_teacher_generations(self):
        example = (
            torch.zeros((7, 9, 9)),
            np.eye(1, 81, dtype=np.float32)[0],
            0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = ShardStore(tmp)
            store.write(0, [example], teacher_gen=0)
            store.write(1, [example, example], teacher_gen=1)
            loaded = store.load_window(1, 10, teacher_gen=1)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(
                store.load_window(1, 10, teacher_gen=2), [])

    def test_decayed_lr_halves_and_floors(self):
        self.assertAlmostEqual(_decayed_lr(1e-3, 0, 25_000, 1e-4), 1e-3)
        self.assertAlmostEqual(_decayed_lr(1e-3, 25_000, 25_000, 1e-4), 5e-4)
        self.assertAlmostEqual(_decayed_lr(1e-3, 50_000, 25_000, 1e-4), 2.5e-4)
        # far past the horizon, the floor holds
        self.assertAlmostEqual(_decayed_lr(1e-3, 500_000, 25_000, 1e-4), 1e-4)
        # 0 disables decay entirely
        self.assertAlmostEqual(_decayed_lr(1e-3, 500_000, 0, 1e-4), 1e-3)

    def test_promotion_requires_head_to_head_and_absolute_progress(self):
        promote, failed = _promotion_decision(
            head_to_head=0.60,
            heur_score=0.32,
            random_score=0.70,
            best_heur=0.28,
            best_random=0.71,
            threshold=0.55,
            absolute_margin=0.02,
            random_tolerance=0.03,
        )
        self.assertTrue(promote)
        self.assertEqual(failed, [])

        promote, failed = _promotion_decision(
            head_to_head=0.60,
            heur_score=0.29,
            random_score=0.70,
            best_heur=0.28,
            best_random=0.71,
            threshold=0.55,
            absolute_margin=0.02,
            random_tolerance=0.03,
        )
        self.assertFalse(promote)
        self.assertIn("winblock", failed)

        promote, failed = _promotion_decision(
            head_to_head=0.60,
            heur_score=1.0,
            random_score=0.99,
            best_heur=1.0,
            best_random=1.0,
            threshold=0.55,
            absolute_margin=0.02,
            random_tolerance=0.03,
        )
        self.assertTrue(promote)
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
