"""Regression tests for the expert-iteration v2 safety gates."""

import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from scripts.expert_iter import (ShardStore, _decayed_lr, _opponent_slice,
                                 _promote_teacher, _promotion_decision)
from scripts.train_alphazero import (
    apply_dihedral_symmetry,
    _blend_value,
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


class _TanhShell(nn.Module):
    """Minimal model with the value_tanh runtime flag (a plain attribute,
    deliberately NOT part of state_dict -- that is the bug class under test)."""

    def __init__(self, value_tanh, fill):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        with torch.no_grad():
            self.lin.weight.fill_(fill)
        self.value_tanh = value_tanh


class _Holder:
    def __init__(self, model):
        self.model = model


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

    def test_gregory_gate_self_arms(self):
        base = dict(head_to_head=0.60, heur_score=0.32, random_score=0.70,
                    best_heur=0.28, best_random=0.71, threshold=0.55,
                    absolute_margin=0.02, random_tolerance=0.03)
        # Unarmed (best below gregory_arm): even a to-zero drop is
        # noise-floor and must stay report-only.
        promote, failed = _promotion_decision(
            **base, gregory_score=0.0, best_gregory=0.05,
            gregory_tolerance=0.03, gregory_arm=0.10)
        self.assertTrue(promote)
        self.assertEqual(failed, [])
        # Armed: a regression beyond tolerance blocks promotion.
        promote, failed = _promotion_decision(
            **base, gregory_score=0.10, best_gregory=0.20,
            gregory_tolerance=0.03, gregory_arm=0.10)
        self.assertFalse(promote)
        self.assertIn("gregory_regression", failed)
        # Armed but within tolerance: passes.
        promote, failed = _promotion_decision(
            **base, gregory_score=0.18, best_gregory=0.20,
            gregory_tolerance=0.03, gregory_arm=0.10)
        self.assertTrue(promote)
        # Legacy call without gregory arguments is unchanged.
        promote, failed = _promotion_decision(**base)
        self.assertTrue(promote)

    def test_gregory_hard_gate_self_arms_independently(self):
        # d3 gate is armed and clearly passing throughout, so any failure here
        # is attributable to the deeper d4 anchor alone.
        base = dict(head_to_head=0.60, heur_score=0.32, random_score=0.70,
                    best_heur=0.28, best_random=0.71, threshold=0.55,
                    absolute_margin=0.02, random_tolerance=0.03,
                    gregory_score=0.30, best_gregory=0.20,
                    gregory_tolerance=0.03, gregory_arm=0.10)
        # Hard anchor unarmed (best below arm): a to-zero drop is noise-floor
        # and stays report-only -- promotion still passes.
        promote, failed = _promotion_decision(
            **base, gregory_hard_score=0.0, best_gregory_hard=0.05)
        self.assertTrue(promote)
        self.assertEqual(failed, [])
        # Hard anchor armed: a regression beyond tolerance blocks, on its own
        # named gate (independent of the shallow gregory gate).
        promote, failed = _promotion_decision(
            **base, gregory_hard_score=0.10, best_gregory_hard=0.20)
        self.assertFalse(promote)
        self.assertIn("gregory_hard_regression", failed)
        self.assertNotIn("gregory_regression", failed)
        # Hard anchor armed but within tolerance: passes.
        promote, failed = _promotion_decision(
            **base, gregory_hard_score=0.18, best_gregory_hard=0.20)
        self.assertTrue(promote)
        # Omitting the hard-anchor arguments leaves the decision unchanged
        # (disabled second anchor is the default path).
        promote, failed = _promotion_decision(**base)
        self.assertTrue(promote)

    def test_opponent_slice_layout_is_stable(self):
        # Legacy two-slice layout: greg_mix omitted must reproduce the pre-S1
        # branch arithmetic exactly (x + 0.0 == x, so identical thresholds).
        self.assertEqual(_opponent_slice(0.0, 0.35, 0.15), "heur")
        self.assertEqual(_opponent_slice(0.34, 0.35, 0.15), "heur")
        self.assertEqual(_opponent_slice(0.36, 0.35, 0.15), "rnd")
        self.assertEqual(_opponent_slice(0.45, 0.35, 0.15), "rnd")
        self.assertIsNone(_opponent_slice(0.51, 0.35, 0.15))
        # S1 suggested first segment (0.30/0.10/0.10): the gregory slice sits
        # between random and self-play; winblock/random boundaries only move
        # via their OWN mix values, never because greg_mix was enabled.
        self.assertEqual(_opponent_slice(0.05, 0.30, 0.10, 0.10), "heur")
        self.assertEqual(_opponent_slice(0.29, 0.30, 0.10, 0.10), "heur")
        self.assertEqual(_opponent_slice(0.31, 0.30, 0.10, 0.10), "rnd")
        self.assertEqual(_opponent_slice(0.39, 0.30, 0.10, 0.10), "rnd")
        self.assertEqual(_opponent_slice(0.41, 0.30, 0.10, 0.10), "greg")
        self.assertEqual(_opponent_slice(0.49, 0.30, 0.10, 0.10), "greg")
        self.assertIsNone(_opponent_slice(0.51, 0.30, 0.10, 0.10))
        self.assertIsNone(_opponent_slice(0.99, 0.30, 0.10, 0.10))
        # greg_mix=0 collapses the gregory slice to empty at any r.
        self.assertIsNone(_opponent_slice(0.50, 0.35, 0.15, 0.0))

    def test_blend_value_targets(self):
        # lam=0 and tactics-shortcut positions (q_root=None, no search ran)
        # keep the pure game outcome z.
        self.assertEqual(_blend_value(1.0, 0.5, 0.0), 1.0)
        self.assertEqual(_blend_value(-1.0, None, 0.4), -1.0)
        # Blend is (1-lam)*z + lam*q_root, both in the same (root player's)
        # perspective.
        self.assertAlmostEqual(_blend_value(1.0, 0.5, 0.4), 0.8)
        self.assertAlmostEqual(_blend_value(-1.0, 0.25, 0.4), -0.5)
        self.assertAlmostEqual(_blend_value(0.0, -0.5, 0.5), -0.25)
        # Defensive clamp: an unbounded (untanhed) root value cannot poison
        # the targets -- the 2026-07-09 value-scale failure class.
        self.assertAlmostEqual(_blend_value(1.0, 3.27, 0.5), 1.0)
        self.assertAlmostEqual(_blend_value(1.0, -3.0, 0.5), 0.0)

    def test_promote_teacher_flips_value_tanh_with_weights(self):
        # The v2 run's fourth failure mode: promotion copied weights via
        # load_state_dict but value_tanh is a runtime attribute, so the live
        # teacher kept feeding unbounded pre-tanh values into MCTS generation.
        teacher = _Holder(_TanhShell(value_tanh=False, fill=0.0))
        student = _Holder(_TanhShell(value_tanh=True, fill=1.0))
        teacher_tanh = _promote_teacher(teacher, student)
        self.assertTrue(teacher_tanh)
        self.assertTrue(teacher.model.value_tanh)
        self.assertTrue(torch.equal(teacher.model.lin.weight,
                                    student.model.lin.weight))

    # ---- shard retention (append-only disk-leak fix) --------------------- #
    def _mk_examples(self, n, ztag=0.0):
        return [(torch.zeros(7, 9, 9),
                 np.zeros(81, dtype=np.float32),
                 float(ztag)) for _ in range(n)]

    def test_shard_prune_tail_keeps_newest_contiguous_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            store = ShardStore(d)
            for i in range(20):
                store.write(i, self._mk_examples(10), teacher_gen=1)
            per = store._shard_files()[0][2]   # shards are byte-identical here
            # Budget for ~3 shards; floor of 1 so the byte budget governs.
            deleted, freed = store.prune_tail(3 * per - 1, min_keep_shards=1)
            kept = [idx for idx, _p, _s in store._shard_files()]
            self.assertEqual(kept, [17, 18, 19])
            self.assertEqual(deleted, 17)
            self.assertEqual(freed, 17 * per)

    def test_shard_prune_tail_respects_min_keep(self):
        with tempfile.TemporaryDirectory() as d:
            store = ShardStore(d)
            for i in range(20):
                store.write(i, self._mk_examples(10), teacher_gen=1)
            # A zero budget would keep 1; the floor forces the newest 5.
            deleted, _freed = store.prune_tail(0, min_keep_shards=5)
            kept = [idx for idx, _p, _s in store._shard_files()]
            self.assertEqual(kept, [15, 16, 17, 18, 19])
            self.assertEqual(deleted, 15)

    def test_shard_prune_tail_noop_when_at_or_under_min_keep(self):
        with tempfile.TemporaryDirectory() as d:
            store = ShardStore(d)
            for i in range(4):
                store.write(i, self._mk_examples(5), teacher_gen=1)
            self.assertEqual(store.prune_tail(0, min_keep_shards=64), (0, 0))
            self.assertEqual(len(store._shard_files()), 4)

    def test_shard_prune_preserves_the_recent_load_window(self):
        # The safety invariant: pruning must never drop a shard load_window
        # would read. load_window reloads `window` CURRENT-gen positions
        # newest-first, so keeping a few-x-window tail leaves it intact.
        with tempfile.TemporaryDirectory() as d:
            store = ShardStore(d)
            for i in range(30):   # 30 shards x 10 positions, tagged by shard
                store.write(i, self._mk_examples(10, ztag=i), teacher_gen=2)
            per = store._shard_files()[0][2]
            store.prune_tail(20 * per, min_keep_shards=1)   # keep ~20 shards
            window = 50
            reloaded = store.load_window(29, window, teacher_gen=2)
            self.assertEqual(len(reloaded), window)
            tags = {int(z) for _x, _pi, z in reloaded}
            self.assertGreaterEqual(min(tags), 25)   # newest 5 shards = 50 pos
            self.assertLessEqual(max(tags), 29)

    def test_prune_tail_to_window_uses_the_window_budget(self):
        with tempfile.TemporaryDirectory() as d:
            store = ShardStore(d)
            for i in range(30):
                store.write(i, self._mk_examples(10), teacher_gen=3)
            # keep_bytes = (size/10) * window(50) * mult(2) = size * 10
            #   -> keep the newest 10 shards.
            deleted, _freed = store.prune_tail_to_window(
                29, 10, window=50, mult=2.0, min_keep_shards=1)
            kept = [idx for idx, _p, _s in store._shard_files()]
            self.assertEqual(kept, list(range(20, 30)))
            self.assertEqual(deleted, 20)

    def test_prune_tail_to_window_disabled_paths_are_noops(self):
        with tempfile.TemporaryDirectory() as d:
            store = ShardStore(d)
            for i in range(5):
                store.write(i, self._mk_examples(4), teacher_gen=1)
            self.assertEqual(store.prune_tail_to_window(4, 4, 50, 0.0), (0, 0))
            self.assertEqual(store.prune_tail_to_window(4, 0, 50, 5.0), (0, 0))
            self.assertEqual(len(store._shard_files()), 5)


if __name__ == "__main__":
    unittest.main()
