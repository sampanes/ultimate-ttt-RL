"""PUCT-style MCTS wrapper around a (policy, value) network.

Compatible with any agent whose `.model` exposes
`forward_both(x) -> (policy_logits[81], value_scalar)` -- currently
NeuralNetAgent3 / NeuralNetAgentPG via the shared ConvNet.

Value sign convention -- VERIFIED CORRECT (2026-06-26, audit + tests):
  The leaf value is the expected outcome for the player to move at that leaf.
  The backup flips sign per ply (zero-sum), so root.Q is the expected outcome
  for the root player. This matches how the value head was trained: in
  `scripts/train_league.py:play_one_game`, rewards are recorded ONLY at the
  active agent's own moves, always from the active player's perspective
  (+1 win / -0.1 draw / -1 loss, +/-0.3 per mini-board), and the value head is
  regressed to that discounted return. So V(s) is the to-move player's expected
  outcome -- exactly what `_expand` returns and `_backup` propagates. The
  per-ply negation in `_best_child` and the `_terminal_value(state, to_play)`
  perspective are consistent with this. `agents/test_mcts.py` locks it in.

KNOWN CAVEAT -- value SCALE mismatch (not a sign error; does NOT invert search):
  The ConvNet value head is `Linear(256, 1)` with NO tanh, trained on a SHAPED,
  discounted return, so its output is unbounded and NOT calibrated to [-1, 1].
  But terminal leaves here use a clean +/-1.0 / 0.0 (`_terminal_value`). MCTS
  therefore mixes two scales: a net leaf rated, say, +1.8 (won a couple of minis,
  expects to win) will outrank a PROVEN terminal win (+1.0), which can pull search
  toward shaky shaped positions over certain wins, and means c_puct was never
  tuned against a clean value scale. This degrades quality, it does not flip the
  search. Fix: retrain with --value_tanh (scripts/train_alphazero.py or
  scripts/train_league.py --value_tanh) so the value head outputs a calibrated
  [-1, 1] result and this comment no longer applies.
  Measure impact with `scripts/benchmark_vs_mcts.py` before tuning c_puct.

Batched leaf evaluation (wave_size > 1):
  When wave_size > 1, each round of the search loop collects `wave_size` leaf
  nodes via virtual loss, then evaluates all leaves in ONE batched forward pass.
  Virtual loss temporarily adds N+1 / W+1 to visited nodes (W is stored from the
  child's to-play perspective and selection scores -Q, so RAISING W penalizes
  the path -- see the sign-bug comment in _run_wave) so each selection in the
  same wave diverges to a different path. After the batch eval the VL is undone
  and real backups are applied. wave_size=1 is the original serial path
  (byte-identical to previous behaviour). wave_size=8 is a good default for the
  oracle benchmark; 16-32 helps for AlphaZero self-play at high n_sims.
  HISTORY: from its introduction until 2026-07-04 the VL sign was inverted
  (W-1), which collapsed whole waves onto a single line and made wave>1 search
  WEAKER than the raw policy -- it poisoned the M4a/M4b AlphaZero runs' visit
  targets. agents/test_mcts.py now locks the wave path too.
"""

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from .agent_base import Agent, board_to_tensor_from_gamestate, wave_planes
from engine.rules import rule_utl_valid_moves
from engine.constants import X, O, DRAW


class MCTSNode:
    __slots__ = ("parent", "children", "prior", "N", "W",
                 "move", "to_play", "is_terminal", "terminal_value")

    def __init__(self, parent=None, prior=0.0, move=None, to_play=None):
        self.parent = parent
        self.children = {}
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.move = move
        self.to_play = to_play
        self.is_terminal = False
        self.terminal_value = 0.0

    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, c_puct, parent_N):
        return c_puct * self.prior * math.sqrt(parent_N) / (1 + self.N)


_DEEPCOPY_WARNED = False


def _clone(state):
    """Fast clone for MCTS rollouts. Uses state.clone() if available."""
    if hasattr(state, "clone"):
        return state.clone()
    global _DEEPCOPY_WARNED
    if not _DEEPCOPY_WARNED:
        # ASCII-only: this can print under a redirected/cp1252 stdout, where a
        # non-ASCII glyph raises UnicodeEncodeError (same class as the engine and
        # train_league import-banner fixes).
        print("[!] MCTS falling back to deepcopy for GameState -- add a clone() "
              "method to the active engine for ~20-50x speedup.")
        _DEEPCOPY_WARNED = True
    return deepcopy(state)


class MCTS:
    _VL = 1.0        # virtual loss magnitude (standard AlphaZero)
    _MIN_WAVES = 16  # floor on waves per search: the tree only deepens between
                     # waves (leaf expansion is deferred to the batched forward
                     # pass), so wave_size ~ n_sims degenerates into a one-ply
                     # breadth probe. Empirical (2026-07-04, edge vs raw net,
                     # after the VL sign fix): 38 waves 0.95, 16 waves 0.80-0.925,
                     # 10 waves 0.70-0.80, 5 waves 0.375-0.80, 1 wave 0.00.

    def __init__(self, model, device, n_sims=100, c_puct=1.5,
                 add_dirichlet_at_root=False, dir_alpha=0.3, dir_eps=0.25,
                 wave_size=1):
        self.model    = model
        self.device   = device
        self.n_sims   = n_sims
        self.c_puct   = c_puct
        self.add_dirichlet = add_dirichlet_at_root
        self.dir_alpha = dir_alpha
        self.dir_eps   = dir_eps
        self.wave_size = max(1, int(wave_size))

    @torch.no_grad()
    def search(self, root_state):
        root = MCTSNode(to_play=root_state.player)
        self._expand(root, root_state, add_noise=self.add_dirichlet)

        if self.wave_size == 1:
            # Original single-sim path -- byte-identical to prior behaviour.
            for _ in range(self.n_sims):
                node  = root
                state = _clone(root_state)

                while node.children and not node.is_terminal:
                    node = self._best_child(node)
                    state.make_move(node.move)
                    if state.winner is not None:
                        node.is_terminal    = True
                        node.terminal_value = self._terminal_value(state, node.to_play)
                        break

                if not node.is_terminal:
                    value = self._expand(node, state, add_noise=False)
                else:
                    value = node.terminal_value

                self._backup(node, value)
        else:
            # Clamp so every search gets at least _MIN_WAVES waves of depth.
            eff_wave = max(1, min(self.wave_size, self.n_sims // self._MIN_WAVES))
            sims_done = 0
            while sims_done < self.n_sims:
                wave = min(eff_wave, self.n_sims - sims_done)
                self._run_wave(root, root_state, wave)
                sims_done += wave

        pi = np.zeros(81, dtype=np.float32)
        for mv, child in root.children.items():
            pi[mv] = child.N
        s = pi.sum()
        if s > 0:
            pi /= s
        return pi, root

    # ------------------------------------------------------------------
    # Batched wave (wave_size > 1 path)
    # ------------------------------------------------------------------

    def _run_wave(self, root, root_state, wave_size):
        """Select `wave_size` leaves with virtual loss, batch-evaluate, backup.

        Each selection applies virtual loss (N+1, W+VL) to visited nodes so
        subsequent selections in the same wave score those paths lower and
        diverge. After the batch forward pass the VL is undone before real
        backup, leaving tree statistics as if each sim ran independently.
        """
        pending = []    # (leaf_node, leaf_state, path_nodes_that_got_vl)

        for _ in range(wave_size):
            node  = root
            state = _clone(root_state)
            path  = []

            while node.children and not node.is_terminal:
                node = self._best_child(node)
                state.make_move(node.move)
                # Virtual loss. SIGN MATTERS: this tree stores W from the CHILD's
                # to-play perspective and _best_child scores -c.Q(), so to make a
                # visited path look BAD to the next sim we must RAISE the child's
                # W (a virtual win for the child = a loss for the selecting
                # parent). W -= VL here inverts that: Q -> -1, -Q -> +1, and the
                # whole wave collapses onto one path (found 2026-07-04: made
                # wave=64 search score 0.000 vs the raw league net; wave=1
                # scored 0.875).
                node.N += 1
                node.W += self._VL
                path.append(node)
                if state.winner is not None:
                    node.is_terminal    = True
                    node.terminal_value = self._terminal_value(state, node.to_play)
                    break

            pending.append((node, state, path))

        # Collect unique unexpanded non-terminal leaves for the batch forward pass.
        # Dedup by node identity: two sims landing on the same unexpanded leaf
        # (possible in a shallow tree early in search) expand it once.
        seen    = {}    # id(node) -> index in to_eval
        to_eval = []    # (pending_idx, node, state)
        for i, (node, state, _) in enumerate(pending):
            if not node.is_terminal and not node.children:
                nid = id(node)
                if nid not in seen:
                    seen[nid] = len(to_eval)
                    to_eval.append((i, node, state))

        # Batch forward pass for all unique leaves.
        leaf_values = {}    # id(node) -> float
        if to_eval:
            # S8: fill all K leaves into one buffer (C++ fill_planes when
            # available) instead of K per-leaf builds + torch.stack.
            xs = wave_planes([s for _, _, s in to_eval], self.device)  # (K,7,9,9)

            logits_b, values_b = self.model.forward_both(xs)
            # forward_both squeezes batch=1 to (81,)/scalar; restore dims.
            if logits_b.dim() == 1:
                logits_b = logits_b.unsqueeze(0)
                values_b = values_b.unsqueeze(0)

            for k, (_, node, state) in enumerate(to_eval):
                self._expand_from_logits(node, state, logits_b[k], add_noise=False)
                leaf_values[id(node)] = float(values_b[k].item())

        # Undo virtual loss then apply real backup for each pending sim.
        for node, state, path in pending:
            for n in path:
                n.N -= 1
                n.W -= self._VL

            if node.is_terminal:
                value = node.terminal_value
            elif id(node) in leaf_values:
                value = leaf_values[id(node)]
            else:
                # Node already had children (expanded by a prior wave or by a
                # sibling sim in this wave) -- use its current Q as estimate.
                value = node.Q()

            self._backup(node, value)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _expand_from_logits(self, node, state, logits, add_noise=False):
        """Populate node.children from pre-computed 1-D (81,) policy logits."""
        valid   = rule_utl_valid_moves(state.board, state.last_move, state.mini_winners)
        mask    = torch.full((81,), float('-inf'), device=logits.device)
        valid_t = torch.tensor(valid, device=logits.device, dtype=torch.long)
        mask.scatter_(0, valid_t, logits[valid_t])
        probs   = F.softmax(mask, dim=0).cpu().numpy()

        if add_noise and len(valid) > 1:
            noise = np.random.dirichlet([self.dir_alpha] * len(valid))
            for i, mv in enumerate(valid):
                probs[mv] = (1 - self.dir_eps) * probs[mv] + self.dir_eps * noise[i]

        next_to_play = O if state.player == X else X
        for mv in valid:
            node.children[mv] = MCTSNode(parent=node, prior=float(probs[mv]),
                                         move=mv, to_play=next_to_play)

    def _expand(self, node, state, add_noise=False):
        """Run one network forward pass for a single leaf, expand it, return value."""
        x = board_to_tensor_from_gamestate(state).unsqueeze(0).to(self.device)
        logits, value = self.model.forward_both(x)
        # forward_both squeezes batch=1 -> (81,) and scalar; both correct here.
        self._expand_from_logits(node, state, logits, add_noise=add_noise)
        # NOTE: `value` is the net's UNBOUNDED shaped-return estimate, not a
        # calibrated [-1, 1] outcome (see module docstring "value SCALE mismatch").
        # It is compared against clean +/-1.0 terminal values in backup/Q.
        return float(value.item())

    def _best_child(self, node):
        # child.Q() is stored from the CHILD's to_play perspective (backup flips
        # sign per ply). The child's mover is the opponent of `node`'s mover, so to
        # score a move from `node`'s perspective we must NEGATE the child's Q.
        # Using +c.Q() here would pick the move that is best for the opponent.
        return max(node.children.values(),
                   key=lambda c: -c.Q() + c.U(self.c_puct, node.N))

    def _backup(self, node, leaf_value):
        v = leaf_value
        while node is not None:
            node.N += 1
            node.W += v
            v = -v
            node = node.parent

    @staticmethod
    def _terminal_value(state, to_play):
        if state.winner in (DRAW, None):
            return 0.0
        return 1.0 if state.winner == to_play else -1.0


class MCTSAgent(Agent):
    """Drop-in arena/eval agent: wraps a trained NN with PUCT search."""

    def __init__(self, nn_agent, n_sims=100, c_puct=1.5, temperature=0.0,
                 add_dirichlet_at_root=False, dir_alpha=0.3, dir_eps=0.25,
                 wave_size=1):
        super().__init__(f"MCTS({nn_agent.name}, n={n_sims})")
        self.nn   = nn_agent
        self.mcts = MCTS(
            model=nn_agent.model,
            device=nn_agent.device,
            n_sims=n_sims,
            c_puct=c_puct,
            add_dirichlet_at_root=add_dirichlet_at_root,
            dir_alpha=dir_alpha,
            dir_eps=dir_eps,
            wave_size=wave_size,
        )
        self.temperature = temperature

    def select_move(self, gamestate):
        pi, _ = self.mcts.search(gamestate)
        if self.temperature == 0.0:
            return int(np.argmax(pi))
        probs  = pi ** (1.0 / self.temperature)
        probs /= probs.sum()
        return int(np.random.choice(81, p=probs))
