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

Solved-node propagation (solve=True, OFF by default):
  Exact game-theoretic status, tracked SEPARATELY from N/W/Q so it can never
  contaminate the averaged neural estimate. `node.solved` is None (unresolved)
  or +1 / 0 / -1 = proven win / draw / loss FOR THAT NODE'S to_play -- the same
  perspective convention as `terminal_value`, so the two are interchangeable at
  backup time.

  Propagation (`_solve_from_children`) is standard proof-number-free AND/OR
  backward induction: a node is a proven WIN as soon as any child is a proven
  loss for its own mover; a proven LOSS only once EVERY legal child is a proven
  win for the opponent; a proven DRAW when all children are solved, none wins,
  and at least one draws.

  Two things make this pay rather than just being bookkeeping:

  1. Terminal children are marked AT EXPANSION (`_mark_terminal_children`), not
     when a simulation happens to wander into them. One clone+make_move per
     legal move, using the engine itself so no win rule is duplicated here. That
     turns every expansion into an exact 1-ply search, so a mate-in-1 at the
     root is proven BEFORE the first simulation and every visit thereafter goes
     to the win.
  2. Selection short-circuits: `_best_child` takes a proven win outright and
     refuses proven losses, and descent stops at any solved node and backs up
     its exact value. A proven subtree then costs one cheap descent per sim with
     NO network evaluation.

  Motivation is measured, not aesthetic. RESULT_DISTILL_PILOT.md found that on
  mate-in-1 positions an 800-sim teacher puts a visit on EVERY legal move and
  keeps only 0.693 mass on the win, where 50 sims keeps 0.825: PUCT's sqrt(N)
  exploration bonus grows with the budget, so more search DILUTES decided
  positions. Proving them stops the sampling instead of correcting it afterwards
  with a temperature hack.

  `search()` also reconciles the returned visit policy with the proof, because
  raw counts can disagree with it: a move that looked good early can collect
  visits and only later be refuted. See the comment at the pi-extraction site.
"""

import math
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from .agent_base import Agent, board_to_tensor_from_gamestate, wave_planes
from engine.rules import rule_utl_valid_moves
from engine.constants import X, O, DRAW


class MCTSNode:
    __slots__ = ("parent", "children", "prior", "N", "W",
                 "move", "to_play", "is_terminal", "terminal_value", "solved")

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
        # Exact status from THIS node's to_play perspective, or None if
        # unresolved. Deliberately NOT folded into W: a proof is not an average,
        # and mixing them would let one backup of +1 masquerade as certainty.
        self.solved = None

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
                 wave_size=1, solve=False):
        self.model    = model
        self.device   = device
        self.n_sims   = n_sims
        self.c_puct   = c_puct
        self.add_dirichlet = add_dirichlet_at_root
        self.dir_alpha = dir_alpha
        self.dir_eps   = dir_eps
        self.wave_size = max(1, int(wave_size))
        # OFF by default on purpose: production expert iteration must not change
        # behaviour just because this landed. Callers opt in explicitly.
        self.solve = bool(solve)
        self.reset_stats()

    def reset_stats(self):
        """Search-cost counters for the search-quality report. Cumulative across
        searches; the caller resets when it wants a per-move figure."""
        self.stat_searches   = 0
        self.stat_expansions = 0   # nodes given children (== net evals of leaves)
        self.stat_nn_evals   = 0   # positions pushed through the net
        self.stat_nn_batches = 0   # forward_both calls
        self.stat_probes     = 0   # clone+make_move terminal probes
        self.stat_solved_roots = 0
        # Own wall clock. A match's total elapsed time cannot be split between
        # two arms after the fact, so each search times itself.
        self.stat_seconds    = 0.0
        # Per-search record, overwritten each call. See _finish_record.
        self.last = {}

    # ------------------------------------------------------------------
    # Per-search proof bookkeeping. All of this is skipped when solve=False.
    # ------------------------------------------------------------------

    def _note_proof(self, root, sims_done):
        """First moment the ROOT becomes proven, and the state of the target then.

        `visits_off_proven` is the metric that matters for distillation: how many
        visits had already been spent on moves other than the proven win by the
        time the proof landed. Zero means the target was never distorted; a large
        number means the proof arrived too late to help the target even though it
        still helps play.
        """
        if self._proof_sim is not None or root.solved is None:
            return
        refuted = sum(c.N for c in root.children.values() if c.solved == 1)
        wins = [c for c in root.children.values() if c.solved == -1]
        off_proven = None
        if wins:
            best = max(wins, key=lambda c: (c.N, -c.move))
            off_proven = root.N - best.N
        self._proof_sim = sims_done
        self._proof_refuted_visits = refuted
        self._proof_off_visits = off_proven
        self._proof_nn_evals = self.stat_nn_evals

    def _finish_record(self, root, pi_raw_argmax, corrected_argmax):
        """One row per search, for the coverage / timing / reconciliation tables."""
        nn_total = self.stat_nn_evals - self._search_nn_start
        rec = {
            "root_solved": root.solved,
            "root_n": root.N,
            "n_sims": self.n_sims,
            "expansions": self.stat_expansions - self._search_exp_start,
            "nn_evals": nn_total,
            "probes": self.stat_probes - self._search_probe_start,
            "raw_argmax": int(pi_raw_argmax),
            "corrected_argmax": int(corrected_argmax),
            "reconciled": int(pi_raw_argmax) != int(corrected_argmax),
            "raw_argmax_solved": None,
            "corrected_argmax_solved": None,
            "proof_sim": self._proof_sim,
            "visits_on_refuted_at_proof": self._proof_refuted_visits,
            "visits_off_proven_at_proof": self._proof_off_visits,
            "nn_evals_at_proof": (None if self._proof_nn_evals is None else
                                  self._proof_nn_evals - self._search_nn_start),
            "visits_on_refuted_final": sum(
                c.N for c in root.children.values() if c.solved == 1),
        }
        raw_child = root.children.get(int(pi_raw_argmax))
        cor_child = root.children.get(int(corrected_argmax))
        if raw_child is not None:
            rec["raw_argmax_solved"] = raw_child.solved
        if cor_child is not None:
            rec["corrected_argmax_solved"] = cor_child.solved
        self.last = rec

    @torch.no_grad()
    def search(self, root_state):
        t_search = time.perf_counter()
        root = MCTSNode(to_play=root_state.player)
        self.stat_searches += 1
        if self.solve:
            self._search_nn_start = self.stat_nn_evals
            self._search_exp_start = self.stat_expansions
            self._search_probe_start = self.stat_probes
            self._proof_sim = None
            self._proof_refuted_visits = None
            self._proof_off_visits = None
            self._proof_nn_evals = None
        else:
            # None, not a stale dict: `last` is only written when solving, and a
            # caller reading the PREVIOUS position's record would be a silent
            # data error rather than a visible one.
            self.last = None
        self._expand(root, root_state, add_noise=self.add_dirichlet)
        if self.solve:
            # A mate-in-1 root is proven by the root expansion itself, before a
            # single simulation runs. Recording that as sim 0 is exact, and it
            # is the case the whole feature exists for.
            self._note_proof(root, 0)

        if self.wave_size == 1:
            # Original single-sim path -- byte-identical to prior behaviour
            # whenever solve=False.
            for sim_i in range(self.n_sims):
                node  = root
                state = _clone(root_state)

                while node.children and not node.is_terminal:
                    node = self._best_child(node)
                    state.make_move(node.move)
                    if state.winner is not None:
                        node.is_terminal    = True
                        node.terminal_value = self._terminal_value(state, node.to_play)
                        if self.solve:
                            self._mark_solved(node, node.terminal_value)
                        break
                    if node.solved is not None:
                        break   # proven subtree: no point searching it again

                if node.solved is not None:
                    value = float(node.solved)
                elif node.is_terminal:
                    value = node.terminal_value
                else:
                    value = self._expand(node, state, add_noise=False)
                    # Expansion may itself have proved the node (a terminal
                    # child was found). An exact value beats the net's guess.
                    if node.solved is not None:
                        value = float(node.solved)

                self._backup(node, value)
                if self.solve:
                    self._note_proof(root, sim_i + 1)
        else:
            # Clamp so every search gets at least _MIN_WAVES waves of depth.
            eff_wave = max(1, min(self.wave_size, self.n_sims // self._MIN_WAVES))
            sims_done = 0
            while sims_done < self.n_sims:
                wave = min(eff_wave, self.n_sims - sims_done)
                self._run_wave(root, root_state, wave, sims_done)
                sims_done += wave
                if self.solve:
                    self._note_proof(root, sims_done)

        pi = np.zeros(81, dtype=np.float32)
        for mv, child in root.children.items():
            pi[mv] = child.N
        raw_argmax = int(pi.argmax())

        if self.solve:
            # Visit counts and the proof can disagree, and when they do the
            # proof is right. A move the net loved can bank hundreds of visits
            # and only later be refuted; raw counts would then hand back a
            # target whose argmax is a move known to LOSE. Two corrections:
            #
            #   proven win  -> one-hot. Any proven win ends the game, so the
            #                  honest target mass is 1.0 on it. This is the
            #                  forced-win dilution from RESULT_DISTILL_PILOT.md
            #                  removed at the source rather than smoothed over.
            #   proven loss -> zero. Unless EVERY move loses, in which case the
            #                  counts stand: something must still be played, and
            #                  visits are as good an ordering as any.
            #
            # Ties among several proven wins go to the most-visited, lowest move
            # index -- fully deterministic, which the fixed panels require.
            wins = [c for c in root.children.values() if c.solved == -1]
            if wins:
                best = max(wins, key=lambda c: (c.N, -c.move))
                pi[:] = 0.0
                pi[best.move] = 1.0
                self.stat_solved_roots += 1
                self._finish_record(root, raw_argmax, best.move)
                self.stat_seconds += time.perf_counter() - t_search
                return pi, root
            losses = [c for c in root.children.values() if c.solved == 1]
            if losses and len(losses) < len(root.children):
                for c in losses:
                    pi[c.move] = 0.0

        s = pi.sum()
        if s > 0:
            pi /= s
        if self.solve:
            self._finish_record(root, raw_argmax, int(pi.argmax()))
        self.stat_seconds += time.perf_counter() - t_search
        return pi, root

    # ------------------------------------------------------------------
    # Batched wave (wave_size > 1 path)
    # ------------------------------------------------------------------

    def _run_wave(self, root, root_state, wave_size, sims_base=0):
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
                    if self.solve:
                        self._mark_solved(node, node.terminal_value)
                    break
                if node.solved is not None:
                    break   # proven subtree: costs one descent, no net eval

            pending.append((node, state, path))

        # NOTE: proofs are NOT recorded here, mid-wave. Virtual loss has inflated
        # every path child's N by +1 per selection and it is not undone until the
        # end of this function, so a visit snapshot taken now would be wrong (and
        # root.N - child.N could even go negative). search() notes proofs at wave
        # boundaries instead, where the counts are real. That costs granularity
        # in proof_sim -- one wave, i.e. n_sims//16 -- and the reported figure is
        # therefore an upper bound on when the proof actually landed.

        # Collect unique unexpanded non-terminal leaves for the batch forward pass.
        # Dedup by node identity: two sims landing on the same unexpanded leaf
        # (possible in a shallow tree early in search) expand it once.
        seen    = {}    # id(node) -> index in to_eval
        to_eval = []    # (pending_idx, node, state)
        for i, (node, state, _) in enumerate(pending):
            if not node.is_terminal and node.solved is None and not node.children:
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
            self.stat_nn_batches += 1
            self.stat_nn_evals += len(to_eval)
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

            # solved first: expansion above may have proved this very node, and
            # an exact result outranks both the terminal cache and the net.
            if node.solved is not None:
                value = float(node.solved)
            elif node.is_terminal:
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

    # ------------------------------------------------------------------
    # Solved-node propagation (all no-ops unless self.solve)
    # ------------------------------------------------------------------

    def _mark_terminal_children(self, node, state):
        """Exact 1-ply search: mark every child that ends the game, then try to
        solve `node` from what that revealed.

        Uses the engine (clone + make_move) rather than a hand-rolled win test.
        A bespoke detector would have to re-derive mini-win, macro-win and
        board-full-draw rules, and a subtle disagreement with the engine would
        silently poison every target built from this search. Measured cost is
        ~2 us per legal move on the C++ engine and ~7 us on the Python one --
        against ~600 ms for an 800-sim search, that is noise.
        """
        for mv, child in node.children.items():
            probe = _clone(state)
            probe.make_move(mv)
            self.stat_probes += 1
            if probe.winner is not None:
                child.is_terminal    = True
                child.terminal_value = self._terminal_value(probe, child.to_play)
                child.solved         = int(child.terminal_value)
        status = self._solve_from_children(node)
        if status is not None:
            self._mark_solved(node, status)

    @staticmethod
    def _solve_from_children(node):
        """Backward induction over `node`'s children, in NODE's perspective.

        Returns +1 / 0 / -1 (proven win / draw / loss for node.to_play) or None.

        A child's `solved` is in the CHILD's perspective, and the child's mover
        is our opponent, so a child solved -1 (lost for whoever moves there) is
        a WIN for us. That sign flip is the same one `_best_child` and `_backup`
        make; getting it backwards here would make the search chase its own
        losses, so it is stated explicitly rather than inlined.
        """
        kids = node.children
        if not kids:
            return None          # unexpanded: absence of children proves nothing
        all_solved = True
        has_draw = False
        for c in kids.values():
            s = c.solved
            if s is None:
                all_solved = False
            elif s == -1:
                return 1         # one refuted reply is enough to prove the win
            elif s == 0:
                has_draw = True
        if not all_solved:
            return None          # loss/draw claims need EVERY legal reply solved
        return 0 if has_draw else -1

    def _mark_solved(self, node, value):
        # Every sim that walks into an already-terminal node re-reaches this;
        # returning early keeps that from re-walking the ancestor chain each
        # time. A proof never changes once made, so re-deriving it is pure cost.
        if node.solved is not None:
            return
        node.solved = int(value)
        self._propagate_solved(node)

    def _propagate_solved(self, node):
        """Carry a fresh proof as far up the tree as it can be justified."""
        cur = node.parent
        while cur is not None and cur.solved is None:
            status = self._solve_from_children(cur)
            if status is None:
                return
            cur.solved = status
            cur = cur.parent

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
        self.stat_expansions += 1
        if self.solve:
            self._mark_terminal_children(node, state)

    def _expand(self, node, state, add_noise=False):
        """Run one network forward pass for a single leaf, expand it, return value."""
        x = board_to_tensor_from_gamestate(state).unsqueeze(0).to(self.device)
        logits, value = self.model.forward_both(x)
        self.stat_nn_batches += 1
        self.stat_nn_evals += 1
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
        kids = node.children.values()
        if self.solve:
            # A proven win ends the discussion -- there is nothing PUCT could
            # find that beats a forced win, and continuing to explore is exactly
            # the dilution this feature exists to stop. Deterministic tie-break.
            wins = [c for c in kids if c.solved == -1]
            if wins:
                return max(wins, key=lambda c: (c.N, -c.move))
            # Never spend another visit on a refuted move. If they are ALL
            # refuted this node is lost regardless, so fall through to PUCT and
            # keep the statistics well-formed rather than special-casing.
            live = [c for c in kids if c.solved != 1]
            if live:
                kids = live
        return max(kids,
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
                 wave_size=1, solve=False):
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
            solve=solve,
        )
        self.temperature = temperature

    def select_move(self, gamestate):
        pi, _ = self.mcts.search(gamestate)
        if self.temperature == 0.0:
            return int(np.argmax(pi))
        probs  = pi ** (1.0 / self.temperature)
        probs /= probs.sum()
        return int(np.random.choice(81, p=probs))
