"""Deterministic mechanics tests for agents/mcts.py.

These lock in the parts of MCTS that are easy to get subtly wrong and that the
MCTS_PLAN flagged for verification -- above all the **value-sign / backup
convention** and **terminal-value perspective**. They use a trivial stub model
(uniform policy, value 0) so the search is driven PURELY by terminal detection
and visit counts: that isolates the tree logic from any trained network, and
makes every assertion deterministic (no Dirichlet, temperature 0, PUCT ties
broken by insertion order -> no RNG anywhere in the search path).

Needs torch at runtime (mcts.py imports it); runs on the home box. Run with:
    python -m agents.test_mcts          # standalone PASS/FAIL runner
    pytest agents/test_mcts.py          # or under pytest
"""

import math
import random

import torch

from engine.game import GameState, _PyGameState
from engine.constants import X, O, DRAW, EMPTY
from engine.rules import rule_utl_valid_moves
from agents.mcts import MCTS, MCTSNode, MCTSAgent


# --------------------------------------------------------------------------- #
# Stub model: uniform policy logits, value 0, for any input. Lets the terminal
# logic and backup signs be the only thing under test.
# --------------------------------------------------------------------------- #
class UniformZeroStub:
    def forward_both(self, x):
        # Mirror ConvNet.forward_both: batched (K,7,9,9) input -> (K,81)/(K,)
        # (used by the wave path), single input squeezed to (81,)/scalar.
        if x.dim() == 4 and x.shape[0] > 1:
            return torch.zeros(x.shape[0], 81), torch.zeros(x.shape[0])
        return torch.zeros(81), torch.tensor(0.0)


def _mcts(n_sims=100, c_puct=1.5):
    return MCTS(model=UniformZeroStub(), device="cpu", n_sims=n_sims,
                c_puct=c_puct, add_dirichlet_at_root=False)


# --------------------------------------------------------------------------- #
# 1. Node math
# --------------------------------------------------------------------------- #
def test_node_q_zero_when_unvisited():
    n = MCTSNode()
    assert n.Q() == 0.0


def test_node_u_formula():
    n = MCTSNode(prior=0.25)
    n.N = 3
    parent_N = 16
    c_puct = 1.5
    expected = c_puct * 0.25 * math.sqrt(parent_N) / (1 + 3)
    assert abs(n.U(c_puct, parent_N) - expected) < 1e-9


# --------------------------------------------------------------------------- #
# 2. Terminal value is from the to_play player's perspective
# --------------------------------------------------------------------------- #
def test_terminal_value_perspective():
    won_by_x = _PyGameState(winner=X)
    assert MCTS._terminal_value(won_by_x, X) == 1.0   # X to move, X won -> +1
    assert MCTS._terminal_value(won_by_x, O) == -1.0  # O to move, X won -> -1

    drawn = _PyGameState(winner=DRAW)
    assert MCTS._terminal_value(drawn, X) == 0.0
    assert MCTS._terminal_value(drawn, O) == 0.0

    ongoing = _PyGameState(winner=None)
    assert MCTS._terminal_value(ongoing, X) == 0.0


# --------------------------------------------------------------------------- #
# 3. Backup flips sign per ply -> each node's Q is in ITS OWN to_play perspective
# --------------------------------------------------------------------------- #
def test_backup_alternates_sign():
    # root(X) -> child(O) -> leaf(X). A leaf value of +1 means "good for X".
    root = MCTSNode(to_play=X)
    child = MCTSNode(parent=root, to_play=O)
    leaf = MCTSNode(parent=child, to_play=X)

    # _backup uses no instance state, but call it bound for clarity.
    _mcts()._backup(leaf, +1.0)

    assert leaf.N == child.N == root.N == 1
    assert leaf.Q() == +1.0   # X's perspective: good for X
    assert child.Q() == -1.0  # O's perspective: bad for O
    assert root.Q() == +1.0   # X's perspective: good for X


# --------------------------------------------------------------------------- #
# 4. THE sign test: an immediate ultimate win must be FOUND, not avoided.
#    If the backup/selection sign were flipped, MCTS would steer AWAY from the
#    winning move and this assertion would fail.
# --------------------------------------------------------------------------- #
def _state_one_move_from_x_win():
    """X already owns minis 0 and 1; X to move can complete mini 2's top row
    (cells 6,7 are X, cell 8 wins it) -> minis 0,1,2 = top row = ultimate X win.
    last_move=None makes it a free move so cell 8 is legal (and the search is
    deterministic over the legal set)."""
    board = [EMPTY] * 81
    board[6] = X
    board[7] = X
    mini_winners = [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    return _PyGameState(board=board, player=X, last_move=None,
                        mini_winners=mini_winners, winner=None)


def test_finds_immediate_win():
    state = _state_one_move_from_x_win()
    # sanity: cell 8 really is the winning move on a real engine clone
    probe = state.clone()
    _, w = probe.make_move(8)
    assert w == X, "test fixture is wrong: move 8 should win the game for X"

    mcts = _mcts(n_sims=100)
    pi, root = mcts.search(state)

    best = int(pi.argmax())
    assert best == 8, f"MCTS picked {best}, not the winning move 8 (sign flip?)"
    # visits should concentrate hard on the proven win, not merely edge it out
    assert pi[8] > 0.5, f"winning move got only {pi[8]:.2f} of visits"


# --------------------------------------------------------------------------- #
# 5. Clone independence -- MCTS rollouts mutate clones; the root must be untouched
# --------------------------------------------------------------------------- #
def test_clone_does_not_mutate_root():
    state = _state_one_move_from_x_win()
    board_before = list(state.board)
    player_before = state.player
    _mcts(n_sims=50).search(state)
    assert state.board == board_before
    assert state.player == player_before


# --------------------------------------------------------------------------- #
# 6. MCTSAgent end-to-end: temperature 0 returns the most-visited (winning) move
# --------------------------------------------------------------------------- #
def test_agent_select_move_argmax():
    class _Wrap:
        def __init__(self):
            self.model = UniformZeroStub()
            self.device = "cpu"
            self.name = "stub"

    agent = MCTSAgent(_Wrap(), n_sims=100, temperature=0.0)
    assert agent.select_move(_state_one_move_from_x_win()) == 8


# --------------------------------------------------------------------------- #
# 7. Wave path (wave_size > 1). Locks two 2026-07-04 bugs:
#    (a) virtual-loss SIGN: this tree stores W from the child's to-play
#        perspective and selection scores -Q, so VL must RAISE W. The inverted
#        sign turned virtual loss into virtual WIN and collapsed whole waves
#        onto one line (search scored 0.000 vs the raw net it wrapped).
#    (b) waves-per-search floor: leaf expansion is deferred to the end of each
#        wave, so wave_size ~ n_sims degenerates to a one-ply breadth probe.
#        search() must clamp wave_size to n_sims // _MIN_WAVES.
# --------------------------------------------------------------------------- #
def test_wave_finds_immediate_win():
    state = _state_one_move_from_x_win()
    for wave in (4, 8, 64):
        mcts = MCTS(model=UniformZeroStub(), device="cpu", n_sims=100,
                    c_puct=1.5, add_dirichlet_at_root=False, wave_size=wave)
        pi, _ = mcts.search(state)
        best = int(pi.argmax())
        assert best == 8, f"wave={wave}: picked {best}, not the winning move 8"
        assert pi[8] > 0.5, f"wave={wave}: winning move got only {pi[8]:.2f}"


def test_wave_clamped_to_min_waves():
    mcts = MCTS(model=UniformZeroStub(), device="cpu", n_sims=100, wave_size=64)
    calls = []
    orig = mcts._run_wave

    def spy(root, root_state, wave, sims_base=0):
        calls.append(wave)
        return orig(root, root_state, wave, sims_base)

    mcts._run_wave = spy
    mcts.search(_state_one_move_from_x_win())
    assert max(calls) <= 100 // MCTS._MIN_WAVES, \
        f"wave not clamped: ran waves of {max(calls)} for n_sims=100"
    assert sum(calls) == 100


def test_virtual_loss_discourages_revisit():
    # Two children, equal priors. Apply VL to one the way _run_wave does and
    # check _best_child now prefers the OTHER -- the inverted sign made the
    # VL'd child MORE attractive instead.
    mcts = _mcts()
    root = MCTSNode(to_play=X)
    root.N = 2
    a = MCTSNode(parent=root, prior=0.5, move=0, to_play=O)
    b = MCTSNode(parent=root, prior=0.5, move=1, to_play=O)
    root.children = {0: a, 1: b}

    a.N += 1
    a.W += MCTS._VL          # virtual loss as _run_wave applies it
    assert mcts._best_child(root) is b, \
        "virtual loss made the visited child MORE attractive (sign inverted?)"


# --------------------------------------------------------------------------- #
# 8. Root Q sign -- S2 (STRENGTH_NEXT) consumes root.Q() as a value target.
#    On a root where the mover has a provable immediate win, root.Q() must be
#    strongly POSITIVE (W is stored per-node in its own to_play perspective,
#    so the root's Q is the ROOT player's expected outcome), while the winning
#    CHILD's Q is exactly -1.0 (stored from the child's to-play = the LOSER's
#    perspective). Confusing those two perspectives is the exact sign mistake
#    the 2026-07-04 virtual-loss bug came from (see _run_wave). This test must
#    pass before trusting blended value targets built from root.Q().
# --------------------------------------------------------------------------- #
def test_root_q_sign_on_won_root():
    state = _state_one_move_from_x_win()
    _, root = _mcts(n_sims=100).search(state)

    assert root.to_play == X, "root perspective should be the player to move"
    assert root.N == 100, f"root got {root.N} backups, expected one per sim"
    assert root.Q() > 0.5, \
        f"root Q {root.Q():+.2f} must be strongly positive on a won root"

    win_child = root.children[8]
    assert win_child.Q() == -1.0, (
        f"winning child's Q is {win_child.Q():+.2f}; every backup through a "
        f"terminal win is exactly -1.0 from the child's (loser's) perspective")


# --------------------------------------------------------------------------- #
# 9. Solved-node propagation (solve=True). These lock the failure mode measured
#    in RESULT_DISTILL_PILOT.md: at 800 sims PUCT put a visit on EVERY legal
#    move of a mate-in-1 and kept only 0.693 mass on the win, because the
#    exploration bonus grows with sqrt(N_total). Proving the position is
#    supposed to make that dilution structurally impossible.
# --------------------------------------------------------------------------- #
def _solver(n_sims=100, wave_size=1):
    return MCTS(model=UniformZeroStub(), device="cpu", n_sims=n_sims,
                c_puct=1.5, add_dirichlet_at_root=False,
                wave_size=wave_size, solve=True)


def test_solve_off_by_default_and_inert():
    """Production must not change behaviour just because this landed."""
    m = _mcts(n_sims=50)
    assert m.solve is False
    pi, root = m.search(_state_one_move_from_x_win())
    assert root.solved is None, "solve=False must never write a proof"
    assert m.stat_probes == 0, "solve=False must not pay for terminal probes"
    assert all(c.solved is None for c in root.children.values())
    # pi is still the plain normalised visit distribution.
    total = sum(c.N for c in root.children.values())
    assert abs(pi[8] - root.children[8].N / total) < 1e-6


def test_solved_mate_in_one_is_one_hot():
    """The headline requirement: forced-win policy mass on the proven move."""
    for wave in (1, 4, 8):
        mcts = _solver(n_sims=100, wave_size=wave)
        pi, root = mcts.search(_state_one_move_from_x_win())
        assert root.solved == 1, f"wave={wave}: mate-in-1 root not proven won"
        assert root.children[8].solved == -1, "winning child not proven lost"
        assert int(pi.argmax()) == 8
        assert abs(pi[8] - 1.0) < 1e-6, \
            f"wave={wave}: proven win got {pi[8]:.4f} of the target, not 1.0"
        assert abs(pi.sum() - 1.0) < 1e-6


def test_solved_mate_in_one_does_not_dilute_with_more_sims():
    """The pilot's actual defect, as a test: mass on the win must not DROP as
    the simulation budget grows. Without solving it fell 0.825 -> 0.693."""
    masses = []
    for sims in (50, 200, 800):
        mcts = _solver(n_sims=sims, wave_size=max(1, sims // MCTS._MIN_WAVES))
        pi, _ = mcts.search(_state_one_move_from_x_win())
        masses.append(float(pi[8]))
    assert all(abs(m - 1.0) < 1e-6 for m in masses), \
        f"forced-win mass diluted with budget: {masses}"


def test_solving_saves_network_evaluations():
    """A proven subtree must cost descents, not forward passes."""
    plain = _mcts(n_sims=200)
    plain.search(_state_one_move_from_x_win())
    solved = _solver(n_sims=200)
    solved.search(_state_one_move_from_x_win())
    assert solved.stat_nn_evals < plain.stat_nn_evals, (
        f"solving did not cut net evals: {solved.stat_nn_evals} vs "
        f"{plain.stat_nn_evals}")


def test_solved_state_is_separate_from_visit_statistics():
    """Requirement: proofs must not contaminate N/W/Q."""
    mcts = _solver(n_sims=100)
    _, root = mcts.search(_state_one_move_from_x_win())
    win = root.children[8]
    # Every sim descends through the proven win, and _backup credits one visit
    # per node on the path -- exactly as it does with solving off.
    assert win.N == 100 and root.N == 100, \
        f"visit accounting disturbed: root.N={root.N} win.N={win.N}"
    # W is still the plain sum of backed-up values, here 100 terminal losses
    # from the child's own perspective.
    assert abs(win.W - (-100.0)) < 1e-6
    assert win.Q() == -1.0


# ---- backward induction, unit-tested directly on synthetic trees ---------- #
def _node_with_children(child_status):
    parent = MCTSNode(to_play=X)
    for i, s in enumerate(child_status):
        c = MCTSNode(parent=parent, move=i, to_play=O)
        c.solved = s
        parent.children[i] = c
    return parent


def test_induction_win_needs_only_one_refuted_reply():
    assert MCTS._solve_from_children(_node_with_children([None, -1, 1])) == 1


def test_induction_loss_needs_every_reply_solved():
    # All replies win for the opponent -> we are lost.
    assert MCTS._solve_from_children(_node_with_children([1, 1, 1])) == -1
    # One reply still unresolved -> nothing is proven yet.
    assert MCTS._solve_from_children(_node_with_children([1, 1, None])) is None


def test_induction_draw_requires_all_solved_and_no_win():
    assert MCTS._solve_from_children(_node_with_children([1, 0, 1])) == 0
    assert MCTS._solve_from_children(_node_with_children([0, None])) is None


def test_induction_unexpanded_node_proves_nothing():
    assert MCTS._solve_from_children(MCTSNode(to_play=X)) is None


def test_proof_propagates_through_ancestors():
    root = MCTSNode(to_play=X)
    mid = MCTSNode(parent=root, move=0, to_play=O)
    root.children[0] = mid
    leaf = MCTSNode(parent=mid, move=1, to_play=X)
    mid.children[1] = leaf

    mcts = _solver()
    mcts._mark_solved(leaf, -1)      # leaf's mover (X) is lost
    assert mid.solved == 1, "mid should be a proven win for O"
    assert root.solved == -1, \
        "root's only move loses, so root must be a proven loss"


def test_refuted_move_gets_zero_target_mass():
    """A move that banked visits before being refuted must not survive in pi."""
    mcts = _solver()
    root = MCTSNode(to_play=X)
    for mv, (n, solved) in enumerate([(400, 1), (10, None)]):
        c = MCTSNode(parent=root, move=mv, to_play=O)
        c.N, c.solved = n, solved
        root.children[mv] = c
    # Drive only the pi-extraction half of search() on this hand-built root.
    import numpy as np
    pi = np.zeros(81, dtype=np.float32)
    for mv, c in root.children.items():
        pi[mv] = c.N
    losses = [c for c in root.children.values() if c.solved == 1]
    if losses and len(losses) < len(root.children):
        for c in losses:
            pi[c.move] = 0.0
    pi /= pi.sum()
    assert pi[0] == 0.0 and pi[1] == 1.0


# ---- soundness against exhaustive minimax -------------------------------- #
def _exact_value(st, to_play, budget):
    """True game value for `to_play`: +1 win, 0 draw, -1 loss.

    budget is a mutable [n] node counter; returns None if it runs out, so a
    position that turns out to be too big is skipped rather than hanging.
    """
    if st.winner is not None:
        if st.winner == DRAW:
            return 0
        return 1 if st.winner == to_play else -1
    budget[0] -= 1
    if budget[0] <= 0:
        return None
    moves = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)
    if not moves:
        return 0
    opp = O if to_play == X else X
    best = -1
    for m in moves:
        child = st.clone()
        child.make_move(m)
        v = _exact_value(child, opp, budget)
        if v is None:
            return None
        best = max(best, -v)
        if best == 1:
            break
    return best


def _random_late_position(rng, plies):
    st = GameState()
    for _ in range(plies):
        moves = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)
        if not moves or st.winner is not None:
            return None
        st.make_move(rng.choice(moves))
    if st.winner is not None:
        return None
    if not rule_utl_valid_moves(st.board, st.last_move, st.mini_winners):
        return None
    return st


def test_solved_claims_agree_with_exhaustive_minimax():
    """SOUNDNESS, the property that actually matters.

    Completeness is not required -- a 200-sim search will leave plenty of
    positions unresolved. But every proof it DOES emit gets compared against a
    full minimax of the same position, because a wrong proof is worse than no
    proof: it would be baked into a one-hot training target with total
    confidence.
    """
    rng = random.Random(20260727)
    checked = proofs = 0
    for _ in range(400):
        st = _random_late_position(rng, rng.randint(58, 70))
        if st is None:
            continue
        truth = _exact_value(st, st.player, [30000])
        if truth is None:
            continue          # too big to verify exhaustively; skip, don't guess
        checked += 1
        _, root = _solver(n_sims=200).search(st)
        if root.solved is not None:
            proofs += 1
            assert root.solved == truth, (
                f"MCTS proved {root.solved:+d} but minimax says {truth:+d} "
                f"for the player to move")
    assert checked >= 40, f"only {checked} positions were verifiable; test is weak"
    assert proofs >= 5, (
        f"only {proofs}/{checked} positions got proved -- solving is firing far "
        f"too rarely to be doing anything")


def test_solver_still_plays_legal_moves():
    rng = random.Random(4242)
    mcts = _solver(n_sims=64, wave_size=4)
    st = GameState()
    for _ in range(30):
        legal = rule_utl_valid_moves(st.board, st.last_move, st.mini_winners)
        if not legal or st.winner is not None:
            break
        pi, _ = mcts.search(st.clone())
        mv = int(pi.argmax())
        assert mv in legal, f"solver returned illegal move {mv}"
        st.make_move(mv)


# --------------------------------------------------------------------------- #
# Standalone runner (mirrors engine/test_tactics.py style)
# --------------------------------------------------------------------------- #
def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001 -- test harness wants the message
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
