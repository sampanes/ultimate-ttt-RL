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

import torch

from engine.game import _PyGameState
from engine.constants import X, O, DRAW, EMPTY
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

    def spy(root, root_state, wave):
        calls.append(wave)
        return orig(root, root_state, wave)

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
