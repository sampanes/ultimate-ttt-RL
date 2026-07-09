"""Shallow tactical lookahead -- pure engine logic, no torch.

Two provably-correct tactical layers to wrap on top of a policy:

  * win-in-1   : a legal move that immediately wins the GAME (ultimate win).
                 Never pass one up.
  * block-in-1 : a legal move that hands the opponent an immediate game-winning
                 reply (depth 2 -- our move + their winning reply). These are the
                 moves to AVOID.

Winning a single *mini-board* is deliberately NOT treated as tactically forced:
in UTTT it can be bad (it may send the opponent to a free/strong board). Only
ultimate wins and losses are unambiguous, so only those are handled here. Every
other choice is left to the policy.

All functions take a GameState and use clone()/make_move()/rule_utl_valid_moves,
so they behave identically on the C++ and pure-Python engines and need no torch.
This keeps the logic unit-testable on a box with no GPU/torch.
"""
from engine.constants import X, O
from engine.rules import (
    _MINI_INDICES,
    rule_utl_get_mini_index,
    rule_utl_get_next_mini,
    rule_utl_valid_moves,
)


_MINI_WIN_PATTERNS = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


def _opponent(player: int) -> int:
    return O if player == X else X


def _valid_for(state, valid):
    if valid is None:
        return rule_utl_valid_moves(state.board, state.last_move, state.mini_winners)
    return valid


def move_wins_mini(board, move, player):
    """True if `player` at global `move` completes that mini-board.

    This is not game-theoretic ground truth by itself: winning a mini-board can
    be strategically bad in UTTT. It is exposed for adversarial data generation
    and deterministic anchors such as WinBlockAgent, not for the default
    proof-only tactical filter below.
    """
    mini = rule_utl_get_mini_index(move)
    cell = rule_utl_get_next_mini(move)
    cells = _MINI_INDICES[mini]
    for pat in _MINI_WIN_PATTERNS:
        if cell not in pat:
            continue
        if all(c == cell or board[cells[c]] == player for c in pat):
            return True
    return False


def mini_winning_moves(state, valid=None, player=None):
    """Legal moves that complete a mini-board for `player`.

    Defaults to state.player. This deliberately answers only the local tactical
    question; callers decide whether the local win/block is strategically useful.
    """
    player = state.player if player is None else player
    return [
        mv for mv in _valid_for(state, valid)
        if move_wins_mini(state.board, mv, player)
    ]


def mini_tactical_filter(state, valid=None):
    """Return local mini-board win/block candidates.

    Returns (wins, blocks, preferred), where preferred is `wins` if any exist,
    else `blocks`. This mirrors WinBlockAgent's local rule and is intended for
    adversarial coverage of that measured blind spot. Unlike tactical_filter(),
    it is heuristic, not proof-only.
    """
    valid = _valid_for(state, valid)
    wins = mini_winning_moves(state, valid, state.player)
    blocks = mini_winning_moves(state, valid, _opponent(state.player))
    return wins, blocks, wins if wins else blocks


def winning_moves(state, valid=None):
    """Legal moves that immediately win the game for state.player (ultimate win)."""
    mover = state.player
    wins = []
    for mv in _valid_for(state, valid):
        s = state.clone()
        _, w = s.make_move(mv)
        if w == mover:
            wins.append(mv)
    return wins


def losing_moves(state, valid=None):
    """Legal moves that allow the opponent an immediate game-winning reply.

    A move is 'losing' iff, after we play it, the game is still live AND the
    opponent has at least one reply that wins the game outright. A move that
    itself ends the game (our win, or a draw) is never 'losing' -- the opponent
    gets no reply. make_move returns the winner (X/O/DRAW) or None if the game
    continues, normalized identically across the C++ and Python engines.
    """
    opp = _opponent(state.player)
    losing = []
    for mv in _valid_for(state, valid):
        s = state.clone()
        _, w = s.make_move(mv)
        if w is not None:
            continue  # game ended on our move (win or draw) -- no opponent reply
        for omv in rule_utl_valid_moves(s.board, s.last_move, s.mini_winners):
            s2 = s.clone()
            _, w2 = s2.make_move(omv)
            if w2 == opp:
                losing.append(mv)
                break
    return losing


def tactical_filter(state, valid=None):
    """Shallow tactical override for state.player. Returns (winning, safe):

        winning : immediate game-winning moves -- if non-empty, play one of them.
        safe    : legal moves that don't allow an immediate opponent win
                  (falls back to all legal moves if *every* move loses).

    Pure -- no model, no torch. A policy/argmax is expected to choose WITHIN the
    returned pool (winning if non-empty, else safe), so tactics only ever
    constrain the choice to provably-correct moves; the net breaks ties.
    """
    valid = _valid_for(state, valid)
    wins = winning_moves(state, valid)
    if wins:
        return wins, wins
    losing = set(losing_moves(state, valid))
    safe = [mv for mv in valid if mv not in losing]
    if not safe:
        safe = list(valid)
    return [], safe
