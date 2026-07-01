import random as _random
from .agent_base import Agent
from engine.rules import (
    rule_utl_valid_moves,
    rule_utl_get_mini_index,
    rule_utl_get_next_mini,
    _MINI_INDICES,
)
from engine.constants import X, O

_MINI_WIN_PATTERNS = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diags
)

# Center cell (local position 4) of each mini-board, as GLOBAL board indices.
# The flat 81-board is row-major, so a mini-board's 9 cells are SCATTERED, not a
# contiguous run of 9 -- derive centers from _MINI_INDICES, never i*9+4.
_CENTER_CELLS = frozenset(_MINI_INDICES[m][4] for m in range(9))


def _move_wins_mini(board, move, player):
    """Return True if placing `player` at global `move` completes a line in that mini-board."""
    mini = rule_utl_get_mini_index(move)   # which mini-board (0-8) the cell belongs to
    cell = rule_utl_get_next_mini(move)    # the cell's local position within that mini (0-8)
    cells = _MINI_INDICES[mini]            # the 9 global indices of that mini-board
    for pat in _MINI_WIN_PATTERNS:
        if cell not in pat:
            continue
        if all(c == cell or board[cells[c]] == player for c in pat):
            return True
    return False


class FirstAvailableAgent(Agent):
    def __init__(self):
        super().__init__("FirstAvailableAgent")

    def select_move(self, gamestate):
        valid_moves = rule_utl_valid_moves(gamestate.board, gamestate.last_move, gamestate.mini_winners)
        return valid_moves[0]


class LastAvailableAgent(Agent):
    def __init__(self):
        super().__init__("LastAvailableAgent")

    def select_move(self, gamestate):
        valid_moves = rule_utl_valid_moves(gamestate.board, gamestate.last_move, gamestate.mini_winners)
        return valid_moves[-1]


class WinBlockAgent(Agent):
    def __init__(self):
        super().__init__("WinBlockAgent")

    def select_move(self, gamestate):
        board = gamestate.board
        valid_moves = rule_utl_valid_moves(board, gamestate.last_move, gamestate.mini_winners)
        player = gamestate.player
        opponent = O if player == X else X

        for move in valid_moves:
            if _move_wins_mini(board, move, player):
                return move

        for move in valid_moves:
            if _move_wins_mini(board, move, opponent):
                return move

        return _random.choice(valid_moves)


class CenterPreferenceAgent(Agent):
    def __init__(self):
        super().__init__("CenterPreferenceAgent")

    def select_move(self, gamestate):
        valid_moves = rule_utl_valid_moves(gamestate.board, gamestate.last_move, gamestate.mini_winners)

        for move in valid_moves:
            if move in _CENTER_CELLS:
                return move

        return _random.choice(valid_moves)
