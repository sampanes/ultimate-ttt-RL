import random as _random
from .agent_base import Agent
from engine.rules import (
    rule_utl_valid_moves,
    _MINI_INDICES,
)
from engine.tactics import move_wins_mini
from engine.constants import X, O

# Center cell (local position 4) of each mini-board, as GLOBAL board indices.
# The flat 81-board is row-major, so a mini-board's 9 cells are SCATTERED, not a
# contiguous run of 9 -- derive centers from _MINI_INDICES, never i*9+4.
_CENTER_CELLS = frozenset(_MINI_INDICES[m][4] for m in range(9))


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
            if move_wins_mini(board, move, player):
                return move

        for move in valid_moves:
            if move_wins_mini(board, move, opponent):
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
