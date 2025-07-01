from .base import Agent
from engine.rules import rule_utl_valid_moves

class FirstAvailableAgent(Agent):
    def __init__(self):
        super().__init__("FirstAvailableAgent")

    def select_move(self, gamestate):
        valid_moves = rule_utl_valid_moves(gamestate.board, gamestate.last_move, gamestate.mini_winners)
        return valid_moves[0]