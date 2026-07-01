import random
from .agent_base import Agent
from engine.rules import rule_utl_valid_moves

class RandomAgent(Agent):
    def __init__(self):
        super().__init__("RandomAgent")

    def select_move(self, gamestate):
        valid_moves = rule_utl_valid_moves(gamestate.board, gamestate.last_move, gamestate.mini_winners)
        return random.choice(valid_moves)