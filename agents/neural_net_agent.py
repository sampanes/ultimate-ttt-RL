# agents/neural_net_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from agents.base import Agent
from engine.constants import EMPTY, X, O, DRAW
from engine.rules import rule_utl_valid_moves
from engine.game import GameState

MODEL_DIR = "models/neural_net"
os.makedirs(MODEL_DIR, exist_ok=True)


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(81, 128)
        self.fc2 = nn.Linear(128, 81)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw scores for all 81 positions


class NeuralNetAgent(Agent):
    def __init__(self, name="NeuralNetAgent", model_path=None):
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if model_path:
            self.load(model_path)
        self.last_game_states = []
        self.last_moves = []
        self.last_players = []

    def board_to_tensor(self, board):
        mapping = {EMPTY: 0, X: 1, O: -1}
        return torch.tensor([mapping[val] for val in board], dtype=torch.float32).to(self.device)

    def select_move(self, gamestate: GameState):
        valid = rule_utl_valid_moves(gamestate.board, gamestate.last_move, gamestate.mini_winners)
        if not valid:
            return None  # Shouldn't happen in valid games

        x = self.board_to_tensor(gamestate.board)
        with torch.no_grad():
            logits = self.model(x)
            logits = logits.cpu().numpy()

        # Mask invalid moves
        mask = [-1e9] * 81
        for i in valid:
            mask[i] = logits[i]

        best_move = int(torch.tensor(mask).argmax())

        # Store game state for learning later
        self.last_game_states.append(x.detach())
        self.last_moves.append(best_move)
        self.last_players.append(gamestate.player)

        return best_move

    def learn(self, winner: int):
        self.model.train()
        DRAW_REWARD = 0.2  # or 0.0 if you want neutral

        for state, move, player in zip(self.last_game_states, self.last_moves, self.last_players):
            if winner == DRAW:
                target_value = DRAW_REWARD
            elif player == winner:
                target_value = 1
            else:
                target_value = -1

            output = self.model(state)
            target = output.clone().detach()
            target[move] = target_value

            loss = F.mse_loss(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.last_game_states.clear()
        self.last_moves.clear()
        self.last_players.clear()


    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()