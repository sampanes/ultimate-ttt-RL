import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from agents.base import Agent, board_to_tensor
from engine.constants import EMPTY, X, O, DRAW
from engine.rules import rule_utl_valid_moves
from engine.game import GameState
import numpy as np

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
        if torch.cuda.is_available():
            print(f"🚀\t{name} is using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("⚠️\tUsing CPU — training will be slower.")

        self.model = SimpleNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if model_path:
            self.load(model_path)
        self.last_game_states = []
        self.last_moves = []
        self.last_players = []
        self.last_rewards = []


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

        best_move = int(np.argmax(mask))

        # Store game state for learning later
        self.last_game_states.append(x.detach())
        self.last_moves.append(best_move)
        self.last_players.append(gamestate.player)

        return best_move

    def learn(self):
        if not self.last_game_states:
            return

        assert len(self.last_game_states) == len(self.last_moves) == len(self.last_rewards), \
            f"Inconsistent lengths: {len(self.last_game_states)}, {len(self.last_moves)}, {len(self.last_rewards)}"
        
        self.model.train()

        states = torch.stack(self.last_game_states).to(self.device)
        outputs = self.model(states)
        targets = outputs.clone().detach()

        for i, (move, reward) in enumerate(zip(self.last_moves, self.last_rewards)):
            if not (0 <= move < 81):
                print(f"⚠️ BAD MOVE: {move} at index {i}")
                continue
            targets[i, move] = reward

        loss = F.mse_loss(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.clear_history()

    def clear_history(self):
        # Clear history
        self.last_game_states.clear()
        self.last_moves.clear()
        self.last_players.clear()
        self.last_rewards.clear()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        print(f"🧠\t{self.name} is loading {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()