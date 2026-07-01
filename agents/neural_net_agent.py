import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from agents.agent_base import Agent, board_to_tensor
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
            print(f"[>>]\t{name} is using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("[!]\tUsing CPU -- training will be slower.")

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

        x = board_to_tensor(gamestate.board).to(self.device)
        with torch.no_grad():
            logits = self.model(x)

        # Mask invalid moves to -inf, then pick the best legal move.
        masked_logits = torch.full_like(logits, float('-inf'))
        valid_t = torch.tensor(valid, dtype=torch.long, device=logits.device)
        masked_logits[valid_t] = logits[valid_t]
        best_move = int(torch.argmax(masked_logits))

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

        states = torch.stack(self.last_game_states).to(self.device)   # (B, 81)
        logits = self.model(states)
        returns = torch.tensor(self.last_rewards, dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.last_moves, dtype=torch.long, device=self.device)

        # REINFORCE: maximize logpi(a|s) weighted by the (shaped) return. Replaces the
        # old MSE-regress-logit-toward-reward objective that collapsed the policy to
        # ~uniform in symmetric self-play. Softmax is over all 81 (flat encoding has
        # no legality channel); illegal moves are masked at selection time and drift
        # down as they are never rewarded.
        log_probs_all = F.log_softmax(logits, dim=1)
        chosen_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        policy_loss = -(chosen_log_probs * returns).mean()
        entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=1).mean()
        loss = policy_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.clear_history()
        return loss.item()

    def clear_history(self):
        # Clear history
        self.last_game_states.clear()
        self.last_moves.clear()
        self.last_players.clear()
        self.last_rewards.clear()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        print(f"[*]\t{self.name} is loading {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()