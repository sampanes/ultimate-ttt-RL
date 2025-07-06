import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base import Agent, ModelConfig
from engine.constants import EMPTY, X, O, DRAW
from engine.rules import rule_utl_valid_moves
from engine.game import GameState

'''
from agents.neural_net_agent_2 import ModelConfig, NeuralNetAgent2

# override whatever you like here
cfg = ModelConfig(
    hidden_sizes=[256, 128, 64],
    learning_rate=5e-4,
    model_dir="models/neural_net_experiment1",
)

# you can still optionally point at a checkpoint
agent = NeuralNetAgent2(cfg, model_path=None)

'''

class ConfigurableNN(nn.Module):
    """
    A fully-connected net whose layers can be changed via ModelConfig.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layers = []
        in_dim = cfg.input_size
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            # Convert function to nn.Module
            if cfg.activation == F.relu:
                layers.append(nn.ReLU())
            elif cfg.activation == F.leaky_relu:
                layers.append(nn.LeakyReLU())
            elif cfg.activation == F.tanh:
                layers.append(nn.Tanh())
            else:
                raise ValueError("Unsupported activation function. Use F.relu, F.leaky_relu, etc.")
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.output_size))
        # Use nn.Sequential for clarity
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def board_to_tensor(board: list[int]) -> torch.Tensor:
    """
    Convert 9x9 board values into a 1D float32 tensor.
    EMPTY->0.0, X->1.0, O->-1.0
    """
    mapping = {EMPTY: 0.0, X: 1.0, O: -1.0}
    arr = [mapping[val] for val in board]
    return torch.tensor(arr, dtype=torch.float32)

class NeuralNetAgent2(Agent):
    def __init__(self, cfg: ModelConfig, model_path: str = None):
        super().__init__(name="NeuralNetAgent2")
        self.cfg = cfg
        self.device = cfg.device
        if torch.cuda.is_available():
            print(f"🚀\t{self.name} is using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("⚠️\tUsing CPU — training will be slower.")
        # build model + optimizer
        self.model = ConfigurableNN(cfg).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # load if a checkpoint provided
        if model_path:
            self.load(model_path)

        # history for training
        self.clear_history()

    def select_move(self, gamestate: GameState) -> int:
        valid = rule_utl_valid_moves(
            gamestate.board, gamestate.last_move, gamestate.mini_winners
        )
        # feature extraction
        x = board_to_tensor(gamestate.board).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        # mask invalid moves
        mask = np.full((self.cfg.output_size,), -np.inf, dtype=float)
        valid = list(valid)
        cpu_logits = logits.cpu().numpy()
        for idx in valid:
            mask[idx] = cpu_logits[idx]
        # pick best
        best_move = int(np.argmax(mask))

        # record for learning
        self.last_game_states.append(x.detach())
        self.last_moves.append(best_move)
        self.last_players.append(gamestate.player)

        return best_move

    def learn(self):
        """
        Perform a regression on the recorded (s, a) pairs toward their rewards.
        """
        if not self.last_game_states:
            return
        assert len(self.last_game_states) == len(self.last_moves) == len(self.last_rewards)

        self.model.train()
        states = torch.stack(self.last_game_states).to(self.device)
        outputs = self.model(states)
        targets = outputs.clone().detach()

        # supervise chosen actions toward their rewards
        for i, (move, reward) in enumerate(zip(self.last_moves, self.last_rewards)):
            targets[i, move] = reward

        loss = F.mse_loss(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.clear_history()

    def clear_history(self):
        """
        Reset the per-episode history before a new self-play batch.
        """
        self.last_game_states = []
        self.last_moves = []
        self.last_players = []
        self.last_rewards = []

    def save(self, path: str):
        '''TODO
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg.__dict__,  # optionally deep copy this
        }, path)
        '''
        print(f"🧠\t{self.name} is saving {path}")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        '''TODO
        ckpt = torch.load(path, map_location=device)
        cfg = ModelConfig(**ckpt['config'])  # rebuild the right architecture
        model = ConfigurableNN(cfg)
        model.load_state_dict(ckpt['model_state_dict'])
        '''
        print(f"🧠\t{self.name} is loading {path}")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()