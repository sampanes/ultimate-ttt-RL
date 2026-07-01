# File: agents/huge_net_agent.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.agent_base import Agent, ModelConfig, board_to_tensor
from engine.rules import rule_utl_valid_moves


class HugeNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HugeNetAgent(Agent):
    """Frozen, eval-only large MLP player.

    NOTE: this agent has no learn()/clear_history() machinery, so it cannot be
    trained through scripts.trainer_base. For a *trainable* large MLP, use
    NeuralNetAgent2 (ConfigurableNN), which accepts arbitrary hidden_sizes.
    """

    def __init__(self, config: ModelConfig, model_path: str = None):
        super().__init__("HugeNetAgent")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.model = HugeNet(
            input_size=config.input_size,
            hidden_sizes=config.hidden_sizes,
            output_size=config.output_size,
        ).to(self.device)
        self.model.eval()

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )

    def select_move(self, gamestate):
        valid = rule_utl_valid_moves(gamestate.board, gamestate.last_move, gamestate.mini_winners)
        if not valid:
            return None
        x = board_to_tensor(gamestate.board).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        masked = torch.full_like(logits, float('-inf'))
        valid_t = torch.tensor(valid, dtype=torch.long, device=logits.device)
        masked[valid_t] = logits[valid_t]
        return int(torch.argmax(masked))
