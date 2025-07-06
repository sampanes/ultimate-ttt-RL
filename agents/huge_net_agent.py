# File: agents/huge_net_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base import Agent, ModelConfig

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
    def __init__(self, config: ModelConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.model = HugeNet(
            input_size=config.input_size,
            hidden_sizes=config.hidden_sizes,
            output_size=config.output_size
        ).to(self.device)

        self.model.eval()

        if config.model_path and os.path.exists(config.model_path):
            self.model.load_state_dict(torch.load(config.model_path, map_location=self.device))

    def select_move(self, board, legal_moves):
        board_tensor = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.model(board_tensor).squeeze(0)
        logits_filtered = torch.full_like(logits, -float('inf'))
        logits_filtered[legal_moves] = logits[legal_moves]
        probs = F.softmax(logits_filtered, dim=0)
        move = torch.multinomial(probs, 1).item()
        return move
