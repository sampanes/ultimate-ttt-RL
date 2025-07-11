import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent_base import Agent, ModelConfigCNN, board_to_tensor_from_gamestate
from engine.constants import EMPTY, X, O, DRAW
from engine.rules import rule_utl_valid_moves
from engine.game import GameState


class ConvNet(nn.Module):
    def __init__(self, cfg: ModelConfigCNN):
        super().__init__()

        layers = []
        in_channels = cfg.input_channels
        for out_channels in cfg.conv_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(self._to_module_activation(cfg.activation))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Compute conv output size (assuming 9x9 input and stride=1, padding=1 keeps size)
        conv_output_dim = in_channels * 9 * 9

        fc_layers = [nn.Flatten()]
        in_dim = conv_output_dim
        for h in cfg.fc_hidden_sizes:
            fc_layers.append(nn.Linear(in_dim, h))
            fc_layers.append(self._to_module_activation(cfg.activation))
            in_dim = h
        fc_layers.append(nn.Linear(in_dim, cfg.output_size))

        self.policy_head = nn.Sequential(*fc_layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        assert x.shape[-2:] == (9, 9), f"Expected 9x9 input, got {x.shape[-2:]}"
        assert x.shape[1] == 7, f"Expected 7 channels, got {x.shape[1]}"
        out = self.policy_head(self.conv_layers(x))
        return out.squeeze(0) if out.shape[0] == 1 else out

    def _to_module_activation(self, act):
        """Map functional activation to nn.Module."""
        if act == F.relu:
            return nn.ReLU()
        elif act == F.leaky_relu:
            return nn.LeakyReLU()
        elif act == F.tanh:
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {act}")
    

class NeuralNetAgent3(Agent):
    def __init__(self, cfg: ModelConfigCNN, model_path: str = None):
        super().__init__(name="NeuralNetAgent3")
        self.cfg = cfg
        self.device = cfg.device
        self.verbose = False
        if torch.cuda.is_available():
            print(f"🚀\t{self.name} is using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("⚠️\tUsing CPU — training will be slower.")
        # build model + optimizer
        self.model = ConvNet(cfg=cfg).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # load if a checkpoint provided
        if model_path:
            self.load(model_path)

        # history for training
        self.clear_history()

    def set_eval(self, is_eval: bool = True):
        self.model.eval() if is_eval else self.model.train()

    def select_move(self, gamestate: GameState) -> int:
        valid = rule_utl_valid_moves(
            gamestate.board, gamestate.last_move, gamestate.mini_winners
        )
        # feature extraction
        x = board_to_tensor_from_gamestate(gamestate).to(self.device)
        # with torch.no_grad():
        logits = self.model(x)
        logits = logits.squeeze(0) if logits.dim() == 2 else logits  # ✅ only flattens if needed
        assert logits.shape == (81,), f"Expected (81,), got {logits.shape}"

        # mask invalid moves
        valid = list(valid)
        if self.verbose:
            print("logits shape:", logits.shape)         # should be (81,)
            print("valid moves:", valid)                # list of ints
            valid_logits = [(i, logits[i].item()) for i in valid]
            valid_logits.sort(key=lambda x: x[1], reverse=True)
            print("Top valid logits:", valid_logits[:5])


        masked_logits = torch.full_like(logits, float('-inf'))
        masked_logits[valid] = logits[valid]
        best_move = int(torch.argmax(masked_logits))

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
        states = torch.stack([s.cpu() for s in self.last_game_states]).to(self.device)
        logits = self.model(states)  # shape: [B, 81]

        # Gather target values and the indices of moves
        target_values = torch.tensor(self.last_rewards, dtype=torch.float32, device=self.device)
        action_indices = torch.tensor(self.last_moves, dtype=torch.long, device=self.device)

        # Pull out the predicted logit for each selected action
        predicted_values = logits[range(len(logits)), action_indices]

        # Loss = MSE between predicted and actual reward
        loss = F.mse_loss(predicted_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.clear_history()
        return loss.item()

    def clear_history(self):
        """
        Reset the per-episode history before a new self-play batch.
        """
        self.last_game_states = []
        self.last_moves = []
        self.last_players = []
        self.last_rewards = []

    def save(self, path: str, verbose=True):
        if verbose:
            p = path.replace("\\","/")
            print(f"🧠\t{self.name} is saving {p}")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        p = path.replace("\\","/")
        print(f"🧠\t{self.name} is loading {p}")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()