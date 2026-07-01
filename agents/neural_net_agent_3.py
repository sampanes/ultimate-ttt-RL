import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import contextlib

from agents.agent_base import Agent, ModelConfigCNN, board_to_tensor_from_gamestate
from engine.constants import EMPTY, X, O, DRAW
from engine.rules import rule_utl_valid_moves
from engine.game import GameState

import glob, re, os

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

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.value_tanh = getattr(cfg, 'value_tanh', False)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        assert x.shape[-2:] == (9, 9), f"Expected 9x9 input, got {x.shape[-2:]}"
        assert x.shape[1] == 7, f"Expected 7 channels, got {x.shape[1]}"
        out = self.policy_head(self.conv_layers(x))
        return out.squeeze(0) if out.shape[0] == 1 else out

    def forward_both(self, x):
        """Returns (policy_logits, value) where policy_logits is shape (81,) or (B, 81)
        and value is shape () or (B,)."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        assert x.shape[-2:] == (9, 9), f"Expected 9x9 input, got {x.shape[-2:]}"
        assert x.shape[1] == 7, f"Expected 7 channels, got {x.shape[1]}"
        conv_out = self.conv_layers(x)
        policy = self.policy_head(conv_out)
        value = self.value_head(conv_out).squeeze(-1)  # (B,)
        if self.value_tanh:
            value = torch.tanh(value)
        squeeze = policy.shape[0] == 1
        if squeeze:
            return policy.squeeze(0), value.squeeze(0)
        return policy, value

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
            print(f"[>>]\t{self.name} is using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("[!]\tUsing CPU -- training will be slower.")
        # build model + optimizer
        self.model = ConvNet(cfg=cfg).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        self.model_load_magic(model_path, cfg)

        # history for training
        self.clear_history()

    def model_load_magic(self, model_path, cfg):
        # load if a checkpoint provided
        if model_path is None:
            self.model_dir = cfg.model_dir
            self.brand_new_weights(cfg)
            return
        else:
            if model_path == "":
                self.model_dir = cfg.get_model_dir()
                pattern = os.path.join(self.model_dir, "version_*.pt")
                candidates = glob.glob(pattern)
                if not candidates:
                    print(f"[!]\t{self.name}: model_path=\"\" but no checkpoints in "
                          f"{self.model_dir} -- starting from RANDOM weights.")
                    self.brand_new_weights(cfg)
                    return
                # pick the highest numbered version
                version, path = max(
                    ((int(re.search(r"version_(\d+)\.pt$", p).group(1)), p)
                     for p in candidates if re.search(r"version_(\d+)\.pt$", p)),
                    key=lambda t: t[0]
                )
                model_path = path
                if self.verbose:
                    print(f"[find] Auto-loaded latest checkpoint {version}: {model_path}")

            self.load(model_path)

    def set_eval(self, is_eval: bool = True):
        self.model.eval() if is_eval else self.model.train()

    def select_move(self, gamestate: GameState) -> int:
        valid = rule_utl_valid_moves(
            gamestate.board, gamestate.last_move, gamestate.mini_winners
        )

        x = board_to_tensor_from_gamestate(gamestate, v_computed=valid).to(self.device)

        # choose a context: no_grad in eval, no-op in train
        ctx = torch.no_grad if not self.model.training else contextlib.nullcontext
        with ctx():
            logits = self.model(x)

        # flatten if needed
        logits = logits.squeeze(0) if logits.dim() == 2 else logits
        assert logits.shape == (81,), f"Expected (81,), got {logits.shape}"

        # mask invalid moves
        masked = torch.full_like(logits, float('-inf'))
        valid_t = torch.tensor(valid, dtype=torch.long, device=logits.device)
        masked.scatter_(0, valid_t, logits[valid_t])

        best_move = int(torch.argmax(masked))
        return best_move

    def learn(self):
        """Masked actor-critic policy-gradient update over recorded (s, a, reward) tuples.

        The previous implementation regressed the *selected move's logit* toward the
        scalar reward via MSE. In near-symmetric self-play every action's expected
        reward is ~0, so that objective pulls all logits to the same value and the
        policy collapses to ~uniform ("flat logits"). Instead we treat the logits as
        a policy and do REINFORCE with the value head as a baseline:
            policy_loss = -(logpi(a|s) * (return - V(s))).mean()
            value_loss  =  mse(V(s), return)
        The legal-move mask is recovered from the stored state's valid-moves channel
        (channel 3 of board_to_tensor_from_gamestate), so illegal moves get zero
        probability mass and never receive gradient.
        """
        if not self.last_game_states:
            return
        assert len(self.last_game_states) == len(self.last_moves) == len(self.last_rewards)

        self.model.train()
        states = torch.stack([s.cpu() for s in self.last_game_states]).to(self.device)  # (B,7,9,9)
        logits, values = self.model.forward_both(states)                                # (B,81),(B,)
        if logits.dim() == 1:   # B==1 edge: forward_both squeezes the batch dim
            logits = logits.unsqueeze(0)
            values = values.unsqueeze(0)

        returns = torch.tensor(self.last_rewards, dtype=torch.float32, device=self.device)  # (B,)
        actions = torch.tensor(self.last_moves, dtype=torch.long, device=self.device)       # (B,)

        # Legal mask from the stored valid-moves channel (ch 3).
        legal = states[:, 3, :, :].reshape(states.shape[0], 81) > 0.5                       # (B,81) bool
        masked_logits = logits.masked_fill(~legal, float('-inf'))
        log_probs_all = F.log_softmax(masked_logits, dim=1)
        chosen_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)         # (B,)

        advantage = returns - values.detach()
        policy_loss = -(chosen_log_probs * advantage).mean()
        value_loss = F.mse_loss(values, returns)

        probs_all = log_probs_all.exp()
        # masked_fill guards 0 * -inf = nan from the masked (illegal) entries.
        entropy = -(probs_all * log_probs_all.masked_fill(probs_all == 0, 0.0)).sum(dim=1).mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

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
            print(f"[*]\t{self.name} is saving {p}")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        p = path.replace("\\","/")
        print(f"[*]\t{self.name} is loading {p}")
        state = torch.load(path, map_location=self.device, weights_only=True)
        # handle old save format with nested state_dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # strict=False tolerates added heads (e.g. value head), but a near-total
        # mismatch means we silently kept random weights -- surface that loudly.
        result = self.model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"[!]\t{self.name}: state_dict mismatch on load -- "
                  f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected key(s). "
                  f"Missing-key weights remain RANDOM.")
        self.model.eval()


    def brand_new_weights(self, cfg):
        # no checkpoint -> this is a fresh network:
        # let's save its initial weights for LTH / rewinding later
        init_path = os.path.join(cfg.model_dir, "initial.pt")
        # make sure the directory exists
        os.makedirs(os.path.dirname(init_path), exist_ok=True)
        torch.save({"initial_state_dict": self.model.state_dict()}, init_path)
        if self.verbose:
            print(f"[box]  Saved initial weights to {init_path}")