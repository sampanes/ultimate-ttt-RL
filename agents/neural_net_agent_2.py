import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent_base import Agent, ModelConfig, board_to_tensor
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
        valid = list(valid)
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
        # bring every recorded state to CPU first
        cpu_states = [s.cpu() for s in self.last_game_states]
        # now stack on CPU (all same device!), then move the entire batch to GPU
        states = torch.stack(cpu_states, dim=0).to(self.device)

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
        '''TODO
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg.__dict__,  # optionally deep copy this
        }, path)
        '''
        if verbose:
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