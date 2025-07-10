from dataclasses import dataclass
import torch
import random
import numpy as np
import torch.nn.functional as F
from engine.constants import X, O, EMPTY
from engine.rules import rule_utl_valid_moves, rule_utl_get_indices_of_mini

@dataclass
class ModelConfig:
    """
    Holds hyperparameters and architecture specs for the neural net.
    Easily extend or override hidden sizes, learning rate, etc.
    """
    input_size: int = 81
    hidden_sizes: list[int] = None  # e.g. [128, 64]
    output_size: int = 81
    learning_rate: float = 1e-3
    model_dir: str = None  # we'll fill this later
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activation: callable = F.relu
    label: str = "uninitiated_label"

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128]
        if self.model_dir is None:
            layers_str = "-".join(map(str, self.hidden_sizes + [self.output_size]))
            self.model_dir = f"models/{self.label}/{layers_str}"

@dataclass
class ModelConfigCNN:
    """
    Holds hyperparameters and architecture specs for the neural net.
    Easily extend or override hidden sizes, learning rate, etc.
    """
    conv_channels: list[int]        # e.g. [32, 64, 64]
    fc_hidden_sizes: list[int]     # e.g. [256]
    input_channels: int = 7
    output_size: int = 81
    activation: callable = F.relu
    learning_rate: float = 1e-3
    model_dir: str = None  # we'll fill this later
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label: str = "uninitiated_label"

    def __post_init__(self):
        if self.fc_hidden_sizes is None:
            self.fc_hidden_sizes = [128]
        if self.model_dir is None:
            layers_str = "-".join(map(str, self.fc_hidden_sizes + [self.output_size]))
            self.model_dir = f"models/{self.label}/{layers_str}"

class Agent:
    def __init__(self, name="UnnamedAgent"):
        self.name = name

    def select_move(self, gamestate):
        """Given a GameState, return a valid move index (0-80)."""
        raise NotImplementedError("select_move must be implemented by subclasses")
    

def board_to_tensor(board):# TODO, player=None):
    """
    Convert a GameState.board (e.g. a 9x9 nested list or numpy array of ints)
    into a 1D float tensor on CPU. You can then call `.to(device)` or `.cpu()`
    on the result.

    Example mapping: empty=0.0, X=1.0, O=-1.0
    """
    # assume board is list[list[int]] or np.array
    arr = np.array(board, dtype=np.float32)
    # If your board uses other encodings, adjust mapping here
    # e.g. arr[arr == some_code] = ...
    return torch.from_numpy(arr).view(-1)  # flatten to vector

def board_to_tensor_from_gamestate(gamestate) -> torch.Tensor:
    # Create a 7-channel (feature planes) 9x9 tensor filled with zeros
    # Channels:
    # [0] X moves (all 0s except X spots, 1)
    # [1] O moves (all 0s except O spots, 1)
    # [2] Current player (X=1, O=-1)
    # [3] Valid moves
    # [4] Mini-board wins (X=1, O=-1)
    # [5] Last move
    # [6] Constant bias layer (all ones)
    tensor = torch.zeros((7, 9, 9), dtype=torch.float32)

    # Populate X and O positions
    for row in range(9):
        for col in range(9):
            idx = row * 9 + col
            val = gamestate.board[idx]
            if val == X:
                tensor[0, row, col] = 1.0
            elif val == O:
                tensor[1, row, col] = 1.0

    # Encode current player: all 1s for X, all -1s for O
    if gamestate.player == X:
        tensor[2, :, :] = 1.0
    elif gamestate.player == O:
        tensor[2, :, :] = -1.0

    # Valid move locations (1 where legal)
    valid = rule_utl_valid_moves(gamestate.board, gamestate.last_move, gamestate.mini_winners)
    for idx in valid:
        r, c = divmod(idx, 9)
        tensor[3, r, c] = 1.0

    # Mark entire mini-boards with the winner (X=1, O=-1)
    for i, mw in enumerate(gamestate.mini_winners):
        if mw != EMPTY:
            mini_idxs = rule_utl_get_indices_of_mini(i)
            for idx in mini_idxs:
                r, c = divmod(idx, 9)
                tensor[4, r, c] = 1.0 if mw == X else -1.0

    # Mark the last move (single 1.0)
    if gamestate.last_move is not None:
        r, c = divmod(gamestate.last_move, 9)
        tensor[5, r, c] = 1.0

    # Constant bias plane (can help learning in early layers)
    tensor[6, :, :] = 1.0  # Bias or constant plane

    return tensor


def get_random_x_o():
    return X if random.random() < 0.5 else O