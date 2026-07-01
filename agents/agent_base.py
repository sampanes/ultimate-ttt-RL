from dataclasses import dataclass
import torch
import random
import numpy as np
import torch.nn.functional as F
from engine.constants import X, O, EMPTY
from engine.rules import rule_utl_valid_moves, _MINI_INDICES

_MINI_IDX_ROWS = [np.array(_MINI_INDICES[i]) // 9 for i in range(9)]
_MINI_IDX_COLS = [np.array(_MINI_INDICES[i]) % 9 for i in range(9)]

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
        self.model_dir = self.get_model_dir()

    def get_model_dir(self):
        if self.model_dir is None:
            layers_str = "-".join(map(str, self.hidden_sizes + [self.output_size]))
            return f"models/{self.label}/{layers_str}"
        else:
            return self.model_dir

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
    value_tanh: bool = False

    def __post_init__(self):
        if self.fc_hidden_sizes is None:
            self.fc_hidden_sizes = [128]
        self.model_dir = self.get_model_dir()

    def get_model_dir(self):
        if self.model_dir is None:
            layers_str = "-".join(map(str, self.fc_hidden_sizes + [self.output_size]))
            return f"models/{self.label}/{layers_str}"
        else:
            return self.model_dir

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
    # Encode antisymmetrically (EMPTY=0, X=+1, O=-1) so the net can exploit the
    # X/O symmetry of the game. The board stores O as 2, so remap it to -1.
    arr[arr == O] = -1.0
    return torch.from_numpy(arr).view(-1)  # flatten to vector

def board_to_tensor_from_gamestate(gamestate, v_computed=None) -> torch.Tensor:
    tensor = torch.zeros((7, 9, 9), dtype=torch.float32)

    # Channels 0,1: X and O positions -- vectorized
    board = np.array(gamestate.board, dtype=np.int8).reshape(9, 9)
    tensor[0] = torch.from_numpy((board == X).astype(np.float32))
    tensor[1] = torch.from_numpy((board == O).astype(np.float32))

    # Channel 2: current player
    tensor[2, :, :] = 1.0 if gamestate.player == X else -1.0

    # Channel 3: valid moves -- vectorized
    if v_computed is not None:
        valid = v_computed
    else:
        valid = rule_utl_valid_moves(gamestate.board, gamestate.last_move, gamestate.mini_winners)
    if valid:
        valid_arr = np.array(valid)
        rows, cols = valid_arr // 9, valid_arr % 9
        tensor[3, rows, cols] = 1.0

    # Channel 4: mini-board winners -- vectorized
    for i, mw in enumerate(gamestate.mini_winners):
        if mw != EMPTY:
            tensor[4, _MINI_IDX_ROWS[i], _MINI_IDX_COLS[i]] = 1.0 if mw == X else -1.0

    # Channel 5: last move
    if gamestate.last_move is not None and gamestate.last_move != -1:
        r, c = divmod(gamestate.last_move, 9)
        tensor[5, r, c] = 1.0

    # Channel 6: bias
    tensor[6, :, :] = 1.0

    return tensor


def get_random_x_o():
    return X if random.random() < 0.5 else O