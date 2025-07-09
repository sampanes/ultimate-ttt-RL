from dataclasses import dataclass
import torch
import random
import numpy as np
import torch.nn.functional as F
from engine.constants import X, O, DRAW

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

class Agent:
    def __init__(self, name="UnnamedAgent"):
        self.name = name

    def select_move(self, gamestate):
        """Given a GameState, return a valid move index (0-80)."""
        raise NotImplementedError("select_move must be implemented by subclasses")
    

def board_to_tensor(board):
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


def get_random_x_o():
    return X if random.random() < 0.5 else O