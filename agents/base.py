from dataclasses import dataclass
import torch
import torch.nn.functional as F

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