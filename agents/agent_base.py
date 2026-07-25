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

    # --- modern-tower options (all OFF by default) ----------------------- #
    # When all three are falsy, ConvNet builds the EXACT legacy module graph,
    # so every existing checkpoint still loads key-for-key. Turning any of
    # them on produces a different (incompatible) state dict on purpose.
    #
    # residual:     interpret conv_channels as [stem_width] + [block_width]*N,
    #               where each entry after the first is one residual block
    #               holding TWO 3x3 convs of that width. Requires every block
    #               width to equal the stem width.
    # norm:         None | "group" | "batch". Normalization after every conv.
    #               "group" has no running stats, so it is batch-size and
    #               train/eval invariant -- safer under the eval server, where
    #               wave sizes vary, and it exports to ONNX cleanly.
    # head_squeeze: 0 keeps the legacy flatten->Linear heads (which spend ~78%
    #               of a 7-layer net's parameters). >0 inserts an AlphaZero
    #               style 1x1 conv down to this many channels BEFORE the
    #               flatten (2 is the usual policy value).
    residual: bool = False
    norm: str = None
    head_squeeze: int = 0

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

_FILL_PLANES = None   # tri-state cache: None=unknown, then bool (per engine)


def _has_fill_planes(state) -> bool:
    """True when the C++ engine exposes fill_planes (S8 fast path)."""
    global _FILL_PLANES
    if _FILL_PLANES is None:
        _FILL_PLANES = hasattr(state, "fill_planes")
    return _FILL_PLANES


def wave_planes(states, device) -> torch.Tensor:
    """S8: build (K, 7, 9, 9) planes for a wave of leaf states in ONE buffer.

    The C++ fast path fills each leaf straight into the buffer (one pybind
    crossing per leaf, no per-leaf torch allocation, no torch.stack). Profiling
    showed board_to_tensor was ~46% of an actor's non-GPU CPU. Byte-identical to
    the per-leaf fallback, used when the pure-Python engine is active.
    """
    k = len(states)
    buf = np.empty((k, 7, 9, 9), dtype=np.float32)
    if k and _has_fill_planes(states[0]):
        for i, s in enumerate(states):
            s.fill_planes(buf[i])
    else:
        for i, s in enumerate(states):
            buf[i] = board_to_tensor_from_gamestate(s).numpy()
    return torch.from_numpy(buf).to(device)


def board_to_tensor_from_gamestate(gamestate, v_computed=None) -> torch.Tensor:
    # S8 fast path: the C++ engine writes all 7 planes in one crossing,
    # byte-identical to the numpy path below (gated by test_hot_path). v_computed
    # is unused here -- fill_planes recomputes valid moves in C++, cheaper than
    # the Python call v_computed was meant to save.
    if _has_fill_planes(gamestate):
        buf = np.empty((7, 9, 9), dtype=np.float32)
        gamestate.fill_planes(buf)
        return torch.from_numpy(buf)

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