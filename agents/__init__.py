from .agent_base              import Agent
from .random_agent      import RandomAgent
from .deterministics import FirstAvailableAgent, LastAvailableAgent, WinBlockAgent, CenterPreferenceAgent
from .neural_net_agent import NeuralNetAgent
from .neural_net_agent_2 import ModelConfig, NeuralNetAgent2
from .neural_net_agent_3 import ModelConfigCNN, NeuralNetAgent3 
from .neural_net_agent_pg import NeuralNetAgentPG
from .huge_net_agent import HugeNetAgent
from .mcts import MCTSAgent
from .gregory import GregoryAgent
import os
import glob, re

BIG8_CFG = ModelConfig(
            hidden_sizes=[256, 512, 1024, 2048, 2048, 1024, 512, 256],
            learning_rate=1e-5,
            label="big_8_layer"
        )
BIG8_DIR = "models/big_8_layer/256-512-1024-2048-2048-1024-512-256-81"
# instead of calling each constructor now, store a factory
AGENT_FACTORIES = {
    "random": RandomAgent,   # classes are callable
    "first": FirstAvailableAgent,
    "last": LastAvailableAgent,
    "winblock": WinBlockAgent,
    "center": CenterPreferenceAgent,
    # Frozen eval-only large MLP (no training API; use NeuralNetAgent2 to train one).
    "huge_net": lambda: HugeNetAgent(
        ModelConfig(hidden_sizes=[2048, 4096, 8192, 8192, 4096],
                    learning_rate=1e-4, label="huge_net"),
    ),
    "nn_old": lambda: NeuralNetAgent(
        model_path="models/neural_net/self_play_trained_50-33-17.pt"
    ),
    "nn": lambda: NeuralNetAgent(
        model_path="models/neural_net/self_play_trained_02.pt"
    ),
    "nn_01": lambda: NeuralNetAgent2(
        cfg = ModelConfig(
            hidden_sizes=[256, 512, 512, 512, 256],
            learning_rate=1e-3,
            model_dir="models/neural_net_2/256-512-512-512-256-81/"
        ),
        model_path="models/neural_net_2/256-512-512-512-256-81/version_01.pt"
    ),
    "nn_06": lambda: NeuralNetAgent2(
        cfg = ModelConfig(
            hidden_sizes=[256, 512, 512, 512, 256],
            learning_rate=1e-3,
            model_dir="models/neural_net_2/256-512-512-512-256-81/"
        ),
        model_path="models/neural_net_2/256-512-512-512-256-81/version_06.pt"
    ),
    "nn2": lambda: NeuralNetAgent2(
        cfg = ModelConfig(
            hidden_sizes=[256, 512, 512, 512, 256],
            learning_rate=1e-3,
            model_dir="models/neural_net_2/256-512-512-512-256-81/"
        ),
        model_path="models/neural_net_2/256-512-512-512-256-81/version_01.pt"
    ),
    "nn_big_8": lambda: NeuralNetAgent2(
        cfg = BIG8_CFG, # Now Hard Defined up top
        model_path=_resolve_best_or_latest("big_8_layer", "256-512-1024-2048-2048-1024-512-256-81", fallback="version_01.pt")
    ),
    "nn_big_8_best": lambda: NeuralNetAgent2(cfg=BIG8_CFG, model_path=os.path.join(BIG8_DIR, "best.pt")),
    "nn_big_8_v01":  lambda: NeuralNetAgent2(cfg=BIG8_CFG, model_path=os.path.join(BIG8_DIR, "version_01.pt")),
    "new_cnn": lambda: NeuralNetAgent3(
        cfg = ModelConfigCNN(
            conv_channels=[32, 64, 64],
            fc_hidden_sizes=[256, 512, 1024, 512, 128],
            learning_rate=1e-5,
            label="new_cnn"
        ),
        model_path=""
    ),
    "lottery": lambda: NeuralNetAgent3(
        cfg = ModelConfigCNN(
            conv_channels=[64, 128, 256, 256],
            fc_hidden_sizes=[2048, 1024, 1024, 512, 256],
            learning_rate=1e-5,
            label="lottery"
        ),
        model_path=""
    ),
    "lottery_no_touchy": lambda: NeuralNetAgent3(
        cfg=ModelConfigCNN(
            conv_channels=[512] * 135,
            fc_hidden_sizes=[4096, 3072, 2048, 1024, 512, 256],
            learning_rate=1e-5,
            label="lottery"
        ),
        model_path=r"models\lottery\4096-3072-2048-1024-512-256-81\several_weeks_no_touchy.pt"
    ),
    "pg_best": lambda: NeuralNetAgentPG(
        cfg=ModelConfigCNN(
            conv_channels=[32, 64, 64],
            fc_hidden_sizes=[256, 512, 256],
            learning_rate=1e-4,
            label="league_pg"
        ),
        model_path=_resolve_latest_archive("models/league_pg/archive")
    ),
    # Gregory: depth-limited alpha-beta. Gene-pool-independent foil + partial oracle.
    # depth=1 fast warm-up | depth=3 solid curriculum anchor | depth=5 partial oracle
    "gregory":      lambda: GregoryAgent(depth=3),
    "gregory_d1":   lambda: GregoryAgent(depth=1),
    "gregory_d2":   lambda: GregoryAgent(depth=2),
    "gregory_deep": lambda: GregoryAgent(depth=5),
    "mcts_lottery": lambda: MCTSAgent(
        NeuralNetAgent3(
            cfg=ModelConfigCNN(
                conv_channels=[512] * 135,
                fc_hidden_sizes=[4096, 3072, 2048, 1024, 512, 256],
                learning_rate=1e-5,
                label="lottery"
            ),
            model_path=r"models\lottery\4096-3072-2048-1024-512-256-81\several_weeks_no_touchy.pt"
        ),
        n_sims=100,
    ),
}

def get_agent(name: str) -> Agent:
    try:
        factory = AGENT_FACTORIES[name]
    except KeyError:
        raise ValueError(f"No such agent: {name!r}")
    return factory()     # <-- only now does the ctor run

def _resolve_best_or_latest(label: str, arch_dir: str, fallback: str = "version_01.pt") -> str:
    model_dir = os.path.join("models", label, arch_dir)

    best = os.path.join(model_dir, "best.pt")
    if os.path.isfile(best):
        return best

    latest = _find_latest_checkpoint_version(model_dir)
    if latest is not None:
        return os.path.join(model_dir, f"version_{latest:02d}.pt")

    return os.path.join(model_dir, fallback)

def _find_latest_checkpoint_version(model_dir: str):
    pattern = os.path.join(model_dir, "version_*.pt")
    candidates = glob.glob(pattern)
    best = None
    for p in candidates:
        m = re.search(r"version_(\d+)\.pt$", p.replace("\\", "/"))
        if not m:
            continue
        v = int(m.group(1))
        best = v if best is None else max(best, v)
    return best

def _resolve_latest_archive(archive_dir: str) -> str:
    pattern = os.path.join(archive_dir, "archive_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return ""
    return max(candidates, key=lambda p: int(re.search(r"archive_(\d+)", p).group(1)))