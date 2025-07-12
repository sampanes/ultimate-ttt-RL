from .agent_base              import Agent
from .random_agent      import RandomAgent
from .first_available_agent import FirstAvailableAgent
from .neural_net_agent import NeuralNetAgent
from .neural_net_agent_2 import ModelConfig, NeuralNetAgent2
from .neural_net_agent_3 import ModelConfigCNN, NeuralNetAgent3 

# instead of calling each constructor now, store a factory
AGENT_FACTORIES = {
    "random": RandomAgent,   # classes are callable
    "first": FirstAvailableAgent,
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
        cfg = ModelConfig(
            hidden_sizes=[256, 512, 1024, 2048, 2048, 1024, 512, 256],
            learning_rate=1e-5,
            label="big_8_layer"
        ),
        model_path="models/big_8_layer/256-512-1024-2048-2048-1024-512-256-81/version_01.pt"
    ),
    "new_cnn": lambda: NeuralNetAgent3(
        cfg = ModelConfigCNN(
            conv_channels=[32, 64, 64],
            fc_hidden_sizes=[256, 512, 1024, 512, 128],
            learning_rate=1e-5,
            label="new_cnn"
        ),
        model_path="models/new_cnn/256-512-1024-512-128-81/version_04.pt"
    )
}

def get_agent(name: str) -> Agent:
    try:
        factory = AGENT_FACTORIES[name]
    except KeyError:
        raise ValueError(f"No such agent: {name!r}")
    return factory()     # <-- only now does the ctor run