from .base              import Agent
from .random_agent      import RandomAgent
from .first_available_agent import FirstAvailableAgent
from .neural_net_agent import NeuralNetAgent

# instead of calling each constructor now, store a factory
AGENT_FACTORIES = {
    "random": RandomAgent,   # classes are callable
    "first": FirstAvailableAgent,
    "nn_old": lambda: NeuralNetAgent(
        model_path="models/neural_net/self_play_trained_50-33-17.pt"
    ),
    "nn": lambda: NeuralNetAgent(
        model_path="models/neural_net/self_play_trained_00.pt"
    ),
}

def get_agent(name: str) -> Agent:
    try:
        factory = AGENT_FACTORIES[name]
    except KeyError:
        raise ValueError(f"No such agent: {name!r}")
    return factory()     # <-- only now does the ctor run