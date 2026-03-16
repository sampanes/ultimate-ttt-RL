import os

from agents.agent_base import ModelConfigCNN
from agents.neural_net_agent_3 import NeuralNetAgent3


SUPER_CFG = ModelConfigCNN(
    conv_channels=[64, 128, 256, 256, 128],
    fc_hidden_sizes=[1024, 2048, 4096, 2048, 1024, 512],
    learning_rate=5e-5,
    label="super_agent",
)

SUPER_MODEL_PATH = os.path.join(SUPER_CFG.model_dir, "super_agent.pt")


def build_super_agent() -> NeuralNetAgent3:
    model_path = SUPER_MODEL_PATH if os.path.isfile(SUPER_MODEL_PATH) else None
    return NeuralNetAgent3(cfg=SUPER_CFG, model_path=model_path)
