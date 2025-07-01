from .base import Agent
from .random_agent import RandomAgent
from .first_available_agent import FirstAvailableAgent

AGENT_REGISTRY = {
    "random": RandomAgent(),
    "first": FirstAvailableAgent(),
    # future agents can be added here
}