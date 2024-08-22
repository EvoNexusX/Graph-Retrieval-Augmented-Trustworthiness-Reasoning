from ..config import BackendConfig

from .base import IntelligenceBackend
from .openaichat import OpenAIChat
from .agentchat import AgentChat
from .agentchat2 import AgentChat2
from .agentchat3 import AgentChat3
from .agentchat4 import AgentChat4
from .agentchat5 import AgentChat5
from .agentchat6 import AgentChat6
from .human import Human
from .hf_transformers import TransformersConversational

ALL_BACKENDS = [
    Human,
    OpenAIChat,
    AgentChat,
    AgentChat2,
    AgentChat3,
    AgentChat4,
    AgentChat5,
    AgentChat6,
    TransformersConversational,
]

BACKEND_REGISTRY = {backend.type_name: backend for backend in ALL_BACKENDS}


# Load a backend from a config dictionary
def load_backend(config: BackendConfig, args=None):
    try:
        backend_cls = BACKEND_REGISTRY[config.backend_type]
    except KeyError:
        raise ValueError(f"Unknown backend type: {config.backend_type}")

    backend = backend_cls.from_config(config, args)
    return backend
import os
from pathlib import Path

FILE = Path(__file__).resolve()

ROOT_DIR = os.path.dirname(FILE)
