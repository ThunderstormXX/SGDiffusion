"""
Model factories and wrappers.
"""

from .base import ModelFactory
from .factories import ClassFactory, build_model_from_config
from .nanogpt import NanoGPTFactory

__all__ = [
    "ModelFactory",
    "NanoGPTFactory",
    "ClassFactory",
    "build_model_from_config",
]


def register_builtin_models():
    """Register all built-in model factories with the global registry."""
    from ..core.registry import registry

    registry.register_model("nanogpt", NanoGPTFactory)
    registry.register_model("class", ClassFactory)


# Auto-register on import
register_builtin_models()
