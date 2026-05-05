"""
Optimizer factories and wrappers.
"""

from .base import OptimizerFactory
from .factories import (
    AdamFactory,
    AdamWFactory,
    NoisySGDFactory,
    RMSpropFactory,
    SGDFactory,
    build_optimizer_from_config,
)
from .wrappers import ScheduledOptimizer

__all__ = [
    "OptimizerFactory",
    "SGDFactory",
    "AdamFactory",
    "AdamWFactory",
    "RMSpropFactory",
    "NoisySGDFactory",
    "ScheduledOptimizer",
    "build_optimizer_from_config",
]


def register_builtin_optimizers():
    """Register all built-in optimizer factories with the global registry."""
    from ..core.registry import registry

    registry.register_optimizer("sgd", SGDFactory)
    registry.register_optimizer("adam", AdamFactory)
    registry.register_optimizer("adamw", AdamWFactory)
    registry.register_optimizer("rmsprop", RMSpropFactory)
    registry.register_optimizer("noisysgd", NoisySGDFactory)


# Auto-register on import
register_builtin_optimizers()
