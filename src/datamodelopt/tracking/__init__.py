"""
Trackers for recording training history (weights, gradients, metrics, etc.).
"""

from .base import Tracker
from .grads import GradientHistoryTracker
from .hessian import HessianTracker
from .metrics import JsonMetricsLogger
from .weights import WeightHistoryTracker

__all__ = [
    "Tracker",
    "JsonMetricsLogger",
    "WeightHistoryTracker",
    "GradientHistoryTracker",
    "HessianTracker",
]


def register_builtin_trackers():
    """Register all built-in trackers with the global registry."""
    from ..core.registry import registry

    registry.register_tracker("json_metrics", JsonMetricsLogger)
    registry.register_tracker("metrics", JsonMetricsLogger)
    registry.register_tracker("weight_history", WeightHistoryTracker)
    registry.register_tracker("weights", WeightHistoryTracker)
    registry.register_tracker("gradient_history", GradientHistoryTracker)
    registry.register_tracker("grads", GradientHistoryTracker)
    registry.register_tracker("hessian", HessianTracker)


# Auto-register on import
register_builtin_trackers()
