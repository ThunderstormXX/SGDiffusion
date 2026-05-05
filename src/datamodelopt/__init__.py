"""
datamodelopt - A modular framework for training experiments with tracking and multi-stage pipelines.

Main components:
- core: Configuration, registry, checkpointing, utilities
- data: Data modules for various datasets
- models: Model factories and wrappers
- optim: Optimizer factories and wrappers
- training: Tasks and trainer implementations
- tracking: Trackers for weights, gradients, metrics, hessians
- experiments: Pipeline and experiment runner
- visualization: Plotting and visualization utilities
"""

from .core.checkpointing import load_model, save_model
from .core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    StageConfig,
    TrackerConfig,
)
from .core.registry import Registry
from .experiments.pipeline import ExperimentRunner, StageRunner

__version__ = "0.1.0"

__all__ = [
    # Config
    "DataConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrackerConfig",
    "StageConfig",
    "ExperimentConfig",
    # Registry
    "Registry",
    # Checkpointing
    "save_model",
    "load_model",
    # Pipeline
    "ExperimentRunner",
    "StageRunner",
]
