"""
Generic model factories for building models by class name.
"""

from typing import Any

import torch
import torch.nn as nn

from ..core.config import ModelConfig
from .base import ModelFactory, ModelWrapper


class ClassFactory(ModelFactory):
    """
    Generic factory that builds models from src.model by class name.

    Example config:
        {"name": "class", "kwargs": {"class_name": "FlexibleMLP", "hidden_dim": 8, ...}}
    """

    def __init__(
        self,
        class_name: str,
        wrap_for_classification: bool = True,
        **model_kwargs,
    ):
        """
        Initialize class factory.

        Args:
            class_name: Name of the class in src.model (e.g., "FlexibleMLP", "CNN").
            wrap_for_classification: If True, wrap model with ModelWrapper for loss computation.
            **model_kwargs: Arguments passed to the model constructor.
        """
        super().__init__(**model_kwargs)
        self.class_name = class_name
        self.wrap_for_classification = wrap_for_classification
        self.model_kwargs = model_kwargs

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> nn.Module:
        """
        Build a model by class name.

        Args:
            device: Target device for the model.
            dtype: Target dtype for the model.

        Returns:
            A model instance.
        """
        # Import from src.model
        import src.model as model_module

        if not hasattr(model_module, self.class_name):
            raise ValueError(
                f"Class '{self.class_name}' not found in src.model. "
                f"Available: {[n for n in dir(model_module) if not n.startswith('_')]}"
            )

        model_cls = getattr(model_module, self.class_name)
        model = model_cls(**self.model_kwargs)

        if self.wrap_for_classification:
            model = ModelWrapper(model)

        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        return model

    def get_info(self) -> dict[str, Any]:
        """Get model configuration info."""
        info = super().get_info()
        info.update({
            "class_name": self.class_name,
            "model_kwargs": self.model_kwargs,
        })
        return info


def build_model_from_config(
    config: ModelConfig,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """
    Build a model from a ModelConfig.

    Args:
        config: The model configuration.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A model instance.
    """
    from ..core.registry import registry

    factory_cls = registry.get_model(config.name)
    factory = factory_cls(**config.kwargs)
    return factory.build(device=device, dtype=dtype)
