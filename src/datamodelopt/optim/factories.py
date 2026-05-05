"""
Optimizer factories for common optimizers.
"""

from collections.abc import Iterator

import torch
import torch.nn as nn

from ..core.config import OptimizerConfig
from .base import OptimizerFactory


class SGDFactory(OptimizerFactory):
    """Factory for SGD optimizer."""

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        **kwargs,
    ):
        super().__init__(lr=lr, **kwargs)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def build(self, params: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )


class AdamFactory(OptimizerFactory):
    """Factory for Adam optimizer."""

    def __init__(
        self,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        **kwargs,
    ):
        super().__init__(lr=lr, **kwargs)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def build(self, params: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )


class AdamWFactory(OptimizerFactory):
    """Factory for AdamW optimizer (Adam with decoupled weight decay)."""

    def __init__(
        self,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        **kwargs,
    ):
        super().__init__(lr=lr, **kwargs)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def build(self, params: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )


class RMSpropFactory(OptimizerFactory):
    """Factory for RMSprop optimizer."""

    def __init__(
        self,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        **kwargs,
    ):
        super().__init__(lr=lr, **kwargs)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered

    def build(self, params: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(
            params,
            lr=self.lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )


class NoisySGDFactory(OptimizerFactory):
    """
    Factory for NoisySGD optimizer.

    Uses src.optimizer.NoisySGD by import.
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        noise_std: float = 0.0,
        **kwargs,
    ):
        super().__init__(lr=lr, **kwargs)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.noise_std = noise_std

    def build(self, params: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
        from src.optimizer import NoisySGD

        return NoisySGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            noise_std=self.noise_std,
        )


def build_optimizer_from_config(
    config: OptimizerConfig,
    params: Iterator[nn.Parameter],
) -> torch.optim.Optimizer:
    """
    Build an optimizer from an OptimizerConfig.

    Args:
        config: The optimizer configuration.
        params: Model parameters to optimize.

    Returns:
        An optimizer instance.
    """
    from ..core.registry import registry

    factory_cls = registry.get_optimizer(config.name)
    factory = factory_cls(**config.kwargs)
    return factory.build(params)
