"""
Base class for optimizer factories.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Iterator
import torch
import torch.nn as nn


class OptimizerFactory(ABC):
    """
    Abstract base class for optimizer factories.
    
    An OptimizerFactory creates and configures an optimizer instance.
    """
    
    def __init__(self, lr: float = 0.001, **kwargs):
        """
        Initialize the factory with configuration.
        
        Args:
            lr: Learning rate.
            **kwargs: Additional optimizer arguments.
        """
        self.lr = lr
        self.kwargs = kwargs
    
    @abstractmethod
    def build(self, params: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
        """
        Build and return an optimizer instance.
        
        Args:
            params: Model parameters to optimize.
        
        Returns:
            A configured torch.optim.Optimizer.
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the optimizer configuration."""
        return {
            "factory": self.__class__.__name__,
            "lr": self.lr,
            "kwargs": self.kwargs,
        }
