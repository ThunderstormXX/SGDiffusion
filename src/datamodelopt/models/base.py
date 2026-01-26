"""
Base class for model factories.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class ModelFactory(ABC):
    """
    Abstract base class for model factories.
    
    A ModelFactory creates and configures a model instance.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the factory with configuration.
        
        Args:
            **kwargs: Configuration arguments for the model.
        """
        self.kwargs = kwargs
    
    @abstractmethod
    def build(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> nn.Module:
        """
        Build and return a model instance.
        
        Args:
            device: Target device for the model.
            dtype: Target dtype for the model.
        
        Returns:
            A configured torch.nn.Module.
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the model configuration."""
        return {"factory": self.__class__.__name__, "kwargs": self.kwargs}


class ModelWrapper(nn.Module):
    """
    Wrapper for models to provide a consistent interface.
    
    For classification models that don't return loss directly,
    this wrapper adds loss computation.
    """
    
    def __init__(self, model: nn.Module, criterion: Optional[nn.Module] = None):
        """
        Initialize the wrapper.
        
        Args:
            model: The underlying model.
            criterion: Loss function. If None, uses CrossEntropyLoss.
        """
        super().__init__()
        self.model = model
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):
        """
        Forward pass with optional loss computation.
        
        Args:
            x: Input tensor.
            y: Optional target tensor.
        
        Returns:
            If y is None: logits
            If y is provided: (logits, loss)
        """
        logits = self.model(x)
        if y is None:
            return logits
        loss = self.criterion(logits, y)
        return logits, loss
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
