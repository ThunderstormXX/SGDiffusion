"""
Optimizer wrappers and utilities.
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Any, Callable


class ScheduledOptimizer:
    """
    Wrapper that combines an optimizer with an optional learning rate scheduler.
    
    Provides a unified interface for stepping both.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
    ):
        """
        Initialize the scheduled optimizer.
        
        Args:
            optimizer: The underlying optimizer.
            scheduler: Optional learning rate scheduler.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform optimization step and scheduler step.
        
        Args:
            closure: Optional closure for optimizers that require it.
        
        Returns:
            Loss if closure is provided, else None.
        """
        loss = self.optimizer.step(closure)
        if self.scheduler is not None:
            self.scheduler.step()
        return loss
    
    @property
    def param_groups(self):
        """Access optimizer param groups."""
        return self.optimizer.param_groups
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def set_lr(self, lr: float) -> None:
        """Set learning rate for all param groups."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        state = {"optimizer": self.optimizer.state_dict()}
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dict."""
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])


def create_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: Optional[int] = None,
) -> _LRScheduler:
    """
    Create a linear warmup scheduler.
    
    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps (for cosine decay after warmup).
    
    Returns:
        A learning rate scheduler.
    """
    if total_steps is not None and total_steps > warmup_steps:
        # Linear warmup + cosine annealing
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)
    else:
        # Linear warmup only
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        return LambdaLR(optimizer, lr_lambda)
