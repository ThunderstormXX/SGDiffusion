"""
Gradient history tracker.
"""

import torch
from typing import List, Optional, Any
from pathlib import Path

from .base import Tracker
from ..core.tensor_utils import flatten_grads


class GradientHistoryTracker(Tracker):
    """
    Tracker that records gradient trajectory.
    
    Records the flattened gradient vector at specified intervals.
    Can optionally compute full (deterministic) gradient over the dataset.
    """
    
    def __init__(
        self,
        every: int = 1,
        filename: str = "grads.pt",
        record_stochastic: bool = True,
        record_full: bool = False,
        **kwargs,
    ):
        """
        Initialize gradient history tracker.
        
        Args:
            every: Record every N steps.
            filename: Output filename.
            record_stochastic: Record stochastic (mini-batch) gradients.
            record_full: Record full-batch gradients (expensive).
            **kwargs: Additional arguments.
        """
        super().__init__(every=every, **kwargs)
        self.filename = filename
        self.record_stochastic = record_stochastic
        self.record_full = record_full
        
        # Storage - list of flattened gradient tensors on CPU
        self.stochastic_grads: List[torch.Tensor] = []
        self.full_grads: List[torch.Tensor] = []
        self.steps: List[int] = []
        
        # Stage tracking
        self.stage_name: str = ""
        self.stage_idx: int = 0
    
    def on_stage_start(self, ctx: Any) -> None:
        """Initialize for new stage."""
        super().on_stage_start(ctx)
        self.stage_name = ctx.stage_name
        self.stage_idx = ctx.stage_idx
        
        # Clear storage for this stage
        self.stochastic_grads = []
        self.full_grads = []
        self.steps = []
    
    def _on_step_end(self, ctx: Any) -> None:
        """Record gradients at this step."""
        self.steps.append(ctx.step_in_stage)
        
        # Record stochastic gradient (already computed during backward)
        if self.record_stochastic:
            flat_grads = flatten_grads(ctx.model).cpu()
            self.stochastic_grads.append(flat_grads)
        
        # Record full gradient if requested (expensive!)
        if self.record_full and ctx.train_loader is not None:
            full_grad = self._compute_full_gradient(ctx)
            if full_grad is not None:
                self.full_grads.append(full_grad.cpu())
    
    def _compute_full_gradient(self, ctx: Any) -> Optional[torch.Tensor]:
        """
        Compute full-batch gradient.
        
        This is expensive and recomputes gradients over the entire dataset.
        """
        model = ctx.model
        train_loader = ctx.train_loader
        device = ctx.device
        
        if train_loader is None:
            return None
        
        # Save current training mode
        was_training = model.training
        model.eval()
        
        # Accumulate gradients
        model.zero_grad()
        total_loss = 0.0
        n_batches = 0
        
        # Import task for forward computation
        # This is a simplified version - assumes loss is accessible
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Try to get loss from model
            output = model(x, y)
            if isinstance(output, tuple):
                _, loss = output
            else:
                # Shouldn't happen but just in case
                continue
            
            loss.backward()
            total_loss += loss.item()
            n_batches += 1
        
        # Get accumulated gradient
        flat_grad = flatten_grads(model).cpu()
        
        # Clear gradients
        model.zero_grad()
        
        # Restore training mode
        if was_training:
            model.train()
        
        return flat_grad
    
    def flush(self, save_dir: str) -> None:
        """Save gradients to .pt file."""
        if not self.stochastic_grads and not self.full_grads:
            return
        
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        
        # Add stage name to filename
        filename = self.filename
        if self.stage_name:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "pt")
            filename = f"{name}_{self.stage_name}.{ext}"
        
        data = {
            "steps": self.steps,
            "stage_name": self.stage_name,
            "stage_idx": self.stage_idx,
        }
        
        # Stack gradients if available
        if self.stochastic_grads:
            data["stochastic_grads"] = torch.stack(self.stochastic_grads)
        if self.full_grads:
            data["full_grads"] = torch.stack(self.full_grads)
        
        save_path = p / filename
        torch.save(data, str(save_path))
    
    def get_stochastic_trajectory(self) -> torch.Tensor:
        """Get stochastic gradient trajectory."""
        if not self.stochastic_grads:
            return torch.empty(0)
        return torch.stack(self.stochastic_grads)
    
    def get_full_trajectory(self) -> torch.Tensor:
        """Get full gradient trajectory."""
        if not self.full_grads:
            return torch.empty(0)
        return torch.stack(self.full_grads)
    
    def reset(self) -> None:
        """Reset storage."""
        super().reset()
        self.stochastic_grads = []
        self.full_grads = []
        self.steps = []
