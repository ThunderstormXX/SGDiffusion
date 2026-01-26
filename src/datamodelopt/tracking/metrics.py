"""
Metrics logging tracker.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import Tracker


class JsonMetricsLogger(Tracker):
    """
    Tracker that logs metrics to a JSON file.
    
    Logs:
        - loss (every step)
        - accuracy (every step)
        - grad_norm (every step)
        - step_time (optional)
        - val_loss (at eval_every intervals)
        - val_accuracy (at eval_every intervals)
    """
    
    def __init__(
        self,
        every: int = 1,
        filename: str = "metrics.json",
        log_eval: bool = True,
        **kwargs,
    ):
        """
        Initialize metrics logger.
        
        Args:
            every: Log every N steps.
            filename: Output filename.
            log_eval: Whether to log evaluation metrics.
            **kwargs: Additional arguments.
        """
        super().__init__(every=every, **kwargs)
        self.filename = filename
        self.log_eval = log_eval
        
        # Storage
        self.steps: List[int] = []
        self.losses: List[float] = []
        self.accuracies: List[float] = []
        self.grad_norms: List[float] = []
        
        self.val_steps: List[int] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        
        self.train_eval_losses: List[float] = []
        self.train_eval_accs: List[float] = []
        
        # Stage tracking
        self.stage_name: str = ""
        self.stage_idx: int = 0
    
    def on_stage_start(self, ctx: Any) -> None:
        """Reset for new stage."""
        super().on_stage_start(ctx)
        self.stage_name = ctx.stage_name
        self.stage_idx = ctx.stage_idx
        
        # Clear storage for this stage
        self.steps = []
        self.losses = []
        self.accuracies = []
        self.grad_norms = []
        self.val_steps = []
        self.val_losses = []
        self.val_accuracies = []
        self.train_eval_losses = []
        self.train_eval_accs = []
    
    def _on_step_end(self, ctx: Any) -> None:
        """Log step metrics."""
        self.steps.append(ctx.step_in_stage)
        self.losses.append(ctx.metrics.get("loss", 0.0))
        self.accuracies.append(ctx.metrics.get("accuracy", 0.0))
        self.grad_norms.append(ctx.metrics.get("grad_norm", 0.0))
        
        # Log eval metrics if available
        if self.log_eval and ctx.val_eval:
            self.val_steps.append(ctx.step_in_stage)
            self.val_losses.append(ctx.val_eval.get("loss", 0.0))
            self.val_accuracies.append(ctx.val_eval.get("accuracy", 0.0))
            
            if ctx.train_eval:
                self.train_eval_losses.append(ctx.train_eval.get("loss", 0.0))
                self.train_eval_accs.append(ctx.train_eval.get("accuracy", 0.0))
    
    def on_stage_end(self, ctx: Any) -> None:
        """Final eval at stage end."""
        if self.log_eval and ctx.val_eval:
            # Make sure last eval is captured
            if not self.val_steps or self.val_steps[-1] != ctx.step_in_stage:
                self.val_steps.append(ctx.step_in_stage)
                self.val_losses.append(ctx.val_eval.get("loss", 0.0))
                self.val_accuracies.append(ctx.val_eval.get("accuracy", 0.0))
                
                if ctx.train_eval:
                    self.train_eval_losses.append(ctx.train_eval.get("loss", 0.0))
                    self.train_eval_accs.append(ctx.train_eval.get("accuracy", 0.0))
    
    def flush(self, save_dir: str) -> None:
        """Save metrics to JSON."""
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        
        data = {
            "stage_name": self.stage_name,
            "stage_idx": self.stage_idx,
            "steps": self.steps,
            "losses": self.losses,
            "accuracies": self.accuracies,
            "grad_norms": self.grad_norms,
            "val_steps": self.val_steps,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "train_eval_losses": self.train_eval_losses,
            "train_eval_accs": self.train_eval_accs,
        }
        
        # Add stage name to filename
        filename = self.filename
        if self.stage_name:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "json")
            filename = f"{name}_{self.stage_name}.{ext}"
        
        with open(p / filename, "w") as f:
            json.dump(data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "steps": self.steps,
            "losses": self.losses,
            "accuracies": self.accuracies,
            "grad_norms": self.grad_norms,
            "val_steps": self.val_steps,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }
    
    def reset(self) -> None:
        """Reset all storage."""
        super().reset()
        self.steps = []
        self.losses = []
        self.accuracies = []
        self.grad_norms = []
        self.val_steps = []
        self.val_losses = []
        self.val_accuracies = []
        self.train_eval_losses = []
        self.train_eval_accs = []
