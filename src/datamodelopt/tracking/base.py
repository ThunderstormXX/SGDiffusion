"""
Base class for trackers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from pathlib import Path


class Tracker(ABC):
    """
    Abstract base class for training trackers.
    
    Trackers observe the training process and record various metrics,
    weight histories, gradients, etc.
    
    Lifecycle:
        1. on_run_start(ctx) - Called once at the start of the experiment
        2. on_stage_start(ctx) - Called at the start of each stage
        3. on_step_end(ctx) - Called after each training step
        4. on_stage_end(ctx) - Called at the end of each stage
        5. on_run_end(ctx) - Called once at the end of the experiment
        6. flush(save_dir) - Save any accumulated data
    """
    
    def __init__(self, every: int = 1, **kwargs):
        """
        Initialize the tracker.
        
        Args:
            every: Record every N steps (default: 1).
            **kwargs: Additional arguments.
        """
        self.every = every
        self.kwargs = kwargs
        self._step_counter = 0
    
    def on_run_start(self, ctx: Any) -> None:
        """Called at the start of the experiment."""
        pass
    
    def on_stage_start(self, ctx: Any) -> None:
        """Called at the start of each stage."""
        self._step_counter = 0
    
    def on_step_end(self, ctx: Any) -> None:
        """
        Called after each training step.
        
        Override _on_step_end for custom behavior.
        """
        self._step_counter += 1
        if self._step_counter % self.every == 0:
            self._on_step_end(ctx)
    
    def _on_step_end(self, ctx: Any) -> None:
        """Internal step end handler. Override this in subclasses."""
        pass
    
    def on_stage_end(self, ctx: Any) -> None:
        """Called at the end of each stage."""
        pass
    
    def on_run_end(self, ctx: Any) -> None:
        """Called at the end of the experiment."""
        pass
    
    @abstractmethod
    def flush(self, save_dir: str) -> None:
        """
        Save accumulated data to disk.
        
        Args:
            save_dir: Directory to save data to.
        """
        pass
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self._step_counter = 0
