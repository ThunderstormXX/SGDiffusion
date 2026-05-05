"""
Weight history tracker.
"""

from pathlib import Path
from typing import Any

import torch

from ..core.tensor_utils import flatten_params
from .base import Tracker


class WeightHistoryTracker(Tracker):
    """
    Tracker that records weight trajectory.

    Records the flattened parameter vector at specified intervals
    and saves to a .pt file.
    """

    def __init__(
        self,
        every: int = 1,
        filename: str = "weights.pt",
        record_initial: bool = True,
        record_final: bool = True,
        **kwargs,
    ):
        """
        Initialize weight history tracker.

        Args:
            every: Record every N steps.
            filename: Output filename.
            record_initial: Whether to record initial weights.
            record_final: Whether to record final weights (even if not on interval).
            **kwargs: Additional arguments.
        """
        super().__init__(every=every, **kwargs)
        self.filename = filename
        self.record_initial = record_initial
        self.record_final = record_final

        # Storage - list of flattened weight tensors on CPU
        self.weights: list[torch.Tensor] = []
        self.steps: list[int] = []

        # Stage tracking
        self.stage_name: str = ""
        self.stage_idx: int = 0
        self._recorded_final = False

    def on_stage_start(self, ctx: Any) -> None:
        """Record initial weights if configured."""
        super().on_stage_start(ctx)
        self.stage_name = ctx.stage_name
        self.stage_idx = ctx.stage_idx
        self._recorded_final = False

        # Clear storage for this stage
        self.weights = []
        self.steps = []

        if self.record_initial:
            flat_weights = flatten_params(ctx.model, detach=True).cpu()
            self.weights.append(flat_weights)
            self.steps.append(0)

    def _on_step_end(self, ctx: Any) -> None:
        """Record weights at this step."""
        flat_weights = flatten_params(ctx.model, detach=True).cpu()
        self.weights.append(flat_weights)
        self.steps.append(ctx.step_in_stage)

    def on_stage_end(self, ctx: Any) -> None:
        """Record final weights if configured."""
        if self.record_final and not self._recorded_final:
            # Check if we already recorded this step
            if not self.steps or self.steps[-1] != ctx.step_in_stage:
                flat_weights = flatten_params(ctx.model, detach=True).cpu()
                self.weights.append(flat_weights)
                self.steps.append(ctx.step_in_stage)
            self._recorded_final = True

    def flush(self, save_dir: str) -> None:
        """Save weights to .pt file."""
        if not self.weights:
            return

        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)

        # Stack into a single tensor [n_snapshots, n_params]
        weight_tensor = torch.stack(self.weights)

        # Add stage name to filename
        filename = self.filename
        if self.stage_name:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "pt")
            filename = f"{name}_{self.stage_name}.{ext}"

        data = {
            "weights": weight_tensor,
            "steps": self.steps,
            "stage_name": self.stage_name,
            "stage_idx": self.stage_idx,
        }

        save_path = p / filename
        torch.save(data, str(save_path))

        # Save metadata txt file
        txt_filename = filename.replace(".pt", "_info.txt")
        txt_path = p / txt_filename
        with open(txt_path, "w") as f:
            f.write("Weight History Tracker\n")
            f.write("=" * 50 + "\n")
            f.write(f"Stage: {self.stage_name} (index: {self.stage_idx})\n")
            f.write(f"Tensor shape: {weight_tensor.shape}\n")
            f.write(f"  - n_snapshots: {weight_tensor.shape[0]}\n")
            f.write(f"  - n_params: {weight_tensor.shape[1]}\n")
            f.write(f"Steps recorded: {len(self.steps)}\n")
            f.write(f"  First step: {self.steps[0]}\n")
            f.write(f"  Last step: {self.steps[-1]}\n")
            f.write(f"  Steps: {self.steps}\n")
            f.write(f"File: {filename}\n")
            f.write(f"File size: {save_path.stat().st_size / 1024:.2f} KB\n")

        # CRITICAL: Clear memory after saving to avoid accumulation
        self.weights.clear()
        self.steps.clear()
        del weight_tensor
        del data

    def get_trajectory(self) -> torch.Tensor:
        """
        Get weight trajectory as a tensor.

        Returns:
            Tensor of shape [n_snapshots, n_params].
        """
        if not self.weights:
            return torch.empty(0)
        return torch.stack(self.weights)

    def reset(self) -> None:
        """Reset storage."""
        super().reset()
        self.weights = []
        self.steps = []
        self._recorded_final = False
