"""
Hessian tracker.

WARNING: Computing Hessians is very expensive (O(n^2) memory, O(n^2) time).
Only use this for small models (< 10,000 parameters).
"""

from pathlib import Path
from typing import Any

import torch

from ..core.tensor_utils import get_param_count
from .base import Tracker


class HessianTracker(Tracker):
    """
    Tracker that computes and records the Hessian matrix.

    WARNING: This is very expensive! Only use for small models.

    Computes the Hessian via autograd.functional.hessian or
    a custom implementation using second-order gradients.
    """

    def __init__(
        self,
        every: int = 100,
        filename: str = "hessians.pt",
        max_params: int = 10000,
        method: str = "autograd",
        batch_size_for_hessian: int | None = None,
        **kwargs,
    ):
        """
        Initialize Hessian tracker.

        Args:
            every: Compute Hessian every N steps.
            filename: Output filename.
            max_params: Maximum number of parameters to allow (safety check).
            method: Method for Hessian computation ("autograd" or "manual").
            batch_size_for_hessian: Batch size for Hessian computation. If None, use full batch.
            **kwargs: Additional arguments.
        """
        super().__init__(every=every, **kwargs)
        self.filename = filename
        self.max_params = max_params
        self.method = method
        self.batch_size_for_hessian = batch_size_for_hessian

        # Storage
        self.hessians: list[torch.Tensor] = []
        self.steps: list[int] = []
        self.eigenvalues: list[torch.Tensor] = []

        # Stage tracking
        self.stage_name: str = ""
        self.stage_idx: int = 0
        self._n_params: int = 0

    def on_stage_start(self, ctx: Any) -> None:
        """Check model size and initialize."""
        super().on_stage_start(ctx)
        self.stage_name = ctx.stage_name
        self.stage_idx = ctx.stage_idx

        # Check model size
        self._n_params = get_param_count(ctx.model)
        if self._n_params > self.max_params:
            print(
                f"WARNING: HessianTracker disabled. Model has {self._n_params} params, "
                f"but max_params={self.max_params}. Set max_params higher to enable."
            )

        # Clear storage
        self.hessians = []
        self.steps = []
        self.eigenvalues = []

    def _on_step_end(self, ctx: Any) -> None:
        """Compute and record Hessian."""
        if self._n_params > self.max_params:
            return

        if ctx.train_loader is None:
            return

        hessian = self._compute_hessian(ctx)
        if hessian is not None:
            self.hessians.append(hessian.cpu())
            self.steps.append(ctx.step_in_stage)

            # Compute eigenvalues
            try:
                eigs = torch.linalg.eigvalsh(hessian.cpu())
                self.eigenvalues.append(eigs)
            except Exception:
                pass

    def _compute_hessian(self, ctx: Any) -> torch.Tensor | None:
        """
        Compute Hessian matrix using second-order gradients.

        Uses a simplified approach suitable for small models.
        """
        model = ctx.model
        device = ctx.device
        train_loader = ctx.train_loader

        # Collect parameters
        params = list(model.parameters())

        # Get a batch of data
        if self.batch_size_for_hessian is not None:
            # Use specified batch size
            data_iter = iter(train_loader)
            try:
                batch = next(data_iter)
            except StopIteration:
                return None
        else:
            # Try to get full dataset (for small models)
            all_x, all_y = [], []
            for x, y in train_loader:
                all_x.append(x)
                all_y.append(y)
            batch = (torch.cat(all_x), torch.cat(all_y))

        x, y = batch
        x, y = x.to(device), y.to(device)

        # Compute loss
        model.zero_grad()
        output = model(x, y)
        if isinstance(output, tuple):
            _, loss = output
        else:
            return None

        # Compute gradient
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.flatten() for g in grads])

        n_params = flat_grad.numel()
        hessian = torch.zeros(n_params, n_params, device=device)

        # Compute Hessian row by row
        for i in range(n_params):
            # Get second derivatives w.r.t. all parameters
            grad2 = torch.autograd.grad(
                flat_grad[i],
                params,
                retain_graph=True,
                allow_unused=True,
            )

            # Flatten and store
            flat_grad2 = []
            for g in grad2:
                if g is not None:
                    flat_grad2.append(g.flatten())
                else:
                    # This shouldn't happen for typical losses
                    flat_grad2.append(torch.zeros(params[0].numel(), device=device))

            hessian[i] = torch.cat(flat_grad2)

        model.zero_grad()

        return hessian.detach()

    def flush(self, save_dir: str) -> None:
        """Save Hessians to .pt file."""
        if not self.hessians:
            return

        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)

        # Add stage name to filename
        filename = self.filename
        if self.stage_name:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "pt")
            filename = f"{name}_{self.stage_name}.{ext}"

        data = {
            "hessians": self.hessians,  # List of tensors (may be large!)
            "eigenvalues": self.eigenvalues,
            "steps": self.steps,
            "stage_name": self.stage_name,
            "stage_idx": self.stage_idx,
            "n_params": self._n_params,
        }

        save_path = p / filename
        torch.save(data, str(save_path))

        # Save metadata txt file
        txt_filename = filename.replace(".pt", "_info.txt")
        txt_path = p / txt_filename
        with open(txt_path, "w") as f:
            f.write("Hessian Tracker\n")
            f.write("=" * 50 + "\n")
            f.write(f"Stage: {self.stage_name} (index: {self.stage_idx})\n")
            f.write(f"Model parameters: {self._n_params}\n")
            f.write(f"Number of Hessians: {len(self.hessians)}\n")
            if self.hessians:
                f.write(f"Hessian shape: {self.hessians[0].shape}\n")
                f.write(f"  - Matrix size: {self._n_params} x {self._n_params}\n")
            if self.eigenvalues:
                f.write(f"Eigenvalues computed: {len(self.eigenvalues)}\n")
                f.write(f"Eigenvalues shape per step: {self.eigenvalues[0].shape}\n")
            f.write(f"Steps recorded: {len(self.steps)}\n")
            if self.steps:
                f.write(f"  First step: {self.steps[0]}\n")
                f.write(f"  Last step: {self.steps[-1]}\n")
                f.write(f"  Steps: {self.steps}\n")
            f.write(f"File: {filename}\n")
            f.write(f"File size: {save_path.stat().st_size / 1024:.2f} KB ({save_path.stat().st_size / (1024*1024):.2f} MB)\n")

        # CRITICAL: Clear memory after saving (Hessians are very large!)
        self.hessians.clear()
        self.eigenvalues.clear()
        self.steps.clear()
        del data

    def get_top_eigenvalues(self, k: int = 5) -> list[torch.Tensor]:
        """Get top k eigenvalues at each recorded step."""
        if not self.eigenvalues:
            return []
        return [eig[-k:] for eig in self.eigenvalues]

    def reset(self) -> None:
        """Reset storage."""
        super().reset()
        self.hessians = []
        self.steps = []
        self.eigenvalues = []
