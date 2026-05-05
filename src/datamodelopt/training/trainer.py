"""
Trainer implementation.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..core.tensor_utils import grad_norm
from .tasks import Task


class TrainingContext:
    """
    Context object passed to trackers during training.

    Contains all information about the current training state.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        dtype: torch.dtype,
        run_dir: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype
        self.run_dir = run_dir

        # Stage info
        self.stage_name: str = ""
        self.stage_idx: int = 0

        # Training progress
        self.global_step: int = 0
        self.step_in_stage: int = 0
        self.epoch: int = 0
        self.epoch_step: int = 0

        # Current batch info
        self.batch: tuple | None = None
        self.loss: torch.Tensor | None = None
        self.metrics: dict[str, Any] = {}

        # Evaluation results (set during eval)
        self.train_eval: dict[str, float] = {}
        self.val_eval: dict[str, float] = {}
        self.eval_step: int | None = None
        self.eval_global_step: int | None = None
        self.eval_epoch: int | None = None
        self.eval_counter: int = 0

        # Data loaders (set by stage runner)
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None


class Trainer:
    """
    A trainer that runs training steps and coordinates with trackers.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        task: Task,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train.
            optimizer: The optimizer to use.
            task: The task defining forward/loss computation.
            device: Training device.
            dtype: Training dtype.
        """
        self.model = model
        self.optimizer = optimizer
        self.task = task
        self.device = device
        self.dtype = dtype

    def train_step(
        self,
        batch: tuple,
    ) -> tuple:
        """
        Perform a single training step.

        Args:
            batch: A batch of data.

        Returns:
            (loss, metrics_dict)
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss, metrics = self.task.forward_and_loss(
            self.model, batch, self.device
        )

        loss.backward()

        # Add gradient norm to metrics
        metrics["grad_norm"] = grad_norm(self.model)

        self.optimizer.step()

        return loss, metrics

    def train_steps(
        self,
        train_loader: DataLoader,
        n_steps: int,
        ctx: TrainingContext,
        trackers: list = None,
        eval_fn: Callable | None = None,
        eval_every: int | None = None,
        progress: bool = True,
    ) -> None:
        """
        Train for a fixed number of steps.

        Args:
            train_loader: Training data loader.
            n_steps: Number of steps to train.
            ctx: Training context.
            trackers: List of trackers to call.
            eval_fn: Optional evaluation function.
            eval_every: Evaluate every N steps.
            progress: Show progress bar.
        """
        trackers = trackers or []
        data_iter = iter(train_loader)

        pbar = tqdm(range(n_steps), desc=f"Stage {ctx.stage_name}", disable=not progress)

        for step in pbar:
            # Get next batch, cycling if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Training step
            loss, metrics = self.train_step(batch)

            # Update context
            ctx.batch = batch
            ctx.loss = loss
            ctx.metrics = metrics
            ctx.metrics["loss"] = loss.item()
            ctx.step_in_stage = step + 1
            ctx.global_step += 1

            # Update progress bar (exclude 'loss' from metrics to avoid duplicate)
            pbar.set_postfix(loss=loss.item(), **{k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, float) and k != "loss"})

            # Call trackers
            for tracker in trackers:
                tracker.on_step_end(ctx)

            # Periodic evaluation
            if eval_fn is not None and eval_every is not None and (step + 1) % eval_every == 0:
                train_metrics, val_metrics = eval_fn()
                ctx.train_eval = train_metrics
                ctx.val_eval = val_metrics
                ctx.eval_step = ctx.step_in_stage
                ctx.eval_global_step = ctx.global_step
                ctx.eval_epoch = None
                ctx.eval_counter += 1

    def train_epochs(
        self,
        train_loader: DataLoader,
        n_epochs: int,
        ctx: TrainingContext,
        trackers: list = None,
        eval_fn: Callable | None = None,
        eval_every: int | None = None,
        progress: bool = True,
    ) -> None:
        """
        Train for a fixed number of epochs.

        Args:
            train_loader: Training data loader.
            n_epochs: Number of epochs to train.
            ctx: Training context.
            trackers: List of trackers to call.
            eval_fn: Optional evaluation function.
            eval_every: Evaluate every N epochs.
            progress: Show progress bar.
        """
        trackers = trackers or []

        for epoch in range(n_epochs):
            ctx.epoch = epoch + 1
            epoch_step = 0

            pbar = tqdm(
                train_loader,
                desc=f"Stage {ctx.stage_name} Epoch {epoch + 1}/{n_epochs}",
                disable=not progress,
            )

            for batch in pbar:
                # Training step
                loss, metrics = self.train_step(batch)

                # Update context
                ctx.batch = batch
                ctx.loss = loss
                ctx.metrics = metrics
                ctx.metrics["loss"] = loss.item()
                ctx.epoch_step = epoch_step + 1
                ctx.step_in_stage += 1
                ctx.global_step += 1
                epoch_step += 1

                # Update progress bar
                pbar.set_postfix(loss=loss.item())

                # Call trackers
                for tracker in trackers:
                    tracker.on_step_end(ctx)

            # Periodic evaluation
            if eval_fn is not None and eval_every is not None and (epoch + 1) % eval_every == 0:
                train_metrics, val_metrics = eval_fn()
                ctx.train_eval = train_metrics
                ctx.val_eval = val_metrics
                ctx.eval_step = ctx.step_in_stage
                ctx.eval_global_step = ctx.global_step
                ctx.eval_epoch = ctx.epoch
                ctx.eval_counter += 1

    def evaluate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple:
        """
        Evaluate model on train and validation sets.

        Returns:
            (train_metrics, val_metrics)
        """
        train_metrics = self.task.evaluate(self.model, train_loader, self.device)
        val_metrics = self.task.evaluate(self.model, val_loader, self.device)
        return train_metrics, val_metrics
