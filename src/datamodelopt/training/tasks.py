"""
Task abstractions for different training objectives.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Task(ABC):
    """
    Abstract base class for training tasks.

    A Task defines how to compute forward pass and loss for a model.
    This allows the same training loop to work with different
    model types (LM, classification, etc.).
    """

    @abstractmethod
    def forward_and_loss(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Perform forward pass and compute loss.

        Args:
            model: The model to use.
            batch: A batch of data (typically (x, y) tuple).
            device: Device to move data to.

        Returns:
            Tuple of (loss, metrics_dict) where metrics_dict contains
            additional metrics like accuracy.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
    ) -> dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            model: The model to evaluate.
            dataloader: DataLoader for evaluation data.
            device: Device to use.

        Returns:
            Dictionary of metric name -> value.
        """
        pass


class LanguageModelingTask(Task):
    """
    Task for language modeling (e.g., NanoGPT).

    Assumes model returns (logits, loss) when targets are provided:
        logits, loss = model(x, y)
    """

    def __init__(self):
        """Initialize the language modeling task."""
        pass

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Forward pass for language modeling.

        Args:
            model: NanoGPT-style model.
            batch: (input_ids, target_ids) tuple.
            device: Target device.

        Returns:
            (loss, {"accuracy": token_accuracy})
        """
        x, y = batch
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)

        # Compute token-level accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == y).float()
            accuracy = correct.mean().item()

        return loss, {"accuracy": accuracy}

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
    ) -> dict[str, float]:
        """
        Evaluate language model.

        Returns:
            {"loss": avg_loss, "accuracy": avg_accuracy}
        """
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)

            total_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            correct = (predictions == y).float()
            total_acc += correct.mean().item()

            n_batches += 1

        return {
            "loss": total_loss / max(1, n_batches),
            "accuracy": total_acc / max(1, n_batches),
        }


class ClassificationTask(Task):
    """
    Task for classification.

    Assumes model returns (logits, loss) when targets are provided
    (via ModelWrapper or similar interface).
    """

    def __init__(self, criterion: nn.Module | None = None):
        """
        Initialize classification task.

        Args:
            criterion: Loss function. If None, expects model to return loss.
        """
        self.criterion = criterion

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Forward pass for classification.

        Args:
            model: Classification model.
            batch: (inputs, labels) tuple.
            device: Target device.

        Returns:
            (loss, {"accuracy": accuracy})
        """
        x, y = batch
        x, y = x.to(device), y.to(device)

        # Squeeze labels if needed
        if y.dim() > 1:
            y = y.squeeze()

        # Try to get logits and loss from model
        output = model(x, y)
        if isinstance(output, tuple):
            logits, loss = output
        else:
            logits = output
            if self.criterion is not None:
                loss = self.criterion(logits, y)
            else:
                raise ValueError("Model must return (logits, loss) or criterion must be provided")

        # Compute accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == y).float()
            accuracy = correct.mean().item()

        return loss, {"accuracy": accuracy}

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
    ) -> dict[str, float]:
        """
        Evaluate classification model.

        Returns:
            {"loss": avg_loss, "accuracy": avg_accuracy}
        """
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            if y.dim() > 1:
                y = y.squeeze()

            output = model(x, y)
            if isinstance(output, tuple):
                logits, loss = output
            else:
                logits = output
                if self.criterion is not None:
                    loss = self.criterion(logits, y)
                else:
                    loss = torch.tensor(0.0)

            total_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            correct = (predictions == y).float()
            total_acc += correct.mean().item()

            n_batches += 1

        return {
            "loss": total_loss / max(1, n_batches),
            "accuracy": total_acc / max(1, n_batches),
        }
