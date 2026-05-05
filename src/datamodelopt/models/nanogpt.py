"""
NanoGPT model factory.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .base import ModelFactory


class NanoGPTFactory(ModelFactory):
    """
    Factory for creating NanoGPT models.

    Uses src.model.NanoGPT by import (no modifications to original file).
    """

    def __init__(
        self,
        vocab_size: int = 65,
        n_embd: int = 8,
        n_head: int = 1,
        n_layer: int = 1,
        block_size: int = 16,
        mlp_ratio: int = 1,
        meta_path: str | None = None,
        **kwargs,
    ):
        """
        Initialize NanoGPT factory.

        Args:
            vocab_size: Vocabulary size. Overridden by meta_path if provided.
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            n_layer: Number of transformer layers.
            block_size: Maximum sequence length. Overridden by meta_path if provided.
            mlp_ratio: MLP hidden dimension ratio.
            meta_path: Optional path to metadata file with vocab_size, block_size.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.mlp_ratio = mlp_ratio
        self.meta_path = meta_path

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> nn.Module:
        """
        Build a NanoGPT model.

        Args:
            device: Target device for the model.
            dtype: Target dtype for the model.

        Returns:
            A NanoGPT model instance.
        """
        # Import from src.model
        from src.model import NanoGPT

        vocab_size = self.vocab_size
        block_size = self.block_size

        # Load metadata if path provided
        if self.meta_path is not None and Path(self.meta_path).exists():
            meta = torch.load(self.meta_path, weights_only=False)
            vocab_size = meta.get("vocab_size", vocab_size)
            block_size = meta.get("block_size", block_size)

        model = NanoGPT(
            vocab_size=vocab_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
            block_size=block_size,
            mlp_ratio=self.mlp_ratio,
        )

        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        return model

    def get_info(self) -> dict[str, Any]:
        """Get model configuration info."""
        info = super().get_info()
        info.update({
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "block_size": self.block_size,
            "mlp_ratio": self.mlp_ratio,
        })
        return info
