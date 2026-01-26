"""
Checkpointing utilities for saving and loading model states.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any


def save_model(
    path: str,
    model: torch.nn.Module,
    extra: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        path: Path to save the checkpoint.
        model: The model to save.
        extra: Optional dictionary with extra data to save.
        optimizer: Optional optimizer to save state.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if extra is not None:
        checkpoint["extra"] = extra
    
    torch.save(checkpoint, str(p))
    
    # Verify save
    if not p.exists():
        raise RuntimeError(f"Failed to save checkpoint to {p}")


def load_model(
    path: str,
    model: torch.nn.Module,
    map_location: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into.
        map_location: Device to map tensors to (e.g., "cpu", "cuda").
        optimizer: Optional optimizer to load state into.
        strict: Whether to strictly enforce that the keys in state_dict match.
    
    Returns:
        Dictionary with extra data from the checkpoint (if any).
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint.get("extra", {})


def save_weights_only(path: str, model: torch.nn.Module) -> None:
    """
    Save only model weights (simpler format).
    
    Args:
        path: Path to save weights.
        model: The model to save.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)


def load_weights_only(
    path: str,
    model: torch.nn.Module,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> None:
    """
    Load only model weights (simpler format).
    
    Args:
        path: Path to the weights file.
        model: The model to load weights into.
        map_location: Device to map tensors to.
        strict: Whether to strictly enforce that the keys match.
    """
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict, strict=strict)
