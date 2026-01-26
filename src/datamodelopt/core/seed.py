"""
Seeding utilities for reproducibility.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: The seed value to use.
        deterministic: If True, enable PyTorch deterministic algorithms where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch versions don't have warn_only
                pass


def seed_worker(worker_id: int) -> None:
    """
    Worker init function for DataLoader to ensure reproducibility.
    
    Usage:
        DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create a PyTorch Generator with optional seed.
    
    Args:
        seed: Optional seed for the generator.
    
    Returns:
        A torch.Generator instance.
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return g
