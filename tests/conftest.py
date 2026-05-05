"""Pytest fixtures for SGDiffusion tests."""
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Get CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    """Get float32 dtype for testing."""
    return torch.float32


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_config() -> dict[str, Any]:
    """Minimal JSON config for fast testing."""
    return {
        "experiment_name": "test_experiment",
        "description": "Test: 10 steps -> 10 epochs -> 10 steps x 3 runs",
        "data": {
            "name": "mnist",
            "sample_size": 100,  # Very small for fast tests
            "replacement": True,
        },
        "model": {
            "name": "class",
            "class_name": "FlexibleMLP",
            "hidden_dim": 4,  # Tiny model
            "num_hidden_layers": 1,
            "input_downsample": 6,
        },
        "tunnels": [
            {
                "name": "tunnel_0",
                "description": "SGD: 10 steps",
                "mode": "steps",
                "steps": 10,
                "optimizer": "sgd",
                "lr": 0.1,
                "batch_size": 10,
                "n_runs": 1,
                "trackers": {"metrics": {"every": 5}},
                "eval_every": 5,
                "save_checkpoint": True,
                "source_mode": None,
            },
            {
                "name": "tunnel_1",
                "description": "GD: 10 epochs",
                "mode": "epochs",
                "epochs": 10,
                "optimizer": "sgd",
                "lr": 0.01,
                "batch_size": 100,
                "dataloader_mode": "fullbatch",
                "n_runs": 1,
                "trackers": {"metrics": {"every": 1}},
                "eval_every": 1,
                "save_checkpoint": True,
                "source_mode": "1to1",
            },
            {
                "name": "tunnel_2",
                "description": "Many SGD: 10 steps x 3 runs",
                "mode": "steps",
                "steps": 10,
                "optimizer": "sgd",
                "lr": 0.1,
                "batch_size": 10,
                "n_runs": 3,
                "trackers": {"metrics": {"every": 5}, "weights": {"every": 5}},
                "eval_every": 5,
                "save_checkpoint": True,
                "source_mode": "cartesian",
            },
        ],
        "device": "cpu",
        "dtype": "float32",
    }


@pytest.fixture
def config_file(temp_dir: Path, minimal_config: dict) -> Path:
    """Create a temporary config file."""
    config_path = temp_dir / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(minimal_config, f, indent=2)
    return config_path
