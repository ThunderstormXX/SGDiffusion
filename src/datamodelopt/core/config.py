"""
Configuration dataclasses for experiments.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class DataConfig:
    """Configuration for a data module."""
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataConfig":
        return cls(**d)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**d)


@dataclass
class OptimizerConfig:
    """Configuration for an optimizer."""
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizerConfig":
        return cls(**d)


@dataclass
class TrackerConfig:
    """Configuration for a tracker."""
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrackerConfig":
        return cls(**d)


@dataclass
class TunnelConfig:
    """
    Configuration for a training tunnel.

    A tunnel is a stage in the pipeline with multiple runs.
    Each tunnel can continue from a previous tunnel (source_tunnel).

    Attributes:
        tunnel_index: Index of this tunnel (0, 1, 2, ...)
        description: Human-readable description (stored in JSON)
        mode: Training mode ("steps" or "epochs")
        steps: Number of steps (if mode="steps")
        epochs: Number of epochs (if mode="epochs")
        optimizer: Optimizer configuration (includes lr, momentum, etc)
        dataloader_mode: Minibatch or fullbatch
        batch_size: Override data config batch size if set
        trackers: List of tracker configurations
        eval_every: Evaluate every N steps/epochs
        save_checkpoint: Whether to save checkpoint after this tunnel
        source_mode: How to load from previous tunnel:
            - None: No source (fresh start)
            - "1to1": run_i continues from source_tunnel/run_i
            - "cartesian": cartesian product from previous tunnel runs
        source_run_index: If source_mode="cartesian", which run to load from
        n_initial_weights: (Only for first tunnel) Number of different initial weights.
            Each initial weight spawns n_runs trajectories.
            Total runs = n_initial_weights * n_runs.
        source_fanout: If source_mode="cartesian", number of runs per source run
    """
    tunnel_index: int
    description: str = ""
    mode: Literal["steps", "epochs"] = "steps"
    steps: int | None = None
    epochs: int | None = None
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(name="sgd"))
    dataloader_mode: Literal["minibatch", "fullbatch"] = "minibatch"
    batch_size: int | None = None
    trackers: list[TrackerConfig] = field(default_factory=list)
    eval_every: int | None = None
    save_checkpoint: bool = True
    source_mode: Literal["1to1", "cartesian"] | None = None
    source_run_index: int = 0
    n_initial_weights: int = 1
    source_fanout: int | None = None

    def __post_init__(self):
        if self.mode == "steps" and self.steps is None:
            raise ValueError("steps must be set when mode='steps'")
        if self.mode == "epochs" and self.epochs is None:
            raise ValueError("epochs must be set when mode='epochs'")
        if self.source_mode == "cartesian" and self.source_run_index is None:
            raise ValueError("source_run_index must be set when source_mode='cartesian'")

    @property
    def name(self) -> str:
        """Generate tunnel name: tunnel_0, tunnel_1, etc."""
        return f"tunnel_{self.tunnel_index}"

    def get_learning_rate(self) -> float:
        """Extract learning rate from optimizer config."""
        return self.optimizer.kwargs.get("lr", 0.0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Add computed fields
        d["name"] = self.name
        d["learning_rate"] = self.get_learning_rate()
        # Convert nested configs
        d["optimizer"] = self.optimizer.to_dict() if self.optimizer else None
        d["trackers"] = [t.to_dict() for t in self.trackers]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TunnelConfig":
        d = d.copy()
        # Remove computed fields
        d.pop("name", None)
        d.pop("learning_rate", None)
        if "optimizer" in d and d["optimizer"] is not None:
            d["optimizer"] = OptimizerConfig.from_dict(d["optimizer"])
        if "trackers" in d:
            d["trackers"] = [TrackerConfig.from_dict(t) for t in d["trackers"]]
        return cls(**d)


# Keep StageConfig for backward compatibility
@dataclass
class StageConfig:
    """DEPRECATED: Use TunnelConfig instead. Kept for backward compatibility."""
    name: str
    mode: Literal["steps", "epochs"] = "steps"
    steps: int | None = None
    epochs: int | None = None
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(name="sgd"))
    dataloader_mode: Literal["minibatch", "fullbatch"] = "minibatch"
    batch_size: int | None = None
    trackers: list[TrackerConfig] = field(default_factory=list)
    eval_every: int | None = None
    load_checkpoint: str | None = None
    save_checkpoint: str | None = None

    def __post_init__(self):
        if self.mode == "steps" and self.steps is None:
            raise ValueError("steps must be set when mode='steps'")
        if self.mode == "epochs" and self.epochs is None:
            raise ValueError("epochs must be set when mode='epochs'")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["optimizer"] = self.optimizer.to_dict() if self.optimizer else None
        d["trackers"] = [t.to_dict() for t in self.trackers]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StageConfig":
        d = d.copy()
        if "optimizer" in d and d["optimizer"] is not None:
            d["optimizer"] = OptimizerConfig.from_dict(d["optimizer"])
        if "trackers" in d:
            d["trackers"] = [TrackerConfig.from_dict(t) for t in d["trackers"]]
        return cls(**d)


@dataclass
class ExperimentConfig:
    """
    Configuration for a full experiment with tunnel-based structure.

    Supports both new tunnel-based and legacy stage-based experiments.

    Attributes:
        run_dir: Base directory for experiment results
        seed: Base random seed
        device: Computation device
        dtype: Data type for tensors
        data: Data configuration
        model: Model configuration
        tunnels: List of tunnel configurations (NEW)
        stages: List of stage configurations (DEPRECATED, for backward compatibility)
    """
    run_dir: str
    seed: int = 42
    device: str = "cpu"  # "cpu", "cuda", "mps"
    dtype: str = "float32"  # "float32", "float64", "bfloat16"
    data: DataConfig = field(default_factory=lambda: DataConfig(name="shakespeare"))
    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="nanogpt"))
    tunnels: list[TunnelConfig] = field(default_factory=list)
    stages: list[StageConfig] = field(default_factory=list)  # DEPRECATED

    @property
    def is_tunnel_based(self) -> bool:
        """Check if this experiment uses tunnel-based structure."""
        return len(self.tunnels) > 0

    @property
    def num_tunnels(self) -> int:
        """Number of tunnels in the experiment."""
        return len(self.tunnels) if self.is_tunnel_based else len(self.stages)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with all metadata."""
        d = {
            "run_dir": self.run_dir,
            "seed": self.seed,
            "is_tunnel_based": self.is_tunnel_based,
            "num_tunnels": self.num_tunnels,
            "device": self.device,
            "dtype": self.dtype,
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
        }
        # Add tunnels if present (NEW)
        if self.tunnels:
            d["tunnels"] = [t.to_dict() for t in self.tunnels]
        # Add stages for backward compatibility (DEPRECATED)
        if self.stages:
            d["stages"] = [s.to_dict() for s in self.stages]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        d = d.copy()
        # Remove computed fields
        d.pop("is_tunnel_based", None)
        d.pop("num_tunnels", None)
        d["data"] = DataConfig.from_dict(d["data"])
        d["model"] = ModelConfig.from_dict(d["model"])
        # Load tunnels if present
        if "tunnels" in d:
            d["tunnels"] = [TunnelConfig.from_dict(t) for t in d["tunnels"]]
        # Load stages for backward compatibility
        if "stages" in d:
            d["stages"] = [StageConfig.from_dict(s) for s in d["stages"]]
        return cls(**d)

    def save_json(self, path: str) -> None:
        """Save config to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)


def get_torch_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def get_device(device_str: str):
    """Get torch device, validating availability."""
    import torch
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    elif device_str == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    elif device_str == "cpu":
        return torch.device("cpu")
    else:
        # Try to parse as a device string (e.g., "cuda:0")
        return torch.device(device_str)
