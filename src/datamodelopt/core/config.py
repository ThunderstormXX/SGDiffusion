"""
Configuration dataclasses for experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for a data module."""
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        return cls(**d)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**d)


@dataclass
class OptimizerConfig:
    """Configuration for an optimizer."""
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        return cls(**d)


@dataclass
class TrackerConfig:
    """Configuration for a tracker."""
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrackerConfig":
        return cls(**d)


@dataclass
class StageConfig:
    """Configuration for a training stage."""
    name: str
    mode: Literal["steps", "epochs"] = "steps"
    steps: Optional[int] = None
    epochs: Optional[int] = None
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(name="sgd"))
    dataloader_mode: Literal["minibatch", "fullbatch"] = "minibatch"
    batch_size: Optional[int] = None  # Override data config batch size if set
    trackers: List[TrackerConfig] = field(default_factory=list)
    eval_every: Optional[int] = None  # Evaluate every N steps/epochs
    load_checkpoint: Optional[str] = None  # Path to load checkpoint from
    save_checkpoint: Optional[str] = None  # Path to save checkpoint to
    
    def __post_init__(self):
        if self.mode == "steps" and self.steps is None:
            raise ValueError("steps must be set when mode='steps'")
        if self.mode == "epochs" and self.epochs is None:
            raise ValueError("epochs must be set when mode='epochs'")
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert nested configs
        d["optimizer"] = self.optimizer.to_dict() if self.optimizer else None
        d["trackers"] = [t.to_dict() for t in self.trackers]
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageConfig":
        d = d.copy()
        if "optimizer" in d and d["optimizer"] is not None:
            d["optimizer"] = OptimizerConfig.from_dict(d["optimizer"])
        if "trackers" in d:
            d["trackers"] = [TrackerConfig.from_dict(t) for t in d["trackers"]]
        return cls(**d)


@dataclass
class ExperimentConfig:
    """Configuration for a full experiment."""
    run_dir: str
    seed: int = 42
    device: str = "cpu"  # "cpu", "cuda", "mps"
    dtype: str = "float32"  # "float32", "float64", "bfloat16"
    data: DataConfig = field(default_factory=lambda: DataConfig(name="shakespeare"))
    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="nanogpt"))
    stages: List[StageConfig] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "run_dir": self.run_dir,
            "seed": self.seed,
            "device": self.device,
            "dtype": self.dtype,
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
        }
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        d = d.copy()
        d["data"] = DataConfig.from_dict(d["data"])
        d["model"] = ModelConfig.from_dict(d["model"])
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
        with open(path, "r") as f:
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
