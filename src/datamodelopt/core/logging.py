"""
Logging utilities for experiments.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class RunLogger:
    """
    A simple logger for tracking metrics during a run.
    
    Unlike global loggers, each RunLogger is independent and can be
    serialized to JSON.
    """
    
    run_dir: Optional[str] = None
    metrics: Dict[str, List[Any]] = field(default_factory=dict)
    _start_time: Optional[float] = field(default=None, repr=False)
    _step_times: List[float] = field(default_factory=list, repr=False)
    
    def start(self) -> None:
        """Start timing the run."""
        self._start_time = time.time()
    
    def log(self, name: str, value: Any) -> None:
        """
        Log a metric value.
        
        Args:
            name: Name of the metric.
            value: Value to log (should be JSON-serializable).
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def log_dict(self, values: Dict[str, Any]) -> None:
        """
        Log multiple metrics at once.
        
        Args:
            values: Dictionary of metric name -> value.
        """
        for name, value in values.items():
            self.log(name, value)
    
    def step_time(self) -> float:
        """
        Record and return the time since the last step (or start).
        
        Returns:
            Time elapsed since last step in seconds.
        """
        now = time.time()
        if len(self._step_times) == 0:
            if self._start_time is None:
                self._start_time = now
            elapsed = now - self._start_time
        else:
            elapsed = now - self._step_times[-1]
        self._step_times.append(now)
        return elapsed
    
    def elapsed(self) -> float:
        """
        Get total elapsed time since start.
        
        Returns:
            Total time in seconds.
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metrics": self.metrics,
            "elapsed_time": self.elapsed(),
        }
    
    def save_json(self, path: str) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            path: Path to save the JSON file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any non-serializable values
        data = self.to_dict()
        with open(p, "w") as f:
            json.dump(data, f, indent=2, default=_json_serializer)
    
    @classmethod
    def load_json(cls, path: str) -> "RunLogger":
        """
        Load metrics from JSON file.
        
        Args:
            path: Path to the JSON file.
        
        Returns:
            A RunLogger instance with loaded metrics.
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        logger = cls()
        logger.metrics = data.get("metrics", {})
        return logger
    
    def get(self, name: str) -> List[Any]:
        """
        Get all values for a metric.
        
        Args:
            name: Name of the metric.
        
        Returns:
            List of values.
        """
        return self.metrics.get(name, [])
    
    def last(self, name: str, default: Any = None) -> Any:
        """
        Get the last value for a metric.
        
        Args:
            name: Name of the metric.
            default: Default value if metric doesn't exist.
        
        Returns:
            The last logged value or default.
        """
        values = self.metrics.get(name, [])
        return values[-1] if values else default


def _json_serializer(obj):
    """Custom JSON serializer for non-standard types."""
    import numpy as np
    import torch
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string.
    
    Args:
        seconds: Time in seconds.
    
    Returns:
        Formatted string like "1h 23m 45s" or "45.2s".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"
