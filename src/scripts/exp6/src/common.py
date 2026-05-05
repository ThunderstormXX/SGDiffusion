#!/usr/bin/env python3
"""Common utilities for exp6 reproducible experiments."""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml


EXP6_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP6_ROOT.parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_yaml(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_status_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def collect_environment() -> dict[str, Any]:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": git_commit(),
        "git_status_short": git_status_short(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / 1024**3,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
    }
    if torch.cuda.is_available():
        env["cuda_device_count"] = torch.cuda.device_count()
        env["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return env


def set_reproducible(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def mean_std_ci(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.mean(arr, axis=axis)
    std = np.std(arr, axis=axis, ddof=1) if arr.shape[axis] > 1 else np.zeros_like(mean)
    n = arr.shape[axis]
    half = 1.96 * std / np.sqrt(max(n, 1))
    return mean, std, mean - half, mean + half


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def run_context(result_dir: Path, config: dict[str, Any]):
    result_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    start = time.perf_counter()
    start_mem = process.memory_info().rss
    yield
    runtime = time.perf_counter() - start
    end_mem = process.memory_info().rss
    runtime_info = {
        "runtime_seconds": runtime,
        "rss_start_mb": start_mem / 1024**2,
        "rss_end_mb": end_mem / 1024**2,
        "rss_peak_note": "psutil RSS sampled at process end; not a true peak.",
    }
    save_json(result_dir / "runtime.json", runtime_info)
    save_json(result_dir / "environment.json", collect_environment())
    save_yaml(result_dir / "config.yaml", config)


def copy_make_figure(result_dir: Path) -> None:
    src = EXP6_ROOT / "scripts" / "make_figure.py"
    dst = result_dir / "make_figure.py"
    shutil.copy2(src, dst)


def write_experiment_readme(result_dir: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    lines = [
        f"# {config['experiment_id']}: {config.get('name', '')}",
        "",
        config.get("description", ""),
        "",
        "## Reproduce",
        "",
        "```bash",
        f"bash {EXP6_ROOT}/scripts/run_one.sh {result_dir / 'config.yaml'}",
        "```",
        "",
        "## Artifacts",
        "",
        "- `config.yaml`: exact configuration used for this run.",
        "- `environment.json`: Python, package, hardware, git metadata.",
        "- `runtime.json`: runtime and RSS memory snapshot.",
        "- `metrics.json`: machine-readable primary metrics.",
        "- `raw_outputs.npz`: raw trajectories/statistics.",
        "- `figure_data.csv`: plotted data with mean/std/95% CI when applicable.",
        "- `make_figure.py`: figure generation from saved artifacts only.",
        "",
        "## Primary Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
    ]
    (result_dir / "README_experiment.md").write_text("\n".join(lines))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(np.ravel(a), np.ravel(b)))
    except Exception:
        aa = np.sort(np.ravel(a))
        bb = np.sort(np.ravel(b))
        n = min(len(aa), len(bb))
        return float(np.mean(np.abs(aa[:n] - bb[:n])))
