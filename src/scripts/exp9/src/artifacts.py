from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig


class ArtifactStore:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.root = cfg.result_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def json(self, name: str, obj: dict) -> None:
        with self.path(name).open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")

    def npz(self, name: str, **arrays: np.ndarray) -> None:
        np.savez_compressed(self.path(name), **arrays)

    def text(self, name: str, value: str) -> None:
        self.path(name).write_text(value, encoding="utf-8")
