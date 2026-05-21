from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    seed: int = 42
    result_name: str = "local_minima_full"
    results_root: str = "src/scripts/exp9/results"
    n_samples: int = 280
    moon_noise: float = 0.18
    hidden_dim: int = 4
    weight_decay: float = 1e-4
    init_scale: float = 1.0
    sgd_lr: float = 0.18
    sgd_momentum: float = 0.85
    train_steps: int = 1800
    lbfgs_steps: int = 350
    n_runs: int = 260
    cluster_eps: float = 0.18
    top_clusters: int = 10
    segment_points: int = 241
    hessian_clusters: int = 8
    probe_radius: float = 0.25
    probe_per_cluster: int = 24
    prior_samples: int = 4096
    prior_refine: int = 32
    prior_scale: float = 3.0
    batch_size: int = 4096
    make_mds: bool = True
    walk_runs: int = 3
    walk_steps: int = 3000
    walk_lr: float = 0.08
    walk_batch_size: int = 32
    walk_eval_every: int = 10
    walk_start_noise: float = 0.02
    pair_neighborhood_samples: int = 1000
    pair_radius_factor: float = 0.12
    geodesic_neighbors: int = 12
    gd_validate_steps: int = 50000
    gd_validate_lr: float = 0.5
    lr_sweep: tuple[float, ...] = (0.02, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5)
    lr_sweep_steps: int = 5000
    lr_sweep_eval_every: int = 50

    @property
    def result_dir(self) -> Path:
        return Path(self.results_root) / self.result_name

    def to_dict(self) -> dict:
        return asdict(self)


def apply_preset(cfg: ExperimentConfig, preset: str) -> ExperimentConfig:
    cfg.result_name = f"local_minima_{preset}"
    if preset == "smoke":
        overrides = {
            "n_runs": 24, "train_steps": 350, "lbfgs_steps": 180, "prior_samples": 256,
            "hessian_clusters": 3, "top_clusters": 5, "probe_per_cluster": 4,
            "prior_refine": 4, "segment_points": 81, "walk_runs": 2, "walk_steps": 180,
            "walk_eval_every": 5, "pair_neighborhood_samples": 240, "geodesic_neighbors": 8,
            "gd_validate_steps": 50000, "make_mds": True,
            "lr_sweep_steps": 2500, "lr_sweep_eval_every": 50,
        }
        for key, value in overrides.items():
            setattr(cfg, key, value)
    elif preset != "full":
        raise ValueError(f"unknown preset: {preset}")
    return cfg


def parse_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="full", choices=["smoke", "full"])
    parser.add_argument("--result_name", default="")
    types = {
        "n_runs": int, "train_steps": int, "lbfgs_steps": int, "seed": int,
        "cluster_eps": float, "probe_per_cluster": int, "walk_runs": int,
        "walk_steps": int, "walk_lr": float, "walk_batch_size": int,
    }
    for name, kind in types.items():
        parser.add_argument(f"--{name}", type=kind, default=None)
    parser.add_argument("--lr_sweep", default="")
    parser.add_argument("--make_mds", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    cfg = apply_preset(ExperimentConfig(), args.preset)
    if args.lr_sweep:
        cfg.lr_sweep = tuple(float(x) for x in args.lr_sweep.split(","))
    keys = ["result_name", "make_mds", *types.keys()]
    for key in keys:
        value = getattr(args, key)
        if value not in (None, ""):
            setattr(cfg, key, value)
    return cfg
