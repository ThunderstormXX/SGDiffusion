#!/usr/bin/env python3
"""
Run a single tunnel from JSON config.

Usage:
    python -m src.scripts.exp5.run_tunnel \
        --config configs/mnist_manysgd/small.json \
        --tunnel 0 \
        --n_runs 1 \
        --seeds 0
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.datamodelopt.core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrackerConfig,
    TunnelConfig,
)
from src.datamodelopt.experiments.tunnel import TunnelRunner


def load_config_from_json(config_path: Path) -> tuple[ExperimentConfig, dict]:
    """Load ExperimentConfig from JSON file.
    
    Returns:
        Tuple of (ExperimentConfig, raw_data dict for extracting n_runs, n_initial_weights)
    """
    with open(config_path) as f:
        data = json.load(f)

    # Build data config
    data_cfg = data["data"]
    data_config = DataConfig(
        name=data_cfg["name"],
        kwargs={
            "sample_size": data_cfg.get("sample_size", 6400),
            "batch_size": 64,  # Default, will be overridden per tunnel
            "replacement": data_cfg.get("replacement", True),
        }
    )

    # Build model config - pass all fields except "name" as kwargs
    model_cfg = data["model"]
    model_kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    model_config = ModelConfig(
        name=model_cfg["name"],
        kwargs=model_kwargs
    )

    # Build tunnel configs
    tunnel_configs = []
    for i, t in enumerate(data["tunnels"]):
        # Build trackers
        trackers = []
        if "trackers" in t:
            for tracker_name, tracker_kwargs in t["trackers"].items():
                trackers.append(TrackerConfig(name=tracker_name, kwargs=tracker_kwargs))

        # Build optimizer config
        optimizer = OptimizerConfig(
            name=t.get("optimizer", "sgd"),
            kwargs={"lr": t.get("lr", 0.1)}
        )

        tunnel_config = TunnelConfig(
            tunnel_index=i,
            description=t.get("description", f"Tunnel {i}"),
            mode=t.get("mode", "steps"),
            steps=t.get("steps"),
            epochs=t.get("epochs"),
            optimizer=optimizer,
            dataloader_mode=t.get("dataloader_mode", "minibatch"),
            batch_size=t.get("batch_size"),
            trackers=trackers,
            eval_every=t.get("eval_every", 100),
            save_checkpoint=t.get("save_checkpoint", True),
            source_mode=t.get("source_mode"),
            source_run_index=t.get("source_run_index", 0),
            n_initial_weights=t.get("n_initial_weights", 1),
        )
        tunnel_configs.append(tunnel_config)

    # Build experiment config
    experiment_config = ExperimentConfig(
        run_dir=f"src/scripts/exp5/exp_results/{data['experiment_name']}",
        seed=data.get("seed", 42),
        device=data.get("device", "cpu"),
        dtype=data.get("dtype", "float32"),
        data=data_config,
        model=model_config,
        tunnels=tunnel_configs,
    )

    return experiment_config, data


def count_runs_in_tunnel(experiment_dir: Path, tunnel_index: int) -> int:
    """Count number of runs in a completed tunnel."""
    tunnel_dir = experiment_dir / f"tunnel_{tunnel_index}"
    if not tunnel_dir.exists():
        return 0
    return len(list(tunnel_dir.glob("run_*")))


def main():
    parser = argparse.ArgumentParser(
        description="Run a single tunnel from JSON config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config file")
    parser.add_argument("--tunnel", type=int, required=True,
                        help="Tunnel index (0, 1, 2, ...)")
    parser.add_argument("--n_runs", type=int, default=None,
                        help="Number of runs per source point (default: from config)")
    parser.add_argument("--n_initial_weights", type=int, default=None,
                        help="Number of initial weights for first tunnel (default: from config)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Random seeds (default: auto 0..total_runs-1)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: from config)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Data type")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable progress bars")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode: show tqdm for each step")

    args = parser.parse_args()

    # Find config file
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        config_path = script_dir / args.config
        if not config_path.exists():
            # Try relative to workspace
            config_path = PROJECT_ROOT / args.config

    if not config_path.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    experiment_config, raw_config = load_config_from_json(config_path)

    # Check tunnel index
    if args.tunnel >= len(experiment_config.tunnels):
        print(f"ERROR: Tunnel {args.tunnel} does not exist (only {len(experiment_config.tunnels)} tunnels)")
        sys.exit(1)

    tunnel_config = experiment_config.tunnels[args.tunnel]
    tunnel_raw = raw_config["tunnels"][args.tunnel]

    # Get n_runs and n_initial_weights from config JSON or CLI
    config_n_runs = tunnel_raw.get("n_runs", 1)
    config_n_initial_weights = tunnel_raw.get("n_initial_weights", 1)
    
    n_runs = args.n_runs if args.n_runs is not None else config_n_runs
    n_initial_weights = args.n_initial_weights if args.n_initial_weights is not None else config_n_initial_weights

    # Override device
    if args.device:
        experiment_config.device = args.device
    experiment_config.dtype = args.dtype

    # Output directory
    if args.output_dir:
        experiment_dir = Path(args.output_dir)
    else:
        experiment_dir = Path(experiment_config.run_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Compute total runs based on tunnel type
    is_first_tunnel = tunnel_config.source_mode is None
    
    if is_first_tunnel:
        # First tunnel: total = n_initial_weights × n_runs
        total_runs = n_initial_weights * n_runs
        print(f"First tunnel: {n_initial_weights} initial weights × {n_runs} runs = {total_runs} total runs")
    elif tunnel_config.source_mode == "cartesian":
        # Cartesian mode: total = n_source_runs × n_runs
        n_source_runs = count_runs_in_tunnel(experiment_dir, args.tunnel - 1)
        if n_source_runs == 0:
            print(f"ERROR: No runs found in previous tunnel (tunnel_{args.tunnel - 1})")
            print(f"       Run previous tunnel first!")
            sys.exit(1)
        total_runs = n_source_runs * n_runs
        print(f"Cartesian mode: {n_source_runs} source runs × {n_runs} runs = {total_runs} total runs")
    else:
        # 1to1 mode or other: total = n_runs
        total_runs = n_runs
        print(f"1to1 mode: {total_runs} total runs")

    # Generate seeds if not provided
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(total_runs))

    # Validate seeds count
    if len(seeds) != total_runs:
        print(f"ERROR: Number of seeds ({len(seeds)}) must match total_runs ({total_runs})")
        sys.exit(1)

    # Get device and dtype
    device = torch.device(experiment_config.device)
    dtype = getattr(torch, experiment_config.dtype)

    # Create tunnel runner
    tunnel_runner = TunnelRunner(
        experiment_config=experiment_config,
        tunnel_config=tunnel_config,
        experiment_dir=experiment_dir,
    )

    # Run tunnel
    tunnel_runner.run(
        n_runs=n_runs,
        seeds=seeds,
        device=device,
        dtype=dtype,
        verbose=not args.quiet,
        debug=args.debug,
        n_initial_weights=n_initial_weights,
    )

    print()
    print("✅ Tunnel complete!")
    print(f"   Results: {tunnel_runner.tunnel_dir}")


if __name__ == "__main__":
    main()
