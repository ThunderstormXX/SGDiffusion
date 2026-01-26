#!/usr/bin/env python3
"""
Run experiment pipeline.

Usage:
    python -m src.scripts.exp5.run_pipeline --preset setup1 --device cpu
    python -m src.scripts.exp5.run_pipeline --config path/to/config.json --device cuda
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datamodelopt.experiments.pipeline import ExperimentRunner
from src.datamodelopt.core.config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run preset setup1 (SGD -> GD -> SGD tunnel)
    python -m src.scripts.exp5.run_pipeline --preset setup1 --device cpu

    # Run from config file
    python -m src.scripts.exp5.run_pipeline --config src/scripts/exp5/configs/setup1_sgd_gd_sgd.json

    # Override run directory and seed
    python -m src.scripts.exp5.run_pipeline --preset setup1 --run_dir ./my_results --seed 123
        """
    )
    
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory to save results (default: from config or preset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config or 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, cuda, mps (default: from config or cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Data type: float32, float64, bfloat16 (default: from config or float32)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=[
            "mnist_small", "mnist_medium", "mnist_large",
            "shakespeare_small", "shakespeare_medium", "shakespeare_large",
        ],
        help="Use a predefined preset configuration",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    
    args = parser.parse_args()
    
    # Must specify either config or preset
    if args.config is None and args.preset is None:
        parser.error("Must specify either --config or --preset")
    
    # Load config
    if args.config is not None:
        print(f"Loading config from: {args.config}")
        config = ExperimentConfig.load_json(args.config)
    else:
        print(f"Loading preset: {args.preset}")
        from src.scripts.exp5.presets import get_preset
        config = get_preset(args.preset)
    
    # Override config with command line arguments
    if args.run_dir is not None:
        config.run_dir = args.run_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.dtype is not None:
        config.dtype = args.dtype
    
    # Create and run experiment
    runner = ExperimentRunner(config)
    runner.run(progress=not args.no_progress)


if __name__ == "__main__":
    main()
