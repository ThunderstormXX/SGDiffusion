#!/usr/bin/env python3
"""
Prepare many-SGD experiments for visualization.

For many-SGD experiments, stages 1-2 are run once (seed 100),
and stage 3 is run many times (seeds 200+). For visualization,
we need each stage 3 run to have the same stages 1-2 data.

This script copies stages 1-2 weight files from the single run
to all stage 3 run directories.

Usage:
    python -m src.scripts.exp5.prepare_manysgd_viz \
        --source_seed 100 \
        --target_seeds 200 201 202 203 204 \
        --exp_dir src/scripts/exp5/exp_results/mnist_manysgd_small
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare many-SGD experiments for visualization")
    parser.add_argument("--source_seed", type=int, required=True,
                        help="Seed of the single run (stages 1-2)")
    parser.add_argument("--target_seeds", type=int, nargs="+", required=True,
                        help="Seeds of the many runs (stage 3)")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory")

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    source_dir = exp_dir / f"run_seed{args.source_seed}"

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return

    # Files to copy from stages 1-2
    files_to_copy = [
        "weights_stage1_sgd.pt",
        "weights_stage1_sgd_info.txt",
        "weights_stage2_gd.pt",
        "weights_stage2_gd_info.txt",
    ]

    print("=" * 60)
    print("PREPARING MANY-SGD VISUALIZATION")
    print("=" * 60)
    print(f"Source:  run_seed{args.source_seed}")
    print(f"Targets: {len(args.target_seeds)} runs")
    print(f"Exp dir: {exp_dir}")
    print("=" * 60)

    copied_count = 0
    skipped_count = 0

    for target_seed in args.target_seeds:
        target_dir = exp_dir / f"run_seed{target_seed}"

        if not target_dir.exists():
            print(f"  [SKIP] Target directory not found: run_seed{target_seed}")
            skipped_count += 1
            continue

        for filename in files_to_copy:
            source_file = source_dir / filename
            target_file = target_dir / filename

            if not source_file.exists():
                # File might not exist (e.g., if weights tracking was disabled)
                continue

            if target_file.exists():
                # Already exists, skip
                continue

            # Copy file
            shutil.copy2(source_file, target_file)
            copied_count += 1

    print()
    print("=" * 60)
    print(f"COMPLETE: Copied {copied_count} files")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} runs (directories not found)")
    print("=" * 60)
    print()
    print("Now you can run visualization:")
    print("  cd src/scripts/exp5/bash")
    print("  ./run_*_manysgd_*.sh 1")


if __name__ == "__main__":
    main()
