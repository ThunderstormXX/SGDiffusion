#!/usr/bin/env python3
"""
Visualize experiment results.

Usage:
    python -m src.scripts.exp5.visualize --run_dir src/scripts/exp5/exp_results/setup1
    python -m src.scripts.exp5.visualize --run_dir src/scripts/exp5/exp_results/setup1 --summary
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datamodelopt.visualization import VisualizationRunner, ManyRunsVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Visualize experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all plots (including weight trajectory percentiles)
    python -m src.scripts.exp5.visualize --run_dir src/scripts/exp5/exp_results/setup1

    # Print summary only
    python -m src.scripts.exp5.visualize --run_dir src/scripts/exp5/exp_results/setup1 --summary

    # Plot specific stage
    python -m src.scripts.exp5.visualize --run_dir src/scripts/exp5/exp_results/setup1 --stage stage1_sgd
    
    # Skip weight percentile plots
    python -m src.scripts.exp5.visualize --run_dir src/scripts/exp5/exp_results/setup1 --no-percentiles
        """
    )
    
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to experiment results directory",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary only, no plots",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Plot only this stage",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Generate only combined plots",
    )
    parser.add_argument(
        "--no-percentiles",
        action="store_true",
        help="Skip weight percentile trajectory plots",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[0, 20, 40, 60, 80],
        help="Percentiles for weight trajectory plots (default: 0 20 40 60 80)",
    )
    
    args = parser.parse_args()
    
    runner = VisualizationRunner(args.run_dir)
    
    if args.summary:
        runner.print_summary()
    elif args.combined_only:
        runner.plot_combined()
    elif args.stage:
        runner.plot_stage_metrics(args.stage)
        runner.plot_weight_trajectory(args.stage)
    else:
        runner.plot_all()
        
        # Also generate weight percentile trajectory plots (single run version)
        if not args.no_percentiles and runner.stages:
            print("\n--- Weight Percentile Trajectories ---")
            try:
                # Use ManyRunsVisualizer with single run for consistent visualization
                viz = ManyRunsVisualizer(
                    run_dirs=[args.run_dir],
                    stages=runner.stages,
                    output_dir=args.run_dir,
                )
                viz.plot_weight_trajectories_combined(percentiles=args.percentiles)
                viz.plot_weight_trajectories_comparison(percentiles=args.percentiles)
            except Exception as e:
                print(f"Warning: Could not generate weight percentile plots: {e}")


if __name__ == "__main__":
    main()
