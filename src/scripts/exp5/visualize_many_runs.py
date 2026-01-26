#!/usr/bin/env python3
"""
Visualize aggregated results from multiple runs (many runs analysis).

This creates plots showing mean ± std of weight trajectories across
multiple runs with the same configuration but different seeds.

Usage:
    # Visualize from multiple run directories
    python -m src.scripts.exp5.visualize_many_runs \
        --run_dirs results/run1 results/run2 results/run3 \
        --output_dir results/many_runs_analysis

    # Or from a pattern
    python -m src.scripts.exp5.visualize_many_runs \
        --pattern "src/scripts/exp5/exp_results/setup1_seed*" \
        --output_dir src/scripts/exp5/exp_results/many_runs_analysis
"""

import sys
import argparse
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datamodelopt.visualization import ManyRunsVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Visualize aggregated results from multiple runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From explicit run directories
    python -m src.scripts.exp5.visualize_many_runs \\
        --run_dirs results/run1 results/run2 results/run3

    # From a glob pattern
    python -m src.scripts.exp5.visualize_many_runs \\
        --pattern "results/setup1_seed*"

    # With custom output directory
    python -m src.scripts.exp5.visualize_many_runs \\
        --run_dirs results/run1 results/run2 \\
        --output_dir results/analysis

    # Specify percentiles
    python -m src.scripts.exp5.visualize_many_runs \\
        --run_dirs results/run1 results/run2 \\
        --percentiles 0 25 50 75 100
        """
    )
    
    parser.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        default=None,
        help="List of run directories to aggregate",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern to find run directories (e.g., 'results/run_*')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output plots (default: first run_dir)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=None,
        help="Stage names to include (default: inferred from config)",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[0, 20, 40, 60, 80],
        help="Percentiles of weights to plot (default: 0 20 40 60 80)",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Generate only combined plot",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Generate only comparison plot (normalized vs raw)",
    )
    
    args = parser.parse_args()
    
    # Get run directories
    if args.run_dirs:
        run_dirs = args.run_dirs
    elif args.pattern:
        run_dirs = sorted(glob.glob(args.pattern))
        if not run_dirs:
            print(f"No directories found matching pattern: {args.pattern}")
            return
    else:
        parser.error("Must specify either --run_dirs or --pattern")
    
    print(f"Found {len(run_dirs)} run directories:")
    for d in run_dirs:
        print(f"  - {d}")
    
    if len(run_dirs) < 2:
        print("\nWarning: Only 1 run directory found. Many runs analysis works best with multiple runs.")
    
    # Create visualizer
    viz = ManyRunsVisualizer(
        run_dirs=run_dirs,
        stages=args.stages,
        output_dir=args.output_dir,
    )
    
    # Generate plots
    if args.combined_only:
        viz.plot_weight_trajectories_combined(percentiles=args.percentiles)
    elif args.comparison_only:
        viz.plot_weight_trajectories_comparison(percentiles=args.percentiles)
    else:
        viz.plot_all()


if __name__ == "__main__":
    main()
