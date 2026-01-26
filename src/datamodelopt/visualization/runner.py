"""
Visualization runner for experiment results.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from .plots import (
    plot_loss_curves,
    plot_accuracy_curves,
    plot_loss_and_accuracy,
    plot_weight_trajectory_2d,
    plot_gradient_norms,
    plot_combined_stages,
    load_metrics_json,
    load_weights_pt,
    # Many runs
    plot_weight_trajectories_percentiles,
    plot_weight_trajectories_combined,
    plot_weight_trajectories_both_normalizations,
)


class VisualizationRunner:
    """
    Runner for generating visualizations from experiment results.
    
    Usage:
        runner = VisualizationRunner("src/scripts/exp5/exp_results/setup1")
        runner.plot_all()
    """
    
    def __init__(self, run_dir: str):
        """
        Initialize visualization runner.
        
        Args:
            run_dir: Path to the experiment results directory.
        """
        self.run_dir = Path(run_dir)
        self.config: Optional[Dict[str, Any]] = None
        self.stages: List[str] = []
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load experiment config to get stage names."""
        config_path = self.run_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
            # Extract stage names
            self.stages = [s["name"] for s in self.config.get("stages", [])]
        else:
            # Try to infer stages from metrics files
            for f in self.run_dir.glob("metrics_*.json"):
                stage_name = f.stem.replace("metrics_", "")
                self.stages.append(stage_name)
    
    def plot_stage_metrics(self, stage_name: str) -> None:
        """
        Generate plots for a single stage.
        
        Args:
            stage_name: Name of the stage.
        """
        metrics_path = self.run_dir / f"metrics_{stage_name}.json"
        if not metrics_path.exists():
            print(f"No metrics file found for stage: {stage_name}")
            return
        
        metrics = load_metrics_json(str(metrics_path))
        
        # Loss and accuracy combined
        plot_loss_and_accuracy(
            metrics,
            title=f"Stage: {stage_name}",
            save_path=str(self.run_dir / f"plot_loss_acc_{stage_name}.png"),
        )
        
        # Gradient norms
        if metrics.get("grad_norms"):
            plot_gradient_norms(
                metrics,
                title=f"Gradient Norms: {stage_name}",
                save_path=str(self.run_dir / f"plot_grad_norm_{stage_name}.png"),
            )
    
    def plot_weight_trajectory(self, stage_name: str) -> None:
        """
        Plot weight trajectory for a stage.
        
        Args:
            stage_name: Name of the stage.
        """
        weights_path = self.run_dir / f"weights_{stage_name}.pt"
        if not weights_path.exists():
            print(f"No weights file found for stage: {stage_name}")
            return
        
        weights_data = load_weights_pt(str(weights_path))
        
        plot_weight_trajectory_2d(
            weights_data,
            title=f"Weight Trajectory: {stage_name}",
            save_path=str(self.run_dir / f"plot_weights_{stage_name}.png"),
        )
    
    def plot_combined(self) -> None:
        """Plot combined metrics from all stages."""
        if not self.stages:
            print("No stages found to plot")
            return
        
        plot_combined_stages(
            str(self.run_dir),
            self.stages,
            save_path=str(self.run_dir / "combined_plots.png"),
        )
    
    def plot_all(self) -> None:
        """Generate all available plots."""
        print(f"Generating plots for: {self.run_dir}")
        print(f"Stages: {self.stages}")
        
        # Per-stage plots
        for stage_name in self.stages:
            print(f"\n--- Stage: {stage_name} ---")
            self.plot_stage_metrics(stage_name)
            self.plot_weight_trajectory(stage_name)
        
        # Combined plot
        print(f"\n--- Combined ---")
        self.plot_combined()
        
        print(f"\nDone! Plots saved to: {self.run_dir}")
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the experiment results.
        
        Returns:
            Dictionary with summary statistics.
        """
        summary = {
            "run_dir": str(self.run_dir),
            "stages": self.stages,
            "stage_metrics": {},
        }
        
        for stage_name in self.stages:
            metrics_path = self.run_dir / f"metrics_{stage_name}.json"
            if metrics_path.exists():
                metrics = load_metrics_json(str(metrics_path))
                
                losses = metrics.get("losses", [])
                val_losses = metrics.get("val_losses", [])
                accuracies = metrics.get("accuracies", [])
                val_accs = metrics.get("val_accuracies", [])
                
                summary["stage_metrics"][stage_name] = {
                    "n_steps": len(losses),
                    "final_train_loss": losses[-1] if losses else None,
                    "final_val_loss": val_losses[-1] if val_losses else None,
                    "final_train_acc": accuracies[-1] if accuracies else None,
                    "final_val_acc": val_accs[-1] if val_accs else None,
                    "min_train_loss": min(losses) if losses else None,
                    "min_val_loss": min(val_losses) if val_losses else None,
                    "max_train_acc": max(accuracies) if accuracies else None,
                    "max_val_acc": max(val_accs) if val_accs else None,
                }
        
        return summary
    
    def print_summary(self) -> None:
        """Print a summary of the experiment results."""
        summary = self.summary()
        
        print(f"\n{'='*60}")
        print(f"Experiment Summary: {summary['run_dir']}")
        print(f"{'='*60}")
        
        for stage_name, metrics in summary["stage_metrics"].items():
            print(f"\n--- {stage_name} ---")
            print(f"  Steps: {metrics['n_steps']}")
            if metrics['final_train_loss'] is not None:
                print(f"  Final Train Loss: {metrics['final_train_loss']:.4f}")
            if metrics['final_val_loss'] is not None:
                print(f"  Final Val Loss: {metrics['final_val_loss']:.4f}")
            if metrics['final_train_acc'] is not None:
                print(f"  Final Train Acc: {metrics['final_train_acc']:.4f}")
            if metrics['final_val_acc'] is not None:
                print(f"  Final Val Acc: {metrics['final_val_acc']:.4f}")


class ManyRunsVisualizer:
    """
    Visualizer for aggregating and plotting results from multiple runs.
    
    This is useful for showing mean ± std trajectories across many runs
    with the same configuration but different seeds.
    
    Usage:
        run_dirs = ["results/run1", "results/run2", "results/run3"]
        viz = ManyRunsVisualizer(run_dirs, stages=["stage1_sgd", "stage2_gd", "stage3_sgd"])
        viz.plot_weight_trajectories()
    """
    
    def __init__(
        self,
        run_dirs: List[str],
        stages: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize many runs visualizer.
        
        Args:
            run_dirs: List of paths to run directories.
            stages: List of stage names. If None, inferred from first run.
            output_dir: Directory for saving plots. If None, uses first run dir.
        """
        self.run_dirs = [Path(d) for d in run_dirs]
        self.output_dir = Path(output_dir) if output_dir else self.run_dirs[0]
        
        # Infer stages from first run if not provided
        if stages is None:
            config_path = self.run_dirs[0] / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                self.stages = [s["name"] for s in config.get("stages", [])]
            else:
                self.stages = []
        else:
            self.stages = stages
        
        # Try to get batch sizes from config
        self.stage_batch_sizes: Dict[str, int] = {}
        self.train_dataset_size: Optional[int] = None
        self._load_batch_info()
    
    def _load_batch_info(self) -> None:
        """Load batch size information from config."""
        config_path = self.run_dirs[0] / "config.json"
        if not config_path.exists():
            return
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Get default batch size from data config
        default_batch_size = config.get("data", {}).get("kwargs", {}).get("batch_size", 32)
        
        # Get train dataset size
        data_kwargs = config.get("data", {}).get("kwargs", {})
        self.train_dataset_size = data_kwargs.get("sample_size") or data_kwargs.get("train_size")
        
        # Get per-stage batch sizes
        for stage in config.get("stages", []):
            stage_name = stage["name"]
            batch_size = stage.get("batch_size") or default_batch_size
            
            if stage.get("dataloader_mode") == "fullbatch":
                # Use actual dataset size for fullbatch
                self.stage_batch_sizes[stage_name] = self.train_dataset_size or -1
            else:
                self.stage_batch_sizes[stage_name] = batch_size
    
    def plot_weight_trajectories(
        self,
        percentiles: List[float] = None,
        normalize_x: bool = False,
    ) -> None:
        """
        Plot weight trajectories for selected percentile weights.
        
        Creates 5 individual plots (one per percentile) showing mean ± std.
        
        Args:
            percentiles: Percentiles to plot (default: [0, 20, 40, 60, 80]).
            normalize_x: If True, normalize X axis by batch size.
        """
        if percentiles is None:
            percentiles = [0, 20, 40, 60, 80]
        
        plot_weight_trajectories_percentiles(
            run_dirs=[str(d) for d in self.run_dirs],
            stages=self.stages,
            percentiles=percentiles,
            save_dir=str(self.output_dir),
            normalize_x=normalize_x,
            stage_batch_sizes=self.stage_batch_sizes if normalize_x else None,
            train_dataset_size=self.train_dataset_size,
            title_prefix=f"[{len(self.run_dirs)} runs] ",
        )
    
    def plot_weight_trajectories_combined(
        self,
        percentiles: List[float] = None,
    ) -> None:
        """
        Plot all percentile weight trajectories in a single figure.
        
        Args:
            percentiles: Percentiles to plot (default: [0, 20, 40, 60, 80]).
        """
        if percentiles is None:
            percentiles = [0, 20, 40, 60, 80]
        
        plot_weight_trajectories_combined(
            run_dirs=[str(d) for d in self.run_dirs],
            stages=self.stages,
            percentiles=percentiles,
            save_path=str(self.output_dir / "weight_trajectories_combined.png"),
            stage_batch_sizes=self.stage_batch_sizes,
            train_dataset_size=self.train_dataset_size,
            title=f"Weight Trajectories (mean ± std, {len(self.run_dirs)} runs)",
        )
    
    def plot_weight_trajectories_comparison(
        self,
        percentiles: List[float] = None,
    ) -> None:
        """
        Plot weight trajectories with both normalized and non-normalized X axes.
        
        Creates a side-by-side comparison plot.
        
        Args:
            percentiles: Percentiles to plot (default: [0, 20, 40, 60, 80]).
        """
        if percentiles is None:
            percentiles = [0, 20, 40, 60, 80]
        
        plot_weight_trajectories_both_normalizations(
            run_dirs=[str(d) for d in self.run_dirs],
            stages=self.stages,
            percentiles=percentiles,
            save_dir=str(self.output_dir),
            stage_batch_sizes=self.stage_batch_sizes,
            train_dataset_size=self.train_dataset_size,
            title=f"Weight Trajectories Comparison ({len(self.run_dirs)} runs)",
        )
    
    def plot_all(self) -> None:
        """Generate all many-runs plots."""
        print(f"Generating many-runs plots from {len(self.run_dirs)} runs")
        print(f"Stages: {self.stages}")
        print(f"Output: {self.output_dir}")
        
        print("\n--- Weight trajectories (individual percentiles) ---")
        self.plot_weight_trajectories(normalize_x=False)
        
        print("\n--- Weight trajectories (normalized X) ---")
        self.plot_weight_trajectories(normalize_x=True)
        
        print("\n--- Combined weight trajectories ---")
        self.plot_weight_trajectories_combined()
        
        print("\n--- Comparison plot ---")
        self.plot_weight_trajectories_comparison()
        
        print(f"\nDone! Plots saved to: {self.output_dir}")
