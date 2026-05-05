#!/usr/bin/env python3
"""
Visualize tunnel-based experiment results.

Creates plots for:
1. Weight trajectories across all tunnels
2. For 1 run: single line
3. For N runs: mean ± std OR individual trajectories (tree view)
4. Hessian eigenvalue spectrum

Supports cartesian product (1×1×N) - shows diverging tree of trajectories.

Usage:
    python -m src.scripts.exp5.visualize_tunnel \
        --exp_dir src/scripts/exp5/exp_results/mnist_manysgd_small_v2 \
        --output_dir src/scripts/exp5/exp_results/mnist_manysgd_small_v2/aggregated
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import torch


class TunnelVisualizer:
    """
    Visualizes tunnel-based experiment results.

    Features:
    - Weight trajectories: single line (1 run) or mean±std (N runs)
    - Tree view: diverging trajectories for cartesian product
    - Normalized X-axis (step × batch_size)
    - Hessian eigenvalue spectrum
    """

    def __init__(self, exp_dir: Path, output_dir: Path):
        if not HAS_MPL:
            raise ImportError("matplotlib is required for visualization")

        self.exp_dir = Path(exp_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Discover tunnels
        self.tunnels = self._discover_tunnels()

        # Load batch sizes from metadata
        self._load_batch_sizes()

    def _discover_tunnels(self) -> list[dict[str, Any]]:
        """Discover all tunnels and their data, including run grouping info."""
        tunnels = []

        for tunnel_dir in sorted(self.exp_dir.glob("tunnel_*")):
            if not tunnel_dir.is_dir():
                continue

            # Load metadata
            metadata_path = tunnel_dir / "tunnel_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Count runs and load run metadata for grouping
            runs = sorted(tunnel_dir.glob("run_*"), key=lambda x: int(x.name.split("_")[1]))
            
            # Discover grouping from run metadata
            grouping_info = self._discover_run_grouping(runs)

            tunnels.append({
                "name": tunnel_dir.name,
                "path": tunnel_dir,
                "n_runs": len(runs),
                "metadata": metadata,
                "runs": runs,
                "grouping": grouping_info,
            })

        return tunnels
    
    def _discover_run_grouping(self, runs: list[Path]) -> dict[str, Any]:
        """
        Discover run grouping from run metadata files.
        
        Returns dict with:
        - groups: dict mapping group_idx -> list of run indices in that group
        - n_groups: number of groups
        - n_runs_per_group: runs per group (if uniform)
        - grouping_type: "initial_weights" | "source_runs" | "none"
        """
        if not runs:
            return {"groups": {0: []}, "n_groups": 1, "n_runs_per_group": 0, "grouping_type": "none"}
        
        groups: dict[int, list[int]] = {}
        grouping_type = "none"
        
        for run_dir in runs:
            run_idx = int(run_dir.name.split("_")[1])
            metadata_path = run_dir / "run_metadata.json"
            
            group_idx = 0  # Default: all in one group
            
            if metadata_path.exists():
                with open(metadata_path) as f:
                    meta = json.load(f)
                
                # Check for grouping info
                if "grouping" in meta:
                    # First tunnel with n_initial_weights
                    group_idx = meta["grouping"].get("initial_weight_index", 0)
                    grouping_type = "initial_weights"
                elif "source_run_index" in meta and meta.get("source_mode") == "cartesian":
                    # Subsequent tunnel with cartesian mode
                    group_idx = meta["source_run_index"]
                    grouping_type = "source_runs"
            
            if group_idx not in groups:
                groups[group_idx] = []
            groups[group_idx].append(run_idx)
        
        # Sort runs within each group
        for g in groups:
            groups[g].sort()
        
        n_groups = len(groups)
        runs_per_group = [len(groups[g]) for g in sorted(groups.keys())]
        n_runs_per_group = runs_per_group[0] if runs_per_group and len(set(runs_per_group)) == 1 else None
        
        return {
            "groups": groups,
            "n_groups": n_groups,
            "n_runs_per_group": n_runs_per_group,
            "grouping_type": grouping_type,
        }

    def _load_batch_sizes(self) -> None:
        """Load batch sizes for each tunnel from metadata."""
        self.batch_sizes = {}
        self.train_size = 6400  # Default

        for tunnel in self.tunnels:
            config = tunnel["metadata"].get("configuration", {})
            dataloader_mode = config.get("dataloader_mode", "minibatch")

            if dataloader_mode == "fullbatch":
                self.batch_sizes[tunnel["name"]] = self.train_size
            else:
                # Try to get from experiment config
                exp_config = tunnel["metadata"].get("experiment_config", {})
                data_kwargs = exp_config.get("data", {}).get("kwargs", {})
                self.batch_sizes[tunnel["name"]] = data_kwargs.get("batch_size", 64)

    def plot_all(self, percentiles: list[int] = None) -> None:
        """Generate all plots."""
        if percentiles is None:
            percentiles = [0, 20, 40, 60, 80]
        print("=" * 70)
        print("VISUALIZING TUNNEL DATA")
        print("=" * 70)
        print(f"Experiment: {self.exp_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Tunnels: {len(self.tunnels)}")
        for t in self.tunnels:
            print(f"  - {t['name']}: {t['n_runs']} runs, batch_size={self.batch_sizes.get(t['name'], 64)}")
        print("=" * 70)

        # Load all weight data
        weights_data = self._load_all_weights()

        if weights_data is not None:
            # Plot weight trajectories for each percentile
            for p in percentiles:
                # Mean±std view
                self._plot_weight_trajectory_meanstd(weights_data, p, normalize_x=False)
                self._plot_weight_trajectory_meanstd(weights_data, p, normalize_x=True)

            # Tree view (individual trajectories)
            for p in percentiles:
                self._plot_weight_trajectory_tree(weights_data, p, normalize_x=False)
                self._plot_weight_trajectory_tree(weights_data, p, normalize_x=True)

        # Load and plot metrics (loss, accuracy)
        metrics_data = self._load_all_metrics()

        if metrics_data is not None:
            # Loss plots
            self._plot_metric_meanstd(metrics_data, "loss", normalize_x=False)
            self._plot_metric_meanstd(metrics_data, "loss", normalize_x=True)
            self._plot_metric_tree(metrics_data, "loss", normalize_x=False)
            self._plot_metric_tree(metrics_data, "loss", normalize_x=True)

            # Accuracy plots
            self._plot_metric_meanstd(metrics_data, "accuracy", normalize_x=False)
            self._plot_metric_meanstd(metrics_data, "accuracy", normalize_x=True)
            self._plot_metric_tree(metrics_data, "accuracy", normalize_x=False)
            self._plot_metric_tree(metrics_data, "accuracy", normalize_x=True)

            # Full-train eval plots (if available)
            has_train_full = any(
                t.get("train_full_losses") is not None and len(t.get("train_full_losses")) > 0
                for t in metrics_data["tunnels"]
            )
            if has_train_full:
                self._plot_metric_meanstd(metrics_data, "train_full_loss", normalize_x=False)
                self._plot_metric_meanstd(metrics_data, "train_full_loss", normalize_x=True)
                self._plot_metric_tree(metrics_data, "train_full_loss", normalize_x=False)
                self._plot_metric_tree(metrics_data, "train_full_loss", normalize_x=True)

                self._plot_metric_meanstd(metrics_data, "train_full_accuracy", normalize_x=False)
                self._plot_metric_meanstd(metrics_data, "train_full_accuracy", normalize_x=True)
                self._plot_metric_tree(metrics_data, "train_full_accuracy", normalize_x=False)
                self._plot_metric_tree(metrics_data, "train_full_accuracy", normalize_x=True)

        # Hessian spectrum
        self._plot_hessian_spectrum()

        print("\n" + "=" * 70)
        print("✅ VISUALIZATION COMPLETE")
        print("=" * 70)

    def _load_all_weights(self) -> dict[str, Any] | None:
        """Load weight data from all tunnels, preserving run order for grouping."""
        all_data = []

        # First pass: load all weights and compute step ranges
        cumulative_step = 0
        cumulative_normalized = 0

        for tunnel in self.tunnels:
            tunnel_name = tunnel["name"]
            n_runs = tunnel["n_runs"]
            batch_size = self.batch_sizes.get(tunnel_name, 64)
            grouping = tunnel.get("grouping", {"groups": {0: list(range(n_runs))}, "n_groups": 1})

            # Load weights from each run in order
            run_weights = []
            run_indices = []  # Track which run each weight belongs to
            run_steps_local = None  # Local steps within tunnel (0, 1, 2, ...)

            for run_dir in tunnel["runs"]:
                run_idx = int(run_dir.name.split("_")[1])
                
                # Try different weight file naming patterns
                weights_path = run_dir / f"weights_{tunnel_name}.pt"
                if not weights_path.exists():
                    weights_path = run_dir / "weights.pt"
                if not weights_path.exists():
                    continue

                data = torch.load(weights_path, map_location="cpu")
                if isinstance(data, dict) and "weights" in data:
                    weights = data["weights"].numpy()
                    steps = data.get("steps", list(range(weights.shape[0])))
                else:
                    weights = data.numpy() if isinstance(data, torch.Tensor) else data
                    steps = list(range(weights.shape[0]))

                run_weights.append(weights)
                run_indices.append(run_idx)
                if run_steps_local is None:
                    run_steps_local = np.array(steps)

            if not run_weights:
                continue

            # Stack runs: [n_runs, n_steps, n_params]
            try:
                stacked = np.stack(run_weights)
            except ValueError:
                # Pad if different lengths
                max_len = max(w.shape[0] for w in run_weights)
                padded = []
                for w in run_weights:
                    if w.shape[0] < max_len:
                        pad = np.zeros((max_len - w.shape[0], w.shape[1]))
                        w = np.vstack([w, pad])
                    padded.append(w)
                stacked = np.stack(padded)
                if run_steps_local is not None and len(run_steps_local) < max_len:
                    run_steps_local = np.arange(max_len)

            n_steps = stacked.shape[1]

            # Compute global steps (offset by cumulative)
            global_steps = run_steps_local + cumulative_step
            global_steps_normalized = run_steps_local * batch_size + cumulative_normalized

            # Store boundary BEFORE this tunnel starts
            start_boundary = cumulative_step
            start_boundary_normalized = cumulative_normalized

            all_data.append({
                "tunnel_name": tunnel_name,
                "n_runs": n_runs,
                "weights": stacked,
                "run_indices": run_indices,  # Which run each weight row belongs to
                "steps": global_steps,
                "steps_normalized": global_steps_normalized,
                "batch_size": batch_size,
                "start_boundary": start_boundary,
                "start_boundary_normalized": start_boundary_normalized,
                "grouping": grouping,  # Grouping info
            })

            # Update cumulative for next tunnel
            cumulative_step += n_steps
            cumulative_normalized += n_steps * batch_size

        if not all_data:
            print("No weight data found")
            return None

        return {
            "tunnels": all_data,
            "n_params": all_data[0]["weights"].shape[2],
            "total_steps": cumulative_step,
            "total_normalized": cumulative_normalized,
        }

    def _plot_weight_trajectory_meanstd(
        self,
        data: dict[str, Any],
        percentile: int,
        normalize_x: bool = False,
    ) -> None:
        """
        Plot weight trajectory with mean ± std, respecting run grouping.

        - 1 run: single solid line (no std)
        - N runs with 1 group: single mean ± std tube
        - N runs with M groups: M separate mean ± std tubes
          (each tube averages runs within that group)
        """
        n_params = data["n_params"]
        param_idx = int(n_params * percentile / 100)

        fig, ax = plt.subplots(figsize=(14, 6))

        # Color palette: use different hues for tunnels, different shades for groups
        tunnel_colors = plt.cm.Set1(np.linspace(0, 1, len(data["tunnels"])))

        for i, tunnel_data in enumerate(data["tunnels"]):
            weights = tunnel_data["weights"][:, :, param_idx]  # [n_loaded_runs, n_steps]
            run_indices = tunnel_data.get("run_indices", list(range(weights.shape[0])))
            n_runs = weights.shape[0]
            
            x = tunnel_data["steps_normalized"] if normalize_x else tunnel_data["steps"]
            
            grouping = tunnel_data.get("grouping", {"groups": {0: run_indices}, "n_groups": 1})
            groups = grouping["groups"]
            n_groups = grouping["n_groups"]
            
            base_color = tunnel_colors[i]
            
            if n_runs == 1:
                # Single run: just plot the line
                ax.plot(x, weights[0], color=base_color, linewidth=2,
                        label=f"{tunnel_data['tunnel_name']} (1 run)")
            elif n_groups == 1:
                # Single group: standard mean ± std
                mean = weights.mean(axis=0)
                std = weights.std(axis=0)

                ax.plot(x, mean, color=base_color, linewidth=2,
                        label=f"{tunnel_data['tunnel_name']} ({n_runs} runs)")
                ax.fill_between(x, mean - std, mean + std, color=base_color, alpha=0.3)
            else:
                # Multiple groups: separate mean ± std tube for each group
                # Create sub-colors for groups
                group_colors = self._get_group_colors(base_color, n_groups)
                
                # Build a mapping from run_idx to weight array index
                run_idx_to_array_idx = {run_idx: arr_idx for arr_idx, run_idx in enumerate(run_indices)}
                
                for group_idx, group_run_indices in sorted(groups.items()):
                    # Get weight array indices for this group
                    array_indices = [run_idx_to_array_idx[rid] for rid in group_run_indices 
                                   if rid in run_idx_to_array_idx]
                    
                    if not array_indices:
                        continue
                    
                    group_weights = weights[array_indices]  # [n_runs_in_group, n_steps]
                    group_color = group_colors[group_idx % len(group_colors)]
                    
                    if len(array_indices) == 1:
                        # Single run in group: just line
                        ax.plot(x, group_weights[0], color=group_color, linewidth=1.5,
                               alpha=0.8)
                    else:
                        # Multiple runs: mean ± std
                        mean = group_weights.mean(axis=0)
                        std = group_weights.std(axis=0)
                        
                        ax.plot(x, mean, color=group_color, linewidth=1.5, alpha=0.9)
                        ax.fill_between(x, mean - std, mean + std, color=group_color, alpha=0.2)
                
                # Single legend entry for the tunnel
                ax.plot([], [], color=base_color, linewidth=2,
                       label=f"{tunnel_data['tunnel_name']} ({n_groups} groups × {len(list(groups.values())[0])} runs)")

        # Add tunnel boundaries (vertical lines at tunnel start)
        for i, tunnel_data in enumerate(data["tunnels"]):
            bnd = tunnel_data["start_boundary_normalized"] if normalize_x else tunnel_data["start_boundary"]
            if bnd > 0:  # Don't draw at x=0
                ax.axvline(x=bnd, color='red', linestyle='--', alpha=0.5)

        xlabel = "Step × Batch Size" if normalize_x else "Step"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Weight (percentile {percentile}%)")
        title_suffix = "(normalized X)" if normalize_x else "(raw steps)"
        ax.set_title(f"Weight Trajectory - Percentile {percentile}% - Mean±Std {title_suffix}")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        suffix = "_normalized" if normalize_x else ""
        filename = f"weight_percentile_{percentile}_meanstd{suffix}.png"
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()

        print(f"  ✓ {filename}")
    
    def _get_group_colors(self, base_color, n_groups: int) -> list:
        """Generate n_groups colors as variations of base_color."""
        import matplotlib.colors as mcolors
        
        if n_groups == 1:
            return [base_color]
        
        # Convert to HSV, vary hue slightly
        hsv = mcolors.rgb_to_hsv(base_color[:3])
        colors = []
        
        for i in range(n_groups):
            # Vary saturation and value slightly to distinguish groups
            h = hsv[0]
            s = max(0.3, min(1.0, hsv[1] + (i - n_groups/2) * 0.1))
            v = max(0.3, min(1.0, hsv[2] - (i - n_groups/2) * 0.05))
            rgb = mcolors.hsv_to_rgb([h, s, v])
            colors.append(rgb)
        
        return colors

    def _plot_weight_trajectory_tree(
        self,
        data: dict[str, Any],
        percentile: int,
        normalize_x: bool = False,
    ) -> None:
        """
        Plot weight trajectory as diverging tree.

        Shows ALL individual trajectories, creating a tree-like structure
        for cartesian product experiments (1→1→N runs).
        """
        n_params = data["n_params"]
        param_idx = int(n_params * percentile / 100)

        fig, ax = plt.subplots(figsize=(14, 6))

        colors = plt.cm.Set1(np.linspace(0, 1, len(data["tunnels"])))

        for i, tunnel_data in enumerate(data["tunnels"]):
            weights = tunnel_data["weights"][:, :, param_idx]  # [n_runs, n_steps]
            n_runs = tunnel_data["n_runs"]

            x = tunnel_data["steps_normalized"] if normalize_x else tunnel_data["steps"]

            # Plot each run as individual trajectory
            for run_idx in range(n_runs):
                # First run gets label, others don't (to avoid legend clutter)
                label = f"{tunnel_data['tunnel_name']} ({n_runs} runs)" if run_idx == 0 else None

                # Thinner lines for many runs
                linewidth = 2 if n_runs == 1 else max(0.5, 2 / np.sqrt(n_runs))
                alpha = 1.0 if n_runs == 1 else max(0.3, 1.0 / np.sqrt(n_runs))

                ax.plot(x, weights[run_idx], color=colors[i], linewidth=linewidth,
                        alpha=alpha, label=label)

        # Add tunnel boundaries
        for i, tunnel_data in enumerate(data["tunnels"]):
            bnd = tunnel_data["start_boundary_normalized"] if normalize_x else tunnel_data["start_boundary"]
            if bnd > 0:
                ax.axvline(x=bnd, color='red', linestyle='--', alpha=0.5)

        xlabel = "Step × Batch Size" if normalize_x else "Step"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Weight (percentile {percentile}%)")
        title_suffix = "(normalized X)" if normalize_x else "(raw steps)"
        ax.set_title(f"Weight Trajectory - Percentile {percentile}% - Tree View {title_suffix}")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        suffix = "_normalized" if normalize_x else ""
        filename = f"weight_percentile_{percentile}_tree{suffix}.png"
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()

        print(f"  ✓ {filename}")

    def _load_all_metrics(self) -> dict[str, Any] | None:
        """Load metrics (loss, accuracy) from all tunnels, preserving run order for grouping."""
        all_data = []

        cumulative_step = 0
        cumulative_normalized = 0

        for tunnel in self.tunnels:
            tunnel_name = tunnel["name"]
            n_runs = tunnel["n_runs"]
            batch_size = self.batch_sizes.get(tunnel_name, 64)
            grouping = tunnel.get("grouping", {"groups": {0: list(range(n_runs))}, "n_groups": 1})

            # Load metrics from each run
            run_losses = []
            run_accuracies = []
            run_indices = []  # Track which run each metric belongs to
            run_steps_local = None
            run_train_full_losses = []
            run_train_full_accs = []
            run_train_full_steps_local = None

            for run_dir in tunnel["runs"]:
                run_idx = int(run_dir.name.split("_")[1])
                
                # Try different metrics file naming patterns
                metrics_path = run_dir / f"metrics_{tunnel_name}.json"
                if not metrics_path.exists():
                    metrics_path = run_dir / "metrics.json"
                if not metrics_path.exists():
                    continue

                with open(metrics_path) as f:
                    data = json.load(f)

                steps = data.get("steps", [])
                losses = data.get("losses", [])
                accuracies = data.get("accuracies", [])
                train_full_losses = data.get("train_full_losses", data.get("train_eval_losses", []))
                train_full_accs = data.get("train_full_accs", data.get("train_eval_accs", []))
                train_full_steps = data.get(
                    "train_full_steps",
                    data.get("train_eval_steps", data.get("val_steps", [])),
                )

                if not steps or not losses:
                    continue

                run_losses.append(np.array(losses))
                run_accuracies.append(np.array(accuracies) if accuracies else np.zeros(len(losses)))
                run_indices.append(run_idx)

                if run_steps_local is None:
                    run_steps_local = np.array(steps)

                if train_full_losses:
                    if not train_full_steps:
                        train_full_steps = list(range(1, len(train_full_losses) + 1))
                    run_train_full_losses.append(np.array(train_full_losses))
                    run_train_full_accs.append(
                        np.array(train_full_accs) if train_full_accs else np.zeros(len(train_full_losses))
                    )
                    if run_train_full_steps_local is None:
                        run_train_full_steps_local = np.array(train_full_steps)

            if not run_losses:
                continue

            # Stack runs: [n_runs, n_steps]
            try:
                stacked_losses = np.stack(run_losses)
                stacked_accs = np.stack(run_accuracies)
            except ValueError:
                # Pad if different lengths
                max_len = max(l.shape[0] for l in run_losses)
                padded_losses = []
                padded_accs = []
                for l, a in zip(run_losses, run_accuracies, strict=False):
                    if l.shape[0] < max_len:
                        l = np.pad(l, (0, max_len - l.shape[0]), constant_values=np.nan)
                        a = np.pad(a, (0, max_len - a.shape[0]), constant_values=np.nan)
                    padded_losses.append(l)
                    padded_accs.append(a)
                stacked_losses = np.stack(padded_losses)
                stacked_accs = np.stack(padded_accs)
                if run_steps_local is not None and len(run_steps_local) < max_len:
                    run_steps_local = np.arange(1, max_len + 1)

            train_full_available = bool(run_train_full_losses)
            if train_full_available:
                try:
                    stacked_train_full_losses = np.stack(run_train_full_losses)
                    stacked_train_full_accs = np.stack(run_train_full_accs)
                except ValueError:
                    max_len = max(l.shape[0] for l in run_train_full_losses)
                    padded_losses = []
                    padded_accs = []
                    for l, a in zip(run_train_full_losses, run_train_full_accs, strict=False):
                        if l.shape[0] < max_len:
                            l = np.pad(l, (0, max_len - l.shape[0]), constant_values=np.nan)
                            a = np.pad(a, (0, max_len - a.shape[0]), constant_values=np.nan)
                        padded_losses.append(l)
                        padded_accs.append(a)
                    stacked_train_full_losses = np.stack(padded_losses)
                    stacked_train_full_accs = np.stack(padded_accs)
                    if run_train_full_steps_local is not None and len(run_train_full_steps_local) < max_len:
                        run_train_full_steps_local = np.arange(1, max_len + 1)
                if run_train_full_steps_local is None:
                    run_train_full_steps_local = np.arange(1, stacked_train_full_losses.shape[1] + 1)

            n_steps = stacked_losses.shape[1]

            # Compute global steps
            global_steps = run_steps_local + cumulative_step
            global_steps_normalized = run_steps_local * batch_size + cumulative_normalized

            if train_full_available:
                global_train_full_steps = run_train_full_steps_local + cumulative_step
                global_train_full_steps_normalized = run_train_full_steps_local * batch_size + cumulative_normalized

            start_boundary = cumulative_step
            start_boundary_normalized = cumulative_normalized

            tunnel_entry = {
                "tunnel_name": tunnel_name,
                "n_runs": n_runs,
                "losses": stacked_losses,
                "accuracies": stacked_accs,
                "run_indices": run_indices,  # Which run each row belongs to
                "steps": global_steps,
                "steps_normalized": global_steps_normalized,
                "batch_size": batch_size,
                "start_boundary": start_boundary,
                "start_boundary_normalized": start_boundary_normalized,
                "grouping": grouping,  # Grouping info
            }
            if train_full_available:
                tunnel_entry.update({
                    "train_full_losses": stacked_train_full_losses,
                    "train_full_accs": stacked_train_full_accs,
                    "train_full_steps": global_train_full_steps,
                    "train_full_steps_normalized": global_train_full_steps_normalized,
                })
            all_data.append(tunnel_entry)

            cumulative_step += n_steps
            cumulative_normalized += n_steps * batch_size

        if not all_data:
            print("No metrics data found")
            return None

        return {
            "tunnels": all_data,
            "total_steps": cumulative_step,
            "total_normalized": cumulative_normalized,
        }

    def _plot_metric_meanstd(
        self,
        data: dict[str, Any],
        metric_name: str,  # "loss" or "accuracy"
        normalize_x: bool = False,
    ) -> None:
        """Plot metric with mean ± std, respecting run grouping."""
        fig, ax = plt.subplots(figsize=(14, 6))

        tunnel_colors = plt.cm.Set1(np.linspace(0, 1, len(data["tunnels"])))

        for i, tunnel_data in enumerate(data["tunnels"]):
            if metric_name == "loss":
                values = tunnel_data["losses"]  # [n_runs, n_steps]
                x = tunnel_data["steps_normalized"] if normalize_x else tunnel_data["steps"]
                label_name = "Loss"
            elif metric_name == "accuracy":
                values = tunnel_data["accuracies"]
                x = tunnel_data["steps_normalized"] if normalize_x else tunnel_data["steps"]
                label_name = "Accuracy"
            elif metric_name == "train_full_loss":
                values = tunnel_data.get("train_full_losses")
                if values is None:
                    continue
                x = tunnel_data["train_full_steps_normalized"] if normalize_x else tunnel_data["train_full_steps"]
                label_name = "Train Full Loss"
            elif metric_name == "train_full_accuracy":
                values = tunnel_data.get("train_full_accs")
                if values is None:
                    continue
                x = tunnel_data["train_full_steps_normalized"] if normalize_x else tunnel_data["train_full_steps"]
                label_name = "Train Full Accuracy"
            else:
                continue

            run_indices = tunnel_data.get("run_indices", list(range(values.shape[0])))
            n_runs = values.shape[0]
            
            grouping = tunnel_data.get("grouping", {"groups": {0: run_indices}, "n_groups": 1})
            groups = grouping["groups"]
            n_groups = grouping["n_groups"]
            
            base_color = tunnel_colors[i]

            if n_runs == 1:
                ax.plot(x, values[0], color=base_color, linewidth=2,
                        label=f"{tunnel_data['tunnel_name']} (1 run)")
            elif n_groups == 1:
                # Single group: standard mean ± std
                mean = np.nanmean(values, axis=0)
                std = np.nanstd(values, axis=0)

                ax.plot(x, mean, color=base_color, linewidth=2,
                        label=f"{tunnel_data['tunnel_name']} ({n_runs} runs)")
                ax.fill_between(x, mean - std, mean + std, color=base_color, alpha=0.3)
            else:
                # Multiple groups: separate mean ± std tube for each group
                group_colors = self._get_group_colors(base_color, n_groups)
                
                # Build a mapping from run_idx to array index
                run_idx_to_array_idx = {run_idx: arr_idx for arr_idx, run_idx in enumerate(run_indices)}
                
                for group_idx, group_run_indices in sorted(groups.items()):
                    array_indices = [run_idx_to_array_idx[rid] for rid in group_run_indices 
                                   if rid in run_idx_to_array_idx]
                    
                    if not array_indices:
                        continue
                    
                    group_values = values[array_indices]
                    group_color = group_colors[group_idx % len(group_colors)]
                    
                    if len(array_indices) == 1:
                        ax.plot(x, group_values[0], color=group_color, linewidth=1.5, alpha=0.8)
                    else:
                        mean = np.nanmean(group_values, axis=0)
                        std = np.nanstd(group_values, axis=0)
                        
                        ax.plot(x, mean, color=group_color, linewidth=1.5, alpha=0.9)
                        ax.fill_between(x, mean - std, mean + std, color=group_color, alpha=0.2)
                
                # Single legend entry for the tunnel
                ax.plot([], [], color=base_color, linewidth=2,
                       label=f"{tunnel_data['tunnel_name']} ({n_groups} groups)")

        # Add tunnel boundaries
        for i, tunnel_data in enumerate(data["tunnels"]):
            bnd = tunnel_data["start_boundary_normalized"] if normalize_x else tunnel_data["start_boundary"]
            if bnd > 0:
                ax.axvline(x=bnd, color='red', linestyle='--', alpha=0.5)

        xlabel = "Step × Batch Size" if normalize_x else "Step"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label_name)
        title_suffix = "(normalized X)" if normalize_x else "(raw steps)"
        ax.set_title(f"{label_name} - Mean±Std {title_suffix}")
        ax.legend(loc='upper right' if metric_name in {"loss", "train_full_loss"} else 'lower right')
        ax.grid(True, alpha=0.3)

        suffix = "_normalized" if normalize_x else ""
        filename = f"{metric_name}_meanstd{suffix}.png"
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()

        print(f"  ✓ {filename}")

    def _plot_metric_tree(
        self,
        data: dict[str, Any],
        metric_name: str,
        normalize_x: bool = False,
    ) -> None:
        """Plot metric as diverging tree (all individual trajectories)."""
        fig, ax = plt.subplots(figsize=(14, 6))

        colors = plt.cm.Set1(np.linspace(0, 1, len(data["tunnels"])))

        for i, tunnel_data in enumerate(data["tunnels"]):
            if metric_name == "loss":
                values = tunnel_data["losses"]
                x = tunnel_data["steps_normalized"] if normalize_x else tunnel_data["steps"]
                label_name = "Loss"
            elif metric_name == "accuracy":
                values = tunnel_data["accuracies"]
                x = tunnel_data["steps_normalized"] if normalize_x else tunnel_data["steps"]
                label_name = "Accuracy"
            elif metric_name == "train_full_loss":
                values = tunnel_data.get("train_full_losses")
                if values is None:
                    continue
                x = tunnel_data["train_full_steps_normalized"] if normalize_x else tunnel_data["train_full_steps"]
                label_name = "Train Full Loss"
            elif metric_name == "train_full_accuracy":
                values = tunnel_data.get("train_full_accs")
                if values is None:
                    continue
                x = tunnel_data["train_full_steps_normalized"] if normalize_x else tunnel_data["train_full_steps"]
                label_name = "Train Full Accuracy"
            else:
                continue

            n_runs = tunnel_data["n_runs"]

            for run_idx in range(n_runs):
                label = f"{tunnel_data['tunnel_name']} ({n_runs} runs)" if run_idx == 0 else None
                linewidth = 2 if n_runs == 1 else max(0.5, 2 / np.sqrt(n_runs))
                alpha = 1.0 if n_runs == 1 else max(0.3, 1.0 / np.sqrt(n_runs))

                ax.plot(x, values[run_idx], color=colors[i], linewidth=linewidth,
                        alpha=alpha, label=label)

        # Add tunnel boundaries
        for i, tunnel_data in enumerate(data["tunnels"]):
            bnd = tunnel_data["start_boundary_normalized"] if normalize_x else tunnel_data["start_boundary"]
            if bnd > 0:
                ax.axvline(x=bnd, color='red', linestyle='--', alpha=0.5)

        xlabel = "Step × Batch Size" if normalize_x else "Step"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label_name)
        title_suffix = "(normalized X)" if normalize_x else "(raw steps)"
        ax.set_title(f"{label_name} - Tree View {title_suffix}")
        ax.legend(loc='upper right' if metric_name in {"loss", "train_full_loss"} else 'lower right')
        ax.grid(True, alpha=0.3)

        suffix = "_normalized" if normalize_x else ""
        filename = f"{metric_name}_tree{suffix}.png"
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()

        print(f"  ✓ {filename}")

    def _plot_hessian_spectrum(self) -> None:
        """Plot Hessian eigenvalue spectrum."""
        eigenvalues_list = []

        for tunnel in reversed(self.tunnels):
            tunnel_name = tunnel["name"]
            for run_dir in tunnel["runs"]:
                hessians_path = run_dir / f"hessians_{tunnel_name}.pt"
                if not hessians_path.exists():
                    hessians_path = run_dir / "hessians.pt"
                if not hessians_path.exists():
                    continue

                data = torch.load(hessians_path, map_location="cpu")
                if "eigenvalues" in data:
                    eigs = data["eigenvalues"]
                    if isinstance(eigs, list):
                        eigs = torch.stack(eigs)
                        avg_eig = eigs.mean(dim=0)
                    else:
                        avg_eig = eigs.mean(dim=0)
                    eigenvalues_list.append(avg_eig.numpy())

            if eigenvalues_list:
                break

        if not eigenvalues_list:
            print("  No Hessian eigenvalues found")
            return

        # Stack and compute mean/std
        all_eigs = np.stack(eigenvalues_list)
        mean_eigs = all_eigs.mean(axis=0)
        std_eigs = all_eigs.std(axis=0) if len(eigenvalues_list) > 1 else np.zeros_like(mean_eigs)

        # Sort eigenvalues
        sorted_indices = np.argsort(mean_eigs)
        mean_sorted = mean_eigs[sorted_indices]
        std_sorted = std_eigs[sorted_indices]

        # Find cutoff (near zero)
        eps = 1e-6
        left_mask = mean_sorted < -eps
        right_mask = mean_sorted > eps

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        n_show_left = min(20, n_left)
        n_show_right = min(20, n_right)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Left part (negative eigenvalues)
        if n_show_left > 0:
            left_indices = np.where(left_mask)[0][:n_show_left]
            x_left = np.arange(len(left_indices))
            mean_left = mean_sorted[left_indices]
            std_left = std_sorted[left_indices]

            ax.plot(x_left, mean_left, 'r-o', markersize=4, label='Negative')
            if len(eigenvalues_list) > 1:
                ax.fill_between(x_left, mean_left, mean_left - std_left, color='red', alpha=0.3)

        # Gap
        gap_x = n_show_left + 0.5
        ax.axvline(x=gap_x, color='gray', linestyle='--', alpha=0.7, label='Cutoff')

        # Right part (positive eigenvalues)
        if n_show_right > 0:
            right_indices = np.where(right_mask)[0][-n_show_right:]
            x_right = np.arange(n_show_left + 1, n_show_left + 1 + len(right_indices))
            mean_right = mean_sorted[right_indices]
            std_right = std_sorted[right_indices]

            ax.plot(x_right, mean_right, 'b-o', markersize=4, label='Positive')
            if len(eigenvalues_list) > 1:
                ax.fill_between(x_right, mean_right, mean_right + std_right, color='blue', alpha=0.3)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel("Sorted Eigenvalue Index")
        ax.set_ylabel("Eigenvalue (mean ± std)" if len(eigenvalues_list) > 1 else "Eigenvalue")
        ax.set_title(f"Hessian Eigenvalue Spectrum ({len(eigenvalues_list)} runs)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "hessian_spectrum.png", dpi=150)
        plt.close()

        print("  ✓ hessian_spectrum.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize tunnel-based experiment data")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for plots")
    parser.add_argument("--percentiles", type=int, nargs="+", default=[0, 20, 40, 60, 80],
                        help="Weight percentiles to plot")

    args = parser.parse_args()

    visualizer = TunnelVisualizer(
        exp_dir=Path(args.exp_dir),
        output_dir=Path(args.output_dir),
    )
    visualizer.plot_all(percentiles=args.percentiles)


if __name__ == "__main__":
    main()
