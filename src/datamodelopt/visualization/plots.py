"""
Plotting functions for experiment results.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


def get_percentile_weight_indices(n_params: int, percentiles: Optional[list[float]] = None) -> list[int]:
    """
    Get weight indices at specified percentiles.

    Args:
        n_params: Total number of parameters.
        percentiles: List of percentiles (0-100). Default: [0, 20, 40, 60, 80].

    Returns:
        List of parameter indices.
    """
    if percentiles is None:
        percentiles = [0, 20, 40, 60, 80]

    indices = []
    for p in percentiles:
        idx = int(p / 100 * (n_params - 1))
        indices.append(idx)

    return indices


def load_metrics_json(path: str) -> dict[str, Any]:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_weights_pt(path: str) -> dict[str, Any]:
    """Load weights from .pt file."""
    if not HAS_TORCH:
        raise ImportError("torch is required to load .pt files")
    return torch.load(path, map_location="cpu", weights_only=False)


def plot_loss_curves(
    metrics: dict[str, Any],
    title: str = "Loss Curves",
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        metrics: Dictionary with 'losses', 'val_losses', 'steps', 'val_steps' keys.
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    steps = metrics.get("steps", list(range(1, len(metrics.get("losses", [])) + 1)))
    losses = metrics.get("losses", [])

    if losses:
        ax.plot(steps, losses, label="Train Loss", alpha=0.8)

    val_steps = metrics.get("val_steps", [])
    val_losses = metrics.get("val_losses", [])

    if val_losses:
        ax.plot(val_steps, val_losses, label="Val Loss", marker="o", markersize=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")

    plt.close(fig)


def plot_accuracy_curves(
    metrics: dict[str, Any],
    title: str = "Accuracy Curves",
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot training and validation accuracy curves.

    Args:
        metrics: Dictionary with 'accuracies', 'val_accuracies' keys.
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    steps = metrics.get("steps", list(range(1, len(metrics.get("accuracies", [])) + 1)))
    accuracies = metrics.get("accuracies", [])

    if accuracies:
        ax.plot(steps, accuracies, label="Train Acc", alpha=0.8)

    val_steps = metrics.get("val_steps", [])
    val_accs = metrics.get("val_accuracies", [])

    if val_accs:
        ax.plot(val_steps, val_accs, label="Val Acc", marker="o", markersize=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")

    plt.close(fig)


def plot_loss_and_accuracy(
    metrics: dict[str, Any],
    title: str = "Training Progress",
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> None:
    """
    Plot loss and accuracy side by side.

    Args:
        metrics: Dictionary with metrics data.
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    ax = axes[0]
    steps = metrics.get("steps", list(range(1, len(metrics.get("losses", [])) + 1)))
    losses = metrics.get("losses", [])

    if losses:
        ax.plot(steps, losses, label="Train Loss", alpha=0.8)

    val_steps = metrics.get("val_steps", [])
    val_losses = metrics.get("val_losses", [])

    if val_losses:
        ax.plot(val_steps, val_losses, label="Val Loss", marker="o", markersize=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} - Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    ax = axes[1]
    accuracies = metrics.get("accuracies", [])

    if accuracies:
        ax.plot(steps, accuracies, label="Train Acc", alpha=0.8)

    val_accs = metrics.get("val_accuracies", [])

    if val_accs:
        ax.plot(val_steps, val_accs, label="Val Acc", marker="o", markersize=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} - Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")

    plt.close(fig)


def plot_gradient_norms(
    metrics: dict[str, Any],
    title: str = "Gradient Norms",
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    log_scale: bool = False,
) -> None:
    """
    Plot gradient norm over training.

    Args:
        metrics: Dictionary with 'grad_norms' key.
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
        log_scale: Use log scale for y-axis.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    steps = metrics.get("steps", list(range(1, len(metrics.get("grad_norms", [])) + 1)))
    grad_norms = metrics.get("grad_norms", [])

    if grad_norms:
        ax.plot(steps, grad_norms, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")

    plt.close(fig)


def plot_weight_trajectory_2d(
    weights_data: dict[str, Any],
    title: str = "Weight Trajectory (PCA)",
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    method: str = "pca",
) -> None:
    """
    Plot 2D projection of weight trajectory.

    Args:
        weights_data: Dictionary with 'weights' tensor.
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
        method: Dimensionality reduction method ('pca' or 'first_two').
    """
    _check_matplotlib()

    if not HAS_TORCH:
        raise ImportError("torch is required for weight trajectory plots")

    weights = weights_data.get("weights")
    if weights is None:
        print("No weights data found")
        return

    if isinstance(weights, torch.Tensor):
        weights = weights.numpy()

    n_snapshots, n_params = weights.shape

    if n_snapshots < 2:
        print("Need at least 2 weight snapshots for trajectory plot")
        return

    # Dimensionality reduction
    if method == "pca" and n_params > 2:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(weights)
            xlabel = "PC1"
            ylabel = "PC2"
        except ImportError:
            # Fall back to first two dimensions
            coords = weights[:, :2]
            xlabel = "Param 0"
            ylabel = "Param 1"
    else:
        coords = weights[:, :2]
        xlabel = "Param 0"
        ylabel = "Param 1"

    fig, ax = plt.subplots(figsize=figsize)

    # Color by step
    colors = np.linspace(0, 1, n_snapshots)
    cmap = plt.cm.viridis  # type: ignore[attr-defined]

    # Plot trajectory
    for i in range(n_snapshots - 1):
        ax.plot(
            coords[i:i+2, 0],
            coords[i:i+2, 1],
            color=cmap(colors[i]),
            alpha=0.7,
            linewidth=1,
        )

    # Mark start and end
    ax.scatter(coords[0, 0], coords[0, 1], c="green", s=100, marker="o", label="Start", zorder=5)
    ax.scatter(coords[-1, 0], coords[-1, 1], c="red", s=100, marker="*", label="End", zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_snapshots))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Step")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")

    plt.close(fig)


def plot_combined_stages(
    run_dir: str,
    stages: list[str],
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 10),
) -> None:
    """
    Plot combined metrics from multiple stages.

    Args:
        run_dir: Directory containing stage results.
        stages: List of stage names (e.g., ["stage1_sgd", "stage2_gd", "stage3_sgd"]).
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    """
    _check_matplotlib()

    run_path = Path(run_dir)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    cumulative_step = 0

    all_losses = []
    all_val_losses = []
    all_accs = []
    all_val_accs = []
    all_steps = []
    all_val_steps = []
    stage_boundaries = [0]

    for stage_name in stages:
        metrics_path = run_path / f"metrics_{stage_name}.json"
        if not metrics_path.exists():
            print(f"Warning: {metrics_path} not found, skipping")
            continue

        metrics = load_metrics_json(str(metrics_path))

        steps = metrics.get("steps", [])
        losses = metrics.get("losses", [])
        accuracies = metrics.get("accuracies", [])
        val_steps = metrics.get("val_steps", [])
        val_losses = metrics.get("val_losses", [])
        val_accs = metrics.get("val_accuracies", [])

        # Offset steps
        offset_steps = [s + cumulative_step for s in steps]
        offset_val_steps = [s + cumulative_step for s in val_steps]

        all_losses.extend(losses)
        all_accs.extend(accuracies)
        all_steps.extend(offset_steps)

        all_val_losses.extend(val_losses)
        all_val_accs.extend(val_accs)
        all_val_steps.extend(offset_val_steps)

        if steps:
            cumulative_step += max(steps)

        stage_boundaries.append(cumulative_step)

    # Plot combined loss
    ax = axes[0, 0]
    if all_losses:
        ax.plot(all_steps, all_losses, alpha=0.7, label="Train Loss")
    if all_val_losses:
        ax.plot(all_val_steps, all_val_losses, "o-", markersize=3, label="Val Loss")

    # Add stage boundaries
    for boundary in stage_boundaries[1:-1]:
        ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Combined Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot combined accuracy
    ax = axes[0, 1]
    if all_accs:
        ax.plot(all_steps, all_accs, alpha=0.7, label="Train Acc")
    if all_val_accs:
        ax.plot(all_val_steps, all_val_accs, "o-", markersize=3, label="Val Acc")

    for boundary in stage_boundaries[1:-1]:
        ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Combined Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot per-stage losses
    ax = axes[1, 0]
    cumulative_step = 0
    for stage_idx, stage_name in enumerate(stages):
        metrics_path = run_path / f"metrics_{stage_name}.json"
        if not metrics_path.exists():
            continue

        metrics = load_metrics_json(str(metrics_path))
        steps = metrics.get("steps", [])
        losses = metrics.get("losses", [])

        if losses:
            offset_steps = [s + cumulative_step for s in steps]
            ax.plot(offset_steps, losses, color=colors[stage_idx % len(colors)], alpha=0.8, label=stage_name)
            if steps:
                cumulative_step += max(steps)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss by Stage")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot gradient norms
    ax = axes[1, 1]
    cumulative_step = 0
    for stage_idx, stage_name in enumerate(stages):
        metrics_path = run_path / f"metrics_{stage_name}.json"
        if not metrics_path.exists():
            continue

        metrics = load_metrics_json(str(metrics_path))
        steps = metrics.get("steps", [])
        grad_norms = metrics.get("grad_norms", [])

        if grad_norms:
            offset_steps = [s + cumulative_step for s in steps]
            ax.plot(offset_steps, grad_norms, color=colors[stage_idx % len(colors)], alpha=0.8, label=stage_name)
            if steps:
                cumulative_step += max(steps)

    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms by Stage")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    else:
        # Default save path
        default_path = run_path / "combined_plots.png"
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {default_path}")

    plt.close(fig)


def load_many_runs_weights(
    run_dirs: list[str],
    stages: list[str],
    stage_batch_sizes: dict[str, int] | None = None,
    train_dataset_size: int | None = None,
) -> dict[str, Any]:
    """
    Load weight trajectories from multiple runs and concatenate stages.

    Args:
        run_dirs: List of paths to run directories.
        stages: List of stage names in order.
        stage_batch_sizes: Dict mapping stage_name -> batch_size for normalization.
        train_dataset_size: Size of training dataset (for GD step normalization).

    Returns:
        Dictionary with:
            - 'weights': List of [n_steps, n_params] arrays per run
            - 'steps': List of step arrays per run
            - 'normalized_steps': List of normalized step arrays per run
            - 'stage_boundaries': List of step indices where stages change
            - 'stage_boundaries_normalized': List of normalized step values at boundaries
            - 'stage_names': List of stage names that were loaded
            - 'n_params': Number of parameters
    """
    if not HAS_TORCH:
        raise ImportError("torch is required to load weights")

    all_weights = []
    all_steps = []
    all_normalized_steps = []
    n_params = None
    stage_boundaries: list[int] = []
    stage_boundaries_normalized = []
    loaded_stages = []

    for run_dir in run_dirs:
        run_path = Path(run_dir)
        run_weights = []
        run_steps = []
        run_normalized_steps = []
        cumulative_step = 0
        cumulative_normalized = 0
        run_boundaries = [0]
        run_boundaries_normalized = [0]
        run_loaded_stages = []

        for stage_name in stages:
            weights_path = run_path / f"weights_{stage_name}.pt"

            if not weights_path.exists():
                # Try to get weights from checkpoint
                checkpoint_path = run_path / f"{stage_name}.pt"
                if checkpoint_path.exists():
                    # Can't get trajectory from checkpoint, skip
                    continue
                continue

            data = torch.load(str(weights_path), map_location="cpu", weights_only=False)
            weights = data.get("weights")
            steps = data.get("steps", [])

            if weights is None:
                continue

            if isinstance(weights, torch.Tensor):
                weights = weights.numpy()

            if n_params is None:
                n_params = weights.shape[1]

            run_loaded_stages.append(stage_name)

            # Offset steps
            offset_steps = [s + cumulative_step for s in steps]
            run_weights.append(weights)
            run_steps.extend(offset_steps)

            # Normalized steps (account for batch size vs full batch)
            batch_size = 1
            if stage_batch_sizes and stage_name in stage_batch_sizes:
                batch_size = stage_batch_sizes[stage_name]
            elif train_dataset_size and "gd" in stage_name.lower():
                # GD uses full batch
                batch_size = train_dataset_size

            # Normalize: each step processes batch_size samples
            # So 1 GD step = train_dataset_size / minibatch_size SGD steps
            normalization_factor = batch_size if stage_batch_sizes else 1
            normalized_steps = [cumulative_normalized + s * normalization_factor for s in steps]
            run_normalized_steps.extend(normalized_steps)

            if steps:
                # Update cumulative counters for next stage
                # The next stage should start after the last step of current stage
                last_offset_step = max(offset_steps)
                cumulative_step = last_offset_step + 1

                last_normalized_step = max(normalized_steps)
                cumulative_normalized = last_normalized_step + normalization_factor

            # Add boundary at the start of NEXT stage
            run_boundaries.append(cumulative_step)
            run_boundaries_normalized.append(cumulative_normalized)

        if run_weights:
            # Concatenate all stages for this run
            all_weights.append(np.concatenate(run_weights, axis=0))
            all_steps.append(np.array(run_steps))
            all_normalized_steps.append(np.array(run_normalized_steps))
            if not stage_boundaries:
                stage_boundaries = run_boundaries
                stage_boundaries_normalized = run_boundaries_normalized
                loaded_stages = run_loaded_stages

    return {
        "weights": all_weights,
        "steps": all_steps,
        "normalized_steps": all_normalized_steps,
        "stage_boundaries": stage_boundaries,
        "stage_boundaries_normalized": stage_boundaries_normalized,
        "stage_names": loaded_stages,
        "n_params": n_params,
    }


def _add_stage_boundaries_and_labels(
    ax,
    stage_names: list[str],
    stage_boundaries: list[int],
    x_axis: np.ndarray,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    label_y_position: str | None = "top",
) -> None:
    """
    Add vertical lines at stage boundaries and labels for stage names.

    Args:
        ax: Matplotlib axis.
        stage_names: List of stage names.
        stage_boundaries: List of boundary step values (not indices!) including 0 and end.
        x_axis: X axis values (actual step numbers).
        y_min, y_max: Y axis limits (for label positioning).
        label_y_position: "top", "bottom", or None (no labels, only lines).
    """
    # stage_boundaries are step VALUES, not array indices
    # Find the x-axis values that correspond to boundaries
    x_min, x_max = x_axis[0], x_axis[-1]

    # Draw vertical lines at boundaries (except first and last)
    for boundary_step in stage_boundaries[1:-1]:
        # boundary_step is the step number, draw line at this x value
        if x_min <= boundary_step <= x_max:
            ax.axvline(x=boundary_step, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    # Skip labels if label_y_position is None
    if label_y_position is None:
        return

    if y_min is None or y_max is None:
        y_min, y_max = ax.get_ylim()

    # Label position
    if label_y_position == "top":
        label_y = y_max - 0.05 * (y_max - y_min)
        va = "top"
    else:
        label_y = y_min + 0.05 * (y_max - y_min)
        va = "bottom"

    # Add stage name labels in the middle of each stage region
    for i, stage_name in enumerate(stage_names):
        # Get start and end step values for this stage
        x_start = stage_boundaries[i]
        x_end = stage_boundaries[i + 1] if i + 1 < len(stage_boundaries) else x_max

        # Clamp to visible range
        x_start = max(x_start, x_min)
        x_end = min(x_end, x_max)

        if x_end <= x_start:
            continue

        # Calculate middle x position for label
        x_mid = (x_start + x_end) / 2

        # Format stage name for display (e.g., "stage1_sgd" -> "SGD")
        display_name = _format_stage_name(stage_name)

        ax.text(
            x_mid, label_y, display_name,
            ha="center", va=va,
            fontsize=10, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
        )


def _format_stage_name(stage_name: str) -> str:
    """
    Format stage name for display.

    Examples:
        "stage1_sgd" -> "SGD"
        "stage2_gd" -> "GD"
        "stage3_adam" -> "Adam"
    """
    # Remove stage prefix
    name = stage_name
    if "_" in name:
        parts = name.split("_")
        # Take the last part (usually the optimizer name)
        name = parts[-1]

    # Capitalize known abbreviations
    upper_names = {"sgd", "gd", "adam", "adamw", "rmsprop"}
    if name.lower() in upper_names:
        return name.upper()

    return name.capitalize()


def plot_weight_trajectories_percentiles(
    run_dirs: list[str],
    stages: list[str],
    percentiles: Optional[list[float]] = None,
    save_dir: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    normalize_x: bool = False,
    stage_batch_sizes: dict[str, int] | None = None,
    train_dataset_size: int | None = None,
    title_prefix: str = "",
) -> None:
    """
    Plot weight trajectories for selected percentile weights across many runs.

    Creates 5 plots (one per percentile) showing mean ± std over runs.

    Args:
        run_dirs: List of paths to run directories (many runs).
        stages: List of stage names in order.
        percentiles: Percentiles to plot (default: [0, 20, 40, 60, 80]).
        save_dir: Directory to save plots. If None, uses first run_dir.
        figsize: Figure size for each plot.
        normalize_x: If True, normalize X axis by batch size.
        stage_batch_sizes: Dict mapping stage_name -> batch_size.
        train_dataset_size: Size of training dataset.
        title_prefix: Prefix for plot titles.
    """
    _check_matplotlib()

    if percentiles is None:
        percentiles = [0, 20, 40, 60, 80]

    # Load all trajectories
    data = load_many_runs_weights(
        run_dirs, stages, stage_batch_sizes, train_dataset_size
    )

    if not data["weights"]:
        print("No weight data found")
        return

    weights_list = data["weights"]  # List of [n_steps, n_params] arrays
    n_params = data["n_params"]
    stage_boundaries = data["stage_boundaries"]
    stage_boundaries_normalized = data.get("stage_boundaries_normalized", stage_boundaries)
    stage_names = data.get("stage_names", stages)

    # Get indices for percentile weights
    weight_indices = get_percentile_weight_indices(n_params, percentiles)

    save_path = Path(save_dir) if save_dir else Path(run_dirs[0])
    save_path.mkdir(parents=True, exist_ok=True)

    # Find common length (minimum across runs)
    min_steps = min(w.shape[0] for w in weights_list)

    # Align all trajectories to same length
    aligned_weights = np.stack([w[:min_steps] for w in weights_list], axis=0)
    # Shape: [n_runs, n_steps, n_params]

    # Choose X axis and boundaries
    if normalize_x and data["normalized_steps"]:
        x_axis = data["normalized_steps"][0][:min_steps]
        x_label = "Step × Batch Size"
        boundaries_for_plot = stage_boundaries_normalized  # Use normalized boundaries
    else:
        x_axis = data["steps"][0][:min_steps]
        x_label = "Step"
        boundaries_for_plot = stage_boundaries

    # Plot each percentile weight
    for p, idx in zip(percentiles, weight_indices, strict=False):
        fig, ax = plt.subplots(figsize=figsize)

        # Extract this weight across all runs
        weight_values = aligned_weights[:, :, idx]  # [n_runs, n_steps]

        mean = np.mean(weight_values, axis=0)
        std = np.std(weight_values, axis=0)

        # Plot mean line
        ax.plot(x_axis, mean, linewidth=2, label=f"Mean (weight #{idx})")

        # Plot std band
        ax.fill_between(
            x_axis,
            mean - std,
            mean + std,
            alpha=0.3,
            label="±1 std"
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Weight Value")
        title = f"{title_prefix}Weight at {p}% percentile (idx={idx})"
        if normalize_x:
            title += " [X normalized]"
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Add stage boundaries and labels
        _add_stage_boundaries_and_labels(
            ax, stage_names, boundaries_for_plot, x_axis,
            label_y_position="top"
        )

        plt.tight_layout()

        suffix = "_normalized" if normalize_x else ""
        plot_path = save_path / f"weight_percentile_{int(p)}{suffix}.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        print(f"[saved] {plot_path}")

        plt.close(fig)


def plot_weight_trajectories_combined(
    run_dirs: list[str],
    stages: list[str],
    percentiles: Optional[list[float]] = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 12),
    normalize_x: bool = False,
    stage_batch_sizes: dict[str, int] | None = None,
    train_dataset_size: int | None = None,
    title: str = "Weight Trajectories (mean ± std)",
) -> None:
    """
    Plot all percentile weight trajectories in a single figure with subplots.

    Args:
        run_dirs: List of paths to run directories (many runs).
        stages: List of stage names in order.
        percentiles: Percentiles to plot (default: [0, 20, 40, 60, 80]).
        save_path: Path to save the combined plot.
        figsize: Figure size.
        normalize_x: If True, normalize X axis by batch size.
        stage_batch_sizes: Dict mapping stage_name -> batch_size.
        train_dataset_size: Size of training dataset.
        title: Main title for the figure.
    """
    _check_matplotlib()

    if percentiles is None:
        percentiles = [0, 20, 40, 60, 80]

    # Load all trajectories
    data = load_many_runs_weights(
        run_dirs, stages, stage_batch_sizes, train_dataset_size
    )

    if not data["weights"]:
        print("No weight data found")
        return

    weights_list = data["weights"]
    n_params = data["n_params"]
    stage_boundaries = data["stage_boundaries"]
    stage_boundaries_normalized = data.get("stage_boundaries_normalized", stage_boundaries)
    stage_names = data.get("stage_names", stages)

    weight_indices = get_percentile_weight_indices(n_params, percentiles)

    min_steps = min(w.shape[0] for w in weights_list)
    aligned_weights = np.stack([w[:min_steps] for w in weights_list], axis=0)

    if normalize_x and data["normalized_steps"]:
        x_axis = data["normalized_steps"][0][:min_steps]
        x_label = "Step × Batch Size"
        boundaries_for_plot = stage_boundaries_normalized
    else:
        x_axis = data["steps"][0][:min_steps]
        x_label = "Step"
        boundaries_for_plot = stage_boundaries

    # Create subplots
    n_plots = len(percentiles)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (p, idx) in enumerate(zip(percentiles, weight_indices, strict=False)):
        ax = axes[i]

        weight_values = aligned_weights[:, :, idx]
        mean = np.mean(weight_values, axis=0)
        std = np.std(weight_values, axis=0)

        color = colors[i % len(colors)]
        ax.plot(x_axis, mean, linewidth=2, color=color, label="Mean")
        ax.fill_between(x_axis, mean - std, mean + std, alpha=0.3, color=color)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Weight Value")
        ax.set_title(f"{int(p)}% percentile (idx={idx})")
        ax.grid(True, alpha=0.3)

        # Add stage boundaries and labels
        _add_stage_boundaries_and_labels(
            ax, stage_names, boundaries_for_plot, x_axis,
            label_y_position="top"
        )

    # Hide empty subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path is None:
        save_path = Path(run_dirs[0]) / "weight_trajectories_combined.png"

    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"[saved] {save_path}")

    plt.close(fig)


def plot_weight_trajectories_both_normalizations(
    run_dirs: list[str],
    stages: list[str],
    percentiles: Optional[list[float]] = None,
    save_dir: str | None = None,
    figsize: tuple[int, int] = (18, 12),
    stage_batch_sizes: dict[str, int] | None = None,
    train_dataset_size: int | None = None,
    title: str = "Weight Trajectories Comparison",
) -> None:
    """
    Plot weight trajectories with both normalized and non-normalized X axes side by side.

    Args:
        run_dirs: List of paths to run directories.
        stages: List of stage names in order.
        percentiles: Percentiles to plot (default: [0, 20, 40, 60, 80]).
        save_dir: Directory to save the plot.
        figsize: Figure size.
        stage_batch_sizes: Dict mapping stage_name -> batch_size.
        train_dataset_size: Size of training dataset.
        title: Main title for the figure.
    """
    _check_matplotlib()

    if percentiles is None:
        percentiles = [0, 20, 40, 60, 80]

    data = load_many_runs_weights(
        run_dirs, stages, stage_batch_sizes, train_dataset_size
    )

    if not data["weights"]:
        print("No weight data found")
        return

    weights_list = data["weights"]
    n_params = data["n_params"]
    stage_boundaries = data["stage_boundaries"]
    stage_boundaries_normalized = data.get("stage_boundaries_normalized", stage_boundaries)
    stage_names = data.get("stage_names", stages)

    weight_indices = get_percentile_weight_indices(n_params, percentiles)

    min_steps = min(w.shape[0] for w in weights_list)
    aligned_weights = np.stack([w[:min_steps] for w in weights_list], axis=0)

    x_raw = data["steps"][0][:min_steps]
    x_normalized = data["normalized_steps"][0][:min_steps] if data["normalized_steps"] else x_raw

    # Create figure with 2 columns: raw steps, normalized steps
    n_rows = len(percentiles)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (p, idx) in enumerate(zip(percentiles, weight_indices, strict=False)):
        weight_values = aligned_weights[:, :, idx]
        mean = np.mean(weight_values, axis=0)
        std = np.std(weight_values, axis=0)

        color = colors[i % len(colors)]

        # Left: raw steps
        ax = axes[i, 0]
        ax.plot(x_raw, mean, linewidth=2, color=color, label="Mean")
        ax.fill_between(x_raw, mean - std, mean + std, alpha=0.3, color=color)
        ax.set_ylabel(f"{int(p)}%\n(idx={idx})")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Raw Steps")
        if i == n_rows - 1:
            ax.set_xlabel("Step")

        # Add stage boundaries and labels (only on first row to avoid clutter)
        _add_stage_boundaries_and_labels(
            ax, stage_names, stage_boundaries, x_raw,
            label_y_position="top" if i == 0 else None
        )

        # Right: normalized steps
        ax = axes[i, 1]
        ax.plot(x_normalized, mean, linewidth=2, color=color, label="Mean")
        ax.fill_between(x_normalized, mean - std, mean + std, alpha=0.3, color=color)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Step × Batch Size")
        if i == n_rows - 1:
            ax.set_xlabel("Step × Batch Size")

        # Add stage boundaries and labels (use normalized boundaries)
        _add_stage_boundaries_and_labels(
            ax, stage_names, stage_boundaries_normalized, x_normalized,
            label_y_position="top" if i == 0 else None
        )

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    save_path = Path(save_dir) if save_dir else Path(run_dirs[0])
    save_path.mkdir(parents=True, exist_ok=True)
    plot_path = save_path / "weight_trajectories_comparison.png"

    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    print(f"[saved] {plot_path}")

    plt.close(fig)


def plot_hessian_spectrum(
    run_dirs: list[str],
    stages: list[str],
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 8),
    n_left: int = 20,
    n_right: int = 20,
    eps: float = 1e-6,
    title: str = "Hessian Eigenvalue Spectrum",
) -> None:
    """
    Plot mean Hessian eigenvalue spectrum across many runs.

    Shows the leftmost (most negative) and rightmost (most positive) eigenvalues,
    cutting out the middle near-zero part. Mean ± std visualization with correct
    fill direction based on sign.

    Args:
        run_dirs: List of paths to run directories (many runs).
        stages: List of stage names in order.
        save_path: Path to save the plot.
        figsize: Figure size.
        n_left: Number of leftmost (negative) eigenvalues to show.
        n_right: Number of rightmost (positive) eigenvalues to show.
        eps: Threshold for near-zero eigenvalues to skip.
        title: Plot title.
    """
    _check_matplotlib()

    if not HAS_TORCH:
        raise ImportError("torch is required to load Hessians")

    # Collect all Hessian eigenvalues from all runs
    all_eigenvalues = []  # List of [n_steps, n_params] arrays per run

    for run_dir in run_dirs:
        run_path = Path(run_dir)

        for stage_name in stages:
            hessian_path = run_path / f"hessians_{stage_name}.pt"

            if not hessian_path.exists():
                continue

            data = torch.load(str(hessian_path), map_location="cpu", weights_only=False)
            eigenvalues = data.get("eigenvalues")

            if eigenvalues is None:
                continue

            if isinstance(eigenvalues, torch.Tensor):
                eigenvalues = eigenvalues.numpy()

            # eigenvalues shape: [n_steps, n_params]
            # Sort each step's eigenvalues
            eigenvalues_sorted = np.sort(eigenvalues, axis=1)
            all_eigenvalues.append(eigenvalues_sorted)

    if not all_eigenvalues:
        print("No Hessian eigenvalues found")
        return

    # Stack all runs: [n_runs, n_steps, n_params]
    # Align to minimum number of steps
    min_steps = min(e.shape[0] for e in all_eigenvalues)
    all_eigenvalues = [e[:min_steps] for e in all_eigenvalues]

    # Average across all steps for each run: [n_runs, n_params]
    avg_eigenvalues_per_run = [np.mean(e, axis=0) for e in all_eigenvalues]
    avg_eigenvalues = np.stack(avg_eigenvalues_per_run, axis=0)  # [n_runs, n_params]

    # Compute mean and std across runs
    mean_eigenvalues = np.mean(avg_eigenvalues, axis=0)  # [n_params]
    std_eigenvalues = np.std(avg_eigenvalues, axis=0)    # [n_params]

    n_params = len(mean_eigenvalues)

    # Extract left and right parts
    left_indices = np.arange(n_left)
    right_indices = np.arange(n_params - n_right, n_params)

    mean_left = mean_eigenvalues[left_indices]
    std_left = std_eigenvalues[left_indices]
    mean_right = mean_eigenvalues[right_indices]
    std_right = std_eigenvalues[right_indices]

    # Create X axis with gap
    x_left = np.arange(n_left)
    x_right = np.arange(n_left + 1, n_left + 1 + n_right)  # Gap of 1

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot left part (negative eigenvalues)
    for i in range(n_left):
        mean_val = mean_left[i]
        std_val = std_left[i]

        if mean_val < 0:
            # Fill downward: [mean, mean - std]
            lower = mean_val - std_val
            upper = mean_val
        else:
            # Fill upward: [mean, mean + std]
            lower = mean_val
            upper = mean_val + std_val

        ax.fill_between([x_left[i] - 0.4, x_left[i] + 0.4],
                        [lower, lower], [upper, upper],
                        alpha=0.3, color='red')
        ax.plot([x_left[i] - 0.4, x_left[i] + 0.4], [mean_val, mean_val],
                linewidth=2, color='darkred')

    # Plot right part (positive eigenvalues)
    for i in range(n_right):
        mean_val = mean_right[i]
        std_val = std_right[i]

        if mean_val < 0:
            # Fill downward: [mean, mean - std]
            lower = mean_val - std_val
            upper = mean_val
        else:
            # Fill upward: [mean, mean + std]
            lower = mean_val
            upper = mean_val + std_val

        ax.fill_between([x_right[i] - 0.4, x_right[i] + 0.4],
                        [lower, lower], [upper, upper],
                        alpha=0.3, color='blue')
        ax.plot([x_right[i] - 0.4, x_right[i] + 0.4], [mean_val, mean_val],
                linewidth=2, color='darkblue')

    # Add vertical dashed line at the gap
    gap_x = n_left + 0.5
    y_min, y_max = ax.get_ylim()
    ax.axvline(x=gap_x, color='gray', linestyle='--', linewidth=2, alpha=0.7)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Labels
    ax.set_xlabel("Eigenvalue Index (sorted)", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_title(title + f" ({len(run_dirs)} runs, averaged over steps)", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Custom x-axis labels
    x_ticks = list(x_left[::max(1, n_left//5)]) + list(x_right[::max(1, n_right//5)])
    x_labels = [str(i+1) for i in left_indices[::max(1, n_left//5)]] + \
               [str(i+1) for i in right_indices[::max(1, n_right//5)]]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(run_dirs[0]) / "hessian_spectrum.png"

    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"[saved] {save_path}")

    plt.close(fig)
