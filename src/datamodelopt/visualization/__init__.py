"""
Visualization utilities for experiment results.
"""

from .plots import (
    # Many runs visualizations
    get_percentile_weight_indices,
    load_many_runs_weights,
    plot_accuracy_curves,
    plot_combined_stages,
    plot_gradient_norms,
    plot_hessian_spectrum,
    plot_loss_and_accuracy,
    plot_loss_curves,
    plot_weight_trajectories_both_normalizations,
    plot_weight_trajectories_combined,
    plot_weight_trajectories_percentiles,
    plot_weight_trajectory_2d,
)
from .runner import ManyRunsVisualizer, VisualizationRunner

__all__ = [
    "plot_loss_curves",
    "plot_accuracy_curves",
    "plot_loss_and_accuracy",
    "plot_weight_trajectory_2d",
    "plot_gradient_norms",
    "plot_combined_stages",
    # Many runs
    "get_percentile_weight_indices",
    "load_many_runs_weights",
    "plot_weight_trajectories_percentiles",
    "plot_weight_trajectories_combined",
    "plot_weight_trajectories_both_normalizations",
    "plot_hessian_spectrum",
    "VisualizationRunner",
    "ManyRunsVisualizer",
]
