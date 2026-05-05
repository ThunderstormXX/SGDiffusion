#!/usr/bin/env python3
"""Generate publication-style figures from saved exp6 artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _load_optional_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def setup_text(result_dir: Path) -> str:
    cfg = _load_optional_yaml(result_dir / "config.yaml")
    metrics = _load_optional_json(result_dir / "metrics.json")
    lines = [
        "SETUP",
        f"exp: {cfg.get('experiment_id', result_dir.name)}",
        f"result: {cfg.get('result_name', result_dir.name)}",
        f"seed: {cfg.get('seed', 'NA')}",
        f"deterministic: {cfg.get('deterministic', 'NA')}",
    ]
    if cfg.get("device"):
        lines.append(f"device: {cfg.get('device')}")

    params = cfg.get("parameters", {})
    if params:
        lines += [
            "",
            "toy dynamics:",
            f"n_runs: {params.get('n_runs', 'NA')}",
            f"steps: {params.get('steps', 'NA')}",
            f"eta/lr: {params.get('lr', 'NA')}",
            f"lambda: {params.get('lambda', 'NA')}",
            f"noise_std: {params.get('noise_std', 'NA')}",
            f"x0: {params.get('x0', 'NA')}",
            "dt convention: one SGD step",
        ]

    dataset = cfg.get("dataset", {})
    training = cfg.get("training", {})
    ensemble = cfg.get("ensemble", {})
    analysis = cfg.get("analysis", {})
    if dataset:
        sample_size = int(dataset.get("sample_size", 0) or 0)
        batch_size = int(training.get("batch_size", 0) or 0)
        epoch_steps = sample_size // batch_size if sample_size and batch_size else None
        steps = ensemble.get("steps")
        epochs = float(steps) / epoch_steps if steps is not None and epoch_steps else None
        model_cfg = cfg.get("model", {})
        model_name = "CNN" if model_cfg.get("type") == "cnn" else "MLP-386"
        lines += [
            "",
            "data/model:",
            f"dataset: {dataset.get('name', 'NA')}",
            f"sample/val: {dataset.get('sample_size', 'NA')}/{dataset.get('val_size', 'NA')}",
            f"subset: {dataset.get('subset', 'NA')}",
            f"normalize: {dataset.get('normalize', 'NA')}",
            f"model: {model_name}",
            "",
            "training:",
            f"batch: {training.get('batch_size', 'NA')}",
            f"replacement ref: {training.get('replacement', 'NA')}",
            f"ref lr/steps: {training.get('lr', 'NA')}/{training.get('reference_steps', 'NA')}",
            "",
            "ensemble:",
            f"n_runs: {ensemble.get('n_runs', 'NA')}",
            f"steps: {steps}",
            f"epochs: {_fmt(epochs) if epochs is not None else 'NA'}",
            f"eta/lr: {ensemble.get('lr', 'NA')}",
            f"methods: {','.join(ensemble.get('methods', []))[:38]}",
        ]
        if "langevin_noise_batches" in ensemble:
            lines.append(f"noise batches: {ensemble['langevin_noise_batches']}")
        if analysis:
            lines.append(f"dirs: {analysis.get('n_directions', 'NA')}")

    metric_keys = [
        "pass",
        "wasserstein_standard_to_sgd",
        "wasserstein_modified_to_sgd",
        "replacement_vs_no_replacement_wasserstein",
        "relative_error",
        "log_correlation",
        "pearson_displacement_test_loss",
        "spearman_displacement_test_loss",
        "fit_improvement",
        "flat_variance_reduction_fraction",
        "suppress_flat_test_loss_delta",
        "theory_high_variance_reduction_fraction",
        "suppress_theory_high_variance_test_loss_delta",
    ]
    present = [(k, metrics[k]) for k in metric_keys if k in metrics]
    if present:
        lines += ["", "key metrics:"]
        for k, v in present[:7]:
            short = k.replace("_to_sgd", "").replace("_displacement_test_loss", "")
            lines.append(f"{short}: {_fmt(v)}")
    return "\n".join(lines)


def draw_setup_box(ax: plt.Axes, text: str) -> None:
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        text,
        ha="left",
        va="top",
        family="monospace",
        fontsize=7.2,
        linespacing=1.18,
        bbox={"boxstyle": "square,pad=0.45", "facecolor": "#f7f7f7", "edgecolor": "#444444", "linewidth": 0.8},
    )


def make_exp7_control(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    loss_rows = read_rows(result_dir / "loss_data.csv")
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(19.0, 5.4),
        gridspec_kw={"width_ratios": [1.15, 1.45, 1.45, 1.45]},
    )
    draw_setup_box(axes[0], setup)

    def plot_variance(ax: plt.Axes, suffix: str, label: str) -> None:
        for method in sorted({r["method"] for r in rows if r["method"].endswith(suffix)}):
            rr = [r for r in rows if r["method"] == method]
            base = method.removesuffix(suffix).removesuffix("_")
            x = np.array([float(r["step"]) for r in rr])
            y = np.array([float(r["mean_direction_variance"]) for r in rr])
            ax.plot(x, y, label=base)
        ax.set_xlabel("step")
        ax.set_ylabel("variance")
        ax.set_title(label)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    plot_variance(axes[1], "_theory_high_variance", "theory-high-variance subspace")
    plot_variance(axes[2], "_sharp", "sharp eigenspace")

    for split, style in [("train", "-"), ("test", "--")]:
        for method in sorted({r["method"] for r in loss_rows}):
            rr = [r for r in loss_rows if r["method"] == method and r["split"] == split]
            if not rr:
                continue
            x = np.array([float(r["step"]) for r in rr])
            y = np.array([float(r["loss_mean"]) for r in rr])
            axes[3].plot(x, y, style, label=f"{method} {split}")
    axes[3].set_xlabel("step")
    axes[3].set_ylabel("loss")
    axes[3].set_title("train/test loss")
    axes[3].grid(alpha=0.25)
    axes[3].legend(fontsize=6)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_exp8_amplification(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    loss_rows = read_rows(result_dir / "loss_data.csv")
    metrics = _load_optional_json(result_dir / "metrics.json")
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22.0, 5.4),
        gridspec_kw={"width_ratios": [1.1, 1.25, 1.25, 1.25, 1.25]},
    )
    draw_setup_box(axes[0], setup)

    methods = sorted({r["method"] for r in loss_rows})
    for split, ax, ylabel in [("train", axes[1], "train loss"), ("test", axes[2], "test loss")]:
        for method in methods:
            rr = [r for r in loss_rows if r["method"] == method and r["split"] == split]
            if not rr:
                continue
            x = np.array([float(r["step"]) for r in rr])
            y = np.array([float(r["loss_mean"]) for r in rr])
            ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.2, label=method)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)

    deltas = []
    labels = []
    for method in methods:
        key = f"{method}_test_loss_mean"
        if method != "baseline_sgd" and key in metrics and "baseline_sgd_test_loss_mean" in metrics:
            labels.append(method.replace("amplify_", "amp_").replace("_high_variance", "_high_var"))
            deltas.append(float(metrics[key]) - float(metrics["baseline_sgd_test_loss_mean"]))
    axes[3].axhline(0.0, color="k", linewidth=0.8)
    axes[3].bar(np.arange(len(deltas)), deltas)
    axes[3].set_xticks(np.arange(len(deltas)), labels, rotation=25, ha="right", fontsize=7)
    axes[3].set_ylabel("final test loss delta")
    axes[3].set_title("final delta vs baseline")
    axes[3].grid(axis="y", alpha=0.25)

    for method in sorted({r["method"] for r in rows if r["method"].endswith("_theory_high_variance")}):
        rr = [r for r in rows if r["method"] == method]
        base = method.removesuffix("_theory_high_variance")
        x = np.array([float(r["step"]) for r in rr])
        y = np.array([float(r["mean_direction_variance"]) for r in rr])
        axes[4].plot(x, y, label=base)
    axes[4].set_xlabel("step")
    axes[4].set_ylabel("variance")
    axes[4].set_title("theory-selected variance")
    axes[4].set_yscale("log")
    axes[4].grid(alpha=0.25)
    axes[4].legend(fontsize=6)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_alpha_sweep(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    loss_rows = read_rows(result_dir / "loss_data.csv")
    final_rows = read_rows(result_dir / "final_alpha_sweep.csv")
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(23.0, 5.4),
        gridspec_kw={"width_ratios": [1.1, 1.45, 1.45, 1.25, 1.25]},
    )
    draw_setup_box(axes[0], setup)
    alphas = sorted({float(r["alpha"]) for r in loss_rows})
    positive = [a for a in alphas if a > 0]
    alpha_floor = min(positive) / 3.0 if positive else 1e-4
    alpha_plot = {a: (a if a > 0 else alpha_floor) for a in alphas}
    cmap = plt.get_cmap("viridis")
    color_for = {a: cmap(i / max(len(alphas) - 1, 1)) for i, a in enumerate(alphas)}

    for split, ax, ylabel in [("train", axes[1], "train loss"), ("test", axes[2], "test loss")]:
        for alpha in alphas:
            rr = [r for r in loss_rows if float(r["alpha"]) == alpha and r["split"] == split]
            if not rr:
                continue
            x = np.array([float(r["step"]) for r in rr])
            y = np.array([float(r["loss_mean"]) for r in rr])
            lo = np.array([float(r.get("loss_ci95_low", r["loss_mean"])) for r in rr])
            hi = np.array([float(r.get("loss_ci95_high", r["loss_mean"])) for r in rr])
            ax.plot(x, y, color=color_for[alpha], linewidth=1.15, label=f"{alpha:g}")
            if np.nanmax(hi - lo) > 0:
                ax.fill_between(x, lo, hi, color=color_for[alpha], alpha=0.12, linewidth=0)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[2].legend(title="alpha", fontsize=6, title_fontsize=7, ncols=2)

    x_raw = np.array([float(r["alpha"]) for r in final_rows])
    x = np.array([alpha_plot[float(r["alpha"])] for r in final_rows])
    train_key = "final_train_loss" if "final_train_loss" in final_rows[0] else "train_loss_mean"
    test_key = "final_test_loss" if "final_test_loss" in final_rows[0] else "test_loss_mean"
    train_acc_key = "final_train_accuracy" if "final_train_accuracy" in final_rows[0] else "train_accuracy_mean"
    test_acc_key = "final_test_accuracy" if "final_test_accuracy" in final_rows[0] else "test_accuracy_mean"
    train = np.array([float(r[train_key]) for r in final_rows])
    test = np.array([float(r[test_key]) for r in final_rows])
    train_acc = np.array([float(r.get(train_acc_key, "nan")) for r in final_rows])
    test_acc = np.array([float(r.get(test_acc_key, "nan")) for r in final_rows])
    train_err = None
    test_err = None
    if "train_loss_ci95_low" in final_rows[0]:
        train_err = np.vstack([
            train - np.array([float(r["train_loss_ci95_low"]) for r in final_rows]),
            np.array([float(r["train_loss_ci95_high"]) for r in final_rows]) - train,
        ])
        test_err = np.vstack([
            test - np.array([float(r["test_loss_ci95_low"]) for r in final_rows]),
            np.array([float(r["test_loss_ci95_high"]) for r in final_rows]) - test,
        ])
    axes[3].errorbar(x, train, yerr=train_err, fmt="o-", capsize=2, label="train")
    axes[3].errorbar(x, test, yerr=test_err, fmt="o-", capsize=2, label="test")
    axes[3].axvline(1.0, color="k", linestyle="--", linewidth=0.8)
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("alpha")
    axes[3].set_ylabel("final loss")
    axes[3].set_title("final loss vs alpha")
    axes[3].grid(alpha=0.25)
    axes[3].legend(fontsize=7)
    if 0.0 in x_raw:
        axes[3].text(alpha_floor, np.nanmax(test), "0", ha="center", va="bottom", fontsize=7)

    if (result_dir / "trajectory_data.csv").exists():
        traj_rows = read_rows(result_dir / "trajectory_data.csv")
        for alpha in alphas:
            rr = [r for r in traj_rows if float(r["alpha"]) == alpha]
            if not rr:
                continue
            tx = np.array([float(r["step"]) for r in rr])
            y_field = "normalized_variance" if "normalized_variance" in rr[0] else "parameter_variance_trace"
            vy = np.array([float(r[y_field]) for r in rr])
            axes[4].plot(tx, vy + 1e-18, color=color_for[alpha], linewidth=1.15, label=f"{alpha:g}")
        axes[4].set_xlabel("step")
        axes[4].set_ylabel("variance / alpha=1 variance" if y_field == "normalized_variance" else "parameter variance trace")
        axes[4].set_title("normalized displacement variance" if y_field == "normalized_variance" else "ensemble displacement variance")
        axes[4].set_yscale("log")
        axes[4].grid(alpha=0.25)
        axes[4].legend(title="alpha", fontsize=6, title_fontsize=7, ncols=2)
    else:
        train_acc_err = None
        test_acc_err = None
        if "train_accuracy_ci95_low" in final_rows[0]:
            train_acc_err = np.vstack([
                train_acc - np.array([float(r["train_accuracy_ci95_low"]) for r in final_rows]),
                np.array([float(r["train_accuracy_ci95_high"]) for r in final_rows]) - train_acc,
            ])
            test_acc_err = np.vstack([
                test_acc - np.array([float(r["test_accuracy_ci95_low"]) for r in final_rows]),
                np.array([float(r["test_accuracy_ci95_high"]) for r in final_rows]) - test_acc,
            ])
        axes[4].errorbar(x, train_acc, yerr=train_acc_err, fmt="o-", capsize=2, label="train")
        axes[4].errorbar(x, test_acc, yerr=test_acc_err, fmt="o-", capsize=2, label="test")
        axes[4].axvline(1.0, color="k", linestyle="--", linewidth=0.8)
        axes[4].set_xscale("log")
        axes[4].set_xlabel("alpha")
        axes[4].set_ylabel("accuracy")
        axes[4].set_title("final accuracy")
        axes[4].set_ylim(0.0, 1.0)
        axes[4].grid(alpha=0.25)
        axes[4].legend(fontsize=7)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_time_variance(rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    methods = sorted({r["method"] for r in rows if "method" in r})
    keys = rows[0].keys()
    has_mean_error = "mean_path_error_to_sgd" in keys
    has_2d_path = {"mean_coord_1", "mean_coord_2"}.issubset(keys)
    nplots = 1 + int(has_mean_error) + int(has_2d_path)
    fig, axes = plt.subplots(1, nplots + 1, figsize=(3.7 + 5.0 * nplots, 5.2), gridspec_kw={"width_ratios": [1.1] + [1.55] * nplots})
    draw_setup_box(axes[0], setup)
    ax = axes[1]
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        x = np.array([float(r["step"]) for r in rr])
        y_key = "variance" if "variance" in rr[0] else "mean_direction_variance"
        y = np.array([float(r[y_key]) for r in rr])
        ax.plot(x, y, label=method)
    ax.set_xlabel("step")
    ax.set_ylabel("variance")
    ax.set_title("ensemble variance")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    idx = 1
    if has_mean_error:
        ax = axes[idx + 1]
        idx += 1
        for method in methods:
            rr = [r for r in rows if r["method"] == method]
            x = np.array([float(r["step"]) for r in rr])
            y = np.array([float(r["mean_path_error_to_sgd"]) for r in rr])
            ax.plot(x, y, label=method)
        ax.set_xlabel("step")
        ax.set_ylabel("distance")
        ax.set_title("mean path error to SGD")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    if has_2d_path:
        ax = axes[idx + 1]
        for method in methods:
            rr = [r for r in rows if r["method"] == method]
            x = np.array([float(r["mean_coord_1"]) for r in rr])
            y = np.array([float(r["mean_coord_2"]) for r in rr])
            ax.plot(x, y, marker="o", markersize=2, linewidth=1.2, label=method)
            ax.plot(x[0], y[0], "k.", markersize=4)
        ax.set_xlabel("mean projection 1")
        ax.set_ylabel("mean projection 2")
        ax.set_title("mean trajectory")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_exp12_langevin_verification(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    one_rows = read_rows(result_dir / "one_step_matching.csv")
    horizon_rows = read_rows(result_dir / "horizon_error.csv")
    cal_rows = read_rows(result_dir / "calibration_sweep.csv")
    audit = _load_optional_json(result_dir / "implementation_audit.json")
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22.0, 5.4),
        gridspec_kw={"width_ratios": [1.1, 1.2, 1.35, 1.35, 1.1]},
    )
    draw_setup_box(axes[0], setup)

    labels = [r["method"].replace("_langevin", "") for r in one_rows]
    mean_err = np.array([float(r["mean_update_error"]) for r in one_rows])
    cov_err = np.array([float(r["covariance_error"]) for r in one_rows])
    x = np.arange(len(labels))
    axes[1].bar(x - 0.18, mean_err, width=0.36, label="mean")
    axes[1].bar(x + 0.18, cov_err, width=0.36, label="cov")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_title("one-step matching")
    axes[1].set_ylabel("relative error")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=7)

    for method in sorted({r["method"] for r in horizon_rows if not r["method"].startswith("modified_c_")}):
        rr = [r for r in horizon_rows if r["method"] == method]
        hx = np.array([float(r["horizon"]) for r in rr])
        hy = np.array([float(r["mean_trajectory_error"]) for r in rr])
        axes[2].plot(hx, hy, "o-", label=method.replace("_langevin", ""))
    axes[2].set_xlabel("horizon")
    axes[2].set_ylabel("mean path error")
    axes[2].set_title("error vs horizon")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=7)

    for method in sorted({r["method"] for r in horizon_rows if not r["method"].startswith("modified_c_")}):
        rr = [r for r in horizon_rows if r["method"] == method]
        hx = np.array([float(r["horizon"]) for r in rr])
        hy = np.array([float(r["variance_error"]) for r in rr])
        axes[3].plot(hx, hy, "o-", label=method.replace("_langevin", ""))
    axes[3].set_xlabel("horizon")
    axes[3].set_ylabel("variance error")
    axes[3].set_title("variance mismatch")
    axes[3].set_yscale("log")
    axes[3].grid(alpha=0.25)
    axes[3].legend(fontsize=7)

    cx = np.array([float(r["calibration"]) for r in cal_rows])
    cy = np.array([float(r["covariance_error"]) for r in cal_rows])
    axes[4].plot(cx, cy, "o-")
    axes[4].axvline(1.0, color="k", linestyle="--", linewidth=0.8)
    axes[4].set_xlabel("modified noise multiplier")
    axes[4].set_ylabel("one-step cov error")
    axes[4].set_title("calibration sweep")
    axes[4].set_yscale("log")
    axes[4].grid(alpha=0.25)
    ratio = audit.get("standard_noise_to_drift_ratio")
    if ratio is not None:
        axes[4].text(0.02, 0.02, f"noise/drift={ratio:.3g}", transform=axes[4].transAxes, fontsize=7)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_scatter(rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    keys = rows[0].keys()
    if {"flat_displacement", "sharp_displacement", "generalization_gap", "final_test_loss"}.issubset(keys):
        fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.2), gridspec_kw={"width_ratios": [1.1, 1.35, 1.35, 1.35]})
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.2), gridspec_kw={"width_ratios": [1.15, 1.7]})
    draw_setup_box(axes[0], setup)
    ax = axes[1]

    def scatter_fit(axis: plt.Axes, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, panel_title: str) -> None:
        axis.plot(x, y, "o", markersize=4, alpha=0.75)
        if len(x) >= 2 and np.std(x) > 0:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            axis.plot(xx, coef[0] * xx + coef[1], "k--", linewidth=1)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(panel_title)
        axis.grid(alpha=0.25)

    if {"measured_variance", "predicted_variance"}.issubset(keys):
        x = np.array([float(r["predicted_variance"]) for r in rows])
        y = np.array([float(r["measured_variance"]) for r in rows])
        ax.loglog(x + 1e-12, y + 1e-12, "o")
        lo, hi = min(x.min(), y.min()) + 1e-12, max(x.max(), y.max()) + 1e-12
        ax.loglog([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("predicted variance")
        ax.set_ylabel("measured variance")
    elif {"flat_displacement", "sharp_displacement", "generalization_gap", "final_test_loss"}.issubset(keys):
        x = np.array([float(r["flat_displacement"]) for r in rows])
        sharp = np.array([float(r["sharp_displacement"]) for r in rows])
        gap = np.array([float(r["generalization_gap"]) for r in rows])
        test = np.array([float(r["final_test_loss"]) for r in rows])
        scatter_fit(ax, x, gap, "flat eigenspace displacement", "generalization gap", "flat vs gap")
        scatter_fit(axes[2], sharp, gap, "sharp eigenspace displacement", "generalization gap", "sharp vs gap")
        scatter_fit(axes[3], x, test, "flat eigenspace displacement", "final test loss", "flat vs test")
    elif {"flat_displacement", "final_test_loss"}.issubset(keys):
        x = np.array([float(r["flat_displacement"]) for r in rows])
        y = np.array([float(r["final_test_loss"]) for r in rows])
        scatter_fit(ax, x, y, "flat-direction displacement", "final test loss", title)
    else:
        ax.text(0.5, 0.5, "No plottable data", ha="center", va="center")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    rows = read_rows(result_dir / "figure_data.csv")
    if not rows:
        return
    out = result_dir / "figure.png"
    title = result_dir.name
    setup = setup_text(result_dir)
    if result_dir.name.startswith("exp12_") and (result_dir / "one_step_matching.csv").exists():
        make_exp12_langevin_verification(result_dir, rows, out, title, setup)
    elif (result_dir.name.startswith("exp10_") or result_dir.name.startswith("exp11_")) and (result_dir / "loss_data.csv").exists():
        make_alpha_sweep(result_dir, rows, out, title, setup)
    elif (result_dir.name.startswith("exp8_") or result_dir.name.startswith("exp9_")) and (result_dir / "loss_data.csv").exists():
        make_exp8_amplification(result_dir, rows, out, title, setup)
    elif (result_dir / "loss_data.csv").exists() and any("_theory_high_variance" in r.get("method", "") for r in rows):
        make_exp7_control(result_dir, rows, out, title, setup)
    elif "method" in rows[0] and "step" in rows[0]:
        make_time_variance(rows, out, title, setup)
    else:
        make_scatter(rows, out, title, setup)
    if result_dir.parent.name == "results":
        figures_dir = result_dir.parent.parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        top_level = figures_dir / f"{result_dir.name}.png"
        top_level.write_bytes(out.read_bytes())
        print(f"[saved] {top_level}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
