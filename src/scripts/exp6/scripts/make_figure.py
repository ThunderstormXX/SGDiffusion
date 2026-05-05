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
            f"lr eta: {params.get('lr', 'NA')}",
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
        model_type = model_cfg.get("type", "mlp")
        if model_type == "cnn":
            model_name = "CNN"
        elif model_type == "nanogpt":
            model_name = "NanoGPT-960"
        else:
            model_name = "MLP-386"
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
            f"lr eta: {ensemble.get('lr', 'NA')}",
            f"Langevin M: {ensemble.get('langevin_substeps', 1)}",
            f"coeff update: {ensemble.get('langevin_coefficient_update_every', 1)}",
            f"drift mode: {ensemble.get('langevin_drift_mode', 'current')}",
            f"noise mode: {ensemble.get('langevin_noise_mode', 'current')}",
            f"methods: {','.join(ensemble.get('methods', []))[:38]}",
        ]
        if ensemble.get("langevin_substeps"):
            try:
                dt = float(ensemble.get("lr")) / float(ensemble.get("langevin_substeps", 1))
                lines.append(f"Langevin dt: {_fmt(dt)}")
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        if "langevin_noise_batches" in ensemble:
            lines.append(f"noise batches: {ensemble['langevin_noise_batches']}")
        if analysis:
            lines.append(f"dirs: {analysis.get('n_directions', 'NA')}")
            if "gradient_batches" in analysis:
                lines.append(f"grad batches: {analysis.get('gradient_batches')}")
            if "hessian_noise_batches" in analysis:
                lines.append(f"hess noise batches: {analysis.get('hessian_noise_batches')}")

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


def make_exp16_evidence(rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18.5, 5.2),
        gridspec_kw={"width_ratios": [1.1, 1.35, 1.35, 1.15]},
    )
    draw_setup_box(axes[0], setup)
    methods = ["standard_langevin", "modified_langevin"]
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        steps = sorted({int(r["step"]) for r in rr})
        means = []
        for step in steps:
            vals = [float(r["cumulative_log_evidence"]) for r in rr if int(r["step"]) == step]
            means.append(np.mean(vals))
        axes[1].plot(steps, means, label=method.replace("_langevin", ""))
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("cumulative log-evidence")
    axes[1].set_title("trajectory evidence")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    rr = [r for r in rows if r["method"] == "standard_langevin"]
    steps = sorted({int(r["step"]) for r in rr})
    direct_mean, formula_mean = [], []
    for step in steps:
        vals_d = [float(r["cumulative_log_ratio"]) for r in rr if int(r["step"]) == step]
        vals_f = [float(r["cumulative_log_ratio_formula"]) for r in rr if int(r["step"]) == step]
        direct_mean.append(np.mean(vals_d))
        formula_mean.append(np.mean(vals_f))
    axes[2].plot(steps, direct_mean, label="direct density ratio")
    axes[2].plot(steps, formula_mean, "--", label="closed-form formula")
    axes[2].axhline(0.0, color="k", linewidth=0.8)
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("cumulative log-ratio")
    axes[2].set_title("modified - standard")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    errors = []
    for step in steps:
        vals = [float(r["formula_error"]) for r in rr if int(r["step"]) == step]
        errors.append(np.max(vals))
    axes[3].plot(steps, errors)
    axes[3].set_yscale("log")
    axes[3].set_xlabel("iteration")
    axes[3].set_ylabel("max formula error")
    axes[3].set_title("formula check")
    axes[3].grid(alpha=0.25)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_exp17_covariance(rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18.5, 5.2),
        gridspec_kw={"width_ratios": [1.05, 1.35, 1.35, 1.35]},
    )
    draw_setup_box(axes[0], setup)
    label_map = {
        "empirical_sgd": "empirical SGD ensemble",
        "standard_fp": "standard FP (no drift-square)",
        "discrete_fp": r"discrete FP (+$\eta^2 H\Pi H$)",
    }
    methods = [m for m in ["empirical_sgd", "standard_fp", "discrete_fp"] if any(r["method"] == m for r in rows)]
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        x = np.array([float(r["step"]) for r in rr])
        x_plot = np.maximum(x, 1.0)
        y = np.array([float(r["covariance_trace"]) for r in rr])
        traces[method] = (x_plot, y)
        style = {"empirical_sgd": "k-", "standard_fp": "r--", "discrete_fp": "b:"}.get(method, "-")
        lw = 2.2 if method == "empirical_sgd" else 2.0
        axes[1].loglog(x_plot, np.maximum(y, 1e-18), style, linewidth=lw, label=label_map.get(method, method))
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("trace covariance")
    axes[1].set_title("absolute trace (discrete overlaps empirical)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    if "empirical_sgd" in traces:
        _, emp = traces["empirical_sgd"]
        for method in ["standard_fp", "discrete_fp"]:
            if method not in traces:
                continue
            x, y = traces[method]
            rel = (y - emp) / np.maximum(emp, 1e-18)
            axes[2].semilogx(x, rel, {"standard_fp": "r--", "discrete_fp": "b:"}[method], linewidth=2.0, label=label_map.get(method, method))
        axes[2].axhline(0.0, color="k", linewidth=0.8)
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("(prediction - empirical) / empirical")
    axes[2].set_title("trace residual makes overlap visible")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    for method in ["standard_fp", "discrete_fp"]:
        rr = [r for r in rows if r["method"] == method]
        x = np.array([float(r["step"]) for r in rr])
        x_plot = np.maximum(x, 1.0)
        y = np.array([float(r["relative_frobenius_error"]) for r in rr])
        axes[3].loglog(x_plot, np.maximum(y, 1e-18), {"standard_fp": "r--", "discrete_fp": "b:"}[method], linewidth=2.0, label=label_map.get(method, method))
    axes[3].set_xlabel("iteration")
    axes[3].set_ylabel("relative Frobenius error")
    axes[3].set_title("covariance error to empirical")
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].grid(alpha=0.25)
    axes[3].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def _npz(result_dir: Path):
    return np.load(result_dir / "raw_outputs.npz")


def _plot_loglog_pair(ax: plt.Axes, x: np.ndarray, empirical: np.ndarray, prediction: np.ndarray, empirical_label: str, prediction_label: str) -> None:
    ax.loglog(x, np.maximum(empirical, 1e-18), "o-", linewidth=1.4, markersize=4, label=empirical_label)
    ax.loglog(x, np.maximum(prediction, 1e-18), "--", linewidth=1.6, label=prediction_label)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7)


def make_eta_scaling(result_dir: Path, out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    etas = data["etas"]
    emp_centered = data["empirical_centered"]
    emp_raw = data["empirical_raw"]
    pred_centered = data["pred_centered"]
    pred_raw = data["pred_raw"]
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 5.2), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2, 1.25]})
    draw_setup_box(axes[0], setup)
    _plot_loglog_pair(axes[1], etas, emp_centered, pred_centered, "empirical centered covariance", r"$\eta^2 \mathrm{tr}\Sigma$")
    axes[1].set_xlabel(r"$\eta$")
    axes[1].set_ylabel("centered second moment")
    axes[1].set_title("centered noise scales as eta^2")

    _plot_loglog_pair(axes[2], etas, emp_raw, pred_raw, "empirical raw second moment", r"$\eta^2 \mathrm{tr}(\Sigma+gg^T)$")
    axes[2].set_xlabel(r"$\eta$")
    axes[2].set_ylabel("raw second moment")
    axes[2].set_title("raw moment includes drift-square")

    axes[3].semilogx(etas, emp_raw / np.maximum(emp_centered, 1e-18), "o-", label="raw / centered")
    axes[3].semilogx(etas, pred_raw / np.maximum(pred_centered, 1e-18), "--", label="prediction")
    axes[3].set_xlabel(r"$\eta$")
    axes[3].set_ylabel("ratio")
    axes[3].set_title("visible ggT contribution")
    axes[3].grid(alpha=0.25, which="both")
    axes[3].legend(fontsize=7)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_exp20_anisotropy(result_dir: Path, out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    par_raw = float(np.mean(data["par_raw"]))
    orth_raw = float(np.mean(data["orth_raw"]))
    par_centered = float(np.mean(data["par_centered"]))
    orth_centered = float(np.mean(data["orth_centered"]))
    raw_vals = np.array([par_raw, orth_raw])
    centered_vals = np.array([par_centered, orth_centered])
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.2), gridspec_kw={"width_ratios": [1.05, 1.1, 1.1, 1.15]})
    draw_setup_box(axes[0], setup)
    labels = ["parallel to g", "orthogonal mean"]
    x = np.arange(2)
    axes[1].bar(x, raw_vals, color=["tab:red", "tab:blue"])
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("raw second moment")
    axes[1].set_title(r"raw moment: $\Sigma + gg^T$")
    axes[1].grid(axis="y", alpha=0.25, which="both")

    axes[2].bar(x, centered_vals, color=["tab:red", "tab:blue"])
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("centered covariance")
    axes[2].set_title(r"centered covariance: $\Sigma$")
    axes[2].grid(axis="y", alpha=0.25, which="both")

    ratios = np.array([par_raw / max(orth_raw, 1e-18), par_centered / max(orth_centered, 1e-18)])
    axes[3].bar(np.arange(2), ratios, color=["tab:purple", "tab:gray"])
    axes[3].axhline(1.0, color="k", linewidth=0.8)
    axes[3].set_xticks(np.arange(2), ["raw", "centered"])
    axes[3].set_yscale("log")
    axes[3].set_ylabel("parallel / orthogonal")
    axes[3].set_title("anisotropy diagnostic")
    axes[3].grid(axis="y", alpha=0.25, which="both")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_batch_scaling(result_dir: Path, out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    batch = data["batch_sizes"]
    emp_centered = data["empirical_centered"]
    emp_raw = data["empirical_raw"]
    pred_centered = data["pred_centered"]
    pred_raw = data["pred_raw"]
    raw_floor = float(np.min(pred_raw))
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 5.2), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2, 1.25]})
    draw_setup_box(axes[0], setup)
    _plot_loglog_pair(axes[1], batch, emp_centered, pred_centered, "empirical centered", r"$\eta^2 \mathrm{tr}\Sigma/B$")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("batch size B")
    axes[1].set_ylabel("centered covariance")
    axes[1].set_title("centered noise falls as 1/B")

    _plot_loglog_pair(axes[2], batch, emp_raw, pred_raw, "empirical raw", r"$\eta^2(\mathrm{tr}\Sigma/B+\|g\|^2)$")
    axes[2].invert_xaxis()
    axes[2].set_xlabel("batch size B")
    axes[2].set_ylabel("raw second moment")
    axes[2].set_title("raw moment has nonzero floor")

    axes[3].loglog(batch, np.maximum(emp_raw - raw_floor, 1e-18), "o-", label="empirical raw minus floor")
    axes[3].loglog(batch, np.maximum(pred_raw - raw_floor, 1e-18), "--", label="prediction minus floor")
    axes[3].invert_xaxis()
    axes[3].set_xlabel("batch size B")
    axes[3].set_ylabel("excess over drift floor")
    axes[3].set_title("1/B part made visible")
    axes[3].grid(alpha=0.25, which="both")
    axes[3].legend(fontsize=7)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_exp35_sampling(result_dir: Path, out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    wr = data["trace_with_replacement"]
    wor = data["trace_without_replacement"]
    steps = np.arange(len(wr))
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.2), gridspec_kw={"width_ratios": [1.05, 1.35, 1.35]})
    draw_setup_box(axes[0], setup)
    axes[1].semilogy(steps, np.maximum(wr, 1e-18), label="with replacement")
    axes[1].semilogy(steps, np.maximum(wor, 1e-18), label="without replacement")
    axes[1].set_xlabel("record index")
    axes[1].set_ylabel("covariance trace")
    axes[1].set_title("sampling changes covariance")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=7)
    axes[2].semilogy(steps, np.maximum(wor / np.maximum(wr, 1e-18), 1e-18))
    axes[2].axhline(1.0, color="k", linewidth=0.8)
    axes[2].set_xlabel("record index")
    axes[2].set_ylabel("without / with")
    axes[2].set_title("variance suppression ratio")
    axes[2].grid(alpha=0.25, which="both")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def _save_clean_copy(fig_fn, out: Path, title: str) -> None:
    clean = out.with_name("figure_clean.png")
    fig_fn(clean, title)


def make_exp37_mean_matching(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    etas = data["etas"]
    std = data["standard_error"]
    cor = data["corrected_error"]

    def draw(path: Path, plot_title: str, with_setup: bool = False) -> None:
        if with_setup:
            fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2), gridspec_kw={"width_ratios": [1.05, 1.35, 1.35]})
            draw_setup_box(axes[0], setup)
            ax1, ax2 = axes[1], axes[2]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
            ax1, ax2 = axes
        ax1.loglog(etas, std, "o--", label="standard generator")
        ax1.loglog(etas, cor, "o-", label=r"corrected generator $-\eta Hg/2$")
        ax1.set_xlabel(r"$\eta$")
        ax1.set_ylabel("mean-map error")
        ax1.set_title("one-step mean map")
        ax1.grid(alpha=0.25, which="both")
        ax1.legend(fontsize=8)
        ax2.semilogx(etas, std / np.maximum(cor, 1e-18), "o-", color="tab:green")
        ax2.axhline(1.0, color="k", linewidth=0.8)
        ax2.set_xlabel(r"$\eta$")
        ax2.set_ylabel("standard / corrected error")
        ax2.set_title("improvement")
        ax2.grid(alpha=0.25, which="both")
        fig.suptitle(plot_title)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(path, dpi=180)
        plt.close()

    draw(out, title, with_setup=True)
    draw(out.with_name("figure_clean.png"), title, with_setup=False)


def make_exp38_moment_matching(result_dir: Path, out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    t = data["times"]
    std_m, cor_m = data["standard_mean_error"], data["corrected_mean_error"]
    std_c, cor_c = data["standard_covariance_error"], data["corrected_covariance_error"]

    def draw(path: Path, plot_title: str, with_setup: bool = False) -> None:
        if with_setup:
            fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.2), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2, 1.25]})
            draw_setup_box(axes[0], setup)
            ax1, ax2, ax3 = axes[1], axes[2], axes[3]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
            ax1, ax2, ax3 = axes
        for ax, ys, ylabel, panel_title in [
            (ax1, (std_m, cor_m), "mean error", "mean evolution"),
            (ax2, (std_c, cor_c), "relative covariance error", "covariance evolution"),
        ]:
            ax.semilogy(t, np.maximum(ys[0], 1e-18), "--", label="standard")
            ax.semilogy(t, np.maximum(ys[1], 1e-18), "-", label="corrected")
            ax.set_xlabel("macro step")
            ax.set_ylabel(ylabel)
            ax.set_title(panel_title)
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=8)
        ax3.semilogy(t, np.maximum(std_m / np.maximum(cor_m, 1e-18), 1e-18), label="mean")
        ax3.semilogy(t, np.maximum(std_c / np.maximum(cor_c, 1e-18), 1e-18), label="covariance")
        ax3.axhline(1.0, color="k", linewidth=0.8)
        ax3.set_xlabel("macro step")
        ax3.set_ylabel("standard / corrected error")
        ax3.set_title("improvement ratio")
        ax3.grid(alpha=0.25, which="both")
        ax3.legend(fontsize=8)
        fig.suptitle(plot_title)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(path, dpi=180)
        plt.close()

    draw(out, title, with_setup=True)
    draw(out.with_name("figure_clean.png"), title, with_setup=False)


def make_exp39_ablation(rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    labels = [r["method"].replace("_", "\n") for r in rows]
    mean = np.array([float(r["mean_error"]) for r in rows])
    cov = np.array([float(r["covariance_error"]) for r in rows])
    nll = np.array([float(r["negative_log_likelihood"]) for r in rows])

    def draw(path: Path, plot_title: str, with_setup: bool = False) -> None:
        if with_setup:
            fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.2), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2, 1.2]})
            draw_setup_box(axes[0], setup)
            axs = axes[1:]
        else:
            fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.6))
        for ax, vals, ylabel, panel_title in [
            (axs[0], mean, "mean error", "drift correction"),
            (axs[1], cov, "covariance error", "covariance matching"),
            (axs[2], nll, "NLL on exact SGD samples", "conditional likelihood"),
        ]:
            plot_vals = np.maximum(vals, 1e-18)
            ax.bar(np.arange(len(vals)), plot_vals)
            ax.set_xticks(np.arange(len(vals)), labels, rotation=30, ha="right", fontsize=7)
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.set_title(panel_title)
            ax.grid(axis="y", alpha=0.25, which="both")
        fig.suptitle(plot_title)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(path, dpi=180)
        plt.close()

    draw(out, title, with_setup=True)
    draw(out.with_name("figure_clean.png"), title, with_setup=False)


def make_exp40_semigroup(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    etas = data["etas"]
    std = data["standard_error"]
    cor = data["corrected_error"]
    mats = [data["G_exact"], data["G_standard"], data["G_corrected"]]
    names = ["exact log(P)/eta", "standard generator", "corrected generator"]

    def draw(path: Path, plot_title: str, with_setup: bool = False) -> None:
        if with_setup:
            fig, axes = plt.subplots(1, 5, figsize=(21.0, 5.2), gridspec_kw={"width_ratios": [1.05, 1.25, 1.0, 1.0, 1.0]})
            draw_setup_box(axes[0], setup)
            ax_err = axes[1]; heat_axes = axes[2:]
        else:
            fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.6), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0, 1.0]})
            ax_err = axes[0]; heat_axes = axes[1:]
        ax_err.loglog(etas, std, "o--", label="standard")
        ax_err.loglog(etas, cor, "o-", label="corrected")
        ax_err.set_xlabel(r"$\eta$")
        ax_err.set_ylabel("relative Frobenius error")
        ax_err.set_title("generator error")
        ax_err.grid(alpha=0.25, which="both")
        ax_err.legend(fontsize=8)
        vmax = max(float(np.max(np.abs(m))) for m in mats)
        for ax, mat, name in zip(heat_axes, mats, names):
            im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(name)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=list(heat_axes), fraction=0.025, pad=0.01)
        fig.suptitle(plot_title)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(path, dpi=180)
        plt.close()

    draw(out, title, with_setup=True)
    draw(out.with_name("figure_clean.png"), title, with_setup=False)


def make_exp41_cumulants(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    emp = data["empirical_increment"]
    gau = data["gaussian_increment"]
    labels = [r["method"].replace("_", "\n") for r in rows]
    skew = np.array([float(r["skewness"]) for r in rows])
    kurt = np.array([float(r["kurtosis"]) for r in rows])

    def draw(path: Path, plot_title: str, with_setup: bool = False) -> None:
        if with_setup:
            fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.2), gridspec_kw={"width_ratios": [1.05, 1.35, 1.0, 1.0]})
            draw_setup_box(axes[0], setup)
            axh, axs, axk = axes[1], axes[2], axes[3]
        else:
            fig, (axh, axs, axk) = plt.subplots(1, 3, figsize=(13.8, 4.6))
        axh.hist(emp, bins=80, density=True, alpha=0.55, label="SGD increment")
        axh.hist(gau, bins=80, density=True, alpha=0.45, label="Gaussian surrogate")
        axh.set_title("increment histogram")
        axh.set_xlabel("projected increment")
        axh.set_ylabel("density")
        axh.legend(fontsize=8)
        axs.bar(np.arange(len(skew)), skew)
        axs.set_xticks(np.arange(len(skew)), labels, rotation=25, ha="right", fontsize=7)
        axs.set_ylabel("skewness")
        axs.set_title("third cumulant")
        axs.grid(axis="y", alpha=0.25)
        axk.bar(np.arange(len(kurt)), kurt)
        axk.axhline(3.0, color="k", linewidth=0.8, linestyle="--")
        axk.set_xticks(np.arange(len(kurt)), labels, rotation=25, ha="right", fontsize=7)
        axk.set_ylabel("kurtosis")
        axk.set_title("fourth moment")
        axk.grid(axis="y", alpha=0.25)
        fig.suptitle(plot_title)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(path, dpi=180)
        plt.close()

    draw(out, title, with_setup=True)
    draw(out.with_name("figure_clean.png"), title, with_setup=False)


def make_exp42_poisson(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    emp = data["empirical_increment"]; gau = data["gaussian_increment"]; poi = data["poisson_increment"]
    emp_m = data["empirical_multi"]; gau_m = data["gaussian_multi"]; poi_m = data["poisson_multi"]
    methods = [r["method"].replace("_", "\n") for r in rows]
    wass = np.array([float(r["wasserstein"]) for r in rows])

    def draw(path: Path, plot_title: str, with_setup: bool = False) -> None:
        if with_setup:
            fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.2), gridspec_kw={"width_ratios": [1.05, 1.35, 1.35, 1.1]})
            draw_setup_box(axes[0], setup)
            ax1, ax2, ax3 = axes[1], axes[2], axes[3]
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 4.6))
        ax1.hist(emp, bins=90, density=True, alpha=0.45, label="SGD")
        ax1.hist(gau, bins=90, density=True, histtype="step", linewidth=1.6, label="Gaussian")
        ax1.hist(poi, bins=90, density=True, histtype="step", linewidth=1.6, label="Poisson")
        ax1.set_title("one-step increment")
        ax1.legend(fontsize=8)
        ax2.hist(emp_m, bins=90, density=True, alpha=0.45, label="SGD")
        ax2.hist(gau_m, bins=90, density=True, histtype="step", linewidth=1.6, label="Gaussian")
        ax2.hist(poi_m, bins=90, density=True, histtype="step", linewidth=1.6, label="Poisson")
        ax2.set_title("multi-step distribution")
        ax2.legend(fontsize=8)
        ax3.bar(np.arange(len(wass)), wass)
        ax3.set_xticks(np.arange(len(wass)), methods, rotation=30, ha="right", fontsize=7)
        ax3.set_yscale("log")
        ax3.set_ylabel("Wasserstein distance")
        ax3.set_title("fit error")
        ax3.grid(axis="y", alpha=0.25, which="both")
        fig.suptitle(plot_title)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(path, dpi=180)
        plt.close()

    draw(out, title, with_setup=True)
    draw(out.with_name("figure_clean.png"), title, with_setup=False)


def make_exp43_mlp_mean_cov(rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    colors = {
        "exact_sgd": "black",
        "standard_langevin": "tab:red",
        "drift_corrected_langevin": "tab:blue",
    }
    labels = {
        "exact_sgd": "exact SGD",
        "standard_langevin": "standard Langevin",
        "drift_corrected_langevin": "drift-corrected Langevin",
    }

    def draw(path: Path, plot_title: str, with_setup: bool = False) -> None:
        if with_setup:
            fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.4), gridspec_kw={"width_ratios": [1.05, 1.55, 1.25]})
            draw_setup_box(axes[0], setup)
            ax, ax_err = axes[1], axes[2]
        else:
            fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(11.8, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
        for method in ["exact_sgd", "standard_langevin", "drift_corrected_langevin"]:
            rr = [r for r in rows if r["method"] == method]
            x = np.array([float(r["mean_x"]) for r in rr])
            y = np.array([float(r["mean_y"]) for r in rr])
            ax.plot(x, y, "-", color=colors[method], linewidth=2.0, label=labels[method])
            ax.plot(x[0], y[0], "o", color=colors[method], markersize=4)
            ax.plot(x[-1], y[-1], "s", color=colors[method], markersize=4)
            for r in rr:
                if int(float(r["is_ellipse_step"])) != 1:
                    continue
                width = max(float(r["ellipse_width"]), 1e-10)
                height = max(float(r["ellipse_height"]), 1e-10)
                if width < 1e-8 and height < 1e-8:
                    continue
                ell = plt.matplotlib.patches.Ellipse(
                    (float(r["mean_x"]), float(r["mean_y"])),
                    width=width,
                    height=height,
                    angle=float(r["ellipse_angle"]),
                    fill=False,
                    edgecolor=colors[method],
                    linewidth=1.0,
                    alpha=0.55,
                )
                ax.add_patch(ell)
        ax.set_xlabel("projection on Hessian eigenvector 1")
        ax.set_ylabel("projection on Hessian eigenvector 2")
        ax.set_title("projected ensemble mean and 1-sigma ellipses")
        ax.axis("equal")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        sgd = {int(float(r["step"])): (float(r["mean_x"]), float(r["mean_y"])) for r in rows if r["method"] == "exact_sgd"}
        for method in ["standard_langevin", "drift_corrected_langevin"]:
            rr = [r for r in rows if r["method"] == method]
            steps = np.array([int(float(r["step"])) for r in rr])
            err = np.array([
                np.linalg.norm(np.array([float(r["mean_x"]), float(r["mean_y"])]) - np.array(sgd[int(float(r["step"]))]))
                for r in rr
            ])
            ax_err.semilogy(steps, np.maximum(err, 1e-18), color=colors[method], label=labels[method])
        ax_err.set_xlabel("step")
        ax_err.set_ylabel("mean error to SGD")
        ax_err.set_title("projected mean error")
        ax_err.grid(alpha=0.25, which="both")
        ax_err.legend(fontsize=8)
        fig.suptitle(plot_title)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(path, dpi=180)
        plt.close()

    draw(out, title, with_setup=True)
    draw(out.with_name("figure_clean.png"), title, with_setup=False)


def make_exp44_rough_landscape(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    wlatt = data["w_lattice"]
    landscapes = data["landscapes"]
    mean_sgd = data["mean_exact_sgd"]
    mean_l = data["mean_standard_langevin"]
    mean_c = data["mean_drift_corrected_langevin"]
    var_sgd = data["variance_exact_sgd"]
    var_l = data["variance_standard_langevin"]
    var_c = data["variance_drift_corrected_langevin"]
    centers = data["hist_centers"]
    hist_sgd = data["hist_exact_sgd"]
    hist_l = data["hist_standard_langevin"]
    hist_c = data["hist_drift_corrected_langevin"]
    steps = np.arange(len(mean_sgd))

    def draw(path: Path, plot_title: str, with_setup: bool = False) -> None:
        if with_setup:
            fig, axes = plt.subplots(1, 5, figsize=(22.5, 5.4), gridspec_kw={"width_ratios": [1.05, 1.25, 1.2, 1.2, 1.2]})
            draw_setup_box(axes[0], setup)
            ax_land, ax_mean, ax_var, ax_hist = axes[1], axes[2], axes[3], axes[4]
        else:
            fig, (ax_land, ax_mean, ax_var, ax_hist) = plt.subplots(1, 4, figsize=(18.0, 4.8))
        for k in range(landscapes.shape[0]):
            ax_land.plot(wlatt, landscapes[k], linewidth=1.0, alpha=0.75)
        ax_land.set_xlabel("w")
        ax_land.set_ylabel("sample loss")
        ax_land.set_title("fluctuating minibatch landscapes")
        ax_land.grid(alpha=0.2)

        ax_mean.plot(steps, mean_sgd, "k-", linewidth=2.0, label="exact SGD")
        ax_mean.plot(steps, mean_l, "r--", linewidth=2.0, label="standard Langevin")
        ax_mean.plot(steps, mean_c, "b-.", linewidth=2.0, label="drift-corrected Langevin")
        ax_mean.set_xlabel("SGD macro-step")
        ax_mean.set_ylabel("ensemble mean")
        ax_mean.set_title("mean trajectory")
        ax_mean.grid(alpha=0.25)
        ax_mean.legend(fontsize=8)

        ax_var.plot(steps, var_sgd, "k-", linewidth=2.0, label="exact SGD")
        ax_var.plot(steps, var_l, "r--", linewidth=2.0, label="standard Langevin")
        ax_var.plot(steps, var_c, "b-.", linewidth=2.0, label="drift-corrected Langevin")
        ax_var.set_xlabel("SGD macro-step")
        ax_var.set_ylabel("variance")
        ax_var.set_title("ensemble variance")
        ax_var.grid(alpha=0.25)
        ax_var.legend(fontsize=8)

        width = centers[1] - centers[0] if len(centers) > 1 else 0.01
        ax_hist.bar(centers, hist_sgd, width=width, alpha=0.45, label="exact SGD")
        ax_hist.step(centers, hist_l, where="mid", linewidth=2.0, color="tab:red", label="standard Langevin")
        ax_hist.step(centers, hist_c, where="mid", linewidth=2.0, color="tab:blue", linestyle="-.", label="drift-corrected Langevin")
        ax_hist.set_xlabel("final w")
        ax_hist.set_ylabel("density")
        ax_hist.set_title("final distribution")
        ax_hist.grid(alpha=0.2)
        ax_hist.legend(fontsize=8)

        fig.suptitle(plot_title)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(path, dpi=180)
        plt.close()

    draw(out, title, with_setup=True)
    draw(out.with_name("figure_clean.png"), title, with_setup=False)


def make_hessian_decoupling(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    cov_orig = data["covariance_original_final"]
    cov_eig = data["covariance_eigenbasis_final"]
    times = data["times"]
    ratios = data["offdiag_over_diag"]
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 5.2), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2, 1.35]})
    draw_setup_box(axes[0], setup)
    im0 = axes[1].imshow(cov_orig, cmap="coolwarm")
    axes[1].set_title("covariance original basis")
    plt.colorbar(im0, ax=axes[1], fraction=0.046)
    im1 = axes[2].imshow(cov_eig, cmap="coolwarm")
    axes[2].set_title("covariance Hessian basis")
    plt.colorbar(im1, ax=axes[2], fraction=0.046)
    axes[3].plot(times, ratios)
    axes[3].set_yscale("log")
    axes[3].set_xlabel("iteration")
    axes[3].set_ylabel("offdiag / diag mass")
    axes[3].set_title("decoupling diagnostic")
    axes[3].grid(alpha=0.25)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_exp27_phase(result_dir: Path, rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    data = _npz(result_dir)
    etas, gammas = data["etas"], data["gammas"]
    empirical = data["empirical_unstable"].T
    discrete = data["discrete_unstable"].T
    standard = data["standard_unstable"].T
    mismatch = np.logical_and(discrete > 0.5, standard < 0.5).astype(float)
    fig, axes = plt.subplots(1, 5, figsize=(22.5, 5.2), gridspec_kw={"width_ratios": [1.05, 1.05, 1.05, 1.05, 1.25]})
    draw_setup_box(axes[0], setup)
    eta_grid = np.linspace(etas.min(), etas.max(), 200)
    lam = _load_optional_json(result_dir / "metrics.json").get("lambda", 1.0)
    panels = [
        (empirical, "empirical SGD unstable"),
        (discrete, "discrete theory unstable"),
        (standard, "standard Langevin unstable"),
        (mismatch, "theory mismatch region"),
    ]
    for ax, (mat, panel_title) in zip(axes[1:5], panels):
        im = ax.imshow(mat, origin="lower", aspect="auto", extent=[etas.min(), etas.max(), gammas.min(), gammas.max()], cmap="Reds", vmin=0, vmax=1)
        ax.plot(eta_grid, lam * (2 - eta_grid * lam) / eta_grid, "k-", linewidth=1.2, label="discrete boundary")
        ax.plot(eta_grid, 2 * lam / eta_grid, "k--", linewidth=1.2, label="standard boundary")
        ax.set_ylim(gammas.min(), gammas.max())
        ax.set_xlabel(r"$\eta$")
        ax.set_title(panel_title)
        ax.grid(alpha=0.15)
    axes[1].set_ylabel(r"$\Gamma$")
    axes[4].legend(fontsize=7, loc="upper right")
    plt.colorbar(im, ax=axes[1:5], fraction=0.018, pad=0.01, label="unstable = 1")
    # Overlay a concrete trace from the mismatch region as an inset-style line on the last panel.
    if "example_variance_trace" in data:
        trace = data["example_variance_trace"]
        axins = axes[4].inset_axes([0.08, 0.08, 0.47, 0.38])
        axins.semilogy(np.arange(len(trace)), np.maximum(trace, 1e-18), color="tab:blue")
        axins.set_title(f"example eta={float(data['example_eta']):.2g}", fontsize=7)
        axins.set_xlabel("record", fontsize=7)
        axins.set_ylabel("var", fontsize=7)
        axins.tick_params(labelsize=6)
        axins.grid(alpha=0.2)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=180)
    plt.close()


def make_exp29_nonstationary(rows: list[dict[str, str]], out: Path, title: str, setup: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.2), gridspec_kw={"width_ratios": [1.05, 1.35, 1.35]})
    draw_setup_box(axes[0], setup)
    for method in ["flat_mean_displacement_abs"]:
        rr = [r for r in rows if r["method"] == method]
        axes[1].plot([float(r["step"]) for r in rr], [float(r["variance"]) for r in rr], label="|mean flat|")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("mean displacement")
    axes[1].set_title("non-stationary drift")
    axes[1].grid(alpha=0.25)
    for method in ["flat_variance", "sharp_variance", "pure_diffusion_prediction_flat", "drift_induced_prediction_flat"]:
        rr = [r for r in rows if r["method"] == method]
        axes[2].loglog(np.maximum([float(r["step"]) for r in rr], 1), np.maximum([float(r["variance"]) for r in rr], 1e-18), label=method.replace("_", " "))
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("variance")
    axes[2].set_title("variance growth")
    axes[2].grid(alpha=0.25, which="both")
    axes[2].legend(fontsize=7)
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

    if {"eigenvalue", "gradient_noise_variance"}.issubset(keys):
        x = np.array([float(r["eigenvalue"]) for r in rows])
        y = np.array([float(r["gradient_noise_variance"]) for r in rows])
        mask = (x > 0) & (y > 0)
        ax.loglog(x[mask] + 1e-12, y[mask] + 1e-12, "o", markersize=4, alpha=0.75)
        if mask.sum() >= 2:
            coef = np.polyfit(x[mask], y[mask], 1)
            xx = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.loglog(xx, np.maximum(coef[0] * xx + coef[1], 1e-12), "k--", linewidth=1)
        ax.set_xlabel("Hessian eigenvalue lambda_i")
        ax.set_ylabel("gradient-noise variance Var(G_i)")
        ax.set_title("gradient noise vs curvature")
    elif {"measured_variance", "predicted_variance"}.issubset(keys):
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
    elif result_dir.name.startswith("exp16_") and "cumulative_log_evidence" in rows[0]:
        make_exp16_evidence(rows, out, title, setup)
    elif (result_dir.name.startswith("exp17_") or result_dir.name.startswith("exp21_")) and "covariance_trace" in rows[0]:
        make_exp17_covariance(rows, out, title, setup)
    elif result_dir.name.startswith("exp18_") or result_dir.name.startswith("exp36_"):
        make_eta_scaling(result_dir, out, title, setup)
    elif result_dir.name.startswith("exp20_"):
        make_exp20_anisotropy(result_dir, out, title, setup)
    elif result_dir.name.startswith("exp23_"):
        make_batch_scaling(result_dir, out, title, setup)
    elif result_dir.name.startswith("exp27_"):
        make_exp27_phase(result_dir, rows, out, title, setup)
    elif result_dir.name.startswith("exp29_"):
        make_exp29_nonstationary(rows, out, title, setup)
    elif result_dir.name.startswith("exp33_"):
        make_hessian_decoupling(result_dir, rows, out, title, setup)
    elif result_dir.name.startswith("exp35_"):
        make_exp35_sampling(result_dir, out, title, setup)
    elif result_dir.name.startswith("exp37_"):
        make_exp37_mean_matching(result_dir, rows, out, title, setup)
    elif result_dir.name.startswith("exp38_"):
        make_exp38_moment_matching(result_dir, out, title, setup)
    elif result_dir.name.startswith("exp39_"):
        make_exp39_ablation(rows, out, title, setup)
    elif result_dir.name.startswith("exp40_"):
        make_exp40_semigroup(result_dir, rows, out, title, setup)
    elif result_dir.name.startswith("exp41_"):
        make_exp41_cumulants(result_dir, rows, out, title, setup)
    elif result_dir.name.startswith("exp42_"):
        make_exp42_poisson(result_dir, rows, out, title, setup)
    elif result_dir.name.startswith("exp43_"):
        make_exp43_mlp_mean_cov(rows, out, title, setup)
    elif result_dir.name.startswith("exp44_"):
        make_exp44_rough_landscape(result_dir, rows, out, title, setup)
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
        clean = out.with_name("figure_clean.png")
        if clean.exists():
            top_level_clean = figures_dir / f"{result_dir.name}_clean.png"
            top_level_clean.write_bytes(clean.read_bytes())
            print(f"[saved] {top_level_clean}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
