#!/usr/bin/env python3
"""Single-trajectory SGD plateau probe for MNIST/CNN.

Runs ordinary SGD epoch by epoch and evaluates full train/test loss after each
epoch. The goal is to estimate when the train-loss curve changes from fast
optimization to a plateau-like regime.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import FlexibleCNN  # noqa: E402
from src.scripts.exp6.src.common import collect_environment, save_json, set_reproducible  # noqa: E402
from src.scripts.exp6.src.experiments import _eval_classification, _mnist_loaders  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def detect_plateau(losses: np.ndarray, window: int, rel_slope_threshold: float, patience: int) -> dict:
    if len(losses) < 2 * window + patience:
        return {"plateau_epoch": None, "reason": "not_enough_points"}
    eps = 1e-12
    log_losses = np.log(np.maximum(losses, eps))
    slopes = np.full(len(losses), np.nan)
    x = np.arange(window)
    for end in range(window, len(losses) + 1):
        y = log_losses[end - window:end]
        slopes[end - 1] = np.polyfit(x, y, 1)[0]
    for i in range(window - 1, len(losses) - patience + 1):
        recent = slopes[i:i + patience]
        if np.all(np.isfinite(recent)) and np.all(np.abs(recent) < rel_slope_threshold):
            return {
                "plateau_epoch": int(i + 1),
                "reason": "log_slope_below_threshold",
                "window": window,
                "patience": patience,
                "rel_slope_threshold": rel_slope_threshold,
                "slope_at_plateau": float(slopes[i]),
            }
    return {
        "plateau_epoch": None,
        "reason": "not_detected",
        "window": window,
        "patience": patience,
        "rel_slope_threshold": rel_slope_threshold,
        "last_slope": float(slopes[np.where(np.isfinite(slopes))[0][-1]]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="src/scripts/exp6/results/sgd_plateau_probe_cnn")
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--rel-slope-threshold", type=float, default=1e-4)
    parser.add_argument("--min-epochs", type=int, default=1000)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible(args.seed, True)

    dataset_cfg = {
        "name": "mnist",
        "data_dir": "./data",
        "sample_size": args.sample_size,
        "val_size": args.val_size,
        "subset": "first",
        "normalize": False,
    }
    _, _, train_loader, full_loader, val_loader = _mnist_loaders(dataset_cfg, args.batch_size, True, args.seed)
    epoch_steps = len(train_loader)
    model = FlexibleCNN(
        input_downsample=14,
        conv_channels=[8, 16],
        conv_use_bn=False,
        pool_after=[True, True],
        gap_size=1,
        mlp_hidden_dim=32,
        mlp_num_layers=1,
    )
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    start = time.perf_counter()

    def evaluate(epoch: int) -> None:
        train = _eval_classification(model, full_loader, torch.device("cpu"))
        test = _eval_classification(model, val_loader, torch.device("cpu"))
        rows.append({
            "epoch": epoch,
            "step": epoch * epoch_steps,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "runtime_seconds": time.perf_counter() - start,
        })

    evaluate(0)
    data_iter = iter(train_loader)
    plateau = {"plateau_epoch": None, "reason": "not_checked"}
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for _ in range(epoch_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        evaluate(epoch)
        if epoch >= args.min_epochs:
            losses = np.asarray([r["train_loss"] for r in rows], dtype=np.float64)
            plateau = detect_plateau(losses, args.window, args.rel_slope_threshold, args.patience)
            if plateau["plateau_epoch"] is not None:
                break

    write_csv(result_dir / "plateau_curve.csv", rows)
    metrics = {
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "epochs_run": rows[-1]["epoch"],
        "epoch_steps": epoch_steps,
        "lr": args.lr,
        "sample_size": args.sample_size,
        "batch_size": args.batch_size,
        "final_train_loss": rows[-1]["train_loss"],
        "final_train_accuracy": rows[-1]["train_accuracy"],
        "final_test_loss": rows[-1]["test_loss"],
        "final_test_accuracy": rows[-1]["test_accuracy"],
        "runtime_seconds": time.perf_counter() - start,
    }
    metrics.update(plateau)
    save_json(result_dir / "metrics.json", metrics)
    save_json(result_dir / "environment.json", collect_environment())
    (result_dir / "README.md").write_text(
        "# SGD Plateau Probe\n\n"
        "Single ordinary-SGD trajectory. Full train/test loss is evaluated after every epoch.\n\n"
        f"- max epochs: `{args.max_epochs}`\n"
        f"- epochs run: `{rows[-1]['epoch']}`\n"
        f"- plateau epoch: `{metrics.get('plateau_epoch')}`\n"
        f"- final train loss: `{metrics['final_train_loss']:.6g}`\n"
        f"- final test loss: `{metrics['final_test_loss']:.6g}`\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
