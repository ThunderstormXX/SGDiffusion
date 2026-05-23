from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_pretrain_losses(logs, sgd_steps, out_path):
    if not logs:
        return
    steps = np.array([row["global_step"] for row in logs], dtype=float)
    batch_loss = np.array([row["batch_loss"] for row in logs], dtype=float)
    full_rows = [row for row in logs if row["full_train_loss"] is not None]
    full_steps = np.array([row["global_step"] for row in full_rows], dtype=float)
    full_loss = np.array([row["full_train_loss"] for row in full_rows], dtype=float)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(0, max(sgd_steps, 0), color="tab:blue", alpha=0.06, label="SGD phase")
    ax.axvspan(max(sgd_steps, 0), steps.max(), color="tab:red", alpha=0.05, label="GD phase")
    ax.plot(steps, batch_loss, color="0.55", lw=0.8, alpha=0.7, label="batch loss")
    ax.plot(full_steps, full_loss, color="tab:red", lw=2.0, marker="o", ms=2.5,
            label="full train loss")
    if sgd_steps > 0:
        ax.axvline(sgd_steps, color="black", lw=1.0, alpha=0.45)
    ax.set_xlabel("pretrain step")
    ax.set_ylabel("cross entropy")
    ax.set_title("Pretrain losses")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_metric(logs, metric_name, out_path):
    rows = [row for row in logs if row.get(metric_name) is not None]
    if not rows:
        return
    steps = np.array([row["global_step"] for row in rows], dtype=float)
    values = np.array([row[metric_name] for row in rows], dtype=float)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, values, color="tab:purple", lw=1.8)
    ax.set_xlabel("pretrain step")
    ax.set_ylabel("fraction")
    ax.set_title(metric_name)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
