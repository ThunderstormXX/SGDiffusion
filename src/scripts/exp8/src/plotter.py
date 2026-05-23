import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def selected_indices(n_params):
    return [0, n_params // 2, n_params - 1]


def plot_param_trajectories(gd, sgd, out_path):
    steps = np.arange(gd.shape[0])
    mean, std = sgd.mean(axis=0), sgd.std(axis=0)
    indices = selected_indices(gd.shape[1])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, idx in zip(axes, indices):
        for run in range(sgd.shape[0]):
            ax.plot(steps, sgd[run, :, idx], color="0.72", lw=0.6, alpha=0.55)
        ax.fill_between(steps, mean[:, idx] - std[:, idx], mean[:, idx] + std[:, idx],
                        color="tab:blue", alpha=0.18, linewidth=0)
        ax.plot(steps, mean[:, idx], color="tab:blue", lw=1.8, label="SGD mean")
        ax.plot(steps, gd[:, idx], color="tab:red", lw=1.8, label="GD")
        ax.set_ylabel(f"p[{idx}]")
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("iteration / GD epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def project_pca(gd, sgd):
    flat_sgd = sgd.reshape(-1, sgd.shape[-1])
    data = np.concatenate([gd, flat_sgd], axis=0).astype(np.float32)
    x = torch.from_numpy(data)
    mean = x.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x - mean, q=2, center=False)
    z = ((x - mean) @ v[:, :2]).numpy()
    gd_z = z[:gd.shape[0]]
    sgd_z = z[gd.shape[0]:].reshape(sgd.shape[0], sgd.shape[1], 2)
    return gd_z, sgd_z


def _add_std_ellipse(ax, points, color, alpha):
    if len(points) < 2:
        return
    cov = np.cov(points.T) + np.eye(2) * 1e-10
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    center = points.mean(axis=0)
    patch = Ellipse(center, 2 * np.sqrt(vals[0]), 2 * np.sqrt(vals[1]), angle=angle,
                    facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)


def plot_pca_trajectories(gd, sgd, out_path):
    gd_z, sgd_z = project_pca(gd, sgd)
    mean = sgd_z.mean(axis=0)
    alpha = min(0.035, max(0.006, 3.0 / max(sgd_z.shape[1], 1)))
    fig, ax = plt.subplots(figsize=(9, 7))
    for run in range(sgd_z.shape[0]):
        ax.plot(sgd_z[run, :, 0], sgd_z[run, :, 1], color="0.72", lw=0.65, alpha=0.5)
    for step in range(sgd_z.shape[1]):
        _add_std_ellipse(ax, sgd_z[:, step, :], "tab:blue", alpha)
    ax.plot(mean[:, 0], mean[:, 1], color="tab:blue", lw=2.0, label="SGD mean")
    ax.plot(gd_z[:, 0], gd_z[:, 1], color="tab:red", lw=2.0, label="GD")
    ax.scatter(mean[0, 0], mean[0, 1], color="black", s=18, label="start", zorder=4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
