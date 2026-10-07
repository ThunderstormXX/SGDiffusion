#!/usr/bin/env python3
"""Figure of the NanoGPT experiment: discrete SGD vs. the Langevin approximation.

For every Hessian eigendirection i and every learning rate eta:
    x = eta * lambda_i                    (learning rate x curvature)
    y = measured variance / Langevin prediction,
where the measured variance is the variance of the SGD iterates along the direction over
N_TRAJ independent trajectories, averaged over the last WINDOW steps, and the Langevin
prediction is eta d_i / (2 lambda_i - eta Gamma_ii) (with its time dependence over the same
window). Langevin predicts y = 1. Discrete SGD predicts
    y = (2 lambda_i - eta Gamma_ii) / (2 lambda_i - eta (lambda_i^2 + Gamma_ii)),
which depends on the direction only through x and the small ratio Gamma_ii / lambda_i^2;
the curve is drawn for the median of that ratio.

Usage:
    python make_figure.py RUN_DIR OUT_DIR                 # from the trajectories of a pipeline run
    python make_figure.py --from-csv POINTS.csv OUT_DIR   # redraw from the saved points
Writes OUT_DIR/fig_nanogpt_finite_step.{pdf,png} and OUT_DIR/figure_points.csv.
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

N_TRAJ = 240      # trajectories per learning rate used for the figure in the paper
WINDOW = 200      # the variance is averaged over the last WINDOW steps of the trajectories

# One-hue ordinal ramp for the ordered learning rates (monotone lightness, distinguishable
# steps, also under colour-vision deficiency) and one contrasting accent for the prediction.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
ACCENT, INK, INK2, RULE, GRID = "#eb6834", "#0b0b0b", "#52514e", "#b9b8b2", "#e9e8e4"
C_GRID = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 1.5])     # planned values of eta * lambda_max


def theory(x, g):
    """Discrete / Langevin plateau ratio as a function of x = eta*lambda and g = Gamma/lambda^2."""
    return (1 - x * g / 2) / (1 - x * (1 + g) / 2)


def points_from_run(run):
    """One row per (learning rate, direction), computed from the first N_TRAJ trajectories."""
    st = json.load(open(os.path.join(run, "stats.json")))
    lam, gam, d = np.array(st["lam"]), np.array(st["Gamma"]), np.array(st["d"])
    K = len(lam)
    parts = {}
    for f in sorted(glob.glob(os.path.join(run, "traj_eta*_w*.npz"))):
        if f.endswith(".tmp.npz"):          # a worker's file in the middle of being replaced
            continue
        z = np.load(f)
        if int(z["done"]):
            parts.setdefault(float(z["eta"]), []).append(z["proj"][:int(z["done"]), :, :K].astype(np.float64))
    rows = []
    for eta in sorted(parts):
        p = np.concatenate(parts[eta])
        if len(p) < N_TRAJ:
            raise SystemExit(f"eta={eta:g}: only {len(p)} trajectories, the figure needs {N_TRAJ}")
        p = p[:N_TRAJ]
        T1 = p.shape[1]
        W = min(WINDOW, T1 - 1)
        emp = p.var(axis=0, ddof=1)[-W:].mean(0)
        n = np.arange(T1 - W, T1)[:, None]
        kl = 2 * lam - eta * gam
        lang = (eta * d[:K] / kl * (1 - np.exp(-kl * eta * n))).mean(0)
        for i in range(K):
            rows.append(dict(eta=eta, eta_lambda_max=eta * lam.max(), lambda_i=lam[i],
                             gamma_over_lambda_sq=gam[i] / lam[i] ** 2, x=eta * lam[i], y=emp[i] / lang[i]))
    return rows


def read_csv(path):
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eta", "eta_lambda_max", "lambda_i", "gamma_over_lambda_sq", "x", "y", "discrete_prediction"])
        for r in rows:
            w.writerow([f"{r['eta']:.6g}", f"{r['eta_lambda_max']:.4g}", f"{r['lambda_i']:.4f}",
                        f"{r['gamma_over_lambda_sq']:.5f}", f"{r['x']:.5f}", f"{r['y']:.5f}",
                        f"{theory(r['x'], r['gamma_over_lambda_sq']):.5f}"])


def draw(rows, path):
    etas = sorted({r["eta"] for r in rows})
    g_med = float(np.median([r["gamma_over_lambda_sq"] for r in rows if r["eta"] == etas[0]]))
    n_dir = sum(r["eta"] == etas[0] for r in rows)
    # Six learning rates on a five-step ramp: the two smallest (no visible effect) share the lightest step.
    cls = {eta: max(0, k - (len(etas) - len(RAMP))) for k, eta in enumerate(etas)}
    names = {}
    for eta in etas:
        c = next(r["eta_lambda_max"] for r in rows if r["eta"] == eta)
        names.setdefault(cls[eta], []).append(float(C_GRID[np.abs(C_GRID - c).argmin()]))
    labels = [("≤ " if len(v) > 1 else "") + f"{max(v):g}" for _, v in sorted(names.items())]

    # Times-like text and math; drawn at final size (one column).
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    xs = np.linspace(0, 1.56, 300)
    ax.axhline(1, color=INK2, lw=1.1, zorder=2)
    ax.plot(xs, theory(xs, g_med), color=ACCENT, lw=1.9, solid_capstyle="round", zorder=2)
    for eta in etas:
        sel = [r for r in rows if r["eta"] == eta]
        ax.scatter([r["x"] for r in sel], [r["y"] for r in sel], s=10, color=RAMP[cls[eta]],
                   edgecolors="white", linewidths=0.35, zorder=3 + cls[eta])

    ax.set_yscale("log")
    ax.set_ylim(0.82, 6.6)
    ax.set_xlim(0, 1.6)
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 5]))
    ax.yaxis.set_major_formatter(FixedFormatter(["1", "1.5", "2", "3", "5"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK2, length=2.5, width=0.6, labelsize=8.5)
    ax.set_xlabel(r"$\eta\,\lambda_i$   (learning rate $\times$ curvature)", color=INK, fontsize=9.5)
    ax.set_ylabel("measured variance / Langevin prediction", color=INK, fontsize=9.5)
    ax.set_title("NanoGPT (6.6M parameters), WikiText-2", loc="left", fontsize=10, color=INK,
                 fontweight="bold", pad=17)
    ax.text(0, 1.04, f"{n_dir} Hessian eigendirections $\\times$ {len(etas)} learning rates",
            transform=ax.transAxes, fontsize=8.5, color=INK2)

    # The two predictions are labelled on the lines themselves (text in ink; position carries identity).
    ax.text(1.58, 2.0, "discrete SGD", color=INK, fontsize=9, ha="right", va="center", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))      # masks the gridline behind the label
    ax.annotate("", xy=(1.39, theory(1.39, g_med)), xytext=(1.39, 2.14),
                arrowprops=dict(arrowstyle="-", color=RULE, lw=0.6, shrinkA=0, shrinkB=3))
    ax.text(1.58, 0.955, "Langevin prediction", color=INK, fontsize=9, ha="right", va="top")

    handles = [Line2D([], [], marker="o", ls="", ms=4.4, mfc=c, mec="white", mew=0.35) for c in RAMP]
    leg = ax.legend(handles, labels, title=r"SGD,  $\eta\,\lambda_{\max}$", loc="upper left", frameon=False,
                    fontsize=8.5, title_fontsize=8.5, handletextpad=0.1, labelspacing=0.22, borderaxespad=0.1)
    leg._legend_box.align = "left"
    leg.get_title().set_color(INK)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.tight_layout(pad=0.4)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=400)
    plt.close(fig)


def main():
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "--from-csv":
        rows, out = read_csv(args[1]), args[2]
    elif len(args) == 2:
        rows, out = points_from_run(args[0]), args[1]
    else:
        raise SystemExit(__doc__)
    os.makedirs(out, exist_ok=True)
    if args[0] != "--from-csv":
        write_csv(rows, os.path.join(out, "figure_points.csv"))
    draw(rows, os.path.join(out, "fig_nanogpt_finite_step"))
    etas = sorted({r["eta"] for r in rows})
    print(f"{len(rows)} points: {len(rows) // len(etas)} directions x {len(etas)} learning rates {etas}")


if __name__ == "__main__":
    main()
