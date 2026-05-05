#!/usr/bin/env python3
"""Corrected-Langevin and rough-landscape EXP37-EXP44 implementations.

This module keeps the newer generator-validation experiments out of the
legacy monolithic experiments.py while preserving the same run_experiment.py
contract and artifact format.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import wasserstein_1d, write_csv
from .experiments import (
    _covariance_np,
    _grad_at_vector_np,
    _hessian_full_batch,
    _real_mlp_context,
    _sample_batch_tensors,
)

def _nonlinear_g_h(w: np.ndarray, a: np.ndarray, cubic: float, quartic: float) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise nonlinear mean gradient and Hessian diagonal."""
    g = a[None, :] * w + cubic * w**2 + quartic * w**3
    h_diag = a[None, :] + 2.0 * cubic * w + 3.0 * quartic * w**2
    return g, h_diag


def _corrected_flow_mean(w: np.ndarray, eta: float, a: np.ndarray, cubic: float, quartic: float, *, corrected: bool, substeps: int) -> np.ndarray:
    """Integrate the standard/corrected generator drift over one macro-time eta."""
    y = np.asarray(w, dtype=np.float64).copy()
    dt = eta / max(int(substeps), 1)
    for _ in range(max(int(substeps), 1)):
        g, h = _nonlinear_g_h(y, a, cubic, quartic)
        drift = -g
        if corrected:
            drift = drift - 0.5 * eta * h * g
        y = y + dt * drift
    return y


def _simulate_generator_ensemble(
    w: np.ndarray,
    eta: float,
    a: np.ndarray,
    cubic: float,
    quartic: float,
    sigma: float,
    rng: np.random.Generator,
    *,
    corrected_drift: bool,
    raw_covariance: bool,
    substeps: int,
) -> np.ndarray:
    """One macro-step of the continuous surrogate integrated over time eta."""
    y = np.asarray(w, dtype=np.float64).copy()
    dt = eta / max(int(substeps), 1)
    dim = y.shape[1]
    for _ in range(max(int(substeps), 1)):
        g, h = _nonlinear_g_h(y, a, cubic, quartic)
        drift = -g
        if corrected_drift:
            drift = drift - 0.5 * eta * h * g
        noise = sigma * rng.normal(size=y.shape)
        if raw_covariance:
            # Diagnostic-only wrong model: add a state-wise scalar ggT contribution along g.
            u = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
            z_par = rng.normal(size=(y.shape[0], 1))
            noise = noise + np.linalg.norm(g, axis=1, keepdims=True) * z_par * u
        y = y + dt * drift + math.sqrt(eta * dt) * noise
        if dim == 0:
            break
    return y


def _generator_base_params(config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    p = config.get("parameters", {})
    dim = int(p.get("dim", 4))
    eig_min = float(p.get("eig_min", 0.4))
    eig_max = float(p.get("eig_max", 1.6))
    a = np.linspace(eig_min, eig_max, dim, dtype=np.float64)
    return p, a


def run_exp37(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Corrected log-generator drift improves finite-eta mean-map matching."""
    p, a = _generator_base_params(config)
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    dim = len(a)
    n_points = int(p.get("n_points", 512))
    etas = np.asarray(p.get("etas", [0.02, 0.05, 0.1, 0.2]), dtype=np.float64)
    cubic = float(p.get("cubic", 0.25))
    quartic = float(p.get("quartic", 0.08))
    point_scale = float(p.get("point_scale", 0.6))
    substeps = int(p.get("substeps", 128))
    w = point_scale * rng.normal(size=(n_points, dim))
    rows = []
    std_errs, corr_errs = [], []
    for eta in etas:
        g, _ = _nonlinear_g_h(w, a, cubic, quartic)
        exact = w - eta * g
        standard = _corrected_flow_mean(w, eta, a, cubic, quartic, corrected=False, substeps=substeps)
        corrected = _corrected_flow_mean(w, eta, a, cubic, quartic, corrected=True, substeps=substeps)
        err_std = np.linalg.norm(standard - exact, axis=1)
        err_cor = np.linalg.norm(corrected - exact, axis=1)
        std_errs.append(float(err_std.mean()))
        corr_errs.append(float(err_cor.mean()))
        rows += [
            {"eta": float(eta), "step": float(eta), "method": "standard_generator", "error": float(err_std.mean()), "error_std": float(err_std.std(ddof=1))},
            {"eta": float(eta), "step": float(eta), "method": "corrected_generator", "error": float(err_cor.mean()), "error_std": float(err_cor.std(ddof=1))},
        ]
    std_arr, corr_arr = np.asarray(std_errs), np.asarray(corr_errs)
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", etas=etas, standard_error=std_arr, corrected_error=corr_arr, points=w, curvature=a)
    return {
        "dim": dim,
        "n_points": n_points,
        "substeps": substeps,
        "standard_error_slope": float(np.polyfit(np.log(etas), np.log(std_arr), 1)[0]),
        "corrected_error_slope": float(np.polyfit(np.log(etas), np.log(corr_arr), 1)[0]),
        "final_improvement_standard_over_corrected": float(std_arr[-1] / max(corr_arr[-1], 1e-18)),
        "pass": bool(corr_arr[-1] < std_arr[-1] and np.mean(corr_arr < std_arr) > 0.75),
    }


def run_exp38(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Moment evolution: corrected generator vs standard generator on nonlinear SGD."""
    p, a = _generator_base_params(config)
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    dim = len(a)
    eta = float(p.get("eta", 0.12))
    steps = int(p.get("num_steps", p.get("steps", 25)))
    n_runs = int(p.get("n_runs", 20000))
    sigma = float(p.get("sigma", 0.4))
    cubic = float(p.get("cubic", 0.25))
    quartic = float(p.get("quartic", 0.08))
    substeps = int(p.get("substeps", 64))
    initial_scale = float(p.get("initial_scale", 0.5))
    w0 = initial_scale * rng.normal(size=(n_runs, dim))
    sgd = w0.copy(); standard = w0.copy(); corrected = w0.copy()
    rows, times, mean_std, mean_cor, cov_std, cov_cor = [], [], [], [], [], []
    for step in range(steps + 1):
        emp_mean = sgd.mean(axis=0)
        emp_cov = _covariance_np(sgd)
        for name, arr in [("standard_generator", standard), ("corrected_generator", corrected)]:
            m_err = float(np.linalg.norm(arr.mean(axis=0) - emp_mean))
            c_err = float(np.linalg.norm(_covariance_np(arr) - emp_cov, ord="fro") / max(np.linalg.norm(emp_cov, ord="fro"), 1e-12))
            rows.append({"step": step, "method": name, "mean_error": m_err, "covariance_error": c_err, "combined_error": m_err + c_err})
            if name.startswith("standard"):
                mean_std.append(m_err); cov_std.append(c_err)
            else:
                mean_cor.append(m_err); cov_cor.append(c_err)
        times.append(step)
        if step == steps:
            break
        g, _ = _nonlinear_g_h(sgd, a, cubic, quartic)
        sgd = sgd - eta * (g + sigma * rng.normal(size=sgd.shape))
        standard = _simulate_generator_ensemble(standard, eta, a, cubic, quartic, sigma, rng, corrected_drift=False, raw_covariance=False, substeps=substeps)
        corrected = _simulate_generator_ensemble(corrected, eta, a, cubic, quartic, sigma, rng, corrected_drift=True, raw_covariance=False, substeps=substeps)
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", times=np.asarray(times), standard_mean_error=np.asarray(mean_std), corrected_mean_error=np.asarray(mean_cor), standard_covariance_error=np.asarray(cov_std), corrected_covariance_error=np.asarray(cov_cor))
    return {
        "dim": dim,
        "eta": eta,
        "num_steps": steps,
        "n_runs": n_runs,
        "final_mean_improvement_standard_over_corrected": float(mean_std[-1] / max(mean_cor[-1], 1e-18)),
        "final_covariance_improvement_standard_over_corrected": float(cov_std[-1] / max(cov_cor[-1], 1e-18)),
        "pass": bool(mean_cor[-1] < mean_std[-1]),
    }


def run_exp39(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Ablate corrected drift vs the wrong raw-moment covariance correction."""
    p, a = _generator_base_params(config)
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    dim = len(a)
    eta = float(p.get("eta", 0.15))
    n_runs = int(p.get("n_runs", 50000))
    sigma = float(p.get("sigma", 0.35))
    cubic = float(p.get("cubic", 0.25))
    quartic = float(p.get("quartic", 0.08))
    substeps = int(p.get("substeps", 96))
    w = float(p.get("point_scale", 0.8)) * rng.normal(size=(1, dim))
    g, _ = _nonlinear_g_h(w, a, cubic, quartic)
    exact_mean = (w - eta * g).ravel()
    exact_cov = eta**2 * sigma**2 * np.eye(dim)
    samples = w + np.zeros((n_runs, dim))
    exact = samples - eta * (g + sigma * rng.normal(size=samples.shape))
    methods = {
        "A_standard": (False, False),
        "B_corrected_drift_only": (True, False),
        "C_wrong_raw_covariance": (False, True),
        "D_full_corrected": (True, False),
    }
    rows, raw = [], {}
    for name, (corr_drift, raw_cov) in methods.items():
        pred = _simulate_generator_ensemble(samples, eta, a, cubic, quartic, sigma, rng, corrected_drift=corr_drift, raw_covariance=raw_cov, substeps=substeps)
        mean_error = float(np.linalg.norm(pred.mean(axis=0) - exact_mean))
        cov_error = float(np.linalg.norm(_covariance_np(pred) - exact_cov, ord="fro") / max(np.linalg.norm(exact_cov, ord="fro"), 1e-12))
        # Conditional likelihood sanity at the exact SGD samples under Gaussian approximation.
        pred_mean = pred.mean(axis=0)
        pred_cov = _covariance_np(pred) + 1e-9 * np.eye(dim)
        sign, logdet = np.linalg.slogdet(pred_cov)
        diff = exact - pred_mean[None, :]
        quad = np.sum(diff * np.linalg.solve(pred_cov, diff.T).T, axis=1)
        nll = float(0.5 * np.mean(dim * np.log(2 * np.pi) + logdet + quad)) if sign > 0 else float("inf")
        rows.append({"method": name, "step": 0, "mean_error": mean_error, "covariance_error": cov_error, "negative_log_likelihood": nll})
        raw[f"{name}_samples"] = pred.astype(np.float32)
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", exact_samples=exact.astype(np.float32), **raw)
    by_name = {r["method"]: r for r in rows}
    return {
        "dim": dim,
        "eta": eta,
        "n_runs": n_runs,
        "corrected_drift_mean_error": by_name["B_corrected_drift_only"]["mean_error"],
        "standard_mean_error": by_name["A_standard"]["mean_error"],
        "raw_covariance_nll_delta_vs_standard": by_name["C_wrong_raw_covariance"]["negative_log_likelihood"] - by_name["A_standard"]["negative_log_likelihood"],
        "pass": bool(by_name["B_corrected_drift_only"]["mean_error"] < by_name["A_standard"]["mean_error"] and by_name["C_wrong_raw_covariance"]["negative_log_likelihood"] > by_name["A_standard"]["negative_log_likelihood"]),
    }


def _one_dim_g_h(x: np.ndarray, a: float, cubic: float, quartic: float) -> tuple[np.ndarray, np.ndarray]:
    return a * x + cubic * x**2 + quartic * x**3, a + 2.0 * cubic * x + 3.0 * quartic * x**2


def _fp_generator_matrix(grid: np.ndarray, eta: float, sigma: float, a: float, cubic: float, quartic: float, corrected: bool) -> np.ndarray:
    n = len(grid)
    dx = float(grid[1] - grid[0])
    g, h = _one_dim_g_h(grid, a, cubic, quartic)
    b = -g - (0.5 * eta * h * g if corrected else 0.0)
    diff = 0.5 * eta * sigma**2
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Forward generator on density with reflecting-ish zero-flux boundary approximation.
        if i > 0:
            G[i - 1, i] += max(b[i], 0.0) / dx + diff / dx**2
        if i < n - 1:
            G[i + 1, i] += max(-b[i], 0.0) / dx + diff / dx**2
        G[i, i] -= (max(b[i], 0.0) if i > 0 else 0.0) / dx
        G[i, i] -= (max(-b[i], 0.0) if i < n - 1 else 0.0) / dx
        G[i, i] -= (diff / dx**2) * ((1 if i > 0 else 0) + (1 if i < n - 1 else 0))
    return G


def _kernel_matrix_1d(grid: np.ndarray, mean: np.ndarray, variance: float, ridge: float) -> np.ndarray:
    dx = float(grid[1] - grid[0])
    var = max(float(variance), 1e-12)
    P = np.exp(-0.5 * (grid[:, None] - mean[None, :]) ** 2 / var) / math.sqrt(2.0 * math.pi * var)
    P = P * dx
    P = P / np.maximum(P.sum(axis=0, keepdims=True), 1e-18)
    return (1.0 - ridge) * P + ridge / len(grid)


def _matrix_log_generator(P: np.ndarray, eta: float, ridge: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(P)
    eigvals = np.where(np.abs(eigvals) < ridge, ridge + 0j, eigvals)
    return np.real(eigvecs @ np.diag(np.log(eigvals)) @ np.linalg.inv(eigvecs)) / eta


def run_exp40(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Semigroup logarithm: compare exact log transition to standard/corrected generators."""
    p = config.get("parameters", {})
    etas = np.asarray(p.get("etas", [0.02, 0.04, 0.08]), dtype=np.float64)
    n_grid = int(p.get("n_grid", 61))
    x_max = float(p.get("x_max", 2.5))
    sigma = float(p.get("sigma", 0.35))
    a = float(p.get("a", 1.0))
    cubic = float(p.get("cubic", 0.2))
    quartic = float(p.get("quartic", 0.05))
    ridge = max(float(p.get("transition_ridge", 1e-5)), 1e-5)
    grid = np.linspace(-x_max, x_max, n_grid)
    rows, std_errs, cor_errs = [], [], []
    mats = {}
    dx = grid[1] - grid[0]
    for eta in etas:
        g, _ = _one_dim_g_h(grid, a, cubic, quartic)
        mean_exact = grid - eta * g
        mean_standard = _corrected_flow_mean(grid[:, None], eta, np.asarray([a]), cubic, quartic, corrected=False, substeps=128).ravel()
        mean_corrected = _corrected_flow_mean(grid[:, None], eta, np.asarray([a]), cubic, quartic, corrected=True, substeps=128).ravel()
        variance = eta**2 * sigma**2
        P = _kernel_matrix_1d(grid, mean_exact, variance, ridge)
        P_standard = _kernel_matrix_1d(grid, mean_standard, variance, ridge)
        P_corrected = _kernel_matrix_1d(grid, mean_corrected, variance, ridge)
        G_exact = _matrix_log_generator(P, eta, ridge)
        G_standard = _matrix_log_generator(P_standard, eta, ridge)
        G_corrected = _matrix_log_generator(P_corrected, eta, ridge)
        err_std = float(np.linalg.norm(G_exact - G_standard, ord="fro") / max(np.linalg.norm(G_exact, ord="fro"), 1e-12))
        err_cor = float(np.linalg.norm(G_exact - G_corrected, ord="fro") / max(np.linalg.norm(G_exact, ord="fro"), 1e-12))
        std_errs.append(err_std); cor_errs.append(err_cor)
        rows += [{"eta": float(eta), "step": float(eta), "method": "standard_generator", "error": err_std}, {"eta": float(eta), "step": float(eta), "method": "corrected_generator", "error": err_cor}]
        if eta == etas[-1]:
            mats = {"G_exact": G_exact, "G_standard": G_standard, "G_corrected": G_corrected}
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", etas=etas, standard_error=np.asarray(std_errs), corrected_error=np.asarray(cor_errs), grid=grid, **mats)
    return {
        "n_grid": n_grid,
        "sigma": sigma,
        "final_improvement_standard_over_corrected": float(std_errs[-1] / max(cor_errs[-1], 1e-18)),
        "corrected_better_fraction": float(np.mean(np.asarray(cor_errs) < np.asarray(std_errs))),
        "pass": bool(cor_errs[-1] < std_errs[-1]),
    }


def _centered_exponential(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
    return rng.exponential(scale=1.0, size=size) - 1.0


def _hist_kl(a: np.ndarray, b: np.ndarray, bins: int = 120) -> float:
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    pa, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    pa = pa + 1e-12; pb = pb + 1e-12
    pa = pa / pa.sum(); pb = pb / pb.sum()
    return float(np.sum(pa * np.log(pa / pb)))


def run_exp41(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Higher cumulants: Gaussian corrected Langevin misses skew/kurtosis of increments."""
    p = config.get("parameters", {})
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    n = int(p.get("n_samples", 100000))
    eta = float(p.get("eta", 0.05))
    gbar = float(p.get("gbar", 1.0))
    sigma = float(p.get("sigma", 0.8))
    exact = -eta * (gbar + sigma * _centered_exponential(rng, (n,)))
    gaussian = -eta * (gbar + sigma * rng.normal(size=n))
    def stats(x: np.ndarray) -> tuple[float, float, float, float]:
        m = float(np.mean(x)); s = float(np.std(x, ddof=1))
        z = (x - m) / max(s, 1e-18)
        return m, s, float(np.mean(z**3)), float(np.mean(z**4))
    exact_s = stats(exact); gauss_s = stats(gaussian)
    rows = [
        {"method": "empirical_sgd_increment", "step": 0, "mean": exact_s[0], "std": exact_s[1], "skewness": exact_s[2], "kurtosis": exact_s[3], "kl_to_empirical": 0.0},
        {"method": "gaussian_corrected_langevin", "step": 0, "mean": gauss_s[0], "std": gauss_s[1], "skewness": gauss_s[2], "kurtosis": gauss_s[3], "kl_to_empirical": _hist_kl(exact, gaussian)},
    ]
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", empirical_increment=exact, gaussian_increment=gaussian)
    return {
        "n_samples": n,
        "eta": eta,
        "empirical_skewness": exact_s[2],
        "gaussian_skewness": gauss_s[2],
        "empirical_kurtosis": exact_s[3],
        "gaussian_kurtosis": gauss_s[3],
        "histogram_kl_empirical_to_gaussian": rows[1]["kl_to_empirical"],
        "pass": bool(abs(exact_s[2] - gauss_s[2]) > 0.5 and abs(exact_s[3] - gauss_s[3]) > 0.5),
    }


def run_exp42(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Exploratory compound-Poisson surrogate for skewed SGD increments."""
    p = config.get("parameters", {})
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    n = int(p.get("n_samples", 100000))
    eta = float(p.get("eta", 0.05))
    gbar = float(p.get("gbar", 1.0))
    sigma = float(p.get("sigma", 0.8))
    multi_steps = int(p.get("multi_steps", 20))
    lambda_true = float(p.get("lambda_true", 0.7))
    jump_scale_true = float(p.get("jump_scale", eta * sigma))
    exact = -eta * gbar + jump_scale_true * (rng.poisson(lambda_true, size=n) - lambda_true)
    mean = float(np.mean(exact)); var = float(np.var(exact, ddof=1))
    centered = exact - mean
    k3 = float(np.mean(centered**3))
    scale = math.copysign(max(abs(k3) / max(var, 1e-18), 1e-8), k3)
    lam = max(var / (scale**2), 1e-6)
    poisson = mean + scale * (rng.poisson(lam, size=n) - lam)
    gaussian = rng.normal(loc=mean, scale=math.sqrt(var), size=n)
    exact_multi = np.zeros(n); poisson_multi = np.zeros(n); gaussian_multi = np.zeros(n)
    for _ in range(multi_steps):
        exact_multi += -eta * gbar + jump_scale_true * (rng.poisson(lambda_true, size=n) - lambda_true)
        poisson_multi += mean + scale * (rng.poisson(lam, size=n) - lam)
        gaussian_multi += rng.normal(loc=mean, scale=math.sqrt(var), size=n)
    w_g = wasserstein_1d(exact, gaussian); w_p = wasserstein_1d(exact, poisson)
    w_gm = wasserstein_1d(exact_multi, gaussian_multi); w_pm = wasserstein_1d(exact_multi, poisson_multi)
    rows = [
        {"method": "gaussian_one_step", "step": 1, "wasserstein": w_g, "histogram_kl": _hist_kl(exact, gaussian)},
        {"method": "poisson_one_step", "step": 1, "wasserstein": w_p, "histogram_kl": _hist_kl(exact, poisson)},
        {"method": "gaussian_multi_step", "step": multi_steps, "wasserstein": w_gm, "histogram_kl": _hist_kl(exact_multi, gaussian_multi)},
        {"method": "poisson_multi_step", "step": multi_steps, "wasserstein": w_pm, "histogram_kl": _hist_kl(exact_multi, poisson_multi)},
    ]
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", empirical_increment=exact, gaussian_increment=gaussian, poisson_increment=poisson, empirical_multi=exact_multi, gaussian_multi=gaussian_multi, poisson_multi=poisson_multi)
    return {
        "n_samples": n,
        "eta": eta,
        "poisson_lambda": float(lam),
        "poisson_scale": float(scale),
        "one_step_wasserstein_gaussian": w_g,
        "one_step_wasserstein_poisson": w_p,
        "multi_step_wasserstein_gaussian": w_gm,
        "multi_step_wasserstein_poisson": w_pm,
        "pass": bool(w_p < w_g),
        "exploratory": True,
    }


def _ellipse_params_2d(cov: np.ndarray) -> tuple[float, float, float]:
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vec = vecs[:, order[0]]
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
    width = 2.0 * math.sqrt(float(vals[0]))
    height = 2.0 * math.sqrt(float(vals[1]))
    return width, height, angle


def run_exp43(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """MLP mean trajectory and projected covariance: SGD vs standard/corrected Langevin."""
    p = config.get("parameters", {})
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    batch_size = int(p.get("batch_size", 64))
    n_runs = int(p.get("n_runs", 24))
    steps = int(p.get("num_steps", p.get("steps", 40)))
    eta = float(p.get("eta", 0.05))
    n_noise_batches = int(p.get("n_noise_batches", 48))
    covariance_ridge = float(p.get("covariance_ridge", 1e-10))
    device, model, _, x_train, y_train, loss_fn, w0 = _real_mlp_context(config, seed, batch_size)
    dim = int(w0.size)

    g0 = _grad_at_vector_np(model, w0, x_train, y_train, loss_fn, device)
    h0 = _hessian_full_batch(model, x_train, y_train, loss_fn).detach().cpu().numpy().astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(h0)
    top = np.argsort(eigvals)[-2:]
    proj_basis = eigvecs[:, top]

    grad_samples = []
    for _ in range(n_noise_batches):
        xb, yb = _sample_batch_tensors(x_train, y_train, batch_size, rng, replacement=True)
        grad_samples.append(_grad_at_vector_np(model, w0, xb, yb, loss_fn, device))
    grad_samples_arr = np.asarray(grad_samples, dtype=np.float64)
    noise = grad_samples_arr - g0[None, :]
    d_cov = _covariance_np(noise)
    vals_d, vecs_d = np.linalg.eigh(d_cov)
    vals_d = np.maximum(vals_d, 0.0)
    d_sqrt = vecs_d @ np.diag(np.sqrt(vals_d + covariance_ridge))
    hg0 = h0 @ g0

    methods = ["exact_sgd", "standard_langevin", "drift_corrected_langevin"]
    states = {m: np.repeat(w0[None, :], n_runs, axis=0).astype(np.float64) for m in methods}
    cursors = [{} for _ in range(n_runs)]
    projected_means = {m: [] for m in methods}
    projected_covs = {m: [] for m in methods}
    projected_traces = {m: [] for m in methods}
    rows: list[dict[str, Any]] = []
    ellipse_steps = sorted(set([0, steps // 4, steps // 2, steps]))

    for step in range(steps + 1):
        for method in methods:
            proj = (states[method] - w0[None, :]) @ proj_basis
            mean2 = proj.mean(axis=0)
            cov2 = _covariance_np(proj)
            width, height, angle = _ellipse_params_2d(cov2)
            projected_means[method].append(mean2)
            projected_covs[method].append(cov2)
            projected_traces[method].append(proj.astype(np.float32))
            rows.append({
                "step": step,
                "method": method,
                "mean_x": float(mean2[0]),
                "mean_y": float(mean2[1]),
                "cov_xx": float(cov2[0, 0]),
                "cov_xy": float(cov2[0, 1]),
                "cov_yy": float(cov2[1, 1]),
                "ellipse_width": float(width),
                "ellipse_height": float(height),
                "ellipse_angle": float(angle),
                "is_ellipse_step": int(step in ellipse_steps),
            })
        if step == steps:
            break

        for r in range(n_runs):
            xb, yb = _sample_batch_tensors(x_train, y_train, batch_size, rng, replacement=True, cursor_state=cursors[r])
            grad = _grad_at_vector_np(model, states["exact_sgd"][r], xb, yb, loss_fn, device)
            states["exact_sgd"][r] = states["exact_sgd"][r] - eta * grad
        gaussian_noise = rng.normal(size=(n_runs, dim)) @ d_sqrt.T
        states["standard_langevin"] = states["standard_langevin"] - eta * g0[None, :] + eta * gaussian_noise
        gaussian_noise = rng.normal(size=(n_runs, dim)) @ d_sqrt.T
        states["drift_corrected_langevin"] = (
            states["drift_corrected_langevin"]
            - eta * g0[None, :]
            - 0.5 * eta**2 * hg0[None, :]
            + eta * gaussian_noise
        )

    mean_sgd = np.asarray(projected_means["exact_sgd"], dtype=np.float64)
    mean_std = np.asarray(projected_means["standard_langevin"], dtype=np.float64)
    mean_cor = np.asarray(projected_means["drift_corrected_langevin"], dtype=np.float64)
    cov_sgd = np.asarray(projected_covs["exact_sgd"], dtype=np.float64)
    cov_std = np.asarray(projected_covs["standard_langevin"], dtype=np.float64)
    cov_cor = np.asarray(projected_covs["drift_corrected_langevin"], dtype=np.float64)
    mean_err_std_t = np.linalg.norm(mean_std - mean_sgd, axis=1)
    mean_err_cor_t = np.linalg.norm(mean_cor - mean_sgd, axis=1)
    cov_err_std_t = np.linalg.norm(cov_std - cov_sgd, axis=(1, 2))
    cov_err_cor_t = np.linalg.norm(cov_cor - cov_sgd, axis=(1, 2))
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        methods=np.asarray(methods),
        eigenvalues=eigvals,
        selected_eigenvector_indices=top,
        projected_mean_exact_sgd=mean_sgd,
        projected_mean_standard_langevin=mean_std,
        projected_mean_drift_corrected_langevin=mean_cor,
        projected_cov_exact_sgd=cov_sgd,
        projected_cov_standard_langevin=cov_std,
        projected_cov_drift_corrected_langevin=cov_cor,
        projected_trajectories_exact_sgd=np.asarray(projected_traces["exact_sgd"], dtype=np.float32),
        projected_trajectories_standard_langevin=np.asarray(projected_traces["standard_langevin"], dtype=np.float32),
        projected_trajectories_drift_corrected_langevin=np.asarray(projected_traces["drift_corrected_langevin"], dtype=np.float32),
        full_gradient=g0,
        hessian=h0,
        gradient_covariance=d_cov,
        mean_error_standard_by_step=mean_err_std_t,
        mean_error_corrected_by_step=mean_err_cor_t,
        covariance_error_standard_by_step=cov_err_std_t,
        covariance_error_corrected_by_step=cov_err_cor_t,
    )
    mean_error_standard = float(np.mean(mean_err_std_t))
    mean_error_corrected = float(np.mean(mean_err_cor_t))
    cov_error_standard = float(np.mean(cov_err_std_t))
    cov_error_corrected = float(np.mean(cov_err_cor_t))
    return {
        "model": "MLP-386",
        "dim": dim,
        "eta": eta,
        "num_steps": steps,
        "n_runs": n_runs,
        "batch_size": batch_size,
        "n_noise_batches": n_noise_batches,
        "local_coefficients": "frozen_at_w0",
        "full_gradient_norm": float(np.linalg.norm(g0)),
        "hessian_gradient_norm": float(np.linalg.norm(hg0)),
        "selected_eigenvalue_0": float(eigvals[top[0]]),
        "selected_eigenvalue_1": float(eigvals[top[1]]),
        "mean_error_standard": mean_error_standard,
        "mean_error_corrected": mean_error_corrected,
        "final_mean_error_standard": float(mean_err_std_t[-1]),
        "final_mean_error_corrected": float(mean_err_cor_t[-1]),
        "cov_error_standard": cov_error_standard,
        "cov_error_corrected": cov_error_corrected,
        "mean_improvement": float(mean_error_standard / max(mean_error_corrected, 1e-18)),
        "cov_improvement": float(cov_error_standard / max(cov_error_corrected, 1e-18)),
        "pass": bool(mean_error_corrected < mean_error_standard),
    }


def _rough_grad_av(w: np.ndarray) -> np.ndarray:
    return 4.0 * w**3 - 0.6 * w + 0.15


def _rough_hess_av(w: np.ndarray) -> np.ndarray:
    return 12.0 * w**2 - 0.6


def _rough_fluct_multipl(w: np.ndarray) -> np.ndarray:
    return 0.1 * w**2 + 1.0


def _rough_grad_stoh(a: np.ndarray, b: np.ndarray, w: np.ndarray, amp: np.ndarray, phase: np.ndarray, length: float) -> np.ndarray:
    angles = 2.0 * np.pi * w[..., None] / length * b[None, :] + np.pi * phase
    return _rough_fluct_multipl(w) * np.sum(a[None, :] * amp * np.cos(angles), axis=-1)


def _rough_grad_disp(w: np.ndarray, a: np.ndarray) -> np.ndarray:
    return _rough_fluct_multipl(w) ** 2 * np.sum(a * a) / 6.0


def run_exp44(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Rough fluctuating 1D landscape: exact SGD vs standard/corrected Langevin."""
    p = config.get("parameters", {})
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    nm = int(p.get("n_modes", 20))
    length = float(p.get("landscape_length", 2.0))
    eta = float(p.get("eta", 0.1))
    steps = int(p.get("num_steps", p.get("steps", 50)))
    n_samples = int(p.get("n_samples", 10000))
    langevin_substeps = int(p.get("langevin_substeps", 100))
    w_start_std = float(p.get("w_start_std", 0.18))
    n_landscapes = int(p.get("n_landscapes", 10))
    w_num = int(p.get("w_num", 300))
    hist_min = float(p.get("hist_min", -0.9))
    hist_max = float(p.get("hist_max", 0.4))
    freq = np.arange(nm, dtype=np.float64) + 1.0
    amplitudes = 0.5 * np.ones(nm, dtype=np.float64) / (freq * freq)

    wlatt = -length / 2.0 + np.arange(w_num, dtype=np.float64) / w_num * length
    dw = length / w_num
    landscape_rows = []
    landscapes = np.zeros((n_landscapes, w_num), dtype=np.float64)
    for k in range(n_landscapes):
        amp = rng.uniform(-1.0, 1.0, nm)
        phase = rng.uniform(-1.0, 1.0, nm)
        values = np.zeros(w_num, dtype=np.float64)
        for i in range(w_num - 1):
            grad = _rough_grad_av(np.asarray([wlatt[i]]))[0] + _rough_grad_stoh(amplitudes, freq, np.asarray([wlatt[i]]), amp, phase, length)[0]
            values[i + 1] = values[i] + dw * grad
        landscapes[k] = values
        for j in range(w_num):
            landscape_rows.append({"landscape": k, "w": float(wlatt[j]), "loss": float(values[j])})

    wstart = rng.normal(0.0, w_start_std, n_samples)
    result = np.zeros((steps + 1, n_samples), dtype=np.float64)
    result[0] = wstart
    for n in range(steps):
        amp = rng.uniform(-1.0, 1.0, size=(n_samples, nm))
        phase = rng.uniform(-1.0, 1.0, size=(n_samples, nm))
        w = result[n]
        stochastic = _rough_grad_stoh(amplitudes, freq, w, amp, phase, length)
        result[n + 1] = w - eta * (_rough_grad_av(w) + stochastic)

    dt = eta / max(langevin_substeps, 1)
    result_l = np.zeros((steps * langevin_substeps + 1, n_samples), dtype=np.float64)
    result_l[0] = wstart
    result_c = np.zeros((steps * langevin_substeps + 1, n_samples), dtype=np.float64)
    result_c[0] = wstart
    for n in range(steps * langevin_substeps):
        w = result_l[n]
        xi = rng.normal(size=n_samples)
        result_l[n + 1] = w - dt * _rough_grad_av(w) + np.sqrt(np.maximum(dt * eta * _rough_grad_disp(w, amplitudes), 0.0)) * xi
        wc = result_c[n]
        xic = rng.normal(size=n_samples)
        corrected_drift = _rough_grad_av(wc) + 0.5 * eta * _rough_hess_av(wc) * _rough_grad_av(wc)
        result_c[n + 1] = wc - dt * corrected_drift + np.sqrt(np.maximum(dt * eta * _rough_grad_disp(wc, amplitudes), 0.0)) * xic
    result_l_red = result_l[np.arange(steps + 1) * langevin_substeps]
    result_c_red = result_c[np.arange(steps + 1) * langevin_substeps]

    avg = result.mean(axis=1)
    var = result.var(axis=1, ddof=1)
    avg_l = result_l_red.mean(axis=1)
    var_l = result_l_red.var(axis=1, ddof=1)
    avg_c = result_c_red.mean(axis=1)
    var_c = result_c_red.var(axis=1, ddof=1)
    mean_err = np.abs(avg_l - avg)
    var_err = np.abs(var_l - var)
    mean_err_c = np.abs(avg_c - avg)
    var_err_c = np.abs(var_c - var)
    rows: list[dict[str, Any]] = []
    for step in range(steps + 1):
        rows += [
            {"step": step, "method": "exact_sgd", "mean": float(avg[step]), "variance": float(var[step]), "mean_error_to_sgd": 0.0, "variance_error_to_sgd": 0.0},
            {"step": step, "method": "standard_langevin", "mean": float(avg_l[step]), "variance": float(var_l[step]), "mean_error_to_sgd": float(mean_err[step]), "variance_error_to_sgd": float(var_err[step])},
            {"step": step, "method": "drift_corrected_langevin", "mean": float(avg_c[step]), "variance": float(var_c[step]), "mean_error_to_sgd": float(mean_err_c[step]), "variance_error_to_sgd": float(var_err_c[step])},
        ]
    hist_bins = int(p.get("hist_bins", 80))
    hist_sgd, edges = np.histogram(result[-1], bins=hist_bins, range=(hist_min, hist_max), density=True)
    hist_l, _ = np.histogram(result_l_red[-1], bins=edges, density=True)
    hist_c, _ = np.histogram(result_c_red[-1], bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for c, hs, hl, hc in zip(centers, hist_sgd, hist_l, hist_c):
        rows.append({"step": steps, "method": "hist_exact_sgd", "mean": float(c), "variance": float(hs), "mean_error_to_sgd": 0.0, "variance_error_to_sgd": 0.0})
        rows.append({"step": steps, "method": "hist_standard_langevin", "mean": float(c), "variance": float(hl), "mean_error_to_sgd": 0.0, "variance_error_to_sgd": 0.0})
        rows.append({"step": steps, "method": "hist_drift_corrected_langevin", "mean": float(c), "variance": float(hc), "mean_error_to_sgd": 0.0, "variance_error_to_sgd": 0.0})
    write_csv(result_dir / "figure_data.csv", rows)
    write_csv(result_dir / "landscape_data.csv", landscape_rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        exact_sgd=result.astype(np.float32),
        standard_langevin=result_l_red.astype(np.float32),
        drift_corrected_langevin=result_c_red.astype(np.float32),
        standard_langevin_full=result_l.astype(np.float32),
        drift_corrected_langevin_full=result_c.astype(np.float32),
        w_lattice=wlatt,
        landscapes=landscapes,
        mean_exact_sgd=avg,
        mean_standard_langevin=avg_l,
        mean_drift_corrected_langevin=avg_c,
        variance_exact_sgd=var,
        variance_standard_langevin=var_l,
        variance_drift_corrected_langevin=var_c,
        hist_centers=centers,
        hist_exact_sgd=hist_sgd,
        hist_standard_langevin=hist_l,
        hist_drift_corrected_langevin=hist_c,
        amplitudes=amplitudes,
        frequencies=freq,
    )
    return {
        "n_modes": nm,
        "eta": eta,
        "num_steps": steps,
        "n_samples": n_samples,
        "langevin_substeps": langevin_substeps,
        "mean_error_l1": float(np.mean(mean_err)),
        "final_mean_error": float(mean_err[-1]),
        "variance_error_l1": float(np.mean(var_err)),
        "corrected_variance_error_l1": float(np.mean(var_err_c)),
        "final_variance_error": float(var_err[-1]),
        "final_corrected_variance_error": float(var_err_c[-1]),
        "final_mean_exact_sgd": float(avg[-1]),
        "final_mean_standard_langevin": float(avg_l[-1]),
        "final_mean_drift_corrected_langevin": float(avg_c[-1]),
        "final_variance_exact_sgd": float(var[-1]),
        "final_variance_standard_langevin": float(var_l[-1]),
        "final_variance_drift_corrected_langevin": float(var_c[-1]),
        "histogram_l1_distance": float(np.mean(np.abs(hist_sgd - hist_l))),
        "corrected_histogram_l1_distance": float(np.mean(np.abs(hist_sgd - hist_c))),
        "corrected_mean_error_l1": float(np.mean(mean_err_c)),
        "final_corrected_mean_error": float(mean_err_c[-1]),
        "mean_improvement_standard_over_corrected": float(np.mean(mean_err) / max(np.mean(mean_err_c), 1e-18)),
        "variance_improvement_standard_over_corrected": float(np.mean(var_err) / max(np.mean(var_err_c), 1e-18)),
        "pass": bool(np.isfinite(mean_err[-1]) and np.isfinite(var_err[-1])),
    }
