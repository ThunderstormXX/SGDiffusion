from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .loss import BCELoss
from .model import SmoothTinyMLP


class LocalOptimizer:
    def __init__(self, model: SmoothTinyMLP, loss: BCELoss, cfg: ExperimentConfig):
        self.model = model
        self.loss = loss
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def init_theta(self) -> np.ndarray:
        scale = self.cfg.init_scale / math.sqrt(self.model.dim)
        return self.rng.normal(scale=scale, size=self.model.dim)

    def sgd(self, theta0: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        theta = theta0.copy()
        velocity = np.zeros_like(theta)
        best = theta.copy()
        best_loss = np.inf
        for _ in range(self.cfg.train_steps):
            value, grad = self.loss.value_grad(theta, X, y)
            if value < best_loss:
                best_loss = value
                best = theta.copy()
            velocity = self.cfg.sgd_momentum * velocity + grad
            theta -= self.cfg.sgd_lr * velocity
            if not np.all(np.isfinite(theta)):
                return best
        return best

    def polish(self, theta0: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
            return self.loss.value_grad(theta, X, y)

        options = {"maxiter": self.cfg.lbfgs_steps, "gtol": 1e-9, "ftol": 1e-13, "maxls": 50}
        res = minimize(fun, theta0, jac=True, method="L-BFGS-B", options=options)
        return np.asarray(res.x, dtype=np.float64)

    def optimize(self, theta0: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
        theta = self.polish(self.sgd(theta0, X, y), X, y)
        value, grad = self.loss.value_grad(theta, X, y)
        return theta, float(value), float(np.linalg.norm(grad))

    def run_many(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        starts = np.array([self.init_theta() for _ in range(self.cfg.n_runs)])
        return self.run_starts(starts, X, y, "multistart optimize")

    def run_starts(
        self, starts: np.ndarray, X: np.ndarray, y: np.ndarray, desc: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        thetas, losses, grad_norms = [], [], []
        for start in tqdm(starts, desc=desc):
            theta, loss, grad_norm = self.optimize(start, X, y)
            thetas.append(theta)
            losses.append(loss)
            grad_norms.append(grad_norm)
        return np.array(thetas), np.array(losses), np.array(grad_norms)
