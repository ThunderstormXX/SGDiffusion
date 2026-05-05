#!/usr/bin/env python3
"""Experiment implementations for exp6."""

from __future__ import annotations

import math
import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Subset, TensorDataset

from src.model import FlexibleCNN, FlexibleMLP, NanoGPT

from .common import mean_std_ci, save_json, wasserstein_1d, write_csv


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return torch.device("cpu")
    return torch.device(name)


def _mnist_loaders(cfg: dict[str, Any], batch_size: int, replacement: bool, seed: int):
    from torchvision import datasets, transforms

    data_dir = cfg.get("data_dir", "./data")
    sample_size = int(cfg.get("sample_size", 6400))
    val_size = int(cfg.get("val_size", min(1000, sample_size)))
    normalize = bool(cfg.get("normalize", False))
    transform_items = [transforms.ToTensor()]
    if normalize:
        transform_items.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_items)
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    val_full = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    if cfg.get("subset", "first") == "random":
        g = torch.Generator().manual_seed(seed)
        train_idx = torch.randperm(len(train_full), generator=g)[:sample_size].tolist()
        val_idx = torch.randperm(len(val_full), generator=g)[:val_size].tolist()
    else:
        train_idx = list(range(sample_size))
        val_idx = list(range(val_size))
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)
    if replacement:
        n = (len(train_ds) // batch_size) * batch_size
        g = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(train_ds, replacement=True, num_samples=n, generator=g)
        train_loader = DataLoader(train_ds, batch_sampler=BatchSampler(sampler, batch_size, drop_last=True))
    else:
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)
    full_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=min(512, len(val_ds)), shuffle=False)
    return train_ds, val_ds, train_loader, full_loader, val_loader


def _mlp_model(device: torch.device) -> nn.Module:
    return FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6).to(device)


def _nanogpt_model(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = cfg.get("model", {})
    meta_path = Path(model_cfg.get("meta_path", "src/data/shakespeare_meta.pt"))
    meta = torch.load(meta_path, map_location="cpu")
    return NanoGPT(
        vocab_size=int(meta["vocab_size"]),
        n_embd=int(model_cfg.get("n_embd", 8)),
        n_head=int(model_cfg.get("n_head", 1)),
        n_layer=int(model_cfg.get("n_layer", 1)),
        block_size=int(meta.get("block_size", model_cfg.get("block_size", 16))),
        mlp_ratio=int(model_cfg.get("mlp_ratio", 1)),
    ).to(device)


def _shakespeare_loaders(cfg: dict[str, Any], batch_size: int, replacement: bool, seed: int):
    train_path = Path(cfg.get("train_path", "src/data/shakespeare_train.pt"))
    val_path = Path(cfg.get("val_path", "src/data/shakespeare_val.pt"))
    x_train, y_train = torch.load(train_path, map_location="cpu")
    x_val, y_val = torch.load(val_path, map_location="cpu")
    sample_size = cfg.get("sample_size", None)
    val_size = cfg.get("val_size", None)
    if sample_size is not None:
        x_train = x_train[: int(sample_size)]
        y_train = y_train[: int(sample_size)]
    if val_size is not None:
        x_val = x_val[: int(val_size)]
        y_val = y_val[: int(val_size)]
    train_ds = TensorDataset(x_train.long(), y_train.long())
    val_ds = TensorDataset(x_val.long(), y_val.long())
    if replacement:
        n = (len(train_ds) // batch_size) * batch_size
        g = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(train_ds, replacement=True, num_samples=n, generator=g)
        train_loader = DataLoader(train_ds, batch_sampler=BatchSampler(sampler, batch_size, drop_last=True))
    else:
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)
    full_batch_size = int(cfg.get("full_batch_size", len(train_ds)))
    full_loader = DataLoader(train_ds, batch_size=min(full_batch_size, len(train_ds)), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=min(int(cfg.get("val_batch_size", batch_size)), len(val_ds)), shuffle=False)
    return train_ds, val_ds, train_loader, full_loader, val_loader


def _configured_mlp_model(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = cfg.get("model", {})
    return FlexibleMLP(
        hidden_dim=int(model_cfg.get("hidden_dim", 8)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 1)),
        input_downsample=int(model_cfg.get("input_downsample", 6)),
    ).to(device)


def _configured_model(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = cfg.get("model", {})
    kind = model_cfg.get("type", "mlp")
    if kind == "cnn":
        return FlexibleCNN(
            input_downsample=model_cfg.get("input_downsample", None),
            conv_channels=list(model_cfg.get("conv_channels", [8, 16])),
            conv_use_bn=bool(model_cfg.get("conv_use_bn", False)),
            pool_after=list(model_cfg.get("pool_after", [True, True])),
            gap_size=int(model_cfg.get("gap_size", 1)),
            mlp_hidden_dim=int(model_cfg.get("mlp_hidden_dim", 32)),
            mlp_num_layers=int(model_cfg.get("mlp_num_layers", 1)),
        ).to(device)
    return _configured_mlp_model(cfg, device)


def _eval_classification(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(loss_fn(logits, y).item())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return {"loss": float(np.mean(losses)), "accuracy": correct / max(total, 1)}


def _dataset_to_tensors(dataset: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs).to(device), torch.tensor(ys, dtype=torch.long, device=device)


def _eval_classification_tensors(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, y.numel(), batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            logits = model(xb)
            total_loss += float(loss_fn(logits, yb).item())
            correct += int((logits.argmax(1) == yb).sum().item())
            total += int(yb.numel())
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


def _loss_value(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module | None) -> torch.Tensor:
    if loss_fn is None:
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Model did not return a loss for supervised language-model batch.")
        return loss
    return loss_fn(model(x), y)


def _grad_vector_general(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module | None) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss = _loss_value(model, x, y, loss_fn)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return torch.cat([g.detach().reshape(-1) for g in grads])


def _hessian_full_batch_general(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module | None) -> torch.Tensor:
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    loss = _loss_value(model, x, y, loss_fn)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    rows = []
    for i in range(n):
        grad2 = torch.autograd.grad(flat[i], params, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in grad2]))
    return torch.stack(rows).detach()


def _train_reference_mlp(cfg: dict[str, Any], device: torch.device, seed: int) -> tuple[nn.Module, Any]:
    train_ds, val_ds, train_loader, full_loader, val_loader = _mnist_loaders(
        cfg["dataset"], cfg["training"]["batch_size"], cfg["training"].get("replacement", True), seed
    )
    model = _mlp_model(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["training"]["lr"])
    loss_fn = nn.CrossEntropyLoss()
    steps = int(cfg["training"].get("reference_steps", 20))
    data_iter = iter(train_loader)
    model.train()
    for _ in range(steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model, (train_ds, val_ds, train_loader, full_loader, val_loader)


def _hessian_full_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
    params = list(model.parameters())
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    rows = []
    for i in range(flat.numel()):
        grad2 = torch.autograd.grad(flat[i], params, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in grad2]))
    return torch.stack(rows).detach()


def _hvp_full_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module, vector: torch.Tensor) -> torch.Tensor:
    params = list(model.parameters())
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    dot = torch.dot(flat_grad, vector)
    hvp = torch.autograd.grad(dot, params)
    return torch.cat([h.detach().reshape(-1) for h in hvp])


def _grad_vector(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return torch.cat([g.detach().reshape(-1) for g in grads])


def _estimate_minibatch_noise_std(
    model: nn.Module,
    full_grad: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    n_batches: int,
    loss_fn: nn.Module,
) -> torch.Tensor:
    _, _, loader, _, _ = _mnist_loaders(cfg["dataset"], cfg["training"]["batch_size"], True, seed)
    grads = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        grads.append(_grad_vector(model, x.to(device), y.to(device), loss_fn) - full_grad)
    if len(grads) < 2:
        return torch.full_like(full_grad, 1e-12)
    noise = torch.stack(grads)
    return noise.std(dim=0, unbiased=True).clamp_min(1e-12)


def _modified_langevin_noise_std(centered_noise_std: torch.Tensor, full_grad: torch.Tensor) -> torch.Tensor:
    """Euler-Maruyama noise amplitude for sqrt(D + grad_bar^2), diagonalized."""
    return torch.sqrt(centered_noise_std.square() + full_grad.square()).clamp_min(1e-12)


def run_exp1a(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    p = config["parameters"]
    n_runs, steps = int(p["n_runs"]), int(p["steps"])
    eta, lam, sigma = float(p["lr"]), float(p["lambda"]), float(p["noise_std"])
    x0 = float(p.get("x0", 1.0))
    rng = np.random.default_rng(config["seed"])
    sgd = np.zeros((n_runs, steps + 1))
    langevin = np.zeros_like(sgd)
    modified = np.zeros_like(sgd)
    sgd[:, 0] = x0
    langevin[:, 0] = x0
    modified[:, 0] = x0
    additive_std = eta * sigma * abs(x0)
    for t in range(steps):
        xi = rng.normal(0.0, sigma, size=n_runs)
        sgd[:, t + 1] = sgd[:, t] * (1.0 - eta * (lam + xi))
        langevin[:, t + 1] = (1.0 - eta * lam) * langevin[:, t] + additive_std * rng.normal(size=n_runs)
        xi2 = rng.normal(0.0, sigma, size=n_runs)
        modified[:, t + 1] = modified[:, t] * (1.0 - eta * (lam + xi2))
    var_sgd = sgd.var(axis=0, ddof=1)
    var_l = langevin.var(axis=0, ddof=1)
    var_m = modified.var(axis=0, ddof=1)
    tail = max(10, steps // 4)
    xs = np.arange(steps + 1)
    slope_sgd = float(np.polyfit(xs[-tail:], var_sgd[-tail:], 1)[0])
    slope_l = float(np.polyfit(xs[-tail:], var_l[-tail:], 1)[0])
    slope_m = float(np.polyfit(xs[-tail:], var_m[-tail:], 1)[0])
    w_l = wasserstein_1d(sgd[:, -1], langevin[:, -1])
    w_m = wasserstein_1d(sgd[:, -1], modified[:, -1])
    mean_sgd = sgd.mean(axis=0)
    mean_error_l = np.abs(langevin.mean(axis=0) - mean_sgd)
    mean_error_m = np.abs(modified.mean(axis=0) - mean_sgd)
    rows = []
    for name, vals in [("sgd", sgd), ("standard_langevin", langevin), ("modified_langevin", modified)]:
        mean, std, lo, hi = mean_std_ci(vals, axis=0)
        for t in range(steps + 1):
            if name == "standard_langevin":
                mean_path_error = mean_error_l[t]
            elif name == "modified_langevin":
                mean_path_error = mean_error_m[t]
            else:
                mean_path_error = 0.0
            rows.append({
                "step": t,
                "method": name,
                "mean": mean[t],
                "std": std[t],
                "ci95_low": lo[t],
                "ci95_high": hi[t],
                "variance": np.var(vals[:, t], ddof=1),
                "mean_path_error_to_sgd": mean_path_error,
                "mean_coord_1": mean[t],
            })
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", sgd=sgd, standard_langevin=langevin, modified_langevin=modified)
    return {
        "sgd_late_variance_slope": slope_sgd,
        "standard_langevin_late_variance_slope": slope_l,
        "modified_langevin_late_variance_slope": slope_m,
        "wasserstein_standard_to_sgd": w_l,
        "wasserstein_modified_to_sgd": w_m,
        "standard_langevin_final_mean_path_error_to_sgd": float(mean_error_l[-1]),
        "modified_langevin_final_mean_path_error_to_sgd": float(mean_error_m[-1]),
        "pass": bool(abs(slope_sgd) > 1.2 * max(abs(slope_l), 1e-12) and w_m < w_l),
    }


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _ensemble_mlp_single(config: dict[str, Any], result_dir: Path, nonstationary: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, list[float]]]:
    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    base_model, loaders = _train_reference_mlp(config, device, seed)
    train_ds, val_ds, _, full_loader, val_loader = loaders
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    loss_fn = nn.CrossEntropyLoss()
    ref_vec = parameters_to_vector(base_model.parameters()).detach().clone()
    n_params = ref_vec.numel()
    ens = int(config["ensemble"]["n_runs"])
    steps = int(config["ensemble"]["steps"])
    lr = float(config["ensemble"]["lr"])
    langevin_substeps = int(config["ensemble"].get("langevin_substeps", 1))
    if langevin_substeps < 1:
        raise ValueError("ensemble.langevin_substeps must be >= 1")
    langevin_dt = lr / float(langevin_substeps)
    langevin_noise_scale = math.sqrt(lr * langevin_dt)
    langevin_seed_stride = (steps + 1) * langevin_substeps + 1
    langevin_coefficient_update_every = int(config["ensemble"].get("langevin_coefficient_update_every", 1))
    if langevin_coefficient_update_every < 1 or langevin_coefficient_update_every > langevin_substeps:
        raise ValueError("ensemble.langevin_coefficient_update_every must be in [1, langevin_substeps]")
    langevin_noise_mode = str(config["ensemble"].get("langevin_noise_mode", "current"))
    if langevin_noise_mode not in {"current", "reference"}:
        raise ValueError("ensemble.langevin_noise_mode must be 'current' or 'reference'")
    langevin_drift_mode = str(config["ensemble"].get("langevin_drift_mode", "current"))
    if langevin_drift_mode not in {"current", "reference"}:
        raise ValueError("ensemble.langevin_drift_mode must be 'current' or 'reference'")
    directions = int(config["analysis"].get("n_directions", 8))
    torch.manual_seed(seed + 1000)
    q, _ = torch.linalg.qr(torch.randn(n_params, directions, device=device))
    methods = config["ensemble"].get("methods", ["sgd_replacement", "sgd_no_replacement", "standard_langevin", "modified_langevin"])
    trajectories: dict[str, np.ndarray] = {}
    final_losses: dict[str, list[float]] = {m: [] for m in methods}
    full_grad = _grad_vector(base_model, full_x, full_y, loss_fn)
    grad_norm = float(torch.linalg.norm(full_grad).item())
    langevin_noise_batches = int(config["ensemble"].get("langevin_noise_batches", 4))
    ref_noise_std = _estimate_minibatch_noise_std(
        base_model,
        full_grad,
        config,
        device,
        seed + 555,
        langevin_noise_batches,
        loss_fn,
    )
    preload_dataset = bool(config["ensemble"].get("preload_dataset", True))
    train_tensors = _dataset_to_tensors(train_ds, device) if preload_dataset else None
    val_tensors = _dataset_to_tensors(val_ds, device) if preload_dataset else None
    for method in methods:
        print(
            f"[ensemble] lr={lr:g} method={method} M={langevin_substeps} "
            f"dt={langevin_dt:g} coeff_update={langevin_coefficient_update_every} "
            f"noise_mode={langevin_noise_mode} drift_mode={langevin_drift_mode}",
            flush=True,
        )
        arr = np.zeros((ens, steps + 1, directions), dtype=np.float32)
        progress_every = max(1, ens // 4)
        for r in range(ens):
            if r == 0 or (r + 1) % progress_every == 0 or r + 1 == ens:
                print(f"[ensemble] lr={lr:g} method={method} run={r + 1}/{ens}", flush=True)
            torch.manual_seed(seed + 10000 * (r + 1) + len(method))
            model = _mlp_model(device)
            vector_to_parameters(ref_vec.clone(), model.parameters())
            if method.startswith("sgd"):
                repl = method == "sgd_replacement"
                opt = torch.optim.SGD(model.parameters(), lr=lr)
                if repl and train_tensors is not None:
                    train_x, train_y = train_tensors
                    batch_size = int(config["training"]["batch_size"])
                    batch_gen = torch.Generator().manual_seed(seed + r)
                    for t in range(steps + 1):
                        vec = parameters_to_vector(model.parameters()).detach() - ref_vec
                        arr[r, t] = (vec @ q).cpu().numpy()
                        if t == steps:
                            break
                        idx = torch.randint(train_y.numel(), (batch_size,), generator=batch_gen, device=train_y.device)
                        opt.zero_grad(set_to_none=True)
                        loss = loss_fn(model(train_x[idx]), train_y[idx])
                        loss.backward()
                        opt.step()
                else:
                    _, _, loader, _, _ = _mnist_loaders(config["dataset"], config["training"]["batch_size"], repl, seed + r)
                    data_iter = iter(loader)
                    for t in range(steps + 1):
                        vec = parameters_to_vector(model.parameters()).detach() - ref_vec
                        arr[r, t] = (vec @ q).cpu().numpy()
                        if t == steps:
                            break
                        try:
                            x, y = next(data_iter)
                        except StopIteration:
                            data_iter = iter(loader)
                            x, y = next(data_iter)
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad(set_to_none=True)
                        loss = loss_fn(model(x), y)
                        loss.backward()
                        opt.step()
            else:
                cached_full_grad = full_grad
                if method == "standard_langevin":
                    cached_noise_std = ref_noise_std
                else:
                    cached_noise_std = _modified_langevin_noise_std(ref_noise_std, full_grad)
                for t in range(steps + 1):
                    vec = parameters_to_vector(model.parameters()).detach() - ref_vec
                    arr[r, t] = (vec @ q).cpu().numpy()
                    if t == steps:
                        break
                    for substep in range(langevin_substeps):
                        substep_index = t * langevin_substeps + substep
                        current = parameters_to_vector(model.parameters()).detach()
                        if langevin_drift_mode == "current" and substep_index % langevin_coefficient_update_every == 0:
                            cached_full_grad = _grad_vector(model, full_x, full_y, loss_fn)
                            if method == "standard_langevin":
                                cached_noise_std = ref_noise_std
                            else:
                                if langevin_noise_mode == "reference":
                                    centered_noise_std = ref_noise_std
                                else:
                                    centered_noise_std = _estimate_minibatch_noise_std(
                                        model,
                                        cached_full_grad,
                                        config,
                                        device,
                                        seed + 70000 + langevin_seed_stride * r + substep_index,
                                        langevin_noise_batches,
                                        loss_fn,
                                    )
                                cached_noise_std = _modified_langevin_noise_std(centered_noise_std, cached_full_grad)
                        noise = cached_noise_std * torch.randn_like(current)
                        vector_to_parameters(current - langevin_dt * cached_full_grad + langevin_noise_scale * noise, model.parameters())
            if val_tensors is not None:
                final_losses[method].append(_eval_classification_tensors(model, *val_tensors)["loss"])
            else:
                final_losses[method].append(_eval_classification(model, val_loader, device)["loss"])
        trajectories[method] = arr
    rows = []
    metrics: dict[str, Any] = {}
    mean_paths = {method: arr.mean(axis=0) for method, arr in trajectories.items()}
    sgd_mean_path = mean_paths.get("sgd_replacement")
    for method, arr in trajectories.items():
        var = arr.var(axis=0, ddof=1).mean(axis=1)
        mean, std, lo, hi = mean_std_ci(arr.mean(axis=2), axis=0)
        mean_path = mean_paths[method]
        if sgd_mean_path is None:
            mean_path_error = np.zeros(steps + 1, dtype=np.float64)
        else:
            mean_path_error = np.linalg.norm(mean_path - sgd_mean_path, axis=1)
        for t in range(steps + 1):
            rows.append({
                "step": t,
                "method": method,
                "mean_projection": mean[t],
                "std_projection": std[t],
                "ci95_low": lo[t],
                "ci95_high": hi[t],
                "mean_direction_variance": var[t],
                "mean_path_error_to_sgd": float(mean_path_error[t]),
                "mean_coord_1": float(mean_path[t, 0]),
                "mean_coord_2": float(mean_path[t, 1]) if mean_path.shape[1] > 1 else 0.0,
            })
        metrics[f"{method}_final_variance"] = float(var[-1])
        metrics[f"{method}_final_mean_path_error_to_sgd"] = float(mean_path_error[-1])
        loss_mean, loss_std, loss_lo, loss_hi = mean_std_ci(np.asarray(final_losses[method]), axis=0)
        metrics[f"{method}_final_loss_mean"] = float(loss_mean)
        metrics[f"{method}_final_loss_std"] = float(loss_std)
        metrics[f"{method}_final_loss_ci95_low"] = float(loss_lo)
        metrics[f"{method}_final_loss_ci95_high"] = float(loss_hi)
    if "sgd_replacement" in trajectories and "standard_langevin" in trajectories:
        metrics["wasserstein_standard_to_sgd"] = wasserstein_1d(trajectories["sgd_replacement"][:, -1, :], trajectories["standard_langevin"][:, -1, :])
    if "sgd_replacement" in trajectories and "modified_langevin" in trajectories:
        metrics["wasserstein_modified_to_sgd"] = wasserstein_1d(trajectories["sgd_replacement"][:, -1, :], trajectories["modified_langevin"][:, -1, :])
    metrics["reference_full_gradient_norm"] = grad_norm
    metrics["langevin_substeps"] = langevin_substeps
    metrics["langevin_dt"] = float(langevin_dt)
    metrics["langevin_time_per_sgd_step"] = float(lr)
    metrics["langevin_noise_scale"] = float(langevin_noise_scale)
    metrics["langevin_coefficient_update_every"] = langevin_coefficient_update_every
    metrics["langevin_noise_mode"] = langevin_noise_mode
    metrics["langevin_drift_mode"] = langevin_drift_mode
    metrics["langevin_discretization"] = "Euler-Maruyama with M*dt=eta and noise scale sqrt(eta*dt)"
    metrics["pass"] = bool(metrics.get("wasserstein_modified_to_sgd", np.inf) < metrics.get("wasserstein_standard_to_sgd", np.inf))
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", **trajectories)
    return metrics, trajectories, final_losses


def _ensemble_mlp(config: dict[str, Any], result_dir: Path, nonstationary: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, list[float]]]:
    lr_grid = config.get("ensemble", {}).get("lr_grid")
    if not lr_grid:
        return _ensemble_mlp_single(config, result_dir, nonstationary=nonstationary)

    all_rows: list[dict[str, Any]] = []
    all_metrics: dict[str, Any] = {
        "lr_grid": [float(x) for x in lr_grid],
        "n_learning_rates": len(lr_grid),
    }
    all_trajs: dict[str, np.ndarray] = {}
    all_losses: dict[str, list[float]] = {}
    modified_better = []
    for lr_i, lr in enumerate(lr_grid):
        lr = float(lr)
        cfg_lr = copy.deepcopy(config)
        cfg_lr["ensemble"]["lr"] = lr
        subdir = result_dir / f"_lr_{lr_i}_{lr:g}"
        metrics, trajs, losses = _ensemble_mlp_single(cfg_lr, subdir, nonstationary=nonstationary)
        label = f"lr={lr:g}"
        for row in _read_csv_rows(subdir / "figure_data.csv"):
            row["lr"] = lr
            row["method_base"] = row["method"]
            row["method"] = f"{row['method']} {label}"
            all_rows.append(row)
        for k, v in metrics.items():
            all_metrics[f"lr_{lr:g}_{k}"] = v
        if "wasserstein_standard_to_sgd" in metrics and "wasserstein_modified_to_sgd" in metrics:
            modified_better.append(metrics["wasserstein_modified_to_sgd"] < metrics["wasserstein_standard_to_sgd"])
            rel_std = float(metrics["wasserstein_standard_to_sgd"])
            rel_mod = float(metrics["wasserstein_modified_to_sgd"])
            all_metrics[f"lr_{lr:g}_relative_error_standard"] = rel_std
            all_metrics[f"lr_{lr:g}_relative_error_modified"] = rel_mod
            all_metrics[f"lr_{lr:g}_relative_improvement"] = float((rel_std - rel_mod) / max(rel_std, 1e-12))
            all_metrics[f"lr_{lr:g}_modified_closer_ratio"] = float(
                metrics["wasserstein_modified_to_sgd"] / max(metrics["wasserstein_standard_to_sgd"], 1e-12)
            )
        for name, arr in trajs.items():
            all_trajs[f"{name}_lr_{lr:g}".replace(".", "p")] = arr
        for name, vals in losses.items():
            all_losses[f"{name}_lr_{lr:g}"] = vals
    if modified_better:
        all_metrics["modified_langevin_closer_fraction"] = float(np.mean(modified_better))
        high_lr = float(max(lr_grid))
        high_standard = all_metrics.get(f"lr_{high_lr:g}_wasserstein_standard_to_sgd", np.inf)
        high_modified = all_metrics.get(f"lr_{high_lr:g}_wasserstein_modified_to_sgd", np.inf)
        all_metrics["high_lr"] = high_lr
        all_metrics["high_lr_modified_langevin_closer"] = bool(high_modified < high_standard)
        all_metrics["modified_better_at_high_lr"] = bool(high_modified < high_standard)
        all_metrics["relative_error_standard"] = float(high_standard)
        all_metrics["relative_error_modified"] = float(high_modified)
        all_metrics["relative_improvement"] = float((high_standard - high_modified) / max(high_standard, 1e-12))
        all_metrics["high_lr_modified_langevin_error_ratio"] = float(high_modified / max(high_standard, 1e-12))
    all_metrics["pass"] = bool(all_metrics.get("high_lr_modified_langevin_closer", False))
    write_csv(result_dir / "figure_data.csv", all_rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", **all_trajs)
    return all_metrics, all_trajs, all_losses


def run_exp1b(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    metrics, _, _ = _ensemble_mlp(config, result_dir, nonstationary=True)
    return metrics


def run_exp2(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    def agreement_metrics(measured_vals: np.ndarray, pred_vals: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        measured_vals = np.asarray(measured_vals, dtype=np.float64)
        pred_vals = np.asarray(pred_vals, dtype=np.float64)
        mask = np.isfinite(measured_vals) & np.isfinite(pred_vals) & (measured_vals >= 0.0) & (pred_vals >= 0.0)
        measured_vals = measured_vals[mask]
        pred_vals = pred_vals[mask]
        eps = 1e-12
        if len(measured_vals) < 3:
            return {
                "n": float(len(measured_vals)),
                "relative_error": float("nan"),
                "log_correlation": float("nan"),
                "log_mse": float("nan"),
                "best_scalar": float("nan"),
                "relative_error_best_scalar": float("nan"),
                "permutation_pvalue_log_correlation": float("nan"),
            }
        log_measured = np.log(measured_vals + eps)
        log_pred = np.log(pred_vals + eps)
        corr = float(np.corrcoef(log_measured, log_pred)[0, 1])
        rel = float(np.linalg.norm(measured_vals - pred_vals) / max(np.linalg.norm(measured_vals), eps))
        mse_log = float(np.mean((log_measured - log_pred) ** 2))
        scalar = float(np.dot(pred_vals, measured_vals) / max(np.dot(pred_vals, pred_vals), eps))
        rel_scaled = float(np.linalg.norm(measured_vals - scalar * pred_vals) / max(np.linalg.norm(measured_vals), eps))
        perm_corrs = []
        for _ in range(200):
            perm_corrs.append(float(np.corrcoef(log_measured, rng.permutation(log_pred))[0, 1]))
        pvalue = float((1.0 + np.sum(np.asarray(perm_corrs) >= corr)) / (len(perm_corrs) + 1.0))
        return {
            "n": float(len(measured_vals)),
            "relative_error": rel,
            "log_correlation": corr,
            "log_mse": mse_log,
            "best_scalar": scalar,
            "relative_error_best_scalar": rel_scaled,
            "permutation_pvalue_log_correlation": pvalue,
        }

    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    rng = np.random.default_rng(seed + 9000)
    model_kind = config.get("model", {}).get("type", "mlp")
    if model_kind == "nanogpt":
        train_ds, val_ds, train_loader, full_loader, _ = _shakespeare_loaders(
            config["dataset"], config["training"]["batch_size"], config["training"].get("replacement", True), seed
        )
        model = _nanogpt_model(config, device)
        checkpoint_path = config.get("training", {}).get("checkpoint_path")
        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            opt_ref = torch.optim.SGD(model.parameters(), lr=float(config["training"]["lr"]))
            data_iter_ref = iter(train_loader)
            for _ in range(int(config["training"].get("reference_steps", 100))):
                try:
                    x_ref, y_ref = next(data_iter_ref)
                except StopIteration:
                    data_iter_ref = iter(train_loader)
                    x_ref, y_ref = next(data_iter_ref)
                x_ref, y_ref = x_ref.to(device), y_ref.to(device)
                opt_ref.zero_grad(set_to_none=True)
                _loss_value(model, x_ref, y_ref, None).backward()
                opt_ref.step()
        loss_fn: nn.Module | None = None
    else:
        model, loaders = _train_reference_mlp(config, device, seed)
        _, _, train_loader, full_loader, _ = loaders
        loss_fn = nn.CrossEntropyLoss()
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    reference_full_grad = _grad_vector_general(model, full_x, full_y, loss_fn)
    reference_full_grad_norm = float(torch.linalg.norm(reference_full_grad).item())
    reference_full_loss = float(_loss_value(model, full_x, full_y, loss_fn).detach().cpu().item())
    hessian_mean_batches = int(config["analysis"].get("hessian_mean_batches", 1))
    hessians_for_mean = []
    for i, (hx, hy) in enumerate(full_loader):
        if i >= hessian_mean_batches:
            break
        hessians_for_mean.append(_hessian_full_batch_general(model, hx.to(device), hy.to(device), loss_fn).cpu())
    h = torch.stack(hessians_for_mean).mean(dim=0) if len(hessians_for_mean) > 1 else hessians_for_mean[0]
    eigvals, eigvecs = torch.linalg.eigh(h)
    grads = []
    max_batches = int(config["analysis"].get("gradient_batches", 16))
    for i, (x, y) in enumerate(train_loader):
        if i >= max_batches:
            break
        grads.append((_grad_vector_general(model, x.to(device), y.to(device), loss_fn).cpu() @ eigvecs).numpy())
    grads_arr = np.stack(grads)
    g_var = grads_arr.var(axis=0, ddof=1)
    eig_np = eigvals.numpy()
    hessian_noise_batches = int(config["analysis"].get("hessian_noise_batches", 0))
    hessian_noise_metrics: dict[str, float] = {}
    hessian_noise_rows: list[dict[str, Any]] = []
    rotated_hessians = None
    if hessian_noise_batches > 0:
        hs = []
        for i, (x, y) in enumerate(train_loader):
            if i >= hessian_noise_batches:
                break
            hb = _hessian_full_batch_general(model, x.to(device), y.to(device), loss_fn).cpu()
            rb = eigvecs.T @ hb @ eigvecs
            hs.append(rb.numpy())
        if hs:
            rotated_hessians = np.stack(hs)
            diag = np.diagonal(rotated_hessians, axis1=1, axis2=2)
            off = rotated_hessians.copy()
            idx = np.arange(off.shape[1])
            off[:, idx, idx] = 0.0
            diag_mass = np.sum(diag**2, axis=1)
            off_mass = np.sum(off**2, axis=(1, 2))
            total_mass = diag_mass + off_mass + 1e-12
            diag_fraction = diag_mass / total_mass
            offdiag_fraction = off_mass / total_mass
            offdiag_diag_ratio = np.sqrt(off_mass / np.maximum(diag_mass, 1e-12))
            diag_fluct = diag - diag.mean(axis=0, keepdims=True)
            if diag_fluct.shape[0] >= 3:
                corr = np.corrcoef(diag_fluct, rowvar=False)
                corr_abs = np.abs(corr[np.triu_indices_from(corr, k=1)])
                mean_abs_diag_corr = float(np.nanmean(corr_abs))
                p95_abs_diag_corr = float(np.nanpercentile(corr_abs, 95))
            else:
                mean_abs_diag_corr = float("nan")
                p95_abs_diag_corr = float("nan")
            hessian_noise_metrics = {
                "hessian_noise_batches": float(len(hs)),
                "hessian_diag_fraction_mean": float(np.mean(diag_fraction)),
                "hessian_diag_fraction_p05": float(np.percentile(diag_fraction, 5)),
                "hessian_offdiag_fraction_mean": float(np.mean(offdiag_fraction)),
                "hessian_offdiag_fraction_p95": float(np.percentile(offdiag_fraction, 95)),
                "hessian_offdiag_diag_ratio_mean": float(np.mean(offdiag_diag_ratio)),
                "hessian_offdiag_diag_ratio_p95": float(np.percentile(offdiag_diag_ratio, 95)),
                "hessian_diag_fluctuation_mean_abs_correlation": mean_abs_diag_corr,
                "hessian_diag_fluctuation_p95_abs_correlation": p95_abs_diag_corr,
            }
            for b in range(len(hs)):
                hessian_noise_rows.append({
                    "batch": b,
                    "diag_fraction": float(diag_fraction[b]),
                    "offdiag_fraction": float(offdiag_fraction[b]),
                    "offdiag_diag_ratio": float(offdiag_diag_ratio[b]),
                    "diag_mass": float(diag_mass[b]),
                    "offdiag_mass": float(off_mass[b]),
                })
    # Parameter-free proxy for Eq. (32): use independently measured gradient-noise
    # power in the Hessian eigenbasis and no post-hoc variance rescaling.
    pred_all = float(config["ensemble"]["lr"]) * g_var / (2.0 * np.abs(eig_np) + 1e-6)

    ref_vec = parameters_to_vector(model.parameters()).detach().clone()
    ens = int(config["ensemble"]["n_runs"])
    steps = int(config["ensemble"]["steps"])
    lr = float(config["ensemble"]["lr"])
    projected = np.zeros((ens, eigvecs.shape[1]), dtype=np.float32)
    for r in range(ens):
        torch.manual_seed(seed + 20000 + r)
        run_model = _nanogpt_model(config, device) if model_kind == "nanogpt" else _mlp_model(device)
        vector_to_parameters(ref_vec.clone(), run_model.parameters())
        if model_kind == "nanogpt":
            _, _, run_loader, _, _ = _shakespeare_loaders(config["dataset"], config["training"]["batch_size"], True, seed + r)
        else:
            _, _, run_loader, _, _ = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, seed + r)
        opt = torch.optim.SGD(run_model.parameters(), lr=lr)
        data_iter = iter(run_loader)
        for _ in range(steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(run_loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = _loss_value(run_model, x, y, loss_fn)
            loss.backward()
            opt.step()
        delta = (parameters_to_vector(run_model.parameters()).detach().cpu() - ref_vec.cpu()) @ eigvecs
        projected[r] = delta.numpy()
    measured_all = projected.var(axis=0, ddof=1)

    n_dirs = int(config["analysis"].get("n_directions", 16))
    selected = np.argsort(np.abs(eig_np))[-n_dirs:]
    measured = measured_all[selected]
    pred_sel = pred_all[selected]
    random_selected = rng.choice(len(eig_np), size=min(n_dirs, len(eig_np)), replace=False)
    top_metrics = agreement_metrics(measured, pred_sel, rng)
    all_metrics = agreement_metrics(measured_all, pred_all, rng)
    random_metrics = agreement_metrics(measured_all[random_selected], pred_all[random_selected], rng)

    split_perm = rng.permutation(len(selected))
    train_idx = split_perm[: max(2, len(split_perm) // 2)]
    test_idx = split_perm[max(2, len(split_perm) // 2):]
    if len(test_idx) >= 2:
        pred_train = pred_sel[train_idx].astype(np.float64)
        meas_train = measured[train_idx].astype(np.float64)
        heldout_scalar = float(np.dot(pred_train, meas_train) / max(np.dot(pred_train, pred_train), 1e-12))
        heldout_rel = float(np.linalg.norm(measured[test_idx] - heldout_scalar * pred_sel[test_idx]) / max(np.linalg.norm(measured[test_idx]), 1e-12))
        heldout_log_corr = float(np.corrcoef(np.log(measured[test_idx] + 1e-12), np.log(pred_sel[test_idx] + 1e-12))[0, 1])
    else:
        heldout_scalar = float("nan")
        heldout_rel = float("nan")
        heldout_log_corr = float("nan")

    gamma = float(np.mean(g_var[selected] / (np.abs(eig_np[selected]) + 1e-6)))
    gamma_seed_list = [int(s) for s in config["analysis"].get("gamma_seeds", [seed])]
    gamma_rows: list[dict[str, Any]] = []
    gamma_values = []
    for gamma_seed in gamma_seed_list:
        if model_kind == "nanogpt":
            _, _, gamma_loader, _, _ = _shakespeare_loaders(config["dataset"], config["training"]["batch_size"], True, gamma_seed)
        else:
            _, _, gamma_loader, _, _ = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, gamma_seed)
        gamma_grads = []
        for i, (x, y) in enumerate(gamma_loader):
            if i >= max_batches:
                break
            gamma_grads.append((_grad_vector_general(model, x.to(device), y.to(device), loss_fn).cpu() @ eigvecs).numpy())
        if len(gamma_grads) >= 2:
            gamma_arr = np.stack(gamma_grads)
            gamma_g_var = gamma_arr.var(axis=0, ddof=1)
            gamma_value = float(np.mean(gamma_g_var[selected] / (np.abs(eig_np[selected]) + 1e-6)))
        else:
            gamma_value = float("nan")
        gamma_values.append(gamma_value)
        gamma_rows.append({"seed": gamma_seed, "gamma_estimate": gamma_value, "gradient_batches": min(max_batches, len(gamma_grads))})
    rows = []
    for out_i, eig_i in enumerate(selected):
        rows.append({"direction": int(eig_i), "eigenvalue": float(eig_np[eig_i]), "measured_variance": float(measured[out_i]), "predicted_variance": float(pred_sel[out_i])})
    write_csv(result_dir / "figure_data.csv", rows)
    write_csv(result_dir / "gamma_estimates.csv", gamma_rows)
    sanity_rows = []
    for name, values in [("top_abs_eigenvalue", top_metrics), ("all_directions", all_metrics), ("random_directions", random_metrics)]:
        row = {"direction_set": name}
        row.update(values)
        sanity_rows.append(row)
    if hessian_noise_rows:
        write_csv(result_dir / "hessian_noise_diagnostics.csv", hessian_noise_rows)
        sanity_rows.append({
            "direction_set": "hessian_noise_structure",
            "n": hessian_noise_metrics.get("hessian_noise_batches", float("nan")),
            "relative_error": hessian_noise_metrics.get("hessian_offdiag_diag_ratio_mean", float("nan")),
            "log_correlation": hessian_noise_metrics.get("hessian_diag_fluctuation_mean_abs_correlation", float("nan")),
            "log_mse": float("nan"),
            "best_scalar": hessian_noise_metrics.get("hessian_offdiag_fraction_mean", float("nan")),
            "relative_error_best_scalar": hessian_noise_metrics.get("hessian_offdiag_fraction_p95", float("nan")),
            "permutation_pvalue_log_correlation": float("nan"),
        })
    sanity_rows.append({
        "direction_set": "top_abs_train_scalar_heldout",
        "n": float(len(test_idx)),
        "relative_error": heldout_rel,
        "log_correlation": heldout_log_corr,
        "log_mse": float("nan"),
        "best_scalar": heldout_scalar,
        "relative_error_best_scalar": heldout_rel,
        "permutation_pvalue_log_correlation": float("nan"),
    })
    write_csv(result_dir / "sanity_checks.csv", sanity_rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        hessian=h.numpy(),
        eigenvalues=eig_np,
        grad_eig=grads_arr,
        measured_all=measured_all,
        predicted_all=pred_all,
        measured=measured,
        predicted=pred_sel,
        selected=selected,
        random_selected=random_selected,
        rotated_hessians=np.asarray(rotated_hessians) if rotated_hessians is not None else np.empty((0, 0, 0)),
        gamma_estimates=np.asarray(gamma_values, dtype=np.float64),
    )
    gamma_arr_np = np.asarray([g for g in gamma_values if np.isfinite(g)], dtype=np.float64)
    gamma_mean, gamma_std, gamma_lo, gamma_hi = mean_std_ci(gamma_arr_np, axis=0) if len(gamma_arr_np) else (np.nan, np.nan, np.nan, np.nan)
    metrics = {
        "reference_full_loss": reference_full_loss,
        "reference_full_grad_norm": reference_full_grad_norm,
        "gamma_estimate": gamma,
        "gamma_seed_count": int(len(gamma_arr_np)),
        "gamma_mean": float(gamma_mean),
        "gamma_std": float(gamma_std),
        "gamma_cv": float(gamma_std / max(abs(gamma_mean), 1e-12)) if np.isfinite(gamma_mean) else float("nan"),
        "gamma_min": float(np.min(gamma_arr_np)) if len(gamma_arr_np) else float("nan"),
        "gamma_max": float(np.max(gamma_arr_np)) if len(gamma_arr_np) else float("nan"),
        "gamma_estimate_mean": float(gamma_mean),
        "gamma_estimate_std": float(gamma_std),
        "gamma_estimate_ci95_low": float(gamma_lo),
        "gamma_estimate_ci95_high": float(gamma_hi),
        "gamma_estimate_cv": float(gamma_std / max(abs(gamma_mean), 1e-12)) if np.isfinite(gamma_mean) else float("nan"),
        "relative_error": top_metrics["relative_error"],
        "log_correlation": top_metrics["log_correlation"],
        "log_mse": top_metrics["log_mse"],
        "top_relative_error_best_scalar": top_metrics["relative_error_best_scalar"],
        "top_best_scalar": top_metrics["best_scalar"],
        "top_permutation_pvalue_log_correlation": top_metrics["permutation_pvalue_log_correlation"],
        "all_directions_log_correlation": all_metrics["log_correlation"],
        "all_directions_relative_error": all_metrics["relative_error"],
        "random_directions_log_correlation": random_metrics["log_correlation"],
        "random_directions_relative_error": random_metrics["relative_error"],
        "heldout_scalar_from_half_top_dirs": heldout_scalar,
        "heldout_relative_error_after_train_scalar": heldout_rel,
        "heldout_log_correlation": heldout_log_corr,
    }
    metrics.update(hessian_noise_metrics)
    metrics["pass"] = bool(metrics["log_correlation"] > 0.2 and metrics["top_permutation_pvalue_log_correlation"] < 0.05)
    return metrics


def run_exp3(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    metrics, trajs, _ = _ensemble_mlp(config, result_dir, nonstationary=False)
    if "sgd_replacement" in trajs and "sgd_no_replacement" in trajs:
        rep = trajs["sgd_replacement"]
        norep = trajs["sgd_no_replacement"]
        metrics["replacement_vs_no_replacement_wasserstein"] = wasserstein_1d(rep[:, -1, :], norep[:, -1, :])
        metrics["replacement_late_slope"] = float(np.polyfit(np.arange(rep.shape[1])[-10:], rep.var(axis=0).mean(axis=1)[-10:], 1)[0])
        metrics["no_replacement_late_slope"] = float(np.polyfit(np.arange(norep.shape[1])[-10:], norep.var(axis=0).mean(axis=1)[-10:], 1)[0])
    metrics["pass"] = bool(metrics.get("replacement_vs_no_replacement_wasserstein", 0.0) > 0.0)
    return metrics


def run_exp4(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    if not config.get("enabled", False):
        rows = [{"status": "not_run", "reason": "disabled_by_default_long_experiment"}]
        write_csv(result_dir / "figure_data.csv", rows)
        np.savez_compressed(result_dir / "raw_outputs.npz", placeholder=np.array([0]))
        return {"status": "skipped", "pass": None, "reason": "EXP4 is long; set enabled=true to run the lite HVP experiment."}

    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    _, _, train_loader, full_loader, val_loader = _mnist_loaders(
        config["dataset"], config["training"]["batch_size"], config["training"].get("replacement", True), seed
    )
    model = _configured_mlp_model(config, device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=float(config["training"]["lr"]))
    data_iter = iter(train_loader)
    for _ in range(int(config["training"].get("reference_steps", 100))):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    ref_vec = parameters_to_vector(model.parameters()).detach().clone()
    n_params = ref_vec.numel()

    power_iters = int(config["analysis"].get("power_iterations", 12))
    torch.manual_seed(seed + 123)
    v = torch.randn(n_params, device=device)
    v = v / torch.linalg.norm(v)
    for _ in range(power_iters):
        hv = _hvp_full_batch(model, full_x, full_y, loss_fn, v)
        norm = torch.linalg.norm(hv)
        if float(norm.item()) == 0.0:
            break
        v = hv / norm
    hv = _hvp_full_batch(model, full_x, full_y, loss_fn, v)
    top_eigenvalue = float(torch.dot(v, hv).item())
    sharp_dir = v.detach()

    n_random = int(config["analysis"].get("random_flat_directions", 8))
    torch.manual_seed(seed + 456)
    random_basis = torch.randn(n_params, n_random, device=device)
    random_basis = random_basis - sharp_dir[:, None] * (sharp_dir @ random_basis)[None, :]
    random_basis, _ = torch.linalg.qr(random_basis)

    ens = int(config["ensemble"]["n_runs"])
    steps = int(config["ensemble"]["steps"])
    lr = float(config["ensemble"]["lr"])
    sharp_proj = np.zeros((ens, steps + 1), dtype=np.float32)
    random_proj = np.zeros((ens, steps + 1, n_random), dtype=np.float32)
    final_losses = []
    for r in range(ens):
        torch.manual_seed(seed + 50000 + r)
        run_model = _configured_mlp_model(config, device)
        vector_to_parameters(ref_vec.clone(), run_model.parameters())
        _, _, run_loader, _, _ = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, seed + r)
        run_iter = iter(run_loader)
        run_opt = torch.optim.SGD(run_model.parameters(), lr=lr)
        for t in range(steps + 1):
            delta = parameters_to_vector(run_model.parameters()).detach() - ref_vec
            sharp_proj[r, t] = float(delta @ sharp_dir)
            random_proj[r, t] = (delta @ random_basis).cpu().numpy()
            if t == steps:
                break
            try:
                x, y = next(run_iter)
            except StopIteration:
                run_iter = iter(run_loader)
                x, y = next(run_iter)
            x, y = x.to(device), y.to(device)
            run_opt.zero_grad(set_to_none=True)
            loss = loss_fn(run_model(x), y)
            loss.backward()
            run_opt.step()
        final_losses.append(_eval_classification(run_model, val_loader, device)["loss"])

    sharp_var = sharp_proj.var(axis=0, ddof=1)
    random_var = random_proj.var(axis=0, ddof=1).mean(axis=1)
    rows = []
    for t in range(steps + 1):
        rows.append({
            "step": t,
            "method": "sharp_top_hvp",
            "mean_projection": float(sharp_proj[:, t].mean()),
            "std_projection": float(sharp_proj[:, t].std(ddof=1)),
            "ci95_low": float(sharp_proj[:, t].mean() - 1.96 * sharp_proj[:, t].std(ddof=1) / math.sqrt(max(ens, 1))),
            "ci95_high": float(sharp_proj[:, t].mean() + 1.96 * sharp_proj[:, t].std(ddof=1) / math.sqrt(max(ens, 1))),
            "mean_direction_variance": float(sharp_var[t]),
            "mean_path_error_to_sgd": 0.0,
            "mean_coord_1": float(sharp_proj[:, t].mean()),
            "mean_coord_2": 0.0,
        })
        rows.append({
            "step": t,
            "method": "random_flat_proxy",
            "mean_projection": float(random_proj[:, t, :].mean()),
            "std_projection": float(random_proj[:, t, :].mean(axis=1).std(ddof=1)),
            "ci95_low": float(random_proj[:, t, :].mean() - 1.96 * random_proj[:, t, :].mean(axis=1).std(ddof=1) / math.sqrt(max(ens, 1))),
            "ci95_high": float(random_proj[:, t, :].mean() + 1.96 * random_proj[:, t, :].mean(axis=1).std(ddof=1) / math.sqrt(max(ens, 1))),
            "mean_direction_variance": float(random_var[t]),
            "mean_path_error_to_sgd": 0.0,
            "mean_coord_1": float(random_proj[:, t, 0].mean()),
            "mean_coord_2": float(random_proj[:, t, 1].mean()) if n_random > 1 else 0.0,
        })

    tail = max(10, steps // 5)
    xs = np.arange(steps + 1)
    sharp_slope = float(np.polyfit(xs[-tail:], sharp_var[-tail:], 1)[0])
    random_slope = float(np.polyfit(xs[-tail:], random_var[-tail:], 1)[0])
    loss_mean, loss_std, loss_lo, loss_hi = mean_std_ci(np.asarray(final_losses), axis=0)
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        sharp_projection=sharp_proj,
        random_projection=random_proj,
        sharp_direction=sharp_dir.cpu().numpy(),
        random_basis=random_basis.cpu().numpy(),
    )
    return {
        "status": "exp4_lite_hvp_power_iteration",
        "n_parameters": int(n_params),
        "top_hvp_eigenvalue_estimate": top_eigenvalue,
        "sharp_final_variance": float(sharp_var[-1]),
        "random_flat_proxy_final_variance": float(random_var[-1]),
        "sharp_late_slope": sharp_slope,
        "random_flat_proxy_late_slope": random_slope,
        "final_loss_mean": float(loss_mean),
        "final_loss_std": float(loss_std),
        "final_loss_ci95_low": float(loss_lo),
        "final_loss_ci95_high": float(loss_hi),
        "pass": bool(np.isfinite(top_eigenvalue) and random_var[-1] > 0.0),
    }


def _run_lr_scaling_buckets(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    model, loaders = _train_reference_mlp(config, device, seed)
    _, _, train_loader, full_loader, _ = loaders
    loss_fn = nn.CrossEntropyLoss()
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    h = _hessian_full_batch(model, full_x, full_y, loss_fn).cpu()
    eigvals, eigvecs = torch.linalg.eigh(h)
    eig_np = eigvals.numpy()
    n_flat = int(config["analysis"].get("n_flat_directions", 4))
    n_sharp = int(config["analysis"].get("n_sharp_directions", 4))
    n_diffusive = int(config["analysis"].get("n_diffusive_directions", n_flat))
    flat_idx = np.argsort(np.abs(eig_np))[:n_flat]
    positive = np.where(eig_np > 0)[0]
    sharp_idx = positive[np.argsort(eig_np[positive])[-n_sharp:]] if len(positive) >= n_sharp else np.argsort(np.abs(eig_np))[-n_sharp:]
    grad_batches = int(config["analysis"].get("gradient_batches", 8))
    grad_eig = []
    for i, (gx, gy) in enumerate(train_loader):
        if i >= grad_batches:
            break
        grad_eig.append((_grad_vector(model, gx.to(device), gy.to(device), loss_fn).cpu() @ eigvecs).numpy())
    grad_eig_arr = np.stack(grad_eig)
    grad_var = grad_eig_arr.var(axis=0, ddof=1)
    predicted_var_proxy = grad_var / (2.0 * np.abs(eig_np) + 1e-6)
    diffusive_idx = np.argsort(predicted_var_proxy)[-n_diffusive:]
    bases = {
        "flat": eigvecs[:, flat_idx].to(device),
        "sharp": eigvecs[:, sharp_idx].to(device),
        "diffusive": eigvecs[:, diffusive_idx].to(device),
    }
    ref_vec = parameters_to_vector(model.parameters()).detach().clone()
    lrs = [float(x) for x in config["ensemble"].get("lr_grid", [config["ensemble"]["lr"]])]
    ens = int(config["ensemble"].get("n_runs", 4))
    steps = int(config["ensemble"]["steps"])
    eval_every = int(config["analysis"].get("eval_every", max(1, steps // 10)))
    eval_steps = list(range(0, steps + 1, eval_every))
    if eval_steps[-1] != steps:
        eval_steps.append(steps)
    rows: list[dict[str, Any]] = []
    raw: dict[str, np.ndarray] = {"eigenvalues": eig_np, "flat_indices": flat_idx, "sharp_indices": sharp_idx}
    metrics: dict[str, Any] = {"lr_grid": lrs, "n_runs": ens, "steps": steps}
    slope_by_lr: dict[float, float] = {}

    for lr in lrs:
        proj = {bucket: np.zeros((ens, len(eval_steps), basis.shape[1]), dtype=np.float32) for bucket, basis in bases.items()}
        for r in range(ens):
            torch.manual_seed(seed + 33000 + r + int(round(lr * 100000)))
            run_model = _mlp_model(device)
            vector_to_parameters(ref_vec.clone(), run_model.parameters())
            _, _, run_loader, _, _ = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, seed + r)
            data_iter = iter(run_loader)
            eval_pos = 0
            for t in range(steps + 1):
                if t == eval_steps[eval_pos]:
                    delta = parameters_to_vector(run_model.parameters()).detach() - ref_vec
                    for bucket, basis in bases.items():
                        proj[bucket][r, eval_pos] = (delta @ basis).detach().cpu().numpy()
                    eval_pos += 1
                    if eval_pos >= len(eval_steps):
                        break
                if t == steps:
                    break
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(run_loader)
                    x, y = next(data_iter)
                x, y = x.to(device), y.to(device)
                grad = _grad_vector(run_model, x, y, loss_fn)
                current = parameters_to_vector(run_model.parameters()).detach()
                vector_to_parameters(current - lr * grad, run_model.parameters())
        for bucket, arr in proj.items():
            raw[f"{bucket}_lr_{lr:g}".replace(".", "p")] = arr
            direction_var = arr.var(axis=0, ddof=1)
            mean_var = direction_var.mean(axis=1)
            std_var = direction_var.std(axis=1, ddof=1) if direction_var.shape[1] > 1 else np.zeros_like(mean_var)
            ci_half = 1.96 * std_var / math.sqrt(max(direction_var.shape[1], 1))
            for j, step in enumerate(eval_steps):
                rows.append({
                    "step": step,
                    "lr": lr,
                    "bucket": bucket,
                    "method": f"{bucket}_lr={lr:g}",
                    "mean_direction_variance": float(mean_var[j]),
                    "std_direction_variance": float(std_var[j]),
                    "ci95_low": float(mean_var[j] - ci_half[j]),
                    "ci95_high": float(mean_var[j] + ci_half[j]),
                })
            tail = max(3, len(eval_steps) // 3)
            slope = float(np.polyfit(np.asarray(eval_steps[-tail:], dtype=np.float64), mean_var[-tail:], 1)[0])
            plateau = float(mean_var[-tail:].mean())
            metrics[f"lr_{lr:g}_{bucket}_late_slope"] = slope
            metrics[f"lr_{lr:g}_{bucket}_late_plateau_mean"] = plateau
            if bucket == "flat":
                slope_by_lr[lr] = slope
    if len(slope_by_lr) >= 3:
        xs = np.asarray(sorted(slope_by_lr), dtype=np.float64)
        ys = np.asarray([slope_by_lr[x] for x in xs], dtype=np.float64)
        metrics["flat_slope_lr_correlation"] = float(np.corrcoef(xs, ys)[0, 1])
        metrics["flat_slope_vs_lr_fit_slope"] = float(np.polyfit(xs, ys, 1)[0])
    if lrs:
        primary_lr = float(max(lrs))
        flat_slope = float(metrics.get(f"lr_{primary_lr:g}_flat_late_slope", float("nan")))
        sharp_slope = float(metrics.get(f"lr_{primary_lr:g}_sharp_late_slope", float("nan")))
        metrics["primary_lr"] = primary_lr
        metrics["flat_slope"] = flat_slope
        metrics["sharp_slope"] = sharp_slope
        metrics["slope_ratio"] = float(flat_slope / max(abs(sharp_slope), 1e-30)) if np.isfinite(flat_slope) and np.isfinite(sharp_slope) else float("nan")
    metrics["pass"] = bool(metrics.get("flat_slope_lr_correlation", 0.0) > 0.3)
    write_csv(result_dir / "figure_data.csv", rows)
    write_csv(result_dir / "bucketed_statistics.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", **raw)
    return metrics


def run_exp5(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    if config.get("analysis", {}).get("mode") == "lr_scaling":
        return _run_lr_scaling_buckets(config, result_dir)
    metrics, _, _ = _ensemble_mlp(config, result_dir, nonstationary=True)
    stationary_error = metrics.get("wasserstein_standard_to_sgd", np.nan)
    nonstationary_error = metrics.get("wasserstein_modified_to_sgd", np.nan)
    metrics["fit_improvement"] = float(stationary_error - nonstationary_error)
    metrics["pass"] = bool(metrics["fit_improvement"] > 0)
    return metrics


def run_exp6(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    model, loaders = _train_reference_mlp(config, device, seed)
    _, _, _, full_loader, val_loader = loaders
    loss_fn = nn.CrossEntropyLoss()
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    h = _hessian_full_batch(model, full_x, full_y, loss_fn).cpu()
    eigvals, eigvecs = torch.linalg.eigh(h)
    eig_np = eigvals.numpy()
    n_dirs = int(config["analysis"].get("n_directions", 16))
    n_flat = int(config["analysis"].get("n_flat_directions", max(2, n_dirs // 2)))
    n_sharp = int(config["analysis"].get("n_sharp_directions", max(2, n_dirs // 2)))
    flat_idx = np.argsort(np.abs(eig_np))[:n_flat]
    positive_order = np.argsort(eig_np)
    sharp_candidates = positive_order[eig_np[positive_order] > 0]
    if len(sharp_candidates) >= n_sharp:
        sharp_idx = sharp_candidates[-n_sharp:]
    else:
        sharp_idx = np.argsort(np.abs(eig_np))[-n_sharp:]
    flat_basis = eigvecs[:, flat_idx].to(device)
    sharp_basis = eigvecs[:, sharp_idx].to(device)
    ref_vec = parameters_to_vector(model.parameters()).detach().clone()

    ens = int(config["ensemble"]["n_runs"])
    steps = int(config["ensemble"]["steps"])
    lr = float(config["ensemble"]["lr"])
    flat_disp = np.zeros(ens, dtype=np.float64)
    sharp_disp = np.zeros(ens, dtype=np.float64)
    total_disp = np.zeros(ens, dtype=np.float64)
    final_train_loss = np.zeros(ens, dtype=np.float64)
    final_test_loss = np.zeros(ens, dtype=np.float64)
    for r in range(ens):
        torch.manual_seed(seed + 80000 + r)
        run_model = _mlp_model(device)
        vector_to_parameters(ref_vec.clone(), run_model.parameters())
        _, _, run_loader, _, _ = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, seed + r)
        data_iter = iter(run_loader)
        opt = torch.optim.SGD(run_model.parameters(), lr=lr)
        for _ in range(steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(run_loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(run_model(x), y)
            loss.backward()
            opt.step()
        delta = parameters_to_vector(run_model.parameters()).detach() - ref_vec
        flat_disp[r] = float(torch.linalg.norm(delta @ flat_basis).item())
        sharp_disp[r] = float(torch.linalg.norm(delta @ sharp_basis).item())
        total_disp[r] = float(torch.linalg.norm(delta).item())
        final_train_loss[r] = _eval_classification(run_model, full_loader, device)["loss"]
        final_test_loss[r] = _eval_classification(run_model, val_loader, device)["loss"]
    gen_gap = final_test_loss - final_train_loss

    def corr_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        try:
            from scipy.stats import pearsonr, spearmanr

            pr = pearsonr(x, y)
            sr = spearmanr(x, y)
            return {
                "pearson": float(pr.statistic),
                "pearson_pvalue": float(pr.pvalue),
                "spearman": float(sr.statistic),
                "spearman_pvalue": float(sr.pvalue),
            }
        except Exception:
            corr = float(np.corrcoef(x, y)[0, 1])
            return {"pearson": corr, "pearson_pvalue": float("nan"), "spearman": corr, "spearman_pvalue": float("nan")}

    flat_gap = corr_stats(flat_disp, gen_gap)
    flat_test = corr_stats(flat_disp, final_test_loss)
    sharp_gap = corr_stats(sharp_disp, gen_gap)
    total_gap = corr_stats(total_disp, gen_gap)

    x_design = np.column_stack([np.ones(ens), flat_disp, sharp_disp, final_train_loss])
    coef, *_ = np.linalg.lstsq(x_design, gen_gap, rcond=None)
    pred = x_design @ coef
    ss_res = float(np.sum((gen_gap - pred) ** 2))
    ss_tot = float(np.sum((gen_gap - gen_gap.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    rows = []
    for i in range(ens):
        rows.append({
            "run": i,
            "flat_displacement": float(flat_disp[i]),
            "sharp_displacement": float(sharp_disp[i]),
            "total_displacement": float(total_disp[i]),
            "final_train_loss": float(final_train_loss[i]),
            "final_test_loss": float(final_test_loss[i]),
            "generalization_gap": float(gen_gap[i]),
        })
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        hessian=h.numpy(),
        eigenvalues=eig_np,
        flat_indices=flat_idx,
        sharp_indices=sharp_idx,
        flat_displacement=flat_disp,
        sharp_displacement=sharp_disp,
        total_displacement=total_disp,
        final_train_loss=final_train_loss,
        final_test_loss=final_test_loss,
        generalization_gap=gen_gap,
    )
    train_mean, train_std, train_lo, train_hi = mean_std_ci(final_train_loss, axis=0)
    test_mean, test_std, test_lo, test_hi = mean_std_ci(final_test_loss, axis=0)
    gap_mean, gap_std, gap_lo, gap_hi = mean_std_ci(gen_gap, axis=0)
    metrics = {
        "flat_eigenvalue_abs_max": float(np.max(np.abs(eig_np[flat_idx]))),
        "sharp_eigenvalue_min": float(np.min(eig_np[sharp_idx])),
        "sharp_eigenvalue_max": float(np.max(eig_np[sharp_idx])),
        "flat_displacement_mean": float(flat_disp.mean()),
        "sharp_displacement_mean": float(sharp_disp.mean()),
        "total_displacement_mean": float(total_disp.mean()),
        "final_train_loss_mean": float(train_mean),
        "final_train_loss_std": float(train_std),
        "final_train_loss_ci95_low": float(train_lo),
        "final_train_loss_ci95_high": float(train_hi),
        "final_test_loss_mean": float(test_mean),
        "final_test_loss_std": float(test_std),
        "final_test_loss_ci95_low": float(test_lo),
        "final_test_loss_ci95_high": float(test_hi),
        "generalization_gap_mean": float(gap_mean),
        "generalization_gap_std": float(gap_std),
        "generalization_gap_ci95_low": float(gap_lo),
        "generalization_gap_ci95_high": float(gap_hi),
        "pearson_flat_displacement_generalization_gap": flat_gap["pearson"],
        "pearson_flat_displacement_generalization_gap_pvalue": flat_gap["pearson_pvalue"],
        "spearman_flat_displacement_generalization_gap": flat_gap["spearman"],
        "spearman_flat_displacement_generalization_gap_pvalue": flat_gap["spearman_pvalue"],
        "pearson_flat_displacement_test_loss": flat_test["pearson"],
        "spearman_flat_displacement_test_loss": flat_test["spearman"],
        "pearson_sharp_displacement_generalization_gap": sharp_gap["pearson"],
        "spearman_sharp_displacement_generalization_gap": sharp_gap["spearman"],
        "pearson_total_displacement_generalization_gap": total_gap["pearson"],
        "spearman_total_displacement_generalization_gap": total_gap["spearman"],
        "regression_gap_intercept": float(coef[0]),
        "regression_gap_flat_coef": float(coef[1]),
        "regression_gap_sharp_coef": float(coef[2]),
        "regression_gap_train_loss_coef": float(coef[3]),
        "regression_gap_r2": float(r2),
    }
    metrics["pass"] = bool(np.isfinite(metrics["spearman_flat_displacement_generalization_gap"]))
    return metrics


def run_exp7(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    model, loaders = _train_reference_mlp(config, device, seed)
    _, _, train_loader, full_loader, val_loader = loaders
    loss_fn = nn.CrossEntropyLoss()
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    h = _hessian_full_batch(model, full_x, full_y, loss_fn).cpu()
    eigvals, eigvecs = torch.linalg.eigh(h)
    eig_np = eigvals.numpy()
    n_flat = int(config["analysis"].get("n_flat_directions", 8))
    n_sharp = int(config["analysis"].get("n_sharp_directions", 8))
    flat_idx = np.argsort(np.abs(eig_np))[:n_flat]
    positive_order = np.argsort(eig_np)
    sharp_candidates = positive_order[eig_np[positive_order] > 0]
    if len(sharp_candidates) >= n_sharp:
        sharp_idx = sharp_candidates[-n_sharp:]
    else:
        sharp_idx = np.argsort(np.abs(eig_np))[-n_sharp:]
    flat_basis = eigvecs[:, flat_idx].to(device)
    sharp_basis = eigvecs[:, sharp_idx].to(device)
    grad_batches = int(config["analysis"].get("gradient_batches", 16))
    grads = []
    for i, (x, y) in enumerate(train_loader):
        if i >= grad_batches:
            break
        grads.append((_grad_vector(model, x.to(device), y.to(device), loss_fn).cpu() @ eigvecs).numpy())
    grads_arr = np.stack(grads)
    g_var = grads_arr.var(axis=0, ddof=1)
    pred_var = float(config["ensemble"]["lr"]) * g_var / (2.0 * np.abs(eig_np) + 1e-6)
    n_theory = int(config["analysis"].get("n_theory_directions", n_flat))
    theory_idx = np.argsort(pred_var)[-n_theory:]
    theory_basis = eigvecs[:, theory_idx].to(device)
    ref_vec = parameters_to_vector(model.parameters()).detach().clone()

    ens = int(config["ensemble"]["n_runs"])
    steps = int(config["ensemble"]["steps"])
    lr = float(config["ensemble"]["lr"])
    eval_every = int(config["analysis"].get("eval_every", max(1, steps // 10)))
    methods = config["ensemble"].get("methods", ["baseline_sgd", "suppress_flat"])
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    final_train_losses: dict[str, list[float]] = {m: [] for m in methods}
    final_test_losses: dict[str, list[float]] = {m: [] for m in methods}
    loss_history: dict[str, dict[int, dict[str, list[float]]]] = {
        m: {} for m in methods
    }

    for method in methods:
        flat_proj = np.zeros((ens, steps + 1, n_flat), dtype=np.float32)
        sharp_proj = np.zeros((ens, steps + 1, n_sharp), dtype=np.float32)
        theory_proj = np.zeros((ens, steps + 1, n_theory), dtype=np.float32)
        total_norm = np.zeros((ens, steps + 1), dtype=np.float32)
        for r in range(ens):
            torch.manual_seed(seed + 90000 + 1000 * len(method) + r)
            run_model = _mlp_model(device)
            vector_to_parameters(ref_vec.clone(), run_model.parameters())
            _, _, run_loader, _, _ = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, seed + r)
            data_iter = iter(run_loader)
            for t in range(steps + 1):
                current = parameters_to_vector(run_model.parameters()).detach()
                delta = current - ref_vec
                flat_proj[r, t] = (delta @ flat_basis).cpu().numpy()
                sharp_proj[r, t] = (delta @ sharp_basis).cpu().numpy()
                theory_proj[r, t] = (delta @ theory_basis).cpu().numpy()
                total_norm[r, t] = float(torch.linalg.norm(delta).item())
                if t % eval_every == 0 or t == steps:
                    if t not in loss_history[method]:
                        loss_history[method][t] = {"train": [], "test": []}
                    loss_history[method][t]["train"].append(_eval_classification(run_model, full_loader, device)["loss"])
                    loss_history[method][t]["test"].append(_eval_classification(run_model, val_loader, device)["loss"])
                if t == steps:
                    break
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(run_loader)
                    x, y = next(data_iter)
                grad = _grad_vector(run_model, x.to(device), y.to(device), loss_fn)
                if method == "suppress_flat":
                    grad = grad - flat_basis @ (flat_basis.T @ grad)
                elif method == "suppress_sharp":
                    grad = grad - sharp_basis @ (sharp_basis.T @ grad)
                elif method == "suppress_theory_high_variance":
                    grad = grad - theory_basis @ (theory_basis.T @ grad)
                elif method == "flat_only":
                    grad = flat_basis @ (flat_basis.T @ grad)
                vector_to_parameters(current - lr * grad, run_model.parameters())
            final_train_losses[method].append(_eval_classification(run_model, full_loader, device)["loss"])
            final_test_losses[method].append(_eval_classification(run_model, val_loader, device)["loss"])
        trajectories[method] = {"flat": flat_proj, "sharp": sharp_proj, "theory": theory_proj, "total_norm": total_norm}

    rows = []
    loss_rows = []
    metrics: dict[str, Any] = {
        "flat_eigenvalue_abs_max": float(np.max(np.abs(eig_np[flat_idx]))),
        "sharp_eigenvalue_min": float(np.min(eig_np[sharp_idx])),
        "sharp_eigenvalue_max": float(np.max(eig_np[sharp_idx])),
        "theory_high_variance_predicted_min": float(np.min(pred_var[theory_idx])),
        "theory_high_variance_predicted_max": float(np.max(pred_var[theory_idx])),
        "theory_high_variance_eigenvalue_abs_min": float(np.min(np.abs(eig_np[theory_idx]))),
        "theory_high_variance_eigenvalue_abs_max": float(np.max(np.abs(eig_np[theory_idx]))),
    }
    for method, vals in trajectories.items():
        flat_var = vals["flat"].var(axis=0, ddof=1).mean(axis=1)
        sharp_var = vals["sharp"].var(axis=0, ddof=1).mean(axis=1)
        theory_var = vals["theory"].var(axis=0, ddof=1).mean(axis=1)
        total_mean, total_std, total_lo, total_hi = mean_std_ci(vals["total_norm"], axis=0)
        for t in range(steps + 1):
            rows.append({
                "step": t,
                "method": f"{method}_flat",
                "mean_projection": float(vals["flat"][:, t, :].mean()),
                "std_projection": float(vals["flat"][:, t, :].mean(axis=1).std(ddof=1)),
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "mean_direction_variance": float(flat_var[t]),
                "mean_path_error_to_sgd": 0.0,
                "mean_coord_1": float(vals["flat"][:, t, 0].mean()),
                "mean_coord_2": float(vals["flat"][:, t, 1].mean()) if n_flat > 1 else 0.0,
            })
            rows.append({
                "step": t,
                "method": f"{method}_sharp",
                "mean_projection": float(vals["sharp"][:, t, :].mean()),
                "std_projection": float(vals["sharp"][:, t, :].mean(axis=1).std(ddof=1)),
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "mean_direction_variance": float(sharp_var[t]),
                "mean_path_error_to_sgd": float(total_mean[t]),
                "mean_coord_1": float(vals["sharp"][:, t, 0].mean()),
                "mean_coord_2": float(vals["sharp"][:, t, 1].mean()) if n_sharp > 1 else 0.0,
            })
            rows.append({
                "step": t,
                "method": f"{method}_theory_high_variance",
                "mean_projection": float(vals["theory"][:, t, :].mean()),
                "std_projection": float(vals["theory"][:, t, :].mean(axis=1).std(ddof=1)),
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "mean_direction_variance": float(theory_var[t]),
                "mean_path_error_to_sgd": float(total_mean[t]),
                "mean_coord_1": float(vals["theory"][:, t, 0].mean()),
                "mean_coord_2": float(vals["theory"][:, t, 1].mean()) if n_theory > 1 else 0.0,
            })
        train_mean, train_std, train_lo, train_hi = mean_std_ci(np.asarray(final_train_losses[method]), axis=0)
        test_mean, test_std, test_lo, test_hi = mean_std_ci(np.asarray(final_test_losses[method]), axis=0)
        gap = np.asarray(final_test_losses[method]) - np.asarray(final_train_losses[method])
        gap_mean, gap_std, gap_lo, gap_hi = mean_std_ci(gap, axis=0)
        for t, values in sorted(loss_history[method].items()):
            for split in ["train", "test"]:
                mean, std, lo, hi = mean_std_ci(np.asarray(values[split]), axis=0)
                loss_rows.append({
                    "step": t,
                    "method": method,
                    "split": split,
                    "loss_mean": float(mean),
                    "loss_std": float(std),
                    "loss_ci95_low": float(lo),
                    "loss_ci95_high": float(hi),
                })
        metrics[f"{method}_flat_final_variance"] = float(flat_var[-1])
        metrics[f"{method}_sharp_final_variance"] = float(sharp_var[-1])
        metrics[f"{method}_theory_high_variance_final_variance"] = float(theory_var[-1])
        metrics[f"{method}_total_displacement_mean"] = float(total_mean[-1])
        metrics[f"{method}_total_displacement_std"] = float(total_std[-1])
        metrics[f"{method}_train_loss_mean"] = float(train_mean)
        metrics[f"{method}_train_loss_std"] = float(train_std)
        metrics[f"{method}_train_loss_ci95_low"] = float(train_lo)
        metrics[f"{method}_train_loss_ci95_high"] = float(train_hi)
        metrics[f"{method}_test_loss_mean"] = float(test_mean)
        metrics[f"{method}_test_loss_std"] = float(test_std)
        metrics[f"{method}_test_loss_ci95_low"] = float(test_lo)
        metrics[f"{method}_test_loss_ci95_high"] = float(test_hi)
        metrics[f"{method}_generalization_gap_mean"] = float(gap_mean)
        metrics[f"{method}_generalization_gap_std"] = float(gap_std)
        metrics[f"{method}_generalization_gap_ci95_low"] = float(gap_lo)
        metrics[f"{method}_generalization_gap_ci95_high"] = float(gap_hi)

    if "baseline_sgd" in methods and "suppress_flat" in methods:
        base_flat = metrics["baseline_sgd_flat_final_variance"]
        proj_flat = metrics["suppress_flat_flat_final_variance"]
        metrics["flat_variance_reduction_fraction"] = float(1.0 - proj_flat / max(base_flat, 1e-12))
        metrics["suppress_flat_test_loss_delta"] = float(metrics["suppress_flat_test_loss_mean"] - metrics["baseline_sgd_test_loss_mean"])
        metrics["suppress_flat_train_loss_delta"] = float(metrics["suppress_flat_train_loss_mean"] - metrics["baseline_sgd_train_loss_mean"])
        metrics["suppress_flat_test_std_reduction_fraction"] = float(
            1.0 - metrics["suppress_flat_test_loss_std"] / max(metrics["baseline_sgd_test_loss_std"], 1e-12)
        )
    else:
        metrics["flat_variance_reduction_fraction"] = float("nan")
        metrics["suppress_flat_test_loss_delta"] = float("nan")
    if "baseline_sgd" in methods and "suppress_theory_high_variance" in methods:
        base_theory = metrics["baseline_sgd_theory_high_variance_final_variance"]
        proj_theory = metrics["suppress_theory_high_variance_theory_high_variance_final_variance"]
        metrics["theory_high_variance_reduction_fraction"] = float(1.0 - proj_theory / max(base_theory, 1e-12))
        metrics["suppress_theory_high_variance_test_loss_delta"] = float(
            metrics["suppress_theory_high_variance_test_loss_mean"] - metrics["baseline_sgd_test_loss_mean"]
        )
        metrics["suppress_theory_high_variance_train_loss_delta"] = float(
            metrics["suppress_theory_high_variance_train_loss_mean"] - metrics["baseline_sgd_train_loss_mean"]
        )
    else:
        metrics["theory_high_variance_reduction_fraction"] = float("nan")
        metrics["suppress_theory_high_variance_test_loss_delta"] = float("nan")

    write_csv(result_dir / "figure_data.csv", rows)
    write_csv(result_dir / "loss_data.csv", loss_rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        hessian=h.numpy(),
        eigenvalues=eig_np,
        predicted_variance=pred_var,
        flat_indices=flat_idx,
        sharp_indices=sharp_idx,
        theory_high_variance_indices=theory_idx,
        **{f"{method}_{space}": arr for method, vals in trajectories.items() for space, arr in vals.items()},
    )
    metrics["pass"] = bool(
        metrics.get("theory_high_variance_reduction_fraction", metrics.get("flat_variance_reduction_fraction", -np.inf)) > 0.5
        and metrics.get("suppress_theory_high_variance_test_loss_delta", metrics.get("suppress_flat_test_loss_delta", np.inf)) < 0.02
    )
    return metrics


def run_exp8(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    model, loaders = _train_reference_mlp(config, device, seed)
    _, _, train_loader, full_loader, val_loader = loaders
    loss_fn = nn.CrossEntropyLoss()
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    h = _hessian_full_batch(model, full_x, full_y, loss_fn).cpu()
    eigvals, eigvecs = torch.linalg.eigh(h)
    eig_np = eigvals.numpy()

    grad_batches = int(config["analysis"].get("gradient_batches", 16))
    grads = []
    for i, (x, y) in enumerate(train_loader):
        if i >= grad_batches:
            break
        grads.append((_grad_vector(model, x.to(device), y.to(device), loss_fn).cpu() @ eigvecs).numpy())
    grads_arr = np.stack(grads)
    g_var = grads_arr.var(axis=0, ddof=1)
    pred_var = float(config["ensemble"]["lr"]) * g_var / (2.0 * np.abs(eig_np) + 1e-6)

    n_theory = int(config["analysis"].get("n_theory_directions", 8))
    n_sharp = int(config["analysis"].get("n_sharp_directions", 8))
    theory_idx = np.argsort(pred_var)[-n_theory:]
    positive_order = np.argsort(eig_np)
    sharp_candidates = positive_order[eig_np[positive_order] > 0]
    if len(sharp_candidates) >= n_sharp:
        sharp_idx = sharp_candidates[-n_sharp:]
    else:
        sharp_idx = np.argsort(np.abs(eig_np))[-n_sharp:]
    rng = np.random.default_rng(seed + 808)
    random_idx = rng.choice(len(eig_np), size=n_theory, replace=False)

    theory_basis = eigvecs[:, theory_idx].to(device)
    sharp_basis = eigvecs[:, sharp_idx].to(device)
    random_basis = eigvecs[:, random_idx].to(device)
    ref_vec = parameters_to_vector(model.parameters()).detach().clone()

    ens = int(config["ensemble"]["n_runs"])
    steps = int(config["ensemble"]["steps"])
    lr = float(config["ensemble"]["lr"])
    alpha = float(config["ensemble"].get("amplification_factor", 2.0))
    eval_every = int(config["analysis"].get("eval_every", max(1, steps // 5)))
    methods = config["ensemble"].get("methods", ["baseline_sgd", "amplify_theory_high_variance", "amplify_random", "amplify_sharp"])

    trajectories: dict[str, dict[str, np.ndarray]] = {}
    loss_history: dict[str, dict[int, dict[str, list[float]]]] = {m: {} for m in methods}
    final_train_losses: dict[str, list[float]] = {m: [] for m in methods}
    final_test_losses: dict[str, list[float]] = {m: [] for m in methods}

    for method in methods:
        theory_proj = np.zeros((ens, steps + 1, n_theory), dtype=np.float32)
        sharp_proj = np.zeros((ens, steps + 1, n_sharp), dtype=np.float32)
        total_norm = np.zeros((ens, steps + 1), dtype=np.float32)
        for r in range(ens):
            torch.manual_seed(seed + 120000 + 1000 * len(method) + r)
            run_model = _mlp_model(device)
            vector_to_parameters(ref_vec.clone(), run_model.parameters())
            _, _, run_loader, _, _ = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, seed + r)
            data_iter = iter(run_loader)
            for t in range(steps + 1):
                current = parameters_to_vector(run_model.parameters()).detach()
                delta = current - ref_vec
                theory_proj[r, t] = (delta @ theory_basis).cpu().numpy()
                sharp_proj[r, t] = (delta @ sharp_basis).cpu().numpy()
                total_norm[r, t] = float(torch.linalg.norm(delta).item())
                if t % eval_every == 0 or t == steps:
                    if t not in loss_history[method]:
                        loss_history[method][t] = {"train": [], "test": []}
                    loss_history[method][t]["train"].append(_eval_classification(run_model, full_loader, device)["loss"])
                    loss_history[method][t]["test"].append(_eval_classification(run_model, val_loader, device)["loss"])
                if t == steps:
                    break
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(run_loader)
                    x, y = next(data_iter)
                grad_batch = _grad_vector(run_model, x.to(device), y.to(device), loss_fn)
                if method == "baseline_sgd":
                    update_grad = grad_batch
                else:
                    grad_full = _grad_vector(run_model, full_x, full_y, loss_fn)
                    noise = grad_batch - grad_full
                    if method == "amplify_theory_high_variance":
                        basis = theory_basis
                    elif method == "amplify_sharp":
                        basis = sharp_basis
                    elif method == "amplify_random":
                        basis = random_basis
                    else:
                        basis = theory_basis
                    noise_selected = basis @ (basis.T @ noise)
                    update_grad = grad_full + noise + (alpha - 1.0) * noise_selected
                vector_to_parameters(current - lr * update_grad, run_model.parameters())
            final_train_losses[method].append(_eval_classification(run_model, full_loader, device)["loss"])
            final_test_losses[method].append(_eval_classification(run_model, val_loader, device)["loss"])
        trajectories[method] = {"theory": theory_proj, "sharp": sharp_proj, "total_norm": total_norm}

    rows = []
    loss_rows = []
    metrics: dict[str, Any] = {
        "amplification_factor": alpha,
        "theory_high_variance_predicted_min": float(np.min(pred_var[theory_idx])),
        "theory_high_variance_predicted_max": float(np.max(pred_var[theory_idx])),
        "theory_high_variance_eigenvalue_abs_min": float(np.min(np.abs(eig_np[theory_idx]))),
        "theory_high_variance_eigenvalue_abs_max": float(np.max(np.abs(eig_np[theory_idx]))),
        "sharp_eigenvalue_min": float(np.min(eig_np[sharp_idx])),
        "sharp_eigenvalue_max": float(np.max(eig_np[sharp_idx])),
    }
    for method, vals in trajectories.items():
        theory_var = vals["theory"].var(axis=0, ddof=1).mean(axis=1)
        sharp_var = vals["sharp"].var(axis=0, ddof=1).mean(axis=1)
        total_mean, total_std, _, _ = mean_std_ci(vals["total_norm"], axis=0)
        for t in range(steps + 1):
            for suffix, arr, var in [("theory_high_variance", vals["theory"], theory_var), ("sharp", vals["sharp"], sharp_var)]:
                rows.append({
                    "step": t,
                    "method": f"{method}_{suffix}",
                    "mean_projection": float(arr[:, t, :].mean()),
                    "std_projection": float(arr[:, t, :].mean(axis=1).std(ddof=1)),
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                    "mean_direction_variance": float(var[t]),
                    "mean_path_error_to_sgd": float(total_mean[t]),
                    "mean_coord_1": float(arr[:, t, 0].mean()),
                    "mean_coord_2": float(arr[:, t, 1].mean()) if arr.shape[2] > 1 else 0.0,
                })
        for t, values in sorted(loss_history[method].items()):
            for split in ["train", "test"]:
                mean, std, lo, hi = mean_std_ci(np.asarray(values[split]), axis=0)
                loss_rows.append({
                    "step": t,
                    "method": method,
                    "split": split,
                    "loss_mean": float(mean),
                    "loss_std": float(std),
                    "loss_ci95_low": float(lo),
                    "loss_ci95_high": float(hi),
                })
        train = np.asarray(final_train_losses[method])
        test = np.asarray(final_test_losses[method])
        gap = test - train
        train_mean, train_std, train_lo, train_hi = mean_std_ci(train, axis=0)
        test_mean, test_std, test_lo, test_hi = mean_std_ci(test, axis=0)
        gap_mean, gap_std, gap_lo, gap_hi = mean_std_ci(gap, axis=0)
        metrics[f"{method}_theory_high_variance_final_variance"] = float(theory_var[-1])
        metrics[f"{method}_sharp_final_variance"] = float(sharp_var[-1])
        metrics[f"{method}_total_displacement_mean"] = float(total_mean[-1])
        metrics[f"{method}_total_displacement_std"] = float(total_std[-1])
        metrics[f"{method}_train_loss_mean"] = float(train_mean)
        metrics[f"{method}_train_loss_std"] = float(train_std)
        metrics[f"{method}_train_loss_ci95_low"] = float(train_lo)
        metrics[f"{method}_train_loss_ci95_high"] = float(train_hi)
        metrics[f"{method}_test_loss_mean"] = float(test_mean)
        metrics[f"{method}_test_loss_std"] = float(test_std)
        metrics[f"{method}_test_loss_ci95_low"] = float(test_lo)
        metrics[f"{method}_test_loss_ci95_high"] = float(test_hi)
        metrics[f"{method}_generalization_gap_mean"] = float(gap_mean)
        metrics[f"{method}_generalization_gap_std"] = float(gap_std)
        metrics[f"{method}_generalization_gap_ci95_low"] = float(gap_lo)
        metrics[f"{method}_generalization_gap_ci95_high"] = float(gap_hi)

    if "baseline_sgd" in methods and "amplify_theory_high_variance" in methods:
        metrics["amplify_theory_test_loss_delta"] = float(metrics["amplify_theory_high_variance_test_loss_mean"] - metrics["baseline_sgd_test_loss_mean"])
        metrics["amplify_theory_train_loss_delta"] = float(metrics["amplify_theory_high_variance_train_loss_mean"] - metrics["baseline_sgd_train_loss_mean"])
        metrics["amplify_theory_variance_ratio"] = float(
            metrics["amplify_theory_high_variance_theory_high_variance_final_variance"]
            / max(metrics["baseline_sgd_theory_high_variance_final_variance"], 1e-12)
        )
    if "baseline_sgd" in methods and "amplify_random" in methods:
        metrics["amplify_random_test_loss_delta"] = float(metrics["amplify_random_test_loss_mean"] - metrics["baseline_sgd_test_loss_mean"])
    if "baseline_sgd" in methods and "amplify_sharp" in methods:
        metrics["amplify_sharp_test_loss_delta"] = float(metrics["amplify_sharp_test_loss_mean"] - metrics["baseline_sgd_test_loss_mean"])

    write_csv(result_dir / "figure_data.csv", rows)
    write_csv(result_dir / "loss_data.csv", loss_rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        hessian=h.numpy(),
        eigenvalues=eig_np,
        predicted_variance=pred_var,
        theory_high_variance_indices=theory_idx,
        sharp_indices=sharp_idx,
        random_indices=random_idx,
        **{f"{method}_{space}": arr for method, vals in trajectories.items() for space, arr in vals.items()},
    )
    metrics["pass"] = bool(
        metrics.get("amplify_theory_test_loss_delta", np.inf)
        < min(metrics.get("amplify_random_test_loss_delta", np.inf), metrics.get("amplify_sharp_test_loss_delta", np.inf), 0.02)
    )
    return metrics


def run_exp9(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    model, loaders = _train_reference_mlp(config, device, seed)
    _, _, train_loader, full_loader, val_loader = loaders
    loss_fn = nn.CrossEntropyLoss()
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    h = _hessian_full_batch(model, full_x, full_y, loss_fn).cpu()
    eigvals, eigvecs = torch.linalg.eigh(h)
    eig_np = eigvals.numpy()

    grad_batches = int(config["analysis"].get("gradient_batches", 16))
    grads = []
    for i, (x, y) in enumerate(train_loader):
        if i >= grad_batches:
            break
        grads.append((_grad_vector(model, x.to(device), y.to(device), loss_fn).cpu() @ eigvecs).numpy())
    grads_arr = np.stack(grads)
    g_var = grads_arr.var(axis=0, ddof=1)
    pred_var = float(config["ensemble"]["lr"]) * g_var / (2.0 * np.abs(eig_np) + 1e-6)

    n_theory = int(config["analysis"].get("n_theory_directions", 8))
    n_sharp = int(config["analysis"].get("n_sharp_directions", 8))
    theory_idx = np.argsort(pred_var)[-n_theory:]
    positive_order = np.argsort(eig_np)
    sharp_candidates = positive_order[eig_np[positive_order] > 0]
    sharp_idx = sharp_candidates[-n_sharp:] if len(sharp_candidates) >= n_sharp else np.argsort(np.abs(eig_np))[-n_sharp:]
    rng = np.random.default_rng(seed + 909)
    random_idx = rng.choice(len(eig_np), size=n_theory, replace=False)

    theory_basis = eigvecs[:, theory_idx].to(device)
    sharp_basis = eigvecs[:, sharp_idx].to(device)
    random_basis = eigvecs[:, random_idx].to(device)
    ref_vec = parameters_to_vector(model.parameters()).detach().clone()

    ens = int(config["ensemble"]["n_runs"])
    steps = int(config["ensemble"]["steps"])
    lr = float(config["ensemble"]["lr"])
    eval_every = int(config["analysis"].get("eval_every", max(1, steps // 5)))
    methods = config["ensemble"].get("methods", ["baseline_sgd", "denoise_theory_high_variance", "denoise_random", "denoise_sharp"])

    trajectories: dict[str, dict[str, np.ndarray]] = {}
    loss_history: dict[str, dict[int, dict[str, list[float]]]] = {m: {} for m in methods}
    final_train_losses: dict[str, list[float]] = {m: [] for m in methods}
    final_test_losses: dict[str, list[float]] = {m: [] for m in methods}

    for method in methods:
        theory_proj = np.zeros((ens, steps + 1, n_theory), dtype=np.float32)
        sharp_proj = np.zeros((ens, steps + 1, n_sharp), dtype=np.float32)
        total_norm = np.zeros((ens, steps + 1), dtype=np.float32)
        for r in range(ens):
            torch.manual_seed(seed + 130000 + 1000 * len(method) + r)
            run_model = _mlp_model(device)
            vector_to_parameters(ref_vec.clone(), run_model.parameters())
            _, _, run_loader, _, _ = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, seed + r)
            data_iter = iter(run_loader)
            for t in range(steps + 1):
                current = parameters_to_vector(run_model.parameters()).detach()
                delta = current - ref_vec
                theory_proj[r, t] = (delta @ theory_basis).cpu().numpy()
                sharp_proj[r, t] = (delta @ sharp_basis).cpu().numpy()
                total_norm[r, t] = float(torch.linalg.norm(delta).item())
                if t % eval_every == 0 or t == steps:
                    if t not in loss_history[method]:
                        loss_history[method][t] = {"train": [], "test": []}
                    loss_history[method][t]["train"].append(_eval_classification(run_model, full_loader, device)["loss"])
                    loss_history[method][t]["test"].append(_eval_classification(run_model, val_loader, device)["loss"])
                if t == steps:
                    break
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(run_loader)
                    x, y = next(data_iter)
                grad_batch = _grad_vector(run_model, x.to(device), y.to(device), loss_fn)
                if method == "baseline_sgd":
                    update_grad = grad_batch
                else:
                    grad_full = _grad_vector(run_model, full_x, full_y, loss_fn)
                    noise = grad_batch - grad_full
                    if method == "denoise_theory_high_variance":
                        basis = theory_basis
                    elif method == "denoise_sharp":
                        basis = sharp_basis
                    elif method == "denoise_random":
                        basis = random_basis
                    else:
                        basis = theory_basis
                    update_grad = grad_batch - basis @ (basis.T @ noise)
                vector_to_parameters(current - lr * update_grad, run_model.parameters())
            final_train_losses[method].append(_eval_classification(run_model, full_loader, device)["loss"])
            final_test_losses[method].append(_eval_classification(run_model, val_loader, device)["loss"])
        trajectories[method] = {"theory": theory_proj, "sharp": sharp_proj, "total_norm": total_norm}

    rows = []
    loss_rows = []
    metrics: dict[str, Any] = {
        "theory_high_variance_predicted_min": float(np.min(pred_var[theory_idx])),
        "theory_high_variance_predicted_max": float(np.max(pred_var[theory_idx])),
        "theory_high_variance_eigenvalue_abs_min": float(np.min(np.abs(eig_np[theory_idx]))),
        "theory_high_variance_eigenvalue_abs_max": float(np.max(np.abs(eig_np[theory_idx]))),
        "sharp_eigenvalue_min": float(np.min(eig_np[sharp_idx])),
        "sharp_eigenvalue_max": float(np.max(eig_np[sharp_idx])),
    }
    for method, vals in trajectories.items():
        theory_var = vals["theory"].var(axis=0, ddof=1).mean(axis=1)
        sharp_var = vals["sharp"].var(axis=0, ddof=1).mean(axis=1)
        total_mean, total_std, _, _ = mean_std_ci(vals["total_norm"], axis=0)
        for t in range(steps + 1):
            for suffix, arr, var in [("theory_high_variance", vals["theory"], theory_var), ("sharp", vals["sharp"], sharp_var)]:
                rows.append({
                    "step": t,
                    "method": f"{method}_{suffix}",
                    "mean_projection": float(arr[:, t, :].mean()),
                    "std_projection": float(arr[:, t, :].mean(axis=1).std(ddof=1)),
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                    "mean_direction_variance": float(var[t]),
                    "mean_path_error_to_sgd": float(total_mean[t]),
                    "mean_coord_1": float(arr[:, t, 0].mean()),
                    "mean_coord_2": float(arr[:, t, 1].mean()) if arr.shape[2] > 1 else 0.0,
                })
        for t, values in sorted(loss_history[method].items()):
            for split in ["train", "test"]:
                mean, std, lo, hi = mean_std_ci(np.asarray(values[split]), axis=0)
                loss_rows.append({
                    "step": t,
                    "method": method,
                    "split": split,
                    "loss_mean": float(mean),
                    "loss_std": float(std),
                    "loss_ci95_low": float(lo),
                    "loss_ci95_high": float(hi),
                })
        train = np.asarray(final_train_losses[method])
        test = np.asarray(final_test_losses[method])
        gap = test - train
        train_mean, train_std, train_lo, train_hi = mean_std_ci(train, axis=0)
        test_mean, test_std, test_lo, test_hi = mean_std_ci(test, axis=0)
        gap_mean, gap_std, gap_lo, gap_hi = mean_std_ci(gap, axis=0)
        metrics[f"{method}_theory_high_variance_final_variance"] = float(theory_var[-1])
        metrics[f"{method}_sharp_final_variance"] = float(sharp_var[-1])
        metrics[f"{method}_total_displacement_mean"] = float(total_mean[-1])
        metrics[f"{method}_total_displacement_std"] = float(total_std[-1])
        metrics[f"{method}_train_loss_mean"] = float(train_mean)
        metrics[f"{method}_train_loss_std"] = float(train_std)
        metrics[f"{method}_train_loss_ci95_low"] = float(train_lo)
        metrics[f"{method}_train_loss_ci95_high"] = float(train_hi)
        metrics[f"{method}_test_loss_mean"] = float(test_mean)
        metrics[f"{method}_test_loss_std"] = float(test_std)
        metrics[f"{method}_test_loss_ci95_low"] = float(test_lo)
        metrics[f"{method}_test_loss_ci95_high"] = float(test_hi)
        metrics[f"{method}_generalization_gap_mean"] = float(gap_mean)
        metrics[f"{method}_generalization_gap_std"] = float(gap_std)
        metrics[f"{method}_generalization_gap_ci95_low"] = float(gap_lo)
        metrics[f"{method}_generalization_gap_ci95_high"] = float(gap_hi)

    if "baseline_sgd" in methods and "denoise_theory_high_variance" in methods:
        metrics["denoise_theory_test_loss_delta"] = float(metrics["denoise_theory_high_variance_test_loss_mean"] - metrics["baseline_sgd_test_loss_mean"])
        metrics["denoise_theory_train_loss_delta"] = float(metrics["denoise_theory_high_variance_train_loss_mean"] - metrics["baseline_sgd_train_loss_mean"])
        metrics["denoise_theory_variance_reduction_fraction"] = float(
            1.0
            - metrics["denoise_theory_high_variance_theory_high_variance_final_variance"]
            / max(metrics["baseline_sgd_theory_high_variance_final_variance"], 1e-12)
        )
    if "baseline_sgd" in methods and "denoise_random" in methods:
        metrics["denoise_random_test_loss_delta"] = float(metrics["denoise_random_test_loss_mean"] - metrics["baseline_sgd_test_loss_mean"])
    if "baseline_sgd" in methods and "denoise_sharp" in methods:
        metrics["denoise_sharp_test_loss_delta"] = float(metrics["denoise_sharp_test_loss_mean"] - metrics["baseline_sgd_test_loss_mean"])

    write_csv(result_dir / "figure_data.csv", rows)
    write_csv(result_dir / "loss_data.csv", loss_rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        hessian=h.numpy(),
        eigenvalues=eig_np,
        predicted_variance=pred_var,
        theory_high_variance_indices=theory_idx,
        sharp_indices=sharp_idx,
        random_indices=random_idx,
        **{f"{method}_{space}": arr for method, vals in trajectories.items() for space, arr in vals.items()},
    )
    metrics["pass"] = bool(
        metrics.get("denoise_theory_variance_reduction_fraction", -np.inf) > 0.5
        and metrics.get("denoise_theory_test_loss_delta", np.inf) <= min(
            metrics.get("denoise_random_test_loss_delta", np.inf),
            metrics.get("denoise_sharp_test_loss_delta", np.inf),
            0.005,
        )
    )
    return metrics


def _run_noise_alpha_sweep(config: dict[str, Any], result_dir: Path, normalized_noise: bool) -> dict[str, Any]:
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    train_ds, val_ds, train_loader, full_loader, val_loader = _mnist_loaders(
        config["dataset"], config["training"]["batch_size"], config["training"].get("replacement", True), seed
    )
    loss_fn = nn.CrossEntropyLoss()
    model = _configured_model(config, device)
    opt = torch.optim.SGD(model.parameters(), lr=float(config["training"]["lr"]))
    data_iter = iter(train_loader)
    reference_steps = int(config["training"].get("reference_steps", 50))
    pretrain_iter = range(reference_steps)
    if tqdm is not None:
        pretrain_iter = tqdm(pretrain_iter, desc="reference pretrain", unit="step", leave=True)
    for _ in pretrain_iter:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    ref_vec = parameters_to_vector(model.parameters()).detach().clone()
    ref_train_eval = _eval_classification(model, full_loader, device)
    ref_test_eval = _eval_classification(model, val_loader, device)

    alphas = [float(a) for a in config["sweep"]["alphas"]]
    seed_list = [int(s) for s in config["sweep"].get("seeds", [seed])]
    steps = int(config["ensemble"]["steps"])
    lr = float(config["ensemble"]["lr"])
    eval_every = int(config["analysis"].get("eval_every", max(1, steps // 20)))
    batch_size = int(config["training"]["batch_size"])
    epoch_steps = max(1, len(train_loader))
    rows = []
    loss_rows = []
    final_rows = []
    metrics: dict[str, Any] = {
        "normalized_noise": bool(normalized_noise),
        "n_alphas": len(alphas),
        "sweep_seeds": seed_list,
        "n_sweep_seeds": len(seed_list),
        "steps": steps,
        "lr": lr,
        "reference_train_loss": ref_train_eval["loss"],
        "reference_train_accuracy": ref_train_eval["accuracy"],
        "reference_test_loss": ref_test_eval["loss"],
        "reference_test_accuracy": ref_test_eval["accuracy"],
    }

    per_seed_loss_rows: list[dict[str, Any]] = []
    per_seed_trajectory_rows: list[dict[str, Any]] = []
    per_seed_final_rows: list[dict[str, Any]] = []
    eval_vectors: dict[tuple[float, int], list[np.ndarray]] = {}
    alpha_iter = alphas
    if tqdm is not None:
        alpha_iter = tqdm(alphas, desc=f"{config['experiment_id']} alpha sweep", unit="alpha", leave=True)
    for alpha in alpha_iter:
        for run_seed in seed_list:
            torch.manual_seed(run_seed + 150000 + int(round(alpha * 1000)))
            run_model = _configured_model(config, device)
            vector_to_parameters(ref_vec.clone(), run_model.parameters())
            _, _, run_loader, _, _ = _mnist_loaders(config["dataset"], batch_size, True, run_seed)
            data_iter = iter(run_loader)
            diverged = False
            last_train = float("nan")
            last_test = float("nan")
            last_train_acc = float("nan")
            last_test_acc = float("nan")
            step_iter = range(steps + 1)
            progress = None
            if tqdm is not None:
                progress = tqdm(step_iter, desc=f"alpha={alpha:g} seed={run_seed}", unit="step", leave=False)
                step_iter = progress
            for t in step_iter:
                if t % eval_every == 0 or t == steps or diverged:
                    current_eval = parameters_to_vector(run_model.parameters()).detach()
                    displacement_norm = float(torch.linalg.norm(current_eval - ref_vec).item()) if not diverged else float("nan")
                    if not diverged:
                        eval_vectors.setdefault((alpha, t), []).append((current_eval - ref_vec).detach().cpu().numpy())
                    train_eval = _eval_classification(run_model, full_loader, device) if not diverged else {"loss": float("nan"), "accuracy": float("nan")}
                    test_eval = _eval_classification(run_model, val_loader, device) if not diverged else {"loss": float("nan"), "accuracy": float("nan")}
                    train_loss, test_loss = train_eval["loss"], test_eval["loss"]
                    train_acc, test_acc = train_eval["accuracy"], test_eval["accuracy"]
                    last_train, last_test = train_loss, test_loss
                    last_train_acc, last_test_acc = train_acc, test_acc
                    for split, loss_value, acc_value in [("train", train_loss, train_acc), ("test", test_loss, test_acc)]:
                        per_seed_loss_rows.append({
                            "step": t,
                            "alpha": alpha,
                            "seed": run_seed,
                            "method": f"alpha={alpha:g}",
                            "split": split,
                            "loss": loss_value,
                            "accuracy": acc_value,
                        })
                    per_seed_trajectory_rows.append({
                        "step": t,
                        "alpha": alpha,
                        "seed": run_seed,
                        "method": f"alpha={alpha:g}",
                        "displacement_norm": displacement_norm,
                        "diverged": diverged,
                    })
                    if progress is not None:
                        progress.set_postfix({
                            "epoch": f"{t / epoch_steps:.1f}",
                            "train": f"{train_loss:.4g}",
                            "test": f"{test_loss:.4g}",
                            "acc": f"{test_acc:.3f}",
                        })
                if t == steps or diverged:
                    break
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(run_loader)
                    x, y = next(data_iter)
                grad_batch = _grad_vector(run_model, x.to(device), y.to(device), loss_fn)
                grad_full = _grad_vector(run_model, full_x, full_y, loss_fn)
                noise = grad_batch - grad_full
                if normalized_noise:
                    noise_norm = torch.linalg.norm(noise).clamp_min(1e-12)
                    full_norm = torch.linalg.norm(grad_full)
                    scaled_noise = alpha * full_norm / noise_norm * noise
                else:
                    scaled_noise = alpha * noise
                update_grad = grad_full + scaled_noise
                current = parameters_to_vector(run_model.parameters()).detach()
                next_vec = current - lr * update_grad
                if not torch.isfinite(next_vec).all():
                    diverged = True
                    continue
                vector_to_parameters(next_vec, run_model.parameters())
            per_seed_final_rows.append({
                "alpha": alpha,
                "seed": run_seed,
                "method": f"alpha={alpha:g}",
                "final_train_loss": last_train,
                "final_test_loss": last_test,
                "final_train_accuracy": last_train_acc,
                "final_test_accuracy": last_test_acc,
                "generalization_gap": last_test - last_train if np.isfinite(last_test) and np.isfinite(last_train) else float("nan"),
                "final_displacement_norm": (
                    float(torch.linalg.norm(parameters_to_vector(run_model.parameters()).detach() - ref_vec).item())
                    if not diverged else float("nan")
                ),
                "diverged": diverged,
            })
        if hasattr(alpha_iter, "set_postfix"):
            vals = [r for r in per_seed_final_rows if abs(r["alpha"] - alpha) < 1e-12]
            alpha_iter.set_postfix({
                "alpha": f"{alpha:g}",
                "test_mean": f"{np.nanmean([r['final_test_loss'] for r in vals]):.4g}",
                "acc_mean": f"{np.nanmean([r['final_test_accuracy'] for r in vals]):.3f}",
            })

    for alpha in alphas:
        alpha_final = [r for r in per_seed_final_rows if abs(r["alpha"] - alpha) < 1e-12]
        row = {"alpha": alpha, "method": f"alpha={alpha:g}", "n_seeds": len(alpha_final)}
        for key in ["final_train_loss", "final_test_loss", "final_train_accuracy", "final_test_accuracy", "generalization_gap", "final_displacement_norm"]:
            vals = np.asarray([r[key] for r in alpha_final if np.isfinite(r[key])], dtype=np.float64)
            mean, std, lo, hi = mean_std_ci(vals, axis=0) if len(vals) else (np.nan, np.nan, np.nan, np.nan)
            short = key.replace("final_", "")
            row[f"{short}_mean"] = float(mean)
            row[f"{short}_std"] = float(std)
            row[f"{short}_ci95_low"] = float(lo)
            row[f"{short}_ci95_high"] = float(hi)
        final_vecs = np.asarray(eval_vectors.get((alpha, steps), []), dtype=np.float64)
        if len(final_vecs) >= 2:
            centered = final_vecs - final_vecs.mean(axis=0, keepdims=True)
            row["parameter_variance_trace"] = float(np.mean(np.sum(centered**2, axis=1)))
        else:
            row["parameter_variance_trace"] = 0.0
        row["diverged_count"] = int(sum(bool(r["diverged"]) for r in alpha_final))
        final_rows.append(row)
        metrics[f"alpha_{alpha:g}_final_train_loss_mean"] = row["train_loss_mean"]
        metrics[f"alpha_{alpha:g}_final_test_loss_mean"] = row["test_loss_mean"]
        metrics[f"alpha_{alpha:g}_final_test_loss_std"] = row["test_loss_std"]
        metrics[f"alpha_{alpha:g}_final_test_accuracy_mean"] = row["test_accuracy_mean"]
        metrics[f"alpha_{alpha:g}_final_displacement_norm_mean"] = row["displacement_norm_mean"]
        metrics[f"alpha_{alpha:g}_final_displacement_norm_std"] = row["displacement_norm_std"]
        metrics[f"alpha_{alpha:g}_final_parameter_variance_trace"] = row["parameter_variance_trace"]
        metrics[f"alpha_{alpha:g}_diverged_count"] = row["diverged_count"]

    baseline_alpha1 = next((r for r in final_rows if abs(float(r["alpha"]) - 1.0) < 1e-12), None)
    var_alpha1 = float(baseline_alpha1.get("parameter_variance_trace", 0.0)) if baseline_alpha1 is not None else 0.0
    disp_alpha1 = float(baseline_alpha1.get("displacement_norm_mean", 0.0)) if baseline_alpha1 is not None else 0.0
    for row in final_rows:
        row["normalized_variance"] = float(row["parameter_variance_trace"] / max(var_alpha1, 1e-30))
        row["normalized_displacement"] = float(row["displacement_norm_mean"] / max(disp_alpha1, 1e-30))
        rows.append(row.copy())
        metrics[f"alpha_{float(row['alpha']):g}_normalized_variance"] = row["normalized_variance"]
        metrics[f"alpha_{float(row['alpha']):g}_normalized_displacement"] = row["normalized_displacement"]

    grouped_loss: dict[tuple[float, int, str], list[dict[str, Any]]] = {}
    for row in per_seed_loss_rows:
        grouped_loss.setdefault((float(row["alpha"]), int(row["step"]), str(row["split"])), []).append(row)
    for (alpha, step, split), vals_rows in sorted(grouped_loss.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        losses = np.asarray([r["loss"] for r in vals_rows if np.isfinite(r["loss"])], dtype=np.float64)
        accs = np.asarray([r["accuracy"] for r in vals_rows if np.isfinite(r["accuracy"])], dtype=np.float64)
        loss_mean, loss_std, loss_lo, loss_hi = mean_std_ci(losses, axis=0) if len(losses) else (np.nan, np.nan, np.nan, np.nan)
        acc_mean, acc_std, acc_lo, acc_hi = mean_std_ci(accs, axis=0) if len(accs) else (np.nan, np.nan, np.nan, np.nan)
        loss_rows.append({
            "step": step,
            "alpha": alpha,
            "method": f"alpha={alpha:g}",
            "split": split,
            "loss_mean": float(loss_mean),
            "loss_std": float(loss_std),
            "loss_ci95_low": float(loss_lo),
            "loss_ci95_high": float(loss_hi),
            "accuracy": float(acc_mean),
            "accuracy_std": float(acc_std),
            "accuracy_ci95_low": float(acc_lo),
            "accuracy_ci95_high": float(acc_hi),
        })

    trajectory_rows: list[dict[str, Any]] = []
    grouped_traj: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for row in per_seed_trajectory_rows:
        grouped_traj.setdefault((float(row["alpha"]), int(row["step"])), []).append(row)
    for (alpha, step), vals_rows in sorted(grouped_traj.items(), key=lambda x: (x[0][0], x[0][1])):
        disps = np.asarray([r["displacement_norm"] for r in vals_rows if np.isfinite(r["displacement_norm"])], dtype=np.float64)
        disp_mean, disp_std, disp_lo, disp_hi = mean_std_ci(disps, axis=0) if len(disps) else (np.nan, np.nan, np.nan, np.nan)
        vecs = np.asarray(eval_vectors.get((alpha, step), []), dtype=np.float64)
        if len(vecs) >= 2:
            centered = vecs - vecs.mean(axis=0, keepdims=True)
            variance_trace = float(np.mean(np.sum(centered**2, axis=1)))
        else:
            variance_trace = 0.0
        trajectory_rows.append({
            "step": step,
            "alpha": alpha,
            "method": f"alpha={alpha:g}",
            "displacement_norm_mean": float(disp_mean),
            "displacement_norm_std": float(disp_std),
            "displacement_norm_ci95_low": float(disp_lo),
            "displacement_norm_ci95_high": float(disp_hi),
            "parameter_variance_trace": variance_trace,
            "n_seeds": len(vals_rows),
        })
    variance_at_alpha1_by_step = {
        int(r["step"]): float(r["parameter_variance_trace"])
        for r in trajectory_rows
        if abs(float(r["alpha"]) - 1.0) < 1e-12
    }
    displacement_at_alpha1_by_step = {
        int(r["step"]): float(r["displacement_norm_mean"])
        for r in trajectory_rows
        if abs(float(r["alpha"]) - 1.0) < 1e-12
    }
    for row in trajectory_rows:
        step = int(row["step"])
        row["normalized_variance"] = float(row["parameter_variance_trace"] / max(variance_at_alpha1_by_step.get(step, 0.0), 1e-30))
        row["normalized_displacement"] = float(row["displacement_norm_mean"] / max(displacement_at_alpha1_by_step.get(step, 0.0), 1e-30))

    finite = [r for r in final_rows if r.get("diverged_count", 0) < r.get("n_seeds", 1) and np.isfinite(r["train_loss_mean"])]
    if finite:
        best_train = min(finite, key=lambda r: r["train_loss_mean"])
        best_test = min(finite, key=lambda r: r["test_loss_mean"])
        metrics["best_train_alpha"] = float(best_train["alpha"])
        metrics["best_train_loss"] = float(best_train["train_loss_mean"])
        metrics["best_test_alpha"] = float(best_test["alpha"])
        metrics["best_test_loss"] = float(best_test["test_loss_mean"])
        best_acc = max(finite, key=lambda r: r["test_accuracy_mean"])
        metrics["best_test_accuracy_alpha"] = float(best_acc["alpha"])
        metrics["best_test_accuracy"] = float(best_acc["test_accuracy_mean"])
        baseline = next((r for r in finite if abs(r["alpha"] - 1.0) < 1e-12), None)
        if baseline is not None:
            metrics["best_train_delta_vs_alpha_1"] = float(best_train["train_loss_mean"] - baseline["train_loss_mean"])
            metrics["best_test_delta_vs_alpha_1"] = float(best_test["test_loss_mean"] - baseline["test_loss_mean"])
            metrics["best_accuracy_delta_vs_alpha_1"] = float(best_acc["test_accuracy_mean"] - baseline["test_accuracy_mean"])
    metrics["pass"] = bool("best_test_delta_vs_alpha_1" in metrics and metrics["best_test_delta_vs_alpha_1"] < 0.0)
    write_csv(result_dir / "figure_data.csv", rows)
    write_csv(result_dir / "loss_data.csv", loss_rows)
    write_csv(result_dir / "final_alpha_sweep.csv", final_rows)
    write_csv(result_dir / "loss_data_by_seed.csv", per_seed_loss_rows)
    write_csv(result_dir / "trajectory_data.csv", trajectory_rows)
    write_csv(result_dir / "trajectory_data_by_seed.csv", per_seed_trajectory_rows)
    write_csv(result_dir / "final_alpha_sweep_by_seed.csv", per_seed_final_rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        alphas=np.asarray(alphas),
        seeds=np.asarray(seed_list),
        trajectory_summary=np.asarray(
            [[r["alpha"], r["step"], r["displacement_norm_mean"], r["parameter_variance_trace"]] for r in trajectory_rows],
            dtype=np.float64,
        ),
        final_rows=np.asarray(
            [[r["alpha"], r["train_loss_mean"], r["test_loss_mean"], r["train_accuracy_mean"], r["test_accuracy_mean"], r["displacement_norm_mean"], r["parameter_variance_trace"]] for r in final_rows],
            dtype=np.float64,
        ),
    )
    return metrics


def run_exp10(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    return _run_noise_alpha_sweep(config, result_dir, normalized_noise=False)


def run_exp11(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    return _run_noise_alpha_sweep(config, result_dir, normalized_noise=True)


def run_exp12(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    model, loaders = _train_reference_mlp(config, device, seed)
    _, _, train_loader, full_loader, _ = loaders
    loss_fn = nn.CrossEntropyLoss()
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    ref_vec = parameters_to_vector(model.parameters()).detach().clone()
    n_params = ref_vec.numel()
    lr = float(config["ensemble"]["lr"])
    one_step_samples = int(config["analysis"].get("one_step_samples", 32))
    langevin_noise_batches = int(config["analysis"].get("langevin_noise_batches", 8))
    horizons = [int(h) for h in config["analysis"].get("horizons", [1, 5, 20, 80])]
    max_horizon = max(horizons)
    n_runs = int(config["ensemble"].get("n_runs", 8))
    n_dirs = int(config["analysis"].get("n_projection_directions", 8))
    calibration = [float(c) for c in config["analysis"].get("calibration_multipliers", [0.5, 0.75, 1.0, 1.25, 1.5])]

    full_grad = _grad_vector(model, full_x, full_y, loss_fn)
    drift = -lr * full_grad
    ref_noise_std = _estimate_minibatch_noise_std(model, full_grad, config, device, seed + 1200, langevin_noise_batches, loss_fn)
    ref_modified_noise_std = _modified_langevin_noise_std(ref_noise_std, full_grad)
    drift_norm = float(torch.linalg.norm(drift).item())
    standard_noise_step_norm = float(lr * torch.linalg.norm(ref_noise_std).item())
    modified_noise_step_norm = float(lr * torch.linalg.norm(ref_modified_noise_std).item())
    sde_sigma_norm = float(math.sqrt(lr) * torch.linalg.norm(ref_noise_std).item())
    audit = {
        "standard_noise_scaling_mode": "Euler-Maruyama with dt=lr: lr * sqrt(D)",
        "modified_noise_scaling_mode": "Euler-Maruyama with dt=lr: lr * sqrt(D + grad_bar^2)",
        "sde_sigma_scaling_mode": "continuous SDE coefficient sigma=sqrt(lr) * sqrt(D); EM step multiplies by sqrt(dt)=sqrt(lr)",
        "delta_t": lr,
        "lr": lr,
        "drift_norm": drift_norm,
        "standard_noise_step_norm": standard_noise_step_norm,
        "modified_noise_step_norm": modified_noise_step_norm,
        "sde_sigma_norm_before_dt": sde_sigma_norm,
        "standard_noise_to_drift_ratio": standard_noise_step_norm / max(drift_norm, 1e-12),
        "modified_noise_to_drift_ratio": modified_noise_step_norm / max(drift_norm, 1e-12),
        "em_scaling_consistent_with_delta_t_equals_lr": True,
        "note": "For dtheta=-grad dt - sqrt(lr A)dW and dt=lr, Euler-Maruyama gives a noise step lr*sqrt(A)*xi.",
    }
    save_json(result_dir / "implementation_audit.json", audit)

    def diag_cov_np(x: np.ndarray) -> np.ndarray:
        return np.var(x, axis=0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1], dtype=np.float64)

    rng = np.random.default_rng(seed + 1212)
    loader = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, seed + 1213)[2]
    sgd_updates = []
    for i, (x, y) in enumerate(loader):
        if i >= one_step_samples:
            break
        g = _grad_vector(model, x.to(device), y.to(device), loss_fn)
        sgd_updates.append((-lr * g).cpu().numpy())
    sgd_updates_np = np.stack(sgd_updates)
    sgd_mean = sgd_updates_np.mean(axis=0)
    sgd_cov_diag = diag_cov_np(sgd_updates_np)
    sgd_cov_trace = float(np.sum(sgd_cov_diag))

    one_step_rows = []
    one_step_metrics: dict[str, float] = {}
    for method, c in [("standard_langevin", 1.0), ("modified_langevin", 1.0)]:
        noise_std_t = ref_noise_std if method == "standard_langevin" else ref_modified_noise_std
        noise_std = noise_std_t.cpu().numpy()
        samples = drift.cpu().numpy()[None, :] + lr * c * rng.standard_normal((one_step_samples, n_params)) * noise_std[None, :]
        mean_err = float(np.linalg.norm(samples.mean(axis=0) - sgd_mean) / max(np.linalg.norm(sgd_mean), 1e-12))
        cov_diag = diag_cov_np(samples)
        cov_err = float(np.linalg.norm(cov_diag - sgd_cov_diag) / max(np.linalg.norm(sgd_cov_diag), 1e-12))
        cov_trace = float(np.sum(cov_diag))
        one_step_rows.append({
            "method": method,
            "calibration": c,
            "mean_update_error": mean_err,
            "covariance_error": cov_err,
            "covariance_trace": cov_trace,
            "sgd_covariance_trace": sgd_cov_trace,
            "noise_to_drift_ratio": float(lr * np.linalg.norm(noise_std) / max(drift_norm, 1e-12)),
        })
        one_step_metrics[f"{method}_mean_update_error"] = mean_err
        one_step_metrics[f"{method}_cov_error"] = cov_err

    calibration_rows = []
    best_cal = None
    for c in calibration:
        noise_std = ref_modified_noise_std.cpu().numpy()
        samples = drift.cpu().numpy()[None, :] + lr * c * rng.standard_normal((one_step_samples, n_params)) * noise_std[None, :]
        cov_diag = diag_cov_np(samples)
        cov_err = float(np.linalg.norm(cov_diag - sgd_cov_diag) / max(np.linalg.norm(sgd_cov_diag), 1e-12))
        mean_err = float(np.linalg.norm(samples.mean(axis=0) - sgd_mean) / max(np.linalg.norm(sgd_mean), 1e-12))
        row = {"calibration": c, "mean_update_error": mean_err, "covariance_error": cov_err}
        calibration_rows.append(row)
        if best_cal is None or cov_err < best_cal["covariance_error"]:
            best_cal = row

    torch.manual_seed(seed + 1214)
    q, _ = torch.linalg.qr(torch.randn(n_params, n_dirs, device=device))

    def run_horizon(method: str, c: float = 1.0) -> np.ndarray:
        out = np.zeros((n_runs, max_horizon + 1, n_params), dtype=np.float32)
        for r in range(n_runs):
            torch.manual_seed(seed + 1300 + r + int(1000 * c) + len(method))
            run_model = _mlp_model(device)
            vector_to_parameters(ref_vec.clone(), run_model.parameters())
            run_loader = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, seed + 1400 + r)[2]
            data_iter = iter(run_loader)
            for t in range(max_horizon + 1):
                current = parameters_to_vector(run_model.parameters()).detach()
                out[r, t] = (current - ref_vec).cpu().numpy()
                if t == max_horizon:
                    break
                if method == "sgd":
                    try:
                        x, y = next(data_iter)
                    except StopIteration:
                        data_iter = iter(run_loader)
                        x, y = next(data_iter)
                    update_grad = _grad_vector(run_model, x.to(device), y.to(device), loss_fn)
                    vector_to_parameters(current - lr * update_grad, run_model.parameters())
                else:
                    fg = _grad_vector(run_model, full_x, full_y, loss_fn)
                    if method == "standard_langevin":
                        ns = ref_noise_std
                    else:
                        centered_ns = _estimate_minibatch_noise_std(run_model, fg, config, device, seed + 1500 + 100 * r + t, langevin_noise_batches, loss_fn)
                        ns = _modified_langevin_noise_std(centered_ns, fg)
                    noise = c * ns * torch.randn_like(current)
                    vector_to_parameters(current - lr * fg + lr * noise, run_model.parameters())
        return out

    trajs = {
        "sgd": run_horizon("sgd"),
        "standard_langevin": run_horizon("standard_langevin"),
        "modified_langevin": run_horizon("modified_langevin"),
    }
    calibration_horizon_errors = {}
    for c in calibration:
        trajs[f"modified_c_{c:g}"] = run_horizon("modified_langevin", c=c)

    sgd_mean_path = trajs["sgd"].mean(axis=0)
    sgd_proj = np.einsum("rtd,dk->rtk", trajs["sgd"], q.cpu().numpy())
    sgd_proj_mean = sgd_proj.mean(axis=0)
    sgd_var = np.mean(np.sum((trajs["sgd"] - sgd_mean_path[None, :, :]) ** 2, axis=2), axis=0)
    horizon_rows = []
    figure_rows = []
    for name, arr in trajs.items():
        mean_path = arr.mean(axis=0)
        proj = np.einsum("rtd,dk->rtk", arr, q.cpu().numpy())
        proj_mean = proj.mean(axis=0)
        var_trace = np.mean(np.sum((arr - mean_path[None, :, :]) ** 2, axis=2), axis=0)
        for h in horizons:
            mean_err = float(np.linalg.norm(mean_path[h] - sgd_mean_path[h]) / max(np.linalg.norm(sgd_mean_path[h]), 1e-12))
            proj_err = float(np.linalg.norm(proj_mean[h] - sgd_proj_mean[h]) / max(np.linalg.norm(sgd_proj_mean[h]), 1e-12))
            var_err = float(abs(var_trace[h] - sgd_var[h]) / max(abs(sgd_var[h]), 1e-12))
            row = {
                "horizon": h,
                "method": name,
                "mean_trajectory_error": mean_err,
                "projection_error": proj_err,
                "variance_error": var_err,
                "variance_trace": float(var_trace[h]),
                "sgd_variance_trace": float(sgd_var[h]),
                "final_displacement_error": mean_err,
            }
            horizon_rows.append(row)
            figure_rows.append({
                "step": h,
                "method": name,
                "mean_projection": float(proj_mean[h].mean()),
                "std_projection": float(proj[:, h, :].mean(axis=1).std(ddof=1)),
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "mean_direction_variance": float(var_trace[h]),
                "mean_path_error_to_sgd": mean_err,
                "mean_coord_1": float(proj_mean[h, 0]),
                "mean_coord_2": float(proj_mean[h, 1]) if proj_mean.shape[1] > 1 else 0.0,
            })
        if name.startswith("modified_c_"):
            cal_h = horizons[min(1, len(horizons) - 1)]
            calibration_horizon_errors[name] = float(
                np.linalg.norm(mean_path[cal_h] - sgd_mean_path[cal_h])
                / max(np.linalg.norm(sgd_mean_path[cal_h]), 1e-12)
            )

    standard_short = next(r for r in horizon_rows if r["method"] == "standard_langevin" and r["horizon"] == horizons[0])
    modified_short = next(r for r in horizon_rows if r["method"] == "modified_langevin" and r["horizon"] == horizons[0])
    standard_long = next(r for r in horizon_rows if r["method"] == "standard_langevin" and r["horizon"] == horizons[-1])
    modified_long = next(r for r in horizon_rows if r["method"] == "modified_langevin" and r["horizon"] == horizons[-1])
    best_horizon_cal_name = min(calibration_horizon_errors, key=calibration_horizon_errors.get) if calibration_horizon_errors else ""
    if best_cal is not None and best_cal["calibration"] < 1.0 and best_cal["covariance_error"] < one_step_metrics["modified_langevin_cov_error"]:
        verdict = "modified_noise_overestimated"
    elif one_step_metrics["modified_langevin_cov_error"] <= one_step_metrics["standard_langevin_cov_error"] and modified_long["mean_trajectory_error"] > standard_long["mean_trajectory_error"]:
        verdict = "implementation_consistent_but_modified_is_only_local"
    elif modified_long["projection_error"] > standard_long["projection_error"] and modified_long["mean_trajectory_error"] <= standard_long["mean_trajectory_error"]:
        verdict = "projection_metric_misleading"
    else:
        verdict = "modified_does_not_improve_local_or_long_horizon_match"

    metrics = {
        "standard_noise_scaling_mode": audit["standard_noise_scaling_mode"],
        "modified_noise_scaling_mode": audit["modified_noise_scaling_mode"],
        "standard_mean_update_error": one_step_metrics["standard_langevin_mean_update_error"],
        "modified_mean_update_error": one_step_metrics["modified_langevin_mean_update_error"],
        "standard_cov_error": one_step_metrics["standard_langevin_cov_error"],
        "modified_cov_error": one_step_metrics["modified_langevin_cov_error"],
        "standard_short_horizon_error": standard_short["mean_trajectory_error"],
        "modified_short_horizon_error": modified_short["mean_trajectory_error"],
        "standard_long_horizon_error": standard_long["mean_trajectory_error"],
        "modified_long_horizon_error": modified_long["mean_trajectory_error"],
        "standard_projection_error": standard_long["projection_error"],
        "modified_projection_error": modified_long["projection_error"],
        "modified_best_calibration": float(best_cal["calibration"]) if best_cal else float("nan"),
        "modified_best_calibration_cov_error": float(best_cal["covariance_error"]) if best_cal else float("nan"),
        "modified_best_horizon_calibration": best_horizon_cal_name,
        "modified_calibration_improves_local_match": bool(best_cal and best_cal["covariance_error"] < one_step_metrics["modified_langevin_cov_error"]),
        "implementation_verdict": verdict,
        "pass": True,
    }
    save_json(result_dir / "metrics.json", metrics)
    write_csv(result_dir / "one_step_matching.csv", one_step_rows)
    write_csv(result_dir / "horizon_error.csv", horizon_rows)
    write_csv(result_dir / "calibration_sweep.csv", calibration_rows)
    write_csv(result_dir / "figure_data.csv", figure_rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", **{k.replace(".", "p"): v for k, v in trajs.items()})
    return metrics


def run_exp13(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Main Langevin falsification diagnostic in Hessian flat/sharp buckets."""
    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    model, loaders = _train_reference_mlp(config, device, seed)
    _, _, train_loader, full_loader, _ = loaders
    loss_fn = nn.CrossEntropyLoss()
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    ref_vec = parameters_to_vector(model.parameters()).detach().clone()
    h = _hessian_full_batch(model, full_x, full_y, loss_fn).cpu()
    eigvals, eigvecs = torch.linalg.eigh(h)
    eig_np = eigvals.numpy()
    n_flat = int(config["analysis"].get("n_flat_directions", 4))
    n_sharp = int(config["analysis"].get("n_sharp_directions", 4))
    n_diffusive = int(config["analysis"].get("n_diffusive_directions", n_flat))
    flat_idx = np.argsort(np.abs(eig_np))[:n_flat]
    positive = np.where(eig_np > 0)[0]
    sharp_idx = positive[np.argsort(eig_np[positive])[-n_sharp:]] if len(positive) >= n_sharp else np.argsort(np.abs(eig_np))[-n_sharp:]
    grad_batches = int(config["analysis"].get("gradient_batches", 8))
    grad_eig = []
    for i, (gx, gy) in enumerate(train_loader):
        if i >= grad_batches:
            break
        grad_eig.append((_grad_vector(model, gx.to(device), gy.to(device), loss_fn).cpu() @ eigvecs).numpy())
    grad_eig_arr = np.stack(grad_eig)
    grad_var = grad_eig_arr.var(axis=0, ddof=1)
    predicted_var_proxy = grad_var / (2.0 * np.abs(eig_np) + 1e-6)
    diffusive_idx = np.argsort(predicted_var_proxy)[-n_diffusive:]
    bases = {
        "flat": eigvecs[:, flat_idx].to(device),
        "sharp": eigvecs[:, sharp_idx].to(device),
        "diffusive": eigvecs[:, diffusive_idx].to(device),
    }

    lr = float(config["ensemble"]["lr"])
    ens = int(config["ensemble"]["n_runs"])
    steps = int(config["ensemble"]["steps"])
    methods = config["ensemble"].get("methods", ["sgd", "standard_langevin", "modified_langevin", "gd"])
    noise_batches = int(config["ensemble"].get("langevin_noise_batches", 4))
    eval_every = int(config["analysis"].get("eval_every", max(1, steps // 20)))
    eval_steps = list(range(0, steps + 1, eval_every))
    if eval_steps[-1] != steps:
        eval_steps.append(steps)

    full_grad_ref = _grad_vector(model, full_x, full_y, loss_fn)
    ref_noise_std = _estimate_minibatch_noise_std(model, full_grad_ref, config, device, seed + 1313, noise_batches, loss_fn)
    trajectories: dict[str, dict[str, np.ndarray]] = {
        m: {bucket: np.zeros((ens, len(eval_steps), basis.shape[1]), dtype=np.float32) for bucket, basis in bases.items()}
        for m in methods
    }

    for method in methods:
        for r in range(ens):
            torch.manual_seed(seed + 130000 + 1000 * r + len(method))
            run_model = _mlp_model(device)
            vector_to_parameters(ref_vec.clone(), run_model.parameters())
            _, _, run_loader, _, _ = _mnist_loaders(config["dataset"], config["training"]["batch_size"], True, seed + 1700 + r)
            data_iter = iter(run_loader)
            eval_pos = 0
            for t in range(steps + 1):
                if t == eval_steps[eval_pos]:
                    delta = parameters_to_vector(run_model.parameters()).detach() - ref_vec
                    for bucket, basis in bases.items():
                        trajectories[method][bucket][r, eval_pos] = (delta @ basis).cpu().numpy()
                    eval_pos += 1
                    if eval_pos >= len(eval_steps):
                        break
                if t == steps:
                    break
                current = parameters_to_vector(run_model.parameters()).detach()
                if method == "sgd":
                    try:
                        x, y = next(data_iter)
                    except StopIteration:
                        data_iter = iter(run_loader)
                        x, y = next(data_iter)
                    grad = _grad_vector(run_model, x.to(device), y.to(device), loss_fn)
                    update = -lr * grad
                else:
                    fg = _grad_vector(run_model, full_x, full_y, loss_fn)
                    if method == "gd":
                        update = -lr * fg
                    else:
                        if method == "standard_langevin":
                            noise_std = ref_noise_std
                        else:
                            centered = _estimate_minibatch_noise_std(run_model, fg, config, device, seed + 1900 + 100 * r + t, noise_batches, loss_fn)
                            noise_std = _modified_langevin_noise_std(centered, fg)
                        update = -lr * fg + lr * noise_std * torch.randn_like(current)
                vector_to_parameters(current + update, run_model.parameters())

    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "flat_eigenvalue_abs_max": float(np.max(np.abs(eig_np[flat_idx]))),
        "sharp_eigenvalue_min": float(np.min(eig_np[sharp_idx])),
        "sharp_eigenvalue_max": float(np.max(eig_np[sharp_idx])),
        "diffusive_predicted_variance_min": float(np.min(predicted_var_proxy[diffusive_idx])),
        "diffusive_predicted_variance_max": float(np.max(predicted_var_proxy[diffusive_idx])),
        "diffusive_eigenvalue_abs_max": float(np.max(np.abs(eig_np[diffusive_idx]))),
        "reference_full_grad_norm": float(torch.linalg.norm(full_grad_ref).item()),
    }
    tail = max(3, len(eval_steps) // 3)
    xs_tail = np.asarray(eval_steps[-tail:], dtype=np.float64)
    raw: dict[str, np.ndarray] = {
        "eigenvalues": eig_np,
        "flat_indices": flat_idx,
        "sharp_indices": sharp_idx,
        "diffusive_indices": diffusive_idx,
        "gradient_variance": grad_var,
        "predicted_variance_proxy": predicted_var_proxy,
    }
    for method, vals in trajectories.items():
        for bucket, arr in vals.items():
            raw[f"{method}_{bucket}"] = arr
            direction_var = arr.var(axis=0, ddof=1)
            mean_var = direction_var.mean(axis=1)
            slope = float(np.polyfit(xs_tail, mean_var[-tail:], 1)[0])
            metrics[f"{method}_{bucket}_late_slope"] = slope
            metrics[f"{method}_{bucket}_final_variance"] = float(mean_var[-1])
            metrics[f"{method}_{bucket}_late_mean_variance"] = float(mean_var[-tail:].mean())
            for j, step in enumerate(eval_steps):
                rows.append({
                    "step": int(step),
                    "method": f"{method}_{bucket}",
                    "mean_projection": float(arr[:, j, :].mean()),
                    "std_projection": float(arr[:, j, :].mean(axis=1).std(ddof=1)),
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                    "mean_direction_variance": float(mean_var[j]),
                    "mean_path_error_to_sgd": 0.0,
                    "mean_coord_1": float(arr[:, j, 0].mean()),
                    "mean_coord_2": float(arr[:, j, 1].mean()) if arr.shape[2] > 1 else 0.0,
                })
    metrics["sgd_vs_standard_flat_slope_ratio"] = float(
        metrics.get("sgd_flat_late_slope", 0.0) / max(abs(metrics.get("standard_langevin_flat_late_slope", 0.0)), 1e-30)
    )
    metrics["sgd_vs_standard_sharp_variance_ratio"] = float(
        metrics.get("sgd_sharp_final_variance", 0.0) / max(abs(metrics.get("standard_langevin_sharp_final_variance", 0.0)), 1e-30)
    )
    metrics["sgd_vs_standard_diffusive_slope_ratio"] = float(
        metrics.get("sgd_diffusive_late_slope", 0.0) / max(abs(metrics.get("standard_langevin_diffusive_late_slope", 0.0)), 1e-30)
    )
    metrics["pass"] = bool(
        metrics.get("sgd_flat_late_slope", -np.inf) > metrics.get("standard_langevin_flat_late_slope", np.inf)
        or metrics.get("sgd_diffusive_late_slope", -np.inf) > metrics.get("standard_langevin_diffusive_late_slope", np.inf)
    )
    write_csv(result_dir / "figure_data.csv", rows)
    write_csv(result_dir / "bucketed_statistics.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", **raw)
    return metrics


def run_exp14(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Learning-rate and batch-size scaling of flat/sharp projected variance."""
    device = _device(config.get("device", "cpu"))
    seed = int(config["seed"])
    model, loaders = _train_reference_mlp(config, device, seed)
    _, _, _, full_loader, _ = loaders
    loss_fn = nn.CrossEntropyLoss()
    full_x, full_y = next(iter(full_loader))
    full_x, full_y = full_x.to(device), full_y.to(device)
    h = _hessian_full_batch(model, full_x, full_y, loss_fn).cpu()
    eigvals, eigvecs = torch.linalg.eigh(h)
    eig_np = eigvals.numpy()
    n_flat = int(config["analysis"].get("n_flat_directions", 4))
    n_sharp = int(config["analysis"].get("n_sharp_directions", 4))
    flat_idx = np.argsort(np.abs(eig_np))[:n_flat]
    positive = np.where(eig_np > 0)[0]
    sharp_idx = positive[np.argsort(eig_np[positive])[-n_sharp:]] if len(positive) >= n_sharp else np.argsort(np.abs(eig_np))[-n_sharp:]
    bases = {"flat": eigvecs[:, flat_idx].to(device), "sharp": eigvecs[:, sharp_idx].to(device)}
    ref_vec = parameters_to_vector(model.parameters()).detach().clone()

    lrs = [float(x) for x in config["ensemble"].get("lr_grid", [config["ensemble"]["lr"]])]
    batches = [int(x) for x in config["ensemble"].get("batch_grid", [config["training"]["batch_size"]])]
    ens = int(config["ensemble"].get("n_runs", 4))
    steps = int(config["ensemble"]["steps"])
    eval_every = int(config["analysis"].get("eval_every", max(1, steps // 10)))
    eval_steps = list(range(0, steps + 1, eval_every))
    if eval_steps[-1] != steps:
        eval_steps.append(steps)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"lr_grid": lrs, "batch_grid": batches, "n_runs": ens, "steps": steps}
    raw: dict[str, np.ndarray] = {"eigenvalues": eig_np, "flat_indices": flat_idx, "sharp_indices": sharp_idx}

    for batch_size in batches:
        for lr in lrs:
            for bucket, basis in bases.items():
                arr = np.zeros((ens, len(eval_steps), basis.shape[1]), dtype=np.float32)
                for r in range(ens):
                    torch.manual_seed(seed + 140000 + r + int(round(lr * 100000)) + batch_size)
                    run_model = _mlp_model(device)
                    vector_to_parameters(ref_vec.clone(), run_model.parameters())
                    _, _, run_loader, _, _ = _mnist_loaders(config["dataset"], batch_size, True, seed + r)
                    data_iter = iter(run_loader)
                    eval_pos = 0
                    for t in range(steps + 1):
                        if t == eval_steps[eval_pos]:
                            delta = parameters_to_vector(run_model.parameters()).detach() - ref_vec
                            arr[r, eval_pos] = (delta @ basis).detach().cpu().numpy()
                            eval_pos += 1
                            if eval_pos >= len(eval_steps):
                                break
                        if t == steps:
                            break
                        try:
                            x, y = next(data_iter)
                        except StopIteration:
                            data_iter = iter(run_loader)
                            x, y = next(data_iter)
                        grad = _grad_vector(run_model, x.to(device), y.to(device), loss_fn)
                        current = parameters_to_vector(run_model.parameters()).detach()
                        vector_to_parameters(current - lr * grad, run_model.parameters())
                key = f"{bucket}_lr_{lr:g}_b_{batch_size}".replace(".", "p")
                raw[key] = arr
                direction_var = arr.var(axis=0, ddof=1)
                mean_var = direction_var.mean(axis=1)
                tail = max(3, len(eval_steps) // 3)
                slope = float(np.polyfit(np.asarray(eval_steps[-tail:], dtype=np.float64), mean_var[-tail:], 1)[0])
                plateau = float(mean_var[-tail:].mean())
                metric_prefix = f"lr_{lr:g}_batch_{batch_size}_{bucket}"
                metrics[f"{metric_prefix}_late_slope"] = slope
                metrics[f"{metric_prefix}_late_mean_variance"] = plateau
                summary_rows.append({"lr": lr, "batch_size": batch_size, "bucket": bucket, "late_slope": slope, "late_mean_variance": plateau})
                for j, step in enumerate(eval_steps):
                    rows.append({
                        "step": int(step),
                        "method": f"{bucket}_lr={lr:g}_B={batch_size}",
                        "lr": lr,
                        "batch_size": batch_size,
                        "bucket": bucket,
                        "mean_projection": float(arr[:, j, :].mean()),
                        "std_projection": float(arr[:, j, :].mean(axis=1).std(ddof=1)),
                        "ci95_low": 0.0,
                        "ci95_high": 0.0,
                        "mean_direction_variance": float(mean_var[j]),
                    })
    flat_rows = [r for r in summary_rows if r["bucket"] == "flat"]
    if len({r["lr"] for r in flat_rows}) >= 3:
        by_lr = []
        for lr in sorted({r["lr"] for r in flat_rows}):
            vals = [r["late_slope"] for r in flat_rows if r["lr"] == lr]
            by_lr.append((lr, float(np.mean(vals))))
        metrics["flat_slope_lr_correlation"] = float(np.corrcoef([x for x, _ in by_lr], [y for _, y in by_lr])[0, 1])
    if len({r["batch_size"] for r in flat_rows}) >= 3:
        by_b = []
        for b in sorted({r["batch_size"] for r in flat_rows}):
            vals = [r["late_slope"] for r in flat_rows if r["batch_size"] == b]
            by_b.append((1.0 / b, float(np.mean(vals))))
        metrics["flat_slope_inv_batch_correlation"] = float(np.corrcoef([x for x, _ in by_b], [y for _, y in by_b])[0, 1])
    metrics["pass"] = bool(metrics.get("flat_slope_lr_correlation", 0.0) > 0.3)
    write_csv(result_dir / "figure_data.csv", rows)
    write_csv(result_dir / "scaling_summary.csv", summary_rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", **raw)
    return metrics


def run_exp15(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Check Var(G_i) alignment with Hessian eigenvalues using saved EXP2 artifacts."""
    source = Path(config["analysis"].get("source_result_dir", "src/scripts/exp6/results/exp2_eq32_full"))
    raw = np.load(source / "raw_outputs.npz")
    eig = np.asarray(raw["eigenvalues"], dtype=np.float64)
    grad = np.asarray(raw["grad_eig"], dtype=np.float64)
    g_var = grad.var(axis=0, ddof=1)
    positive_mask = eig > float(config["analysis"].get("positive_lambda_min", 1e-8))
    selected = np.where(positive_mask)[0]
    if len(selected) < 3:
        selected = np.argsort(np.abs(eig))[-min(48, len(eig)):]
    x = eig[selected]
    y = g_var[selected]
    eps = 1e-12
    log_corr = float(np.corrcoef(np.log(x + eps), np.log(y + eps))[0, 1]) if len(selected) >= 3 else float("nan")
    lin_coef = np.polyfit(x, y, 1) if len(selected) >= 2 else [float("nan"), float("nan")]
    y_hat = lin_coef[0] * x + lin_coef[1]
    r2 = float(1.0 - np.sum((y - y_hat) ** 2) / max(np.sum((y - y.mean()) ** 2), eps)) if len(selected) >= 2 else float("nan")
    gamma_i = y / np.maximum(x, eps)
    gamma_mean, gamma_std, gamma_lo, gamma_hi = mean_std_ci(gamma_i, axis=0)
    rows = []
    for i in selected:
        rows.append({
            "direction": int(i),
            "eigenvalue": float(eig[i]),
            "gradient_noise_variance": float(g_var[i]),
            "gamma_i": float(g_var[i] / max(eig[i], eps)),
        })
    write_csv(result_dir / "figure_data.csv", rows)
    write_csv(result_dir / "alignment_summary.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", eigenvalues=eig, gradient_variance=g_var, selected=selected)
    return {
        "source_result_dir": str(source),
        "n_directions": int(len(selected)),
        "positive_lambda_min": float(np.min(x)) if len(x) else float("nan"),
        "positive_lambda_max": float(np.max(x)) if len(x) else float("nan"),
        "grad_noise_lambda_log_correlation": log_corr,
        "linear_fit_slope": float(lin_coef[0]),
        "linear_fit_intercept": float(lin_coef[1]),
        "linear_fit_r2": r2,
        "gamma_mean": float(gamma_mean),
        "gamma_std": float(gamma_std),
        "gamma_cv": float(gamma_std / max(abs(gamma_mean), eps)),
        "gamma_ci95_low": float(gamma_lo),
        "gamma_ci95_high": float(gamma_hi),
        "pass": bool(np.isfinite(log_corr) and log_corr > 0.3),
    }


def _make_spd_matrix_np(dim: int, eig_min: float, eig_max: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eigvals = np.linspace(eig_min, eig_max, dim)
    return q @ np.diag(eigvals) @ q.T


def _log_gaussian_density_np(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    dim = x.shape[0]
    chol = np.linalg.cholesky(cov)
    diff = x - mean
    y = np.linalg.solve(chol, diff)
    quad = float(y @ y)
    logdet = float(2.0 * np.sum(np.log(np.diag(chol))))
    return float(-0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad))


def _estimate_quadratic_gradient_cov_np(w: np.ndarray, sigma_g: float, sigma_h: float, dim: int) -> np.ndarray:
    return sigma_g**2 * np.eye(dim) + sigma_h**2 * ((w @ w) * np.eye(dim) + np.outer(w, w)) / dim


def _run_evidence_quadratic_trajectory(
    *,
    dim: int,
    num_steps: int,
    eta: float,
    sigma_g: float,
    sigma_h: float,
    eig_min: float,
    eig_max: float,
    ridge: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h_bar = _make_spd_matrix_np(dim, eig_min=eig_min, eig_max=eig_max, seed=seed)
    w = rng.normal(size=dim)
    log_lan_steps = np.zeros(num_steps, dtype=np.float64)
    log_mod_steps = np.zeros(num_steps, dtype=np.float64)
    log_ratio_direct_steps = np.zeros(num_steps, dtype=np.float64)
    log_ratio_formula_steps = np.zeros(num_steps, dtype=np.float64)

    for step in range(num_steps):
        g_sample = sigma_g * rng.normal(size=dim)
        a = rng.normal(size=(dim, dim))
        e = (a + a.T) / np.sqrt(2.0 * dim)
        h_sample = h_bar + sigma_h * e
        grad_sample = g_sample + h_sample @ w
        w_next = w - eta * grad_sample

        mean_grad = h_bar @ w
        sigma = _estimate_quadratic_gradient_cov_np(w, sigma_g=sigma_g, sigma_h=sigma_h, dim=dim)
        sigma = sigma + ridge * np.eye(dim)
        sigma_mod = sigma + np.outer(mean_grad, mean_grad)
        transition_mean = w - eta * mean_grad
        cov_lan = eta**2 * sigma
        cov_mod = eta**2 * sigma_mod

        log_lan = _log_gaussian_density_np(w_next, transition_mean, cov_lan)
        log_mod = _log_gaussian_density_np(w_next, transition_mean, cov_mod)
        residual = w_next - w + eta * mean_grad
        logdet_sigma = np.linalg.slogdet(sigma)[1]
        logdet_sigma_mod = np.linalg.slogdet(sigma_mod)[1]
        quad_lan = residual @ np.linalg.solve(sigma, residual)
        quad_mod = residual @ np.linalg.solve(sigma_mod, residual)
        log_ratio_formula = 0.5 * (logdet_sigma - logdet_sigma_mod + (quad_lan - quad_mod) / (eta**2))

        log_lan_steps[step] = log_lan
        log_mod_steps[step] = log_mod
        log_ratio_direct_steps[step] = log_mod - log_lan
        log_ratio_formula_steps[step] = log_ratio_formula
        w = w_next

    return {
        "log_lan_steps": log_lan_steps,
        "log_mod_steps": log_mod_steps,
        "log_ratio_direct_steps": log_ratio_direct_steps,
        "log_ratio_formula_steps": log_ratio_formula_steps,
        "cumulative_lan": np.cumsum(log_lan_steps),
        "cumulative_mod": np.cumsum(log_mod_steps),
        "cumulative_ratio_direct": np.cumsum(log_ratio_direct_steps),
        "cumulative_ratio_formula": np.cumsum(log_ratio_formula_steps),
    }


def run_exp16(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Trajectory log-evidence comparison for quadratic SGD vs Gaussian surrogates."""
    params = config.get("parameters", {})
    dim = int(params.get("dim", 10))
    num_steps = int(params.get("num_steps", params.get("steps", 1000)))
    eta = float(params.get("eta", params.get("lr", 0.03)))
    sigma_g = float(params.get("sigma_g", 0.5))
    sigma_h = float(params.get("sigma_h", 0.2))
    eig_min = float(params.get("eig_min", 0.3))
    eig_max = float(params.get("eig_max", 3.0))
    ridge = float(params.get("ridge", 1e-8))
    n_runs = int(params.get("n_runs", 1))
    base_seed = int(config.get("seed", params.get("seed", 42)))

    runs = []
    rows: list[dict[str, Any]] = []
    final_lan = []
    final_mod = []
    final_ratio = []
    max_formula_errors = []
    mean_formula_errors = []
    for run in range(n_runs):
        out = _run_evidence_quadratic_trajectory(
            dim=dim,
            num_steps=num_steps,
            eta=eta,
            sigma_g=sigma_g,
            sigma_h=sigma_h,
            eig_min=eig_min,
            eig_max=eig_max,
            ridge=ridge,
            seed=base_seed + run,
        )
        runs.append(out)
        err = np.abs(out["log_ratio_direct_steps"] - out["log_ratio_formula_steps"])
        final_lan.append(float(out["cumulative_lan"][-1]))
        final_mod.append(float(out["cumulative_mod"][-1]))
        final_ratio.append(float(out["cumulative_ratio_direct"][-1]))
        max_formula_errors.append(float(err.max()))
        mean_formula_errors.append(float(err.mean()))
        for step in range(num_steps):
            rows.append({
                "run": run,
                "step": step + 1,
                "method": "standard_langevin",
                "cumulative_log_evidence": float(out["cumulative_lan"][step]),
                "cumulative_log_ratio": float(out["cumulative_ratio_direct"][step]),
                "cumulative_log_ratio_formula": float(out["cumulative_ratio_formula"][step]),
                "step_log_evidence": float(out["log_lan_steps"][step]),
                "step_log_ratio": float(out["log_ratio_direct_steps"][step]),
                "formula_error": float(err[step]),
            })
            rows.append({
                "run": run,
                "step": step + 1,
                "method": "modified_langevin",
                "cumulative_log_evidence": float(out["cumulative_mod"][step]),
                "cumulative_log_ratio": float(out["cumulative_ratio_direct"][step]),
                "cumulative_log_ratio_formula": float(out["cumulative_ratio_formula"][step]),
                "step_log_evidence": float(out["log_mod_steps"][step]),
                "step_log_ratio": float(out["log_ratio_direct_steps"][step]),
                "formula_error": float(err[step]),
            })

    final_lan_arr = np.asarray(final_lan, dtype=np.float64)
    final_mod_arr = np.asarray(final_mod, dtype=np.float64)
    final_ratio_arr = np.asarray(final_ratio, dtype=np.float64)
    ratio_mean, ratio_std, ratio_lo, ratio_hi = mean_std_ci(final_ratio_arr, axis=0)
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        **{f"run_{i}_{k}": v for i, d in enumerate(runs) for k, v in d.items()},
        final_log_evidence_standard_langevin=final_lan_arr,
        final_log_evidence_modified_langevin=final_mod_arr,
        final_log_ratio_modified_minus_standard=final_ratio_arr,
    )
    return {
        "dim": dim,
        "num_steps": num_steps,
        "eta": eta,
        "sigma_g": sigma_g,
        "sigma_h": sigma_h,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "ridge": ridge,
        "n_runs": n_runs,
        "final_log_evidence_standard_langevin_mean": float(final_lan_arr.mean()),
        "final_log_evidence_modified_langevin_mean": float(final_mod_arr.mean()),
        "final_log_ratio_modified_minus_standard_mean": float(ratio_mean),
        "final_log_ratio_modified_minus_standard_std": float(ratio_std),
        "final_log_ratio_modified_minus_standard_ci95_low": float(ratio_lo),
        "final_log_ratio_modified_minus_standard_ci95_high": float(ratio_hi),
        "final_log_ratio_positive_fraction": float(np.mean(final_ratio_arr > 0.0)),
        "max_formula_error": float(np.max(max_formula_errors)),
        "mean_formula_error": float(np.mean(mean_formula_errors)),
        "pass": bool(ratio_mean < 0.0 and np.max(max_formula_errors) < 1e-7),
    }


def run_exp17(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Ensemble covariance evolution: standard FP vs discrete/raw-moment FP."""
    params = config.get("parameters", {})
    dim = int(params.get("dim", 10))
    num_steps = int(params.get("num_steps", params.get("steps", 1000)))
    eta = float(params.get("eta", params.get("lr", 0.03)))
    sigma_g = float(params.get("sigma_g", 0.5))
    eig_min = float(params.get("eig_min", 0.3))
    eig_max = float(params.get("eig_max", 3.0))
    n_runs = int(params.get("n_runs", 10000))
    initial_scale = float(params.get("initial_scale", 1.0))
    seed = int(config.get("seed", params.get("seed", 42)))
    rng = np.random.default_rng(seed)

    h = _make_spd_matrix_np(dim, eig_min=eig_min, eig_max=eig_max, seed=seed)
    a = np.eye(dim) - eta * h
    sigma_g_mat = sigma_g**2 * np.eye(dim)
    w = initial_scale * rng.normal(size=(n_runs, dim))
    pi_standard = np.cov(w, rowvar=False, ddof=1)
    pi_discrete = pi_standard.copy()
    eval_every = int(params.get("eval_every", max(1, num_steps // 100)))
    eval_steps = list(range(0, num_steps + 1, eval_every))
    if eval_steps[-1] != num_steps:
        eval_steps.append(num_steps)

    rows: list[dict[str, Any]] = []
    empirical_trace = []
    standard_trace = []
    discrete_trace = []
    standard_error = []
    discrete_error = []
    eval_set = set(eval_steps)

    def record(step: int) -> None:
        emp = np.cov(w, rowvar=False, ddof=1)
        err_std = float(np.linalg.norm(emp - pi_standard, ord="fro") / max(np.linalg.norm(emp, ord="fro"), 1e-12))
        err_disc = float(np.linalg.norm(emp - pi_discrete, ord="fro") / max(np.linalg.norm(emp, ord="fro"), 1e-12))
        empirical_trace.append(float(np.trace(emp)))
        standard_trace.append(float(np.trace(pi_standard)))
        discrete_trace.append(float(np.trace(pi_discrete)))
        standard_error.append(err_std)
        discrete_error.append(err_disc)
        for method, trace, err in [
            ("empirical_sgd", float(np.trace(emp)), 0.0),
            ("standard_fp", float(np.trace(pi_standard)), err_std),
            ("discrete_fp", float(np.trace(pi_discrete)), err_disc),
        ]:
            rows.append({
                "step": step,
                "method": method,
                "covariance_trace": trace,
                "relative_frobenius_error": err,
                "mean_direction_variance": trace / dim,
                "mean_projection": trace / dim,
                "std_projection": 0.0,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "mean_path_error_to_sgd": err,
            })

    record(0)
    for step in range(1, num_steps + 1):
        g = sigma_g * rng.normal(size=(n_runs, dim))
        w = w @ a.T - eta * g
        pi_standard = pi_standard - eta * (h @ pi_standard + pi_standard @ h) + eta**2 * sigma_g_mat
        pi_discrete = a @ pi_discrete @ a.T + eta**2 * sigma_g_mat
        if step in eval_set:
            record(step)

    eval_steps_arr = np.asarray(eval_steps, dtype=np.int64)
    emp_arr = np.asarray(empirical_trace, dtype=np.float64)
    std_arr = np.asarray(standard_trace, dtype=np.float64)
    disc_arr = np.asarray(discrete_trace, dtype=np.float64)
    std_err_arr = np.asarray(standard_error, dtype=np.float64)
    disc_err_arr = np.asarray(discrete_error, dtype=np.float64)
    improvement = std_err_arr / np.maximum(disc_err_arr, 1e-12)
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        eval_steps=eval_steps_arr,
        empirical_trace=emp_arr,
        standard_trace=std_arr,
        discrete_trace=disc_arr,
        standard_error=std_err_arr,
        discrete_error=disc_err_arr,
        hessian=h,
        sigma_g=sigma_g_mat,
    )
    return {
        "dim": dim,
        "num_steps": num_steps,
        "eta": eta,
        "sigma_g": sigma_g,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "n_runs": n_runs,
        "initial_scale": initial_scale,
        "final_standard_relative_frobenius_error": float(std_err_arr[-1]),
        "final_discrete_relative_frobenius_error": float(disc_err_arr[-1]),
        "final_error_improvement_standard_over_discrete": float(improvement[-1]),
        "mean_error_improvement_standard_over_discrete": float(np.mean(improvement[1:])) if len(improvement) > 1 else float(improvement[0]),
        "discrete_better_fraction": float(np.mean(disc_err_arr <= std_err_arr)),
        "final_empirical_covariance_trace": float(emp_arr[-1]),
        "final_standard_covariance_trace": float(std_arr[-1]),
        "final_discrete_covariance_trace": float(disc_arr[-1]),
        "pass": bool(
            std_err_arr[-1] > 2.0 * disc_err_arr[-1]
            and float(np.mean(disc_err_arr <= std_err_arr)) > 0.8
        ),
    }


def _gradient_covariance_np(w: np.ndarray, sigma_g: float, sigma_h: float, dim: int, batch_size: int = 1) -> np.ndarray:
    return _estimate_quadratic_gradient_cov_np(w, sigma_g, sigma_h, dim) / max(int(batch_size), 1)


def _covariance_np(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(x.shape[0] - 1, 1)


def _moment_rows(x: np.ndarray, ys: dict[str, np.ndarray], x_name: str = "step") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, y in ys.items():
        for xi, yi in zip(x, y):
            rows.append({"step": float(xi), "method": method, "variance": float(yi), x_name: float(xi)})
    return rows


def _mlp386_reference_and_grads(config: dict[str, Any], n_batches: int, batch_size: int, seed: int, device: torch.device):
    cfg = copy.deepcopy(config)
    cfg.setdefault("dataset", {"name": "mnist", "sample_size": 1024, "val_size": 256, "subset": "first", "normalize": False})
    cfg.setdefault("training", {"batch_size": batch_size, "replacement": True, "lr": 0.1, "reference_steps": 20})
    cfg["training"]["batch_size"] = batch_size
    model, loaders = _train_reference_mlp(cfg, device, seed)
    train_ds, _, _, _, _ = loaders
    x_train, y_train = _dataset_to_tensors(train_ds, device)
    loss_fn = nn.CrossEntropyLoss()
    full_grad = _grad_vector(model, x_train, y_train, loss_fn).detach().cpu().numpy()
    rng = np.random.default_rng(seed + 101)
    grads = []
    for _ in range(n_batches):
        idx = rng.integers(0, len(train_ds), size=batch_size)
        xb = x_train[torch.as_tensor(idx, dtype=torch.long, device=device)]
        yb = y_train[torch.as_tensor(idx, dtype=torch.long, device=device)]
        grads.append(_grad_vector(model, xb, yb, loss_fn).detach().cpu().numpy())
    grad_samples = np.asarray(grads, dtype=np.float64)
    return full_grad.astype(np.float64), grad_samples, int(sum(p.numel() for p in model.parameters()))


def run_exp18(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Learning-rate scaling of centered covariance and raw second moment."""
    p = config.get("parameters", {})
    mode = p.get("mode", "toy")
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    etas = np.asarray(p.get("etas", [0.01, 0.03, 0.1, 0.2]), dtype=np.float64)
    n_runs = int(p.get("n_runs", 20000))
    if mode == "mlp386":
        device = _device(config.get("device", "cpu"))
        full_grad, grad_samples, dim = _mlp386_reference_and_grads(
            config, int(p.get("n_batches", n_runs)), int(p.get("batch_size", config.get("training", {}).get("batch_size", 64))), seed, device
        )
        noise = grad_samples - full_grad[None, :]
        sigma_trace = float(np.sum(np.var(noise, axis=0, ddof=1)))
        mean_grad = full_grad
    else:
        dim = int(p.get("dim", 10))
        h = _make_spd_matrix_np(dim, float(p.get("eig_min", 0.3)), float(p.get("eig_max", 3.0)), seed)
        w = rng.normal(size=dim)
        mean_grad = h @ w
        sigma = _gradient_covariance_np(w, float(p.get("sigma_g", 0.5)), float(p.get("sigma_h", 0.0)), dim)
        noise = rng.multivariate_normal(np.zeros(dim), sigma, size=n_runs)
        sigma_trace = float(np.trace(sigma))

    empirical_centered, empirical_raw, pred_centered, pred_raw = [], [], [], []
    base_noise = noise[:n_runs]
    for eta in etas:
        delta = -eta * (mean_grad[None, :] + base_noise)
        centered = delta - delta.mean(axis=0, keepdims=True)
        empirical_centered.append(float(np.mean(np.sum(centered**2, axis=1))))
        empirical_raw.append(float(np.mean(np.sum(delta**2, axis=1))))
        pred_centered.append(float(eta**2 * sigma_trace))
        pred_raw.append(float(eta**2 * (sigma_trace + mean_grad @ mean_grad)))
    emp_c = np.asarray(empirical_centered)
    emp_r = np.asarray(empirical_raw)
    pred_c = np.asarray(pred_centered)
    pred_r = np.asarray(pred_raw)
    rows = _moment_rows(etas, {
        "empirical_centered_covariance": emp_c,
        "eta2_Tr_Sigma": pred_c,
        "empirical_raw_second_moment": emp_r,
        "eta2_Tr_Sigma_plus_ggT": pred_r,
    }, x_name="eta")
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", etas=etas, empirical_centered=emp_c, empirical_raw=emp_r, pred_centered=pred_c, pred_raw=pred_r, mean_grad=mean_grad)
    return {
        "mode": mode,
        "dim": int(dim),
        "n_runs": int(len(base_noise)),
        "mean_grad_norm": float(np.linalg.norm(mean_grad)),
        "centered_loglog_slope": float(np.polyfit(np.log(etas), np.log(emp_c), 1)[0]),
        "raw_loglog_slope": float(np.polyfit(np.log(etas), np.log(emp_r), 1)[0]),
        "centered_prediction_relative_error": float(np.linalg.norm(emp_c - pred_c) / max(np.linalg.norm(emp_c), 1e-12)),
        "raw_prediction_relative_error": float(np.linalg.norm(emp_r - pred_r) / max(np.linalg.norm(emp_r), 1e-12)),
        "raw_over_centered_ratio_at_max_eta": float(emp_r[-1] / max(emp_c[-1], 1e-12)),
        "pass": bool(np.linalg.norm(emp_r - pred_r) < np.linalg.norm(emp_r - pred_c)),
    }


def run_exp20(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Directional raw moment: parallel to gradient vs orthogonal directions."""
    p = config.get("parameters", {})
    mode = p.get("mode", "toy")
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    n_runs = int(p.get("n_runs", 20000))
    eta = float(p.get("eta", 0.03))
    if mode == "mlp386":
        device = _device(config.get("device", "cpu"))
        full_grad, grad_samples, dim = _mlp386_reference_and_grads(config, int(p.get("n_batches", n_runs)), int(p.get("batch_size", 64)), seed, device)
        gbar = full_grad
        delta = -eta * grad_samples
    else:
        dim = int(p.get("dim", 10))
        h = _make_spd_matrix_np(dim, float(p.get("eig_min", 0.3)), float(p.get("eig_max", 3.0)), seed)
        w = float(p.get("point_scale", 2.0)) * rng.normal(size=dim)
        gbar = h @ w
        noise = float(p.get("sigma_g", 0.5)) * rng.normal(size=(n_runs, dim))
        delta = -eta * (gbar[None, :] + noise)
    u = gbar / max(np.linalg.norm(gbar), 1e-12)
    q, _ = np.linalg.qr(rng.normal(size=(int(dim), int(dim))))
    q = q - u[:, None] * (u @ q)[None, :]
    q, _ = np.linalg.qr(q)
    orth = q[:, : int(dim) - 1]
    centered = delta - delta.mean(axis=0, keepdims=True)
    par_raw = (delta @ u) ** 2
    orth_raw = (delta @ orth) ** 2
    par_c = (centered @ u) ** 2
    orth_c = (centered @ orth) ** 2
    rows = [{"step": 0, "method": "parallel_to_gradient_raw", "variance": float(par_raw.mean())},
            {"step": 0, "method": "orthogonal_raw_mean", "variance": float(orth_raw.mean())},
            {"step": 0, "method": "parallel_to_gradient_centered", "variance": float(par_c.mean())},
            {"step": 0, "method": "orthogonal_centered_mean", "variance": float(orth_c.mean())}]
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        mean_grad=gbar,
        gradient_direction=u,
        orth_basis=orth,
        par_raw=par_raw,
        orth_raw=orth_raw,
        par_centered=par_c,
        orth_centered=orth_c,
    )
    return {
        "mode": mode,
        "dim": int(dim),
        "mean_grad_norm": float(np.linalg.norm(gbar)),
        "parallel_raw_second_moment": float(par_raw.mean()),
        "orthogonal_raw_second_moment_mean": float(orth_raw.mean()),
        "parallel_over_orthogonal_raw_ratio": float(par_raw.mean() / max(orth_raw.mean(), 1e-12)),
        "parallel_centered_covariance": float(par_c.mean()),
        "orthogonal_centered_covariance_mean": float(orth_c.mean()),
        "pass": bool(par_raw.mean() > orth_raw.mean()),
    }


def run_exp21(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Long-time ensemble covariance mismatch: standard FP vs discrete FP."""
    return run_exp17(config, result_dir)


def run_exp23(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Batch-size scaling: centered noise vanishes as 1/B, raw moment keeps ggT floor."""
    p = config.get("parameters", {})
    mode = p.get("mode", "toy")
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    eta = float(p.get("eta", 0.03))
    batch_sizes = np.asarray(p.get("batch_sizes", [1, 2, 4, 8, 16, 64, 256, 1024]), dtype=np.int64)
    n_runs = int(p.get("n_runs", 20000))
    if mode == "mlp386":
        device = _device(config.get("device", "cpu"))
        base_b = int(p.get("base_batch_size", 32))
        full_grad, _, dim = _mlp386_reference_and_grads(config, 2, base_b, seed, device)
        cfg = copy.deepcopy(config)
        cfg.setdefault("dataset", {"name": "mnist", "sample_size": 1024, "val_size": 256, "subset": "first", "normalize": False})
        cfg.setdefault("training", {"batch_size": base_b, "replacement": True, "lr": 0.1, "reference_steps": 20})
        model, loaders = _train_reference_mlp(cfg, device, seed)
        train_ds, _, _, _, _ = loaders
        x_train, y_train = _dataset_to_tensors(train_ds, device)
        loss_fn = nn.CrossEntropyLoss()
        empirical_centered, empirical_raw, pred_centered, pred_raw = [], [], [], []
        for b in batch_sizes:
            grads = []
            for _ in range(n_runs):
                idx = rng.integers(0, len(train_ds), size=int(b))
                xb = x_train[torch.as_tensor(idx, dtype=torch.long, device=device)]
                yb = y_train[torch.as_tensor(idx, dtype=torch.long, device=device)]
                grads.append(_grad_vector(model, xb, yb, loss_fn).detach().cpu().numpy())
            g_arr = np.asarray(grads)
            noise = g_arr - full_grad[None, :]
            delta = -eta * g_arr
            centered = delta - delta.mean(axis=0, keepdims=True)
            empirical_centered.append(float(np.mean(np.sum(centered**2, axis=1))))
            empirical_raw.append(float(np.mean(np.sum(delta**2, axis=1))))
            sigma_trace = float(np.sum(np.var(noise, axis=0, ddof=1)))
            pred_centered.append(float(eta**2 * sigma_trace))
            pred_raw.append(float(eta**2 * (sigma_trace + full_grad @ full_grad)))
        mean_grad = full_grad
    else:
        dim = int(p.get("dim", 10))
        h = _make_spd_matrix_np(dim, float(p.get("eig_min", 0.3)), float(p.get("eig_max", 3.0)), seed)
        w = float(p.get("point_scale", 2.0)) * rng.normal(size=dim)
        mean_grad = h @ w
        empirical_centered, empirical_raw, pred_centered, pred_raw = [], [], [], []
        for b in batch_sizes:
            sigma = _gradient_covariance_np(w, float(p.get("sigma_g", 0.5)), float(p.get("sigma_h", 0.0)), dim, int(b))
            noise = rng.multivariate_normal(np.zeros(dim), sigma, size=n_runs)
            delta = -eta * (mean_grad[None, :] + noise)
            centered = delta - delta.mean(axis=0, keepdims=True)
            empirical_centered.append(float(np.mean(np.sum(centered**2, axis=1))))
            empirical_raw.append(float(np.mean(np.sum(delta**2, axis=1))))
            pred_centered.append(float(eta**2 * np.trace(sigma)))
            pred_raw.append(float(eta**2 * (np.trace(sigma) + mean_grad @ mean_grad)))
    emp_c, emp_r, pred_c, pred_r = map(np.asarray, [empirical_centered, empirical_raw, pred_centered, pred_raw])
    rows = _moment_rows(batch_sizes, {
        "empirical_centered_covariance": emp_c,
        "eta2_Tr_Sigma_over_B": pred_c,
        "empirical_raw_second_moment": emp_r,
        "eta2_Tr_Sigma_over_B_plus_ggT": pred_r,
    }, x_name="batch_size")
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", batch_sizes=batch_sizes, empirical_centered=emp_c, empirical_raw=emp_r, pred_centered=pred_c, pred_raw=pred_r, mean_grad=mean_grad)
    floor = eta**2 * float(mean_grad @ mean_grad)
    return {
        "mode": mode,
        "mean_grad_norm": float(np.linalg.norm(mean_grad)),
        "centered_large_batch_ratio_to_b1": float(emp_c[-1] / max(emp_c[0], 1e-12)),
        "raw_large_batch_ratio_to_b1": float(emp_r[-1] / max(emp_r[0], 1e-12)),
        "large_batch_raw_over_deterministic_floor": float(emp_r[-1] / max(floor, 1e-12)),
        "pass": bool(emp_c[-1] < emp_c[0] and emp_r[-1] > floor * 0.8),
    }


def run_exp27(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Stability boundary where standard Langevin predicts stability but discrete SGD is unstable."""
    p = config.get("parameters", {})
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    etas = np.asarray(p.get("etas", [0.05, 0.1, 0.2, 0.3, 0.4]), dtype=np.float64)
    gammas = np.asarray(p.get("gammas", [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]), dtype=np.float64)
    lam = float(p.get("lambda", 1.0))
    d = float(p.get("d", 0.2))
    steps = int(p.get("num_steps", p.get("steps", 500)))
    n_runs = int(p.get("n_runs", 8000))
    threshold = float(p.get("unstable_variance_threshold", 20.0))
    rows = []
    empirical_unstable = np.zeros((len(etas), len(gammas)), dtype=np.float64)
    discrete_unstable = np.zeros_like(empirical_unstable)
    standard_unstable = np.zeros_like(empirical_unstable)
    example = None
    for i, eta in enumerate(etas):
        for j, gamma in enumerate(gammas):
            theta = np.zeros(n_runs, dtype=np.float64)
            trace = []
            sample_keep = None
            for step in range(steps + 1):
                if step % max(1, steps // 100) == 0:
                    trace.append(float(np.var(theta, ddof=1)))
                if step == steps:
                    break
                h = lam + np.sqrt(gamma) * rng.normal(size=n_runs)
                g = np.sqrt(d) * rng.normal(size=n_runs)
                theta = (1.0 - eta * h) * theta - eta * g
                if sample_keep is None and (2.0 * lam - eta * lam * lam < eta * gamma < 2.0 * lam):
                    sample_keep = theta[: min(64, n_runs)].copy()
            final_var = float(np.var(theta, ddof=1))
            emp_unst = bool(
                final_var > threshold
                or (len(trace) > 5 and trace[-1] > 10.0 * max(trace[1], 1e-12))
                or (len(trace) > 0 and max(trace) > threshold)
            )
            disc_stable = eta * gamma < lam * (2.0 - eta * lam)
            std_stable = eta * gamma < 2.0 * lam
            empirical_unstable[i, j] = float(emp_unst)
            discrete_unstable[i, j] = float(not disc_stable)
            standard_unstable[i, j] = float(not std_stable)
            rows.append({
                "eta": float(eta),
                "gamma": float(gamma),
                "method": "phase",
                "variance": final_var,
                "empirical_unstable": int(emp_unst),
                "discrete_predicts_unstable": int(not disc_stable),
                "standard_predicts_unstable": int(not std_stable),
                "mismatch_region": int((not disc_stable) and std_stable),
            })
            if example is None and (not disc_stable) and std_stable:
                example = (eta, gamma, np.asarray(trace))
    mismatch_mask = (discrete_unstable > 0.5) & (standard_unstable < 0.5)
    empirical_in_mismatch = empirical_unstable[mismatch_mask] if mismatch_mask.any() else np.asarray([])
    write_csv(result_dir / "figure_data.csv", rows)
    extra = {}
    if example is not None:
        extra = {"example_eta": example[0], "example_gamma": example[1], "example_variance_trace": example[2]}
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        etas=etas,
        gammas=gammas,
        empirical_unstable=empirical_unstable,
        discrete_unstable=discrete_unstable,
        standard_unstable=standard_unstable,
        **extra,
    )
    return {
        "lambda": lam,
        "d": d,
        "num_steps": steps,
        "n_runs": n_runs,
        "mismatch_grid_points": int(mismatch_mask.sum()),
        "empirical_unstable_fraction_in_mismatch_region": float(empirical_in_mismatch.mean()) if empirical_in_mismatch.size else float("nan"),
        "pass": bool(empirical_in_mismatch.size > 0 and empirical_in_mismatch.mean() > 0.5),
    }


def run_exp29(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Non-stationary drift-induced spreading with eta^2 Gamma mu(t)^2 source."""
    p = config.get("parameters", {})
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    steps = int(p.get("num_steps", p.get("steps", 800)))
    n_runs = int(p.get("n_runs", 30000))
    eta = float(p.get("eta", 0.02))
    lam_flat = float(p.get("lambda_flat", 0.0))
    lam_sharp = float(p.get("lambda_sharp", 1.0))
    gamma_flat = float(p.get("gamma_flat", 0.8))
    gamma_sharp = float(p.get("gamma_sharp", 0.1))
    d = float(p.get("d", 0.02))
    gbar = float(p.get("gbar_flat", 0.5))
    y = np.zeros((n_runs, 2), dtype=np.float64)
    lambdas = np.array([lam_flat, lam_sharp])
    gammas = np.array([gamma_flat, gamma_sharp])
    g_mean = np.array([gbar, 0.0])
    rows, times, mu_hist, var_hist, pred_diff, pred_drift = [], [], [], [], [], []
    mu = np.zeros(2)
    diff_accum = 0.0
    drift_accum = 0.0
    for step in range(steps + 1):
        if step % max(1, steps // 100) == 0:
            var = np.var(y, axis=0, ddof=1)
            mu_emp = y.mean(axis=0)
            times.append(step)
            mu_hist.append(mu_emp)
            var_hist.append(var)
            pred_diff.append(diff_accum)
            pred_drift.append(drift_accum)
            rows += [
                {"step": step, "method": "flat_variance", "variance": float(var[0])},
                {"step": step, "method": "sharp_variance", "variance": float(var[1])},
                {"step": step, "method": "flat_mean_displacement_abs", "variance": float(abs(mu_emp[0]))},
                {"step": step, "method": "pure_diffusion_prediction_flat", "variance": float(diff_accum)},
                {"step": step, "method": "drift_induced_prediction_flat", "variance": float(diff_accum + drift_accum)},
            ]
        if step == steps:
            break
        h_noise = np.sqrt(gammas)[None, :] * rng.normal(size=(n_runs, 2))
        g_noise = np.sqrt(d) * rng.normal(size=(n_runs, 2))
        h_sample = lambdas[None, :] + h_noise
        g_sample = g_mean[None, :] + g_noise
        y = (1.0 - eta * h_sample) * y - eta * g_sample
        diff_accum += eta**2 * d
        drift_accum += eta**2 * gamma_flat * float(mu[0] ** 2)
        mu = (1.0 - eta * lambdas) * mu - eta * g_mean
    times_arr, mu_arr, var_arr = np.asarray(times), np.asarray(mu_hist), np.asarray(var_hist)
    tail = max(5, len(times_arr) // 2)
    log_t = np.log(np.maximum(times_arr[-tail:], 1.0))
    log_v = np.log(np.maximum(var_arr[-tail:, 0], 1e-18))
    power = float(np.polyfit(log_t, log_v, 1)[0])
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(
        result_dir / "raw_outputs.npz",
        times=times_arr,
        mean=mu_arr,
        variance=var_arr,
        pure_diffusion_prediction_flat=np.asarray(pred_diff),
        drift_induced_prediction_flat=np.asarray(pred_drift),
    )
    return {
        "num_steps": steps,
        "n_runs": n_runs,
        "eta": eta,
        "flat_variance_power_law_tail": power,
        "final_flat_variance": float(var_arr[-1, 0]),
        "final_pure_diffusion_prediction_flat": float(pred_diff[-1]),
        "final_drift_induced_prediction_flat": float(pred_diff[-1] + pred_drift[-1]),
        "drift_over_diffusion_prediction_final": float(pred_drift[-1] / max(pred_diff[-1], 1e-18)),
        "pass": bool(power > 1.5 and pred_drift[-1] > pred_diff[-1]),
    }


def _real_mlp_context(config: dict[str, Any], seed: int, batch_size: int):
    device = _device(config.get("device", "cpu"))
    cfg = copy.deepcopy(config)
    cfg.setdefault("dataset", {"name": "mnist", "sample_size": 512, "val_size": 256, "subset": "first", "normalize": False})
    cfg.setdefault("training", {"batch_size": batch_size, "replacement": True, "lr": 0.1, "reference_steps": 20})
    cfg["training"]["batch_size"] = batch_size
    model, loaders = _train_reference_mlp(cfg, device, seed)
    train_ds, _, _, _, _ = loaders
    x_train, y_train = _dataset_to_tensors(train_ds, device)
    loss_fn = nn.CrossEntropyLoss()
    w0 = parameters_to_vector(model.parameters()).detach().cpu().numpy().astype(np.float64)
    return device, model, train_ds, x_train, y_train, loss_fn, w0


def _grad_at_vector_np(model: nn.Module, vec: np.ndarray, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module, device: torch.device) -> np.ndarray:
    vt = torch.as_tensor(vec, dtype=next(model.parameters()).dtype, device=device)
    vector_to_parameters(vt, model.parameters())
    return _grad_vector(model, x, y, loss_fn).detach().cpu().numpy().astype(np.float64)


def _sample_batch_tensors(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    rng: np.random.Generator,
    replacement: bool,
    cursor_state: dict[str, Any] | None = None,
):
    n = int(y_train.numel())
    if replacement:
        idx = rng.integers(0, n, size=batch_size)
    else:
        if cursor_state is None:
            cursor_state = {}
        if "perm" not in cursor_state or cursor_state.get("pos", 0) + batch_size > n:
            cursor_state["perm"] = rng.permutation(n)
            cursor_state["pos"] = 0
        pos = int(cursor_state["pos"])
        idx = cursor_state["perm"][pos : pos + batch_size]
        cursor_state["pos"] = pos + batch_size
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=x_train.device)
    return x_train[idx_t], y_train[idx_t]


def _run_real_mlp_ensemble(
    config: dict[str, Any],
    *,
    seed: int,
    n_runs: int,
    steps: int,
    lr: float,
    batch_size: int,
    replacement: bool = True,
    initial_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    device, model, _, x_train, y_train, loss_fn, w0 = _real_mlp_context(config, seed, batch_size)
    dim = w0.size
    vectors = initial_vectors.copy() if initial_vectors is not None else np.repeat(w0[None, :], n_runs, axis=0)
    traces = np.zeros((steps + 1, n_runs, dim), dtype=np.float32)
    cursors = [{} for _ in range(n_runs)]
    for step in range(steps + 1):
        traces[step] = vectors.astype(np.float32)
        if step == steps:
            break
        for r in range(n_runs):
            xb, yb = _sample_batch_tensors(x_train, y_train, batch_size, rng, replacement, cursors[r])
            grad = _grad_at_vector_np(model, vectors[r], xb, yb, loss_fn, device)
            vectors[r] = vectors[r] - lr * grad
    return traces, w0, x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy()


def run_exp33(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Hessian eigenbasis decoupling on real MLP ensemble trajectories."""
    p = config.get("parameters", {})
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    batch_size = int(p.get("batch_size", 64))
    n_runs = int(p.get("n_runs", 16))
    steps = int(p.get("num_steps", p.get("steps", 50)))
    lr = float(p.get("eta", 0.03))
    k_subspace = int(p.get("n_directions", 12))
    device, model, _, x_train, y_train, loss_fn, w0 = _real_mlp_context(config, seed, batch_size)
    h = _hessian_full_batch(model, x_train, y_train, loss_fn).detach().cpu().numpy().astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(h)
    traces, *_ = _run_real_mlp_ensemble(config, seed=seed + 2000, n_runs=n_runs, steps=steps, lr=lr, batch_size=batch_size, replacement=True)
    rows, times, ratios = [], [], []
    sel = np.concatenate([np.argsort(np.abs(eigvals))[: k_subspace // 2], np.argsort(eigvals)[-(k_subspace - k_subspace // 2) :]])
    cov_orig_final = cov_eig_final = None
    for step in range(steps + 1):
        disp = traces[step].astype(np.float64) - w0[None, :]
        proj = disp @ eigvecs[:, sel]
        cov_eig = _covariance_np(proj)
        cov_orig = _covariance_np(disp[:, : len(sel)])
        diag_mass = float(np.sum(np.diag(cov_eig) ** 2))
        offdiag_mass = float(np.sum(cov_eig**2) - diag_mass)
        ratio = offdiag_mass / max(diag_mass, 1e-18)
        rows.append({"step": step, "method": "offdiag_over_diag_hessian_basis", "variance": ratio})
        times.append(step); ratios.append(ratio); cov_orig_final, cov_eig_final = cov_orig, cov_eig
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", times=np.asarray(times), offdiag_over_diag=np.asarray(ratios), hessian=h, eigenvalues=eigvals, selected=sel, covariance_original_final=cov_orig_final, covariance_eigenbasis_final=cov_eig_final)
    return {
        "model": "MLP-386",
        "dim": int(w0.size),
        "n_runs": n_runs,
        "num_steps": steps,
        "final_offdiag_over_diag": float(ratios[-1]),
        "diagonal_mass_fraction_final": float(1.0 / (1.0 + ratios[-1])),
        "n_directions": int(len(sel)),
        "pass": bool(ratios[-1] < float(p.get("ratio_threshold", 2.0))),
    }


def run_exp35(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """With vs without replacement on real MLP trajectories."""
    p = config.get("parameters", {})
    seed = int(config.get("seed", 42))
    batch_size = int(p.get("batch_size", 64))
    n_runs = int(p.get("n_runs", 12))
    steps = int(p.get("num_steps", p.get("steps", 80)))
    lr = float(p.get("eta", 0.03))
    wr, w0, *_ = _run_real_mlp_ensemble(config, seed=seed + 4000, n_runs=n_runs, steps=steps, lr=lr, batch_size=batch_size, replacement=True)
    wor, *_ = _run_real_mlp_ensemble(config, seed=seed + 4000, n_runs=n_runs, steps=steps, lr=lr, batch_size=batch_size, replacement=False, initial_vectors=np.repeat(w0[None, :], n_runs, axis=0))
    rows, tr_wr, tr_wor = [], [], []
    for step in range(steps + 1):
        c_wr = _covariance_np(wr[step].astype(np.float64) - w0[None, :])
        c_wor = _covariance_np(wor[step].astype(np.float64) - w0[None, :])
        tr1, tr2 = float(np.trace(c_wr)), float(np.trace(c_wor))
        tr_wr.append(tr1); tr_wor.append(tr2)
        rows += [{"step": step, "method": "with_replacement", "variance": tr1}, {"step": step, "method": "without_replacement", "variance": tr2}]
    write_csv(result_dir / "figure_data.csv", rows)
    np.savez_compressed(result_dir / "raw_outputs.npz", trace_with_replacement=np.asarray(tr_wr), trace_without_replacement=np.asarray(tr_wor))
    return {
        "model": "MLP-386",
        "n_runs": n_runs,
        "num_steps": steps,
        "final_trace_with_replacement": float(tr_wr[-1]),
        "final_trace_without_replacement": float(tr_wor[-1]),
        "without_over_with_trace_final": float(tr_wor[-1] / max(tr_wr[-1], 1e-18)),
        "pass": bool(tr_wor[-1] < tr_wr[-1]),
    }


def run_exp36(config: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    """Learning-rate scaling on real MLP via actual minibatch-gradient increments."""
    return run_exp18(config, result_dir)


from .experiments_corrected_langevin import (
    run_exp37,
    run_exp38,
    run_exp39,
    run_exp40,
    run_exp41,
    run_exp42,
    run_exp43,
    run_exp44,
)


RUNNERS = {
    "EXP1A": run_exp1a,
    "EXP1B": run_exp1b,
    "EXP2": run_exp2,
    "EXP3": run_exp3,
    "EXP4": run_exp4,
    "EXP5": run_exp5,
    "EXP6": run_exp6,
    "EXP7": run_exp7,
    "EXP8": run_exp8,
    "EXP9": run_exp9,
    "EXP10": run_exp10,
    "EXP11": run_exp11,
    "EXP12": run_exp12,
    "EXP13": run_exp13,
    "EXP14": run_exp14,
    "EXP15": run_exp15,
    "EXP16": run_exp16,
    "EXP17": run_exp17,
    "EXP18": run_exp18,
    "EXP20": run_exp20,
    "EXP21": run_exp21,
    "EXP23": run_exp23,
    "EXP27": run_exp27,
    "EXP29": run_exp29,
    "EXP33": run_exp33,
    "EXP35": run_exp35,
    "EXP36": run_exp36,
    "EXP37": run_exp37,
    "EXP38": run_exp38,
    "EXP39": run_exp39,
    "EXP40": run_exp40,
    "EXP41": run_exp41,
    "EXP42": run_exp42,
    "EXP43": run_exp43,
    "EXP44": run_exp44,
}
