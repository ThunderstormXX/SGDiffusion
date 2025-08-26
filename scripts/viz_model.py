import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def _human(n: int) -> str:
    """Красивый формат числа параметров."""
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n}{unit}"
        n //= 1000
    return f"{n}T"

def _count_params(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def print_model_tree(model: nn.Module, show_buffers: bool = False) -> None:
    """Печатает древо модулей с количеством параметров."""
    def recurse(module: nn.Module, prefix: str = "", is_last: bool = True, name: str = ""):
        total, trainable = _count_params(module)
        connector = "└─" if is_last else "├─"
        cls = module.__class__.__name__
        line = f"{prefix}{connector} {name} ({cls})  params={_human(total)} / train={_human(trainable)}"
        print(line)

        children = list(module.named_children())
        n = len(children)
        for i, (child_name, child) in enumerate(children):
            new_prefix = prefix + ("   " if is_last else "│  ")
            recurse(child, new_prefix, i == n - 1, child_name)

    root_total, root_train = _count_params(model)
    print(f"{model.__class__.__name__}  [params={_human(root_total)} / train={_human(root_train)}]")
    for i, (name, child) in enumerate(model.named_children()):
        recurse(child, "", i == len(list(model.named_children())) - 1, name)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _save_hist(t: torch.Tensor, title: str, out: str):
    arr = t.detach().float().cpu().view(-1).numpy()
    plt.figure()
    plt.hist(arr, bins=80)
    plt.title(title)
    plt.xlabel("value"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def _save_heatmap_2d(t: torch.Tensor, title: str, out: str):
    """Для 2D-тензоров: показываем как теплокарту (матрица весов Linear)."""
    W = t.detach().float().cpu().numpy()
    max_side = 512
    h, w = W.shape
    sh, sw = h, w
    if h > max_side:
        idx = np.linspace(0, h - 1, max_side).astype(int)
        W = W[idx, :]
        sh = max_side
    if w > max_side:
        idx = np.linspace(0, w - 1, max_side).astype(int)
        W = W[:, idx]
        sw = max_side

    plt.figure()
    plt.imshow(W, aspect='auto', interpolation='nearest')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"{title}  ({h}x{w}→{sh}x{sw})")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def visualize_weights_per_module(model: nn.Module, out_dir: str = "weights_viz") -> Dict[str, Dict[str, str]]:
    """Сохраняет изображения весов для каждого подмодуля."""
    _ensure_dir(out_dir)
    artifacts: Dict[str, Dict[str, str]] = {}

    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            tag = full_name or "root.Linear"
            artifacts.setdefault(tag, {})
            
            # heatmap для матрицы весов
            heatmap_path = os.path.join(out_dir, f"{tag.replace('.', '_')}_W_heatmap.png")
            _save_heatmap_2d(module.weight, f"{tag}.weight", heatmap_path)
            artifacts[tag]["heatmap"] = heatmap_path

            # гистограммы весов/смещений
            hist_w = os.path.join(out_dir, f"{tag.replace('.', '_')}_W_hist.png")
            _save_hist(module.weight, f"{tag}.weight", hist_w)
            artifacts[tag]["hist_weight"] = hist_w

            if module.bias is not None:
                hist_b = os.path.join(out_dir, f"{tag.replace('.', '_')}_b_hist.png")
                _save_hist(module.bias, f"{tag}.bias", hist_b)
                artifacts[tag]["hist_bias"] = hist_b

    return artifacts

def describe_and_visualize(model: nn.Module, out_dir: str = "weights_viz", show_buffers: bool = False) -> Dict[str, Dict[str, str]]:
    """
    1) Печатает структуру модели (древо + параметры).
    2) Сохраняет картинки весов/гистограмм по модулям.
    3) Возвращает словарь путей к артефактам.
    """
    print_model_tree(model, show_buffers=show_buffers)
    arts = visualize_weights_per_module(model, out_dir=out_dir)
    print(f"\nSaved weight visualizations to: {os.path.abspath(out_dir)}")
    return arts

def set_model_params_from_vector(model: nn.Module, param_vector: np.ndarray):
    """Устанавливает параметры модели из вектора."""
    param_vector = torch.from_numpy(param_vector).float()
    idx = 0
    
    for param in model.parameters():
        param_size = param.numel()
        param.data = param_vector[idx:idx + param_size].view(param.shape)
        idx += param_size
    
    if idx != len(param_vector):
        raise ValueError(f"Размер вектора {len(param_vector)} не соответствует количеству параметров модели {idx}")