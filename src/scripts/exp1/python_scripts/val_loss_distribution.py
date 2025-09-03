#!/usr/bin/env python3
import os, sys, argparse, json, gc, glob
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP, FlexibleCNN
from src.utils import MNIST  # при необходимости заменишь на load_similar_mnist_data


# ---- device ----
def get_device():
    return torch.device("cpu")


def clear_device_caches():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try: torch.mps.empty_cache()
        except Exception: pass


# ---- models / io ----
def create_mlp():  # те же гиперы, что в твоём трейнере
    return FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6)

def create_cnn():
    return FlexibleCNN(in_channels=1, conv_channels=[12], conv_kernels=[3], conv_strides=[1],
                       conv_use_relu_list=[True], conv_dropouts=[0.0], conv_use_bn=True,
                       pool_after=[False], gap_size=1, mlp_hidden_dims=[11],
                       mlp_use_relu_list=[True], mlp_dropouts=[0.0], output_dim=10)

@torch.no_grad()
def params_vector(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().float()

@torch.no_grad()
def vector_to_model_(model: nn.Module, vec: torch.Tensor):
    vec = vec.to(next(model.parameters()).device, non_blocking=True)
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


# ---- data ----
def build_val_loader(batch_size=512, sample_size=6000):
    # Берём вал датасет из твоей утилиты MNIST
    _, val_dataset, _, _ = MNIST(batch_size=batch_size, sample_size=sample_size)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0), len(val_dataset)


# ---- eval ----
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    tot_loss, tot_correct, tot_count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        tot_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        tot_correct += (preds == y).sum().item()
        tot_count += x.size(0)
    return tot_loss / max(1, tot_count), tot_correct / max(1, tot_count)


def autodetect_checkpoint(weights_file: str):
    # ищем рядом любые checkpoint*.pt (или init*.pt)
    cand_dir = os.path.dirname(weights_file)
    pats = ["checkpoint*.pt", "init*.pt", "*template*.pt", "*model*.pt"]
    cands = []
    for p in pats:
        cands.extend(glob.glob(os.path.join(cand_dir, p)))
    # отфильтруем сам файл весов
    cands = [c for c in cands if os.path.basename(c) != os.path.basename(weights_file)]
    return sorted(cands)[0] if cands else None


def pick_arch_by_nparams(n_params: int, device: torch.device):
    # выбираем шаблон по числу параметров
    m_mlp = create_mlp().to(device)
    m_cnn = create_cnn().to(device)
    n_mlp = params_vector(m_mlp).numel()
    n_cnn = params_vector(m_cnn).numel()
    if n_params == n_mlp: return "mlp", m_mlp
    if n_params == n_cnn: return "cnn", m_cnn
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_file", required=True, help="путь к weights_lr*.pt")
    ap.add_argument("--out_dir", required=True, help="директория для CSV/JSON/PNG")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    criterion = nn.CrossEntropyLoss()

    # 1) Загружаем тензор траекторий
    wf = args.weights_file
    print(f"[load] {wf}")
    trajs = torch.load(wf, map_location="cpu")  # [n_samples, steps, n_params]
    if trajs.ndim != 3:
        print(f"[error] Ожидалась форма [n_samples, steps, n_params], получено {tuple(trajs.shape)}"); sys.exit(2)
    n_samples, n_steps, n_params = map(int, trajs.shape)
    print(f"[info] shape = (samples={n_samples}, steps={n_steps}, n_params={n_params})")

    # 2) Строим валидационный лоадер
    val_loader, val_size = build_val_loader()
    print(f"[info] val_size={val_size}, device={device}")

    # 3) Выбор архитектуры по размеру вектора и попытка подгрузить checkpoint
    arch, model = pick_arch_by_nparams(n_params, device)
    if arch is None:
        print("[warn] Не удалось однозначно определить архитектуру по числу параметров.")
        print("       Проверь, что гиперпараметры create_mlp/create_cnn совпадают с тренером.")
        sys.exit(3)

    ckpt_path = autodetect_checkpoint(wf)
    if ckpt_path:
        try:
            sd = torch.load(ckpt_path, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            model.load_state_dict(sd, strict=False)  # strict=False на случай лишних ключей
            print(f"[info] checkpoint loaded: {ckpt_path}")
        except Exception as e:
            print(f"[warn] checkpoint load failed: {e}")

    # 4) Берём последний срез траекторий
    end_step = n_steps - 1
    # Для средней модели понадобится средний вектор
    end_vecs_mean = trajs[:, end_step, :].mean(dim=0)  # [n_params], CPU

    # 5) Прогон по всем сэмплам (с tqdm)
    base = os.path.splitext(os.path.basename(wf))[0]
    csv_path = os.path.join(args.out_dir, f"{base}__val_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample_idx,val_loss,val_acc\n")

    losses, accs = [], []
    pbar = tqdm(range(n_samples), desc="eval samples", dynamic_ncols=True)
    for si in pbar:
        vec = trajs[si, end_step]                  # CPU float32 [n_params]
        vector_to_model_(model, vec)               # в модель (на device)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        losses.append(val_loss); accs.append(val_acc)
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{si},{val_loss:.8f},{val_acc:.6f}\n")
        pbar.set_postfix(loss=f"{val_loss:.4f}", acc=f"{val_acc:.4f}")

    # 6) Метрики средней по сэмплам модели
    vector_to_model_(model, end_vecs_mean)
    mean_loss_line, mean_acc_line = evaluate(model, val_loader, device, criterion)

    # 7) Summary JSON
    losses_t = torch.tensor(losses, dtype=torch.float32)
    accs_t   = torch.tensor(accs,   dtype=torch.float32)
    summary = {
        "weights_file": wf,
        "n_samples": n_samples,
        "n_steps": n_steps,
        "end_step_index": end_step,
        "n_params": n_params,
        "val_loss_mean": float(losses_t.mean().item()),
        "val_loss_std":  float(losses_t.std(unbiased=False).item()),
        "val_acc_mean":  float(accs_t.mean().item()),
        "val_acc_std":   float(accs_t.std(unbiased=False).item()),
        "mean_model_val_loss": float(mean_loss_line),
        "mean_model_val_acc":  float(mean_acc_line),
        "csv": csv_path,
    }
    json_path = os.path.join(args.out_dir, f"{base}__summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 8) Гистограммы с вертикальной линией средней модели
    def save_hist(values, vline, title, xlabel, out_png):
        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=40)
        plt.axvline(vline, linestyle="--", linewidth=2)
        plt.title(title); plt.xlabel(xlabel); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    loss_png = os.path.join(args.out_dir, f"{base}__loss_hist.png")
    acc_png  = os.path.join(args.out_dir, f"{base}__acc_hist.png")
    save_hist(losses, mean_loss_line, f"Val loss distribution ({base})", "val_loss", loss_png)
    save_hist(accs,  mean_acc_line,  f"Val accuracy distribution ({base})", "val_acc", acc_png)

    print(f"[saved] csv={csv_path}")
    print(f"[saved] summary={json_path}")
    print(f"[saved] plots: {loss_png}, {acc_png}")

    # 9) cleanup
    del trajs; gc.collect(); clear_device_caches()


if __name__ == "__main__":
    main()
