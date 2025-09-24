#!/usr/bin/env python3
import os, sys, argparse, json, gc
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
from src.utils import load_saved_data


# ---- device ----
def get_device(auto: bool, device: str):
    if auto:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device(device)


def clear_device_caches():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# ---- models ----
def create_model(model_type):
    if model_type == "mlp":
        return FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6)
    else:
        return FlexibleCNN(
            in_channels=1, conv_channels=[12], conv_kernels=[3], conv_strides=[1],
            conv_use_relu_list=[True], conv_dropouts=[0.0], conv_use_bn=True,
            pool_after=[False], gap_size=1, mlp_hidden_dims=[11],
            mlp_use_relu_list=[True], mlp_dropouts=[0.0], output_dim=10
        )


@torch.no_grad()
def vector_to_model_(model: nn.Module, vec: torch.Tensor):
    vec = vec.to(next(model.parameters()).device, non_blocking=True)
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
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



def save_hist(values, vline, title, xlabel, out_png):
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=40)
    plt.axvline(vline, linestyle="--", linewidth=2, color="red")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def process_iteration(trajs, step_idx, model, val_loader, criterion, device, base, out_dir, tag):
    n_samples = trajs.shape[0]
    csv_path = os.path.join(out_dir, f"{base}__{tag}_val_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample_idx,val_loss,val_acc\n")

    losses, accs = [], []
    pbar = tqdm(range(n_samples), desc=f"eval {tag} samples", dynamic_ncols=True)
    for si in pbar:
        vec = trajs[si, step_idx]  # [n_params]
        vector_to_model_(model, vec)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        losses.append(val_loss); accs.append(val_acc)
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{si},{val_loss:.8f},{val_acc:.6f}\n")
        pbar.set_postfix(loss=f"{val_loss:.4f}", acc=f"{val_acc:.4f}")

    # усреднённая модель
    mean_vec = trajs[:, step_idx, :].mean(dim=0)
    vector_to_model_(model, mean_vec)
    mean_loss, mean_acc = evaluate(model, val_loader, criterion, device)

    # summary
    losses_t, accs_t = torch.tensor(losses), torch.tensor(accs)
    stats = {
        f"{tag}_val_loss_mean": float(losses_t.mean().item()),
        f"{tag}_val_loss_std": float(losses_t.std(unbiased=False).item()),
        f"{tag}_val_acc_mean": float(accs_t.mean().item()),
        f"{tag}_val_acc_std": float(accs_t.std(unbiased=False).item()),
        f"{tag}_mean_model_val_loss": float(mean_loss),
        f"{tag}_mean_model_val_acc": float(mean_acc),
        f"{tag}_csv": csv_path,
    }

    # гистограммы
    dist_dir = os.path.join(out_dir, "val_loss_distribution")
    os.makedirs(dist_dir, exist_ok=True)
    loss_png = os.path.join(dist_dir, f"{base}__{tag}_loss_hist.png")
    acc_png  = os.path.join(dist_dir, f"{base}__{tag}_acc_hist.png")
    save_hist(losses, mean_loss, f"Val loss distribution ({tag})", "val_loss", loss_png)
    save_hist(accs,  mean_acc,  f"Val acc distribution ({tag})", "val_acc",  acc_png)
    stats[f"{tag}_loss_png"] = loss_png
    stats[f"{tag}_acc_png"]  = acc_png

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_file", required=True, help="путь к sgd_weights_lr*.pt")
    ap.add_argument("--out_dir", required=True, help="директория для CSV/JSON/PNG")
    ap.add_argument("--dataset_train", choices=["mnist", "mnist_similar"], required=True)
    ap.add_argument("--model", choices=["mlp", "cnn"], required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--auto_device", action="store_true")
    ap.add_argument("--data_loader", default="default")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.auto_device, args.device)
    criterion = nn.CrossEntropyLoss()

    # --- траектории ---
    trajs = torch.load(args.weights_file, map_location="cpu")  # [n_samples, steps, n_params]
    if trajs.ndim != 3:
        print(f"[error] Bad shape {tuple(trajs.shape)}"); sys.exit(2)
    n_samples, n_steps, n_params = map(int, trajs.shape)
    print(f"[info] shape = (samples={n_samples}, steps={n_steps}, n_params={n_params})")

    # --- данные ---
    if args.dataset_train == "mnist":
        train_path = "src/data/mnist_train.pt"
        test_path  = "src/data/mnist_test.pt"
    else:
        raise Exception("пока не надо similar data")

    replacement = args.data_loader == "replacement"
    train_ds, val_ds, train_loader, val_loader = load_saved_data(
        train_path=train_path, test_path=test_path,
        batch_size=args.batch_size, replacement=replacement
    )
    print(f"[info] val_size={len(val_ds)}, device={device}")

    # --- модель ---
    model = create_model(args.model).to(device)

    base = os.path.splitext(os.path.basename(args.weights_file))[0]
    summary = {
        "weights_file": args.weights_file,
        "n_samples": n_samples,
        "n_steps": n_steps,
        "n_params": n_params,
    }

    # start (нулевой шаг)
    summary.update(process_iteration(trajs, 0, model, val_loader, criterion, device, base, args.out_dir, "start"))
    # end (последний шаг)
    summary.update(process_iteration(trajs, n_steps - 1, model, val_loader, criterion, device, base, args.out_dir, "end"))

    # summary.json
    json_path = os.path.join(args.out_dir, f"{base}__summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[saved] summary={json_path}")

    del trajs; gc.collect(); clear_device_caches()


if __name__ == "__main__":
    main()
