#!/usr/bin/env python3
import os, sys, argparse, random, numpy as np, gc, json
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP, FlexibleCNN
from src.utils import MNIST, load_similar_mnist_data


# -------------------- утилиты --------------------
def get_device(force_auto: bool = False) -> torch.device:
    if force_auto:
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def clear_device_caches():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

def create_model(model_type):
    if model_type == 'mlp':
        return FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6)
    else:
        return FlexibleCNN(in_channels=1, conv_channels=[12], conv_kernels=[3], conv_strides=[1],
                           conv_use_relu_list=[True], conv_dropouts=[0.0], conv_use_bn=True,
                           pool_after=[False], gap_size=1, mlp_hidden_dims=[11],
                           mlp_use_relu_list=[True], mlp_dropouts=[0.0], output_dim=10)

@torch.no_grad()
def params_vector(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().float()

def disable_dropout_in_train(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.p = 0.0

def load_data(dataset_train: str, batch_size: int, sample_size: int | None, seed: int):
    if dataset_train == "mnist":
        train_dataset, test_dataset, _, _ = MNIST(batch_size=batch_size, sample_size=sample_size)
    else:
        train_dataset, test_dataset, _, _ = load_similar_mnist_data(
            batch_size=batch_size, sample_size=sample_size or 1000
        )

    # GD = полный батч
    train_loader_full = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker
    )

    return train_dataset, test_dataset, train_loader_full, val_loader


# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', choices=['mnist', 'mnist_similar'], required=True)
    parser.add_argument('--model', choices=['mlp', 'cnn'], required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lrs', default='0.1,0.01')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--checkpoint_in', required=True)
    parser.add_argument('--results_dir', default='src/scripts/exp_results')
    parser.add_argument('--sample_size', type=int, default=6400)
    parser.add_argument('--auto_device', action='store_true')
    args = parser.parse_args()

    device = get_device(args.auto_device)
    lrs = [float(x) for x in args.lrs.split(',')]

    train_ds, val_ds, train_loader_full, val_loader = load_data(
        dataset_train=args.dataset_train,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        seed=args.seed
    )
    criterion = nn.CrossEntropyLoss()
    os.makedirs(args.results_dir, exist_ok=True)

    ckpt_path = os.path.join(args.results_dir, args.checkpoint_in) \
        if not os.path.isabs(args.checkpoint_in) else args.checkpoint_in

    model_template = create_model(args.model).to(device)
    model_template.load_state_dict(torch.load(ckpt_path, map_location=device))
    n_params = params_vector(model_template).numel()

    for lr in lrs:
        print(f"[info] lr={lr} | steps={args.steps} | n_params={n_params} "
              f"| train_size={len(train_ds)} | val_size={len(val_ds)}")

        weights_trajs = torch.empty((args.steps, n_params), dtype=torch.float32, device='cpu')
        logs = {"train_losses": [], "val_losses": [], "val_accs": []}

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        model = create_model(args.model).to(device)
        model.load_state_dict(model_template.state_dict())
        disable_dropout_in_train(model)
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # --- подготовка полного батча один раз ---
        full_data, full_target = next(iter(train_loader_full))
        full_data, full_target = full_data.to(device), full_target.to(device)

        step_bar = tqdm(range(args.steps), desc=f"GD training (lr={lr})")
        for step in step_bar:
            # логируем ПЕРЕД шагом
            v = params_vector(model)
            weights_trajs[step].copy_(v)

            # один шаг GD (полный батч фиксированный)
            optimizer.zero_grad(set_to_none=True)
            out = model(full_data)
            loss = criterion(out, full_target)
            loss.backward()
            optimizer.step()
            logs["train_losses"].append(loss.item())

            # валидация
            model.eval()
            vloss, correct = 0.0, 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    out = model(data)
                    vloss += criterion(out, target).item()
                    pred = out.argmax(dim=1)
                    correct += (pred == target).sum().item()
            logs["val_losses"].append(vloss / max(1, len(val_loader)))
            logs["val_accs"].append(correct / len(val_loader.dataset))
            model.train()

            if step % 10 == 0 or step == args.steps - 1:
                step_bar.set_postfix(train_loss=float(loss.detach().cpu()),
                                     val_loss=logs["val_losses"][-1],
                                     val_acc=logs["val_accs"][-1])


        # сохраняем траекторию весов и логи
        out_path = os.path.join(args.results_dir, f"gd_weights_lr{lr}.pt")
        torch.save(weights_trajs, out_path)
        print(f"[saved] {out_path} shape={tuple(weights_trajs.shape)}")

        logs_path = os.path.join(args.results_dir, f"gd_logs_lr{lr}.json")
        with open(logs_path, "w") as f:
            json.dump(logs, f)
        print(f"[saved] {logs_path}")

        del model, optimizer, weights_trajs
        gc.collect()
        clear_device_caches()

    print("[done] all lrs processed")


if __name__ == '__main__':
    main()
