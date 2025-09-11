#!/usr/bin/env python3
import os, sys, argparse, random, numpy as np, gc, json
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP, FlexibleCNN
from src.utils import MNIST, load_similar_mnist_data
from src.utils import load_saved_data


def get_device(force_auto: bool = True) -> torch.device:
    if force_auto:
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, criterion, device):
    model.eval()
    vloss, correct, total = 0.0, 0, 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        vloss += criterion(out, target).item()
        pred = out.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return vloss / max(1, len(val_loader)), correct / total


def clear_device_caches():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', choices=['mnist', 'mnist_similar'], required=True)
    parser.add_argument('--model', choices=['mlp', 'cnn'], required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lrs', default='0.1,0.01')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--checkpoint_in', required=True)
    parser.add_argument('--results_dir', default='src/scripts/exp_results')
    parser.add_argument('--sample_size', type=int, default=6400)
    parser.add_argument('--auto_device', action='store_true')
    parser.add_argument("--data_loader", default='default')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = args.device
    lrs = [float(x) for x in args.lrs.split(',')]

    if args.dataset_train == 'mnist': 
        train_path = 'src/data/mnist_train.pt'
        test_path = 'src/data/mnist_test.pt'
    else:
        Exception('пока не надо similar data')
    load_data_fn = load_saved_data
    replacement = args.data_loader == 'replacement'
    train_ds, val_ds, train_loader, val_loader = load_data_fn(
        train_path=train_path, test_path=test_path, batch_size=args.batch_size, replacement=replacement
    )

    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.results_dir, exist_ok=True)

    # шаблон
    model_template = create_model(args.model).to(device)
    model_template.load_state_dict(torch.load(os.path.join(args.results_dir, args.checkpoint_in),
                                              map_location=device))
    n_params = params_vector(model_template).numel()

    for lr in lrs:
        print(f"[info] lr={lr} | n_samples={args.n_samples} | steps={args.steps} | n_params={n_params}")

        weights_trajs = torch.empty((args.n_samples, args.steps, n_params), dtype=torch.float32, device='cpu')
        logs = {"train_losses": [], "val_losses": [], "val_accs": []}

        for si in tqdm(range(args.n_samples), desc=f"samples (lr={lr})"):
            torch.manual_seed(args.seed + si)

            model = create_model(args.model).to(device)
            model.load_state_dict(model_template.state_dict())
            optimizer = optim.SGD(model.parameters(), lr=lr)

            data_iter = iter(train_loader)
            step_bar = tqdm(range(args.steps), desc=f"steps s={si}", leave=False)
            for step in step_bar:
                try:
                    data, target = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    data, target = next(data_iter)

                data, target = data.to(device), target.to(device)

                v = params_vector(model)
                weights_trajs[si, step].copy_(v)

                optimizer.zero_grad(set_to_none=True)
                out = model(data)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                logs["train_losses"].append(loss.item())

                # раз в эпоху (проход train_loader) делаем валидацию
                if (step + 1) % len(train_loader) == 0:
                    vloss, vacc = evaluate(model, val_loader, criterion, device)
                    logs["val_losses"].append(vloss)
                    logs["val_accs"].append(vacc)

            del model, optimizer
            clear_device_caches()

        # сохраняем веса и логи
        out_path = os.path.join(args.results_dir, f"sgd_weights_lr{lr}.pt")
        torch.save(weights_trajs, out_path)
        print(f"[saved] {out_path} shape={tuple(weights_trajs.shape)}")

        logs_path = os.path.join(args.results_dir, f"sgd_logs_lr{lr}.json")
        with open(logs_path, "w") as f:
            json.dump(logs, f)
        print(f"[saved] {logs_path}")

        del weights_trajs, logs
        gc.collect()
        clear_device_caches()

    print("[done] all lrs processed")


if __name__ == '__main__':
    main()
