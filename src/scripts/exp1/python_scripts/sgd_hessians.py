#!/usr/bin/env python3
import os, sys, argparse, json, random, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP, FlexibleCNN
from src.utils import MNIST, load_similar_mnist_data
from src.utils import load_data_with_replacement, load_data

def get_device(force_auto: bool = True) -> torch.device:
    if force_auto:
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# reproducible randomness for dataloader workers
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

def compute_hessian(model, data, target, criterion):
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    hessian = torch.zeros(n_params, n_params, device=data.device)

    out = model(data)
    loss = criterion(out, target)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    for i in range(n_params):
        grad2 = torch.autograd.grad(flat_grad[i], params, retain_graph=True)
        hessian[i] = torch.cat([g.reshape(-1) for g in grad2])

    return hessian

def safe_load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def aggregate_step_losses(step_losses, steps_per_epoch: int):
    """
    Превращаем покадровые (по батчам) лоссы в эпохальные:
    - каждые steps_per_epoch шагов -> среднее = одна точка
    - остаток (< steps_per_epoch) -> если есть, усредняем и добавляем одну точку
    """
    if steps_per_epoch <= 0:
        return [float(sum(step_losses)/max(1, len(step_losses)))] if step_losses else []

    agg = []
    n = len(step_losses)
    full_epochs = n // steps_per_epoch
    for e in range(full_epochs):
        chunk = step_losses[e*steps_per_epoch:(e+1)*steps_per_epoch]
        agg.append(float(sum(chunk) / len(chunk)))
    rem = n % steps_per_epoch
    if rem > 0:
        chunk = step_losses[-rem:]
        agg.append(float(sum(chunk) / len(chunk)))
    return agg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', choices=['mnist', 'mnist_similar'], required=True)
    parser.add_argument('--dataset_val', default='mnist')  # не используется
    parser.add_argument('--model', choices=['mlp', 'cnn'], required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lrs', default='0.1,0.01')
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--checkpoint_in', required=True)
    parser.add_argument('--results_dir', default='src/scripts/exp_results')
    parser.add_argument('--sample_size', type=int, default=6400)
    parser.add_argument('--auto_device', action='store_true')
    parser.add_argument("--data_loader", default='default')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = args.device
    lrs = [float(x) for x in args.lrs.split(',')]
    load_data_fn = (
        load_data_with_replacement if args.data_loader == 'replacement'
        else load_data if args.data_loader == 'default'
        else None
    )
    train_dataset, test_dataset, train_loader, val_loader = load_data_fn(  args.dataset_train, args.batch_size, args.sample_size, args.seed)
    train_size = len(train_dataset)

    steps_per_epoch = max(1, len(train_loader))  # сколько батчей в «эпохе»
    criterion = nn.CrossEntropyLoss()

    # логи SGD/GD по эпохам (могут отсутствовать — тогда пустые)
    sgd_logs = safe_load_json(os.path.join(args.results_dir, "logs_sgd.json"),
                              {"train_losses": [], "val_losses": [], "val_accs": []})
    gd_logs  = safe_load_json(os.path.join(args.results_dir, "logs_gd.json"),
                              {"train_losses": [], "val_losses": [], "val_accs": []})

    for lr in lrs:
        torch.manual_seed(args.seed)
        model = create_model(args.model).to(device)
        model.load_state_dict(torch.load(os.path.join(args.results_dir, args.checkpoint_in), map_location=device))
        optimizer = optim.SGD(model.parameters(), lr=lr)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Hessian SGD: lr={lr}, steps={args.steps}, batch_size={args.batch_size}, "
              f"params={n_params}, train_size={train_size}, steps_per_epoch={steps_per_epoch}")

        # Траектории (как раньше)
        weights_traj = torch.zeros(args.steps, n_params, device=device)
        hessians_traj = torch.zeros(args.steps, n_params, n_params, device=device)

        # Батчевые лоссы текущего прогона
        step_losses = []

        data_iter = iter(train_loader)
        for step in tqdm(range(args.steps), desc=f"steps (lr={lr})"):
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                data, target = next(data_iter)

            data, target = data.to(device), target.to(device)

            params_vec = torch.cat([p.reshape(-1) for p in model.parameters()])
            weights_traj[step] = params_vec

            hessian = compute_hessian(model, data, target, criterion)
            hessians_traj[step] = hessian

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            step_losses.append(float(loss.detach().cpu()))

        os.makedirs(args.results_dir, exist_ok=True)
        torch.save(weights_traj.cpu(), os.path.join(args.results_dir, f"weights_traj_lr{lr}.pt"))
        torch.save(hessians_traj.cpu(), os.path.join(args.results_dir, f"hessians_traj_lr{lr}.pt"))

        # --- Агрегируем батчевые лоссы в эпохальные точки ---
        aggregated_epoch_losses = aggregate_step_losses(step_losses, steps_per_epoch)

        # --- Объединяем с SGD/GD по эпохам ---
        combined_train = (sgd_logs.get("train_losses", []) or []) + \
                         (gd_logs.get("train_losses", []) or []) + \
                         aggregated_epoch_losses

        # JSON с подробностями
        merged_json_path = os.path.join(args.results_dir, f"logs_sgd_gd_plus_sgdEpochified_lr{lr}.json")
        with open(merged_json_path, "w") as f:
            json.dump({
                "steps_per_epoch": steps_per_epoch,
                "total_steps": args.steps,
                "sgd_train_losses": sgd_logs.get("train_losses", []),
                "gd_train_losses": gd_logs.get("train_losses", []),
                "current_sgd_step_losses": step_losses,             # сырой батч-уровень
                "current_sgd_epochified_losses": aggregated_epoch_losses,  # усреднённые по эпохам
                "combined_train_losses": combined_train
            }, f)

        # График: точки по эпохам для SGD, GD и текущего прогона
        plt.figure(figsize=(12, 4))
        plt.plot(combined_train, label="Train loss: SGD(epochs) + GD(epochs) + SGD@lr (epochified)")
        n_sgd = len(sgd_logs.get("train_losses", []))
        n_gd  = len(gd_logs.get("train_losses", []))
        if n_sgd:
            plt.axvline(n_sgd, color='red', linestyle='--', alpha=0.6, label="end SGD(epochs)")
        if n_gd:
            plt.axvline(n_sgd + n_gd, color='green', linestyle='--', alpha=0.6, label="end GD(epochs)")
        plt.title(f"Combined epoch-level train losses (lr={lr})")
        plt.xlabel("epochs (SGD, GD), then epochified-steps for current SGD")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(args.results_dir, f"loss_sgd_gd_plus_sgdEpochified_lr{lr}.png")
        plt.savefig(out_png)
        plt.close()

if __name__ == '__main__':
    main()
