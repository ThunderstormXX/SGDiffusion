#!/usr/bin/env python3
import os, sys, argparse, random, numpy as np, gc
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP, FlexibleCNN
from src.utils import MNIST, load_similar_mnist_data

def get_device(force_auto: bool = False) -> torch.device:
    if force_auto:
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_train_loader(dataset_train: str, batch_size: int, sample_size: int, seed: int):
    if dataset_train == "mnist":
        train_dataset, _, _, _ = MNIST(batch_size=batch_size, sample_size=sample_size)
    else:
        train_dataset, _, _, _ = load_similar_mnist_data(batch_size=batch_size, sample_size=sample_size)

    N = len(train_dataset)
    eff = (N // batch_size) * batch_size  # кратно batch_size → все батчи ровно batch_size
    g = torch.Generator().manual_seed(seed)

    sampler = RandomSampler(train_dataset, replacement=True, num_samples=eff, generator=g)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    return train_loader, N

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
    # CPU float32
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().float()

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
    parser.add_argument('--sample_size', type=int, default=6000)
    parser.add_argument('--auto_device', action='store_true')
    args = parser.parse_args()

    device = get_device(args.auto_device)
    lrs = [float(x) for x in args.lrs.split(',')]

    train_loader, train_size = load_train_loader(
        args.dataset_train, args.batch_size, args.sample_size, args.seed
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.results_dir, exist_ok=True)

    # шаблон + размерность вектора параметров
    model_template = create_model(args.model).to(device)
    model_template.load_state_dict(torch.load(os.path.join(args.results_dir, args.checkpoint_in),
                                              map_location=device))
    n_params = params_vector(model_template).numel()

    for lr in lrs:
        print(f"[info] lr={lr} | n_samples={args.n_samples} | steps={args.steps} | n_params={n_params} | train_size={train_size}")

        # БОЛЬШОЙ тензор на CPU под все траектории для этого lr
        weights_trajs = torch.empty((args.n_samples, args.steps, n_params), dtype=torch.float32, device='cpu')

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

                # логируем ПЕРЕД шагом оптимизации
                v = params_vector(model)                    # [n_params] CPU float32
                weights_trajs[si, step].copy_(v)            # запись в общий тензор

                optimizer.zero_grad(set_to_none=True)
                out = model(data)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

            # небольшая разгрузка девайса между сэмплами
            del model, optimizer
            clear_device_caches()

        # сохраняем один большой .pt для этого lr и очищаем память
        out_path = os.path.join(args.results_dir, f"weights_lr{lr}.pt")
        torch.save(weights_trajs, out_path)
        print(f"[saved] {out_path} shape={tuple(weights_trajs.shape)}")

        del weights_trajs
        gc.collect()
        clear_device_caches()

    print("[done] all lrs processed")

if __name__ == '__main__':
    main()
