import random
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Subset
from torchvision import datasets, transforms


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_dataset(name, data_dir, train_size, val_size, test_size, batch_size, seed):
    if name != "mnist":
        raise ValueError(f"unknown dataset: {name}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    train_perm = torch.randperm(len(train_full), generator=generator).tolist()
    test_perm = torch.randperm(len(test_full), generator=generator).tolist()
    train_n = min(train_size, len(train_full))
    val_n = min(val_size, len(train_full) - train_n)
    test_n = min(test_size, len(test_full))
    train_ds = Subset(train_full, train_perm[:train_n])
    val_ds = Subset(train_full, train_perm[train_n:train_n + val_n])
    test_ds = Subset(test_full, test_perm[:test_n])
    gd_loader = DataLoader(
        train_ds,
        batch_size=len(train_ds),
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_ds, gd_loader, val_loader, test_loader


def make_sgd_loader(train_ds, batch_size, iterations, seed):
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(
        train_ds,
        replacement=True,
        num_samples=batch_size * iterations,
        generator=generator,
    )
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
    return DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
