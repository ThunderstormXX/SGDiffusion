#!/usr/bin/env python3
import os, sys, argparse, random
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import MNIST, load_similar_mnist_data


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def dataset_to_tensors(dataset):
    """Конвертируем Dataset -> (X, y) тензоры"""
    X = torch.stack([dataset[i][0] for i in range(len(dataset))])
    y = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_train", choices=["mnist", "mnist_similar"], required=True)
    parser.add_argument("--dataset_val", default="mnist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_size", type=int, default=6400)
    parser.add_argument("--results_dir", default="src/data")
    args = parser.parse_args()

    seed_all(args.seed)

    # Загружаем train
    if args.dataset_train == "mnist":
        train_ds, _, _, _ = MNIST(batch_size=args.sample_size, sample_size=args.sample_size)
    elif args.dataset_train == "mnist_similar":
        train_ds, _, _, _ = load_similar_mnist_data(batch_size=args.sample_size, sample_size=args.sample_size)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_train}")

    # Загружаем test (обычный MNIST)
    _, test_ds, _, _ = MNIST(batch_size=args.sample_size, sample_size=args.sample_size)

    os.makedirs(args.results_dir, exist_ok=True)

    # Конвертируем в (X, y)
    train_tensors = dataset_to_tensors(train_ds)
    test_tensors = dataset_to_tensors(test_ds)

    train_file = os.path.join(args.results_dir, f"{args.dataset_train}_train.pt")
    test_file = os.path.join(args.results_dir, f"{args.dataset_val}_test.pt")

    print(f"Saving train dataset tensors to {train_file}")
    torch.save(train_tensors, train_file)

    print(f"Saving test dataset tensors to {test_file}")
    torch.save(test_tensors, test_file)


if __name__ == "__main__":
    main()
