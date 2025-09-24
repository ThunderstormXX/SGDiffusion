#!/usr/bin/env python3
import os, sys, argparse, json, random, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import NanoGPT
from src.utils import load_shakespeare_data


def get_device(force_auto: bool = True) -> torch.device:
    if force_auto:
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




def create_model(meta_path):
    """Create NanoGPT model based on metadata"""
    meta = torch.load(meta_path)
    model = NanoGPT(
        vocab_size=meta['vocab_size'],
        n_embd=8,
        n_head=1,
        n_layer=1,
        block_size=meta['block_size'],
        mlp_ratio=1
    )
    return model


def compute_hessian(model, data, target):
    """
    Compute full Hessian matrix like in exp1
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    hessian = torch.zeros(n_params, n_params, device=data.device)

    logits, loss = model(data, target)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    for i in range(n_params):
        grad2 = torch.autograd.grad(flat_grad[i], params, retain_graph=True)
        hessian[i] = torch.cat([g.reshape(-1) for g in grad2])

    return hessian


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', default='shakespeare')
    parser.add_argument('--dataset_val', default='shakespeare')
    parser.add_argument('--model', default='nanogpt')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lrs', default='0.1,0.01')
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--checkpoint_in', required=True)
    parser.add_argument('--results_dir', default='src/scripts/exp2/exp_results')
    parser.add_argument('--auto_device', action='store_true')
    parser.add_argument("--data_loader", default='default')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = args.device
    lrs = [float(x) for x in args.lrs.split(',')]
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load Shakespeare data
    train_path = 'src/data/shakespeare_train.pt'
    val_path = 'src/data/shakespeare_val.pt'
    meta_path = 'src/data/shakespeare_meta.pt'
    
    # Determine if replacement sampling should be used
    replacement = args.data_loader == 'replacement'
    train_ds, val_ds, train_loader, val_loader = load_shakespeare_data(
        train_path, val_path, args.batch_size, replacement=replacement, seed=args.seed
    )

    os.makedirs(args.results_dir, exist_ok=True)

    for lr in lrs:
        print(f"[info] Computing Hessian trajectories for lr={lr}")
        
        model = create_model(meta_path).to(device)
        model.load_state_dict(torch.load(os.path.join(args.results_dir, args.checkpoint_in), 
                                        map_location=device))
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Pre-allocate memory for trajectories
        n_params = sum(p.numel() for p in model.parameters())
        weights_traj = torch.zeros(args.steps, n_params, device=device)
        hessians_traj = torch.zeros(args.steps, n_params, n_params, device=device)
        
        data_iter = iter(train_loader)
        
        for step in tqdm(range(args.steps), desc=f"Hessian steps (lr={lr})"):
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                data, target = next(data_iter)
                
            data, target = data.to(device), target.to(device)
            
            # Save current weights
            params_vec = torch.cat([p.reshape(-1) for p in model.parameters()])
            weights_traj[step] = params_vec
            
            # Compute full Hessian
            hessian = compute_hessian(model, data, target)
            hessians_traj[step] = hessian
            
            # Training step
            optimizer.zero_grad()
            logits, loss = model(data, target)
            loss.backward()
            optimizer.step()
        
        # Save results
        os.makedirs(args.results_dir, exist_ok=True)
        weights_path = os.path.join(args.results_dir, f"weights_traj_lr{lr}.pt")
        torch.save(weights_traj.cpu(), weights_path)
        print(f"[saved] {weights_path}")
        
        hessian_path = os.path.join(args.results_dir, f"hessians_traj_lr{lr}.pt")
        torch.save(hessians_traj.cpu(), hessian_path)
        print(f"[saved] {hessian_path}")

    print("[done] Hessian analysis completed")


if __name__ == '__main__':
    main()
