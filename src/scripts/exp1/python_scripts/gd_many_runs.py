#!/usr/bin/env python3
import os, sys, argparse, random, numpy as np, gc, json
import torch, torch.nn as nn, torch.optim as optim
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FlexibleMLP, FlexibleCNN
from src.utils import load_saved_data

# -------------------- утилиты --------------------
def get_device(force_auto: bool = True) -> torch.device:
    if force_auto:
        if torch.cuda.is_available(): 
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
            return torch.device("mps")
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

def create_model(model_type, dtype):
    if model_type == 'mlp':
        return FlexibleMLP(hidden_dim=8, num_hidden_layers=1, input_downsample=6).to(dtype)
    else:
        return FlexibleCNN(
            in_channels=1, conv_channels=[12], conv_kernels=[3], conv_strides=[1],
            conv_use_relu_list=[True], conv_dropouts=[0.0], conv_use_bn=True,
            pool_after=[False], gap_size=1, mlp_hidden_dims=[11],
            mlp_use_relu_list=[True], mlp_dropouts=[0.0], output_dim=10
        ).to(dtype)

@torch.no_grad()
def params_vector(model: nn.Module, dtype: torch.dtype) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().to(dtype)

def disable_dropout_in_train(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.p = 0.0


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
    parser.add_argument("--lr_scaling", default='default')
    parser.add_argument("--dtype", default="float32",
                        help="torch dtype: float32, float64, bfloat16, float16")
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # --- dtype mapping ---
    dtype_map = {
        "float32": torch.float32, "float": torch.float32,
        "float64": torch.float64, "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16, "half": torch.float16,
    }
    if args.dtype.lower() not in dtype_map:
        raise ValueError(f"Unsupported dtype {args.dtype}. Choices: {list(dtype_map.keys())}")
    dtype = dtype_map[args.dtype.lower()]

    device = args.device
    lrs = [float(x) for x in args.lrs.split(',')]

    if args.dataset_train == 'mnist': 
        train_path = 'src/data/mnist_train.pt'
        test_path = 'src/data/mnist_test.pt'
    else:
        Exception('пока не надо similar data')
    load_data_fn = load_saved_data
    train_ds, val_ds, train_loader_full, val_loader = load_data_fn(
        train_path=train_path, test_path=test_path, batch_size=args.sample_size
    )
    
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.results_dir, exist_ok=True)

    ckpt_path = os.path.join(args.results_dir, args.checkpoint_in) \
        if not os.path.isabs(args.checkpoint_in) else args.checkpoint_in

    model_template = create_model(args.model, dtype).to(device=device)
    model_template.load_state_dict(torch.load(ckpt_path, map_location=device))
    model_template = model_template.to(dtype)
    n_params = params_vector(model_template, dtype).numel()

    for lr in lrs:
        print(f"[info] lr={lr} | steps={args.steps} | n_params={n_params} "
              f"| train_size={len(train_ds)} | val_size={len(val_ds)} | dtype={dtype}")

        weights_trajs = torch.empty((args.steps, n_params), dtype=dtype, device='cpu')
        logs = {"train_losses": [], "val_losses": [], "val_accs": []}

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        model = create_model(args.model, dtype).to(device=device)
        model.load_state_dict(model_template.state_dict())
        model = model.to(dtype)
        disable_dropout_in_train(model)

        gd_lr = lr if args.lr_scaling == 'default' else lr / args.sample_size * args.batch_size if args.lr_scaling == 'N/B' else None
        optimizer = optim.SGD(model.parameters(), lr=gd_lr)

        # --- подготовка полного батча один раз ---
        full_data, full_target = next(iter(train_loader_full))
        full_data = full_data.to(device=device, dtype=dtype)
        full_target = full_target.to(device=device)  # labels остаются long

        step_bar = tqdm(range(args.steps), desc=f"GD training (lr={lr})")
        for step in step_bar:
            # логируем ПЕРЕД шагом
            v = params_vector(model, dtype)
            weights_trajs[step].copy_(v)

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
                    data = data.to(device=device, dtype=dtype)
                    target = target.to(device=device)  # long
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
        out_path = os.path.join(args.results_dir, f"gd_weights_lr{lr}_dtype{args.dtype}.pt")
        torch.save(weights_trajs, out_path)
        print(f"[saved] {out_path} shape={tuple(weights_trajs.shape)} dtype={weights_trajs.dtype}")

        logs_path = os.path.join(args.results_dir, f"gd_logs_lr{lr}_dtype{args.dtype}.json")
        with open(logs_path, "w") as f:
            json.dump(logs, f)
        print(f"[saved] {logs_path}")

        del model, optimizer, weights_trajs
        gc.collect()
        clear_device_caches()

    print("[done] all lrs processed")


if __name__ == '__main__':
    main()
