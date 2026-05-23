import copy
import random
import numpy as np
import torch
from tqdm.auto import tqdm

from dataloader import make_sgd_loader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name):
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "mps" and has_mps:
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "auto":
        if has_mps:
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def params_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().float()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def _step(model, batch, optimizer, device):
    criterion = torch.nn.CrossEntropyLoss()
    x, y = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_gd(model, loader, val_loader, test_loader, lr, epochs, device, eval_every):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    batch = next(iter(loader))
    traj, logs = [params_vector(model)], []
    bar = tqdm(range(1, epochs + 1), desc="GD epochs", position=0)
    for epoch in bar:
        model.train()
        loss = _step(model, batch, optimizer, device)
        traj.append(params_vector(model))
        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate(model, val_loader, device)
            test = evaluate(model, test_loader, device)
            logs.append((epoch, loss, val[1], test[1]))
            bar.set_postfix(loss=f"{loss:.3f}", val=f"{val[1]:.3f}", test=f"{test[1]:.3f}")
    return torch.stack(traj).numpy(), logs


def train_sgd_runs(factory, init_state, train_ds, val_loader, test_loader, args, device):
    all_traj, logs = [], []
    run_bar = tqdm(range(args.runs), desc="SGD runs", position=0)
    for run in run_bar:
        set_seed(args.seed + 1000 + run)
        model = factory().to(device)
        model.load_state_dict(copy.deepcopy(init_state))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        loader = make_sgd_loader(train_ds, args.batch_size, args.sgd_iterations, args.seed + 2000 + run)
        traj = [params_vector(model)]
        step_bar = tqdm(loader, total=args.sgd_iterations, desc=f"run {run + 1}/{args.runs}", leave=False, position=1)
        for step, batch in enumerate(step_bar, 1):
            loss = _step(model, batch, optimizer, device)
            traj.append(params_vector(model))
            if step % args.eval_every == 0 or step == args.sgd_iterations:
                val = evaluate(model, val_loader, device)
                test = evaluate(model, test_loader, device)
                step_bar.set_postfix(loss=f"{loss:.3f}", val=f"{val[1]:.3f}", test=f"{test[1]:.3f}")
        logs.append((run, *evaluate(model, val_loader, device), *evaluate(model, test_loader, device)))
        all_traj.append(torch.stack(traj).numpy())
    return np.stack(all_traj), logs
