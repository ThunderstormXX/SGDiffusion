import argparse
from pathlib import Path
import torch
from tqdm.auto import tqdm

from dataloader import load_dataset, make_sgd_loader
from model import build_model, count_parameters
from pretrain_plotter import plot_metric, plot_pretrain_losses
from tracker import OscillationFraction, TrainingLogger
from trainer import _step, evaluate, pick_device, set_seed


def parse_args():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--architecture", default="flexible_mlp")
    parser.add_argument("--train_size", type=int, default=6400)
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain_sgd_steps", type=int, default=1000)
    parser.add_argument("--pretrain_gd_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--input_downsample", type=int, default=14)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--data_dir", default=str(root / "src/data/MNIST"))
    parser.add_argument("--checkpoint_out", default=str(root / "src/scripts/exp8/results/pretrained_points/pretrained_point.pt"))
    parser.add_argument("--figures_dir", default=str(root / "src/scripts/exp8/figures/pretrain"))
    parser.add_argument("--logs_dir", default=str(root / "src/scripts/exp8/results/pretrained_points"))
    return parser.parse_args()


def train_phase(model, iterable, steps, lr, desc, offset, train_loader, args, device, logger):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    bar = tqdm(iterable, total=steps, desc=desc)
    for step, batch in enumerate(bar, 1):
        loss = _step(model, batch, optimizer, device)
        full_loss = None
        if step % args.eval_every == 0 or step == steps:
            full_loss = evaluate(model, train_loader, device)[0]
            bar.set_postfix(batch=f"{loss:.3f}", full=f"{full_loss:.3f}")
        logger.log(model, phase=desc, global_step=offset + step, phase_step=step,
                   batch_loss=loss, full_train_loss=full_loss)


def train_pretrain(model, train_ds, gd_loader, args, device, logger):
    if args.pretrain_sgd_steps > 0:
        sgd_loader = make_sgd_loader(train_ds, args.batch_size, args.pretrain_sgd_steps, args.seed + 3000)
        train_phase(model, sgd_loader, args.pretrain_sgd_steps, args.lr,
                    "pretrain SGD", 0, gd_loader, args, device, logger)
    if args.pretrain_gd_epochs > 0:
        gd_batches = [next(iter(gd_loader))] * args.pretrain_gd_epochs
        train_phase(model, gd_batches, args.pretrain_gd_epochs, args.lr,
                    "pretrain GD", args.pretrain_sgd_steps, gd_loader, args, device, logger)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    train_ds, gd_loader, val_loader, test_loader = load_dataset(
        args.dataset, args.data_dir, args.train_size, args.val_size,
        args.test_size, args.batch_size, args.seed,
    )
    model = build_model(args.architecture, args.hidden_dim, args.num_hidden_layers,
                        args.input_downsample, args.dropout).to(device)
    print(f"device={device} params={count_parameters(model)}")
    print(f"pretrain SGD steps={args.pretrain_sgd_steps} GD epochs={args.pretrain_gd_epochs} lr={args.lr}")
    logger = TrainingLogger([OscillationFraction()])
    logger.prime(model)
    train_pretrain(model, train_ds, gd_loader, args, device, logger)
    logs = logger.rows
    val_loss, val_acc = evaluate(model, val_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    out_path = Path(args.checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"val_loss": val_loss, "val_acc": val_acc, "test_loss": test_loss, "test_acc": test_acc}
    torch.save({"model_state": model.state_dict(), "args": vars(args), "logs": logs, **metrics}, out_path)
    fig_path = Path(args.figures_dir) / "pretrain_losses.png"
    plot_pretrain_losses(logs, args.pretrain_sgd_steps, fig_path)
    osc_path = Path(args.figures_dir) / "oscillation_fraction.png"
    plot_metric(logs, "oscillation_fraction", osc_path)
    logger.save_jsonl(Path(args.logs_dir) / "pretrain_metrics.jsonl")
    logger.save_csv(Path(args.logs_dir) / "pretrain_metrics.csv")
    print(f"saved checkpoint: {out_path}")
    print(f"saved pretrain figure: {fig_path}")
    print(f"saved oscillation figure: {osc_path}")
    print(f"final val_acc={val_acc:.3f} test_acc={test_acc:.3f}")


if __name__ == "__main__":
    main()
