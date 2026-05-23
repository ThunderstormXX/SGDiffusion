from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import FlexibleMLP


def build_model(
    architecture="flexible_mlp",
    hidden_dim=48,
    num_hidden_layers=1,
    input_downsample=14,
    dropout=0.0,
):
    if architecture != "flexible_mlp":
        raise ValueError(f"unknown architecture: {architecture}")
    input_downsample = None if input_downsample <= 0 else input_downsample
    return FlexibleMLP(
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        input_downsample=input_downsample,
        dropout_list=[dropout] * num_hidden_layers,
        use_relu_list=[True] * num_hidden_layers,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
