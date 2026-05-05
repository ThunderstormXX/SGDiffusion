"""
Tensor utilities for parameter manipulation.
"""


import torch
import torch.nn as nn


def flatten_params(model: nn.Module, detach: bool = True) -> torch.Tensor:
    """
    Flatten all model parameters into a single 1D tensor.

    Args:
        model: The model whose parameters to flatten.
        detach: Whether to detach the tensor from computation graph.

    Returns:
        A 1D tensor containing all parameters concatenated.
    """
    params = []
    for p in model.parameters():
        if detach:
            params.append(p.data.detach().flatten())
        else:
            params.append(p.flatten())
    return torch.cat(params)


def flatten_grads(model: nn.Module) -> torch.Tensor:
    """
    Flatten all model gradients into a single 1D tensor.

    Args:
        model: The model whose gradients to flatten.

    Returns:
        A 1D tensor containing all gradients concatenated.
    """
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros_like(p.data).flatten())
    return torch.cat(grads)


def unflatten_params(
    flat_params: torch.Tensor,
    model: nn.Module,
    copy: bool = True,
) -> None:
    """
    Unflatten a 1D tensor back into model parameters.

    Args:
        flat_params: The flattened parameter tensor.
        model: The model to unflatten into.
        copy: If True, copy data; if False, use the tensor directly.
    """
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        chunk = flat_params[offset:offset + numel].view(p.shape)
        if copy:
            p.data.copy_(chunk)
        else:
            p.data = chunk
        offset += numel


def get_param_count(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Get the number of parameters in a model.

    Args:
        model: The model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_param_shapes(model: nn.Module) -> list[tuple[str, torch.Size]]:
    """
    Get the shapes of all parameters.

    Args:
        model: The model.

    Returns:
        List of (name, shape) tuples.
    """
    return [(name, p.shape) for name, p in model.named_parameters()]


def grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute the gradient norm.

    Args:
        model: The model.
        norm_type: Type of norm (default: L2).

    Returns:
        The gradient norm as a float.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0

    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type,
    )
    return total_norm.item()


def params_to_cpu(model: nn.Module) -> torch.Tensor:
    """
    Get flattened parameters on CPU.

    Args:
        model: The model.

    Returns:
        Flattened parameters on CPU.
    """
    return flatten_params(model, detach=True).cpu()


def grads_to_cpu(model: nn.Module) -> torch.Tensor:
    """
    Get flattened gradients on CPU.

    Args:
        model: The model.

    Returns:
        Flattened gradients on CPU.
    """
    return flatten_grads(model).cpu()
