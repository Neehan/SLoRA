import torch
from typing import List, Tuple


def get_lora_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    """
    Extract LoRA parameters from PEFT model.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        List of (name, parameter) tuples for LoRA parameters
    """
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_" in name.lower() or "lora_a" in name.lower() or "lora_b" in name.lower()):
            lora_params.append((name, param))
    return lora_params


def lora_grad_vec(model) -> torch.Tensor:
    """
    Extract flattened gradient vector from LoRA parameters.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        Flattened gradient tensor concatenated from all LoRA parameters
    """
    grads = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if param.requires_grad and ("lora_" in name.lower() or "lora_a" in name.lower() or "lora_b" in name.lower()):
            grads.append(param.grad.detach().flatten())

    assert len(grads) > 0, "No LoRA gradients found. Check that LoRA modules exist and require_grad=True."
    return torch.cat(grads)


def compute_lora_dim(model) -> int:
    """
    Compute total dimension of flattened LoRA parameter vector.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        Total number of LoRA parameters
    """
    lora_params = get_lora_params(model)
    assert len(lora_params) > 0, "No LoRA parameters found in model."
    total_dim = sum(p.numel() for _, p in lora_params)
    return total_dim
