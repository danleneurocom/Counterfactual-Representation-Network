from __future__ import annotations

import torch
from torch import Tensor


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * grad_output, None


def gradient_reverse(value: Tensor, strength: float = 1.0) -> Tensor:
    """Reverse gradients for adversarial proxy heads."""

    return _GradientReverse.apply(value, float(strength))
