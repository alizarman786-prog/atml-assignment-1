"""Gradient Reversal Layer (GRL) and the DANN alpha schedule, shared by
DANN (Task 2 Step 3) and CDAN (Task 2 Step 4).
"""
from __future__ import annotations

import math

import torch
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Identity on the forward pass; on the backward pass, send the
        # OPPOSITE gradient, scaled by alpha -- this is the entire mechanism
        # that turns a shared feature extractor into an adversary against
        # its own domain discriminator.
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradientReversalFunction.apply(x, alpha)


def dann_alpha(p: float, max_alpha: float = 1.0) -> float:
    """The assignment's schedule: alpha(p) = 2/(1+exp(-10p)) - 1, p in [0,1]
    training progress. `max_alpha` scales the schedule's [0,1] output range,
    used by Task 2 Step 6's controlled study (varying maximum
    gradient-reversal strength over {0.25, 0.5, 1}) while keeping the same
    schedule shape."""
    p = max(0.0, min(1.0, p))
    return max_alpha * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
