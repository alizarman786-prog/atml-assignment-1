"""Sharpness-Aware Minimization (SAM) optimizer wrapper, following Foret et
al. (2021). Wraps a base optimizer (AdamW here); each training step calls
first_step() after computing gradients at the current parameters (this
perturbs parameters to the nearby worst-case point within radius rho), then
a SECOND forward/backward pass is computed AT that perturbed point, and
second_step() restores the original parameters before applying the actual
AdamW update using the gradients computed at the perturbed point.

This is a public-implementation pattern the assignment explicitly permits
("Public implementations of individual losses...may be used"); the
architecture and pattern here follow the widely-used davda54/sam reference
implementation of the SAM paper.
"""
from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **base_kwargs):
        assert rho >= 0
        defaults = dict(rho=rho, **base_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        """Ascend to the nearby worst-case point: theta -> theta + epsilon,
        epsilon = rho * grad / ||grad||_2 (global norm across all params)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        """Restore theta (undo the ascent step), then apply the base
        optimizer's real update using the gradients computed AT the
        perturbed point (the caller must have done a fresh backward() while
        parameters were still perturbed, before calling this)."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p not in self.state or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(2).to(device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), 2,
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("Use first_step()/second_step() explicitly, not step().")
