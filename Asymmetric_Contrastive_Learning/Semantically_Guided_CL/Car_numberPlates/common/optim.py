"""Optimizer and scheduler factories shared by both experiments."""

from __future__ import annotations

import torch
from torch import nn


class LARS(torch.optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling optimizer for large-batch contrastive learning."""

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-6,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            trust_coefficient = group["trust_coefficient"]
            eps = group["eps"]
            apply_lars = group.get("apply_lars", True)
            apply_weight_decay = group.get("apply_weight_decay", True)

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                if apply_weight_decay and weight_decay != 0:
                    update = update.add(parameter, alpha=weight_decay)

                if apply_lars:
                    param_norm = torch.norm(parameter)
                    update_norm = torch.norm(update)
                    if param_norm > 0 and update_norm > 0:
                        q = trust_coefficient * param_norm / (update_norm + eps)
                        update = update.mul(q)

                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.clone(update).detach()
                else:
                    state["momentum_buffer"].mul_(momentum).add_(update)
                parameter.add_(state["momentum_buffer"], alpha=-lr)

        return loss


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str = "lars",
    momentum: float = 0.9,
    trust_coefficient: float = 0.001,
) -> torch.optim.Optimizer:
    """Build an optimizer suitable for both baseline and semantic-guided runs."""

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name != "lars":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    regularized: list[nn.Parameter] = []
    excluded: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            excluded.append(parameter)
        else:
            regularized.append(parameter)

    return LARS(
        [
            {"params": regularized, "apply_lars": True, "apply_weight_decay": True},
            {"params": excluded, "apply_lars": False, "apply_weight_decay": False},
        ],
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        trust_coefficient=trust_coefficient,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 10,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup followed by cosine decay over epochs."""

    warmup_epochs = max(0, warmup_epochs)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress_denominator = max(1, epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / progress_denominator))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
