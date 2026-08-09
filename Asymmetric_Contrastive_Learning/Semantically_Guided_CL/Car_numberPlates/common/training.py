"""Training utilities shared by both experiments."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


class AverageMeter:
    """Track the running average of a scalar metric."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_distributed() -> tuple[bool, int, int, int]:
    """Initialize torch.distributed from torchrun environment variables."""

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def reduce_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Average scalar metrics across distributed workers."""

    if not dist.is_available() or not dist.is_initialized():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(value.item()) for key, value in zip(keys, values)}


def epoch_timer() -> float:
    return time.perf_counter()


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    is_best: bool,
) -> None:
    if not is_main_process():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "backbone": unwrapped.backbone.state_dict(),
        "projection_head": unwrapped.projection_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, output_dir / "latest.pt")
    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
