"""Trainer for semantic-guided SimCLR."""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from common.config import TrainingConfig
from common.losses import nt_xent_loss
from common.training import AverageMeter, append_jsonl, is_main_process, reduce_metrics, save_checkpoint


class SemanticGuidedTrainer:
    """Train global SimCLR plus annotation-derived plate-context NT-Xent."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        dataloader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")
        self.best_loss = float("inf")

    def train(self) -> None:
        for epoch in range(1, self.config.epochs + 1):
            if hasattr(self.dataloader.sampler, "set_epoch"):
                self.dataloader.sampler.set_epoch(epoch)
            metrics = self.train_epoch(epoch)
            metrics = reduce_metrics(metrics)
            self.scheduler.step()
            is_best = metrics["total_loss"] < self.best_loss
            if is_best:
                self.best_loss = metrics["total_loss"]
            save_checkpoint(
                self.config.output_dir,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                metrics,
                is_best,
            )
            append_jsonl(self.config.output_dir / "metrics.jsonl", metrics)
            if is_main_process():
                print(
                    f"epoch={epoch:04d} global_loss={metrics['global_loss']:.4f} "
                    f"patch_loss={metrics['patch_loss']:.4f} total_loss={metrics['total_loss']:.4f} "
                    f"patches_per_image={metrics['avg_plate_patches_per_image']:.2f} "
                    f"lr={metrics['lr']:.6g} "
                    f"time={metrics['epoch_time_sec']:.1f}s"
                )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        global_meter = AverageMeter()
        patch_meter = AverageMeter()
        total_meter = AverageMeter()
        patch_count_meter = AverageMeter()
        start = time.perf_counter()

        for view_1, view_2, patch_view_1, patch_view_2, _paths in self.dataloader:
            view_1 = view_1.to(self.device, non_blocking=True)
            view_2 = view_2.to(self.device, non_blocking=True)
            patch_view_1 = patch_view_1.to(self.device, non_blocking=True)
            patch_view_2 = patch_view_2.to(self.device, non_blocking=True)
            patch_count_meter.update(1.0, view_1.size(0))

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                z1 = self.model(view_1)
                z2 = self.model(view_2)
                global_loss = nt_xent_loss(z1, z2, self.config.temperature)

                patch_z1 = self.model(patch_view_1)
                patch_z2 = self.model(patch_view_2)
                patch_loss = nt_xent_loss(patch_z1, patch_z2, self.config.temperature)

                total_loss = global_loss + self.config.patch_loss_weight * patch_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = view_1.size(0)
            global_meter.update(global_loss.item(), batch_size)
            patch_meter.update(patch_loss.item(), batch_size)
            total_meter.update(total_loss.item(), batch_size)

        return {
            "epoch": float(epoch),
            "global_loss": global_meter.avg,
            "patch_loss": patch_meter.avg,
            "total_loss": total_meter.avg,
            "avg_plate_patches_per_image": patch_count_meter.avg,
            "lr": self.optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.perf_counter() - start,
        }
