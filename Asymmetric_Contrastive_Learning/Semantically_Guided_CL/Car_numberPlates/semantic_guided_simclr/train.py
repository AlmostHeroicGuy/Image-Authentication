"""Run semantic-guided SimCLR training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import TrainingConfig
from common.modeling import SimCLRModel
from common.optim import build_optimizer, build_scheduler
from common.training import cleanup_distributed, get_device, init_distributed, is_main_process, set_seed
from semantic_guided_simclr.dataset import build_dataset
from semantic_guided_simclr.trainer import SemanticGuidedTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train semantic-guided SimCLR.")
    parser.add_argument("--data-root", type=Path, required=True, help="Generated CCPD root with subset/real and subset/manipulated folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/semantic_guided_simclr"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size when launched with torchrun.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("lars", "adamw"), default="lars")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--trust-coefficient", type=float, default=0.001)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-loss-weight", type=float, default=10.0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--bbox-area-scale", type=float, default=1.8)
    parser.add_argument("--from-scratch", action="store_true", help="Disable ImageNet-pretrained ResNet50 initialization.")
    parser.add_argument("--no-sync-bn", action="store_true", help="Disable SyncBatchNorm in DDP.")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    config = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        momentum=args.momentum,
        trust_coefficient=args.trust_coefficient,
        warmup_epochs=args.warmup_epochs,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        use_amp=not args.no_amp,
        patch_loss_weight=args.patch_loss_weight,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        bbox_area_scale=args.bbox_area_scale,
        pretrained_backbone=not args.from_scratch,
        sync_batchnorm=not args.no_sync_bn,
    )
    set_seed(config.seed + rank)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else get_device()

    dataset = build_dataset(
        config.data_root,
        config.image_size,
        train_fraction=config.train_fraction,
        split_seed=config.split_seed,
        bbox_area_scale=config.bbox_area_scale,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = SimCLRModel(config.projection_dim, config.hidden_dim, config.pretrained_backbone).to(device)
    if distributed and device.type == "cuda" and config.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = build_optimizer(
        model,
        config.learning_rate,
        config.weight_decay,
        config.optimizer,
        config.momentum,
        config.trust_coefficient,
    )
    scheduler = build_scheduler(optimizer, config.epochs, config.warmup_epochs)
    if is_main_process():
        print(
            f"distributed={distributed} world_size={world_size} "
            f"batch_per_gpu={config.batch_size} global_batch={config.batch_size * world_size} "
            f"optimizer={config.optimizer} lr={config.learning_rate} warmup_epochs={config.warmup_epochs} "
            f"image_size={config.image_size} sync_bn={config.sync_batchnorm} "
            f"bbox_area_scale={config.bbox_area_scale} patch_loss_weight={config.patch_loss_weight}"
        )
    trainer = SemanticGuidedTrainer(
        model,
        optimizer,
        scheduler,
        dataloader,
        config,
        device,
    )
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
