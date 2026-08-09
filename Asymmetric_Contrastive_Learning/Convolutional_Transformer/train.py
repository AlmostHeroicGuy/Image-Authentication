"""
train.py - Optimized training entry point for the hybrid forensic encoder.

Launch options:
  torchrun --standalone --nproc_per_node=8 train.py
  python train.py   # single-GPU fallback if CUDA is available
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from CNN import create_resnet50_backbone
from augmentations import GlobalBenignAugmentation, LocalMaliciousForgery
from loss import AsymmetricNTXentLoss
from tokenizer import SpatialTokenizer
from transformer import CustomTransformerBlock, SemanticProjector


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the hybrid forensic encoder.")
    parser.add_argument("--dataset-path", default="Faces/train",
                        help="Folder containing pristine face images.")
    parser.add_argument("--batch-per-gpu", type=int, default=64,
                        help="Per-process batch size.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Number of linear warmup epochs.")
    parser.add_argument("--base-lr", type=float, default=1e-4,
                        help="Base learning rate before world-size scaling.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay.")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="NT-Xent temperature.")
    parser.add_argument("--alpha", type=float, default=20.0,
                        help="Hard-negative weighting term.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm.")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save a named checkpoint every N epochs.")
    parser.add_argument("--latest-checkpoint", default="latest_checkpoint.pth",
                        help="Path to the rolling checkpoint file.")
    parser.add_argument("--num-workers", type=int, default=7,
                        help="DataLoader worker processes per rank.")
    parser.add_argument("--enable-checkpointing", action="store_true",
                        help="Enable gradient checkpointing in every transformer block.")
    return parser.parse_args()


def validate_args(args):
    if args.batch_per_gpu < 1:
        raise ValueError("--batch-per-gpu must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs cannot be negative.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be at least 1.")


def setup_training():
    """
    Initialize NCCL when launched with torchrun. If the distributed environment
    variables are absent, fall back to single-GPU execution for easier debugging.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train this model.")

    required_env = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    present_env = [name for name in required_env if name in os.environ]
    if present_env and len(present_env) != len(required_env):
        missing = [name for name in required_env if name not in os.environ]
        raise RuntimeError(
            "Incomplete distributed launch environment. "
            f"Missing {missing}. Launch with torchrun or clear the partial env vars."
        )

    if len(present_env) == len(required_env):
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_distributed = (world_size > 1)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        is_distributed = False

    torch.cuda.set_device(local_rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank, global_rank, world_size, is_distributed


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class PristineFacesDataset(Dataset):
    """
    Loads raw images from disk and resizes them to 512x512.
    All augmentations happen GPU-side in the training loop.
    """

    def __init__(self, folder_path: str = "Faces/train"):
        self.folder_path = Path(folder_path).expanduser()
        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"Dataset folder '{self.folder_path}' does not exist. "
                "Pass the correct path via --dataset-path."
            )
        if not self.folder_path.is_dir():
            raise NotADirectoryError(
                f"Dataset path '{self.folder_path}' is not a directory."
            )

        self.image_paths = sorted(
            path for path in self.folder_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if not self.image_paths:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
            raise FileNotFoundError(
                f"No images found in '{self.folder_path}'. "
                f"Supported extensions: {supported}. "
                "Check your working directory and --dataset-path argument."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
            img = TF.resize(img, [512, 512])
            return TF.to_tensor(img)


class HybridForensicModel(nn.Module):
    """
    Wraps CNN backbone -> tokenizer -> transformer stack -> semantic projector
    into a single forward pass that returns the 2048-D semantic hash.
    """

    def __init__(self):
        super().__init__()
        self.cnn_backbone = create_resnet50_backbone()
        self.tokenizer = SpatialTokenizer(in_channels=1024, spatial_size=32, hidden_dim=768)
        self.transformer_layers = nn.ModuleList([
            CustomTransformerBlock(hidden_dim=768, num_heads=12)
            for _ in range(6)
        ])
        self.projector = SemanticProjector(transformer_dim=768, hash_dim=2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_map = self.cnn_backbone(x)
        z = self.tokenizer(f_map)
        for layer in self.transformer_layers:
            z = layer(z)
        return self.projector(z)


def build_lr_lambda(warmup_epochs: int, total_epochs: int):
    import math

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def train(args):
    validate_args(args)
    device, local_rank, global_rank, world_size, is_distributed = setup_training()
    is_main = (global_rank == 0)

    batch_per_gpu = args.batch_per_gpu
    epochs = args.epochs
    warmup_epochs = args.warmup_epochs
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    temperature = args.temperature
    alpha = args.alpha
    grad_clip = args.grad_clip
    dataset_path = args.dataset_path
    ckpt_interval = args.checkpoint_interval
    latest_ckpt_path = Path(args.latest_checkpoint).expanduser()

    scaled_lr = base_lr * world_size

    if is_main:
        print("=" * 60)
        print("  Hybrid Forensic Encoder")
        print(f"  Mode      : {'DDP' if is_distributed else 'single GPU'}")
        print(f"  GPUs      : {world_size}")
        print(f"  Batch/GPU : {batch_per_gpu}  ->  effective: {batch_per_gpu * world_size}")
        print(f"  Epochs    : {epochs}")
        print(f"  LR scaled : {scaled_lr:.2e}")
        print(f"  Dataset   : {Path(dataset_path).expanduser()}")
        print("=" * 60)

    try:
        dataset = PristineFacesDataset(folder_path=dataset_path)
        sampler = None
        if is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=True,
                drop_last=False,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_per_gpu,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=4,
        )
        if len(dataloader) == 0:
            raise RuntimeError(
                "DataLoader produced zero batches. "
                "Check --dataset-path and --batch-per-gpu."
            )

        samples_per_rank = len(sampler) if sampler is not None else len(dataset)
        if is_main and samples_per_rank < batch_per_gpu:
            print(
                f"  Warning: only {samples_per_rank} samples per rank are available "
                f"each epoch, so batches will be smaller than the requested {batch_per_gpu}."
            )

        model = HybridForensicModel().to(device)
        if args.enable_checkpointing:
            for layer in model.transformer_layers:
                layer.use_checkpoint = True

        if is_distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        aug_global = GlobalBenignAugmentation().to(device)
        aug_local = LocalMaliciousForgery(tamper_size=(96, 192)).to(device)

        criterion = AsymmetricNTXentLoss(
            temperature=temperature,
            alpha=alpha,
            distributed=is_distributed,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=scaled_lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            build_lr_lambda(warmup_epochs, epochs),
        )

        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

        if is_main:
            print(f"  AMP dtype : {amp_dtype}  |  GradScaler: {amp_dtype == torch.float16}")

        start_epoch = 0
        if latest_ckpt_path.exists():
            ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=True)
            unwrap_model(model).load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            if is_main:
                print(f"  Resumed from epoch {start_epoch}")

        if is_distributed:
            dist.barrier()

        latest_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, epochs):
            model.train()
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_loss = 0.0

            for batch_idx, x_pristine in enumerate(dataloader):
                optimizer.zero_grad(set_to_none=True)
                x_pristine = x_pristine.to(device, non_blocking=True)

                with torch.no_grad():
                    x_global = aug_global(x_pristine)
                    x_tampered = aug_local(x_pristine)

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    x_all = torch.cat([x_pristine, x_global, x_tampered], dim=0)
                    h_all = model(x_all)
                    h_anchor, h_positive, h_tampered = h_all.chunk(3, dim=0)
                    loss = criterion(h_anchor, h_positive, h_tampered)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss = epoch_loss + loss.item()

                if is_main and batch_idx % 10 == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(
                        f"  Epoch [{epoch + 1}/{epochs}]  "
                        f"Batch [{batch_idx:4d}/{len(dataloader)}]  "
                        f"Loss: {loss.item():.4f}  LR: {lr_now:.2e}"
                    )

            scheduler.step()

            loss_tensor = torch.tensor(epoch_loss / len(dataloader), device=device)
            if is_distributed:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensor = loss_tensor / world_size

            if is_main:
                print("\n" + "-" * 60)
                print(
                    f"  Epoch {epoch + 1:3d}/{epochs} complete  |  "
                    f"Avg Loss: {loss_tensor.item():.4f}  |  "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                print("-" * 60 + "\n")

                ckpt_state = {
                    "epoch": epoch,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss": loss_tensor.item(),
                }
                torch.save(ckpt_state, latest_ckpt_path)

                if (epoch + 1) % ckpt_interval == 0:
                    epoch_ckpt = latest_ckpt_path.with_name(
                        f"hybrid_encoder_epoch_{epoch + 1:03d}.pth"
                    )
                    torch.save(ckpt_state, epoch_ckpt)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    train(parse_args())
