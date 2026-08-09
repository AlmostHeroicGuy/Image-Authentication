"""Configuration objects shared by the baseline and semantic-guided runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Hyperparameters that must remain identical across both methods."""

    data_root: Path
    output_dir: Path
    image_size: int = 512
    batch_size: int = 64
    epochs: int = 100
    num_workers: int = 4
    learning_rate: float = 0.3
    weight_decay: float = 1e-4
    optimizer: str = "lars"
    momentum: float = 0.9
    trust_coefficient: float = 0.001
    warmup_epochs: int = 10
    temperature: float = 0.5
    projection_dim: int = 128
    hidden_dim: int = 2048
    seed: int = 42
    use_amp: bool = True
    patch_loss_weight: float = 10.0
    train_fraction: float = 0.9
    split_seed: int = 42
    bbox_area_scale: float = 1.8
    pretrained_backbone: bool = True
    sync_batchnorm: bool = True
    checkpoint_metric: str = "loss"

    @property
    def train_original_dir(self) -> Path:
        """Return a legacy train/original path when older scripts still ask for it."""

        split_dir = self.data_root / "train" / "original"
        if split_dir.exists():
            return split_dir
        return self.data_root
