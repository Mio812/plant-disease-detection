"""Typed configuration loaded from a YAML file."""

from dataclasses import dataclass, field

import yaml


@dataclass
class DataConfig:
    root: str = "data/PlantVillage/raw/color"
    image_size: int = 128
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 64
    num_workers: int = 4


@dataclass
class ModelConfig:
    name: str = "resnet18"
    pretrained: bool = True


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    early_stop_patience: int = 7


@dataclass
class SeverityConfig:
    thresholds: tuple = (0.05, 0.20, 0.50)
    levels: tuple = ("healthy", "mild", "moderate", "severe")


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            seed=raw.get("seed", 42),
            output_dir=raw.get("output_dir", "outputs"),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            severity=SeverityConfig(**raw.get("severity", {})),
        )
