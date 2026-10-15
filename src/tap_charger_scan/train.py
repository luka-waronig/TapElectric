import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import Swin2SRForImageSuperResolution

from tap_charger_scan.dataset import DatasetRecord, read_manifest, resolve_path
from tap_charger_scan.restoration import MODEL_ID, MODEL_REVISION, image_to_tensor, resolve_device
from tap_charger_scan.synthetic import INPUT_SIZE, TARGET_SIZE


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    manifest: Path
    output_dir: Path
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-5
    seed: int = 42
    device: str = "auto"
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")


class RestorationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, manifest: Path, split: str) -> None:
        self._manifest = manifest
        self._records = [
            record
            for record in read_manifest(manifest)
            if record.split == split and record.target_path
        ]
        if not self._records:
            raise ValueError(f"no paired records found for the {split} split")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self._records[index]
        input_image = _load_rgb(resolve_path(self._manifest, record.input_path))
        target_image = _load_rgb(resolve_path(self._manifest, _target_path(record)))
        if input_image.size != (INPUT_SIZE, INPUT_SIZE):
            raise ValueError(f"{record.sample_id}: input must be {INPUT_SIZE}x{INPUT_SIZE}")
        if target_image.size != (TARGET_SIZE, TARGET_SIZE):
            raise ValueError(f"{record.sample_id}: target must be {TARGET_SIZE}x{TARGET_SIZE}")
        return image_to_tensor(input_image), image_to_tensor(target_image)


def train(config: TrainingConfig, model: Any | None = None) -> dict[str, list[float]]:
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    train_data = RestorationDataset(config.manifest, "train")
    validation_data = RestorationDataset(config.manifest, "validation")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size)

    if model is None:
        model = Swin2SRForImageSuperResolution.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            use_safetensors=True,
        )
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    history = {"train_loss": [], "validation_loss": []}
    best_validation_loss = float("inf")
    best_epoch = 0
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        history["train_loss"].append(
            _run_epoch(model, train_loader, device, optimizer, config.max_steps)
        )
        validation_loss = _run_epoch(model, validation_loader, device, optimizer=None)
        history["validation_loss"].append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            model.save_pretrained(config.output_dir, safe_serialization=True)

    metadata = {
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "best_epoch": best_epoch,
        "config": {
            **asdict(config),
            "manifest": str(config.manifest),
            "output_dir": str(config.output_dir),
        },
        "history": history,
    }
    (config.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return history


def _run_epoch(
    model: Any,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    optimizer: AdamW | None,
    max_steps: int | None = None,
) -> float:
    model.train(optimizer is not None)
    losses: list[float] = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for step, (inputs, targets) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            reconstruction = model(pixel_values=inputs).reconstruction
            reconstruction = reconstruction[..., : targets.shape[-2], : targets.shape[-1]]
            loss = functional.l1_loss(reconstruction, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            losses.append(loss.detach().item())
    if not losses:
        raise ValueError("no training steps were executed")
    return sum(losses) / len(losses)


def _target_path(record: DatasetRecord) -> str:
    if record.target_path is None:
        raise ValueError(f"{record.sample_id}: target path is missing")
    return record.target_path


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Swin2SR on QR restoration pairs")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    history = train(
        TrainingConfig(
            manifest=args.manifest,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            max_steps=args.max_steps,
        )
    )
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
