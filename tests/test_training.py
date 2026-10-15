from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

from tap_charger_scan.synthetic import generate_dataset
from tap_charger_scan.train import TrainingConfig, train


def test_swin2sr_contract_maps_64_pixels_to_128_without_weights() -> None:
    model = Swin2SRForImageSuperResolution(
        Swin2SRConfig(embed_dim=12, depths=(1,), num_heads=(3,), upscale=2)
    )

    output = model(pixel_values=torch.rand(1, 3, 64, 64)).reconstruction

    assert output.shape == (1, 3, 128, 128)


class TinyRestorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        restored = functional.interpolate(
            pixel_values, scale_factor=2, mode="bilinear", align_corners=False
        )
        return SimpleNamespace(reconstruction=restored + self.offset)

    def save_pretrained(self, output_dir: Path, *, safe_serialization: bool) -> None:
        torch.save(self.state_dict(), output_dir / "tiny-model.pt")


def test_training_loop_runs_with_an_injected_tiny_model(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path / "data", payload_count=6, variants_per_payload=1)
    output_dir = tmp_path / "model"

    history = train(
        TrainingConfig(
            manifest=manifest,
            output_dir=output_dir,
            batch_size=2,
            max_steps=1,
            device="cpu",
        ),
        model=TinyRestorer(),
    )

    assert len(history["train_loss"]) == 1
    assert len(history["validation_loss"]) == 1
    assert (output_dir / "training_metadata.json").exists()
