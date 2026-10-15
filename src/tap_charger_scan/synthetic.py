import argparse
import random
from pathlib import Path

import numpy as np
import qrcode
from PIL import Image, ImageFilter
from qrcode.constants import ERROR_CORRECT_M

from tap_charger_scan.dataset import DatasetRecord, assign_group_splits, write_manifest

INPUT_SIZE = 64
SCALE = 2
TARGET_SIZE = INPUT_SIZE * SCALE


def generate_dataset(
    output_dir: Path,
    *,
    payload_count: int = 20,
    variants_per_payload: int = 3,
    seed: int = 42,
) -> Path:
    if payload_count < 3:
        raise ValueError("payload_count must be at least three")
    if variants_per_payload < 1:
        raise ValueError("variants_per_payload must be positive")

    payloads = [f"tap://charger/C{index:04d}" for index in range(payload_count)]
    groups = [f"charger-C{index:04d}" for index in range(payload_count)]
    split_by_group = assign_group_splits(groups, seed)
    rng = random.Random(seed)
    records: list[DatasetRecord] = []

    for payload, group_id in zip(payloads, groups, strict=True):
        split = split_by_group[group_id]
        clean = _render_qr(payload)
        for variant in range(variants_per_payload):
            sample_id = f"{group_id}-{variant:02d}"
            degraded, parameters = _degrade(clean, rng)
            input_path = Path("images") / split / f"{sample_id}.png"
            target_path = Path("targets") / split / f"{sample_id}.png"
            _save(degraded, output_dir / input_path)
            _save(clean, output_dir / target_path)
            records.append(
                DatasetRecord(
                    sample_id=sample_id,
                    group_id=group_id,
                    source="synthetic",
                    split=split,
                    input_path=input_path.as_posix(),
                    target_path=target_path.as_posix(),
                    expected_payload=payload,
                    degradation=parameters,
                )
            )

    manifest_path = output_dir / "manifest.jsonl"
    write_manifest(records, manifest_path)
    return manifest_path


def _render_qr(payload: str) -> Image.Image:
    code = qrcode.QRCode(error_correction=ERROR_CORRECT_M, box_size=8, border=4)
    code.add_data(payload)
    code.make(fit=True)
    image = code.make_image(fill_color="black", back_color="white").convert("RGB")
    return image.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.NEAREST)


def _degrade(clean: Image.Image, rng: random.Random) -> tuple[Image.Image, dict[str, float]]:
    blur_radius = rng.uniform(0.2, 1.0)
    noise_sigma = rng.uniform(0.0, 10.0)
    dark_edge = rng.uniform(0.45, 0.85)

    low_resolution = clean.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
    blurred = low_resolution.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    pixels = np.asarray(blurred, dtype=np.float32)
    lighting = np.linspace(dark_edge, 1.0, INPUT_SIZE, dtype=np.float32)
    if rng.random() < 0.5:
        lighting = lighting[::-1]
    pixels *= lighting[np.newaxis, :, np.newaxis]
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, noise_sigma, pixels.shape)
    degraded = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
    return degraded, {
        "blur_radius": round(blur_radius, 4),
        "noise_sigma": round(noise_sigma, 4),
        "dark_edge": round(dark_edge, 4),
    }


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grouped synthetic QR restoration data")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--payload-count", type=int, default=20)
    parser.add_argument("--variants-per-payload", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = generate_dataset(
        args.output_dir,
        payload_count=args.payload_count,
        variants_per_payload=args.variants_per_payload,
        seed=args.seed,
    )
    print(manifest)


if __name__ == "__main__":
    main()
