from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from tap_charger_scan.dataset import DatasetRecord, write_manifest
from tap_charger_scan.evaluate import evaluate


class PixelDecoder:
    def decode(self, image: Image.Image) -> SimpleNamespace:
        return SimpleNamespace(payload="expected" if image.getpixel((0, 0))[0] > 200 else None)


def test_evaluation_compares_exact_decodes_and_failures(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    Image.new("RGB", (8, 8), "black").save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            DatasetRecord(
                sample_id="one",
                group_id="one",
                source="synthetic",
                split="test",
                input_path=image_path.name,
                expected_payload="expected",
            )
        ],
        manifest,
    )

    report = evaluate(
        manifest,
        {
            "opencv": lambda image: image,
            "pretrained_swin2sr": lambda image: Image.new("RGB", image.size, "white"),
        },
        decoder=PixelDecoder(),
    )

    assert report["systems"]["opencv"]["no_decode"] == 1
    assert report["systems"]["pretrained_swin2sr"]["exact_accuracy"] == 1.0
    assert report["systems"]["pretrained_swin2sr"]["accuracy_change_vs_opencv"] == 1.0
