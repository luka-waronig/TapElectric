from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image

from tap_charger_scan.dataset import export_verified_scans, read_manifest, resolve_path
from tap_charger_scan.models import VerifiedScan
from tap_charger_scan.synthetic import INPUT_SIZE, TARGET_SIZE, generate_dataset


def test_synthetic_data_is_grouped_before_degradation(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path, payload_count=6, variants_per_payload=2, seed=7)
    records = read_manifest(manifest)

    groups_by_split = {
        split: {record.group_id for record in records if record.split == split}
        for split in ("train", "validation", "test")
    }
    assert groups_by_split["train"].isdisjoint(groups_by_split["validation"])
    assert groups_by_split["train"].isdisjoint(groups_by_split["test"])
    assert groups_by_split["validation"].isdisjoint(groups_by_split["test"])

    first = records[0]
    with Image.open(resolve_path(manifest, first.input_path)) as input_image:
        assert input_image.size == (INPUT_SIZE, INPUT_SIZE)
    assert first.target_path is not None
    with Image.open(resolve_path(manifest, first.target_path)) as target_image:
        assert target_image.size == (TARGET_SIZE, TARGET_SIZE)


class VerifiedRepository:
    async def list_verified(self) -> list[VerifiedScan]:
        return [
            VerifiedScan(uuid4(), f"scan-{index}", f"payload-{index}", f"group-{index}", None)
            for index in range(3)
        ]


class MemoryObjectStore:
    def __init__(self) -> None:
        output = BytesIO()
        Image.new("RGB", (32, 32), "white").save(output, format="PNG")
        self.image = output.getvalue()

    async def get(self, *, key: str) -> bytes:
        return self.image


async def test_verified_scans_export_to_the_same_evaluation_manifest(tmp_path: Path) -> None:
    manifest = await export_verified_scans(VerifiedRepository(), MemoryObjectStore(), tmp_path)
    records = read_manifest(manifest)

    assert len(records) == 3
    assert {record.source for record in records} == {"collected"}
    assert all(record.target_path is None for record in records)
    assert all(resolve_path(manifest, record.input_path).exists() for record in records)
