import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image

from tap_charger_scan.models import VerifiedScan
from tap_charger_scan.ports import ImageObjectStore, ScanRepository

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    sample_id: str
    group_id: str
    source: Literal["synthetic", "collected"]
    split: Split
    input_path: str
    expected_payload: str
    target_path: str | None = None
    degradation: dict[str, float] | None = None


def assign_group_splits(group_ids: Sequence[str], seed: int) -> dict[str, Split]:
    """Assign whole charger/sticker groups before any sample augmentation."""
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 3:
        raise ValueError("at least three groups are required for train/validation/test splits")

    random.Random(seed).shuffle(unique_groups)
    validation_count = max(1, round(len(unique_groups) * 0.15))
    test_count = max(1, round(len(unique_groups) * 0.15))
    train_count = len(unique_groups) - validation_count - test_count
    if train_count < 1:
        raise ValueError("not enough groups for a non-empty training split")

    splits: dict[str, Split] = {}
    for group_id in unique_groups[:train_count]:
        splits[group_id] = "train"
    for group_id in unique_groups[train_count : train_count + validation_count]:
        splits[group_id] = "validation"
    for group_id in unique_groups[train_count + validation_count :]:
        splits[group_id] = "test"
    return splits


def write_manifest(records: Iterable[DatasetRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_manifest(path: Path, *, split: Split | None = None) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = DatasetRecord(**json.loads(line))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid manifest record at line {line_number}") from exc
            if split is None or record.split == split:
                records.append(record)
    return records


def resolve_path(manifest_path: Path, record_path: str) -> Path:
    return (manifest_path.parent / record_path).resolve()


async def export_verified_scans(
    repository: ScanRepository,
    object_store: ImageObjectStore,
    output_dir: Path,
    *,
    seed: int = 42,
) -> Path:
    """Materialize verified production scans into the shared manifest format.

    A verified payload is enough for evaluation. A reference image is optional,
    because pixel-supervised restoration training requires a clean, aligned target.
    """
    scans = await repository.list_verified()
    if not scans:
        raise ValueError("no verified scans are available")
    split_by_group = assign_group_splits([scan.group_id for scan in scans], seed)
    records: list[DatasetRecord] = []

    for scan in scans:
        split = split_by_group[scan.group_id]
        input_path = Path("images") / split / f"{scan.scan_id}.png"
        await _save_object_as_png(object_store, scan.image_key, output_dir / input_path)
        target_path = await _export_reference(scan, object_store, output_dir, split)
        records.append(
            DatasetRecord(
                sample_id=str(scan.scan_id),
                group_id=scan.group_id,
                source="collected",
                split=split,
                input_path=input_path.as_posix(),
                target_path=target_path.as_posix() if target_path else None,
                expected_payload=scan.expected_payload,
            )
        )

    manifest_path = output_dir / "manifest.jsonl"
    write_manifest(records, manifest_path)
    return manifest_path


async def _export_reference(
    scan: VerifiedScan,
    object_store: ImageObjectStore,
    output_dir: Path,
    split: Split,
) -> Path | None:
    if scan.reference_image_key is None:
        return None
    target_path = Path("targets") / split / f"{scan.scan_id}.png"
    await _save_object_as_png(object_store, scan.reference_image_key, output_dir / target_path)
    return target_path


async def _save_object_as_png(
    object_store: ImageObjectStore,
    key: str,
    destination: Path,
) -> None:
    data = await object_store.get(key=key)
    with Image.open(BytesIO(data)) as image:
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(destination, format="PNG")
