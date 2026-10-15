import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from tap_charger_scan.baseline import OpenCVQRDecoder
from tap_charger_scan.dataset import DatasetRecord, read_manifest, resolve_path
from tap_charger_scan.restoration import Swin2SRRestorer

ImageTransform = Callable[[Image.Image], Image.Image]


def evaluate(
    manifest: Path,
    pipelines: dict[str, ImageTransform],
    *,
    decoder: Any | None = None,
) -> dict[str, Any]:
    records = read_manifest(manifest, split="test")
    if not records:
        raise ValueError("the manifest has no test records")
    decoder = decoder or OpenCVQRDecoder()
    outcomes: dict[str, list[tuple[str, float]]] = {name: [] for name in pipelines}

    for record in records:
        with Image.open(resolve_path(manifest, record.input_path)) as source:
            image = source.convert("RGB")
        for name, transform in pipelines.items():
            started = perf_counter()
            decoded = decoder.decode(transform(image.copy())).payload
            latency_ms = (perf_counter() - started) * 1000
            status = _outcome(decoded, record)
            outcomes[name].append((status, latency_ms))

    report = {
        "sample_count": len(records),
        "systems": {name: _summarize(values) for name, values in outcomes.items()},
    }
    baseline_accuracy = report["systems"]["opencv"]["exact_accuracy"]
    for name, metrics in report["systems"].items():
        metrics["accuracy_change_vs_opencv"] = round(
            metrics["exact_accuracy"] - baseline_accuracy, 6
        )
    return report


def build_pipelines(
    *,
    device: str = "auto",
    fine_tuned_checkpoint: Path | None = None,
) -> dict[str, ImageTransform]:
    pretrained = Swin2SRRestorer.from_pretrained(device)
    pipelines: dict[str, ImageTransform] = {
        "opencv": _identity,
        "pretrained_swin2sr": pretrained.restore,
    }
    if fine_tuned_checkpoint is not None:
        fine_tuned = Swin2SRRestorer.from_checkpoint(fine_tuned_checkpoint, device)
        pipelines["fine_tuned_swin2sr"] = fine_tuned.restore
    return pipelines


def _identity(image: Image.Image) -> Image.Image:
    return image


def _outcome(decoded: str | None, record: DatasetRecord) -> str:
    if decoded is None:
        return "no_decode"
    if decoded == record.expected_payload:
        return "correct"
    return "incorrect"


def _summarize(values: Sequence[tuple[str, float]]) -> dict[str, float | int]:
    count = len(values)
    correct = sum(status == "correct" for status, _ in values)
    incorrect = sum(status == "incorrect" for status, _ in values)
    no_decode = count - correct - incorrect
    latencies = sorted(latency for _, latency in values)
    return {
        "correct": correct,
        "incorrect_payloads": incorrect,
        "no_decode": no_decode,
        "exact_accuracy": round(correct / count, 6),
        "incorrect_payload_rate": round(incorrect / count, 6),
        "no_decode_rate": round(no_decode / count, 6),
        "latency_ms_p50": round(_percentile(latencies, 0.50), 3),
        "latency_ms_p95": round(_percentile(latencies, 0.95), 3),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare QR decoding pipelines on held-out data")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--fine-tuned-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.manifest,
        build_pipelines(
            device=args.device,
            fine_tuned_checkpoint=args.fine_tuned_checkpoint,
        ),
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
