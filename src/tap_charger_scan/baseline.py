from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class DecodeResult:
    payload: str | None
    detected: bool
    preprocessing: Literal["original", "clahe"] | None
    latency_ms: float


class OpenCVQRDecoder:
    """OpenCV baseline with one targeted fallback for uneven lighting."""

    def __init__(self) -> None:
        self._detector = cv2.QRCodeDetector()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def decode(self, image: Image.Image | np.ndarray) -> DecodeResult:
        started = perf_counter()
        bgr = _as_bgr(image)

        payload, points, _ = self._detector.detectAndDecode(bgr)
        if payload:
            return DecodeResult(payload, True, "original", _elapsed_ms(started))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        fallback_payload, fallback_points, _ = self._detector.detectAndDecode(enhanced)
        return DecodeResult(
            payload=fallback_payload or None,
            detected=points is not None or fallback_points is not None,
            preprocessing="clahe" if fallback_payload else None,
            latency_ms=_elapsed_ms(started),
        )


def _as_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a grayscale or three-channel image")
    return image


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000
