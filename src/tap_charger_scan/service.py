import hashlib
from datetime import UTC, datetime
from io import BytesIO
from uuid import UUID, uuid4

from PIL import Image, UnidentifiedImageError

from tap_charger_scan.models import Scan, ScanLabel
from tap_charger_scan.ports import ImageObjectStore, ScanRepository
from tap_charger_scan.schemas import ScanLabelRequest, ScanMetadata


class InvalidImageError(ValueError):
    pass


class ImageTooLargeError(ValueError):
    pass


class IdempotencyConflictError(ValueError):
    pass


class ScanNotFoundError(ValueError):
    pass


class ScanIngestionService:
    def __init__(
        self,
        repository: ScanRepository,
        object_store: ImageObjectStore,
        *,
        object_key_prefix: str = "scans",
        max_image_bytes: int = 10 * 1024 * 1024,
        max_image_pixels: int = 25_000_000,
    ) -> None:
        self._repository = repository
        self._object_store = object_store
        self._object_key_prefix = object_key_prefix.strip("/")
        self.max_image_bytes = max_image_bytes
        self._max_image_pixels = max_image_pixels

    async def ingest(
        self,
        metadata: ScanMetadata,
        image_data: bytes,
        declared_content_type: str | None,
    ) -> Scan:
        if not image_data:
            raise InvalidImageError("image is empty")
        if len(image_data) > self.max_image_bytes:
            raise ImageTooLargeError("image exceeds the configured size limit")

        content_type, extension, width, height = self._inspect_image(image_data)
        accepted_declared_types = {content_type}
        if content_type == "image/jpeg":
            accepted_declared_types.add("image/jpg")
        if declared_content_type not in accepted_declared_types:
            raise InvalidImageError("declared content type does not match the image")

        image_sha256 = hashlib.sha256(image_data).hexdigest()
        existing = await self._repository.get_by_client_scan_id(metadata.client_scan_id)
        if existing:
            if existing.image_sha256 != image_sha256:
                raise IdempotencyConflictError(
                    "client_scan_id has already been used for a different image"
                )
            return existing

        scan_id = uuid4()
        image_key = f"{self._object_key_prefix}/{scan_id}/{image_sha256}.{extension}"
        location = metadata.location
        scan = Scan(
            id=scan_id,
            client_scan_id=metadata.client_scan_id,
            captured_at=metadata.captured_at,
            image_key=image_key,
            image_sha256=image_sha256,
            image_content_type=content_type,
            image_size_bytes=len(image_data),
            image_width=width,
            image_height=height,
            latitude=location.latitude if location else None,
            longitude=location.longitude if location else None,
            location_accuracy_m=location.accuracy_m if location else None,
            device_model=metadata.device_model,
            created_at=datetime.now(UTC),
        )

        await self._object_store.put(
            key=image_key,
            data=image_data,
            content_type=content_type,
            sha256=image_sha256,
        )
        try:
            await self._repository.add(scan)
        except Exception:
            await self._object_store.delete(key=image_key)
            raise
        return scan

    async def verify_label(self, scan_id: UUID, request: ScanLabelRequest) -> ScanLabel:
        if await self._repository.get_by_id(scan_id) is None:
            raise ScanNotFoundError("scan was not found")

        label = ScanLabel(
            scan_id=scan_id,
            expected_payload=request.expected_payload,
            group_id=request.group_id,
            source=request.source,
            reference_image_key=request.reference_image_key,
            verified_at=datetime.now(UTC),
        )
        await self._repository.upsert_label(label)
        return label

    def _inspect_image(self, image_data: bytes) -> tuple[str, str, int, int]:
        try:
            with Image.open(BytesIO(image_data)) as image:
                image_format = image.format
                width, height = image.size
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("image is not a valid JPEG or PNG") from exc

        formats = {
            "JPEG": ("image/jpeg", "jpg"),
            "PNG": ("image/png", "png"),
        }
        if image_format not in formats:
            raise InvalidImageError("only JPEG and PNG images are supported")
        if width * height > self._max_image_pixels:
            raise ImageTooLargeError("image dimensions exceed the configured pixel limit")

        content_type, extension = formats[image_format]
        return content_type, extension, width, height
