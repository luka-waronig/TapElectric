from typing import Protocol
from uuid import UUID

from tap_charger_scan.models import Scan, ScanLabel, VerifiedScan


class ScanRepository(Protocol):
    async def get_by_id(self, scan_id: UUID) -> Scan | None: ...

    async def get_by_client_scan_id(self, client_scan_id: UUID) -> Scan | None: ...

    async def add(self, scan: Scan) -> None: ...

    async def upsert_label(self, label: ScanLabel) -> None: ...

    async def list_verified(self) -> list[VerifiedScan]: ...


class ImageObjectStore(Protocol):
    async def put(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str,
        sha256: str,
    ) -> None: ...

    async def get(self, *, key: str) -> bytes: ...

    async def delete(self, *, key: str) -> None: ...
