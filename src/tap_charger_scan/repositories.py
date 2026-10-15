from dataclasses import asdict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tap_charger_scan.models import (
    Scan,
    ScanLabel,
    ScanLabelRow,
    ScanRow,
    VerifiedScan,
)


class PostgresScanRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_by_id(self, scan_id: UUID) -> Scan | None:
        async with self._session_factory() as session:
            row = await session.get(ScanRow, scan_id)
            return _to_scan(row) if row else None

    async def get_by_client_scan_id(self, client_scan_id: UUID) -> Scan | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ScanRow).where(ScanRow.client_scan_id == client_scan_id)
            )
            row = result.scalar_one_or_none()
            return _to_scan(row) if row else None

    async def add(self, scan: Scan) -> None:
        async with self._session_factory() as session:
            session.add(ScanRow(**asdict(scan)))
            await session.commit()

    async def upsert_label(self, label: ScanLabel) -> None:
        async with self._session_factory() as session:
            row = await session.get(ScanLabelRow, label.scan_id)
            values = asdict(label)
            if row is None:
                session.add(ScanLabelRow(**values))
            else:
                for name, value in values.items():
                    if name != "scan_id":
                        setattr(row, name, value)
            await session.commit()

    async def list_verified(self) -> list[VerifiedScan]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ScanRow, ScanLabelRow).join(ScanLabelRow, ScanLabelRow.scan_id == ScanRow.id)
            )
            return [
                VerifiedScan(
                    scan_id=scan.id,
                    image_key=scan.image_key,
                    expected_payload=label.expected_payload,
                    group_id=label.group_id,
                    reference_image_key=label.reference_image_key,
                )
                for scan, label in result.all()
            ]


def _to_scan(row: ScanRow) -> Scan:
    return Scan(
        id=row.id,
        client_scan_id=row.client_scan_id,
        captured_at=row.captured_at,
        image_key=row.image_key,
        image_sha256=row.image_sha256,
        image_content_type=row.image_content_type,
        image_size_bytes=row.image_size_bytes,
        image_width=row.image_width,
        image_height=row.image_height,
        latitude=row.latitude,
        longitude=row.longitude,
        location_accuracy_m=row.location_accuracy_m,
        device_model=row.device_model,
        created_at=row.created_at,
    )
