from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, String, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


@dataclass(frozen=True, slots=True)
class Scan:
    id: UUID
    client_scan_id: UUID
    captured_at: datetime
    image_key: str
    image_sha256: str
    image_content_type: str
    image_size_bytes: int
    image_width: int
    image_height: int
    latitude: float | None
    longitude: float | None
    location_accuracy_m: float | None
    device_model: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ScanLabel:
    scan_id: UUID
    expected_payload: str
    group_id: str
    source: str
    reference_image_key: str | None
    verified_at: datetime


@dataclass(frozen=True, slots=True)
class VerifiedScan:
    scan_id: UUID
    image_key: str
    expected_payload: str
    group_id: str
    reference_image_key: str | None


class Base(DeclarativeBase):
    pass


class ScanRow(Base):
    __tablename__ = "scans"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    client_scan_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), unique=True, nullable=False, index=True
    )
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    image_key: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    image_content_type: Mapped[str] = mapped_column(String(20), nullable=False)
    image_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    image_width: Mapped[int] = mapped_column(nullable=False)
    image_height: Mapped[int] = mapped_column(nullable=False)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    location_accuracy_m: Mapped[float | None] = mapped_column(Float)
    device_model: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ScanLabelRow(Base):
    __tablename__ = "scan_labels"

    scan_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("scans.id", ondelete="CASCADE"),
        primary_key=True,
    )
    expected_payload: Mapped[str] = mapped_column(String(2048), nullable=False)
    group_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    reference_image_key: Mapped[str | None] = mapped_column(String(500))
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
