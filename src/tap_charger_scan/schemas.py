from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScanLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy_m: float = Field(gt=0)


class ScanMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_scan_id: UUID
    captured_at: datetime
    location: ScanLocation | None = None
    device_model: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("captured_at")
    @classmethod
    def captured_at_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must include a timezone")
        return value


class ScanCreated(BaseModel):
    scan_id: UUID
    image_sha256: str
    status: Literal["stored"] = "stored"


class ScanLabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_payload: str = Field(min_length=1, max_length=2048)
    group_id: str = Field(
        min_length=1,
        max_length=200,
        description="Stable charger/sticker identity used to keep splits independent",
    )
    source: Literal["charger_registry", "reference_asset", "manual_review"]
    reference_image_key: str | None = Field(default=None, min_length=1, max_length=500)


class ScanLabelStored(BaseModel):
    scan_id: UUID
    status: Literal["verified"] = "verified"
