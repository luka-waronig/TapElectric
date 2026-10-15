from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration supplied by the deployment environment."""

    model_config = SettingsConfigDict(env_prefix="TAP_SCAN_", extra="ignore")

    database_url: str
    s3_bucket: str
    s3_endpoint_url: str | None = None
    s3_region: str = "eu-west-1"
    object_key_prefix: str = "scans"
    max_image_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=25_000_000, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
