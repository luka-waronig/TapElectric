from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

import boto3
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from tap_charger_scan.config import Settings, get_settings
from tap_charger_scan.object_store import S3ImageObjectStore
from tap_charger_scan.repositories import PostgresScanRepository
from tap_charger_scan.schemas import ScanCreated, ScanLabelRequest, ScanLabelStored, ScanMetadata
from tap_charger_scan.service import (
    IdempotencyConflictError,
    ImageTooLargeError,
    InvalidImageError,
    ScanIngestionService,
    ScanNotFoundError,
)


@dataclass(frozen=True)
class ServiceResources:
    service: ScanIngestionService
    engine: AsyncEngine | None = None


def build_service(settings: Settings) -> ServiceResources:
    engine = create_async_engine(settings.database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    repository = PostgresScanRepository(sessions)
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
    )
    object_store = S3ImageObjectStore(s3_client, settings.s3_bucket)
    service = ScanIngestionService(
        repository,
        object_store,
        object_key_prefix=settings.object_key_prefix,
        max_image_bytes=settings.max_image_bytes,
        max_image_pixels=settings.max_image_pixels,
    )
    return ServiceResources(service=service, engine=engine)


def create_app(service: ScanIngestionService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources = (
            ServiceResources(service=service)
            if service is not None
            else build_service(get_settings())
        )
        app.state.ingestion_service = resources.service
        yield
        if resources.engine is not None:
            await resources.engine.dispose()

    application = FastAPI(
        title="Tap Electric charger scan service",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post(
        "/v1/scans",
        response_model=ScanCreated,
        status_code=status.HTTP_201_CREATED,
        tags=["scans"],
    )
    async def create_scan(
        image: Annotated[UploadFile, File(description="Original JPEG or PNG scan image")],
        metadata: Annotated[str, Form(description="JSON-encoded scan metadata")],
        ingestion_service: Annotated[ScanIngestionService, Depends(_get_service)],
    ) -> ScanCreated:
        try:
            parsed_metadata = ScanMetadata.model_validate_json(metadata)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=exc.errors(include_url=False, include_context=False),
            ) from exc

        image_data = await image.read(ingestion_service.max_image_bytes + 1)
        try:
            scan = await ingestion_service.ingest(
                parsed_metadata,
                image_data,
                image.content_type,
            )
        except ImageTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(exc),
            ) from exc
        except InvalidImageError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exc),
            ) from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        return ScanCreated(scan_id=scan.id, image_sha256=scan.image_sha256)

    @application.put(
        "/v1/scans/{scan_id}/label",
        response_model=ScanLabelStored,
        tags=["scans"],
    )
    async def verify_scan_label(
        scan_id: UUID,
        label: ScanLabelRequest,
        ingestion_service: Annotated[ScanIngestionService, Depends(_get_service)],
    ) -> ScanLabelStored:
        try:
            await ingestion_service.verify_label(scan_id, label)
        except ScanNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return ScanLabelStored(scan_id=scan_id)

    return application


def _get_service(request: Request) -> ScanIngestionService:
    return request.app.state.ingestion_service


app = create_app()
