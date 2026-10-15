import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from uuid import UUID, uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

from tap_charger_scan.api import create_app
from tap_charger_scan.models import Scan, ScanLabel, VerifiedScan
from tap_charger_scan.service import ScanIngestionService


class InMemoryScanRepository:
    def __init__(self) -> None:
        self.scans: dict[UUID, Scan] = {}
        self.labels: dict[UUID, ScanLabel] = {}

    async def get_by_id(self, scan_id: UUID) -> Scan | None:
        return next((scan for scan in self.scans.values() if scan.id == scan_id), None)

    async def get_by_client_scan_id(self, client_scan_id: UUID) -> Scan | None:
        return self.scans.get(client_scan_id)

    async def add(self, scan: Scan) -> None:
        self.scans[scan.client_scan_id] = scan

    async def upsert_label(self, label: ScanLabel) -> None:
        self.labels[label.scan_id] = label

    async def list_verified(self) -> list[VerifiedScan]:
        return []


class InMemoryImageStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, *, key: str, data: bytes, content_type: str, sha256: str) -> None:
        self.objects[key] = data

    async def delete(self, *, key: str) -> None:
        self.objects.pop(key, None)

    async def get(self, *, key: str) -> bytes:
        return self.objects[key]


def make_png(color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", (16, 12), color=color).save(output, format="PNG")
    return output.getvalue()


def metadata(client_scan_id: UUID) -> str:
    return json.dumps(
        {
            "client_scan_id": str(client_scan_id),
            "captured_at": "2026-10-15T10:30:00Z",
            "location": {
                "latitude": 52.3676,
                "longitude": 4.9041,
                "accuracy_m": 12.5,
            },
            "device_model": "Test Phone",
        }
    )


def build_app() -> tuple[FastAPI, InMemoryScanRepository, InMemoryImageStore]:
    repository = InMemoryScanRepository()
    object_store = InMemoryImageStore()
    service = ScanIngestionService(repository, object_store)
    return create_app(service), repository, object_store


@asynccontextmanager
async def api_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_create_scan_stores_image_and_metadata() -> None:
    app, repository, object_store = build_app()
    client_scan_id = uuid4()
    image = make_png()

    async with api_client(app) as client:
        response = await client.post(
            "/v1/scans",
            data={"metadata": metadata(client_scan_id)},
            files={"image": ("scan.png", image, "image/png")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "stored"
    assert UUID(body["scan_id"])
    assert len(repository.scans) == 1
    stored_scan = repository.scans[client_scan_id]
    assert stored_scan.image_width == 16
    assert stored_scan.image_height == 12
    assert stored_scan.latitude == 52.3676
    assert object_store.objects[stored_scan.image_key] == image


async def test_create_scan_is_idempotent_for_the_same_image() -> None:
    app, repository, object_store = build_app()
    client_scan_id = uuid4()
    request = {
        "data": {"metadata": metadata(client_scan_id)},
        "files": {"image": ("scan.png", make_png(), "image/png")},
    }

    async with api_client(app) as client:
        first = await client.post("/v1/scans", **request)
        second = await client.post("/v1/scans", **request)

    assert first.status_code == second.status_code == 201
    assert first.json()["scan_id"] == second.json()["scan_id"]
    assert len(repository.scans) == 1
    assert len(object_store.objects) == 1


async def test_reusing_client_scan_id_for_different_image_returns_conflict() -> None:
    app, _, _ = build_app()
    client_scan_id = uuid4()

    async with api_client(app) as client:
        first = await client.post(
            "/v1/scans",
            data={"metadata": metadata(client_scan_id)},
            files={"image": ("scan.png", make_png("white"), "image/png")},
        )
        second = await client.post(
            "/v1/scans",
            data={"metadata": metadata(client_scan_id)},
            files={"image": ("scan.png", make_png("black"), "image/png")},
        )

    assert first.status_code == 201
    assert second.status_code == 409


async def test_invalid_image_is_rejected() -> None:
    app, _, _ = build_app()

    async with api_client(app) as client:
        response = await client.post(
            "/v1/scans",
            data={"metadata": metadata(uuid4())},
            files={"image": ("scan.png", b"not an image", "image/png")},
        )

    assert response.status_code == 415


async def test_verified_label_is_associated_with_a_scan() -> None:
    app, repository, _ = build_app()

    async with api_client(app) as client:
        created = await client.post(
            "/v1/scans",
            data={"metadata": metadata(uuid4())},
            files={"image": ("scan.png", make_png(), "image/png")},
        )
        scan_id = UUID(created.json()["scan_id"])
        response = await client.put(
            f"/v1/scans/{scan_id}/label",
            json={
                "expected_payload": "tap://charger/NL-TAP-E123",
                "group_id": "NL-TAP-E123",
                "source": "charger_registry",
            },
        )

    assert response.status_code == 200
    assert repository.labels[scan_id].expected_payload == "tap://charger/NL-TAP-E123"
