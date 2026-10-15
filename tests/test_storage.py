from io import BytesIO

from tap_charger_scan.object_store import S3ImageObjectStore


class StubS3Client:
    def __init__(self) -> None:
        self.put_request: dict[str, object] | None = None
        self.delete_request: dict[str, object] | None = None
        self.get_request: dict[str, object] | None = None

    def put_object(self, **request: object) -> None:
        self.put_request = request

    def delete_object(self, **request: object) -> None:
        self.delete_request = request

    def get_object(self, **request: object) -> dict[str, BytesIO]:
        self.get_request = request
        return {"Body": BytesIO(b"stored image")}


async def test_s3_object_store_forwards_image_metadata() -> None:
    client = StubS3Client()
    store = S3ImageObjectStore(client, "scan-bucket")

    await store.put(
        key="scans/id/image.png",
        data=b"image",
        content_type="image/png",
        sha256="abc123",
    )
    await store.delete(key="scans/id/image.png")
    loaded = await store.get(key="scans/id/image.png")

    assert client.put_request == {
        "Bucket": "scan-bucket",
        "Key": "scans/id/image.png",
        "Body": b"image",
        "ContentType": "image/png",
        "Metadata": {"sha256": "abc123"},
    }
    assert client.delete_request == {
        "Bucket": "scan-bucket",
        "Key": "scans/id/image.png",
    }
    assert loaded == b"stored image"
