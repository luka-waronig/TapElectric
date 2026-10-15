# Tap Electric charger scan service

A compact reference implementation for collecting failed EV-charger QR scans, building labeled
datasets, fine-tuning an image-restoration model, and measuring whether restoration improves exact
QR decoding.

This repository supplies the application boundaries and executable experiment framework. It does
not provision PostgreSQL/S3, download datasets, run a full training job, or claim real-world model
improvement without real held-out scans.

## Approach

The deployed pipeline would be:

```text
mobile scan -> FastAPI -> original image in S3 + metadata in PostgreSQL
                              |
                    verified payload/group label
                              |
              grouped train / validation / test manifest
                              |
 degraded QR -> Swin2SR restoration -> OpenCV QR decoder -> exact payload
```

The learned model is not the text decoder. It learns a supervised mapping from a degraded 64×64
QR crop to its clean 128×128 counterpart. OpenCV then performs deterministic QR detection and
payload extraction. The restored image is useful only if it increases exact charger-payload
recovery.

The selected checkpoint is
[`caidas/swin2SR-classical-sr-x2-64`](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64),
pinned to a revision in code. Its x2 classical-super-resolution objective and 64-pixel training
patch size match the synthetic input/target pairs without an invented resize workaround. It is a
general restoration starting point, not a QR-specific model; QR-specific behavior comes from
fine-tuning on paired QR examples and must be validated by downstream decode accuracy.

## Repository layout

```text
src/tap_charger_scan/
  api.py             FastAPI routes and dependency wiring
  schemas.py         request/response contracts
  models.py          domain records and SQLAlchemy tables
  ports.py           repository and object-store interfaces
  repositories.py    PostgreSQL adapter
  object_store.py    S3-compatible adapter
  service.py         ingestion and verified-label use cases
  baseline.py        OpenCV decoder with one CLAHE fallback
  dataset.py         manifest, grouped splits, real-scan export path
  synthetic.py       deterministic paired QR generator
  restoration.py     pinned Hugging Face Swin2SR wrapper
  train.py           PyTorch fine-tuning workflow
  evaluate.py        held-out pipeline comparison
tests/                small offline component tests
```

PostgreSQL and object storage are separate adapters even though infrastructure and migrations are
outside this assessment. The service layer depends only on their protocols, which keeps API tests
in-memory and makes either adapter replaceable.

## Data collection API

`POST /v1/scans` accepts `multipart/form-data`:

- `image`: original JPEG or PNG scan.
- `metadata`: JSON with an idempotency key, capture time, and optional location/device context.

```json
{
  "client_scan_id": "9d4b2e2d-f9c8-4968-9fe3-8091e3fd36a4",
  "captured_at": "2026-10-15T10:30:00Z",
  "location": {
    "latitude": 52.3676,
    "longitude": 4.9041,
    "accuracy_m": 12.5
  },
  "device_model": "Example Phone"
}
```

The endpoint validates the image, calculates its checksum, stores the original bytes, and writes
the scan metadata. Repeating a request with the same `client_scan_id` and image is idempotent;
reusing the ID for different bytes returns `409 Conflict`.

`PUT /v1/scans/{scan_id}/label` associates reviewed ground truth:

```json
{
  "expected_payload": "tap://charger/NL-TAP-E123",
  "group_id": "NL-TAP-E123",
  "source": "charger_registry",
  "reference_image_key": null
}
```

`expected_payload` is the exact decoder truth. `group_id` is a stable sticker/charger identity used
to prevent related images crossing data splits. `reference_image_key` is optional: a verified
payload alone supports evaluation, while pixel-supervised restoration training needs a clean,
geometrically aligned reference image. In production, the label route would be an authenticated
internal review/registry integration rather than a public mobile endpoint.

## Baseline and data

The baseline uses only OpenCV `QRCodeDetector.detectAndDecode`. It tries the original image and then
one CLAHE grayscale fallback. CLAHE can recover local contrast under uneven illumination, but may
amplify sensor noise; the held-out evaluation, not visual appearance, decides whether it helps.
See the [OpenCV QRCodeDetector API](https://docs.opencv.org/4.x/de/dc3/classcv_1_1QRCodeDetector.html).

The synthetic generator creates QR codes with known payloads, then applies a small set of relevant
degradations: downsampling, Gaussian blur, sensor-like noise, and an uneven lighting gradient. It
assigns each payload/group to train, validation, or test **before** creating variants, so no version
of the same QR sticker leaks across splits. Synthetic data demonstrates the workflow; it does not
replace real scan validation.

The paired synthetic experiment assumes the QR has already been localized as a crop. The API keeps
the original image; a production data-preparation step would use detected corner points or reviewed
annotations to create aligned crops rather than silently stretching an entire camera frame.

`dataset.export_verified_scans(...)` materializes labeled images from the repository/object store
into the same JSONL manifest. Thus future real scans require no evaluator change. Records without a
clean reference remain evaluation-only; aligned reference pairs can also enter restoration
training after satisfying the documented 64×64 input and 128×128 target contract.

## Setup and demonstration

Python 3.11 or newer is required. `pyproject.toml` is the single dependency/configuration source.

Create an environment and install the API, development tools, and optional ML workflow:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,ml]'
```

Run all offline checks. These tests do not download model weights or require PostgreSQL/S3:

```bash
ruff check .
ruff format --check .
pytest
```

Run a small end-to-end smoke flow:

```bash
tap-generate-data data/smoke --payload-count 6 --variants-per-payload 1 --seed 42

# Downloads and evaluates the pinned pretrained checkpoint against OpenCV.
tap-evaluate data/smoke/manifest.jsonl \
  --output artifacts/pretrained-evaluation.json

# Performs one genuine parameter-update step and writes a reloadable checkpoint.
tap-train data/smoke/manifest.jsonl artifacts/swin2sr-qr-smoke \
  --epochs 1 --batch-size 1 --max-steps 1 --device cpu

# Compares OpenCV, pretrained Swin2SR, and the saved fine-tuned checkpoint.
tap-evaluate data/smoke/manifest.jsonl \
  --fine-tuned-checkpoint artifacts/swin2sr-qr-smoke \
  --output artifacts/fine-tuned-evaluation.json
```

This smoke dataset is intentionally tiny and its metrics are execution evidence, not a claim of
model quality. A defensible model result requires a suitably sized, real, verified held-out set.
Generated `data/`, `artifacts/`, model weights, and local download caches are ignored by Git.

The API can be installed without ML dependencies using `python -m pip install -e '.[dev]'`. To run
it, configure existing external services:

```bash
export TAP_SCAN_DATABASE_URL='postgresql+asyncpg://user:password@localhost/tap_scans'
export TAP_SCAN_S3_BUCKET='tap-charger-scans'
export TAP_SCAN_S3_ENDPOINT_URL='http://localhost:9000'  # optional for S3-compatible stores
uvicorn tap_charger_scan.api:app
```

The SQLAlchemy models describe the expected tables; schema migrations and infrastructure are
deliberately not created here. Importing or running the API does not load OpenCV, PyTorch,
Transformers, or model weights.

The report compares:

- OpenCV on the collected/degraded image;
- pretrained Swin2SR followed by the same OpenCV decoder;
- fine-tuned Swin2SR followed by the same OpenCV decoder.

For each system it reports exact payload accuracy, incorrect-payload count/rate, no-decode
count/rate, and p50/p95 end-to-end latency. A candidate should improve exact accuracy without an
unacceptable rise in wrong payloads or latency. Final model selection must use real, verified,
held-out charger groups and should report uncertainty over a sufficiently large sample.

## Automated checks

The tests cover ingestion/idempotency, label association, S3 adapter behavior, clean OpenCV
decoding, grouped split isolation, synthetic dimensions, real-scan manifest export, evaluation
metrics, and one PyTorch optimization step using an injected tiny model. They do not download a
checkpoint or require PostgreSQL/S3.

```bash
ruff check .
ruff format --check .
pytest
```

## Deliberate omissions

For a production system, add migrations, authentication/authorization, malware/content scanning,
encryption and retention controls, background queues, observability, dataset/model versioning,
approval gates, and scalable distributed training. Multiple QR libraries, large augmentation
catalogues, OCR, and QR-specific detection networks are reasonable later experiments, but are not
needed to establish this assignment's clean baseline and improvement loop.
