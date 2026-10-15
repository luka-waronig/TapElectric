Exported conversation with Codex:

# Plan QR scan training service

> For the purposes of a technical interview assessment, here I've created a new folder in which we will be building out a  reference application repo for the task that I'm about to explain to you. I want to first discuss and plan the layout of the repo structure and implementation, and then I will let you code it. Here is the TASK: In the Tap Electric app there is a feature called “scan a charger”. When you click this button in our app, your phone camera opens in the app. You point the camera at the QR sticker and our app will capture this. This QR sticker also contains some IDs, so with the QR sticker, the location of the EV-driver, and the other information of this sticker, we are normally able to find the right charger for the customer. However, there are sometimes issues, like the camera not getting a sharp image, too dark, too light, the sticker is vading etc. Currently we can’t help the driver in those cases.
>
> We need to create a service that can be used to train a model for the improvement of the scanning of the QR stickers. It should contain the following:
>
> - API endpoint to store the data. Every scan will be pushed towards this endpoint, including useful information. You can design this endpoint and determine what information you want to receive.
> - Determine what kind of database(s) you want to use. Don’t implement the real infrastructure, just write the code as if the database(s) exist.
> - A model that we can use to test and improve our image to text generation (you can find many open source models on https://huggingface.co/models). Select one and implement it inside your code base.
> - A system inside this service that can train and improve your selected model based on the collected data
> - Think about how to validate your trained model.
>
> We don’t need to actually test this with real data for this assignment right now, but build the right foundations and logic. We will discuss design, code, model selection and model improvement. Use the language you think is best suited for this job (Python I would assume). Now, what is really important is that we are clean, precise, and direct in our implementation, so no weird workarounds, hardcoding, and overcomplicating. I want to keep this simple and interpretable. And before we start doing anything, I would really like you to explain to me a little bit more the scope of what you think is best we do here ( i.e. i know that for production we would maybe use postgreSQL for IDs and metadata and S3 object storage for the image files, so we should pretend they exist) . And I guess the most important thing I need your help with right now to begin with is distinguishing this essential functionality for this repository versus any further optional enhancements and improvements that we could add onto it that go beyond and more in depth.  Of course we separate ingestion and training, and of course we should have a requirements.txt  and a README .  And since I'm not familiar with necessarily decoding QR scans, you can find out and tell me what kind of baseline models would be good to try out first before a machine learning model, and then what kind of new, relevant perhaps vision transformer methods there might exist for this. You do a little bit of research and you come up with what would be an appropriate solution for this. And based on the instructions, I assume we should mainly, like first begin with implementing the FastAPI endpoint. Or well, I assume FastAPI. You tell me if you think something else is better. So yeah, just before writing any code, let's walk through the problem a bit more. Especially, I need some more clarification on, since there are different requirements here when I think about this pipeline, and the images of the stickers with the QR codes, and then the actual information about what/where charger that is, and what is the ground truth attached to it (like a clear QR code), so that we can actually decode the scanned image, right? So let's just walk through those requirements a little bit more in detail to begin with. Okay, so go ahead. Before writing any files or installing anything, please give me the overall architectural recommendation, like we just discussed, and identify any important trade-offs that I need to be aware of, and any assumptions that we should not forget about, etc.

<details><summary>4 previous messages</summary>

> I’ll keep this at the architecture-and-research stage—no files, installs, or repo changes. I’m using the OpenAI docs skill only to check whether any Codex-specific repository conventions matter here; for the QR/model choices I’ll rely on primary project documentation and papers.
>
> <details><summary>Searched the web</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `~/.codex/skills/.system/openai-docs/SKILL.md`
>
> </details>
>
> Searched the web for `OpenCV QRCodeDetector official documentation WeChatQRCode official docs ...`
>
> </details>
>
> The core distinction is already clear: this should not begin as “OCR for QR codes.” A QR payload is arbitrary encoded data with built-in geometry and error correction, so the first system should be a deterministic decoder plus controlled preprocessing; machine learning should improve recovery of the symbol, not invent the payload. I’m checking which restoration architecture gives us the cleanest fine-tuning story.
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `site:huggingface.co SegFormer model documentation nvidia mit-b0 official ...`
>
> Searched the web for `site:huggingface.co Restormer image deblurring model ...`
>
> Searched the web for `site:fastapi.tiangolo.com FastAPI features OpenAPI Pydantic official ...`
>
> Searched the web for `https://huggingface.co/caidas/swin2SR-classical-sr-x2-64`
>
> </details>
</details>

The best foundation is a small Python service with three clearly separated concerns:

1. Ingestion: FastAPI receives and durably stores scan attempts.
2. Decoding and labeling: deterministic QR decoders establish the baseline and produce candidate results.
3. Training and evaluation: an offline pipeline improves difficult images and measures whether decoding actually improves.

The most important design decision is this:

> A QR code is not ordinary text. We should not ask a generative vision model to “read” arbitrary QR payloads directly.

QR payloads may contain URLs, identifiers, checksums, or arbitrary bytes. A generative OCR/VLM can hallucinate a plausible but incorrect character, which is substantially worse than returning “unable to decode.” The machine-learning model should restore or normalize the QR image; a deterministic QR decoder should still produce the payload.

## Recommended architecture

```text
Tap Electric app
    |
    | POST /v1/scans
    v
FastAPI ingestion
    |--------------------> PostgreSQL
    |                       scan metadata, status,
    |                       labels, decoder runs,
    |                       dataset/model versions
    |
    `--------------------> S3-compatible storage
                            original images,
                            derived crops,
                            dataset manifests,
                            model artifacts

Offline processing
    |
    |--> deterministic QR baseline
    |--> payload and charger-registry validation
    |--> labeling / dataset snapshot
    |--> model training
    `--> held-out evaluation
```

I would keep these as separate Python modules and separately runnable processes, but not create multiple repositories or microservices. One cohesive repository is much easier to explain and review.

## FastAPI is the right choice

FastAPI fits this assessment particularly well because:

- The training code will already be in Python.
- Pydantic gives us a precise, documented input schema.
- Multipart image uploads are straightforward.
- It generates OpenAPI and interactive API documentation automatically.
- Dependencies such as the database and object store can be injected and replaced with fakes in tests.

Those are first-class FastAPI capabilities documented in its [official feature overview](https://fastapi.tiangolo.com/features/). Flask would require more manual validation and API documentation, while Django would introduce more framework than this service needs.

## The ingestion contract

I recommend:

```text
POST /v1/scans
Content-Type: multipart/form-data

image:      original JPEG or PNG
metadata:   JSON document
```

The metadata should include:

- `client_scan_id`: UUID used for idempotency.
- `captured_at`: UTC timestamp.
- `app_version`, `platform`, and OS version.
- Device and camera model where available.
- Image width, height, orientation, and encoding.
- Camera settings when the mobile SDK exposes them:
  - ISO
  - exposure duration
  - focus state/distance
  - zoom
  - torch/flash state
- Optional latitude, longitude, and horizontal accuracy.
- On-device decoder name, version, result, and failure type.
- Detected QR bounding box, if the client found one.
- Retry number or scan-session identifier.
- Candidate location/charger IDs already considered by the app.
- Consent/data-policy version.

The server should independently calculate:

- SHA-256 of the original bytes.
- Actual MIME type and dimensions.
- Basic quality measurements such as brightness, contrast, blur score, and saturation/clipping.
- Storage object key and ingestion status.

A successful request should return `201 Created` only after the original image and metadata are durable. The image should not be decoded synchronously as part of this endpoint.

Because PostgreSQL and S3 cannot share a transaction, the scan record should have a small state machine such as `uploading → stored` or `uploading → failed`. On retry, the same idempotency key and checksum can safely resume the operation.

## Storage choices

PostgreSQL and S3-compatible object storage are sufficient. No vector database or document database is warranted.

PostgreSQL would contain approximately these entities:

- `scan_attempt`
  - identifiers, timestamps, object key, checksum, dimensions, location, camera metadata, ingestion status.
- `decoder_run`
  - scan, pipeline version, preprocessing variant, decoder, result, bounding box, runtime, and error category.
- `scan_label`
  - expected payload, canonical charger/EVSE ID, provenance, confidence, reviewer, and verification status.
- `training_run`
  - dataset manifest, base model revision, configuration, Git commit, seed, artifact URI, and metrics.
- `model_version`
  - promoted artifact and evaluation summary.

S3 would hold:

- Immutable original uploads.
- Derived QR crops and rectified images.
- Clean reference sticker assets where available.
- Immutable training/evaluation manifests.
- Model checkpoints and final exported artifacts.

The payload itself may contain a token or sensitive URL. We should not log it, and it may need application-level encryption at rest.

## Ground truth: the most important data issue

There are three different pieces of information that must not be conflated:

- The QR payload: the exact bytes or text encoded in the symbol.
- The charger identity: the canonical charger, connector, or EVSE the payload resolves to.
- Context: GPS and nearby charger candidates.

GPS is useful for resolving or ranking candidates, but it is not reliable ground truth. A nearby charger may not be the sticker being photographed.

The strongest ground-truth sources are, in order:

1. The original digital sticker/QR asset from the system that generated the label.
2. The exact expected payload from the authoritative charger registry, validated against the physical sticker.
3. A successful independent scan that passes payload-schema and registry validation.
4. A manually reviewed operator record.
5. Location or a user-selected charger, which should remain a weak/provisional label.

A subtle point: knowing only the payload does not necessarily reproduce the exact same QR bitmap. QR version, error-correction level, encoding mode, mask selection, quiet zone, and artwork can differ. Therefore:

- Exact payload is sufficient for end-to-end evaluation.
- A canonical clean image is useful for synthetic training.
- Pixel-aligned real-image restoration ideally needs the original print asset and generation parameters.

If Tap Electric can obtain the original QR artwork from the charger/sticker provisioning system, it will significantly improve the training options.

I would model labels explicitly as `provisional` or `verified`, with a provenance field. A model prediction must never silently become its own training label.

## Baseline before custom machine learning

The first benchmark should be a bounded deterministic pipeline:

1. Apply image orientation.
2. Attempt decoding on the untouched original.
3. Detect and perspective-rectify a QR crop.
4. Try a small, documented set of transformations:
   - grayscale
   - gamma correction
   - CLAHE
   - adaptive thresholding
   - inversion
   - 2× interpolation
   - mild sharpening
5. Decode each candidate.
6. Validate the decoded payload against the expected Tap Electric schema and charger registry.

I would use:

- [ZXing-C++](https://github.com/zxing-cpp/zxing-cpp) through its Python bindings as the primary decoder. It has maintained QR support, Python bindings, and exposes format/checksum failures.
- OpenCV’s [WeChat QRCode detector](https://github.com/opencv/opencv_zoo/tree/main/models/qrcode_wechatqrcode) as an independent fallback.
- OpenCV `QRCodeDetector` as another inexpensive comparison.

Running multiple finite variants is acceptable. Generating dozens of arbitrary filters until something returns a string would be difficult to interpret and increases false-positive risk.

Every output should be classified as:

- no QR detected
- detected but not decoded
- checksum/format failure
- decoded but rejected by business validation
- verified decode

## Recommended model

For this repository, I would select:

[`caidas/swin2SR-classical-sr-x2-64`](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64)

Swin2SR is a Swin Transformer V2 image-restoration and super-resolution architecture with direct Hugging Face Transformers support. Its model interface is documented by [Hugging Face](https://huggingface.co/docs/transformers/model_doc/swin2sr).

Why this is an appropriate first model:

- It improves an image rather than generating a plausible payload.
- It has a clean Hugging Face loading and fine-tuning path.
- The 2× version is less aggressive than 4× super-resolution.
- It can operate on the QR crop rather than a full phone image.
- It lets us compare the original and restored images through the same deterministic decoder.

It should be treated as a challenger branch:

```text
original crop ────────────────> deterministic decoder
       |
       `─> Swin2SR restored crop ─> deterministic decoder
```

We retain whichever result passes strict validation. The model-enhanced image never replaces a successful original decode.

Other relevant approaches include:

- [Restormer](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html), which is especially relevant for deblurring and denoising.
- A recent QR-specific research direction, [Edge-Guided Restormer](https://arxiv.org/abs/2510.12098), which optimizes restoration around QR edge structure and decoding rate. It is interesting follow-up work, but too specialized and immature to make the foundation of a clean assessment repository.
- Diffusion restoration models, which I would avoid here because latency, complexity, and hallucinated structure are bad trade-offs for exact identifiers.

If the stickers also contain a human-readable printed charger ID, that is a separate OCR task. In that case, [`microsoft/trocr-base-printed`](https://huggingface.co/microsoft/trocr-base-printed) is a sensible optional transformer for cropped text lines. TrOCR should not be used to decode the QR matrix itself.

## Training strategy

The initial training data should combine synthetic pairs and verified real scans.

Synthetic data is especially valuable because we can create unlimited exact pairs:

1. Generate known QR codes from representative Tap Electric payloads.
2. Render them with appropriate versions and error-correction levels.
3. Apply realistic degradations:
   - motion and defocus blur
   - perspective distortion
   - distance/downsampling
   - JPEG compression
   - under/overexposure
   - shadows and glare
   - noise
   - fading, scratches, dirt, and partial occlusion
4. Train the model to reconstruct the clean image.

Real scans should then be used for fine-tuning only when there is a trustworthy clean reference or alignment. Payload-only real examples remain extremely valuable for evaluation even when they cannot support pixel reconstruction loss.

Every training run should use an immutable dataset manifest recording scan IDs and label versions. Splits must be grouped by physical sticker or charger, so multiple photographs of the same QR code cannot leak across training and test sets.

## Validation

The primary metric should not be PSNR, SSIM, or how attractive the enhanced image looks. It should be exact end-to-end decode behavior.

Report at least:

- Exact raw-payload decode rate.
- Canonical charger-resolution rate.
- Recovery rate on images the baseline could not decode.
- Regression rate on images the baseline decoded successfully.
- Wrong-but-valid payload rate.
- Abstention/failure rate.
- P50/P95 inference latency and memory.
- Results by failure slice:
  - blur
  - low light
  - glare
  - faded/damaged sticker
  - device family
  - QR size in pixels
  - perspective severity
  - error-correction level

The most important safety metric is the wrong-payload rate. In this domain, a confident incorrect charger is usually worse than asking the driver to retry or select manually.

## Essential versus optional scope

| Essential for the assessment | Valuable later |
|---|---|
| FastAPI ingestion endpoint | Real-time decode endpoint |
| Typed request and response schemas | Queue/Celery/Kafka processing |
| PostgreSQL metadata models | Full infrastructure-as-code |
| S3 storage adapter | Labeling web UI |
| Idempotent, validated uploads | Multi-frame/burst fusion |
| ZXing/OpenCV baseline | Active learning |
| Explicit label provenance | MLflow/DVC/model registry platform |
| Swin2SR adapter and training job | On-device Core ML/TFLite export |
| Synthetic degradation generator | Shadow deployment and A/B testing |
| Reproducible dataset splitting | Automated drift monitoring |
| End-to-end evaluation script | Automatic retraining schedules |
| Unit and integration tests | OCR branch for printed sticker text |

Authentication, retention controls, encrypted storage, and auditability are production necessities, but the assessment implementation can expose clean interfaces and document them without building the surrounding identity or infrastructure platform.

## Proposed repository layout

```text
.
├── README.md
├── requirements.txt
├── requirements-ml.txt
├── requirements-dev.txt
├── .env.example
├── configs/
│   └── training.yaml
├── migrations/
├── src/
│   └── charger_scan/
│       ├── config.py
│       ├── api/
│       │   ├── main.py
│       │   ├── schemas.py
│       │   └── scans.py
│       ├── db/
│       │   ├── models.py
│       │   ├── repositories.py
│       │   └── session.py
│       ├── storage/
│       │   └── s3.py
│       ├── decoding/
│       │   ├── pipeline.py
│       │   ├── preprocessing.py
│       │   ├── zxing_decoder.py
│       │   ├── opencv_decoder.py
│       │   └── payload_validation.py
│       └── training/
│           ├── dataset.py
│           ├── augmentations.py
│           ├── model.py
│           ├── train.py
│           └── evaluate.py
└── tests/
    ├── unit/
    └── integration/
```

The ingestion process would use only `requirements.txt`; PyTorch and Transformers would live in `requirements-ml.txt`. That prevents the API container from unnecessarily carrying the full training stack.

## Assumptions to confirm before implementation

- Are the QR stickers static, or can their payloads expire or rotate?
- Can we access the original digital sticker assets and exact payloads?
- Does the sticker have a printed human-readable charger/EVSE identifier?
- Is a single image uploaded, or can the app provide a short burst of frames?
- What payload formats and domains are considered valid?
- Can the mobile app provide camera and failed-decoder metadata?
- What retention and consent rules apply to images and precise driver locations?
- Is the initial service only for data collection/training, or must it eventually participate in the live scan path?

The first coding slice should indeed be the FastAPI ingestion path—but as a complete vertical slice: request validation, idempotency, PostgreSQL record, S3 upload, clean error handling, and tests with fake repositories. Immediately after that, I would build the deterministic decoder benchmark. Training a transformer before measuring that baseline would give us no trustworthy definition of improvement.

No files or dependencies have been changed yet.

> Okay, thanks for the initial confirmations. I agree with the overall direction here, and of course the importance of, you know, verifying what are the ground truth labels and the need to establish some kind of a baseline before training a model. That's obviously what I would expect from an experimental point of view. But before implementing anything, I just want us to try to simplify this architecture you propose, because this is for the purposes of a technical interview assessment, not that we are building the production deployment right here. So we don't need the actual real database infrastructure implemented, nor do we need to actually do training on real data right now, at least. My priority is therefore a, you know, small enough and interpretable repository that I can explain and justify all the components, obviously. If you think that's good, we can start with trying out the OpenCV QR detector as our baseline, but not necessarily integrating multiple separate decoders for now. We can think about that down the line, but not at the moment. So maybe, yeah, maybe one or two image processing methods, like the grayscale, as you mentioned, or the CLAHE, but only if it's really meaningful, and no C++ here right now, also please. And also, can you just explain to me a little bit more what exactly would this addition of CLAHE do, and when it would kind of help or hurt our pipeline? Why would we want to test it out in the baseline? It's what I'm asking. And then as for the model from Hugging Face, I mean, sounds good. Just kind of help me understand how to justify in what way is it a genuinely suitable pre-trained starting point for our conditions? Do we focus more on a general model to restore the images, or a QR-specific model to restore the images? Like what would be more appropriate? And so, while definitely I agree with Postgres SQL for structured metadata and the S3-compatible storage for images, I've also worked on that in some other projects that I've had. However, the schemas you proposed maybe seem more elaborate than we need for the beginning. So yeah, kind of just help me identify the minimum information we need to collect for this training and evaluation. So basically, what I want to demonstrate now is the full workflow of, so being able to collect and store a scan, then we associate it with a verified label where available. We construct these training examples, maybe including the synthetic data, and then we fine-tune the selected model, and then we evaluate if it actually improved the correct determination compared with the baseline. So we should prioritize exact decoding accuracy, incorrect payloads, and also the latency during our inference, right? And keep in mind, we have to be careful about avoiding any sort of data leakage between training and evaluations. Okay. And then once we've agreed on the design, we will go ahead and implement it through a couple of steps, starting with the API endpoint.

Yes—this should be simplified into an interview-sized reference implementation. We can demonstrate the complete experimental workflow without queues, model registries, production infrastructure, multiple decoders, or elaborate database schemas.

## Simplified workflow

```text
Upload scan
   ↓
Store image in S3 + metadata in PostgreSQL
   ↓
Associate verified payload/sticker label when available
   ↓
Run OpenCV baseline
   ↓
Build leakage-safe dataset manifest
   ↓
Generate degraded/clean synthetic training pairs
   ↓
Fine-tune restoration model
   ↓
Compare baseline vs model-assisted decoding
```

The repository can contain code against PostgreSQL and S3 interfaces while tests use simple in-memory fakes. We do not need Docker, Terraform, Celery, MLflow, or an actual training run.

## Minimal baseline

I agree with using only OpenCV initially.

The baseline would be:

1. Run `cv2.QRCodeDetector.detectAndDecode()` on the original image.
2. If that fails, convert to grayscale, apply CLAHE, and retry.
3. Validate any returned payload against the verified label during evaluation.

This provides one standard attempt and one explainable fallback. We do not need ZXing, WeChat QRCode, or a large collection of preprocessing variants yet.

### What CLAHE does

CLAHE means Contrast Limited Adaptive Histogram Equalization.

Ordinary histogram equalization adjusts contrast across the entire image. CLAHE divides the image into small tiles, enhances contrast independently inside each tile, and then blends the tiles together. It also limits how much the histogram can be amplified.

That makes it potentially useful when:

- One side of the sticker is in shadow.
- Lighting is uneven.
- The sticker has faded and black/white separation is weak.
- The QR region has low local contrast despite the complete photo having reasonable global contrast.

It may hurt when:

- The image already has strong black/white contrast.
- The image contains heavy sensor noise or JPEG artifacts, because those may be amplified.
- There is glare or clipping; CLAHE cannot reconstruct information lost in a fully white highlight.
- Very small QR modules become noisier or their edges become less stable.

OpenCV’s detector already performs internal image processing, so CLAHE is not guaranteed to help. That is precisely why it is suitable as a baseline experiment: it is cheap, deterministic, addresses a failure mode we expect, and can be measured separately.

We should only use it as a fallback. A successful original-image decode should always take priority.

## General restoration model versus QR-specific model

The best compromise is:

> Use a general pretrained restoration architecture, then fine-tune it for QR-specific degradations.

A general model gives us useful pretrained visual features and a maintained implementation. The QR-specific synthetic fine-tuning teaches it that module edges and decoder success matter more than making the result aesthetically pleasing.

Building or adopting a fully QR-specific architecture would be attractive scientifically, but less appropriate for this assessment because:

- Mature pretrained QR-restoration checkpoints are uncommon.
- We would need more custom architecture and training code.
- The quality and transferability of third-party QR datasets are uncertain.
- It becomes harder to explain which improvement came from pretraining, architecture, or synthetic data.

### Why Swin2SR is a reasonable starting point

[`caidas/swin2SR-classical-sr-x2-64`](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64) is reasonable because:

- It is an image-to-image restoration model rather than a text generator.
- It has direct Hugging Face Transformers support.
- It starts with pretrained weights instead of training from scratch.
- The 2× model can help when the QR crop is small or downsampled.
- Its windowed transformer structure models local module edges while still incorporating wider image context.
- Fine-tuning is a straightforward paired-image reconstruction problem.

The justification should be deliberately narrow: Swin2SR is a promising starting point for low-resolution and compressed QR crops. It is not automatically ideal for every listed failure mode.

For example:

- Low resolution/compression: good architectural match.
- Mild blur: may improve after QR-specific fine-tuning.
- Uneven lighting: synthetic fine-tuning can cover this.
- Severe glare or missing modules: no model can reliably reconstruct unknown information.
- Physical fading: potentially recoverable only if enough structure remains.
- Severe motion blur: a deblurring-oriented model such as Restormer may ultimately be better.

A QR-specific recent architecture such as [Edge-Guided Restormer](https://arxiv.org/abs/2510.12098) is a useful discussion point, but I would not make it the reference implementation’s dependency. It is better presented as a future experiment.

Most importantly, we will decode both the original and restored versions. Swin2SR will not be trusted to produce the payload itself:

```text
Baseline:
original → OpenCV
original → grayscale + CLAHE → OpenCV

Challenger:
same baseline attempts
failed QR crop → Swin2SR → OpenCV
```

This makes the model additive and keeps the final decoding deterministic.

## Minimal data to collect

We can reduce the database to two tables.

### `scans`

Required:

- `id`: server UUID.
- `client_scan_id`: idempotency identifier.
- `captured_at`.
- `image_key`: S3 object key.
- `image_sha256`.
- `created_at`.

Optional but useful:

- `latitude`.
- `longitude`.
- `location_accuracy_m`.
- `device_model`.

Location should not be an input to the restoration model. It is contextual information for charger resolution and later analysis.

We do not need to persist brightness, blur scores, OpenCV results, crops, or every experiment in PostgreSQL. Those can be recomputed and included in evaluation output.

### `scan_labels`

- `scan_id`.
- `expected_payload`.
- `group_id`.
- `source`.
- `verified_at`.

`group_id` is essential for preventing leakage. Ideally, it identifies the physical sticker. If that is unavailable, we can use a canonical charger/sticker identifier or a hash of the expected payload.

`source` can be a small enum:

- `registry`
- `reference_asset`
- `manual`

Only verified labels enter the primary evaluation set.

For the assessment, storing labels separately is worth the extra table because it keeps collected observations separate from ground truth. It also naturally supports scans that have not yet been labeled.

## API scope

The first implementation step should expose:

```text
POST /v1/scans
```

It accepts:

- The image.
- `client_scan_id`.
- `captured_at`.
- Optional location and device information.

It stores the image and metadata and returns the generated `scan_id`.

A later, small internal endpoint can demonstrate association of ground truth:

```text
PUT /v1/scans/{scan_id}/label
```

That would not necessarily be a public mobile endpoint. It represents a trusted registry, reviewer, or labeling process.

## Constructing training examples

There are two categories.

### Synthetic paired examples

These are the initial training source:

1. Take verified payloads assigned to the training split.
2. Render clean QR codes.
3. Apply randomized but recorded degradation:
   - downsampling
   - JPEG compression
   - mild motion or defocus blur
   - brightness and contrast changes
   - uneven illumination
   - noise
   - mild perspective distortion
4. Use the degraded image as input and the clean QR as target.

This gives us exact input/target pairs without claiming that a generated QR is the original physical print asset.

### Real collected examples

Real scans with verified payloads are immediately useful for evaluation.

They should only be used as paired restoration training data when we also have a trustworthy clean reference asset. Otherwise, the exact restoration target is unknown.

The collected payloads still influence training because we generate synthetic QR examples from the real payload distribution. Later, if original sticker artwork becomes available, real paired fine-tuning can be added without changing the architecture.

## Preventing leakage

The sequence matters:

1. Assign `group_id` values to train, validation, or test.
2. Persist that assignment in a dataset manifest.
3. Generate synthetic examples only after the split.
4. Never generate training examples from a payload or physical sticker assigned to the test set.
5. Keep all retries and photographs of one sticker in the same split.

The manifest can be a simple JSONL file:

```json
{
  "scan_id": "...",
  "image_key": "...",
  "expected_payload": "...",
  "group_id": "...",
  "split": "test"
}
```

The test split should primarily contain real scans. Synthetic validation is useful for debugging, but it is not a convincing measurement of real-world improvement.

## Evaluation

We should compare two pipelines on exactly the same labeled test scans:

- Baseline: OpenCV original, then CLAHE fallback.
- Challenger: baseline plus Swin2SR restoration on failed scans.

Primary metrics:

- **Exact decoding accuracy:** correct payloads divided by all labeled scans.
- **Incorrect payload rate:** non-empty but wrong payloads divided by all labeled scans.
- **No-decode rate:** scans returning no payload.
- **Recovery rate:** baseline failures correctly recovered by the model branch.
- **Latency:** P50 and P95 per scan.

Latency should:

- Exclude one-time model loading.
- Use batch size one.
- Run on documented hardware.
- Measure the complete challenger path.
- Separately report baseline and restoration time.

Exact payload comparison should be strict unless the business explicitly defines safe canonicalization. We should never silently treat two different URLs as equal merely because they appear to resolve to the same charger.

## Smaller repository layout

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   └── qr_service/
│       ├── config.py
│       ├── api.py
│       ├── schemas.py
│       ├── db.py
│       ├── object_store.py
│       ├── baseline.py
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       └── evaluate.py
└── tests/
    ├── test_api.py
    ├── test_baseline.py
    ├── test_dataset.py
    └── test_evaluation.py
```

This is intentionally flat. Each file has one obvious responsibility, and the full repository can be explained quickly.

The implementation sequence should be:

1. FastAPI scan-ingestion endpoint and storage interfaces.
2. Label association.
3. OpenCV baseline with the CLAHE fallback.
4. Leakage-safe dataset manifest and synthetic generator.
5. Swin2SR wrapper and fine-tuning entry point.
6. Comparative evaluation.

That is small enough for an assessment while still demonstrating the full scientific and engineering workflow.

> All right, now I understand what CLAHE means, and definitely that sounds like something  appropriate for our use case, precisely because, you know, lighting may be uneven, one side of the sticker can be in shadow, like exactly like you said. So that makes sense. But if you choose the src/ type layout for the repo, shouldn't there be a pyproject.toml file? And why is there no test\_training.py inside test/, but the rest is? I also need to clarify something that I just remembered. So, in the task instructions, they mention image-to-text generation. But that's not really necessarily what we are doing or should be doing. Because what you're telling me that we should choose an image restoration model, so that this learned restoration could actually improve the success of a subsequent downstream image-to-text decoding. So, when, you know, you have texts on this image and it visual needs to be decoded into actual text, right? So, as long as we clearly establish for what purpose our inference is, at what stage, we would be fine. So, in our proposed pipeline, how would the image-to-text then be added? Because, you know, our driver needs to scan a QR sticker. The app needs to send that image and metadata to our API endpoint, and store the image and corresponding database record. And then after that, a training process would retrieve the scans with a reliable ground truth label, labels, and then prepare suitable training examples and use those to fine-tune our selected model. And then evaluation would be comparing the original decoding system against this model-enhanced decoding system, right? On a held out test set of images that were never seen before, of course. And if this trained model then improves correct identification of the scans, then we would have a candidate for what we want to deploy later. So, I need you to just help me, or help yourself, distinguish a model that directly recognizes text from an image, a model that improves QR image quality before we do decoding in a deterministic way, right? And also distinguish between a model that's specifically trained to improve the detection of QR codes. So, I need the simplest approach here that clearly satisfies what we're doing in this task, in this assignment. So it's important that we're able to train and improve our chosen model using the collected scan data. So I want us to settle this modeling decisions before we proceed.

You’re right on both repository points:

- A `src/` layout should have a `pyproject.toml` so the package is installed and imported correctly.
- Omitting `test_training.py` was an oversight. The training entry point should have a lightweight smoke test.

More importantly, we can now settle the modeling decision precisely.

## Final modeling decision

We will build a model-assisted QR decoding pipeline:

```text
Image
  ↓
Optional learned image restoration: Swin2SR
  ↓
Deterministic QR detection and decoding: OpenCV
  ↓
Payload text
```

Therefore:

- The complete pipeline is image-to-text—or more accurately, image-to-QR-payload.
- The learned model is image-to-image.
- OpenCV performs the exact image-to-payload conversion.
- We will not use an OCR model such as TrOCR.
- We will not introduce a trainable QR detector in the initial implementation.

This is the simplest technically defensible interpretation of the assignment.

## The three different model categories

| Category | Input → output | What it solves | Decision |
|---|---|---|---|
| OCR/image-to-text model | Printed text image → characters | Human-readable letters and numbers | Not appropriate for the QR matrix |
| QR detection model | Photograph → QR bounding box | Locating the QR code | Not needed initially |
| Image-restoration model | Degraded QR image → improved QR image | Blur, compression, poor contrast, low resolution | Selected approach |

### Direct image-to-text/OCR

A model such as TrOCR recognizes visible characters—for example, a printed charger ID like `NL-ABC-123`.

QR modules are not written characters. A direct generative model would have to generate arbitrary URLs and identifiers and could silently change one character. That is unnecessarily risky when QR codes already have a deterministic decoding format and error correction.

If stickers have a separate human-readable charger ID, OCR could later become a fallback branch. It is not part of our minimum implementation.

### QR detection model

Detection means locating the QR code and returning its corners or bounding box. It does not necessarily decode the payload.

OpenCV’s `QRCodeDetector` already performs both detection and decoding. We should first measure whether failures come from:

- `not_detected`: OpenCV cannot locate the QR.
- `detected_not_decoded`: corners are found, but module data cannot be decoded.
- `decoded`: a payload is returned.

Our restoration model mainly targets the second category. It may indirectly help detection when enhancement is applied before OpenCV, but detection is not its explicit training objective.

If real data later shows that most failures are `not_detected`, a dedicated QR detector would become the appropriate next experiment.

### QR image restoration

This is our chosen learned step.

Swin2SR takes a degraded QR image and produces a higher-resolution restored image. OpenCV then processes the restored result exactly as it processes the original image.

That means the ML model never gets authority to invent the payload.

## Exact baseline and challenger

The baseline will be:

```text
1. OpenCV on original image
2. If unsuccessful:
   grayscale + CLAHE → OpenCV
```

The model-assisted challenger will be:

```text
1. Run the same baseline
2. If still unsuccessful:
   original QR image → fine-tuned Swin2SR → OpenCV
```

The model should receive the original RGB image or QR crop, not the CLAHE output, unless we explicitly train it on CLAHE-processed inputs.

For the assessment, enhancement can be applied to a reasonably sized full image or central scan region. A production implementation would usually use a client-provided or OpenCV-detected crop to reduce latency.

## Why Swin2SR remains the selected model

We will use [`caidas/swin2SR-classical-sr-x2-64`](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64).

The justification is:

1. It is pretrained for image restoration/super-resolution.
2. It has a supported Hugging Face Transformers implementation.
3. It accepts paired low-quality/high-quality image training.
4. The 2× version is conservative compared with 4× super-resolution.
5. We can fine-tune it on QR-specific synthetic degradations.
6. It preserves a clean separation between learned restoration and deterministic decoding.

We should not claim that its natural-image pretraining is already optimal for QR codes. The pretraining is merely a useful initialization.

Our actual strategy is:

> General pretrained restoration architecture + QR-specific fine-tuning.

That is preferable to building a QR-specific transformer from scratch because it requires less data, less custom code, and is easier to explain.

## How collected scans can improve the model

This needs one important distinction: a verified payload label is enough for evaluating a scan, but it is not always enough to provide a pixel-perfect restoration target.

The dataset will support two types of training examples.

### Synthetic paired examples

Given a verified training payload:

1. Render a clean QR code.
2. Produce a lower-resolution version.
3. Apply blur, poor contrast, uneven lighting, compression, or noise.
4. Use the degraded image as the input.
5. Use the clean rendered QR as the target.

This is naturally compatible with Swin2SR’s 2× objective.

The collected dataset contributes the real payload formats and distribution, while the degradation generator produces exact training pairs.

### Real paired examples

When we have both:

- The collected degraded scan.
- The corresponding original clean sticker/QR asset.

We can align and use them as a real training pair.

The label therefore should optionally include:

```text
reference_image_key
```

If no reference asset exists, the real scan remains useful for evaluation but should not automatically be treated as a pixel-aligned restoration pair.

That is an honest and clean foundation: the training pipeline supports real pairs when available, while synthetic pairs make the assessment executable without real production data.

## Minimal label schema

The label table becomes:

```text
scan_id
expected_payload
group_id
source
reference_image_key     nullable
verified_at
```

Where:

- `expected_payload` is the exact correct QR output.
- `group_id` identifies the physical sticker or equivalent grouping key.
- `source` records how the label was established.
- `reference_image_key` points to the clean QR/sticker asset when available.

This supports both evaluation and paired training without additional schema complexity.

## Training objective

Keep the initial training loss simple:

```text
L1 reconstruction loss
```

We do not need GAN, perceptual, language, or decoder-aware losses initially.

The real success criterion is not the training loss; it is whether OpenCV can decode the restored result correctly. L1 is only the optimization signal needed to learn clean reconstruction.

A more QR-specific module-edge loss could be an enhancement later.

## Evaluation design

We create the split before constructing or augmenting examples:

```text
group_id → train, validation, or test
```

All images, retries, reference assets, and generated derivatives associated with a group remain in the same split.

Then:

- Fine-tune only with training groups.
- Use validation groups for checkpoint selection.
- Evaluate once on unseen test groups.
- Prefer real collected scans in the final test set.
- Never generate training examples from test payloads.

The comparison is:

| Pipeline | Exact accuracy | Incorrect payloads | No decode | Latency |
|---|---:|---:|---:|---:|
| Original + CLAHE baseline | measured | measured | measured | P50/P95 |
| Baseline + Swin2SR fallback | measured | measured | measured | P50/P95 |

We should additionally report:

- How many baseline failures Swin2SR recovered correctly.
- How often Swin2SR produced an incorrect non-empty payload.
- Detection failures versus detected-but-not-decoded failures.
- Model inference time separately from OpenCV time.

## Revised repository layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── src/
│   └── qr_service/
│       ├── __init__.py
│       ├── config.py
│       ├── api.py
│       ├── schemas.py
│       ├── db.py
│       ├── object_store.py
│       ├── baseline.py
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       └── evaluate.py
└── tests/
    ├── test_api.py
    ├── test_baseline.py
    ├── test_dataset.py
    ├── test_model.py
    ├── test_training.py
    └── test_evaluation.py
```

`pyproject.toml` will define:

- The build system.
- Package name and Python version.
- `src` package discovery.
- Pytest configuration.
- Formatting and linting configuration.

Because the assignment explicitly requests `requirements.txt`, we should keep it as the dependency source and configure `pyproject.toml` to read those dependencies dynamically. That avoids maintaining two independent dependency lists.

`test_training.py` should not attempt a full Hugging Face training run or download weights. It will use a tiny model/test double and verify that:

- One training batch can run.
- The loss is finite.
- Backpropagation occurs.
- A checkpoint or training result is produced.

## Settled scope

The repository will demonstrate:

1. Collecting and storing QR scan images and metadata.
2. Associating scans with verified payload labels.
3. Running an OpenCV baseline with a CLAHE fallback.
4. Creating leakage-safe dataset splits.
5. Generating QR-specific paired training examples.
6. Fine-tuning a pretrained Swin2SR restoration model.
7. Passing restored images through OpenCV.
8. Measuring exact payload accuracy, incorrect payloads, failures, and latency.

This satisfies the model-improvement requirement while keeping the payload-generation stage deterministic and interpretable.

> Okay, now I'm a little bit happier. One small clarification that I think you misunderstood. I didn't say that the assignment explicitly requires requirements.txt. That was just me saying that, from what I know, typically, how I structure my repositories. But it's not like they said that this was a required file. So the TOML file is enough, you know, we need to choose one approach, not mix them. So we don't need to make both. That's all I'm saying. And all the tests, I guess, should be kept minimum but sufficient. And about the image-to-text vs image-to-QR discrepancy, it is not either or, I assume you know this, as in production deployment we may need both of these capabilities obviously. So, just one more time briefly explain exactly what is the model learning from the collected QR images, and how does that learning make TAP's charger identification more reliable?

Agreed: we’ll use `pyproject.toml` as the single dependency, packaging, and tooling configuration—no `requirements.txt`. Tests will be minimal and behavior-focused.

The model is not learning charger identities or QR payload text. It learns a visual transformation:

```text
degraded QR image → clean, higher-quality QR image
```

From paired examples, it learns recurring degradation patterns such as blur, low local contrast, uneven lighting, compression artifacts, noise, and reduced resolution. It learns to restore the sharp black-and-white module boundaries that a QR decoder needs.

Training pairs come from:

- Clean QR/reference assets with synthetic degradations.
- Collected scans paired with their clean reference asset, when available.
- Verified payloads used to generate representative synthetic QR examples.

A verified payload without a clean reference remains useful for evaluation, but is not automatically a pixel-perfect restoration target.

At inference:

```text
Failed scan
  → restoration model
  → improved QR image
  → deterministic OpenCV decoding
  → verified payload
  → charger lookup
```

This makes Tap’s identification more reliable by increasing the probability that OpenCV can correctly recover the QR payload. The model does not guess which charger is nearby; it improves the visual evidence used by the exact decoder. The decoded payload can then be combined with Tap’s charger registry, location, and—later—the OCR result from any human-readable sticker text.

In production, both branches can coexist:

- QR restoration and deterministic decoding for the encoded payload.
- OCR for visible charger IDs or other printed sticker text.

For this assessment, we focus the trainable component on QR restoration because it directly addresses the stated capture failures while keeping the final identification deterministic and measurable.

> One clarification before we implement.
> TAP's assignment explicitly states that we do not need to test the system with real data. We currently have no access to actual charger sticker images, or you know, verified charger payloads.
> So I don't want us to overinvest in obtaining data or running extensive ML experiments.
> Our objective is to just implement a working training and evaluation framework, not to claim that we've already demonstrated improvements on real-world scans.
> So if you were going to implement some synthetic data example, then do some lightweight generator that creates QR codes with known payloads and applies these realistic degradations like blur, noise and bad lighting. Then we would use these generated examples to demonstrate the data pipeline and support a reproducible training experiment, right? And our evaluation needs to be able to compare normal OpenCV decoding, pretrained Swin2SR and the fine-tuned Swin2SR. This evaluation should, you know, be compatible with real, verified scan data that could be collected through our API in the future. Include only small automated tests that verify the components work correctly without requiring a full model training run. &#x20;
> Since the assignment is primarily about design, code, model selection and model improvement, I want us to focus on making these capabilities technically correct and easy to explain.
> Please keep the synthetic dataset generator simple, avoid downloading external datasets, and do not make extensive model training a prerequisite for running the service. Keep code clean and elegant, shorter if/where possible.

Yes—this fixes the scope clearly. We will implement a framework, not claim real-world model improvement.

## Frozen implementation scope

### Service

- FastAPI endpoint for storing a scan and minimal metadata.
- PostgreSQL repository interface and S3-compatible object-store interface.
- No database or object-storage infrastructure.
- In-memory fakes for automated tests.
- API startup will not import or load Swin2SR.

### Synthetic data

A small deterministic generator will:

1. Generate QR codes with known synthetic payloads.
2. Produce clean targets.
3. Apply a limited set of realistic degradations:
   - Gaussian or motion blur.
   - Gaussian noise.
   - Reduced or uneven lighting.
4. Save a manifest containing:
   - sample ID
   - payload/group ID
   - degraded image
   - clean target
   - train/validation/test split
   - degradation parameters

A fixed random seed will make the experiment reproducible. No external datasets will be downloaded.

Synthetic payloads will be split before generating image variations so the same payload cannot appear in both training and evaluation.

### Three evaluation pipelines

All three will ultimately use the same deterministic OpenCV decoder:

```text
1. Baseline
   original image → OpenCV
   optional CLAHE fallback → OpenCV

2. Pretrained
   original image → pretrained Swin2SR → OpenCV

3. Fine-tuned
   original image → fine-tuned Swin2SR → OpenCV
```

The evaluation will report:

- Exact payload accuracy.
- Incorrect non-empty payload rate.
- No-decode rate.
- P50 and P95 latency.
- Recovery over the OpenCV baseline.

The report will clearly label synthetic results as framework demonstrations rather than evidence of production improvement.

### Compatibility with future real data

Synthetic and future real examples will share one manifest/sample schema:

```text
sample_id
group_id
image location
expected payload
optional clean target
source: synthetic | collected
split
```

Future verified API scans can therefore be exported into the same evaluation format. A clean target is required for restoration training, but only an expected payload is required for end-to-end decoding evaluation.

### Optional ML execution

Model work will be exposed through explicit commands, conceptually:

```text
generate-synthetic
train
evaluate
```

Running the API will not:

- Download Hugging Face weights.
- Generate a dataset.
- Start training.
- Require a GPU.

Pretrained Swin2SR weights will only be downloaded when the user explicitly runs the relevant training or evaluation command. A fine-tuned checkpoint will be an optional input to evaluation.

`pyproject.toml` will be the only package/dependency configuration. ML and development dependencies can be optional extras so the ingestion service remains lightweight.

### Small automated test suite

The tests will verify behavior without downloading Swin2SR or performing a real training run:

- API upload with fake database and object storage.
- Clean QR decoding and CLAHE fallback behavior.
- Deterministic synthetic generation and leakage-safe splits.
- Evaluation metric correctness.
- One lightweight training-step smoke test using a tiny model/test double.
- Swin2SR adapter behavior with its loader mocked.

This gives us confidence that the plumbing works while keeping test execution fast and offline.

The selected learned capability remains QR image restoration. The repository demonstrates how to fine-tune it and how to determine whether it improves downstream image-to-payload decoding. It does not claim that synthetic performance will transfer to Tap’s real charger stickers without later validation.

No files have been created yet.

> Okay, go ahead and implement it. I already initialized the repository myself, and published the main branch to GitHub: currently only includes an empty README. Start implementation with the API endpoint, schemas and storage layer. Keep the implementation minimal, testable and consistent with the agreed requirements.

<details><summary>8 previous messages</summary>

> I’ll implement only the first vertical slice: project configuration, FastAPI scan ingestion, minimal schemas, PostgreSQL/S3-facing abstractions, and small offline tests using fakes. I’ll first inspect the repository instructions and current state so I preserve the initialized Git history.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `pwd && rg --files -g 'AGENTS.md' -g '!**/.git/**' .. . 2>/dev/null | sort -u && git status --short --branch && rg --files -g '!**/.git/**' | sort`
> - Ran `sed -n '1,200p' README.md && git log -1 --oneline && python3 --version`
>
> </details>
>
> The repository is clean and contains only the initial README. I’m going with one small package and no infrastructure files: a `POST /v1/scans` route, an ingestion service, PostgreSQL and S3 adapters behind protocols, and tests that inject in-memory fakes. The API will remain completely independent of the later ML dependencies.
>
> <details><summary>Created 11 files, edited 3 files, ran 3 commands</summary>
>
> - Ran `for p in python3.13 python3.12 python3.11 python3.10 python3; do command -v "$p" >/dev/null 2>&1 && "$p" --version; done
> python3 - <<'PY'
> mods = ['fastapi', 'sqlalchemy', 'pydantic_settings', 'boto3', 'pytest', 'httpx']
> for mod in mods:
>     try:
>         module = __import__(mod)
>         print(f'{mod}={getattr(module, "__version__", "installed")}')
>     except Exception as exc:
>         print(f'{mod}=missing ({type(exc).__name__})')
> PY`
> - Ran `command -v uv || true
> command -v rye || true
> command -v poetry || true
> command -v pipx || true`
> - Wrote `./.gitignore` (+9 -0)
> - Wrote `./README.md` (+68 -1)
> - Wrote `./pyproject.toml` (+42 -0)
> - Wrote `./src/tap_charger_scan/__init__.py` (+1 -0)
> - Wrote `./src/tap_charger_scan/api.py` (+114 -0)
> - Wrote `./src/tap_charger_scan/config.py` (+23 -0)
> - Wrote `./src/tap_charger_scan/models.py` (+49 -0)
> - Wrote `./src/tap_charger_scan/schemas.py` (+35 -0)
> - Wrote `./src/tap_charger_scan/service.py` (+120 -0)
> - Wrote `./src/tap_charger_scan/storage.py` (+92 -0)
> - Wrote `./tests/test_api.py` (+135 -0)
> - Wrote `./tests/test_storage.py` (+38 -0)
> - Ran `UV_CACHE_DIR=/tmp/tap-electric-uv-cache uv python list --only-installed`
> - Wrote `./src/tap_charger_scan/api.py` (+4 -1)
> - Wrote `./src/tap_charger_scan/storage.py` (+2 -1)
>
> </details>
>
> The first slice is in place. I’ve also kept the data model intentionally narrow: one scan record with the original object key, checksum, dimensions, optional location/device context, and an idempotency key. I’m moving through lint and tests now; no external database or bucket will be contacted.
>
> <details><summary>Edited 3 files, deleted a file, ran 11 commands</summary>
>
> - Ran `UV_CACHE_DIR=/tmp/tap-electric-uv-cache uv sync --python 3.12 --extra dev`
> - Ran `UV_CACHE_DIR=/tmp/tap-electric-uv-cache uv sync --python 3.12 --extra dev`
> - Ran `.venv/bin/ruff check .`
> - Ran `.venv/bin/pytest -q`
> - Wrote `./src/tap_charger_scan/api.py` (+3 -1)
> - Wrote `./src/tap_charger_scan/service.py` (+2 -2)
> - Ran `.venv/bin/ruff check . && .venv/bin/ruff format --check .`
> - Ran `git diff --check && git status --short`
> - Ran `.venv/bin/pytest -q`
> - Wrote `./tests/test_api.py` (+32 -21)
> - Ran `.venv/bin/ruff check . && .venv/bin/ruff format --check .`
> - Ran `.venv/bin/pytest -q`
> - Ran `wc -l uv.lock && git diff --stat && git diff -- README.md pyproject.toml src tests .gitignore`
> - Ran `find src tests -type f -maxdepth 3 -print -exec sed -n '1,240p' {} \;`
> - Deleted `./uv.lock` (+0 -1304)
>
> </details>
>
> The endpoint now passes its offline test suite. The behavior covered is the core contract: valid storage, image validation, stable retry behavior, conflict on idempotency-key reuse, and the S3 adapter request shape. I’m doing a final repository review for accidental generated files and packaging consistency.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `.venv/bin/python -m compileall -q src`
> - Ran `git diff --check && git status --short && git ls-files --others --exclude-standard | sort`
> - Ran `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q`
>
> </details>
</details>

Implemented the first API/storage slice.

Key pieces:

- [`pyproject.toml`](./pyproject.toml) is the sole dependency and tooling configuration.
- [`api.py`](./src/tap_charger_scan/api.py) provides `POST /v1/scans` and `/health`.
- [`schemas.py`](./src/tap_charger_scan/schemas.py) defines strict metadata and response schemas.
- [`service.py`](./src/tap_charger_scan/service.py) handles validation, hashing, idempotency, and persistence coordination.
- [`storage.py`](./src/tap_charger_scan/storage.py) contains PostgreSQL and S3-compatible adapters behind small protocols.
- [`models.py`](./src/tap_charger_scan/models.py) contains the minimal scan domain and SQLAlchemy models.
- [`README.md`](./README.md) documents setup, configuration, and the endpoint.

The endpoint:

- Accepts multipart JPEG/PNG plus JSON metadata.
- Validates actual image contents, size, dimensions, and MIME type.
- Stores the original bytes without re-encoding.
- Calculates a SHA-256 checksum.
- Uses `client_scan_id` for retry safety.
- Cleans up the S3 object if the database write fails.
- Does not connect to external infrastructure during tests.

Verification completed:

- 5 tests passed.
- Ruff lint and formatting checks passed.
- Python compilation passed.
- No ML dependencies or model loading were added.

> I assume one single storage.py like this would never make it to production, but maybe my expectations are too high anyways for the limited time I have for this now. Ok, let's continue with the rest of the implementation. I want a complete, compact and defensible submission addressing every requirement of the assignment. So, implement the baseline. Like we discussed, keep this simple, okay? We don't need multiple libraries or preprocessing methods for this, like we discussed. Then implement, I guess, the synthetic dataset and ground truths. Make the splits, of course, before applying the transformations and augmentations to prevent leakage, of course. And make sure you're providing a clear path for when we would be using future real scanned images from our storage layer, once everything would become available. And then implement the model using the pre-trained Hugging Face checkpoint and, you know, implement the train.py file using, you know, PyTorch, of course. And we need to make sure that the expected input dimensions are all aligned. We need to make the training executable, but a full training run is not required for this assignment. So we don't have to make training the full model or downloading anything necessary for just being able to run our API or doing the automated tests. I guess that's my idea based on our discussion. And then implement the evaluation, like we talked about. And then we can leave the automated tests and documentation for last, after you do this.

<details><summary>15 previous messages</summary>

> I’ll complete the remaining pipeline as optional ML tooling, while keeping the API dependency path unchanged. I’ll also split the current combined storage module now—the compact submission can still have separate repository and object-store adapters without becoming elaborate. First I’m checking the current Hugging Face/PyTorch interfaces and the installed OpenCV capabilities so the executable paths match actual library APIs.
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `site:huggingface.co/docs/transformers Swin2SRForImageSuperResolution training reconstruction loss pixel_values labels ...`
>
> Searched the web for `site:github.com/huggingface/transformers modeling_swin2sr.py reconstruction_loss labels Swin2SRForImageSuperResolution ...`
>
> Searched the web for `'loss_fct'`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> I’ll finish this as one coherent submission: split the persistence adapters, add verified-label support, then build the OpenCV baseline, grouped synthetic/real-data manifests, Swin2SR training, and comparative evaluation. Model dependencies will remain optional, so the API and ordinary tests won’t download a checkpoint.
>
> <details><summary>Created 3 files, edited 6 files, deleted a file, explored 5 files, a search, ran a command</summary>
>
> - Ran `pwd && rg --files -g '!*.pyc' -g '!__pycache__' && git status --short && sed -n '1,240p' pyproject.toml && sed -n '1,260p' src/tap_charger_scan/storage.py && sed -n '1,260p' src/tap_charger_scan/api.py`
>
> <details><summary>Explored 5 files</summary>
>
> - Read `./src/tap_charger_scan/models.py`
> - Read `./src/tap_charger_scan/schemas.py`
> - Read `./src/tap_charger_scan/service.py`
> - Read `./tests/test_api.py`
> - Read `./tests/test_storage.py`
>
> </details>
>
> - Wrote `./src/tap_charger_scan/models.py` (+35 -1)
> - Wrote `./src/tap_charger_scan/object_store.py` (+38 -0)
> - Wrote `./src/tap_charger_scan/ports.py` (+31 -0)
> - Wrote `./src/tap_charger_scan/repositories.py` (+85 -0)
> - Deleted `./src/tap_charger_scan/storage.py` (+0 -93)
>
> - Wrote `./src/tap_charger_scan/api.py` (+21 -2)
> - Wrote `./src/tap_charger_scan/schemas.py` (+18 -0)
> - Wrote `./src/tap_charger_scan/service.py` (+23 -4)
> - Wrote `./tests/test_api.py` (+37 -1)
> - Wrote `./tests/test_storage.py` (+10 -1)
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Searched for `storage|ScanLabel|class S3|verify_label` in `src`
> - Read `./src/tap_charger_scan/service.py`
> - Read `./src/tap_charger_scan/models.py`
>
> </details>
>
> </details>
>
> The data contract now distinguishes two kinds of truth that matter: a verified QR payload is sufficient for evaluation, while supervised restoration training additionally needs an aligned clean target image. Synthetic samples provide both; future collected scans enter the same manifest immediately for evaluation and become trainable only when that stronger target is available.
>
> <details><summary>Created 6 files, edited 2 files, ran 3 commands</summary>
>
> - Wrote `./pyproject.toml` (+12 -0)
> - Wrote `./src/tap_charger_scan/baseline.py` (+56 -0)
> - Wrote `./src/tap_charger_scan/dataset.py` (+137 -0)
> - Wrote `./src/tap_charger_scan/synthetic.py` (+114 -0)
> - Wrote `./src/tap_charger_scan/evaluate.py` (+129 -0)
> - Wrote `./src/tap_charger_scan/restoration.py` (+69 -0)
> - Wrote `./src/tap_charger_scan/train.py` (+177 -0)
> - Ran `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q`
> - Wrote `./src/tap_charger_scan/train.py` (+8 -2)
> - Ran `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q`
> - Ran `.venv/bin/ruff format src/tap_charger_scan/repositories.py src/tap_charger_scan/restoration.py && .venv/bin/pytest -q`
>
> </details>
>
> The implementation boundary is in place: the API imports no ML modules, while the `ml` extra owns OpenCV, QR generation, PyTorch, and Transformers. I’m installing that optional extra only to exercise the lightweight component tests; I will not download the Swin2SR weights or run a full training job.
>
> <details><summary>Ran a command</summary>
>
> - Ran `uv pip install --python .venv/bin/python -e '.[dev,ml]'`
>
> </details>
>
> The optional environment is still finishing the PyTorch wheel installation. While that runs, the code path remains offline-safe: neither importing the FastAPI app nor running its tests initializes Transformers or contacts Hugging Face.
>
> <details><summary>Created 4 files, ran 4 commands</summary>
>
> - Ran `.venv/bin/python - <<'PY'
> from transformers import Swin2SRForImageSuperResolution
> print(Swin2SRForImageSuperResolution.__name__)
> PY
> .venv/bin/python -m tap_charger_scan.synthetic /tmp/tap-electric-smoke --payload-count 6 --variants-per-payload 1 --seed 7
> .venv/bin/python - <<'PY'
> from pathlib import Path
> from tap_charger_scan.baseline import OpenCVQRDecoder
> from tap_charger_scan.dataset import read_manifest, resolve_path
> from PIL import Image
> p=Path('/tmp/tap-electric-smoke/manifest.jsonl')
> r=read_manifest(p)[0]
> with Image.open(resolve_path(p,r.input_path)) as im:
>  print(im.size, OpenCVQRDecoder().decode(im).payload)
> with Image.open(resolve_path(p,r.target_path)) as im:
>  print(im.size, OpenCVQRDecoder().decode(im).payload)
> PY`
> - Wrote `./tests/test_baseline.py` (+13 -0)
> - Wrote `./tests/test_dataset.py` (+57 -0)
> - Wrote `./tests/test_evaluation.py` (+44 -0)
> - Wrote `./tests/test_training.py` (+43 -0)
> - Ran `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q`
> - Ran `sed -n '1,360p' README.md && find src tests -type f | sort && git diff --stat && git status --short`
> - Ran `cat .gitignore && git status --short --untracked-files=all | sed -n '1,200p'`
>
> </details>
>
> All lightweight checks now pass, including a one-step train-loop test with an injected tiny PyTorch model. That test exercises batching, 64→128 shape alignment, loss, optimization, and checkpoint metadata without fetching Hugging Face weights. I’m finishing the README around the actual design boundaries and reproducible commands.
>
> <details><summary>Edited 7 files, explored 4 files, ran 3 commands</summary>
>
> - Wrote `./README.md` (+157 -28)
>
> <details><summary>Explored 4 files</summary>
>
> - Read `./src/tap_charger_scan/__init__.py`
> - Read `./src/tap_charger_scan/evaluate.py`
> - Read `./src/tap_charger_scan/train.py`
> - Read `./src/tap_charger_scan/dataset.py`
>
> </details>
>
> - Ran `.venv/bin/python - <<'PY'
> from inspect import signature
> from transformers import Swin2SRConfig
> print(signature(Swin2SRConfig))
> c=Swin2SRConfig()
> print(c)
> PY`
>
> - Ran `.venv/bin/python - <<'PY'
> import torch
> from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution
> m=Swin2SRForImageSuperResolution(Swin2SRConfig(embed_dim=12, depths=(1,), num_heads=(3,), window_size=8, upscale=2))
> y=m(pixel_values=torch.rand(1,3,64,64)).reconstruction
> print(y.shape)
> PY`
>
> - Wrote `./README.md` (+2 -2)
> - Wrote `./src/tap_charger_scan/restoration.py` (+1 -1)
> - Wrote `./src/tap_charger_scan/synthetic.py` (+1 -1)
> - Wrote `./tests/test_training.py` (+11 -0)
>
> - Wrote `./src/tap_charger_scan/__init__.py` (+1 -1)
> - Wrote `./src/tap_charger_scan/object_store.py` (+5 -1)
> - Wrote `./src/tap_charger_scan/train.py` (+11 -6)
>
> - Wrote `./src/tap_charger_scan/train.py` (+8 -0)
>
> - Ran `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q && .venv/bin/python -m compileall -q src tests && .venv/bin/tap-generate-data --help >/dev/null && .venv/bin/tap-train --help >/dev/null && .venv/bin/tap-evaluate --help >/dev/null`
>
> </details>
>
> The repository is functionally complete and the full offline suite passes. I also verified the real Transformers Swin2SR class—instantiated from a tiny local config, with no weights download—produces the expected `[1, 3, 64, 64] → [1, 3, 128, 128]` contract. I’m doing the final diff and packaging audit now.
>
> <details><summary>Edited a file, ran 4 commands</summary>
>
> - Ran `git diff --check && git status --short --untracked-files=all && git diff -- README.md && sed -n '1,240p' pyproject.toml && find src/tap_charger_scan tests -type f \( -name '*.py' -o -name '*.toml' \) | sort`
> - Ran `.venv/bin/python - <<'PY'
> import sys
> import tap_charger_scan.api
> assert 'torch' not in sys.modules
> assert 'transformers' not in sys.modules
> assert 'cv2' not in sys.modules
> print('API import is ML-free')
> PY
> uv pip check --python .venv/bin/python`
> - Ran `UV_CACHE_DIR=/tmp/tap-electric-uv-cache uv pip check --python .venv/bin/python`
> - Wrote `./README.md` (+4 -0)
> - Ran `git diff --check && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q`
>
> </details>
</details>

Implemented the complete compact workflow.

Key pieces:

- FastAPI scan ingestion and verified-label endpoint in [api.py](./src/tap_charger_scan/api.py).
- Persistence split into repository, object-store, and protocol modules—no longer one `storage.py`.
- OpenCV `QRCodeDetector` baseline with one CLAHE fallback in [baseline.py](./src/tap_charger_scan/baseline.py).
- Reproducible synthetic QR generator with blur, noise, and uneven lighting. Payload groups are split before augmentation.
- Shared manifest format and export path for future verified S3/PostgreSQL scans in [dataset.py](./src/tap_charger_scan/dataset.py).
- Pinned [Swin2SR x2 checkpoint](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64), with explicit 64×64 → 128×128 handling.
- Executable PyTorch fine-tuning with L1 restoration loss, validation, and best-checkpoint saving in [train.py](./src/tap_charger_scan/train.py).
- Held-out comparison of OpenCV, pretrained Swin2SR, and fine-tuned Swin2SR, reporting accuracy, incorrect payloads, failures, and latency in [evaluate.py](./src/tap_charger_scan/evaluate.py).
- Optional ML dependencies kept separate in [pyproject.toml](./pyproject.toml); importing the API does not import Torch, Transformers, or OpenCV.
- Full design rationale, assumptions, commands, and production omissions documented in [README.md](./README.md).

Validation completed:

- 12 tests passed.
- Ruff lint and formatting passed.
- Python compilation and all three CLI entry points verified.
- Dependency compatibility passed.
- No Hugging Face weights or external datasets were downloaded, and no full training run was performed.

> when you say ''I will not download the Swin2SR weights'',  what exactly do you mean? because we do want to demonstrate that model loading and inference work; their instructions stated that we are to take a model from hugging face and implement it in our code base, not only to choose the model. I'm just making sure we are on the same page here. I know I asked for no external downloads, but we dont only want to be initialising the model with random weights. Basically, I want you to confirm that the actual Hugging Face weights can be loaded and used for a successful inference pass.

<details><summary>2 previous messages</summary>

> You’re right to distinguish those two checks. So far I verified the Hugging Face architecture and tensor contract using a small randomly initialized config, but I deliberately had not fetched the published checkpoint. I’ll now load the actual pinned Hugging Face weights and run a real inference pass through our wrapper; this is a model verification step, not a training run or external dataset download.
>
> <details><summary>Ran a command</summary>
>
> - Ran `HF_HOME=/tmp/tap-electric-hf-cache .venv/bin/python - <<'PY'
> from pathlib import Path
>
> import torch
> from PIL import Image
>
> from tap_charger_scan.baseline import OpenCVQRDecoder
> from tap_charger_scan.dataset import read_manifest, resolve_path
> from tap_charger_scan.restoration import MODEL_ID, MODEL_REVISION, Swin2SRRestorer
> from tap_charger_scan.synthetic import generate_dataset
>
> root = Path('/tmp/tap-electric-pretrained-smoke')
> manifest = generate_dataset(root, payload_count=3, variants_per_payload=1, seed=42)
> record = read_manifest(manifest, split='test')[0]
> with Image.open(resolve_path(manifest, record.input_path)) as source:
>     input_image = source.convert('RGB')
>
> restorer = Swin2SRRestorer.from_pretrained(device='cpu')
> output_image = restorer.restore(input_image)
> decoder = OpenCVQRDecoder()
>
> parameter_count = sum(parameter.numel() for parameter in restorer.model.parameters())
> parameter_l1 = sum(parameter.detach().abs().sum().item() for parameter in restorer.model.parameters())
> output_path = root / 'pretrained_output.png'
> output_image.save(output_path)
>
> print({
>     'model_id': MODEL_ID,
>     'revision': MODEL_REVISION,
>     'model_class': type(restorer.model).__name__,
>     'parameter_count': parameter_count,
>     'parameter_l1': round(parameter_l1, 3),
>     'input_size': input_image.size,
>     'output_size': output_image.size,
>     'output_range': (min(output_image.getdata())[0], max(output_image.getdata())[0]),
>     'expected_payload': record.expected_payload,
>     'decoded_before': decoder.decode(input_image).payload,
>     'decoded_after': decoder.decode(output_image).payload,
>     'output_path': str(output_path),
>     'torch_version': torch.__version__,
> })
> PY`
>
> </details>
</details>

Yes—we are on the same page. “Not downloading weights” meant only that checkpoint download is not required for API startup or automated tests. The actual implementation uses:

```python
Swin2SRForImageSuperResolution.from_pretrained(
    "caidas/swin2SR-classical-sr-x2-64",
    revision="9a57888931c46364d7df003ca9b7a27347ea9539",
    use_safetensors=True,
)
```

I have now downloaded those published [Hugging Face weights](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64) and successfully ran our wrapper on a synthetic degraded QR:

- Loaded model: `Swin2SRForImageSuperResolution`
- Parameters: 12,091,571
- Input: 64×64 RGB
- Output: 128×128 RGB
- OpenCV before restoration: no decode
- OpenCV after pretrained restoration: exact expected payload, `tap://charger/C0002`

So this was a genuine pretrained inference pass—not random initialization. Random weights are used only in lightweight automated tests that validate dimensions without requiring network access. Full training remains intentionally unexecuted.

> Okay, I unfortunately don't have much time. So I'm going to ask you now to verify what we did against the objective of this task assignment. So the code for fine-tuning, essentially, it needs to genuinely be able to update the pre-trained parameters and save some kind of a reloadable checkpoint, even though we're not doing a full training run. And also verify that we're not fabricating any results for experiments that are not actually executed. And also make sure the README contains clear, concise, understandable, and accurate setup instructions. And so any model weights or datasets we generate or cached files, whatever needs to be ignored from Git, make sure we are ignoring from Git. And if you discover any issues, please fix them without expanding the architecture or adding new features or new workarounds. And then finally, you would need to give me a list of exact commands that are a good suitable flow for me to execute and demonstrate that all the components function properly.

<details><summary>4 previous messages</summary>

> I’ll audit only the requested acceptance points: actual parameter updates, reloadable Hugging Face checkpoint, honesty of reported results, setup/demo commands, and Git exclusions. I’ll run one real pretrained-weight optimization step in `/tmp`, reload the saved checkpoint locally, and fix only concrete issues found.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `cat .gitignore && sed -n '1,260p' README.md && sed -n '1,260p' src/tap_charger_scan/train.py && sed -n '1,180p' src/tap_charger_scan/restoration.py && git status --short --ignored`
> - Ran `HF_HOME=/tmp/tap-electric-hf-cache .venv/bin/python - <<'PY'
> from pathlib import Path
>
> import torch
> from transformers import Swin2SRForImageSuperResolution
>
> from tap_charger_scan.dataset import read_manifest, resolve_path
> from tap_charger_scan.restoration import MODEL_ID, MODEL_REVISION, image_to_tensor
> from tap_charger_scan.synthetic import generate_dataset
> from tap_charger_scan.train import TrainingConfig, train
> from PIL import Image
>
> root = Path('/tmp/tap-electric-finetune-verification')
> data_dir = root / 'data'
> checkpoint_dir = root / 'checkpoint'
> manifest = generate_dataset(data_dir, payload_count=3, variants_per_payload=1, seed=19)
> model = Swin2SRForImageSuperResolution.from_pretrained(
>     MODEL_ID,
>     revision=MODEL_REVISION,
>     use_safetensors=True,
>     local_files_only=True,
> )
> parameter_name, parameter = next(
>     (name, value) for name, value in model.named_parameters()
>     if value.requires_grad and value.numel() > 1000
> )
> before = parameter.detach().clone()
> history = train(
>     TrainingConfig(
>         manifest=manifest,
>         output_dir=checkpoint_dir,
>         epochs=1,
>         batch_size=1,
>         learning_rate=1e-4,
>         max_steps=1,
>         device='cpu',
>         seed=19,
>     ),
>     model=model,
> )
> after = dict(model.named_parameters())[parameter_name].detach()
> max_parameter_change = (after - before).abs().max().item()
> changed_elements = torch.count_nonzero(after != before).item()
>
> reloaded = Swin2SRForImageSuperResolution.from_pretrained(
>     checkpoint_dir,
>     use_safetensors=True,
>     local_files_only=True,
> )
> reloaded_parameter = dict(reloaded.named_parameters())[parameter_name].detach()
> record = read_manifest(manifest, split='test')[0]
> with Image.open(resolve_path(manifest, record.input_path)) as image:
>     input_tensor = image_to_tensor(image).unsqueeze(0)
> with torch.no_grad():
>     output = reloaded(pixel_values=input_tensor).reconstruction
>
> print({
>     'base_model': MODEL_ID,
>     'parameter_checked': parameter_name,
>     'changed_elements': changed_elements,
>     'max_abs_parameter_change': max_parameter_change,
>     'train_loss': history['train_loss'][0],
>     'validation_loss': history['validation_loss'][0],
>     'checkpoint_files': sorted(path.name for path in checkpoint_dir.iterdir()),
>     'reload_matches_saved_parameter': torch.equal(after, reloaded_parameter),
>     'reloaded_output_shape': tuple(output.shape),
>     'checkpoint_dir': str(checkpoint_dir),
> })
> PY`
>
> </details>
>
> The real fine-tuning audit passed: one optimizer step changed 4,860 values in the pretrained first convolution, the best checkpoint contained `config.json` and `model.safetensors`, reloading reproduced the saved parameter exactly, and the reloaded model produced a 128×128 output. I found one concrete repository issue: generated datasets/checkpoints were not yet covered by `.gitignore`; I’m fixing that and tightening the README into a single copy-paste smoke flow.
>
> <details><summary>Edited 2 files, ran 2 commands</summary>
>
> - Wrote `./.gitignore` (+14 -0)
> - Wrote `./README.md` (+35 -25)
> - Ran `git check-ignore -v --no-index data/example.png artifacts/model/model.safetensors checkpoints/run/model.safetensors outputs/report.json .cache/huggingface/blob .huggingface/token sample.pt && rg -n "accuracy|improv|result|trained|fine-tun|download" README.md src tests && git diff --check && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q`
> - Ran `git status --short --ignored && find . -maxdepth 3 -type f \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name 'pytorch_model*.bin' \) -print && sed -n '120,225p' README.md && git diff --stat`
>
> </details>
</details>

Verified and fixed against the assignment objectives.

- Real pretrained Swin2SR parameters were updated: one optimization step changed 4,860 values in `swin2sr.first_convolution.weight`.
- The best model was saved as `config.json` plus `model.safetensors`.
- Reloading through `from_pretrained(checkpoint_dir)` reproduced the saved parameter exactly.
- The reloaded checkpoint completed inference with shape `[1, 3, 64, 64] → [1, 3, 128, 128]`.
- These are smoke-verification results, not model-quality claims. The README explicitly says synthetic metrics are not evidence of real-world improvement.
- Generated datasets, reports, checkpoints, weight formats, and local caches are now excluded in [.gitignore](./.gitignore).
- Setup and demonstration instructions were consolidated in [README.md](./README.md).
- No weights or generated datasets currently exist inside the repository.
- Final verification: 12 tests passed; lint, formatting, and diff checks passed.

The fine-tuning and reloading logic is in [train.py](./src/tap_charger_scan/train.py) and [restoration.py](./src/tap_charger_scan/restoration.py).

## Recommended demonstration flow

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,ml]'
```

Verify the offline components, including API behavior and the lightweight training loop:

```bash
ruff check .
ruff format --check .
pytest
```

Generate a small grouped synthetic dataset:

```bash
tap-generate-data data/smoke \
  --payload-count 6 \
  --variants-per-payload 1 \
  --seed 42
```

Evaluate OpenCV against the actual pretrained Hugging Face model:

```bash
tap-evaluate data/smoke/manifest.jsonl \
  --output artifacts/pretrained-evaluation.json
```

Perform one genuine fine-tuning step and save a reloadable checkpoint:

```bash
tap-train data/smoke/manifest.jsonl artifacts/swin2sr-qr-smoke \
  --epochs 1 \
  --batch-size 1 \
  --max-steps 1 \
  --device cpu
```

Finally, reload that checkpoint and compare all three systems:

```bash
tap-evaluate data/smoke/manifest.jsonl \
  --fine-tuned-checkpoint artifacts/swin2sr-qr-smoke \
  --output artifacts/fine-tuned-evaluation.json
```

That sequence demonstrates ingestion tests, baseline decoding, synthetic data and leakage-safe splits, actual Hugging Face inference, parameter updates, checkpoint serialization/reloading, and comparative evaluation.

> Can you just walk me through the structure of the repository and explain what are some of these files that you added that I didn't explicitly ask for (like ports.py). I am not saying I want to remove them now, just need to have the complexity justified by the assignment.

The repository has two deliberately separate workflows:

```text
Scan collection:
API → validation/use case → PostgreSQL + S3-compatible storage

Model improvement:
stored/labeled data or synthetic data → manifest → training → evaluation
```

## Root files

- [README.md](./README.md) explains the design, assumptions, and demonstration commands.
- [pyproject.toml](./pyproject.toml) is the single dependency and tooling definition. The ML packages are optional so running the API does not load PyTorch or Transformers.
- [.gitignore](./.gitignore) prevents virtual environments, generated datasets, evaluation reports, caches, and model checkpoints from entering Git.

## API and storage side

- [api.py](./src/tap_charger_scan/api.py) defines the FastAPI application:
  - `POST /v1/scans` stores a scan.
  - `PUT /v1/scans/{scan_id}/label` associates verified ground truth.
  - It also constructs the production PostgreSQL and S3 adapters.

- [schemas.py](./src/tap_charger_scan/schemas.py) contains Pydantic API contracts. Keeping external request validation separate from database models prevents the HTTP payload from implicitly becoming the database schema.

- [service.py](./src/tap_charger_scan/service.py) contains the actual ingestion rules:
  - validates file type and dimensions;
  - calculates the checksum;
  - handles idempotency;
  - writes the image and metadata;
  - cleans up the image if the database write fails;
  - associates verified labels.

  This keeps business behavior testable without sending HTTP requests or connecting to AWS.

- [models.py](./src/tap_charger_scan/models.py) contains both the small domain records and SQLAlchemy table definitions for scans and verified labels. It describes what PostgreSQL would contain but does not create infrastructure or migrations.

- [repositories.py](./src/tap_charger_scan/repositories.py) implements PostgreSQL operations.

- [object_store.py](./src/tap_charger_scan/object_store.py) implements the S3-compatible image operations.

### Why `ports.py` exists

[ports.py](./src/tap_charger_scan/ports.py) defines two small Python `Protocol` interfaces:

```text
ScanRepository
ImageObjectStore
```

It is effectively the contract between the application logic and external infrastructure.

This is useful here for three concrete reasons:

1. The assignment says to write code as though PostgreSQL and S3 exist without implementing that infrastructure.
2. Automated tests must run without PostgreSQL or S3, so they provide in-memory implementations of the same contracts.
3. The real-data exporter needs to retrieve verified records and image bytes without being coupled directly to SQLAlchemy or boto3.

It is not intended as a large abstraction layer or a generic framework. It is approximately the smallest way to express: “the service needs something capable of these operations, but it does not care whether that is PostgreSQL, S3, or a test double.”

We could technically place these protocols in `service.py`, but that would mix infrastructure contracts with use-case implementation. A separate `ports.py` makes the dependency direction visible:

```text
service → ports ← PostgreSQL/S3 adapters
```

That is a modest complexity cost and, in this assignment, defensible because external infrastructure is intentionally unavailable.

- [config.py](./src/tap_charger_scan/config.py) reads database, bucket, endpoint, and upload-limit settings from environment variables. This avoids hardcoding deployment-specific values.

## Baseline and dataset side

- [baseline.py](./src/tap_charger_scan/baseline.py) contains the OpenCV QR decoder. It tries:
  1. the original image;
  2. grayscale CLAHE as the single fallback.

- [synthetic.py](./src/tap_charger_scan/synthetic.py) generates QR codes with known payloads and applies blur, noise, downsampling, and uneven lighting. It assigns charger groups to splits before creating degraded variants.

- [dataset.py](./src/tap_charger_scan/dataset.py) defines the shared JSONL manifest and grouped splitting logic. It also provides the path from future real data to the experiment:

```text
verified database record + S3 image → same manifest used by evaluation/training
```

This file is important because synthetic and real data should not require separate training and evaluation implementations.

## Model side

- [restoration.py](./src/tap_charger_scan/restoration.py) is the focused Hugging Face wrapper. It:
  - identifies and pins the selected Swin2SR checkpoint;
  - loads pretrained or fine-tuned weights;
  - converts images to tensors;
  - runs inference;
  - converts the restored tensor back to an image.

- [train.py](./src/tap_charger_scan/train.py) implements actual fine-tuning:
  - loads train and validation pairs;
  - checks the 64×64 and 128×128 dimensions;
  - loads pretrained parameters;
  - computes L1 restoration loss;
  - performs backpropagation and optimizer updates;
  - saves the best validation checkpoint using Hugging Face’s reloadable format.

- [evaluate.py](./src/tap_charger_scan/evaluate.py) performs the assignment’s final comparison:

```text
original image → OpenCV
original image → pretrained Swin2SR → OpenCV
original image → fine-tuned Swin2SR → OpenCV
```

It reports exact accuracy, incorrect payloads, no-decode failures, and latency.

## Tests

The `tests/` directory mirrors the important boundaries:

- `test_api.py`: ingestion, validation, idempotency, and labels.
- `test_storage.py`: S3 adapter behavior.
- `test_baseline.py`: clean QR decoding.
- `test_dataset.py`: split isolation, dimensions, and real-data export.
- `test_training.py`: tensor dimensions and a genuine optimization step without downloading weights.
- `test_evaluation.py`: metric calculations and pipeline comparison behavior.

So the additional files are primarily separation-of-responsibility files, not additional features. The most arguably optional one is `ports.py`, but it earns its place by letting us represent nonexistent production infrastructure cleanly while keeping the service and tests executable.
