# GhostMesh

**Human pose from Wi-Fi CSI. Camera-free inference, inspectable uncertainty, reproducible experiments.**

GhostMesh turns synchronized channel-state information into a 17-joint 2D
skeleton. Its local application plays real recorded inference with a signal
map, joint uncertainty, reference overlays and explicit failure states. No
boards, camera, cloud account or billed API are required for replay.

**Research status:** software checks have passed and the final experimental
cohort is prepared: 27 actions, 20 people, four environments and 160,380 frames.
The full 40-person expansion is blocked by a verified Google Drive quota.
Final held-out experiments are being frozen; no accuracy improvement or “10×”
claim is established yet. See
[experiment scope](docs/EXPERIMENTS.md) and [data provenance](docs/DATA_ACCESS.md).

## Run locally

Python 3.12 and [uv](https://docs.astral.sh/uv/) are the tested environment.
From a checkout:

```bash
uv sync --locked --extra test
uv run ghostmesh serve
```

Open the localhost address printed by the command. The release replay is
selected by default. While experiments are unfinished, use an explicitly
labeled diagnostic with `--replay examples/diagnostic-replay.json`.

The viewer provides play/pause, frame stepping, timestamp seeking, playback
speed, reference labels, a fixed sequence viewport and a full annotation-plane
view. Invalid signal clears the prediction. Unknown domains and low confidence
remain visible. Uploading a replay stays in the browser.

## What is original

- An independently implemented CSI-to-pose pipeline, with exact frame pairing,
  causal windows, per-link robust normalization and integrity-checked datasets.
- A compact learned spectral encoder and causal temporal network, paired with a
  last-frame encoder control. Gaussian joint uncertainty is evaluated rather
  than presented as a guaranteed probability of correctness.
- A source-only teacher consistency strategy, tested through normalization,
  augmentation and consistency ablations. It does not adapt on held-out people
  or environments.
- An experiment runner that freezes settings before testing, selects epochs
  with validation labels, compares a no-signal pose prior, and resamples whole
  sequences and subjects rather than treating overlapping frames as independent.
- A local application that exposes the recording, checkpoint, split and failure
  evidence behind each skeleton.

This is an original implementation and experimental study. The components
build on established ideas; no claim of being the first CSI pose system is made.
[Model card](docs/MODEL_CARD.md) · [Research references](docs/REFERENCES.md)

## Reproduce the data path

MM-Fi supplies paired CSI and 2D teacher keypoints under CC BY-NC 4.0. GhostMesh
downloads only those streams, verifies archive contents and pairs files by
frame number. It does not load camera images or convert the separate 3D labels
into purported 2D truth.

```bash
python scripts/fetch_mmfi.py --env E01 --actions A02 A06 --inventory
python scripts/fetch_mmfi.py --env E01 --actions A02 A06 \
  --out data/mmfi-raw --max-payload-mb 400
```

Repeat for E02–E04. Follow [the data guide](docs/DATA_ACCESS.md) for the complete
27-action acquisition, coordinate conventions, consent record and split rules.
Preparation refuses an existing output directory:

```bash
uv run ghostmesh prepare-mmfi \
  --source data/mmfi-raw --out data/mmfi-prepared \
  --width 640 --height 640 --fps 10 \
  --provenance docs/mmfi-source.json
uv run ghostmesh audit --manifest data/mmfi-prepared/manifest.json
```

The square reference plane is an explicit isotropic normalization convention,
not a recovered camera resolution. Inspect the coordinate bounds of newly
acquired actions before choosing it. Distances retain pixel-space geometry.

## Train, evaluate and replay

```bash
uv run ghostmesh train \
  --manifest data/mmfi-prepared/manifest.json \
  --config configs/cohort20-temporal.json --out experiments/my-run
uv run ghostmesh evaluate \
  --manifest data/mmfi-prepared/manifest.json \
  --checkpoint experiments/my-run/best.pt --out experiments/my-run/test.json
uv run ghostmesh export \
  --manifest data/mmfi-prepared/manifest.json \
  --checkpoint experiments/my-run/best.pt --out examples/my-replay.json
uv run ghostmesh serve --replay examples/my-replay.json
```

Use a configuration matching the prepared data's protocol and intended scope.
The `cohort20` configurations describe the balanced 27-action, 20-person cohort;
`subset` configurations retain the earlier two-action pilot.
For a controlled comparison, run the frozen suite instead of selecting models
by their test scores:

```bash
uv run python scripts/run_suite.py \
  --subject-manifest data/mmfi-prepared/manifest.json \
  --environment-manifest data/mmfi-prepared/cross-environment.json \
  --profile cohort20 --out experiments/my-suite
```

For annotation-free inference, provide only CSI amplitude and timestamps in a
compatible NPZ recording. Sensor layout is checked against the checkpoint:

```bash
uv run ghostmesh infer-recording --recording examples/heldout-csi.npz \
  --checkpoint models/ghostmesh.pt --out examples/recording-replay.json
```

This mode does not read pose labels. Its domain remains explicitly unverified.
It cannot make an ESP32 recording compatible with MM-Fi weights by reshaping it.

## Validation

```bash
uv run pytest -q
node --test tests/test_replay.mjs
```

Tests cover causal dependencies, gradients, missing-data and timestamp failure
paths, exact frame joins, manifest integrity, split leakage, validation-only
selection, metric definitions, uncertainty, paired comparisons, annotation-free
inference, ESP32 parsing, actual HTTP responses and replay controls.

PCK uses 20% of each frame's valid reference bounding-box diagonal. Normalized
joint error is Euclidean joint error divided by that diagonal. Reports retain
excluded-window coverage, per-person/environment/sequence scores, confidence
reliability and measured processing latency. These 2D metrics are not directly
comparable with MM-Fi's published 3D millimeter results. The recorded stream is
10 Hz regardless of model processing FPS.

## ESP32 mode

Hardware acquisition is a separate, opt-in signal path. It requires a real
ESP-CSI device, a chosen serial port and permission from recorded participants.
No hardware accuracy or cross-device transfer is claimed.

```bash
uv sync --locked --extra hardware
uv run ghostmesh esp-record --port /dev/cu.YOUR_DEVICE --out data/my-session \
  --seconds 60 --consent-note "Consenting participants in the authorized test area"
uv run ghostmesh esp-convert --source data/my-session/csi.csv \
  --out data/my-session/amplitude.npz
```

[Hardware guide and limits](docs/HARDWARE.md)

## Repository map

| Path | Purpose |
| --- | --- |
| `ghostmesh/` | Preparation, model, training, evaluation, inference and local server |
| `ghostmesh/web/` | Browser application, without a CDN or build service |
| `configs/` | Versioned experiment settings and source provenance |
| `scripts/` | Bounded dataset acquisition, suite execution and demo capture |
| `tests/` | Numerical, integrity, inference, parser and application checks |
| `docs/` | Model, data, application, hardware and experiment records |

Original software: [MIT](LICENSE). MM-Fi data, derived weights and replay assets:
[separate noncommercial terms](data/LICENSE.md). [Attribution](NOTICE.md).
Use consenting recordings and authorized sites. GhostMesh is a single-person
2D research system, not a surveillance, identification or medical safety product.
