# Local replay application

The viewer is an offline inspection instrument for exported GhostMesh replay JSON. It uses local JavaScript and system fonts, with no remote dependencies, accounts or API services. The Python server listens on loopback only. Its endpoints expose one selected replay and an exact list of static assets; it accepts no uploads. Opening another JSON file in the viewer reads it inside the browser, without sending it to the server.

Run `python -m ghostmesh serve --replay examples/diagnostic-replay.json` from the project directory, then open the printed localhost URL. The diagnostic replay contains synthetic signals and illustrative joints. It verifies synchronized playback and failure handling, and provides no evidence of CSI pose accuracy.

Use **Open replay** to inspect an exported sequence. Play/pause, seek, step one frame, restart and choose 0.25–2× playback. Reference labels are off by default and can be shown when available. **Fit pose** uses one fixed square viewport derived from model predictions across the sequence, with a minimum 0.35 coordinate span. Reference labels do not influence this crop. Disable it to see the full declared annotation plane. Both views preserve equal x/y scale, so a wide screen never stretches the body. With focus outside a control, Space toggles playback and the left/right arrow keys step frames. No replay autoplays, including with reduced motion enabled.

The large canvas uses the declared normalized 2D annotation plane, with x and y increasing rightward and downward. This is a coordinate convention, not a claim that original camera calibration is known. It draws only the current frame. Invalid CSI windows withhold the pose completely, so an old skeleton never appears to be a current estimate. Low confidence and out-of-domain states retain a visible warning and the exporter's reasons. A domain flag may come from known recording metadata; it does not establish learned detection of unknown rooms. Per-joint scores are uncalibrated Gaussian mass within a fixed image radius, not a probability of correctness. The exporter/model card defines the radius and uncertainty model; held-out calibration must be assessed separately. Source cadence describes timestamp spacing, not inference throughput. Recorded per-frame latency is displayed only when supplied. The CSI panel shows subcarrier amplitude vertically against causal frames horizontally; it is not a Doppler spectrogram.

## Replay schema, version 1

```json
{
  "schema_version": 1,
  "provenance": {
    "kind": "dataset_inference",
    "dataset": "MM-Fi",
    "split": "test",
    "subject": "S05",
    "environment": "E01",
    "sequence": "E01/S05/A01",
    "checkpoint": "path or hash",
    "coordinate_system": "normalized_image",
    "note": "Actual data and experiment provenance"
  },
  "joint_names": ["joint_0", "joint_1"],
  "edges": [[0, 1]],
  "frames": [{
    "time": 0.0,
    "pose": [[0.4, 0.3], [0.6, 0.5]],
    "confidence": [0.8, 0.5],
    "spectrum": [[0.0, 0.5, 1.0]],
    "status": "ok",
    "reasons": [],
    "latency_ms": null
  }]
}
```

Allowed provenance kinds: `synthetic_diagnostic`, `dataset_inference`, `recording_inference`. The last kind is annotation-free inference from a CSI recording. Its subject/environment may be unknown, and its replay does not establish held-out accuracy. Allowed frame states: `ok`, `low_confidence`, `invalid_signal`, `out_of_domain`. Timestamps are strictly increasing nonnegative seconds. A frame may contain optional `ground_truth` coordinates, optional `ground_truth_valid` boolean joint mask, and optional normalized [0,1] rectangular `spectrum`. `pose` may be null or empty only for an invalid-signal frame. `confidence` always contains one finite [0,1] score per joint. Optional top-level `metrics` can carry measured evaluation numbers; the viewer never computes research accuracy from an illustrative animation. Exporter provenance is displayed as supplied and does not constitute independent certification.

Limits: at most 100 named joints, 300 skeleton edges, 200,000 frames and 100 MiB per file. Large experiments should export a bounded qualitative sequence. The full training/evaluation outputs remain the source of aggregate accuracy and throughput evidence.

## Verification

`python -m unittest discover -s tests -p test_server.py -v` exercises real HTTP responses, immutable replay snapshots, static file allowlisting, rejected nonlocal hosts, nonlocal binding rejection, HEAD responses and the absent upload endpoint. `node --test tests/test_replay.mjs` checks replay validation, timestamp-based seeking, irregular cadence, rejected malformed data and diagnostic provenance. Packaged `ghostmesh/web/replay.js` exports these functions independently of the DOM. Visual and interaction inspection must use the current source and actual replay file.

Headless Chrome inspection of the current application verified playback advance, frame seeking, reference and fit controls, speed changes, invalid-signal withholding, low-confidence and domain warnings, and preservation of the previous replay after a malformed file. Desktop (1440 px) and mobile (390 px) layouts showed no horizontal overflow. The reduced-motion mobile inspection did not autoplay; no uncaught JavaScript errors occurred. The diagnostic PNGs in `docs/assets/` show generated application test inputs and are explicitly labeled. They are not empirical pose results.

## Reproduce a release recording

`scripts/capture_demo.mjs` captures the actual application at explicit timestamps from a real `dataset_inference` replay. It seeks the normal viewer, takes screenshots and encodes a silent MP4 and GIF using free Playwright, a browser and FFmpeg. It does not generate or interpolate skeleton motion. A capture refuses an existing output directory and synthetic inputs, records source/replay hashes, checks identical pixels after a backward seek and fully decodes its MP4 before delivery.

```bash
node scripts/capture_demo.mjs \
  --replay ghostmesh/examples/heldout-replay.json \
  --out docs/assets/demo-v1 \
  --python .venv/bin/python \
  --browser "/path/to/Chrome executable" \
  --start 1 --seconds 10 --reference
```

Install Playwright locally as a development tool, or pass `--playwright-module /absolute/path/to/playwright/index.mjs` for an existing installation. FFmpeg is resolved from `PATH`, or can be selected with `--ffmpeg`. The capture owns and stops its temporary localhost server and browser; generated frame scratch is removed only after successful delivery. Inspect the encoded video and its `capture.json` record before using a new recording in the README.
