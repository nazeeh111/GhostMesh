# Reproduce GhostMesh

The release supports CSI-only sample recomputation, independent held-out evaluation using the exact prepared cohort, re-rendering archived study receipts, and full retraining. Sample recomputation and receipt rendering do not require the full cohort. The optional prepared-data archive supplies the original cohort without upstream acquisition. Full retraining requires that cohort, both exact split manifests and their matching provenance receipt. Provider quota limits can prevent fresh acquisition; see [data access](DATA_ACCESS.md). No physical Wi-Fi device or capture is needed for these recorded-data paths.

From the repository root, install the locked Python 3.12 training environment:

```sh
uv sync --locked --extra test
```

## Recompute the shipped example

```sh
uv run python scripts/reproduce_demo.py
```

This checks all valid pose/confidence values and timestamps against the bundled held-out replay, using only `examples/heldout-csi.npz` and `models/ghostmesh.pt`. The NPZ must contain exactly CSI amplitude and timestamps, without pose annotations. The default tolerance is 2e-5; measured timing and unknown-domain metadata are outside the comparison. To save a fresh replay at a new path:

```sh
uv run ghostmesh infer-recording --recording examples/heldout-csi.npz \
  --checkpoint models/ghostmesh.pt --out examples/my-recomputed-replay.json
```

## Independently evaluate the released checkpoints

Download the optional [prepared cohort archive](https://github.com/nazeeh111/GhostMesh/releases/download/v0.1.0/ghostmesh-mmfi-cohort20-v1.tar.gz) and compare its SHA256 with the release receipt. It contains the exact 540 NPZ sequences and both original manifests: 20 people, 27 actions, four environments and 160,380 paired frames. These data and their supplied model-generated 2D labels retain [CC BY-NC 4.0](../data/LICENSE.md) attribution and noncommercial terms. They include no RGB images or 3D arrays.

From the repository root, extract into a new folder:

```sh
mkdir ghostmesh-data-extracted
tar -xzf ghostmesh-mmfi-cohort20-v1.tar.gz -C ghostmesh-data-extracted
```

Verify all member hashes using the extracted `README_DATASET.md`. Then, from the repository root, copy the prepared folder and evaluate both released models:

```sh
uv run python -c "import shutil; shutil.copytree('ghostmesh-data-extracted/ghostmesh-mmfi-cohort20-v1/data/mmfi-cohort20', 'data/mmfi-cohort20')"
uv run ghostmesh evaluate --manifest data/mmfi-cohort20/manifest.json \
  --checkpoint models/ghostmesh.pt --out results/my-primary-test.json
uv run ghostmesh evaluate --manifest data/mmfi-cohort20/cross-environment.json \
  --checkpoint models/ghostmesh-environment.pt --out results/my-environment-test.json
```

The copy refuses an existing destination; preserve any existing prepared data. Evaluation outputs must also be new. These commands recompute dense held-out predictions and metrics, rather than re-rendering saved scores. Timing and small numerical differences can depend on hardware/runtime. All twelve validation-selected checkpoints are in the separate study archive; evaluate each against its registered protocol manifest. Reference labels are supplied teacher estimates, not independently verified manual ground truth. Nominal 10 Hz timestamps and the 640-square annotation plane retain the [documented limits](DATA_ACCESS.md).

## Run a new complete study with supplemental controls

Use the exact prepared archive installed above, so the subject-manifest hash and both manifests' sequence records match the unchanged `docs/cohort-provenance.json` release receipt. A fresh `prepare-mmfi` import can change manifest metadata or NPZ compression bytes even when source arrays match. It therefore cannot be assumed interchangeable with the released weights or receipt. Treat a fresh import as a separate study with its own matching provenance receipt in a separate checkout; the current full-suite validator reads that receipt at `docs/cohort-provenance.json`. Preserve the release receipt and frozen plans.

For the exact released cohort, use new output directories and unchanged registered configs/runtime sources:

```sh
uv run python scripts/run_study.py \
  --subject-manifest data/mmfi-cohort20/manifest.json \
  --environment-manifest data/mmfi-cohort20/cross-environment.json \
  --out experiments/my-cohort20-study \
  --controls-out results/my-signal-controls
```

The launcher starts the unchanged cohort20 CPU runner with all twelve 40-epoch arms. After registration appears, it publishes any missing registered configs as exact canonical JSON using an exclusive atomic link. An existing config is preserved and checked byte-for-byte and semantically. This permits the supplemental control plan to freeze before the original runner reaches any test evaluation. The launcher waits for the unchanged main study to finish, then runs the existing frozen control evaluation. It does not alter model settings, inspect partial test scores, select models, acquire data or publish results. The controls remain descriptive; read [their frozen scope](SIGNAL_CONTROLS.md).

A new full study has its own preregistration and supplemental plan. It does not amend the release study or its already frozen plan. CPU/runtime versions and data/source hashes are checked by the existing runners. Different hardware/runtime versions or newly prepared bytes can change results or fail identity checks; re-rendering archived results is distinct from independent retraining.

The launcher refuses existing or nested output destinations. Startup waits at most 120 seconds for registration/config-directory creation and fails if the child exits early. On startup/freeze failure or interruption, it interrupts only its owned main child, waits for exit and preserves all suite files. If the child does not respond, it terminates that child. Training keeps atomic epoch-boundary recovery checkpoints; an unfinished epoch can be lost. The launcher itself has no resume mode.

## Recover an interrupted study

Keep the original source, configs, data, runtime, suite directory and supplemental `plan.json`. Do not start a second writer while the original child is active. After it has exited, recover through the unchanged runner:

```sh
uv run python scripts/run_suite.py --resume \
  --subject-manifest data/mmfi-cohort20/manifest.json \
  --environment-manifest data/mmfi-cohort20/cross-environment.json \
  --profile cohort20 --device cpu --out experiments/my-cohort20-study
uv run python scripts/run_signal_controls.py \
  --suite experiments/my-cohort20-study \
  --subject-manifest data/mmfi-cohort20/manifest.json \
  --environment-manifest data/mmfi-cohort20/cross-environment.json \
  --out results/my-signal-controls
```

Reuse the same frozen control plan; do not refreeze after test reports exist. The original runner verifies identity and resumes available complete epochs. If startup failed after registration but before creating a control plan, while the main runner is stopped and before any test report exists, run:

```sh
uv run python -c "from pathlib import Path; from scripts.run_study import materialize_configs; materialize_configs(Path('experiments/my-cohort20-study'))"
uv run python scripts/run_signal_controls.py --freeze \
  --suite experiments/my-cohort20-study \
  --subject-manifest data/mmfi-cohort20/manifest.json \
  --environment-manifest data/mmfi-cohort20/cross-environment.json \
  --out results/my-signal-controls
```

Then use the original `--resume` and control-evaluation commands above. If registration itself is absent, preserve the failed directory and choose a new study/output pair. If test evaluation happened without a control freeze, a pre-test diagnostic registration cannot be retroactively created. Existing diagnostic result files are preserved and replacement is refused.

## Validate and re-render the study archive

Follow [STUDY_BUNDLE.md](STUDY_BUNDLE.md) and the archive's `README_STUDY.md` to verify hashes and render stored tables/figures in a separate plotting environment. The bundle includes frozen manifests, selected checkpoints and completed receipts. It excludes raw/prepared CSI, caches and optimizer/RNG recovery states. Re-rendering stored scores and paired intervals is not independent prediction evaluation, interval recomputation or retraining. The CSI-only example above independently recomputes one released recording; it does not reproduce every study endpoint.
