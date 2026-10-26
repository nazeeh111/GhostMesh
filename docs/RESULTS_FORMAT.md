# Cohort20 results renderer

`scripts/report_results.py` renders the completed `cohort20` suite. It never imports the model runtime, loads a checkpoint, trains, evaluates, or recomputes bootstrap intervals.

Install the optional plotting requirement in a separate environment, keeping an active training environment untouched:

```sh
uv venv --python 3.12 .report-venv
uv pip install --python .report-venv/bin/python -r requirements-plots.txt
.report-venv/bin/python scripts/report_results.py \
  --suite experiments/cohort20-final-v1 \
  --out results/cohort20-v1 \
  --provenance docs/cohort-provenance.json
```

The output directory must be new. The renderer checks all input bindings before making it. A rendering failure retains diagnostic bytes and writes `RENDER_FAILED.txt`; that directory must not be published. A successful run writes `renderer_manifest.json` last, after rehashing inputs to catch concurrent modification.

## Required evidence

- `process.json` must say `completed` / `evaluated`.
- `suite_report.json` must bind to the exact `preregistration.json` bytes.
- Twelve registered variants must be present. Primary last-frame/temporal pairs use exactly seeds 2027 and 2028 in both subject and environment protocols; the four cross-subject ablations use seed 2027.
- Current runtime modules and runner must match preregistered SHA-256 fingerprints. Reporting code can be developed independently, but changing frozen model/runtime source invalidates report rendering.
- Both prepared manifests must match their registered hashes and the provenance receipt’s cohort, frame counts, and prepared sequence hashes. The cross-subject receipt is bound to the exact subject-manifest bytes. E04 is the environment target.
- Every completed training/test report must bind to the registered config, manifest, runtime source, runtime environment and exact checkpoint bytes. All test and mean-pose reports must explicitly say `dataset_experiment` and `evaluated` on split `test`. Synthetic software-check reports are refused.
- The suite summary must agree with individual tests. All thirteen expected paired comparisons and both interval types must exist. Coverage, finite metrics, calibration event and timing scope are checked.

Checkpoint hashing does not unpickle model content or independently reproduce scores. Provenance receipts compile importer hash records; they do not claim a new read of all raw files. Fingerprints are artifact identity evidence, not proof that a scientific model is correct.

## Outputs

- `results.md`: readable method, protocol, raw metrics, complete paired intervals, coverage, calibration, timing and limitations.
- `results.csv`: one row per variant/protocol, including static priors, all three PCK thresholds, NJE, valid counts, coverage, ECE, and measured latency.
- `comparisons.csv`: one row per metric and cluster type from each stored paired comparison, preserving the interval values and bootstrap metadata.
- `results.json`: full test, mean-pose and training reports, preregistration, provenance, flattened metrics and stored comparisons. The production purpose is `dataset_experiment`.
- `primary_metrics.svg/.png`: three PCK thresholds plus NJE for both protocols, showing both seeds separately and the static prior. Absolute PCK axes start at zero. No absolute-score error bars are fabricated from paired intervals.
- `paired_changes.svg/.png`: primary temporal-plus-teacher minus last-frame changes with the stored sequence-block and subject-cluster intervals. Difference axes have a zero reference line.
- `ablations.svg/.png`: seed-2027 cross-subject PCK@.2 and NJE. Other thresholds remain in the raw tables.
- `renderer_manifest.json`: input and output SHA-256 fingerprints, renderer code hash, Python/Matplotlib versions, timestamp, command and scope. The manifest itself is excluded from its output hash map to avoid a circular hash.

Absolute scores pool valid joints. Paired changes weight sequence means equally, so their point estimates can differ from subtraction of the pooled scores. Intervals reflect held-out sampling for fixed checkpoints, not training-seed variability. Four and five held-out people give limited subject-cluster sensitivity. Two training seeds are individual sensitivity checks, not a stable variance estimate.

ECE uses an absolute normalized-image radius of .05; PCK uses a fraction of each target frame’s bounding-box diagonal. These are different correctness events. Neural latency is batch-one forward only, excludes preprocessing/I/O/network/rendering, and FPS is the reciprocal of median forward time.

## Software tests and visual checks

`tests/test_report_results.py` creates fabricated JSON and placeholder checkpoint bytes exclusively in pytest temporary directories. It executes no model code. Its isolated `root` injection labels generated images, Markdown, machine JSON and renderer manifests as software-test diagnostics. These files are not empirical research results. The production CLI offers no fixture mode and still refuses non-dataset purposes.

Run only the reporting tests in the isolated environment:

```sh
uv pip install --python .report-venv/bin/python pytest==8.4.2
.report-venv/bin/python -m pytest -q tests/test_report_results.py
```

Inspect a fresh successful diagnostic render to check layout, then inspect the actual successful final render before external publication. Diagnostic previews cannot establish final result quality or model performance.
