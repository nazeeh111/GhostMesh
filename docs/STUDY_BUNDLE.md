# Frozen study bundle

`scripts/package_study.py` creates a compact, independently renderable archive
of a completed cohort20 study. All twelve arms and their held-out evaluations are complete. The release
archive preserves their selected models and exact receipts; the separate
prepared-cohort archive supports independent re-evaluation and retraining.
Download both from the [v0.1.0 release](https://github.com/nazeeh111/GhostMesh/releases/tag/v0.1.0).

The packager uses only Python's standard library and the reporting validator.
It calls `report_results.load_suite()` before creating an archive, requiring
completed status and matching source, config, manifest, checkpoint and report
bindings. It never imports PyTorch, loads a model, evaluates data or acquires
recordings. Do not package a running or incomplete suite.

## Build after completion

From the repository root, with a new output path:

```sh
python3 scripts/package_study.py \
  --suite experiments/cohort20-final-v1 \
  --out dist/ghostmesh-study-v1.tar.gz \
  --provenance docs/cohort-provenance.json
```

The command prints the finished archive's SHA-256 and byte size for the release
receipt. Existing destinations and their `.partial` paths are refused.
Validation failures create no archive. Failures after writing starts retain a
`.partial` file for diagnosis; it is not a distributable artifact. Choose a new
path or explicitly resolve that failed artifact before retrying.

## Contents and evidence boundary

Under one `ghostmesh-study-v1/` prefix, the archive preserves exact source bytes
for both full prepared manifests, all twelve selected `best.pt` checkpoints,
frozen configs, each arm’s standalone `config.json`, `history.json` and
`run_identity.json`, training and test reports, both mean-pose reports,
preregistration, suite/process receipts and cohort provenance. It includes the
five frozen runtime modules, suite runner, results renderer, plotting
requirement, packagers, protocol/reporting documentation and license notices.
Supplemental control sources, frozen plan and completed diagnostics, integrated
findings and the four report-ranked qualitative cases are also retained.
`README_STUDY.md` provides extraction, hash-verification and rendering commands.

Raw or prepared CSI arrays, raw RGB, prepared NPZ files, window caches,
`latest.pt`, optimizer/random-state recovery and training dependencies are
excluded. This bundle supports validation and re-rendering of recorded results
without raw CSI or PyTorch. It does not independently reproduce predictions,
retrain models or recompute confidence intervals. Full manifests retain source
hash receipts, so the archive is intended as a downloadable release asset,
not a Git-tracked replacement for the compact provenance summary.

`bundle_manifest.json` lists every other member's relative path, SHA-256 and
byte size. It excludes itself to avoid a circular hash. All inputs must resolve
inside the project root. They are checked when read, copied without JSON
rewriting and rehashed before the completed archive is published locally.
Standalone history epochs must equal the training report’s history, and each
standalone config must equal its registered config. Run identity is preserved
as a hash-bound receipt without adding another validation contract.
Members are sorted regular files with modification time zero, mode `0644`,
empty owner names and user/group IDs zero; gzip has time zero and no source
filename. Identical inputs on the same Python/compression stack produce the
same archive bytes. Different compression-library versions can differ.
Internal hashes establish consistency; verify the release-provided archive
hash for identity. No authenticity signature is implied.

## Extract and re-render

Extract into a new directory and follow its `README_STUDY.md`:

```sh
mkdir ghostmesh-study-extracted
tar -xzf ghostmesh-study-v1.tar.gz -C ghostmesh-study-extracted
cd ghostmesh-study-extracted/ghostmesh-study-v1
uv venv --python 3.12 .report-venv
uv pip install --python .report-venv/bin/python -r requirements-plots.txt
MPLCONFIGDIR=/tmp/ghostmesh-mpl-cache .report-venv/bin/python scripts/report_results.py \
  --suite experiments/cohort20-final-v1 \
  --out results/cohort20-v1 \
  --provenance docs/cohort-provenance.json
```

The commands use `uv` to select Python 3.12 explicitly, avoiding older macOS
default Python versions. Only Matplotlib and its dependencies are needed for
the plotting environment.
The output directory must be new. The renderer reads the exact frozen files
from the extracted project layout and writes fresh tables, figures and a
renderer manifest. Inspect actual final outputs before sharing.

See [the protocol](EXPERIMENTS.md) and [results format](RESULTS_FORMAT.md) for
cohort limits, static-prior comparisons, pooled metrics versus paired changes,
calibration event and measured timing scope. Software has the MIT license;
MM-Fi-derived artifacts retain CC BY-NC 4.0 terms under the included
`NOTICE.md` and `data/LICENSE.md`.
