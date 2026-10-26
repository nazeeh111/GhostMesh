# Experiment artifacts

Generated run directories and raw/prepared data are excluded from Git. The
release study is `cohort20-final-v1`: 27 actions, 20 people, four environments,
540 sequences and 160,380 synchronized frames. The earlier two-action pilot
was stopped and is not efficacy evidence. The final report and release study
bundle are pending completion and review.

The frozen suite has 12 neural arms with 40 epochs each. Primary last-frame/
temporal-plus-teacher pairs use seeds 2027 and 2028 in cross-subject and E04
transfer protocols; the four cross-subject ablations use seed 2027. Subject
splits contain 12/4/4 people; environment splits contain 12/3/5 people with E04
held out. Room and participant changes are confounded; every test action is
seen during training. Windows contain eight causal frames, with stride 8 for
training/validation and stride 1 for test.

The runner writes `preregistration.json` before training, completes all arms'
source-only training before decoding final-test windows, selects checkpoints
by validation NJE, then evaluates those checkpoints and the training-only
static mean-pose prior. Configs, runtime source, manifests, checkpoints and
reports carry matching fingerprints. `suite_report.json` and completed process
status mark successful suite completion; a directory or checkpoint alone does
not. Partial final-test scores are not used to revise settings.

Reproduce from the repository root with a new destination, or add `--resume`
only for an identical frozen suite:

```sh
uv run python scripts/run_suite.py \
  --subject-manifest data/mmfi-cohort20/manifest.json \
  --environment-manifest data/mmfi-cohort20/cross-environment.json \
  --profile cohort20 --device cpu --out experiments/cohort20-final-v1
```

After completion, render the planned study report:

```sh
MPLCONFIGDIR=/private/tmp/ghostmesh-mpl-cache .report-venv/bin/python scripts/report_results.py \
  --suite experiments/cohort20-final-v1 \
  --out results/cohort20-v1 \
  --provenance docs/cohort-provenance.json
```

See [the protocol](../docs/EXPERIMENTS.md) for frozen teacher/direct-kernel
settings, splits, metrics and inference limits, and
[the results format](../docs/RESULTS_FORMAT.md) for plotting setup and output
hashes. Paired sequence and subject-cluster intervals describe sampling for
fixed checkpoints, not training variability; absolute bars receive no borrowed
intervals. Selected small model/replay artifacts and the study bundle require
final verification before release. Raw datasets and unfinished runs remain
outside the release bundle.
