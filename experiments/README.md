# Experiment artifacts

Generated run directories and raw/prepared data are excluded from Git. The
release study is `cohort20-final-v1`: 27 actions, 20 people, four environments,
540 sequences and 160,380 paired frames. The earlier two-action pilot
was stopped and is not efficacy evidence. All twelve arms and original held-out
evaluations completed. Read [the measured findings](../results/README.md),
including inconsistent temporal gains, strong room priors and poor unseen-room
calibration. Original packet-level synchronization was not independently verified.

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

For a new study, install the exact prepared cohort and follow the
[reproduction guide](../docs/REPRODUCE.md). The launcher freezes supplemental
controls before the main suite reaches testing; both destinations must be new:

```sh
uv run python scripts/run_study.py \
  --subject-manifest data/mmfi-cohort20/manifest.json \
  --environment-manifest data/mmfi-cohort20/cross-environment.json \
  --out experiments/my-cohort20-study --controls-out results/my-signal-controls
```

After the new study completes, render its report into a new destination:

```sh
MPLCONFIGDIR=/tmp/ghostmesh-mpl-cache .report-venv/bin/python scripts/report_results.py \
  --suite experiments/my-cohort20-study \
  --out results/my-cohort20-study \
  --provenance docs/cohort-provenance.json
```

See [the protocol](../docs/EXPERIMENTS.md) for frozen teacher/direct-kernel
settings, splits, metrics and inference limits, and
[the results format](../docs/RESULTS_FORMAT.md) for plotting setup and output
hashes. Paired sequence and subject-cluster intervals describe sampling for
fixed checkpoints, not training variability; absolute bars receive no borrowed
intervals. The separate study bundle preserves selected checkpoints and
completed receipts without raw/prepared CSI or unfinished recovery states.
The optional prepared-cohort archive supplies the exact NPZ files independently;
neither archive includes RGB images. See [archive boundaries](../docs/STUDY_BUNDLE.md).
