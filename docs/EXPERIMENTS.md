# Cohort20 experimental protocol

The release study is the frozen `cohort20` suite at
`experiments/cohort20-final-v1`. All twelve training arms, dense held-out evaluations, static priors and paired
comparisons completed successfully. Measured findings are in [the integrated results](../results/README.md). Settings, software checks, partial runs and
artifact directories do not establish sensing efficacy. The earlier A02+A06
pilot was stopped and superseded by this study; it is not efficacy evidence
and its outputs must not be relabeled as cohort20 results.

## Cohort and held-out protocols

The prepared cohort contains **27 actions, 20 people, four environments,
540 sequences and 160,380 paired frames**, with 297 frames per sequence.
It includes the people whose complete 27-action recordings were available
before the documented provider quota blocked the 40-person expansion. It is
not the full 40-person dataset. All evaluated actions occur in training.

| Protocol | Training | Validation | Final test |
|---|---|---|---|
| Cross subject | 12 people; 324 sequences | 4 people; 108 sequences | 4 people; 108 sequences |
| Cross environment | 12 people; 324 sequences in E01–E03 | 3 people; 81 sequences in E01–E03 | 5 people; 135 sequences in E04 |

People are disjoint across training, validation and test. Cross-subject splits
retain all four environments. E04 is excluded from environment-protocol
training and validation. Environment and participant identity change together,
so the E04 test measures combined room/person transfer, not isolated room
invariance. No unseen-action, 3D or cross-dataset conclusion follows.

The manifests record exact sequence assignments. The
[cohort provenance receipt](cohort-provenance.json) binds the subject manifest,
prepared sequence hashes and importer-recorded raw-source hashes. Its evidence
basis is receipt compilation, not an independent second read of every raw
source file. Dataset licensing, annotation provenance and consent basis are
recorded there and in [the model card](MODEL_CARD.md).

## Frozen comparisons and training

The suite contains **12 neural arms, each with a 40-epoch budget**:

- Last-frame encoder and temporal-plus-teacher pairs, each at seeds 2027 and
  2028, for both protocols: eight arms.
- Cross-subject removal of teacher consistency, normalization or augmentation,
  plus raw-coordinate consistency: four ablations, seed 2027 only.

All arms use an eight-frame causal window, packet-moment encoder of width 64,
batch size 64, one CPU thread, training/validation stride 8 and final-test
stride 1. Temporal arms use the frozen direct temporal kernel and dilations
1, 2 and 4. The last-frame encoder shares the same causal-window preprocessing
and normalization as the combined model; it is not a raw history-free control.
The no-normalization arm is the declared exception to normalization.

The combined strategy uses source-only perturbation and an exponential moving
average teacher. The teacher sees unperturbed source examples and evolves only
from source training; validation and test examples update neither network.
The frozen primary consistency loss scales coordinate differences by detached
teacher variance bounded to `[0.05², 0.5²]`, with weight 0.2. Raw consistency is
retained as a separate ablation. Removing augmentation also removes student/
teacher perturbation diversity, so that ablation changes a coupled mechanism.
Settings are frozen; teacher weights are not held constant during training.

The runner writes preregistration before training, completes **every arm's
training before computing final-test predictions or scores**, and selects each checkpoint
only by validation normalized joint error. It then evaluates the frozen
checkpoints and the training-label mean-pose prior on matched test windows.
The static prior consumes no CSI input or action IDs and fits no held-out
labels. Test scores never select settings, an epoch, or a model.

Preregistration binds configs, manifests, runtime modules, runner and runtime
environment. Reports bind the exact checkpoint, config, source and manifest
bytes. Resuming requires identical frozen inputs and verifies completed
artifacts; epoch recovery retains student, teacher, optimizer and random-state
information. No partial final-test inspection is used to revise the study.

## Metrics and uncertainty

PCK@.05, .1 and .2 are valid-joint success fractions at those multiples of each
frame's valid ground-truth 2D bounding-box diagonal. Normalized joint error
(NJE) is mean Euclidean error divided by that diagonal. Coordinates use a
declared isotropic 640×640 annotation plane; neither metric is millimeters or
3D error. Degenerate boxes are excluded and counted. Reports disclose
candidate, accepted, rejected and metric-eligible windows and valid joint N;
overlapping windows are not independent observations.

Absolute metrics pool valid joints. Paired changes instead weight sequence
means equally and report candidate minus reference. The suite's paired
percentile-bootstrap 95% intervals resample whole sequences; subject-cluster
sensitivity retains all action sequences for each resampled person. These
intervals describe held-out sampling for fixed checkpoints, not training-seed
variance or a causal mechanism. Four or five held-out people limit cluster
inference. Two training seeds are sensitivity checks. Paired-change intervals
must not become error bars on absolute scores.

Expected calibration error uses the event that joint Euclidean error is at
most .05 in normalized image coordinates. This absolute-radius event differs
from bounding-box-normalized PCK@.05. Gaussian radial confidence is evaluated
without post-hoc calibration and need not be reliable. Model-only latency
reports measured median, p95 and reciprocal-median FPS for batch-one forward
calls, after warmup; it excludes preprocessing, disk, network and rendering.

## Reproduction and release

Use the exact prepared cohort archive and matching provenance receipt described
in [the reproduction guide](REPRODUCE.md). From the repository root, start a new
study with the launcher so supplemental controls freeze before testing. Both
output destinations must be new. Recovery uses the unchanged suite runner's
`--resume` path and the same frozen control plan; see the guide.

```sh
uv run python scripts/run_study.py \
  --subject-manifest data/mmfi-cohort20/manifest.json \
  --environment-manifest data/mmfi-cohort20/cross-environment.json \
  --out experiments/my-cohort20-study --controls-out results/my-signal-controls
```

After that new suite completes, render its report into a separate new path:

```sh
.report-venv/bin/python scripts/report_results.py \
  --suite experiments/my-cohort20-study \
  --out results/my-cohort20-study \
  --provenance docs/cohort-provenance.json
```

[Results format and plotting setup](RESULTS_FORMAT.md) describes the separate
reporting environment, strict artifact checks, raw tables and scientific
figures. The [completed release results](../results/README.md) retain negative
and ambiguous comparisons alongside favorable ones. Temporal-plus-teacher
gains are inconsistent, room priors are strong, and unseen-room confidence
is unreliable. These findings do not establish robust dynamic pose recovery.
The separate [study bundle](STUDY_BUNDLE.md) preserves the original completed
reports; rendering those receipts is distinct from rerunning predictions.
