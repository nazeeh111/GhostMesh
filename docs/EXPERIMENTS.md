# Experimental protocol

Current status: implementation checks passed; real pilot training is live.
Results are not yet claimed. The initial validation-only timing run trained
the final-frame control for one epoch. It did not evaluate test labels.
The preregistered pilot is under `experiments/mmfi-pilot-v1` locally; generated
raw/prepared datasets and in-progress runs are excluded from Git history.

## Required evidence

1. Complete exact-frame pairing, finite 2D annotations, source/data license and
   consent basis, sensor layout, content hashes and split audit.
2. Separate source/validation/final-test people; hold E04 out of environment
   development. Environment/person changes remain confounded in MM-Fi.
3. Reproducible last-frame neural and no-signal mean-pose baselines. The
   last-frame encoder shares causal-window normalization, so it is not a raw
   history-free baseline. Compare it against the temporal/no-consistency
   configuration to isolate the temporal encoder more closely.
4. Fixed settings and seed budget before final test evaluation. Best epochs
   use validation normalized joint error. Test scores never choose an epoch.
5. PCK, normalized joint error, joint-level results, window/metric coverage,
   per-domain performance, confidence reliability and measured latency.
6. Normalization, augmentation and teacher-consistency ablations; repeated
   training seeds; paired sequence and subject-cluster uncertainty intervals.
7. Current-source real-data replay with recorded provenance, successes and
   failures. Diagnostic/synthetic files cannot establish sensing efficacy.

## Pilot and full-action separation

The A02+A06 pilot contains 80 sequences and 23,760 frames. Its settings are in
`configs/subset-*.json`. It exercises the full experiment pipeline and provides
a bounded initial comparison. It cannot establish coverage of all actions.

The complete modality-selected MM-Fi inventory contains 1,080 sequences,
320,760 synchronized frames and 27 actions. Its acquisition and preparation
must finish and be audited independently. Full-action experiments require
separately identified configurations and output directories. Pilot scores may
not be relabeled as full-dataset results.

## Definitions and uncertainty

Coordinates use a declared isotropic annotation plane. PCK@0.2 counts a valid
joint whose Euclidean error is at most 0.2 times the valid reference pose's
bounding-box diagonal. NJE averages error divided by that diagonal. Degenerate
boxes are excluded and counted. No metric is measured in 3D or millimeters.

Gaussian confidence is model-implied radial mass inside a stated radius in
the reference plane. It is uncalibrated; reliability bins and expected
calibration error test this claim against held-out annotations. A confident
prediction can still be wrong. The supplied teacher labels may also be wrong.

The comparison bootstrap resamples entire sequence blocks. A separate
subject-cluster analysis keeps both action sequences for a person together.
Intervals quantify the observed sample, not an independent number of frames,
a causal mechanism or all future rooms. Two training seeds are a sensitivity
check, not a reliable distribution of training variance.

Model FPS excludes file access, preprocessing, acquisition and browser
rendering. Replay export also measures preprocessing plus forward pass;
warmup/context and the original 10 Hz sampling rate remain separate.

## Failure accounting

Reports retain every strided complete candidate window and the accepted,
rejected and metric-eligible counts. Reasons can overlap. Startup context and
stride are disclosed separately. Replay withholds invalid signal predictions
instead of presenting a stale skeleton. Unseen/unknown domain status comes
from metadata, not a learned out-of-distribution detector.

Any final result must state whether temporal learning improves the matched
baseline and the static prior, whether the effect survives ablation and seed
changes, and whether it holds on unseen environments. Negative or ambiguous
results remain part of the release.
