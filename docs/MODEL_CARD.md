# GhostMesh model and evaluation contract

GhostMesh implements an original compact 2D pose model. Its purpose is consented
wireless-perception research using paired CSI and pose annotations. A trained
model consumes only CSI at inference. A separate annotation path supplies image
coordinates during training and evaluation. No component validates 3D pose,
through-wall surveillance, identity recognition, fall safety or medical use.

Implementation: `ghostmesh/models.py`, `engine.py`, `metrics.py`. No code, model
weights or figures were copied from the WiSPPN-derived CSI-Pose/Poseborne model
whose upstream redistribution grant was unclear. This document describes the
method and evaluation rules; measured results belong in the experiment reports.

## Inputs, outputs and causality

Input is `[batch, time, link, subcarrier, packet]` nonnegative CSI amplitude.
The model uses the latest complete causal window to predict its final frame.
Labels are `[joint, x/y]` coordinates in a declared annotation reference plane.
The primary cohort divides both pixel axes by 640, preserving isotropic
geometry without claiming a recovered camera resolution. The skeleton ordering is declared by the prepared dataset, not inferred
from coordinate values. The model must match the dataset's link and joint count.

Preprocessing rejects excessive missing, negative and flat signal. It fills a
small number of missing measurements from the current window, applies log1p,
and optionally subtracts the per-link median and divides by its robust scale.
All these statistics use only the window ending at the prediction time. No
whole-test-sequence or future-frame statistics are fitted. Clipping bounds the
processed features. Read `preprocess.py` for the exact thresholds and scales.

The primary frame encoder computes mean and standard deviation across each
released block's ten packets. It retains the three antennas as distinct
channels, pools subcarrier moments to 32 ordered bins and learns a projection
through two linear layers with LayerNorm and SiLU. This marginalizes packet
order, whose original timestamps were not supplied, while retaining coarse
frequency and antenna structure. Width 64 was chosen in a bounded,
validation-only three-epoch comparison before final testing. The older
two-convolution frame encoder remains available as `encoder="cnn"`.

Three residual temporal blocks use
depthwise causal convolutions with dilations 1, 2 and 4, followed by feature
mixing and per-timestep LayerNorm. Left padding preserves causality. The final
time's features produce bounded 2D coordinates and one log standard deviation
per joint. No camera or action label enters this model.

The primary `direct` temporal kernel computes the same three-tap depthwise
cross-correlation and pointwise projection as the retained `native` PyTorch
convolution. It preserves parameter names and shapes. Tests compare outputs,
gradients, checkpoint reload and causality across widths, batches, dilations
and float32/float64. This removes CPU dispatch overhead; it is an implementation
optimization, not a different learned temporal architecture. Kernel-only
timings do not establish a whole-system speedup.

The comparison model shares the frame encoder and head, and uses only the
final preprocessed frame. Both models receive the same window preprocessing,
so the baseline retains information from the causal-window normalization
statistics. It is a final-frame feature baseline, not a history-free raw-signal
baseline. Actual parameter counts are recorded, rather than called identical.

## Training and proposed generalization mechanism

For each valid joint the supervised objective is a 2D isotropic Gaussian
negative log likelihood, omitting an additive constant:

`0.5 * ||prediction - target||² * exp(-2*log_sigma) + 2*log_sigma`.

Invalid annotations contribute no loss. A batch with no finite supervision is
an error. The output standard deviation is constrained to a finite range.
Gradient clipping prevents an isolated large gradient from dominating a step.

The proposed source-only generalization strategy combines causal-window
normalization, per-link multiplicative gain perturbations, subcarrier masking
and a slowly updated teacher. The teacher is an exponential moving average of
the student's parameters. It sees unperturbed source training examples; the
student receives perturbed examples and an additional coordinate consistency
loss. The primary consistency objective divides squared coordinate distance
by twice the detached teacher variance, clamped to `[0.05²,0.5²]` in the
annotation plane. This bounds teacher precision and puts the term on a
comparable scale to the supervised Gaussian objective. A source-only diagnostic
found raw coordinate MSE had a negligible gradient at the original weight;
the raw formulation remains an explicit ablation. Teacher uncertainty is not
assumed calibrated. A short ramp reduces early teacher influence. Validation and test
examples never update either network. This is a domain-generalization
hypothesis, not demonstrated target adaptation or a replacement for labeled
source training. There is no unlabeled target-domain fitting mode.

Normalization could remove useful absolute-position cues. Masking can destroy
weak signals. A teacher can reinforce a student's errors. These are reasons to
measure ablations, not assume the combination works. The full model is allowed
to lose to the simpler baseline or the no-signal mean-pose prior.

## Fixed protocols and controls

The manifest stores sequence hashes, subjects, environments, splits, coordinate
conventions and provenance. Windows never cross recording boundaries. Adjacent
frames remain within one split. A cross-subject protocol rejects overlapping
people. A cross-environment protocol additionally rejects test environments
present in training or validation. MM-Fi groups people by room, so its unseen
room result also changes the people. These two causes cannot be isolated by
that split.

Every checkpoint is selected by validation normalized joint error only. The
training function never constructs a test loader. The checkpoint binds the
exact manifest hash. Evaluation refuses a changed manifest. Thus test
predictions are from a frozen model and a frozen evaluation population.

The train-mean-pose baseline fits each joint's mean and residual variance using
only source training targets from the same window convention. It consumes no
CSI features or action IDs. A small action subset can achieve misleadingly high
PCK from a static pose prior; GhostMesh compares against this prior explicitly.

## Metrics and uncertainty

PCK is the fraction of valid joints whose Euclidean error is at most `0.2` of
that frame's valid ground-truth 2D bounding-box diagonal, unless a config states
a different threshold. Normalized joint error (NJE) is the mean Euclidean
error divided by the same diagonal. Frames with fewer than two valid joints or
a degenerate box are excluded and counted. Neither metric is millimeters or
3D MPJPE. Normalized image coordinates may weight horizontal and vertical
pixels differently when image aspect ratio is not one; use the declared
convention when comparing other research.

The network's sigma is an estimated standard deviation, not a probability of
correctness. The optional confidence readout is the isotropic Gaussian mass
within a stated radius, by default `0.05` in normalized-image coordinate space.
Held-out expected calibration error compares these probabilities with actual
success within that absolute radius, using ten bins. It is separate from PCK,
whose radius uses the ground-truth body scale. No calibration is claimed merely
because the model emits a confidence number. Out-of-domain inputs can still be
overconfident; signal-quality rejection and uncertainty are incomplete failure
detectors.

Evaluation reports per-joint, per-subject, per-environment and per-sequence
scores. Qualitative candidates are ranked actual low/high-error frames, not
hand-authored success animations. The latency benchmark uses five warmup calls
and 30 synchronized, batch-one model calls. Median/p95 and model FPS exclude
preprocessing, file I/O, rendering and network transport. They are not an
end-to-end application FPS guarantee.

## Reproduction and ablations

Versioned configs cover the final-frame baseline, temporal model, removal of
teacher consistency, removal of window normalization, removal of augmentation,
and a separate unseen-environment experiment. Setting augmentation to zero
also removes teacher perturbation diversity, so that ablation changes this
coupled mechanism. All variants retain the same data-window convention.

`scripts/run_suite.py` freezes configs and manifest hashes before training,
trains all requested models using source/validation data, then evaluates their
selected checkpoints on test. `configs/subset-*.json` use width 32, eight
frames, stride four and a 20-epoch budget for the bounded two-action MM-Fi
subset. Temporal and baseline variants have seeds 2027 and 2028. A smaller
budget can be fixed before a suite starts; it is saved in preregistration.
The `cohort20` profile uses all 27 actions for the 20 people whose recordings
were complete before the provider quota. It fixes packet-moment width 64,
eight frames, training/validation stride 8, dense test stride 1, 40 epochs,
batch 64 and one CPU thread. Primary temporal
and last-frame encoder pairs use seeds 2027/2028 in both subject and environment
protocols. Individual consistency/normalization/augmentation ablations use
seed 2027. Configurations establish intended experiments, not completed results.

Optional read-only window caches preserve the exact prepared tensors and
window order. Their identities bind the data/preprocessing source, manifest,
split, normalization, stride and shape; all bytes and headers are verified
before reuse. Epoch-boundary resume restores student, teacher, optimizer and
all RNG states. It requires identical frozen inputs/source/runtime/device,
retains recovery bytes and never uses test labels for checkpoint selection.

Paired comparison intervals resample whole sequence blocks, comparing each
model on the same sequence targets. The point estimate weights sequence means
equally; overall frame-weighted metrics can therefore differ. Since action
recordings of one person remain correlated, the suite also resamples whole
subject clusters as a sensitivity analysis. Small cluster counts make both
intervals unstable. These intervals describe held-out sampling for fixed
checkpoints, not training randomness. Two seeds are a sensitivity check, not a
reliable estimate of training variance. Predefined ablations are reported
together; test rankings must not be used to retune the architecture and then
present the same test set as untouched.

Synthetic fixtures are explicitly marked `software_check_only`. Their scores
validate mechanics and cannot establish Wi-Fi pose estimation, generalization,
novelty, real-world calibration or improvement. Dataset-backed subset results
must retain their action/subject/room coverage and license attribution; a
two-action result cannot stand in for the complete MM-Fi benchmark.

## Known limits

CPU is the tested backend for this release. A synthetic check on the local
Apple M4 GPU found that PyTorch 2.9.0 MPS cannot execute the packet encoder's
non-divisible adaptive pooling from 114 subcarriers to 32 bins. It raised before
prediction; no GPU accuracy or speed result is claimed. CUDA was unavailable.
The CLI's device choices also serve other configurations and do not establish
backend support for every architecture.

The model assumes one labeled person, a fixed sensor layout and one defined
2D coordinate view. Sensor placement, radio bandwidth, subcarrier layout,
clothing, multipath, packet loss, unseen actions and multiple people can break
these assumptions. MM-Fi CSI and ESP32 capture do not share an interchangeable
sensor contract. An ESP32 model requires paired data and independent hardware
evaluation. The software does not turn an MM-Fi checkpoint into a validated
ESP32 pose estimator. Use only authorized recordings from consenting people.
