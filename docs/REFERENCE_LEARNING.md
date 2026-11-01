# Reference learning and full-amplitude source pilots

Neither pilot qualifies for unseen-room evaluation. These experiments test
specific failure mechanisms on **E01 source people**, before any new room score.
The released models and replay application are unchanged.
[All results, candidate checkpoints, frozen plans and audit summaries](domain-v2-reference-learning-results.json).

## Shared data boundary

Both pilots use all 27 actions, with eight whole E01 people for training
(S01/S02/S03/S05/S06/S07/S08/S10, 7,050 accepted windows) and two different people
for validation (S04/S09, 1,773 windows). These validation people have prior
project development exposure. No adjacent frames or overlapping windows cross
person splits. Each input contains eight causal frames; an RF-only forty-frame
quality guard precedes the endpoint. No query pose, hip, initialization pose or
record mean enters inference. Teacher bounding boxes normalize evaluation only.
Targets are supplied, own-hip-relative 17-joint **2D** annotations.

Each source-calibration proxy uses exactly **120 reference poses** from two
training people: S01/S02 × A03/A07/A12 × 20 endpoints. Its queries are the same
two validation people in the already represented E01 environment. This measures
cross-person reference transfer within a source environment. It does **not**
establish unseen-room calibration. Learned comparisons share the same global
training labels and reference budget. Raw reference-only kernels use fewer
global labels, so they are additional controls, not a matched training-budget
comparison.

E03 metadata are counted to verify source isolation. E03 annotations, old
E03-trained weights and old normalization statistics do not enter these fits.
E03 is historically exposed development data elsewhere in GhostMesh, not an
untouched project test set. The protected E04 experiment remains unscored by
these candidates. [Full final protocol and exposure ledger](DOMAIN_RESEARCH.md).

## Differentiable reference adaptation

An independently implemented residual ridge solver adapts a 128-dimensional
CSI code using labeled support poses. Training differentiates through the
closed-form solve, so the radio representation can learn to support adaptation
across people. This uses the established
[R2D2 principle](https://www.robots.ox.ac.uk/~vgg/publications/2019/bertinetto19/);
it is not a claim of a new general meta-learning algorithm.

Three arms share initialization, episodes, parameter count and 2,000 updates:
ordinary supervision, **episodic adaptation**, and support-label permutation
within each training record. Each episode has 24 support histories from two
people and 24 queries from two other people. All arms use 96,000 RF endpoint
occurrences and 144,000 absolute target terms. Temporal difference losses reuse
these same labels. All three final models finish before validation scoring.

The solve normalizes codes, retains an unpenalized residual mean and uses
CPU float64 Cholesky with λ = 0.1 × reference count / 24. Gradients return to the
encoder. Native MPS requires transferring to CPU before casting to float64;
qualification caught and repaired the original combined-transfer error before
real training. The native gradient and recovery paths are tested.

| Mode / control | NJE ↓ | Motion R² ↑ | PCK@0.05 ↑ | PCK@0.20 ↑ |
|---|---:|---:|---:|---:|
| Episodic zero-shot | 0.080292 | -0.403893 | 0.515145 | 0.906440 |
| Ordinary zero-shot | 0.080544 | -0.405614 | 0.496102 | 0.908696 |
| Source static mean | 0.067113 | 0.000000 | 0.630769 | 0.931223 |
| Episodic + 120 refs | 0.080061 | -0.393470 | 0.515842 | 0.906407 |
| Ordinary + 120 refs | 0.079720 | -0.394131 | 0.507813 | 0.909094 |
| Raw linear + 120 refs | 0.070503 | -0.007781 | 0.625162 | 0.908065 |
| Raw RBF + 120 refs | 0.071555 | -0.001807 | 0.612753 | 0.912179 |
| Reference static mean | 0.070417 | 0.000000 | 0.626721 | 0.913772 |

Both frozen gates fail. Motion R² centers predictions and annotations within
records **for evaluation only**; zero means a static prediction explains no
movement, and negative means worse than that motion reference. Low joint error
alone can reward a static average pose. Neither calibration nor its learned
representation recovered useful motion on these people.

## Full magnitude and source-person nuisance learning

The second pilot retains the complete **3 × 114 × 10** magnitude matrix in every
frame. MM-Fi values are already logarithmic. No second logarithm, receiver ratio,
phase transform or packet averaging is applied. Missing cells use their own
frame's packet-column mean; the whole frame is then affine-normalized to [0,1].
This independently implements the inspected toolbox's mathematical convention;
no upstream implementation was copied. Values are relative log magnitudes,
not calibrated RSSI or dBm.

A 267,834-parameter frame CNN and causal temporal convolution predict the 2D
body. Three arms use the same initial state, source pairs, parameter count and
4,000 updates: ordinary, receiver-gain augmentation, and **gain augmentation
plus source-person gradient reversal**. Source-person IDs are nuisance-loss
labels during training only. They are not inputs or an identity-recognition
feature. ±6 dB receiver offsets approximate gain calibration changes; they are
not a validated simulation of room multipath. This is source-only nuisance
learning, not full target-data DANN.

All three training budgets finish before validation. Every 0/500/…/4000-step
checkpoint is then scored. Each arm and mode selects highest source-validation
motion R², then lower NJE, then earlier step. All 27 candidates remain in the
machine-readable report. **Every arm in both modes selected step zero**, the
static initial pose head. This is a failed learning result, not a trained-model
success.

| Primary selected mode | Step | NJE ↓ | Motion R² ↑ | PCK@0.05 ↑ | PCK@0.20 ↑ |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 0 | 0.067113 | 0.000000 | 0.630769 | 0.931223 |
| 120-reference proxy | 0 | 0.077822 | -0.114986 | 0.521814 | 0.904582 |

## Verification and reproduction

The episodic auditor checks all 8,823 raw target/frame joins, 12 checkpoint
seals, CPU predictions from all three models, independent reference equations,
both raw reference equations, 341 pooled/person/action/shuffle metric groups,
and both frozen gates. Maximum metric disagreement was 1.34 × 10⁻¹⁵.

The amplitude auditor checks the same raw target joins, all 27 candidate
checkpoint seals and reference equations, every reported candidate metric,
selected checkpoints and gates. Its independent normalizer replays selected
models on CPU. Maximum metric disagreement was zero and maximum relative
reference coefficient disagreement was 4.27 × 10⁻¹⁵. It checks saved raw-baseline
prediction metrics, but does not independently refit their CV/equations.
Both audits share the actual model classes. They validate these failed source
measurements; they do not prove generalization or accurate sensing.

After obtaining the appropriately licensed dataset and building the typed RF
cache as documented in [data access](DATA_ACCESS.md) and
[the source30 protocol](DOMAIN_RESEARCH.md), run from the repository root:

```bash
uv sync --locked --extra test
uv run python scripts/build_moment_cache.py \
  --raw work/domain-source30-rf-v1 --out work/domain-moment-source20-v1
uv run python scripts/domain_episodic_probe.py \
  --cache work/domain-moment-source20-v1 --raw work/domain-source30-rf-v1 \
  --config configs/episodic-reference-e03-v1.json \
  --out work/domain-episodic-e01-source-v1 --device mps
uv run python scripts/verify_episodic_source.py \
  --study work/domain-episodic-e01-source-v1 --cache work/domain-moment-source20-v1 \
  --raw work/domain-source30-rf-v1 --out work/episodic-audit
uv run python scripts/domain_amplitude_probe.py \
  --cache work/domain-moment-source20-v1 --raw work/domain-source30-rf-v1 \
  --config configs/amplitude-person-e03-v1.json \
  --out work/domain-amplitude-e01-source-v1 --device mps
uv run python scripts/verify_amplitude_source.py \
  --study work/domain-amplitude-e01-source-v1 --cache work/domain-moment-source20-v1 \
  --raw work/domain-source30-rf-v1 \
  --out work/amplitude-audit
```

The measured studies used Python 3.12, NumPy 2.2.6, SciPy 1.16.3, Torch 2.9.0
and MPS. Both training commands also support `--device cpu`; cross-backend
optimization is not promised to reproduce identical weights or scores.
The episodic audit uses the original MPS backend to recover exact native
reference encodings and CPU to check model predictions. Its exact coefficient
readback therefore requires MPS. The amplitude auditor is CPU-only. No paid API
or cloud compute is used. Output directories must be fresh. Sealed checkpoint
recovery uses `--resume` with the same command and backend after the prior
writer exits; input/config/source identity changes are rejected.

These are source-feasibility commands. They do not provide an eligible new-room
candidate or modify default application inference. A mode must pass its frozen
source gate, independent audit and separately frozen target-fit protocol before
held-room query annotations can be scored. No application latency/FPS claim is
made for these unpromoted prototypes; training durations are not inference
benchmarks. The working replay application's measurements remain in the
[reproduction guide](REPRODUCE.md).
