# Unlabeled startup research

A frozen reference from the beginning of each recording **does not fix unseen-room pose or recover useful motion**. The complete E01–E03 source-development experiment fails its predeclared gate in all three rooms. Numerical reproduction passes; small average joint-error gains are insufficient evidence of reconstruction.

This is a separate **unlabeled startup mode**. It consumes the current recording's early CSI and assumes its reset boundary is known. It is neither no-enrollment zero-shot inference nor [room calibration using separate reference recordings](CALIBRATION_RESEARCH.md). No calibrated product is established.

## Measured room transfer

Every method below is evaluated on exactly the same ten people and all 27 actions in each held source room. The target is the supplied 2D skeleton after removing hip-midpoint translation. These are articulation results, without absolute position or 3D reconstruction.

| Method, hip-relative NJE ↓ | E01 | E02 | E03 |
|---|---:|---:|---:|
| Declared primary: startup shape, 456 dimensions | .071969 | .072114 | .069500 |
| Source training pose mean | .072199 | .072157 | .069529 |
| Source endpoint-clock control | .072247 | .072206 | .069564 |
| Frozen raw-current covariance | .072859 | .072427 | .069474 |
| Frozen startup input, no current motion | .072962 | .072702 | .070290 |
| Frozen original temporal model | .071214 | .072394 | .069429 |
| Separate startup shape + volume ablation | .071980 | .072026 | .069513 |

NJE divides each joint's Euclidean error by that reference skeleton's bounding-box diagonal. Lower is better. The primary has tiny improvements over a static mean, but also has negative record-centered articulation R² in every room: **−.003865 / −.006561 / −.006401**. For this diagnostic, prediction and teacher are each centered over time within the recording. Their means are never supplied to the predictor.

| Fine-joint PCK at .05 ↑ | E01 | E02 | E03 |
|---|---:|---:|---:|
| Primary startup shape | .555700 | .551846 | .573488 |
| Source training pose mean | .557367 | .552919 | .576919 |

The primary loses fine-joint PCK to both static and clock controls in every room. Shuffling primary predictions within each recording gives NJE .071973 / .072120 / .069493, which **improves E03**. Articulation R² is nonpositive for 8/10, 8/10 and 9/10 people, and 20/27, 23/27 and 25/27 actions. These results do not support reliable time-specific pose reconstruction. The separate volume arm also has negative pooled motion R² in all three rooms and cannot replace the frozen primary.

[Machine-readable results](domain-v2-startup-results.json) retain every method's aligned and shuffled NJE/PCK/R², all person/action scores and reversals, complete source-selection grids, gate checks and provenance. This is source-development evidence, not an untouched final test. E04 is not consumed.

## What the startup reference contains

For each recording, take its earliest quality-retained endpoint with 40-frame validity context. Decode the **eight current covariance states** ending there, each with 19 frequency bins and a 3×3 complex receiver covariance. Normalize each state by its determinant before averaging. Freeze that reference for all later queries in the same recording. The primary uses generalized covariance spectra against this fixed reference, with shape only.

There are no reference pose labels, query initial poses, query pose means, future reference updates or other-person enrollment statistics in feature inference. No action or person identifier is a model feature. Recording identity routes the correct frozen state; a known recording/reset boundary is required. The first scored query may be the startup endpoint because all its required CSI has arrived.

The observed first-valid endpoints are 39 for 806 recordings, 47 for one and 71 for three. At nominal 10 Hz, that is **3.9 / 4.7 / 7.1 seconds from frame zero**, not measured live acquisition latency. The eight current states themselves span .7 nominal seconds. An action-segment boundary supplied by the dataset is not automatically available in a continuous live deployment.

The startup reference does not reveal the unknown initial pose. A generalized spectrum is noninjective, and a fixed receiver-mixing model has not been established for actual rooms. The optional log-volume branch does not cancel independent frame gain. The frozen-startup raw control repeats the same input through the recording and tests reset/template information without current motion.

## Local observability and reset failure

A separately frozen local control uses E01/S01/A12 with 79 training, 13 validation and 72 held-time windows. Its 40-frame observation supports stay inside purged blocks `[0,118)`, `[126,178)` and `[186,297)`. The primary uses the training startup at endpoint 39 and freezes it across later blocks. **Total feature raw supports share that historical training summary**, even though the current observation supports remain disjoint. No held CSI or teacher enters its fitted anchor or readout.

| Source-validation-selected local method | Held NJE ↓ | Record articulation R² ↑ |
|---|---:|---:|
| Continuous training-startup shape, linear | .030376 | .775429 |
| Independent reset stress, RBF | .062000 | −.077249 |
| Training mean / constant startup input | .058573 | approximately 0 |
| Raw-current covariance, linear | .021364 | .881063 |
| Frozen original temporal control, linear | .026739 | .825134 |

The continuous primary passes its local gate, with aligned-minus-shuffled R² 2.196028 and NJE .518611 of static. It is weaker than both matched raw-current and old temporal controls. A shape RBF scores better on held time, but validation selected linear; the held-preferred kernel is not substituted.

The separate reset stress uses earliest valid endpoints 39/165/225 inside the three blocks. It fails, is worse than static and improves under shuffling. This exposes sensitivity to unknown initial configuration. It cannot replace the declared continuous primary. Neither local result establishes unseen-person, unseen-room or whole-action transfer.

The old temporal control retains its original absolute-pose validation selector and image-clipped predictions, then is projected to hip-relative coordinates only for matched scoring. Its observation rows and 40-frame mask match, but it has no shared startup summary. This difference is explicit rather than treating the controls as identical pipelines.

## Split, freeze and independent verification

All 26,556 preverified rows from 810 source recordings, 30 people and 27 actions are retained. The unused 174-row allocated cache tail is excluded. Each outer held-room fold has 16 source-training, four different source-validation and ten held people:

| Held room | Training windows | Validation windows | Held windows |
|---|---:|---:|---:|
| E01 | 14,190 | 3,543 | 8,823 |
| E02 | 14,124 | 3,546 | 8,886 |
| E03 | 14,166 | 3,543 | 8,847 |

Feature and random-Fourier-feature statistics use source training only: SD floor .001, 128 training samples for median bandwidth, 512 random Fourier features, seed 2027 and ridge grid .001/.01/.1/1/10. Equal-room source-validation NJE selects alpha. All nine new selections across all three folds are written before any held score; every arm selects alpha 10. Raw-current maps/readouts are frozen from the prior study and reapplied to all ten people, with their previously saved eight-person predictions checked. The original temporal selector is also retained.

The independent local verifier reconstructs all four anchors and 328 variant feature rows using SciPy generalized eigensolves rather than the fitter's Cholesky whitening. All 12 readouts, 48 candidate equations, two constant controls, selectors, predictions and gates match. Maximum feature difference is 4.61e-13, normal-equation relative residual 1.34e-14 and independent direct-solve prediction difference 2.64e-15. Saved-prediction and metric differences are zero. Saved normalization means, scales, training designs and bandwidths each have explicit shape, finite-value and numerical-tolerance checks. Twenty-two corruption cases invoke the actual local verifier and reject large offsets and NaNs in saved normalizers, anchors, volume means, features, predictions and metric scalars. The synthetic corruption fixture reuses its already reconstructed features to isolate downstream readback checks; the real-data audit separately reconstructs every feature row.

The full verifier reconstructs all 810 anchors and every row of four feature arms. Independent shape/volume reconstruction differs by at most **2.384e-7 after float32 storage**; raw-current and frozen-input features match exactly. It checks 12 source maps, 45 new and 15 frozen raw-current candidate equations. Maximum equation residual is 3.20e-14; independent validation prediction difference is 6.29e-13. Saved selected predictions match exactly and all 819 global/subgroup scores agree within 1.50e-15. Selectors, resets, permutations and failed gates match.

Both public verifiers check saved numeric arrays and metric scalars for matching shapes, finite values and explicit numerical tolerances before reducing maximum differences. This prevents NaNs from disappearing in a Python maximum reduction. Current public-verifier and shared-checker source hashes, plus fresh verification receipts, are retained separately from the original executed audit identities in the JSON report. Original fitted arrays and receipts remain unchanged.

Independent feature-reconstruction tolerance is separate from exact saved-feature model readback. A first full audit incorrectly demanded bit equality from a sqrt2 coordinate roundtrip. Its failure and source are preserved; the repaired verifier accepts machine-precision error with relative tolerance 4e-15 and absolute tolerance 1e-10. Fitted source and predictions were unchanged. Byte manifests identify artifacts; they do not certify efficacy, rights or permanent immutability.

The full result file records 39.82 seconds for CPU1 feature build, fitting and scoring; the local result records .87 seconds. These are experiment timings, not live inference latency, application FPS or acquisition times. Raw RF was not rebuilt or rehashed. No E04, RGB, GPU or paid compute was used.

## Reproduce

First prepare appropriately licensed source data and the verified predecessor artifacts through [full-cache/local covariance preparation](COVARIANCE_RESEARCH.md) and [the frozen raw-current reference study](CALIBRATION_RESEARCH.md). The commands below consume dataset-derived artifacts; the repository does not redistribute the source recordings or all experiment arrays. Existing outputs are preserved, so use fresh destinations.

```bash
# Local fit, then an independent audit without refitting.
python scripts/domain_startup_covariance_probe.py \
  --predecessor work/domain-covariance-positive-v1 \
  --out work/my-startup-local
python -W error scripts/verify_startup_covariance_probe.py \
  --experiment work/my-startup-local \
  --predecessor work/domain-covariance-positive-v1 \
  --output work/my-startup-local/verification.json
python scripts/seal_research_artifacts.py --root work/my-startup-local

# Full source-only fit. The verified local receipt is a prerequisite.
python scripts/domain_startup_covariance_transfer_probe.py \
  --cache work/domain-covariance-full-cache-v1 \
  --old work/domain-covariance-full27-v1 \
  --raw work/domain-reference-covariance-v1 \
  --local work/my-startup-local \
  --out work/my-startup-transfer
python -W error scripts/verify_startup_covariance_transfer_probe.py \
  --experiment work/my-startup-transfer \
  --cache work/domain-covariance-full-cache-v1 \
  --old work/domain-covariance-full27-v1 \
  --raw work/domain-reference-covariance-v1 \
  --local work/my-startup-local \
  --output work/my-startup-transfer/verification.json
python scripts/seal_research_artifacts.py --root work/my-startup-transfer

pytest -q -W error tests/test_startup_covariance_probe.py \
  tests/test_startup_covariance_transfer_probe.py \
  tests/test_startup_covariance_verifier.py tests/test_startup_local_verifier.py
```

The public local verifier is a portable refactoring of the preserved original independent audit. Replaying the original saved artifacts reproduces every original receipt field except the new verifier's own source hash. The original audit source, receipt and all fitted artifacts remain unchanged.
