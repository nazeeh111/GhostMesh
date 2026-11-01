# Supervised residual calibration

The predeclared 120-label residual calibration experiment **fails its performance gate in all three rooms**. Source-only nested validation selects an active correction in E01 and E02, where it makes held-person joint error worse than the same frozen predictor without calibration. It selects no correction in E03. This route does not establish useful room calibration or reliable time-specific pose recovery.

## Held-person results

All methods use the same eight query people per room, different from both enrollment people. The target is the supplied 2D joint annotation after hip-root centering. The predictor receives no query root, query annotation, query recording mean, initial pose or query action ID.

| Held room | Query windows | Source mean NJE | Frozen current prior NJE | Primary residual NJE | Selected alpha / beta | Primary articulation R² |
|---|---:|---:|---:|---:|---:|---:|
| E01 | 7,065 | .073749 | .074323 | **.074541** | .1 / .25 | -.005729 |
| E02 | 7,111 | .071465 | .071817 | **.072050** | 1 / .5 | -.006407 |
| E03 | 7,075 | .070077 | .069839 | **.069839** | .001 / 0 | -.002457 |

Lower normalized joint error (NJE) is better; it divides each joint error by that frame's reference skeleton bounding-box diagonal. Record-centered articulation R² assesses time-varying configuration, with zero corresponding to a recording-constant prediction after diagnostic centering. That centering is an evaluation operation, not an inference input.

The primary loses to the frozen current predictor for 7/8 query people in both E01 and E02 and ties all eight in E03. It loses to the source mean for 5/8, 6/8 and 3/8 query people. It loses to that mean in 19/27, 16/27 and 10/27 action strata. Pooled articulation R² is negative in every room, and shuffling the predictions within each query recording improves NJE in E02. E03 uses beta zero, so it exactly reproduces the frozen current prior. A positive correction was part of the gate.

The independently selected reference-label-permuted residual control scores .074577/.072051/.069839 NJE. The primary's slight E01/E02 advantage over this deliberately mismatched control does not overcome its losses to the uncalibrated prior and static mean. The absolute KRR and mean-mixture controls independently select beta zero in every fold. Target reference pose means are worse than source means in all three rooms.

[All control scores, thresholds, per-person/action reversals, nested choices, reference identities and cohort roles](domain-v2-reference-kernel-results.json).

## What changes

The earlier [CSI-only fixed-anchor experiment](CALIBRATION_RESEARCH.md) used reference CSI to form invariant descriptors and reserved target reference joints for mean controls. This separately labeled **supervised calibration** route uses those same 120 pose labels to fit a target-room residual readout. It is ordinary regularized radial-basis kernel regression, not a new learning algorithm.

The primary starts with the frozen, source-trained raw-current covariance predictor `f0`. Its features contain eight current covariance states across 19 frequency bins: 1,368 real coordinates. For each enrollment window, the residual target is `Yref - f0(Xref)`. An RBF kernel uses the reference feature mean and standard deviation, with .001 scale floor, and the median squared distance of **all unordered reference pairs**. No query data fit those statistics.

The readout solves `(K + 120 * alpha * I) * dual = residual` without residual centering or a target intercept. Prediction is `f0(Xquery) + beta * Kquery,ref * dual`. A query far from every reference returns to `f0`. Outputs remain unprojected, unclipped hip-root-relative coordinates, matching the frozen predictor. No query-specific ground-truth alignment is applied.

## Nested source-only selection

For each outer held room, only its other two rooms select hyperparameters. In each direction A to B, an inner base predictor uses A's eight training people; its feature statistics, RFF512 bandwidth and readout fit those people only. A's people 4 and 9 select its base ridge value. B's first two people supply the 120 pseudo-target references, while B's people 4 and 9 are the validation queries. Reversing A and B supplies the second direction. Equal weighting of these two directions selects the calibration alpha and beta separately for each method. Choices are never pooled across outer folds.

The alpha grid is .001, .01, .1, 1, 10; beta is 0, .25, .5, .75, 1. The two directional validation queries are disjoint from their target enrollment people and from their source base room. All three outer choices are saved before any outer-held query score is computed. Inner bases use eight training people, while the frozen final bases use 16 people from the two source rooms; this mismatch is a material limitation of the development protocol.

Controls receive the same reference budget. They are the source mean, target reference mean, independently tuned source/reference mean mixture, frozen current predictor, independently tuned constant residual bias, independently tuned absolute KRR around the source mean, independently tuned residual KRR with reference labels permuted within each reference recording, and the frozen temporal zero-shot predictor. The permutation is seeded before fitting and preserves each recording's 20-label distribution. Each room must contain the primary and all eight controls for the gate to evaluate.

The frozen gate requires primary NJE below every control in every room, PCK .05/.2 at least as good as both mean mixture and frozen current prior, positive articulation R², lower NJE than within-record query shuffling, and positive beta. All three room gates fail. No ablation or candidate is promoted after the held scores.

## Enrollment burden and scope

Each room uses exactly 120 labeled endpoint **windows**, from two people and six recordings: A03/A07/A12, with 10 lag-eight pairs per recording. The paired endpoints are 39/47, 63/71, continuing every 24 frames through 255/263. The eight-current-state feature means that those 120 labels can use up to **960 distinct CSI covariance states**. The preceding anchor route averaged only one current covariance at each of 120 endpoints. The label budget is unchanged; the CSI history usage differs explicitly.

At nominal 10 Hz timing, the six supplied recording prefixes total about 158.4 seconds, excluding setup and pauses. This is neither a live collection measurement nor a short-enrollment promise. Every method retains the existing 40-frame data-quality guard. The completed fit and evaluation run took 22.01 seconds on one CPU thread after reusing the verified feature cache; this is not inference latency or dataset preparation time.

This is development evidence from all 27 actions in E01–E03. E04 remains unused. These targets and metrics do not establish 3D reconstruction, absolute translation, unknown initial pose, calibrated deployment quality or strong unseen-room generalization.

## Reproduction

Prepare and verify the [full covariance study](COVARIANCE_RESEARCH.md) and [fixed-anchor/current-state cache](CALIBRATION_RESEARCH.md) first. Preserve their fitted artifacts. The residual run refuses to replace an existing output directory.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -W error scripts/domain_reference_kernel_probe.py \
  --cache work/domain-covariance-full-cache-v1 \
  --anchor-probe work/domain-reference-covariance-v1 \
  --temporal-probe work/domain-covariance-full27-v1 \
  --out work/my-reference-kernel-study
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -W error scripts/verify_reference_kernel_probe.py \
  --cache work/domain-covariance-full-cache-v1 \
  --anchor-probe work/domain-reference-covariance-v1 \
  --temporal-probe work/domain-covariance-full27-v1 \
  --out work/my-reference-kernel-study
```

Before fitting, the run saves source snapshots, input hashes, exact roles/reference identities, within-record label permutations, enrollment burden and source-independent RFF initialization. It later saves every candidate coefficient, nested curve, selected parameter, prediction and query shuffle. The independent verification CLI audits saved coefficients with a separate dense solver, replays source-only maps and selections, and recomputes all output metrics. Its boundary checks bind the saved receipts; execution order is established by the reviewed frozen runner, not by hashes alone.

All 27 method-by-room prediction sets reproduce exactly. All 135 saved target-kernel systems pass an independent dense LU solve and equation audit: maximum relative residual 7.90e-15 and coefficient difference 3.61e-15. Three source-only inner maps/priors, six nested validation curves, three outer choices, references, shuffles, every metric stratum and the failed gates reproduce. The original audit receipt's `models_verified: 27` field counts prediction sets, including static controls; it does not mean 27 learned models.

Nineteen focused tests pass under `-W error`, including all-three-room feature/label poisoning, exact saved/source-only-map parity, reference-only normalization, finite/degenerate guards, no hidden residual intercept, far-reference fallback, independent coefficient solves and tamper rejection. Fresh independent reviews of both fitting and verification source pass. Numerical receipts and final source bindings are recorded in the linked result artifact.
