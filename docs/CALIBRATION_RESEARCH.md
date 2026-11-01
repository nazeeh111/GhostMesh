# Calibration research

The fixed-room CSI-reference experiment **does not demonstrate useful room calibration or reliable pose motion**. Its mathematical invariance, software checks and numerical reproduction pass. Its predeclared performance gate fails in every room.

This experiment changes the observation from a difference between two query times to the current covariance relative to a fixed enrollment reference. Unlike a temporal difference, that descriptor can retain some static configuration information. It remains noninjective, and actual rooms need not obey the assumed receiver-mixing model.

## Measured result

The primary representation was fixed before fitting: three semantic reference anchors, shape only. All scores below use the **same eight query people per room**, excluding both enrollment people. They assess supplied 2D annotations after hip-root centering, not absolute translation or 3D.

| Held room | Query windows | Source static NJE | Matched raw-current NJE | Frozen temporal zero-shot NJE | Primary calibrated NJE | Primary articulation R² |
|---|---:|---:|---:|---:|---:|---:|
| E01 | 7,065 | 0.073749 | 0.074323 | 0.072794 | 0.073619 | -0.004060 |
| E02 | 7,111 | 0.071465 | 0.071817 | 0.071922 | 0.071812 | -0.005627 |
| E03 | 7,075 | 0.070077 | 0.069839 | 0.069974 | 0.070543 | -0.008601 |

Lower NJE is better. It divides joint error by the current reference skeleton's bounding-box diagonal. Record-centered articulation R² measures time-varying joint configuration; zero corresponds to a static recording prediction after diagnostic centering. The model does not receive query means for inference.

Primary articulation R² is nonpositive for 7/8, 8/8 and 8/8 query people, and 21/27, 27/27 and 24/27 action strata. Shuffling the primary predictions within each recording slightly **improves NJE in all three rooms**. These scores do not support time-specific reconstruction. Every one of the 15 fitted representations has negative pooled articulation R².

The target 120-reference pose mean is worse than the source mean in every room: NJE .077294/.075497/.074317. Independent source-validation mean mixtures select reference weight zero in all three folds. Thus the supplied reference-label distribution does not automatically provide a better room prior. It samples three action categories, while the queries span all 27.

All 15 readouts select the largest registered ridge value, 10, using source validation. The four physics arms include one global versus three semantic anchors, each with shape-only versus explicit log-volume features. A matched raw-current covariance control uses the same eight current frames, RFF512 capacity, preprocessing and source-validation objective. No ablation is relabeled as the primary after observing held scores.

[Every threshold, control, action/person reversal, enrollment identity, exact cohort role and source binding](domain-v2-reference-covariance-results.json).

## Enrollment and separate modes

Each room supplies exactly 120 labeled endpoint poses from **two people and six recordings**: A03/A07/A12, 20 endpoints per recording, paired eight frames apart. The source pose regressor uses ordinary source training labels. The primary physics feature construction uses only the 120 endpoint CSI covariances and known enrollment action groups, without target joint coordinates. Target reference joints are reserved for the declared mean/mixture controls.

The four reference-based representations are calibrated modes because they consume target-room enrollment CSI. The raw-current and frozen temporal controls remain zero-shot comparators. Enrollment people and all eight query people are disjoint. Query action IDs, query joint coordinates, query recording means and query initial pose are not predictor inputs.

At nominal 10 Hz frame timing, the final reference endpoint occurs 26.3 seconds into each supplied recording. The six recorded prefixes total approximately 158 seconds, excluding setup and human pauses. This is not a 30-second enrollment promise or a live capture measurement. Software generation of reference matrices for **all three room sets together** took .092 seconds; that is neither acquisition nor inference latency.

For the anchor means, only one current covariance per labeled endpoint is averaged: 120 matrices per room, 40 per known action. The eight-frame query context is separate. The inherited 40-frame data-quality mask remains identical for every model/control. Decoding the cached float32 covariances introduces no additional rejected windows; all 26,556 windows remain eligible before the person split.

## Mathematical scope

Each calibration covariance is normalized by its own determinant **before** averaging. A shared invertible complex receiver mixing within a frequency bin then acts by the same congruence on reference and query shapes. Existing paired-reference shrinkage preserves the generalized spectra under that ideal model. Normalizing a raw average instead would not remove independent calibration-frame gains correctly.

A single global mean can discard orientation through its unitary stabilizer. Three action-specific anchors can retain more relative orientation information, but their spectra still do not uniquely identify arbitrary covariance matrices or poses. Different people performing the same action need not produce homologous anchor distributions. The invariance test transforms the same reference/query ensemble mathematically; it is not proof that real rooms follow that transformation.

The optional log-volume branch cancels a fixed receiver mixing but **does not** cancel independent frame gains. It is a separate ablation. Radio-coordinate invariance also does not determine an arbitrary annotation/camera coordinate frame. Output remains hip-root-relative articulation in the declared supplied annotation plane.

## Reproduction and verification

Whole-person folds contain 16 source-training and four source-validation people. Each held room contributes two enrollment and eight different query people. Input statistics and median kernel bandwidth use source training only; the alpha grid is .001,.01,.1,1,10. Alpha and mean-mixture choices minimize equal-room source-validation NJE. No target query labels select parameters.

Before the first real anchor fit, the run saves source snapshots, raw/cache receipts, all 120 reference IDs, exact cohort indices, source-independent random bases and the full SPD preflight. Fitted anchors, source feature maps, every candidate readout and predictions are retained. The CLI verifier rebuilds all fixed anchors and every feature window, checks source-only statistics/bandwidth/selection, and reproduces models, predictions, controls, mean mixtures, shuffles and strata.

All 15 models and predictions match exactly. All 75 normal equations verify with maximum relative residual 2.54e-14. Eleven meaningful tests and independent pre-fit review pass, including complex mixing/frame-gain invariance, the volume caveat, static information, multiple-anchor orientation, prefix causality, label-independent anchor construction, person separation, source-only selection and the fixed-primary gate.

After preparing the verified full-source covariance cache and frozen temporal study:

```bash
python scripts/domain_reference_covariance_probe.py \
  --cache work/domain-covariance-full-cache-v1 \
  --old-probe work/domain-covariance-full27-v1 \
  --out work/my-room-reference-study
python -W error scripts/domain_reference_covariance_probe.py --verify \
  --cache work/domain-covariance-full-cache-v1 \
  --old-probe work/domain-covariance-full27-v1 \
  --out work/my-room-reference-study
```

[Full-cache preparation and earlier controls](COVARIANCE_RESEARCH.md). This is a source-development investigation, not an untouched final test. No E04 arrays are consumed and no calibrated product is established.
