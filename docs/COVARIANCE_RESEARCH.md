# Receiver covariance research

The full 27-action follow-up finds a small CSI-dependent displacement signal in every source-room fold. **It does not consistently improve pose over a static prior.** The result is a reason to test a different motion decoder, not a solved unseen-room system.

The follow-up uses all 30 E01-E03 people and 810 recordings. Each held room has ten people absent from fitting and source-validation selection. All action categories occur in source training. These are development results, one seed 2027, following the earlier three-action source experiment. E04 remains sealed.

| Held room | Static articulation NJE | Invariant articulation NJE | Raw displacement R² | Shuffled displacement R² | People with nonpositive displacement R² |
|---|---:|---:|---:|---:|---:|
| E01 | 0.072199 | 0.071214 | 0.007535 | -0.003615 | 0/10 |
| E02 | 0.072157 | 0.072394 | 0.000532 | -0.019241 | 5/10 |
| E03 | 0.069529 | 0.069429 | 0.000387 | -0.022110 | 3/10 |

E02 loses to the source static prior on articulation NJE, and E02/E03 both lose on PCK@.05 and PCK@.2. Articulation NJE reverses for 6/10 E02 people and 4/10 E03 people. Displacement R² is nonpositive for 12/27, 17/27 and 17/27 action strata, respectively. The positive pooled displacement values are tiny, especially in E02/E03. Record-centered articulation R² is negative in every room. The original frozen pose gate fails: macro R²−.019797 and aligned-minus-shuffled gain .004322.

A later motion-only development continuation rule, chosen after the first limited-action E01 score, remains satisfied. It asks for positive raw displacement R² and better aligned than shuffled correspondence in all three rooms. This is not final acceptance, a preregistered efficacy claim or proof of consistency across people/actions.

All comparisons retain 26,556/26,730 endpoints (99.35%) under the same strict 40-frame validity mask. The reusable full cache imports 90 already independently verified recordings and computes 720 new recordings. The new full probe consumes that cache with the same512-feature random Fourier maps, source-only normalization/bandwidth, validation grid and whole-person splits as the limited probe. The stopped, unverified v3 partial cache is preserved and supplies no reused bytes.

Independent readback reconstructs the 720 new CSI feature records, checks the 90 reused records against their existing verification, and verifies raw-pose target joins for all 810 records. All nine saved RFF designs and predictions match exactly. All 45 normal equations verify, maximum relative residual 2.57e-14. The cache-aware fitter also reproduces every weight and prediction of all nine earlier limited-action fits exactly; held-label perturbation tests cannot change the fitted parameters or selection.

These metrics assess hip-root-relative joint articulation and its eight-frame displacement. Translation is not recovered. NJE/PCK use the current reference body's bounding-box diagonal; raw displacement R² uses zero displacement as its baseline without a query recording mean repair. The baseline and physics controls are matched; historical neural studies have different data/targets/budgets and are not paired controls here. Full2Dpose, calibrated-product efficacy and unknown initial pose remain unresolved.

[All full-action baselines, thresholds, person/action reversals and provenance](domain-v2-covariance-full27-results.json).

Reproduce after building the integrity-checked typed source30 cache:

```bash
python scripts/domain_covariance_full_cache.py \
  --cache work/domain-source30-rf-v1 --limited work/domain-covariance-actions3-v1 \
  --out work/my-full-covariance-cache
python scripts/domain_covariance_cached_probe.py \
  --cache work/my-full-covariance-cache --out work/my-full-covariance-study
python -W error scripts/verify_covariance_cached_probe.py \
  --base work/domain-source30-rf-v1 --cache work/my-full-covariance-cache \
  --limited work/domain-covariance-actions3-v1 --out work/my-full-covariance-study
python scripts/report_covariance_source_probe.py --out work/my-full-covariance-study
```

The full cache builder first requires the independently verified limited cache produced by the commands below. Archived partial attempts are not treated as verified inputs.

## Exact window-swap constraint: negative follow-up

A further controlled test extracts only the lag-eight covariance pairs and makes the displacement decoder change sign exactly when its two eight-frame windows are swapped. The model applies the same 512-feature Fourier basis to both orientations and subtracts their features, with zero target intercept. Its ordinary control receives the same source examples and algebraically negated labels, the same basis, capacity, normalization energy, ridge grid and source-validation budget.

| Held room | Augmented ordinary raw displacement R² | Exact-odd raw displacement R² | Exact-odd shuffled R² |
|---|---:|---:|---:|
| E01 | 0.0000305 | -0.0004331 | -0.0010357 |
| E02 | -0.0002309 | -0.0007025 | -0.0010892 |
| E03 | -0.0006831 | -0.0016819 | -0.0022264 |

The frozen continuation gate fails. Every arm selects the largest registered ridge value, 10, by source validation. The exact-odd decoder loses to zero displacement in every room and to its matched augmented control. Worse shuffled scores do not rescue a model that loses to zero. This is a rejected decoder/feature restriction, not a claim that all covariance representations lack useful information.

Source training statistics and median bandwidth use only the symmetrically augmented training set. Both arms process two algebraic orientations of each measured pose pair; those are not twice as many independent measurements. Feature columns have unit source energy except at the disclosed .001 standard-deviation or RMS floor. The odd model has no centering offset or target mean, so a zero-change input produces exactly zero output. No query recording mean repairs predictions.

All six saved models and predictions reproduce exactly; all 30 normal equations verify, maximum relative residual 4.10e-14. Four meaningful tests cover swap involution, correct lag extraction, source-only statistics, actual swapped-input antisymmetry, exact static zero, held-label independence and direct augmented-system solutions. The actual RF span is 16 frames, while the inherited 40-frame availability mask remains identical for both arms. Odd inference requires two Fourier-map evaluations, compared with one for ordinary inference without the swap diagnostic. No deployment timing claim is made.

[Complete matched scores, person/action/zero/shuffle breakdowns and provenance](domain-v2-antisymmetric-results.json).

```bash
python scripts/domain_antisymmetric_probe.py \
  --cache work/my-full-covariance-cache --full-probe work/my-full-covariance-study \
  --out work/my-antisymmetric-study
python -W error scripts/domain_antisymmetric_probe.py --verify \
  --cache work/my-full-covariance-cache --out work/my-antisymmetric-study
```

## Earlier fixed three-action experiment

The new covariance representation reduces pooled articulation error in all three source-room development folds. Its motion scores are small and uneven, and it does **not** establish robust unseen-room pose recovery.

This is one seed on fixed actions A03, A07 and A12, with all 30 source people. In each fold, the other two rooms supply 16 training and four validation people; ten people in the held room are absent from fitting and selection. Those action categories occur in training. E04 remains sealed.

| Held room | Source static NJE | Invariant NJE | Invariant PCK@.05 | Static PCK@.05 | Raw motion R² | Shuffled motion R² |
|---|---:|---:|---:|---:|---:|---:|
| E01 | .064296 | .057110 | .65148 | .61780 | .021061 | −.007324 |
| E02 | .062680 | .057360 | .62658 | .59403 | .000508 | −.123171 |
| E03 | .062160 | .058688 | .63655 | .63691 | .024110 | −.016138 |

NJE and PCK here assess **hip-root-relative articulation**, removing translation from both predictions and references. The denominator is the current reference skeleton's bounding-box diagonal. These are not full absolute-pose scores. Raw motion R² compares predicted eight-frame displacement with zero displacement, without repairing predictions using a query recording's mean. The shuffle preserves each recording's predicted values while destroying their temporal alignment.

Untreated covariance and determinant-normalized log-covariance controls use identical temporal coverage, samples, full feature dimensions, RFF512 capacity and validation grids. Their raw displacement R² values are negative in every held room. The static control uses only the source-training relative-pose mean and predicts zero displacement. A matched neural control was not run in this limited-action trial; earlier full-action neural studies have different budgets and are not paired controls.

The pooled gains conceal material failures. Displacement R² is nonpositive for 1/10 E01 people, **7/10 E02 people**, and 1/10 E03 people. A03 is nonpositive in every room, and A07 also reverses in E02. Relative NJE loses to the static prior for E03 people S23 and S26. E03 fine PCK is slightly below the prior. Record-centered articulation R² is .000847, −.018277 and −.024156, so most point-error gains do not demonstrate recovered within-record pose motion.

The original, frozen continuation gate fails: macro record-centered pose R² is −.013862 and the aligned-minus-shuffled gain is .005672. A later development-only motion criterion, introduced after the first E01 score, asks for positive raw displacement R² and better aligned than shuffled displacement in all three rooms. That criterion passes, and supports another controlled experiment. It is not a preregistered success, statistical significance test or final acceptance criterion.

The representation averages complex three-receiver covariance matrices in 19 bins of six subcarriers. Paired generalized eigenvalues cancel a shared invertible receiver mixing within a bin, under the stated mathematical model. Real room changes need not obey that model. Constant covariances have identical zero motion features for different static poses, so a reference or prior is required to identify an unknown initial pose. The optimized implementation caches repeated determinant calculations and matches the original features bit for bit.

All arms retain 2,950 of 2,970 possible endpoints. Four rejected raw frames in three records remove their containing 40-frame histories uniformly. Source statistics, median kernel bandwidth and target-specific regularization are fitted or selected using source training/validation only. No preliminary 64-dimensional random projection is used.

Verification reconstructs all 90 selected recordings from actual CSI, including every retained feature and target join. Nine saved RFF designs and predictions match exactly. Forty-five normal equations have maximum relative residual 1.85e-14. Input hashes, source snapshots, source-only statistics, bandwidths and validation choices verify. Two report tests additionally check the zero-motion reference and undefined zero-energy case.

A separate chronological E01/S01/A12 feasibility control retains useful motion after invariant compression: selected linear model R² .825134, NJE .038369 versus static .103006, with non-overlapping 40-frame split histories. That result is local to one person/action/room and does not strengthen the room-transfer claim.

[Complete metrics, controls, person/action reversals and provenance](domain-v2-covariance-results.json) contain the exact evidence. Full 27-action source expansion is subsequent development; no final-room or calibrated-product result follows from this limited trial.

To reproduce from the verified typed source30 cache:

```bash
python scripts/domain_covariance_source_probe.py \
  --cache work/domain-source30-rf-v1 --out work/my-covariance-actions3 \
  --limited-actions --cached-shapes
python -W error scripts/verify_covariance_source_probe.py \
  --cache work/domain-source30-rf-v1 --out work/my-covariance-actions3
python scripts/report_covariance_source_probe.py --out work/my-covariance-actions3
```

The all-action builds preceding this limited trial stopped before fitting, first at a strict missing-vector guard and later to limit compute. Their partial bytes and executed source snapshots are preserved locally. The three-action scope was declared before these fits or held-room scores; it was not selected from successful held actions.

## From displacement to articulation

A subsequent causal observer integrates predicted increments around a source or
reference mean. It receives no query initial pose or query mean. Calibration
supplies exactly 120 poses from two people, three actions and ten pairs of
frames per action/person. Those pairs provide 60 lag-eight displacements.
The eight query people in each room are different people. The static controls
receive the same references and select their own mean interpolation.

This mean-anchored observer does not solve the reconstruction problem.
Source validation selects zero motion gain in E01; E02 gains slightly in both
modes (NJE .062247 versus .062330 static), while E03 calibration loses
(.063921 versus .063780). Zero-shot E03 returns the static prior.
No nonzero integration persistence is selected. This illustrates why a small
increment signal does not identify the current skeleton.
[All parameters, source identities and scores](domain-v2-observer-results.json).

The source-validation reference people occur in the frozen decoder's training
set, whereas target-room reference people are unseen by that decoder. This can
make validation calibration residuals optimistic. The observer is limited to
articulation and three represented actions; it makes no absolute-position or
calibrated-product claim.

The completed complementary filter adds the frozen RF pose head as its current
configuration estimate. Calibration uses the same 120 poses and can fit a small
shared 2D correction to reference displacements, with a prior toward the identity.
This is an empirical correction, not recovered camera geometry. Zero-shot and
calibrated results use the same eight query people, with an additional ten-person
zero-shot report. All choices use source validation before held-room scoring.

It also fails to add consistent value. E01 and E02 select zero displacement gain;
their predictions equal the corresponding no-motion filters. In E03 the combined
model reaches NJE .060603, losing to the independently selected no-motion filter
at .060118 for five of eight people and two of three actions. Calibration worsens
E02 slightly. All 63 saved prediction readbacks match exactly.
[Complete controls, parameters and reversals](domain-v2-complementary-results.json).

## Learning on the physical representation

A matched neural experiment learns directly from the full invariant features,
using 66,156 parameters. Both networks predict current and previous articulation
from the same current CSI history. The previous pose is an auxiliary prediction,
never an input annotation. One objective supervises both poses; the other replaces
the previous-pose loss with a displacement loss. Architecture, initialization,
1,000 optimizer steps and the 128-window batch schedule are identical.

This uses all 27 actions in the E02 development fold: 14,124 training, 3,546
source-validation and 8,886 held-room windows. Source-only normalization includes
both current and previous label occurrences. Both select step 100 by current
source-validation articulation NJE. Best and last checkpoints are retained.

| E02 method | Articulation NJE | PCK@.05 | PCK@.2 | Displacement R² |
|---|---:|---:|---:|---:|
| Source static prior | .072157 | .552919 | .927672 | .000000 |
| Matched absolute objective | .072446 | .538885 | .927023 | .001613 |
| Motion objective | .071810 | .549589 | .928135 | −.000394 |
| Fixed invariant RFF | .072394 | .543214 | .926189 | .000532 |

The motion objective slightly improves mean error but loses fine PCK to the
prior, has negative displacement R² and loses NJE for six of ten people and
14 of 27 actions. Centered pose-motion R² is negative for both networks. It does
not qualify for final-room testing or replacement of the released model.
Only the two neural objectives have matched optimization budgets; the RFF model
uses the same rows and targets with a different fitting method.

All four checkpoint readbacks match exactly. Independent metrics, source
statistics and all 1,000 schedule batches verify; a CPU spot check differs from
MPS by at most 2.98e−8. The run takes 181 seconds on local MPS, including checks.
No inference FPS or uncertainty estimator is claimed for this prototype.
[Full scores, source identities and independent audit](domain-v2-covariance-neural-results.json).

```sh
python scripts/domain_covariance_neural_probe.py \
  --cache work/my-full-covariance-cache --baseline work/my-full-covariance-study \
  --out work/my-covariance-neural --device cpu
python scripts/domain_complementary_observer.py \
  --experiment work/my-covariance-actions3 --out work/my-complementary-observer
```

## Test the premise before global alignment

A further diagnostic checks whether distances between current RF covariance
states resemble distances between joint configurations over an entire recording.
That would support investigating an offline trajectory-alignment decoder.
It does not use reference annotations as inference inputs.

Both tested geometries fail their frozen continuation criteria. Across 90
recordings, median correlations after accounting for time lag are −.0186 and
−.0194, with only 39 and 37 positive cases. All 5,720 circular-shift controls
reproduce. No alignment model is fitted on this failed premise. This finding
concerns current covariance distances; it does not contradict the strong local
full-CSI model or establish that every RF representation lacks pose information.
[All recording-level results, gates and independent verification](domain-v2-trajectory-geometry-results.json).
