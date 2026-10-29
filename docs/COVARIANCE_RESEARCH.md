# Receiver covariance research: limited source result

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
calibrated-product claim. A complementary filter using the independent RF pose
head as its current configuration estimate is the next development experiment.
