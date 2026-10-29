# Coordinate diagnostic

**Centroid, scale and global 2D-frame correction do not recover unseen-room pose performance for the frozen raw-packet predictor. All three registered diagnostic gates fail.** This source-development experiment tests a possible explanation for the [packet-spectrum transfer failures](PACKET_TIME_RESEARCH.md). It does not introduce another RF model or claim a solution.

The supplied reference poses are 2D camera-plane annotations. Their frame can differ with viewpoint and body placement, even if the RF representation improves. We therefore tested a separately labeled canonical target: subtract each pose's own 17-joint centroid and divide by its own root-mean-square joint radius. Predictions use their own centroid and radius, never the query reference pose. A constant 2×2 rotation, or reflection followed by rotation, is fitted using source validation only. This operation cannot remove depth or yaw foreshortening and is not 3D reconstruction.

## Original image-plane scores remain unchanged

NJE divides each joint error by the reference skeleton's bounding-box diagonal. Lower is better. PCK is the fraction of joints within the stated fraction of that diagonal. Higher is better. These are dimensionless scores, not errors in millimeters.

| Held source room | Frozen family | Original NJE | PCK .05 | PCK .1 | PCK .2 |
|---|---|---:|---:|---:|---:|
| E01 | Natural RFF512 | .124433 | .192732 | .540512 | .842311 |
| E02 | Natural linear380 | .124998 | .196983 | .506224 | .847677 |
| E03 | Natural RFF512 | .134399 | .132002 | .459285 | .822944 |

The diagnostic asserts exact PCK readback and NJE agreement within 1e-12 against the frozen raw-packet report. It retains the original source-validation model choice; it does not select a different family using canonical or held scores.

## Canonical target still loses to matched controls

Each held-room fold has ten held people; the other rooms provide sixteen training and four source-validation people. All 27 action categories appear in training. Training mean and endpoint-clock controls are fitted **directly in each canonical target space**, rather than normalizing an old image-plane mean. Each control receives its own source-validation rotation/reflection choice. The clock control uses training labels at the same canonical endpoint, without an RF input. It is a control, not an input to the RF predictor.

| Held source room | Canonical RF NJE | TRAIN mean NJE | Endpoint clock NJE | Shuffled RF NJE | RF motion R² |
|---|---:|---:|---:|---:|---:|
| E01 | .065753 | .063798 | .063844 | .065640 | −.032047 |
| E02 | .064875 | .062313 | .062352 | .064796 | −.054044 |
| E03 | .063291 | .062073 | .062109 | .063188 | −.014169 |

For motion R², subtract each record's prediction mean from its predictions and its reference mean from its references, then compute one minus squared-error energy divided by reference variation energy. It evaluates temporal variation after removing both record means; it does not measure absolute-pose accuracy. The source-validation rotation choices for RF are proper rotations of .268142°, .062048° and .014410°. All three source-room transform choices were frozen before held scoring. Prediction shuffling is within each held recording, preserves its marginal poses, and improves RF NJE in every room.

The gate required lower canonical NJE and no worse PCK .05/.2 than **both** independently aligned controls, positive motion R², worse shuffled RF NJE, and full coverage in every room. All three gates fail. Valid coverage is 8,823/8,884/8,847 held frames; no degenerate or nonfinite actual predictions or targets were removed. Even a passing canonical gate would not satisfy the original image-plane acceptance criteria without a separately validated decoder back to that target.

Aggregation does not hide subgroup failures. Against the matched mean, RF wins on 4/10, 2/10 and 6/10 people in the native target, but **0/10 people in every room** in the canonical target. Of 27 actions, native-to-canonical win counts are 8→6, 0→1 and 16→6. Individual actions change in both directions: win→loss / loss→win counts are 7/5, 0/1 and 11/1. These are strict NJE comparisons across different targets, not statistical significance or evidence of original-task improvement. The [machine-readable report](domain-v2-coordinate-results.json) lists exact person/action IDs and includes the separate clock comparisons.

## What the reference geometry shows

The descriptive audit uses all 240,570 source reference frames: three rooms × ten people × 27 actions × 297 frames. After removing global action-by-endpoint means, room-associated shares of squared coordinate variation are .134116 in native coordinates, .068593 after centering, and .069088 after centering/scaling. These are associations confounded with people, camera viewpoint, execution and reference-label behavior. They do not identify camera change as the cause of failure.

A separate measure, within-record temporal variation divided by total target energy, is .005128/.039380/.037105 in those three spaces. It has a different denominator and is not an additional additive part of the preceding variance partition. Neither changing that ratio nor reducing raw squared variation establishes an accuracy improvement.

Twelve projected limb/torso/width lengths are divided by each pose's shoulder-midpoint to hip-midpoint distance. Shoulder-width medians are .594381/.592803/.610127 by room; the report includes all twelve ratios, room quartiles and within-record variation. These are 2D projected ratios, not physical bone lengths or camera-independent anatomy. No invalid torso frames occurred.

## Verification and reproduction

Seven small tests run with warnings treated as errors. They check own-pose normalization, zero-scale and nonfinite rejection, source-validation rotation/reflection, training-only canonical controls, variance identities, continuation requirements, coverage accounting and overwrite refusal. A separate task-local checker verifies 18 source-validation rotation candidates using an analytic atan2 solution independent of the builder's SVD, 60 held metric/shuffle readbacks, canonical training controls and all balanced variance identities. Maximum matrix/prior/metric differences are below 8e-16; variance sum-of-squares identities agree within 2.6e-9.

The first finite-data run is preserved. A subsequent synthetic infinity check exposed a helper warning before invalid rows were masked. The repaired helper masks rows before arithmetic and restores NaNs before return; it does not fill a pose. The fresh v2 run has **exactly the same actual metrics, subgroup scores and gates** as v1, and byte-identical label-geometry output. This repair is not an accuracy improvement. Its synthetic reproduction and original source bytes remain task-local.

Use an existing consented source-data setup following [data access](DATA_ACCESS.md) and [packet-spectrum reproduction](PACKET_TIME_RESEARCH.md). The diagnostic consumes the verified source pose cache, packet feature cache and completed frozen packet-transfer experiment. It does not reconstruct raw RF or fit an RF readout. From the repository root:

```sh
uv run pytest -q -W error tests/test_coordinate_diagnostic.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python -W error::RuntimeWarning scripts/domain_coordinate_diagnostic.py \
  --base work/domain-source30-rf-v1 \
  --cache work/domain-packet-full-cache-v1 \
  --experiment work/domain-packet-transfer-v1 \
  --out work/domain-coordinate-diagnostic-fresh
uv run python scripts/seal_research_artifacts.py \
  --root work/domain-coordinate-diagnostic-fresh
```

The output directory must be new. An existing output is refused. The script writes its configuration, frozen input/source identities and executed source snapshot before computation; it writes all source-validation choices before held scores. Private caches and reference poses are not redistributed. The byte manifest records file identity, not scientific certification or immutability.

Executed diagnostic SHA-256: `cb4aeb6b5f30965b8e2ce4fe203bd2a537430c8720b62b41231f01c1e967e332`. Test SHA-256: `fc2bf27f53e2ca79909525023a350a14671716f28c5000fdcd220db7f2b46f6b`. The [public JSON](domain-v2-coordinate-results.json) includes exact configuration, all bound input/source hashes, selected transforms, original and canonical metrics, subgroup reversals, geometry summaries and verification results. All work is SOURCE E01–E03 development. There was no RGB, E04, paid service, query-specific reference alignment or additional RF training.
