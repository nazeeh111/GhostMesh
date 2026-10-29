# Packet-spectrum research

**Unseen-room pose recovery remains unresolved. Both raw-power and normalized spectral-shape experiments fail their registered motion and pose gates.** A fixed local recording contains recoverable motion, but that result does not generalize reliably to different people in different source rooms.

These are source-development experiments, seed 2027, following earlier failed source experiments. E04 remains sealed. No target calibration, query label statistics, previous reference pose input, camera input at inference, hardware or paid service was used. Provided 2D reference labels are the evaluation targets; this is not independent 3D validation.

## What changed

The [MM-Fi paper, sections 3.1–3.2](https://arxiv.org/html/2305.10345v2), describes roughly 100 Hz firmware CSI, ten-column matrices over 100 ms, then synchronization at 10 Hz. Released per-packet timestamps were unavailable. An RF-only continuity audit passed its heuristic in 89/90 fixed A03/A07/A12 recordings, including E01/S01/A12. Continuity supports testing the column-order premise; it does not independently establish packet order, phase coherence or a calibrated sampling clock.

The new feature retains complex receiver ratios, subtracts each causal window's complex mean, applies a Hann window and an 80-sample complex FFT. Five absolute nominal-frequency bands on each sign retain positive/negative power separately. Averaging powers across six adjacent subcarriers produces 2 links × 19 bins × 10 bands = 380 features. The signal spans eight grouped frames; the inherited conservative 40-frame validity mask also applies. Constant amplitude with changing phase remains valid. A trace requires at least 76/80 finite samples; any remaining missing samples use that causal window's complex mean. All variants share the same rejection mask. There is no amplitude-only flatness rejection.

A separate, predeclared follow-up transforms each ten-band dB vector into log-power fractions:

`fraction_dB = x − (10/ln 10) logsumexp(x ln 10/10)`

This pointwise operation cancels a common additive cached-band power shift without learning target statistics. Its physical gain interpretation requires gain to be constant within the window and common across the bands and averaged subcarriers. Multipath, unequal subcarrier gains, time-varying gain and earlier clipping/floors limit that assumption. Total dynamic power is discarded. None of the actual cached bands reached the numerical pre-log floor. This is a narrow nuisance-normalization hypothesis, not universal room invariance.

These are nominal signed spectral bands, not body velocity profiles, calibrated Doppler, physical velocity, angle of arrival or 3D pose. Three co-located receive channels do not supply the spatial geometry needed for those stronger claims.

## Local positive control

The fixed E01/S01/A12 recording uses purged, nonoverlapping raw time blocks. Current-pose counts are 111/45/104 train/validation/test; lag-eight displacement counts are 110/44/103 because both reference endpoints must remain within the block. Each kernel and regularization value is selected on validation only.

The selected natural-spectrum RBF reaches hip-relative pose motion R² 0.770880 and pose NJE 0.043921 versus static NJE 0.103537. Full-joint displacement R² is 0.886979, versus −0.467938 after prediction shuffling. Within-frame packet permutation scores 0.865580 and whole-window permutation 0.796736. **Freshly paired raw-CSI and quotient RBF displacement controls are stronger: 0.948651 and 0.971184.** The FFT is useful compression locally, not a local accuracy upgrade. This previously studied recording is a development control, not an untouched person/room test.

## Full source-room transfer

All 810 recordings, 30 people and 27 action categories in E01–E03 are included. Each fold holds out one room's ten people; the other rooms supply 16 training and four validation people. All action categories occur in training. The common quality mask retains 26,554 of 26,556 covariance endpoints, with only two additional spectral drops. Every current/lag-eight label join and hip-relative projection is checked against the original source annotations and covariance metadata.

Four arms share these rows: natural linear 380, natural random Fourier features (RFF) 512, within-frame ten-column permutation RFF512, and whole-80-sample permutation RFF512. Training-only means/scales, median bandwidth, seed and ridge grid `.001/.01/.1/1/10` are identical across representations. Each target independently selects its ridge value using source validation: NJE for current full 2D pose, raw mean-square error for full-joint/root/hip-relative displacement. Natural linear versus RFF is also selected separately per target using source validation. All three folds' choices are frozen before held scoring.

Full-joint lag-eight displacement R², using **zero displacement** as the reference:

| Representation / arm | E01 | E02 | E03 |
|---|---:|---:|---:|
| Raw power / natural linear | −.002336 | −.001667 | −.006468 |
| Raw power / natural RFF | −.003267 | −.015781 | −.012579 |
| Raw power / within-frame permutation | −.003922 | −.017081 | −.014074 |
| Raw power / whole-window permutation | −.003417 | −.001955 | −.012314 |
| Fractions / natural linear | .002382 | .000550 | −.006688 |
| Fractions / natural RFF | .002087 | .000641 | .001390 |
| Fractions / within-frame permutation | .002243 | −.000042 | .000104 |
| Fractions / whole-window permutation | .000246 | .000266 | −.000487 |

Normalization improves the matched natural RFF in every room, but its positive values explain only about 0.06–0.21% of zero-reference displacement energy. The E01 packet-permuted RFF is slightly better. The actual source-selected normalized family is RFF/RFF/**linear**, scoring .002087/.000641/**−.006688**. Choosing the held-preferred E03 RFF would violate the selector. Source-selected prediction-shuffled R² is −.002299/−.004809/−.013289; worse shuffling alone does not establish useful recovery. The endpoint-only source clock control, with no RF, scores −.002527/−.002681/−.002166. Main RF models never receive the clock feature.

Current **full 2D pose** NJE, lower is better:

| Representation / arm | E01 | E02 | E03 |
|---|---:|---:|---:|
| Raw power / natural linear | .121673 | .124998 | .133157 |
| Raw power / natural RFF | .124433 | .127404 | .134399 |
| Raw power / within-frame permutation | .124660 | .122926 | .134371 |
| Raw power / whole-window permutation | .124714 | .122092 | .134372 |
| Fractions / natural linear | .119563 | .119745 | .135682 |
| Fractions / natural RFF | .119566 | .121281 | .135190 |
| Fractions / within-frame permutation | .120032 | .120870 | .135726 |
| Fractions / whole-window permutation | .121101 | .117356 | .134647 |
| Source static mean | .121134 | .116088 | .134400 |

Source validation selects normalized linear pose in all rooms. Its PCK@.05/@.2 is .225747/.850598, .216602/.860807 and .132793/.821880, compared with static .181684/.854811, .239432/.868177 and .135466/.823270. Even E01's lower NJE loses at the coarser PCK threshold. Normalized pose NJE reverses against static for 1/10, 9/10 and 7/10 people; full displacement R² is nonpositive for 2/10, 3/10 and 8/10 people and 13/27, 17/27 and 22/27 action strata. The [aggregate evidence](domain-v2-packet-time-results.json) includes every arm, component target, control, selection and reversal identity.

NJE/PCK divide image-plane joint error by the current reference body's bounding-box diagonal. Displacement R² uses raw image-plane differences without query-mean repair. Separately selected component heads are not combined into an exact decomposition. Frozen covariance comparisons are restricted to their existing hip-relative pose/displacement heads on identical retained held rows; their temporal context, optimization and original training mask differ. No missing full-pose/root prediction is invented.

## Verification and reproduction

All 22 focused tests pass with warnings as errors; Ruff passes. The public verifier reconstructs every source map/bandwidth/validation choice and independently solves all 60 actual ridge systems per representation. Maximum relative residual is 1.57e−14 raw and 3.98e−14 normalized; independent direct/eigen coefficient differences are at most 6.68e−13 and 1.08e−13. All 24 saved-model prediction replays match exactly. Each study also rechecks 12 prediction-shuffle reports, nine static/zero/endpoint controls, nine covariance comparisons, person/action breakdowns and both gates. Raw mode additionally checks all annotation joins and regenerates 2,430 fixed RF windows with all three variants; normalized mode checks every pointwise transformation and identical metadata. This spot-check count is not a claim to regenerate every raw FFT window during verification.

Build the typed source cache and independently verified full covariance inputs using [data access instructions](DATA_ACCESS.md) and [the covariance commands](COVARIANCE_RESEARCH.md). Source data and fitted arrays remain local; they are not bundled in Git. Then use fresh output paths:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python scripts/domain_packet_transfer_probe.py build --out work/my-packet-cache
python scripts/domain_packet_transfer_probe.py fit \
  --cache work/my-packet-cache --out work/my-packet-raw
python scripts/verify_packet_transfer_probe.py --mode raw \
  --experiment work/my-packet-raw --cache work/my-packet-cache \
  --report work/my-packet-raw/independent-verification.json
python scripts/domain_spectral_shape_probe.py normalize \
  --cache work/my-packet-cache --out work/my-packet-shape-cache
python scripts/domain_spectral_shape_probe.py fit \
  --cache work/my-packet-shape-cache --raw-experiment work/my-packet-raw \
  --out work/my-packet-shape
python scripts/verify_packet_transfer_probe.py --mode shape \
  --experiment work/my-packet-shape --cache work/my-packet-shape-cache \
  --raw-cache work/my-packet-cache --raw-experiment work/my-packet-raw \
  --report work/my-packet-shape/independent-verification.json
python -m pytest -q -W error tests/test_doppler_features.py \
  tests/test_packet_transfer_probe.py tests/test_spectral_shape_probe.py \
  tests/test_verify_packet_transfer_probe.py
```

The build/fit defaults expect the documented `work/domain-source30-rf-v1`, `work/domain-covariance-full-cache-v1` and verified `work/domain-covariance-full27-v1`; pass the corresponding `--base`, `--covariance`, `--full-probe`, `--baseline`, `--raw-source` and `--covariance-cache` options when using different locations. Existing output/report paths are refused. For the local continuity/A12 control, first build, verify and seal the same-condition predecessor as described in [local observability reproduction](LOCAL_OBSERVABILITY.md#reproduce), then run the exact audit and positive commands:

```bash
python scripts/domain_same_condition_probe.py \
  --cache work/domain-source30-rf-v1 --out work/local-reproduction
python scripts/verify_same_condition_probe.py \
  --cache work/domain-source30-rf-v1 --out work/local-reproduction
python scripts/seal_research_artifacts.py --root work/local-reproduction
python scripts/domain_packet_time_probe.py audit \
  --cache work/domain-source30-rf-v1 --out work/my-packet-clock
python scripts/domain_packet_time_probe.py positive \
  --cache work/domain-source30-rf-v1 --clock work/my-packet-clock \
  --predecessor work/local-reproduction --out work/my-packet-positive
```

The predecessor supplies the earlier raw/quotient pose references; the new positive probe independently recomputes displacement controls on its paired endpoint rows.

Cache extraction took 210.8 seconds; raw/normalized fit plus evaluation took 82.7/44.0 seconds with local single-thread CPU settings. A public compute-only benchmark is available:

```bash
python scripts/benchmark_packet_transfer.py --mode shape \
  --experiment work/my-packet-shape --report work/my-packet-shape/latency.json
```

Two observed warmed 200-window runs varied from 0.87 to 1.93 ms median, with p95 0.92–2.47 ms, illustrating host-load sensitivity. This includes quotient/FFT/normalization/RFF and four readouts on one fixed real window. It excludes disk, acquisition, UI, synchronization and conservative mask handling. These timings are not application FPS, do not change nominal 10 Hz output, and do not establish model efficacy. Source/config/data/model hashes and exact numerical receipts are retained in the linked JSON and original task artifacts. No failed score is overwritten or promoted to a solved-room result.
