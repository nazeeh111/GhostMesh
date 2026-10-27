# Measurement interpretation correction

## MM-Fi magnitude units

The original MM-Fi files contain `CSIamp` and `CSIphase`. An independent
numerical audit strongly supports interpreting `CSIamp` as **20 log10 of
complex-coefficient magnitude in an unknown receiver reference scale**, with
phase in radians. This is equivalent to 10 log10 of squared magnitude. It is
not established as calibrated RSSI in dBm, physical power, or raw ADC samples.

For 405 development recordings from E01–E03, all **411,369,834 finite
coefficients** reconstructed as
`10**(CSIamp/20) * exp(1j*CSIphase)` have integer real and imaginary components
within absolute tolerance `1e-8`. The maximum rounding residual is
`3.979e-13`. An independent 18-file check reproduced the finding on 61,560
additional coefficients. The inspected original documentation does not
explicitly certify this unit convention; this is a strong empirical inference.
No E04 recordings were decoded for this audit.

[Full audit receipt](mmfi-unit-audit.json) binds counts and ordered source-file
digests. Three original float64 source frames, with no pose labels or images,
are included for a small independent check:

```sh
uv sync --locked --extra test
uv run python scripts/check_mmfi_units.py
```

[Sample provenance](../examples/rf-units-source.json) records their original
MAT hashes. MM-Fi data remains CC BY-NC 4.0 with the authors' attribution in
[NOTICE](../NOTICE.md) and [data terms](../data/LICENSE.md).

## Effect on version 0.1.0

The legacy preprocessor applies `log1p` before robust normalization. On these
MM-Fi fields that adds nonlinear compression to an already logarithmic
magnitude, so its linear-amplitude physical interpretation was inaccurate.
The published scores, splits, checkpoints and real replays remain measurements
of that exact frozen transform. They have not been recomputed or relabeled as
results of corrected preprocessing.

An additional monotone transform is not intrinsically an invalid neural
feature map. This observation alone does not explain all model failures or
prove that removing it improves accuracy. A separately versioned experiment
must assess explicit log-magnitude and phase representations with matched
controls. Do not feed newly transformed features to an old checkpoint.

The generic synthetic and ESP32 routes can supply actual linear magnitudes;
they require their own declared measurement contracts. Removing logarithms
globally would introduce another unit error.

Phase is also available in original MAT files but was omitted from the
amplitude-only prepared release. Inter-antenna phase/quotient experiments are
separate research, with no validated replacement model or performance claim
implied by this correction. Known room-transfer and motion-recovery limits
remain stated in the [measured findings](../results/README.md).
