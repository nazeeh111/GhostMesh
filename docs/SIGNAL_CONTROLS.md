# Supplemental signal controls

These diagnostics are frozen before the first original suite test report. They do not change the twelve-arm study, training, validation selection, architectures, or primary hypotheses. The freeze records the exact plan/config, five supplemental files, report validator, five frozen runtime modules, original suite runner/preregistration, runtime versions and both manifests. Manifests bind prepared sequence byte hashes. Validation-selected checkpoint hashes are bound at execution after training; they are not chosen using diagnostic outcomes.

Run from the repository root using the same locked environment as the original suite. Before any `experiments/cohort20-final-v1/*/test.json` exists:

```sh
.venv/bin/python scripts/run_signal_controls.py --freeze \
  --suite experiments/cohort20-final-v1 \
  --subject-manifest data/mmfi-cohort20/manifest.json \
  --environment-manifest data/mmfi-cohort20/cross-environment.json \
  --out results/signal-controls-v1
```

Do not edit frozen supplemental files after this step. Complete all twelve training jobs and run the unchanged main suite to completion before evaluating controls:

```sh
.venv/bin/python scripts/run_signal_controls.py \
  --suite experiments/cohort20-final-v1 \
  --subject-manifest data/mmfi-cohort20/manifest.json \
  --environment-manifest data/mmfi-cohort20/cross-environment.json \
  --out results/signal-controls-v1
```

The source-room prior fits one mean 2D position and isotropic coordinate variance per joint/environment using only accepted training endpoints and masks at stride 8. It uses no CSI features or action IDs. Room metadata is an explicit oracle input. An unseen room such as E04 uses the global training fit; a joint without room labels also uses that joint's global training fit. No held-out labels enter fitting. Variance is floored at 1e-6, matching the original mean-pose baseline convention.

For each protocol, one NumPy default_rng permutation with seed 90210 independently shuffles complete accepted test windows within each environment in sorted room order. Both baseline and temporal seed 2027 receive identical indices for that protocol. Protocols have different accepted endpoint sets, so each protocol constructs its own permutation. Each causal window retains its internal eight-frame order. Evaluation is independent across windows in eval mode, so collecting aligned outputs once and permuting pose/log_sigma together is equivalent to permuting complete input windows. Targets, masks, timestamp/frame-index endpoint order, and coverage stay fixed. Uniform random permutations may contain fixed points, which are retained and counted. The output records SHA256 of little-endian int64 indices and ordered endpoint metadata for reproducibility.

Before scoring a permutation, aligned PCK, normalized joint error, eligible/excluded frame counts, valid-joint counts and accepted coverage must reproduce the original test report (only the operational cache-reuse flag may differ) (PCK/NJE absolute tolerance 1e-6). Frozen-source, runtime, manifest, all-twelve-training and completed original suite guards run before data/model loading. The existing report validator verifies the original complete suite and all artifact bindings. New files/config/source drift, late freezes and existing result replacement are refused.

The JSON contains all aligned/permuted/prior aggregate metric fields, calibration diagnostics, fitted training priors/counts, coverage and hashes; CSV and Markdown provide compact tables. These are descriptive controls without p-values, significance claims, further permutations, tuning or model selection. Differences test sample-specific CSI-target correspondence beyond static room pose distributions. Person, action and recording cues can remain; a drop after permutation is not causal motion or localization proof. A no-drop result limits claims about sample-specific signal use. The source-room oracle prior is especially relevant where camera/room views differ; E04 fallback cannot remove the room/person confound in the cross-environment protocol.

Software-only checks, with no research data or model evaluation:

```sh
.venv/bin/python -m pytest tests/test_controls.py -q
```
