# Trained artifacts

`local-kernel.npz` is a separately labeled **same-condition diagnostic** added
during v2 research. It stores 118 source CSI frames (111 causal reference
windows), training-only normalization, RBF bandwidth and dual coefficients.
Its 3.2 MiB compressed representation reconstructs the original double-precision
kernel design rather than storing that 45 MiB design matrix. It predicts
separated time blocks of E01/S01/A12; room/person/action transfer is unvalidated.
The dataset-derived reference CSI and coefficients retain CC BY-NC 4.0 terms.
No uncertainty estimator is supplied. [Evidence and reproduction](../docs/LOCAL_OBSERVABILITY.md).

```sh
uv run python scripts/local_kernel_replay.py infer \
  --model models/local-kernel.npz --recording examples/local-csi.npz \
  --out work/local-recomputed.json
```

SHA-256: `71e9ebb8f78f172d7573a8e2502d6833120c7a9b64622fc02ba3cc8dedd8838a`.

`ghostmesh.pt` is the predeclared primary temporal-plus-teacher model, seed 2027,
selected by validation normalized joint error. It has **54,451 parameters**.
Training used a 40-epoch budget; the selected checkpoint is epoch 1.
The release does not choose a different primary model from test rankings.

Input is nonnegative CSI amplitude, with three channels, 114 subcarriers and ten
packets per released frame. Inference uses eight causal frames ending at the
prediction time. The shipped sample contains CSI and timestamps only. Its 17-joint
outputs use the declared 640-square annotation reference plane. Source data,
normalization, runtime and manifest identities are stored inside the checkpoint.

```sh
uv run ghostmesh infer-recording --recording examples/heldout-csi.npz \
  --checkpoint models/ghostmesh.pt --out examples/recomputed.json
uv run python scripts/reproduce_demo.py
```

The second command checks all valid pose/confidence values against the shipped
replay. It compares predictions, not measured latency or unknown-domain metadata.

SHA-256 of `ghostmesh.pt`:

```text
14fd4f01cd04c41c7393222d50954bb13ffeea994141d575ec288ad62ee2a078
```

These are research weights. Held-out scores, ablations, uncertainty reliability
and transfer limits are documented in the [release results](../results/cohort20-v1/results.md) and
[model card](../docs/MODEL_CARD.md). No 3D, multi-person or ESP32 pose validation
is claimed. Sensor layouts cannot be made compatible by reshaping their arrays.

MM-Fi-derived weights retain [CC BY-NC 4.0](../data/LICENSE.md) attribution and
noncommercial terms. The original implementation uses [MIT](../LICENSE).

## Unseen-room checkpoint

`ghostmesh-environment.pt` is the separately trained seed-2027 temporal model
whose development excludes E04. Its validation-selected checkpoint is epoch 6.
Use it with `examples/unseen-room-csi.npz` and the E04 replay. SHA-256:

```text
7a19bdd6f8a81630253a21f3709063c45d85cf7ffc4e2e93748b16a3d12fc5ef
```

The test results do not establish reliable unseen-room pose recovery; confidence
is poorly calibrated in that domain. [Integrated findings](../results/README.md).
