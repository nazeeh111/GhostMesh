# Trained artifacts

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
