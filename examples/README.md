# Replay inputs

`heldout-csi.npz` contains **CSI amplitude and timestamps only**, with no pose
annotations or camera images. It is the predeclared E01/S05/A02 example from
the primary cohort's cross-subject test split. S05 is absent from that model's
training and validation people. The action category occurs in development.

There are 297 frames with shape `[297, 3, 114, 10]`, at nominal 10 Hz. Timestamps
are reconstructed from source frame IDs. The first seven frames provide startup
context for the eight-frame causal window and withhold the pose. The coordinate
reference plane and original synchronization limits are documented in the
[data guide](../docs/DATA_ACCESS.md).

```sh
uv run ghostmesh infer-recording --recording examples/heldout-csi.npz \
  --checkpoint models/ghostmesh.pt --out examples/recomputed.json
uv run python scripts/reproduce_demo.py
uv run ghostmesh serve --replay examples/recomputed.json
```

The recomputed recording replay marks the domain as unverified because this
input file carries no subject/environment metadata. The saved dataset replay
at `ghostmesh/examples/heldout-replay.json` separately retains the manifest's
test identity and optional supplied keypoints. Those reference labels are
attached after prediction and never enter the model. Reproduction compares
poses, confidence and timestamps; timing and domain metadata may differ.

`diagnostic-replay.json` is generated software-test data. Its illustrative
motion and signal do not establish sensing accuracy.

MM-Fi excerpts and derived replays retain [CC BY-NC 4.0](../data/LICENSE.md)
terms, attribution and noncommercial scope. Aggregate accuracy belongs to the
[full held-out report](../results/cohort20-v1/results.md), rather than one
selected animation.

`unseen-room-csi.npz` is the separately preselected E04/S31/A02 example.
Use `models/ghostmesh-environment.pt`, whose development excludes E04, and
`unseen-room-replay.json`. Both recordings reproduced all 290 valid predictions
and confidence values exactly in the verified release environment.
