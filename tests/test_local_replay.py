"""Real held-time RF-only readback; the restricted local model is not a transfer claim."""

import json
from pathlib import Path

import numpy as np

from ghostmesh.local_kernel import infer_recording, load_model, predict

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_local_kernel_reproduces_all_104_held_time_predictions(tmp_path):
    model = ROOT / "models/local-kernel.npz"
    recording = ROOT / "examples/local-csi.npz"
    reference = json.loads((ROOT / "examples/local-motion-replay.json").read_text())
    with np.load(recording, allow_pickle=False) as source:
        assert set(source.files) == {
            "rf",
            "timestamps",
            "magnitude_units",
            "phase_units",
        }
        assert source["rf"].shape == (111, 6, 114, 10)
        prediction = predict(load_model(model), source["rf"])
    expected = np.array([f["pose"] for f in reference["frames"][7:]])
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-9)
    result = infer_recording(model, recording, tmp_path / "inference.json")
    assert len(result["frames"]) == 111
    assert result["provenance"]["subject"] == "unknown"
    assert result["provenance"]["confidence_available"] is False
    assert all("ground_truth" not in f for f in result["frames"])
    assert sum(f["pose"] is not None for f in result["frames"]) == 104
    np.testing.assert_allclose(
        np.array([f["pose"] for f in result["frames"][7:]]), expected, rtol=0, atol=1e-9
    )
