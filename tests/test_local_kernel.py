import numpy as np
import pytest
from scipy.linalg.blas import dgemm

from ghostmesh.local_kernel import (
    KIND,
    WIDTH,
    design,
    infer_replay,
    load_model,
    predict,
)


def model(tmp_path):
    rng = np.random.default_rng(21)
    rf = rng.normal(size=(12, 6, 114, 10)).astype("float32")
    rf[:, :3] += 40
    checkpoint = tmp_path / "kernel.npz"
    np.savez(
        checkpoint,
        kind=np.asarray(KIND),
        magnitude_units=np.asarray("log_magnitude_db"),
        phase_units=np.asarray("radians"),
        reference_rf=rf,
        input_mean=np.zeros(WIDTH),
        input_scale=np.ones(WIDTH),
        dual=rng.normal(size=(5, 34)) * 0.01,
        target_mean=np.full(34, 0.5),
        bandwidth_squared=np.asarray(500.0),
    )
    return load_model(checkpoint), rf, checkpoint


def test_kernel_prediction_matches_direct_formula_and_is_causal(tmp_path):
    m, rf, _ = model(tmp_path)
    x = design(rf)
    d = np.square(x[:, None] - x[None, :]).sum(2)
    expected = (
        (0.5 + dgemm(1.0, np.exp(-d / 1000.0), m["dual"])).reshape(-1, 17, 2).clip(0, 1)
    )
    np.testing.assert_allclose(predict(m, rf), expected, rtol=0, atol=1e-12)
    changed = rf.copy()
    changed[10:] += 50
    np.testing.assert_array_equal(predict(m, rf)[:3], predict(m, changed)[:3])


def test_unverified_recording_does_not_inherit_training_identity_or_fake_confidence(
    tmp_path,
):
    m, rf, _ = model(tmp_path)
    result = infer_replay(
        m,
        rf,
        np.arange(12) / 10.0,
        checkpoint_name="kernel.npz",
        checkpoint_sha256="x",
        recording_sha256="y",
    )
    assert result["provenance"]["kind"] == "recording_inference"
    assert result["provenance"]["subject"] == "unknown"
    assert result["provenance"]["confidence_available"] is False
    assert result["provenance"]["context_frames"] == 8
    slow = infer_replay(
        m,
        rf,
        np.arange(12) / 5.0,
        checkpoint_name="k",
        checkpoint_sha256="x",
        recording_sha256="y",
    )
    assert "context_seconds" not in slow["provenance"]
    assert slow["provenance"]["training_context_seconds"] == 0.7
    assert not any("ground_truth" in f for f in result["frames"])
    assert all(f["pose"] is None for f in result["frames"][:7])
    assert all(f["pose"] is not None for f in result["frames"][7:])
    gap_times = np.arange(12) / 10.0
    gap_times[8:] += 1
    gaps = infer_replay(
        m,
        rf,
        gap_times,
        checkpoint_name="k",
        checkpoint_sha256="x",
        recording_sha256="y",
    )
    assert gaps["frames"][7]["pose"] is not None
    assert all(f["pose"] is None for f in gaps["frames"][8:])
    rf[:, 3:] = np.nan
    invalid = infer_replay(
        m,
        rf,
        np.arange(12) / 10.0,
        checkpoint_name="k",
        checkpoint_sha256="x",
        recording_sha256="y",
    )
    assert all(f["pose"] is None for f in invalid["frames"])


def test_checkpoint_units_and_shapes_are_enforced(tmp_path):
    _, _, path = model(tmp_path)
    with np.load(path, allow_pickle=False) as source:
        fields = {k: source[k] for k in source.files}
    for change in (
        {"magnitude_units": np.asarray("linear")},
        {"input_scale": np.zeros(WIDTH)},
        {"dual": np.zeros((6, 34))},
    ):
        np.savez(tmp_path / "bad.npz", **(fields | change))
        with pytest.raises(ValueError):
            load_model(tmp_path / "bad.npz")
