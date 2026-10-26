import json
import numpy as np
import pytest
from ghostmesh.data import WindowDataset, load_manifest, read_sequence, sha256
from ghostmesh.prepare import make_fixture, prepare_mmfi, split_subject
from ghostmesh.preprocess import prepare_window, signal_quality


@pytest.fixture
def manifest(tmp_path):
    return make_fixture(tmp_path / "fixture", frames=16)


def test_window_is_causal_and_uses_end_label(manifest):
    d = WindowDataset(manifest, "train", window=4)
    record, end = d.windows[3]
    seq = read_sequence(manifest.parent / d.entries[record]["path"])
    item = d[3]
    np.testing.assert_allclose(
        item["csi"].numpy(), prepare_window(seq["csi"][end - 3 : end + 1])
    )
    np.testing.assert_array_equal(item["pose"].numpy(), seq["pose"][end])
    assert item["time"] == seq["timestamps"][end]
    assert item["csi"].shape == (4, 3, 24, 4)


def test_invalid_signal_removes_affected_windows(manifest):
    d = WindowDataset(manifest, "test", window=4)
    assert d.rejected_windows == 5
    coverage = d.coverage()
    assert coverage["candidate_windows"] == 26
    assert coverage["accepted_windows"] == 21
    assert coverage["accepted_fraction"] == pytest.approx(21 / 26)
    assert coverage["rejection_reasons"] == {"missing_signal": 5}
    assert coverage["warmup_frames"] == 6
    for i in range(len(d)):
        assert np.isfinite(d[i]["csi"].numpy()).all()


def test_coverage_tracks_overlapping_signal_gap_and_label_failures(manifest):
    from ghostmesh.metrics import pose_metrics

    m = json.loads(manifest.read_text())
    row = m["sequences"][0]
    p = manifest.parent / row["path"]
    seq = read_sequence(p)
    seq["csi"][4] = np.nan
    seq["timestamps"][10:] += 0.5
    seq["valid"][12] = False  # This target also lies inside a timestamp gap.
    seq["valid"][15] = False
    seq["valid"][15, 0] = True  # Useful joint supervision, undefined pose-box scale.
    np.savez_compressed(p, **seq)
    row["sha256"] = sha256(p)
    manifest.write_text(json.dumps(m))
    d = WindowDataset(manifest, "train", window=4)
    c = d.coverage()
    assert c["candidate_windows"] == 65
    assert c["accepted_windows"] == 58
    assert c["rejected_windows"] == 7
    assert c["rejection_reasons"] == {
        "missing_signal": 4,
        "timestamp_gap": 3,
        "no_valid_target_joints": 1,
    }
    assert sum(c["rejection_reasons"].values()) > c["rejected_windows"]
    assert c["sequences"][row["id"]]["candidate_windows"] == 13
    assert c["sequences"][row["id"]]["accepted_fraction"] == pytest.approx(6 / 13)
    targets = np.stack([d[i]["pose"].numpy() for i in range(len(d))])
    valid = np.stack([d[i]["valid"].numpy() for i in range(len(d))])
    metrics = pose_metrics(targets, targets, valid)
    assert metrics["valid_frames"] == 57
    assert metrics["excluded_frames"] == 1


def test_hash_modification_detected(manifest):
    m = load_manifest(manifest)
    p = manifest.parent / m["sequences"][0]["path"]
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(manifest)


def test_split_subject_leakage_detected(manifest):
    m = json.loads(manifest.read_text())
    m["protocol"] = "cross_subject"
    m["sequences"][-1]["subject"] = m["sequences"][0]["subject"]
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Subject leakage"):
        load_manifest(manifest)


def test_nonmonotonic_timestamps_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(
        p,
        csi=np.ones((3, 3, 4, 2)),
        pose=np.ones((3, 17, 2)) * 0.5,
        valid=np.ones((3, 17), bool),
        timestamps=[0.0, 0.1, 0.1],
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        read_sequence(p)


def test_signal_processing_missing_and_degenerate():
    rng = np.random.default_rng(1)
    x = rng.uniform(1, 3, (4, 3, 24, 4)).astype("float32")
    x[0, 0, 0, 0] = np.nan
    assert np.isfinite(prepare_window(x)).all()
    assert not signal_quality(np.ones_like(x))[0]
    x[:2] = np.nan
    with pytest.raises(ValueError, match="missing_signal"):
        prepare_window(x)


def test_mmfi_join_and_3d_guard(tmp_path):
    from scipy.io import savemat

    raw = tmp_path / "raw" / "E01" / "S01" / "A02"
    (raw / "wifi-csi").mkdir(parents=True)
    (raw / "rgb").mkdir()
    savemat(raw / "wifi-csi" / "frame001.mat", {"CSIamp": np.ones((3, 114, 10))})
    np.save(raw / "rgb" / "frame002.npy", np.zeros((17, 2)))
    p = {
        "dataset": "MM-Fi",
        "source": "test fixture",
        "license": "CC BY-NC 4.0",
        "consent_basis": "test",
    }
    with pytest.raises(ValueError, match="Unpaired"):
        prepare_mmfi(
            tmp_path / "raw", tmp_path / "out1", width=640, height=480, provenance=p
        )
    (raw / "rgb" / "frame002.npy").rename(raw / "rgb" / "frame001.npy")
    np.save(raw / "rgb" / "frame001.npy", np.zeros((17, 3)))
    with pytest.raises(ValueError, match="no implicit 3D"):
        prepare_mmfi(
            tmp_path / "raw", tmp_path / "out2", width=640, height=480, provenance=p
        )


def test_official_test_subjects_and_unseen_environment():
    assert [
        s
        for i in range(1, 41)
        if (s := f"S{i:02d}")
        and split_subject(s, f"E{(i - 1) // 10 + 1:02d}", "cross_subject") == "test"
    ] == [f"S{i:02d}" for i in range(5, 41, 5)]
    assert split_subject("S31", "E04", "cross_environment") == "test"


def test_manifest_cannot_escape_directory(manifest):
    m = json.loads(manifest.read_text())
    m["sequences"][0]["path"] = "../other.npz"
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="inside manifest"):
        load_manifest(manifest)


@pytest.mark.parametrize("normalize", [True, False])
def test_window_memmap_is_exact_readonly_and_reused_without_decoding(
    manifest, monkeypatch, normalize
):
    import ghostmesh.data as module

    uncached = WindowDataset(manifest, "test", window=4, normalize=normalize, stride=2)
    cached = WindowDataset(
        manifest, "test", window=4, normalize=normalize, stride=2, cache_windows=True
    )
    assert cached.coverage() == uncached.coverage()
    assert cached.storage_info()["reused"] is False
    assert cached.storage_info()["mode"] == "readonly_window_memmap"
    assert cached.storage_info()["checksums_verified"] is True
    assert cached._mapped["csi"].flags.writeable is False
    assert len(cached.cache) == 0
    for i in range(len(cached)):
        a, b = uncached[i], cached[i]
        for key in ("csi", "pose", "valid"):
            np.testing.assert_array_equal(a[key].numpy(), b[key].numpy())
        for key in ("sequence", "subject", "environment", "frame_index", "time"):
            assert a[key] == b[key]
    expected = cached[0]["csi"].numpy().copy()
    cached[0]["csi"].fill_(999.0)
    np.testing.assert_array_equal(cached[0]["csi"].numpy(), expected)
    monkeypatch.setattr(
        module,
        "read_sequence",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("Cache reuse must not decode raw NPZ")
        ),
    )
    reused = WindowDataset(
        manifest, "test", window=4, normalize=normalize, stride=2, cache_windows=True
    )
    assert reused.storage_info()["reused"] is True
    assert reused.coverage() == uncached.coverage()
    np.testing.assert_array_equal(reused[0]["csi"].numpy(), expected)


def test_window_cache_refuses_corruption_without_overwriting(manifest):
    cached = WindowDataset(manifest, "train", window=4, cache_windows=True)
    directory = (
        manifest.parent / ".ghostmesh-cache" / cached.storage_info()["cache_key"]
    )
    p = directory / "csi.npy"
    with p.open("r+b") as stream:
        stream.seek(-1, 2)
        old = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([old[0] ^ 1]))
    changed = sha256(p)
    with pytest.raises(ValueError, match="checksum mismatch: csi"):
        WindowDataset(manifest, "train", window=4, cache_windows=True)
    assert sha256(p) == changed


def test_window_cache_keys_bind_split_stride_normalization_and_preprocess(
    manifest, monkeypatch
):
    import ghostmesh.data as module

    datasets = [
        WindowDataset(
            manifest,
            split,
            window=4,
            stride=stride,
            normalize=normalize,
            cache_windows=True,
        )
        for split, stride, normalize in (
            ("train", 1, True),
            ("validation", 1, True),
            ("train", 2, True),
            ("train", 1, False),
        )
    ]
    keys = {d.storage_info()["cache_key"] for d in datasets}
    assert len(keys) == 4
    original = module.sha256
    monkeypatch.setattr(
        module,
        "sha256",
        lambda p: "f" * 64 if str(p).endswith("preprocess.py") else original(p),
    )
    changed = WindowDataset(manifest, "train", window=4, cache_windows=True)
    assert changed.storage_info()["cache_key"] not in keys


def test_incomplete_window_cache_is_not_silently_rebuilt(manifest):
    cached = WindowDataset(manifest, "train", window=4, cache_windows=True)
    directory = (
        manifest.parent / ".ghostmesh-cache" / cached.storage_info()["cache_key"]
    )
    marker = directory / "metadata.sha256"
    marker.unlink()  # Test-owned generated completion marker only.
    with pytest.raises(ValueError, match="incomplete or corrupt"):
        WindowDataset(manifest, "train", window=4, cache_windows=True)
    assert not marker.exists()
