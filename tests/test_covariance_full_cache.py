"""Verified reuse boundaries and logical-prefix contracts for cache-only build."""

import json

import numpy as np
import pytest

from scripts import domain_covariance_full_cache as cache


def metadata():
    return {
        "pose": np.zeros((3, 17, 2)),
        "delta": np.zeros((3, 17, 2)),
        "record": np.asarray(["E01/S01/A03"] * 2 + ["E01/S01/A07"]),
        "room": np.asarray(["E01"] * 3),
        "person": np.asarray(["S01"] * 3),
        "action": np.asarray(["A03", "A03", "A07"]),
        "end": np.asarray([39, 55, 47]),
    }


def quality():
    return [
        {"record": "E01/S01/A03", "retained_endpoints": [39, 55]},
        {"record": "E01/S01/A07", "retained_endpoints": [47]},
    ]


def test_exact_record_cursors_come_from_labels_and_retained_endpoints():
    blocks = cache.record_blocks(
        metadata(), quality(), [q["record"] for q in quality()]
    )
    assert blocks["E01/S01/A03"][0] == slice(0, 2)
    assert blocks["E01/S01/A07"][0] == slice(2, 3)
    changed = metadata()
    changed["end"][1] = 47
    with pytest.raises(ValueError, match="cursor"):
        cache.record_blocks(changed, quality(), [q["record"] for q in quality()])
    changed = metadata()
    changed["record"][1] = "E01/S01/A07"
    with pytest.raises(ValueError, match="metadata"):
        cache.record_blocks(changed, quality(), [q["record"] for q in quality()])


def test_physical_tail_is_never_a_logical_feature_or_record(tmp_path):
    for arm in cache.ARMS:
        value = np.zeros(
            (5, cache.study.CONFIG["raw_dimensions"][arm]), dtype="float32"
        )
        value[3:] = np.nan
        np.save(tmp_path / (arm + "-features.npy"), value)
    result = cache.logical_features(tmp_path, 3, 5)
    assert all(
        value.shape[0] == 3 and np.isfinite(value).all() for value in result.values()
    )
    with pytest.raises(ValueError, match="logical"):
        cache.logical_features(tmp_path, 6, 5)
    with pytest.raises(ValueError, match="allocation"):
        cache.logical_features(tmp_path, 3, 6)


def test_changed_input_bytes_are_refused(tmp_path):
    source = tmp_path / "input.npy"
    source.write_bytes(b"original input")
    expected = {source.name: cache.study.digest(source)}
    cache.verify_hashes(tmp_path, expected)
    source.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="Input hash mismatch"):
        cache.verify_hashes(tmp_path, expected)
    with pytest.raises(ValueError, match="Unsafe"):
        cache.verify_hashes(tmp_path, {"../outside": "0"})


def test_unverified_interrupted_arrays_cannot_be_blindly_reused(tmp_path):
    for arm in cache.ARMS:
        np.save(tmp_path / (arm + "-features.npy"), np.ones((10, 1)))
    (tmp_path / "registration.json").write_text(json.dumps({"records": 810}))
    before = {p.name: cache.study.digest(p) for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="independently verified"):
        cache.load_verified_limited(tmp_path, {}, "unused")
    after = {p.name: cache.study.digest(p) for p in tmp_path.iterdir()}
    assert before == after


def test_unknown_or_duplicate_quality_rows_are_refused():
    q = quality()
    q[0]["retained_endpoints"] = [39, 39]
    with pytest.raises(ValueError, match="Invalid retained"):
        cache.record_blocks(metadata(), q, [r["record"] for r in q])
    with pytest.raises(ValueError, match="ordering"):
        cache.record_blocks(metadata(), quality(), ["E04/S31/A03"])
