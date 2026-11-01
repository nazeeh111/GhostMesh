"""Synthetic causal startup, matched controls and source-label isolation tests."""

import json

import numpy as np
import pytest

from ghostmesh.covariance_dynamics import determinant_shape, hermitian_coordinates
from scripts.domain_startup_covariance_probe import (
    LOCAL_CONFIG,
    articulation,
    fit_startups,
    local,
    local_design,
    local_fit,
    local_gate,
    transform_startups,
)


def matrices(n, seed=120):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8, 19, 3, 3)) + 1j * rng.normal(size=(n, 8, 19, 3, 3))
    return np.einsum("...ij,...kj->...ik", x, x.conj()) + np.eye(3)


def encode(current):
    value = np.zeros((len(current), 8, 19, 7, 9))
    value[:, :, :, 0] = hermitian_coordinates(current)
    return value.reshape(len(value), -1)


def test_anchor_earliest_available_eight_current_states_normalized_before_mean():
    current = matrices(3)
    bank = fit_startups(
        encode(current), np.repeat("session", 3), np.array([55, 47, 63])
    )
    anchor = bank["session"]
    assert anchor.end == 47 and anchor.reference.group_counts == (8,)
    np.testing.assert_allclose(
        anchor.reference.anchors[0], determinant_shape(current[1]).mean(0), atol=1e-14
    )
    assert not anchor.raw_current.flags.writeable
    assert not anchor.reference.anchors.flags.writeable
    # Averaging before normalizing is generally a different operation.
    assert not np.allclose(
        determinant_shape(current[1].mean(0)), determinant_shape(current[1]).mean(0)
    )


def test_future_changes_do_not_change_anchor_or_earlier_prediction_features():
    x = encode(matrices(4))
    record, ends = np.repeat("own", 4), np.array([39, 47, 55, 63])
    before = fit_startups(x, record, ends)
    changed = x.copy()
    changed[2:] = (
        np.nan
    )  # This data is in the future and must never be decoded for anchor fitting.
    after = fit_startups(changed, record, ends)
    np.testing.assert_array_equal(
        before["own"].reference.anchors, after["own"].reference.anchors
    )
    first = transform_startups(x[:2], record[:2], ends[:2], before)
    second = transform_startups(changed[:2], record[:2], ends[:2], after)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    with pytest.raises(ValueError, match="precedes"):
        transform_startups(
            x[:1],
            record[:1],
            np.array([39]),
            fit_startups(x[1:2], record[:1], np.array([47])),
        )


def test_record_reset_isolation_batch_independence_and_frozen_template_control():
    x = encode(matrices(4))
    record, ends = np.array(["a", "b", "a", "b"]), np.array([39, 47, 55, 63])
    bank = fit_startups(x, record, ends)
    all_rows = transform_startups(x, record, ends, bank)
    for i in range(4):
        one = transform_startups(x[i : i + 1], record[i : i + 1], ends[i : i + 1], bank)
        for key in one:
            np.testing.assert_allclose(one[key][0], all_rows[key][i], atol=0, rtol=0)
    np.testing.assert_array_equal(
        all_rows["startup_frozen_raw"][0], all_rows["startup_frozen_raw"][2]
    )
    changed = x.copy()
    changed[[1, 3]] = encode(matrices(2, seed=121))
    changed_bank = fit_startups(changed, record, ends)
    actual = transform_startups(changed, record, ends, changed_bank)
    for key in all_rows:
        np.testing.assert_array_equal(actual[key][[0, 2]], all_rows[key][[0, 2]])
    with pytest.raises(ValueError, match="Duplicate"):
        fit_startups(x[:2], np.repeat("a", 2), np.array([39, 39]))


def test_primary_shape_fixed_mixing_and_independent_gain_assumption_but_not_volume():
    rng = np.random.default_rng(123)
    current = matrices(3)
    g = rng.normal(size=(19, 3, 3)) * 0.1 + np.eye(3)
    gains = np.exp(rng.normal(size=(3, 8, 19)))
    changed = (
        np.einsum("bij,ntbjk,blk->ntbil", g, current, g.conj()) * gains[..., None, None]
    )
    r, e = np.repeat("a", 3), np.array([39, 47, 55])
    x, y = encode(current), encode(changed)
    original = transform_startups(x, r, e, fit_startups(x, r, e))
    transformed = transform_startups(y, r, e, fit_startups(y, r, e))
    np.testing.assert_allclose(
        original["startup_shape"], transformed["startup_shape"], atol=2e-13, rtol=0
    )
    assert not np.allclose(
        original["startup_shape_volume"], transformed["startup_shape_volume"]
    )
    assert original["startup_shape"].shape == (3, 456)
    assert original["startup_shape_volume"].shape == (3, 608)


def test_training_only_normalization_width_and_labels_do_not_read_held_targets():
    rng = np.random.default_rng(124)
    x, y = rng.normal(size=(14, 6)), rng.normal(size=(14, 17, 2))
    train, val = np.arange(14) < 8, (np.arange(14) >= 8) & (np.arange(14) < 11)
    kernels, norm, constant = local_design(x, train)
    assert not constant
    state, grid = local_fit(kernels["rbf"], y, train, val)
    x2, y2 = x.copy(), y.copy()
    x2[11:] += 1e6
    y2[11:] = np.nan  # Neither weights nor source validation may use held labels.
    again, other_norm, _ = local_design(x2, train)
    second, other_grid = local_fit(again["rbf"], y2, train, val)
    for key in norm:
        np.testing.assert_array_equal(norm[key], other_norm[key])
    for key in state:
        np.testing.assert_array_equal(state[key], second[key])
    assert grid == other_grid


def test_zero_rank_startup_control_has_no_fabricated_rbf_bandwidth():
    x = np.repeat(np.arange(4)[None].astype(float), 12, axis=0)
    train = np.arange(12) < 7
    kernels, normalization, constant = local_design(x, train)
    assert constant and kernels == {} and normalization["bandwidth_squared"] == 0
    q = np.random.default_rng(125).normal(size=(4, 17, 2))
    shifted = q + np.random.default_rng(126).normal(size=(4, 1, 2))
    np.testing.assert_allclose(articulation(q), articulation(shifted), atol=1e-15)


def test_local_continuous_anchor_primary_and_reset_stress_are_declared_not_swapped():
    assert LOCAL_CONFIG["primary_variant"] == "continuous"
    assert LOCAL_CONFIG["variants"] == ["continuous", "reset_stress"]
    assert "SHARE" in LOCAL_CONFIG["continuous_anchor"]

    def value(nje, r2):
        return dict(metrics=dict(normalized_joint_error=nje), record_articulation_r2=r2)

    primary = dict(test=value(0.04, 0.7), shuffle=value(0.07, 0.1))
    assert local_gate(primary, value(0.1, 0), value(0.1, 0))["passed"]
    primary["test"]["record_articulation_r2"] = None
    assert not local_gate(primary, value(0.1, 0), value(0.1, 0))["passed"]
    primary["test"]["record_articulation_r2"] = 0.7
    assert not local_gate(primary, value(0.01, 0), value(0.1, 0))["passed"]


def test_refuses_existing_output_before_any_cache_access(tmp_path):
    with pytest.raises(FileExistsError):
        local(tmp_path / "missing", tmp_path)


def test_synthetic_full_local_path_freezes_both_variants_and_preserves_support(
    tmp_path,
):
    from scripts.domain_background_probe import digest

    predecessor = tmp_path / "predecessor"
    predecessor.mkdir()
    split = np.array(["train"] * 79 + ["validation"] * 13 + ["test"] * 72)
    ends = np.r_[np.arange(39, 118), np.arange(165, 178), np.arange(225, 297)]
    rng = np.random.default_rng(127)
    target = rng.normal(size=(164, 17, 2))
    support = ends[:, None] - np.arange(39, -1, -1)[None]
    np.savez(
        predecessor / "indices.npz",
        split=split,
        end=ends,
        raw_support=support,
        target=target,
        records=np.repeat("E01/S01/A12", 164),
    )
    np.save(predecessor / "untreated_covariance-features.npy", encode(matrices(164)))
    (predecessor / "registration.json").write_text("{}\n")
    (predecessor / "verification.json").write_text(
        json.dumps(
            dict(
                raw_RF_and_target_bindings_verified=True,
                forty_frame_histories_disjoint_between_splits=True,
            )
        )
    )
    (predecessor / "artifact-manifest-final.json").write_text(
        json.dumps({p.name: digest(p) for p in predecessor.iterdir()})
    )
    out = tmp_path / "result"
    local(predecessor, out)
    choices = json.loads((out / "FROZEN-LOCAL-SELECTIONS.json").read_text())
    assert choices["primary_variant"] == "continuous"
    identities = json.loads((out / "startup-identities.json").read_text())
    assert identities["continuous"]["E01/S01/A12"]["end"] == 39
    assert identities["continuous"]["E01/S01/A12"][
        "conservative_validity_support"
    ] == list(range(40))
    assert [
        identities["reset_stress"][name]["end"]
        for name in ("train", "validation", "test")
    ] == [39, 165, 225]
    with np.load(out / "continuous-startup_frozen_raw-constant-predictions.npz") as z:
        np.testing.assert_array_equal(
            z["prediction"],
            np.broadcast_to(articulation(target[:79]).mean(0), target.shape),
        )
    results = json.loads((out / "results.json").read_text())
    assert (
        results["primary_continuation_passed"]
        == results["local_gates"]["continuous"]["passed"]
    )
