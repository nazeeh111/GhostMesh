"""Analytic local-audit checks independent of fitted experiment data."""

import numpy as np
import pytest
from scripts import verify_startup_covariance_probe as audit


def diagonal_episode():
    d = np.array([1.0, 2.0, 5.0])[None, None, None, :] * np.exp(
        np.arange(3)[:, None, None, None]
        * np.array([-0.2, 0.1, 0.3])[None, None, None, :]
    )
    d = np.broadcast_to(d, (3, 8, 19, 3)).copy()
    current = np.zeros((3, 8, 19, 3, 3), complex)
    current[..., (0, 1, 2), (0, 1, 2)] = d
    coords = np.zeros((3, 8, 19, 9))
    coords[..., :3] = d
    return current, coords, d


def test_scipy_generalized_spectrum_matches_diagonal_closed_form_and_prefix():
    current, coords, d = diagonal_episode()
    features, anchor, volume = audit.independent_features(current, coords, 0)
    shape = d / np.exp(np.log(d).mean(-1))[..., None]
    b = shape[0].mean(0)
    b /= np.exp(np.log(b).mean(-1))[..., None]
    midpoint = (shape + b) / 2
    expected = np.sort(
        np.log((shape + 1e-4 * midpoint) / (b + 1e-4 * midpoint)), axis=-1
    )
    expected -= expected.mean(-1, keepdims=True)
    np.testing.assert_allclose(
        features["startup_shape"], expected.reshape(3, 456), atol=1e-15
    )
    np.testing.assert_allclose(
        np.diagonal(anchor, axis1=-2, axis2=-1), shape[0].mean(0), atol=1e-15
    )
    np.testing.assert_allclose(volume, np.log(d[0]).mean(-1).mean(0), atol=1e-15)
    prefix, prefix_anchor, _ = audit.independent_features(current[:2], coords[:2], 0)
    for arm in features:
        np.testing.assert_array_equal(prefix[arm], features[arm][:2])
    np.testing.assert_array_equal(prefix_anchor, anchor)
    np.testing.assert_array_equal(
        features["startup_frozen_raw"][0], features["startup_frozen_raw"][2]
    )


def test_nonpositive_covariance_and_invalid_anchor_refused():
    current, coords, _ = diagonal_episode()
    with pytest.raises(ValueError, match="anchor"):
        audit.independent_features(current, coords, 3)
    current[1, 0, 0, 0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        audit.independent_features(current, coords, 0)


def test_metrics_use_reference_box_and_center_only_for_motion_diagnostic():
    rng = np.random.default_rng(19)
    target = rng.normal(size=(8, 17, 2))
    offset = np.array([3.0, -2.0])
    actual = audit.metrics(target + offset, target)
    expected_nje = np.mean(
        np.linalg.norm(offset) / np.linalg.norm(target.max(1) - target.min(1), axis=1)
    )
    assert actual["nje"] == pytest.approx(expected_nje)
    assert actual["r2"] == pytest.approx(1)
    assert actual["pck"]["0.05"] == 0
    with pytest.raises(ValueError, match="finite"):
        audit.metrics(target * np.nan, target)
    with pytest.raises(ValueError, match="Undefined"):
        audit.metrics(np.ones((3, 17, 2)), np.ones((3, 17, 2)))


def test_owned_output_and_dangling_symlink_refused_before_input_access(tmp_path):
    output = tmp_path / "owned.json"
    output.write_text("owned")
    with pytest.raises(FileExistsError):
        audit.verify(tmp_path / "missing", tmp_path / "missing2", output)
    assert output.read_text() == "owned"
    link = tmp_path / "dangling.json"
    link.symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileExistsError):
        audit.verify(tmp_path / "missing", tmp_path / "missing2", link)
    assert not (tmp_path / "missing-target").exists()


@pytest.fixture(scope="module")
def local_audit_inputs(tmp_path_factory):
    """Prepare a dataset-free saved-input audit up to the first readout.

    The real verify() entry point decodes the saved covariances and reconstructs
    anchors/features before reading the normalizer we deliberately corrupt.
    No fitted readout is needed because rejection must precede model access.
    """
    import json
    import shutil

    root = tmp_path_factory.mktemp("local-audit-normalizers")
    base, pre = root / "experiment", root / "predecessor"
    base.mkdir()
    pre.mkdir()
    n = 164
    endpoints = np.concatenate(
        [np.arange(39, 118), np.arange(165, 178), np.arange(225, 297)]
    )
    split = np.array(["train"] * 79 + ["validation"] * 13 + ["test"] * 72)
    rng = np.random.default_rng(120)
    teacher = rng.normal(size=(n, 17, 2))
    target = teacher - teacher[:, [11, 12]].mean(1, keepdims=True)
    record = np.repeat("E01/S01/A12", n)
    raw_support = endpoints[:, None] - np.arange(39, -1, -1)[None]
    np.savez(
        pre / "indices.npz",
        split=split,
        target=teacher,
        records=record,
        end=endpoints,
        raw_support=raw_support,
    )
    diagonal = np.exp(rng.normal(0, 0.12, (n, 8, 19, 3)))
    current = np.zeros((n, 8, 19, 3, 3), complex)
    current[..., (0, 1, 2), (0, 1, 2)] = diagonal
    coordinates = np.zeros((n, 8, 19, 9))
    coordinates[..., :3] = diagonal
    untreated = np.zeros((n, 8, 19, 7, 9))
    untreated[:, :, :, 0] = coordinates
    np.save(pre / "untreated_covariance-features.npy", untreated.reshape(n, -1))
    source_files = [
        "scripts/domain_startup_covariance_probe.py",
        "ghostmesh/reference_covariance.py",
        "ghostmesh/covariance_dynamics.py",
        "ghostmesh/metrics.py",
        "scripts/domain_background_probe.py",
    ]
    source_hashes = {name: audit.sha(audit.ROOT / name) for name in source_files}
    for name in source_files:
        destination = base / "executed-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audit.ROOT / name, destination)
    consumed = {
        name: audit.sha(pre / name)
        for name in ["indices.npz", "untreated_covariance-features.npy"]
    }
    (pre / "artifact-manifest-final.json").write_text(json.dumps({"sha256": consumed}))
    registration = dict(
        full_recipe=audit.FULL_CONFIG,
        local_recipe=audit.LOCAL_CONFIG,
        source_sha256=source_hashes,
        consumed_predecessor_sha256=consumed,
        predecessor_manifest_sha256=audit.sha(pre / "artifact-manifest-final.json"),
    )
    (base / "registration.json").write_text(json.dumps(registration))
    np.savez(base / "targets-and-rows.npz", articulation_target=target)
    (base / "startup-identities.json").write_text(
        json.dumps(
            {
                "continuous": {
                    "E01/S01/A12": {
                        "current_state_ends": list(range(32, 40)),
                        "conservative_validity_support": list(range(40)),
                    }
                }
            }
        )
    )
    (base / "FROZEN-LOCAL-SELECTIONS.json").write_text("{}")
    (base / "results.json").write_text("{}")
    np.save(base / "test-permutation.npy", np.random.default_rng(90210).permutation(72))
    folder = base / "continuous"
    folder.mkdir()
    feature, anchor, volume = audit.independent_features(current, coordinates, 0)
    np.savez(
        folder / "E01-S01-A12-startup.npz",
        anchors=anchor[None],
        mean_log_volume=volume,
        raw_current=coordinates[0].reshape(-1),
    )
    x = feature["startup_shape"]
    np.save(folder / "startup_shape-features.npy", x)
    mean, scale = x[:79].mean(0), np.maximum(x[:79].std(0), 0.05)
    z = (x - mean) / scale
    cross = audit.mm(z, z[:79].T)
    distance = np.maximum(
        np.square(z).sum(1)[:, None] + np.square(z[:79]).sum(1)[None] - 2 * cross, 0
    )
    width = np.median(distance[:79][np.triu_indices(79, 1)])
    normalizer = folder / "startup_shape-normalization.npz"
    np.savez(
        normalizer,
        mean=mean,
        scale=scale,
        bandwidth_squared=np.asarray(width),
        training_design=z[:79],
    )
    # A direct first readout lets saved-prediction and metric attacks reach
    # their actual verification sites, without fitting the remaining arms.
    from scipy.linalg import solve

    kernel = cross / x.shape[1]
    y = target[:79].reshape(79, 34)
    ymean = y.mean(0)
    weights, grid = [], []
    for alpha in audit.LOCAL_CONFIG["lambdas"]:
        weight = solve(kernel[:79] + alpha * np.eye(79), y - ymean, assume_a="pos")
        weights.append(weight)
        candidate = (audit.mm(kernel, weight) + ymean).reshape(n, 17, 2)
        grid.append(
            dict(validation_nje=audit.metrics(candidate[79:92], target[79:92])["nje"])
        )
    selected = min(range(4), key=lambda i: grid[i]["validation_nje"])
    np.savez(
        folder / "startup_shape-linear-model.npz",
        weights=np.asarray(weights),
        target_mean=ymean,
        selected=np.asarray(selected),
    )
    prediction = (audit.mm(kernel, weights[selected]) + ymean).reshape(n, 17, 2)
    np.savez(
        base / "continuous-startup_shape-linear-predictions.npz", prediction=prediction
    )
    key = "continuous/startup_shape/linear"
    (base / "FROZEN-LOCAL-SELECTIONS.json").write_text(
        json.dumps(dict(candidates={key: grid}))
    )

    def score(p, t):
        value = audit.metrics(p, t)
        return dict(
            metrics=dict(normalized_joint_error=value["nje"], pck_curve=value["pck"]),
            record_articulation_r2=value["r2"],
        )

    scores = {
        role: score(prediction[mask], target[mask])
        for role, mask in [
            ("train", split == "train"),
            ("validation", split == "validation"),
            ("test", split == "test"),
        ]
    }
    scores["shuffle"] = score(
        prediction[92:][np.random.default_rng(90210).permutation(72)], target[92:]
    )
    (base / "results.json").write_text(json.dumps(dict(results={key: scores})))
    return base, pre, normalizer, (feature, anchor, volume)


def cache_fixture_features(monkeypatch, fixture):
    # The readback tests exercise verify(), its file reads, state checks and
    # first real direct readout. Reuse this fixture's independently rebuilt
    # features so each deliberate saved-state corruption does not repeat the
    # unrelated 25,000 generalized eigensolves. Analytic tests above and real
    # dataset audit replays separately exercise reconstruction itself.
    cached = fixture[3]
    monkeypatch.setattr(audit, "independent_features", lambda *_: cached)


@pytest.mark.parametrize(
    "field", ["mean", "scale", "training_design", "bandwidth_squared"]
)
@pytest.mark.parametrize("corruption", ["offset", "nan"])
def test_verify_rejects_corrupted_saved_normalizer_before_readout(
    local_audit_inputs, tmp_path, monkeypatch, field, corruption
):
    base, pre, normalization_path, _ = local_audit_inputs
    cache_fixture_features(monkeypatch, local_audit_inputs)
    original_load = np.load

    class CorruptedArchive:
        def __enter__(self):
            with original_load(normalization_path, allow_pickle=False) as archive:
                self.state = {name: archive[name] for name in archive.files}
            if corruption == "offset":
                self.state[field] = self.state[field] + 1e6
            else:
                self.state[field] = np.full_like(self.state[field], np.nan)
            return self.state

        def __exit__(self, *_):
            return False

    def intercepted_load(path, *args, **kwargs):
        if path == normalization_path:
            return CorruptedArchive()
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(audit.np, "load", intercepted_load)
    output = tmp_path / "verification.json"
    with pytest.raises(ValueError, match="Saved normalization mismatch: " + field):
        audit.verify(base, pre, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "kind", ["anchors", "mean_log_volume", "feature", "prediction"]
)
@pytest.mark.parametrize("corruption", ["nan", "offset"])
def test_verify_rejects_corrupted_saved_numeric_arrays(
    local_audit_inputs, tmp_path, monkeypatch, kind, corruption
):
    base, pre, _, _ = local_audit_inputs
    cache_fixture_features(monkeypatch, local_audit_inputs)
    original_load = np.load
    anchor_path = base / "continuous/E01-S01-A12-startup.npz"
    target_path = (
        base / "continuous/startup_shape-features.npy"
        if kind == "feature"
        else base / "continuous-startup_shape-linear-predictions.npz"
        if kind == "prediction"
        else anchor_path
    )

    def change(value):
        return np.full_like(value, np.nan) if corruption == "nan" else value + 1e6

    class CorruptedArchive:
        def __enter__(self):
            with original_load(target_path, allow_pickle=False) as archive:
                state = {name: archive[name] for name in archive.files}
            state[kind] = change(state[kind])
            return state

        def __exit__(self, *_):
            return False

    def intercepted_load(path, *args, **kwargs):
        if path == target_path:
            return (
                change(original_load(path, *args, **kwargs))
                if kind == "feature"
                else CorruptedArchive()
            )
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(audit.np, "load", intercepted_load)
    output = tmp_path / "verification.json"
    with pytest.raises(ValueError, match="Numeric readback mismatch"):
        audit.verify(base, pre, output)
    assert not output.exists()


@pytest.mark.parametrize("field", ["nje", "pck", "r2"])
@pytest.mark.parametrize("corruption", ["nan", "offset"])
def test_verify_rejects_corrupted_saved_metric_scalars(
    local_audit_inputs, tmp_path, monkeypatch, field, corruption
):
    base, pre, _, _ = local_audit_inputs
    cache_fixture_features(monkeypatch, local_audit_inputs)
    original_read = audit.read

    def intercepted_read(path):
        value = original_read(path)
        if path == base / "results.json":
            score = value["results"]["continuous/startup_shape/linear"]["train"]
            container, key = (
                (score["metrics"], "normalized_joint_error")
                if field == "nje"
                else (score["metrics"]["pck_curve"], "0.1")
                if field == "pck"
                else (score, "record_articulation_r2")
            )
            container[key] = (
                float("nan") if corruption == "nan" else container[key] + 1e6
            )
        return value

    monkeypatch.setattr(audit, "read", intercepted_read)
    output = tmp_path / "verification.json"
    with pytest.raises(ValueError, match="Numeric readback mismatch: metric"):
        audit.verify(base, pre, output)
    assert not output.exists()
