"""Independent root-readback software fixtures, never real pose efficacy evidence."""

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import domain_masked_fullpose_probe as fitter
from scripts import verify_masked_fullpose_probe as audit


def make_map(feature, train, dimensions=16):
    source = feature[train].astype(np.float64)
    mean, scale = source.mean(0), np.maximum(source.std(0), 0.001)
    ids = np.random.default_rng(2027).choice(np.flatnonzero(train), 128, replace=False)
    sample = (feature[ids] - mean) / scale
    distance = np.array(
        [
            np.square(sample[i] - sample[j]).sum()
            for i in range(128)
            for j in range(i + 1, 128)
        ]
    )
    width = np.median(distance)
    rng = np.random.default_rng(2027)
    weight = rng.normal(size=(feature.shape[1], dimensions)) / np.sqrt(width)
    offset = rng.uniform(0, 2 * np.pi, dimensions)
    phi = np.sqrt(2 / dimensions) * np.cos(
        np.einsum("ij,jk->ik", (feature - mean) / scale, weight, optimize=False)
        + offset
    )
    rm, rs = phi[train].mean(0), np.maximum(phi[train].std(0), 0.001)
    return dict(
        feature_mean=mean,
        feature_scale=scale,
        bandwidth_squared=np.asarray(width),
        bandwidth_sample_indices=ids,
        weight=weight,
        offset=offset,
        rff_mean=rm,
        rff_scale=rs,
    ), (phi - rm) / rs


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    base = tmp_path_factory.mktemp("full-root-independent")
    experiment, source = base / "root", base / "source"
    experiment.mkdir()
    source.mkdir()
    people, rooms, ends = [], [], []
    training_people = [
        f"S{i:02}"
        for i in range(1, 31)
        if i not in (4, 9, 14, 19, 24, 29) and not 11 <= i <= 20
    ]
    for person in range(1, 31):
        name = f"S{person:02}"
        room = f"E{(person - 1) // 10 + 1:02}"
        count = (
            2
            if room == "E02"
            else (
                6
                if name in audit.VALIDATION_PEOPLE
                else 8 + (name in training_people[:4])
            )
        )
        people.extend([name] * count)
        rooms.extend([room] * count)
        ends.extend(39 + 8 * np.arange(count))
    people, rooms, ends = np.array(people), np.array(rooms), np.array(ends)
    held = rooms == "E02"
    validation = ~held & np.isin(people, sorted(audit.VALIDATION_PEOPLE))
    roles = dict(train=~(held | validation), validation=validation, held=held)
    n = len(ends)
    rng = np.random.default_rng(194)
    truth_body = rng.normal(0, 0.1, (n, 17, 2))
    truth_body -= truth_body[:, [11, 12]].mean(1, keepdims=True)
    target = truth_body + (np.array([0.4, 0.5]) + rng.normal(0, 0.035, (n, 2)))[:, None]
    body = truth_body * 0.95
    data = dict(
        person=people,
        room=rooms,
        end=ends,
        record=np.array(
            [f"{room}/{person}/A01" for room, person in zip(rooms, people)]
        ),
        action=np.repeat("A01", n),
        pose=truth_body,
    )
    cached = np.zeros((n, 8, 19, 7, 9), np.float64)
    cached[:, :, :, 0] = rng.normal(size=(n, 8, 19, 9))
    designs, bodies, maps = {}, {}, {}
    for arm in audit.ARMS:
        feature = (
            cached[:, :, :, 0].reshape(n, 1368)
            if arm == "raw_current"
            else rng.normal(size=(n, 512)).astype(np.float32)
        )
        # Source mapping statistics use float64, as the frozen real producer does.
        feature64 = feature.astype(np.float64)
        mapping, z = make_map(feature64, roles["train"])
        maps[arm], designs[arm], bodies[arm] = mapping, z, body.copy()
        if arm != "raw_current":
            np.save(source / (arm + "-embedding.npy"), feature)
        np.savez(source / (arm + "-rff.npz"), **mapping)
        np.savez(source / (arm + "-readout.npz"), prediction=body)
    states, choices, predictions = fitter.fit_arrays(
        designs, bodies, target, data, **roles
    )
    for arm, state in states.items():
        np.savez(experiment / (arm + "-root-state.npz"), **state)
    for arm, prediction in predictions.items():
        np.save(experiment / (arm + "-full-prediction.npy"), prediction)
    (experiment / "frozen-fullpose-selections.json").write_text(json.dumps(choices))
    return dict(
        experiment=experiment,
        source=source,
        cached=cached.reshape(n, -1),
        data=data,
        target=target,
        roles=roles,
        designs=designs,
        bodies=bodies,
        states=states,
        choices=choices,
        maps=maps,
    )


def root_args(bundle, *, arm="masked_temporal"):
    b = bundle
    return [
        b["designs"][arm],
        b["bodies"][arm],
        b["target"],
        b["roles"],
        b["data"]["room"],
        copy.deepcopy(b["states"][arm]),
        copy.deepcopy(b["choices"][arm]),
    ]


def numerical_args(bundle):
    return [
        bundle[k] for k in ("experiment", "source", "cached", "data", "target", "roles")
    ]


def test_actual_fitter_artifacts_are_qualified_with_general_lu_without_fitter(
    bundle, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Independent audit called fitter/metric")

    for name in (
        "fit",
        "fit_root",
        "fit_arrays",
        "equal_room_nje",
        "predict_root",
        "scores",
        "score_frozen",
    ):
        monkeypatch.setattr(fitter, name, forbidden)
    proofs = audit.audit_numeric(*numerical_args(bundle), dimensions=16)
    assert set(proofs) == set(audit.ARMS)
    assert sum(p["candidate_equations_verified"] for p in proofs.values()) == 30
    assert (
        sum(p["source_validation_candidates_verified"] for p in proofs.values()) == 150
    )
    assert (
        max(p["maximum_general_lu_coefficient_difference"] for p in proofs.values())
        < 1e-12
    )
    assert max(p["maximum_full_prediction_difference"] for p in proofs.values()) < 1e-12


def test_actual_root_and_priors_do_not_score_or_depend_on_held_teachers(bundle):
    args = root_args(bundle)
    first = audit.audit_root(*args)
    args[2] = args[2].copy()
    args[2][bundle["roles"]["held"]] = np.nan
    second = audit.audit_root(*args)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    audit.audit_priors(bundle["experiment"], args[2], bundle["data"], bundle["roles"])


def test_verify_writes_a_finite_receipt_from_saved_fitter_artifacts_without_refitting(
    bundle, tmp_path, monkeypatch
):
    # A small synthetic cohort replaces only the expensive real-data loader;
    # all on-disk maps, states, choices, priors and prediction checks remain real.
    experiment, source = bundle["experiment"], bundle["source"]

    def seal(directory, filename):
        ready = dict(
            artifacts_sha256={
                path.name: audit.source_audit.digest(path)
                for path in directory.iterdir()
                if path.is_file() and path.name != filename
            }
        )
        (directory / filename).write_text(json.dumps(ready))
        return ready, audit.source_audit.digest(directory / filename)

    ready, expected_full = seal(experiment, "fullpose-source-ready.json")
    source_ready, expected_source = seal(source, "source-ready.json")

    def load(*args):
        assert args[-2:] == (expected_full, expected_source)
        return (
            bundle["data"],
            bundle["target"],
            bundle["roles"],
            bundle["cached"],
            ready,
            source_ready,
        )

    numeric = audit.audit_numeric
    monkeypatch.setattr(audit, "load_contract", load)
    monkeypatch.setattr(audit, "audit_numeric", lambda *a: numeric(*a, dimensions=16))
    for name in (
        "fit",
        "fit_root",
        "fit_arrays",
        "equal_room_nje",
        "predict_root",
        "scores",
        "score_frozen",
    ):
        monkeypatch.setattr(
            fitter, name, lambda *a, **k: pytest.fail("Fitter used during audit")
        )
    output = tmp_path / "audit.json"
    report = audit.verify(
        experiment,
        source,
        tmp_path,
        tmp_path,
        output,
        expected_fullpose_sha256=expected_full,
        expected_source_sha256=expected_source,
    )
    assert json.loads(output.read_text()) == report
    assert report["root_models_verified"] == 6
    assert report["root_candidate_equations_verified"] == 30
    assert report["source_validation_alpha_beta_candidates_verified"] == 150
    assert (
        not report["held_metrics_computed"] and not report["model_training_performed"]
    )
    assert report["maximum_target_join_difference"] == 0


@pytest.mark.parametrize(
    "field", ["x_mean", "root_mean", "weights", "beta", "selected"]
)
@pytest.mark.parametrize("damage", ["nan", "complex", "offset"])
def test_root_state_real_finite_and_selection_contract(bundle, field, damage):
    args = root_args(bundle)
    value = args[5][field]
    args[5][field] = (
        np.full_like(value, np.nan, dtype=float)
        if damage == "nan"
        else (value.astype(complex) + 1e6j if damage == "complex" else value + 1)
    )
    with pytest.raises(ValueError):
        audit.audit_root(*args)


@pytest.mark.parametrize(
    "field",
    ["alpha", "beta", "source_validation_nje", "normal_equation_relative_residual"],
)
def test_grid_rejects_wrong_or_nonfinite_source_selection(bundle, field):
    args = root_args(bundle)
    args[6]["grid"][0][field] = float("nan")
    with pytest.raises(ValueError):
        audit.audit_root(*args)


def test_chosen_alpha_beta_and_recorded_midpoint_are_independently_verified(bundle):
    for key in (
        "alpha",
        "beta",
        "source_validation_nje",
        "normal_equation_relative_residual",
    ):
        args = root_args(bundle)
        args[6]["choice"][key] += 0.1
        with pytest.raises(ValueError):
            audit.audit_root(*args)
    args = root_args(bundle)
    args[6]["maximum_body_hip_midpoint"] = np.nan
    with pytest.raises(ValueError):
        audit.audit_root(*args)
    args = root_args(bundle)
    args[1] = args[1] + 0.01
    with pytest.raises(ValueError, match="midpoint"):
        audit.audit_root(*args)


def test_static_root_uses_beta_zero_and_strongest_alpha_tie(bundle):
    args = root_args(bundle)
    target = args[1] + np.array([0.5, 0.5])[None, None]
    state, selection = fitter.fit_root(
        args[0], args[1], target, args[3]["train"], args[3]["validation"], args[4]
    )
    selection["maximum_body_hip_midpoint"] = float(
        np.max(np.abs(args[1][:, [11, 12]].mean(1)))
    )
    args[2], args[5], args[6] = target, state, selection
    _, proof = audit.audit_root(*args)
    assert proof["selected_alpha"] == 10 and proof["selected_beta"] == 0


@pytest.mark.parametrize("damage", ["nan", "complex", "offset"])
def test_saved_full_prediction_corruption_rejected(bundle, tmp_path, damage):
    import shutil

    experiment = tmp_path / "changed"
    shutil.copytree(bundle["experiment"], experiment)
    path = experiment / "random_frozen-full-prediction.npy"
    values = np.load(path)
    values = (
        values + 0.01
        if damage == "offset"
        else (
            values.astype(complex) + 1e6j
            if damage == "complex"
            else np.full_like(values, np.nan)
        )
    )
    np.save(path, values)
    args = numerical_args(bundle)
    args[0] = experiment
    with pytest.raises(ValueError):
        audit.audit_numeric(*args, dimensions=16)


@pytest.mark.parametrize("arm", ["source_mean", "source_endpoint_clock"])
@pytest.mark.parametrize("damage", ["nan", "complex", "offset"])
def test_static_prior_readback_is_real_finite_and_exact(bundle, tmp_path, arm, damage):
    import shutil

    experiment = tmp_path / "changed"
    shutil.copytree(bundle["experiment"], experiment)
    path = experiment / (arm + "-full-prediction.npy")
    values = np.load(path)
    values = (
        values + 0.01
        if damage == "offset"
        else (
            values.astype(complex) + 1e6j
            if damage == "complex"
            else np.full_like(values, np.nan)
        )
    )
    np.save(path, values)
    with pytest.raises(ValueError):
        audit.audit_priors(
            experiment, bundle["target"], bundle["data"], bundle["roles"]
        )


@pytest.mark.parametrize(
    "field", ["feature_mean", "feature_scale", "bandwidth_squared", "weight"]
)
def test_saved_source_map_normalization_corruption_rejected(bundle, tmp_path, field):
    import shutil

    source = tmp_path / "source"
    shutil.copytree(bundle["source"], source)
    path = source / "random_frozen-rff.npz"
    mapping = audit.npz(path)
    mapping[field] = np.full_like(mapping[field], np.nan)
    np.savez(path, **mapping)
    args = numerical_args(bundle)
    args[1] = source
    with pytest.raises(ValueError):
        audit.audit_numeric(*args, dimensions=16)


def test_roles_require_exact_source_person_partition(bundle):
    audit.validate_roles(bundle["data"], bundle["roles"])
    damaged = copy.deepcopy(bundle["data"])
    held = bundle["roles"]["held"]
    damaged["person"][np.flatnonzero(held)[0]] = damaged["person"][0]
    with pytest.raises(ValueError):
        audit.validate_roles(damaged, bundle["roles"])
    roles = copy.deepcopy(bundle["roles"])
    roles["train"][np.flatnonzero(held)[0]] = True
    with pytest.raises(ValueError):
        audit.validate_roles(bundle["data"], roles)


def test_root_numeric_boundary_refuses_a_held_row_in_training(bundle):
    args = root_args(bundle)
    args[3] = copy.deepcopy(args[3])
    args[3]["train"][np.flatnonzero(args[3]["held"])[0]] = True
    with pytest.raises(ValueError, match="SOURCE rooms"):
        audit.audit_root(*args)


def test_independent_full_target_join_catches_wrong_frames_validity_and_complex():
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(20, 17, 2)).astype(np.float32)
    validity = np.ones((20, 17), bool)
    records = [
        dict(id="E01/S01/A01", offset=0, frames=10),
        dict(id="E03/S21/A01", offset=10, frames=10),
    ]
    data = dict(record=np.array([r["id"] for r in records]), end=np.array([7, 8]))
    target = raw[[7, 18]].astype(float)
    data["pose"] = target - target[:, [11, 12]].mean(1, keepdims=True)
    actual, indices = audit.raw_target_join(raw, validity, records, data)
    np.testing.assert_array_equal(actual, target)
    np.testing.assert_array_equal(indices, [7, 18])
    damaged = copy.deepcopy(data)
    damaged["end"][0] = 8
    with pytest.raises(ValueError):
        audit.raw_target_join(raw, validity, records, damaged)
    with pytest.raises(ValueError):
        audit.raw_target_join(raw.astype(complex), validity, records, data)
    validity[7, 0] = False
    with pytest.raises(ValueError):
        audit.raw_target_join(raw, validity, records, data)


def test_both_external_ready_hashes_checked_before_any_array_read(
    tmp_path, monkeypatch
):
    experiment, source = tmp_path / "root", tmp_path / "source"
    experiment.mkdir()
    source.mkdir()
    root_ready, source_ready = (
        experiment / "fullpose-source-ready.json",
        source / "source-ready.json",
    )
    root_ready.write_text("{}")
    source_ready.write_text("{}")
    monkeypatch.setattr(
        audit.np,
        "load",
        lambda *a, **k: pytest.fail("Array read before ready identities"),
    )
    for expected_root, expected_source in [
        ("wrong", audit.source_audit.digest(source_ready)),
        (audit.source_audit.digest(root_ready), "wrong"),
    ]:
        with pytest.raises(ValueError, match="ready SHA"):
            audit.verify(
                experiment,
                source,
                tmp_path,
                tmp_path,
                tmp_path / "audit.json",
                expected_fullpose_sha256=expected_root,
                expected_source_sha256=expected_source,
            )


@pytest.mark.parametrize("damage", ["missing-root", "missing-encoder", "changed-state"])
def test_ready_required_artifacts_and_leaf_hashes_reject_before_array_reads(
    tmp_path, monkeypatch, damage
):
    experiment, source = tmp_path / "root", tmp_path / "source"
    experiment.mkdir()
    source.mkdir()
    names = {
        "registration.json",
        "roles.npz",
        "canonical-metadata.npz",
        "full-targets.npy",
        "raw-frame-indices.npy",
        "frozen-fullpose-selections.json",
    }
    names.update(arm + "-root-state.npz" for arm in audit.ARMS)
    names.update(
        arm + "-full-prediction.npy"
        for arm in (*audit.ARMS, "source_mean", "source_endpoint_clock")
    )
    for name in names:
        (experiment / name).write_bytes(b"opaque synthetic artifact")
    manifest = {name: audit.source_audit.digest(experiment / name) for name in names}
    if damage == "missing-root":
        del manifest["masked_temporal-root-state.npz"]
    root_ready = experiment / "fullpose-source-ready.json"
    root_ready.write_text(
        json.dumps(
            dict(
                kind="masked-csi-fullimage-source-ready",
                held_metrics_computed=False,
                artifacts_sha256=manifest,
            )
        )
    )
    encoder_ready = source / "source-ready.json"
    encoder_ready.write_text(
        json.dumps(
            dict(
                kind="masked-csi-source-only-ready",
                held_metrics_computed=False,
                artifacts_sha256={},
            )
        )
    )
    if damage == "changed-state":
        (experiment / "masked_temporal-root-state.npz").write_bytes(
            b"changed after ready seal"
        )
    monkeypatch.setattr(
        audit.np, "load", lambda *a, **k: pytest.fail("Unverified array read")
    )
    with pytest.raises(ValueError, match="(omits required|Hash mismatch)"):
        audit.load_contract(
            experiment,
            source,
            tmp_path,
            tmp_path,
            audit.source_audit.digest(root_ready),
            audit.source_audit.digest(encoder_ready),
        )


def source_provenance_fixture(tmp_path, *, registration_kind=None, damage=None):
    """Hash-valid opaque arrays reach provenance checks without real datasets."""
    import shutil

    experiment, source, raw, cache = [
        tmp_path / name for name in ("root", "encoder", "raw", "cache")
    ]
    for directory in (experiment, source, raw, cache):
        directory.mkdir()

    def write_json(path, value):
        path.write_text(json.dumps(value))

    def seal(directory, filename, **fields):
        manifest = {
            str(path.relative_to(directory)): audit.source_audit.digest(path)
            for path in directory.rglob("*")
            if path.is_file() and path.name != filename
        }
        path = directory / filename
        write_json(path, dict(fields, artifacts_sha256=manifest))
        return audit.source_audit.digest(path)

    records = [
        dict(id=f"E{room:02}/S{person:02}/A{action:02}", frames=297, offset=i * 297)
        for i, (room, person, action) in enumerate(
            (room, person, action)
            for room in range(1, 4)
            for person in range((room - 1) * 10 + 1, room * 10 + 1)
            for action in range(1, 28)
        )
    ]
    for name in ("rf", "pose", "valid"):
        (raw / (name + ".npy")).write_bytes(b"opaque raw array")
    raw_arrays = {
        name: audit.source_audit.digest(raw / (name + ".npy"))
        for name in ("rf", "pose", "valid")
    }
    write_json(raw / "receipt.json", dict(arrays=raw_arrays, records=records))
    for name in ("targets-and-metadata.npz", "untreated_covariance-features.npy"):
        (cache / name).write_bytes(b"opaque cache array")
    arrays = {path.name: audit.source_audit.digest(path) for path in cache.iterdir()}
    write_json(
        cache / "receipt.json",
        dict(
            kind="full-source-covariance-feature-cache",
            fits_performed=0,
            raw_receipt_sha256=audit.source_audit.digest(raw / "receipt.json"),
            raw_arrays_sha256=raw_arrays,
            arrays_sha256=arrays,
            records=records,
            logical_rows=26556,
            nominal_rows=26730,
        ),
    )
    write_json(
        cache / "artifact_manifest.json",
        {path.name: audit.source_audit.digest(path) for path in cache.iterdir()},
    )
    lineage = dict(
        raw_receipt_sha256=audit.source_audit.digest(raw / "receipt.json"),
        cache_receipt_sha256=audit.source_audit.digest(cache / "receipt.json"),
    )

    def sources(kind, directory, required):
        names = set(required)
        if registration_kind == kind:
            if damage == "unrelated-only":
                names = {"tests/test_masked_fullpose_verifier.py"}
            elif damage == "producer":
                names.remove(
                    "scripts/domain_masked_fullpose_probe.py"
                    if kind == "root"
                    else "scripts/domain_masked_pretrain_probe.py"
                )
            elif damage == "kernel":
                names.remove(
                    "ghostmesh/reference_covariance.py"
                    if kind == "root"
                    else "scripts/domain_reference_covariance_probe.py"
                )
        manifest = {
            name: audit.source_audit.digest(audit.ROOT / name) for name in names
        }
        for name in names:
            destination = directory / "executed-source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audit.ROOT / name, destination)
        return manifest

    encoder_names = {
        "roles.npz",
        "targets-and-metadata.npz",
        "initial.pt",
        "schedule.npy",
        "preoptimization-seal.json",
        "frozen-source-selections.json",
        "raw_current-rff.npz",
        "raw_current-readout.npz",
    }
    encoder_names.update(
        arm + suffix
        for arm in audit.ENCODERS
        for suffix in (
            "-embedding.npy",
            "-final.pt",
            "-rff.npz",
            "-readout.npz",
            "-training.json",
            "-complete.json",
        )
    )
    encoder_names.update(arm + "-optimizer.pt" for arm in audit.ENCODERS[1:])
    encoder_names.update(
        f"{arm}-step-{step:05}{suffix}"
        for arm in audit.ENCODERS[1:]
        for step in (1000, 2000, 3000, 4000)
        for suffix in (".pt", ".json")
    )
    for name in encoder_names:
        (source / name).write_bytes(b"opaque source array")
    (source / "targets-and-metadata.npz").write_bytes(
        (cache / "targets-and-metadata.npz").read_bytes()
    )
    write_json(
        source / "registration.json",
        dict(
            lineage,
            config=dict(held_room="E02", steps=4000, seed=2027),
            source_sha256=sources("encoder", source, audit.REQUIRED_ENCODER_SOURCES),
        ),
    )
    expected_source = seal(
        source,
        "source-ready.json",
        kind="masked-csi-source-only-ready",
        held_metrics_computed=False,
    )
    root_names = {
        "roles.npz",
        "canonical-metadata.npz",
        "full-targets.npy",
        "raw-frame-indices.npy",
        "frozen-fullpose-selections.json",
    }
    root_names.update(arm + "-root-state.npz" for arm in audit.ARMS)
    root_names.update(
        arm + "-full-prediction.npy"
        for arm in (*audit.ARMS, "source_mean", "source_endpoint_clock")
    )
    for name in root_names:
        (experiment / name).write_bytes(b"opaque root array")
    write_json(
        experiment / "registration.json",
        dict(
            lineage,
            config=dict(room="E02", alphas=audit.ALPHAS, betas=audit.BETAS),
            held_metrics_computed=False,
            source_ready_sha256=expected_source,
            source_sha256=sources("root", experiment, audit.REQUIRED_ROOT_SOURCES),
        ),
    )
    expected_root = seal(
        experiment,
        "fullpose-source-ready.json",
        kind="masked-csi-fullimage-source-ready",
        held_metrics_computed=False,
        source_ready_sha256=expected_source,
        **lineage,
    )
    return experiment, source, raw, cache, expected_root, expected_source


@pytest.mark.parametrize("registration_kind", ["root", "encoder"])
@pytest.mark.parametrize("damage", ["producer", "kernel", "unrelated-only"])
def test_resealed_missing_runtime_provenance_rejects_before_array_reads(
    tmp_path, monkeypatch, registration_kind, damage
):
    args = source_provenance_fixture(
        tmp_path, registration_kind=registration_kind, damage=damage
    )
    monkeypatch.setattr(
        audit.np,
        "load",
        lambda *a, **k: pytest.fail("Missing provenance reached array reader"),
    )
    with pytest.raises(ValueError, match="Missing required " + registration_kind):
        audit.load_contract(*args)


def test_complete_independent_source_sets_reach_array_reader(tmp_path, monkeypatch):
    args = source_provenance_fixture(tmp_path)

    def stop(*args, **kwargs):
        raise RuntimeError("All declared producer/runtime provenance verified")

    monkeypatch.setattr(audit.np, "load", stop)
    with pytest.raises(RuntimeError, match="All declared producer/runtime"):
        audit.load_contract(*args)


def test_existing_outputs_and_dangling_symlinks_refused_before_contract_access(
    tmp_path,
):
    output = tmp_path / "audit.json"
    output.write_text("owned")
    for path in (output, tmp_path / "dangling.json"):
        if path != output:
            path.symlink_to(tmp_path / "absent")
        with pytest.raises(ValueError, match="already exists"):
            audit.verify(
                tmp_path,
                tmp_path,
                tmp_path,
                tmp_path,
                path,
                expected_fullpose_sha256="unused",
                expected_source_sha256="unused",
            )
    assert output.read_text() == "owned" and not (tmp_path / "absent").exists()


def test_no_fitter_import_optimizer_or_held_metric_calls():
    tree = ast.parse(Path(audit.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not any(
                word in (node.module or "")
                for word in (
                    "domain_masked_fullpose_probe",
                    "domain_masked_pretrain_probe",
                )
            )
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
            )
            assert name not in {
                "fit_root",
                "fit_arrays",
                "optimizer_step",
                "scores",
                "score_frozen",
                "backward",
                "Adam",
                "AdamW",
            }
