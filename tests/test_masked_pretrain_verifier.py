"""Independent auditor tests; no real RF, learned fitting, or MPS calls."""

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import verify_masked_pretrain_probe as audit


def product(a, b):
    return np.einsum("ij,jk->ik", a, b, optimize=False)


def synthetic_map(feature, train, dimensions=16):
    source = feature[train].astype(np.float64)
    mean, scale = source.mean(0), np.maximum(source.std(0), 0.001)
    rng = np.random.default_rng(2027)
    ids = rng.choice(np.flatnonzero(train), 128, replace=False)
    sample = (feature[ids] - mean) / scale
    distance = ((sample[:, None] - sample[None]) ** 2).sum(-1)
    bandwidth = np.median(distance[np.triu_indices(128, 1)])
    rng = np.random.default_rng(2027)
    weight = rng.normal(size=(feature.shape[1], dimensions)) / np.sqrt(bandwidth)
    offset = rng.uniform(0, 2 * np.pi, dimensions)
    z = np.sqrt(2 / dimensions) * np.cos(
        product((feature - mean) / scale, weight) + offset
    )
    rm, rs = z[train].mean(0), np.maximum(z[train].std(0), 0.001)
    return dict(
        feature_mean=mean,
        feature_scale=scale,
        weight=weight,
        offset=offset,
        bandwidth_squared=np.asarray(bandwidth),
        bandwidth_sample_indices=ids,
        rff_mean=rm,
        rff_scale=rs,
    ), (z - rm) / rs


def synthetic_readout(z, pose, train, validation, rooms):
    xm, ym = z[train].mean(0), pose[train].reshape(-1, 34).mean(0)
    x, y = z[train] - xm, pose[train].reshape(-1, 34) - ym
    gram, right = product(x.T, x), product(x.T, y)
    weights, grid = [], []
    for alpha in audit.ALPHAS:
        weight = np.linalg.solve(gram + len(x) * alpha * np.eye(z.shape[1]), right)
        weights.append(weight)
        pred = (product(z[validation] - xm, weight) + ym).reshape(-1, 17, 2)
        target = pose[validation]
        diagonal = np.sqrt(((target.max(1) - target.min(1)) ** 2).sum(1))
        error = np.sqrt(((pred - target) ** 2).sum(2)) / diagonal[:, None]
        nje = np.mean(
            [error[rooms[validation] == room].mean() for room in ["E01", "E03"]]
        )
        grid.append(
            dict(
                alpha=alpha,
                source_validation_nje=float(nje),
                normal_equation_relative_residual=0.0,
            )
        )
    selected = int(np.argmin([row["source_validation_nje"] for row in grid]))
    state = dict(
        x_mean=xm,
        y_mean=ym,
        weights=np.stack(weights),
        selected=np.asarray(selected),
        prediction=(product(z - xm, weights[selected]) + ym).reshape(-1, 17, 2),
    )
    return state, dict(grid=grid, selected=selected)


@pytest.fixture
def toy():
    rng = np.random.default_rng(81)
    feature = rng.normal(size=(176, 11))
    pose = rng.normal(size=(176, 17, 2))
    train = np.arange(176) < 132
    validation = (np.arange(176) >= 132) & (np.arange(176) < 156)
    rooms = np.array(["E01", "E03"] * 88)
    rooms[156:] = "E02"
    mapping, z = synthetic_map(feature, train)
    state, selection = synthetic_readout(z, pose, train, validation, rooms)
    return feature, pose, train, validation, rooms, mapping, z, state, selection


def test_independent_rff_and_direct_readout_match_and_never_call_fitter(
    toy, monkeypatch
):
    from scripts import domain_masked_pretrain_probe as fitter

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Numerical audit must not invoke model fitting or held scoring"
        )

    for name in [
        "run",
        "optimizer_step",
        "fit_readout",
        "rff_transform",
        "score_frozen",
        "scores",
    ]:
        monkeypatch.setattr(fitter, name, forbidden)
    feature, pose, train, validation, rooms, mapping, expected_z, state, selection = toy
    z, errors = audit.independent_rff(feature, train, mapping, dimensions=16)
    np.testing.assert_allclose(z, expected_z, atol=1e-13)
    result = audit.independent_readout(
        z, pose[train], pose[validation], train, validation, rooms, state, selection
    )
    assert max(errors.values()) < 1e-12
    assert result["maximum_direct_coefficient_error"] < 1e-12
    assert result["maximum_prediction_error"] < 1e-12
    assert result["selected"] == selection["selected"]


@pytest.mark.parametrize(
    "field",
    [
        "feature_mean",
        "feature_scale",
        "weight",
        "offset",
        "bandwidth_squared",
        "rff_mean",
        "rff_scale",
    ],
)
def test_rff_nan_fields_cannot_hide_behind_max_error(toy, field):
    feature, _, train, _, _, mapping, *_ = toy
    damaged = copy.deepcopy(mapping)
    damaged[field] = np.full_like(damaged[field], np.nan)
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.independent_rff(feature, train, damaged, dimensions=16)


def test_rff_source_sample_mismatch_rejected_and_held_cannot_influence_map(toy):
    feature, _, train, _, _, mapping, *_ = toy
    poisoned = feature.copy()
    poisoned[156:] = poisoned[156:] * 1e4 + 1e6
    audit.independent_rff(poisoned, train, mapping, dimensions=16)
    damaged = copy.deepcopy(mapping)
    damaged["bandwidth_sample_indices"][0] = 175
    with pytest.raises(ValueError, match="bandwidth source sample"):
        audit.independent_rff(feature, train, damaged, dimensions=16)


@pytest.mark.parametrize("field", ["weights", "prediction", "x_mean", "y_mean"])
def test_readout_nonfinite_or_wrong_fields_rejected(toy, field):
    _, pose, train, validation, rooms, _, z, state, selection = toy
    damaged = copy.deepcopy(state)
    damaged[field] = np.full_like(damaged[field], np.nan)
    with pytest.raises(ValueError):
        audit.independent_readout(
            z,
            pose[train],
            pose[validation],
            train,
            validation,
            rooms,
            damaged,
            selection,
        )


def test_wrong_coefficients_and_source_selection_rejected(toy):
    _, pose, train, validation, rooms, _, z, state, selection = toy
    damaged = copy.deepcopy(state)
    damaged["weights"][1, 0, 0] += 0.01
    with pytest.raises(ValueError, match="ridge equation"):
        audit.independent_readout(
            z,
            pose[train],
            pose[validation],
            train,
            validation,
            rooms,
            damaged,
            selection,
        )
    bad_selection = copy.deepcopy(selection)
    bad_selection["selected"] = (selection["selected"] + 1) % len(audit.ALPHAS)
    with pytest.raises(ValueError, match="alpha selection"):
        audit.independent_readout(
            z,
            pose[train],
            pose[validation],
            train,
            validation,
            rooms,
            state,
            bad_selection,
        )
    bad_grid = copy.deepcopy(selection)
    bad_grid["grid"][0]["source_validation_nje"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.independent_readout(
            z, pose[train], pose[validation], train, validation, rooms, state, bad_grid
        )


def test_held_room_is_rejected_by_source_metric(toy):
    _, pose, _, validation, rooms, *_ = toy
    rooms = rooms.copy()
    rooms[validation] = "E02"
    with pytest.raises(ValueError, match="Held room"):
        audit.source_validation_nje(
            pose[validation], pose[validation], rooms[validation]
        )


def test_default_audit_refuses_partial_fit_before_final_artifact_reads(
    tmp_path, monkeypatch
):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "masked_only-final.pt").write_bytes(b"partial")
    output = tmp_path / "report.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("Partial learned artifacts must not be opened")

    monkeypatch.setattr(audit, "preoptimization", forbidden)
    with pytest.raises(ValueError, match="SOURCE_READY"):
        audit.verify(experiment, tmp_path, tmp_path, tmp_path, output)
    assert not output.exists()


def test_preoptimization_only_does_not_read_or_score_learned_artifacts(
    tmp_path, monkeypatch
):
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        audit,
        "preoptimization",
        lambda *args: ({}, {}, {}, [], None, dict(stage="preoptimization")),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Preoptimization audit must not read learned artifacts")

    monkeypatch.setattr(audit, "checkpoint_audit", forbidden)
    monkeypatch.setattr(audit, "replay_embeddings", forbidden)
    result = audit.verify(
        tmp_path, tmp_path, tmp_path, tmp_path, output, preoptimization_only=True
    )
    assert (
        result["held_metrics_computed"] is False
        and result["model_training_performed"] is False
    )
    assert json.loads(output.read_text())["stage"] == "preoptimization"
    with pytest.raises(ValueError, match="already exists"):
        audit.verify(
            tmp_path, tmp_path, tmp_path, tmp_path, output, preoptimization_only=True
        )


def test_seal_hash_and_required_artifact_tampering_refused(tmp_path):
    file = tmp_path / "model.pt"
    file.write_bytes(b"original")
    manifest = {file.name: audit.digest(file)}
    audit.verify_hashes(tmp_path, manifest)
    file.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Hash mismatch"):
        audit.verify_hashes(tmp_path, manifest)
    with pytest.raises(ValueError, match="Unsafe"):
        audit.verify_hashes(tmp_path, {"../outside": "x"})


def test_verifier_contains_no_fitter_import_or_optimizer_calls():
    source = Path(audit.__file__).read_text()
    tree = ast.parse(source)
    imports = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any("domain_masked_pretrain_probe" in name for name in imports)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"backward", "step", "Adam", "AdamW"}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {
                "fit_readout",
                "optimizer_step",
                "scores",
                "score_frozen",
            }


def role_bundle():
    records = [
        dict(id=f"E{room:02}/S{person:02}/A{action:02}", frames=297, offset=i * 297)
        for i, (room, person, action) in enumerate(
            (room, person, action)
            for room in range(1, 4)
            for person in range((room - 1) * 10 + 1, room * 10 + 1)
            for action in range(1, 28)
        )
    ]
    names = np.array([r["id"] for r in records])
    data = dict(
        record=names,
        room=np.array([x.split("/")[0] for x in names]),
        person=np.array([x.split("/")[1] for x in names]),
        action=np.array([x.split("/")[2] for x in names]),
        end=np.full(810, 39),
        pose=np.ones((810, 17, 2)),
        delta=np.zeros((810, 17, 2)),
    )
    held = data["room"] == "E02"
    validation = ~held & np.isin(data["person"], sorted(audit.VALIDATION_PEOPLE))
    roles = dict(train=~(held | validation), validation=validation, held=held)
    rng = np.random.default_rng(2027)
    groups = [
        np.flatnonzero(roles["train"] & (data["room"] == room))
        for room in ["E01", "E03"]
    ]
    schedule = np.stack(
        [
            rng.choice(groups[int(rng.integers(2))], 16, replace=False)
            for _ in range(4000)
        ]
    )
    reg = dict(
        config=dict(seed=2027, held_room="E02", steps=4000, batch=16),
        counts={key: int(mask.sum()) for key, mask in roles.items()},
        exposure=dict(
            optimizer_histories_per_trained_arm=64000,
            optimizer_frame_occurrences_per_trained_arm=512000,
            optimizer_unique_endpoint_rows=len(np.unique(schedule)),
            supervised_optimizer_endpoint_label_occurrences=64000,
            ssl_optimizer_pose_labels=0,
            frozen_probe_training_endpoint_labels=int(roles["train"].sum()),
        ),
    )
    seal = dict(
        roles={
            key: dict(
                people=sorted(set(data["person"][mask])),
                records=sorted(set(names[mask])),
                rows=int(mask.sum()),
            )
            for key, mask in roles.items()
        },
        optimizer_unique_raw_frames=len(np.unique(schedule)) * 8,
    )
    return data, records, reg, roles, schedule, seal


def test_exact_schedule_and_causal_role_boundary_and_exposure_tampering():
    data, records, reg, roles, schedule, seal = role_bundle()
    expected, offsets = audit.roles_and_schedule(
        data, records, reg, roles, schedule, seal
    )
    assert len(offsets) == 810 and expected["train"][schedule].all()
    bad_schedule = schedule.copy()
    bad_schedule[0, 0] = np.flatnonzero(roles["held"])[0]
    with pytest.raises(ValueError, match="non-TRAIN"):
        audit.roles_and_schedule(data, records, reg, roles, bad_schedule, seal)
    bad_reg = copy.deepcopy(reg)
    bad_reg["exposure"]["ssl_optimizer_pose_labels"] = 1
    with pytest.raises(ValueError, match="Exposure"):
        audit.roles_and_schedule(data, records, bad_reg, roles, schedule, seal)
    bad_data = copy.deepcopy(data)
    bad_data["end"][0] = 303
    with pytest.raises(ValueError, match="Noncanonical"):
        audit.roles_and_schedule(bad_data, records, reg, roles, schedule, seal)


def test_checkpoint_numeric_nan_rejected_without_mps(tmp_path):
    import torch

    path = tmp_path / "checkpoint.pt"
    torch.save(dict(model=dict(weight=torch.tensor([float("nan")]))), path)
    with pytest.raises(ValueError, match="Nonfinite/device tensor"):
        audit.load_torch(path)


@pytest.fixture
def checkpoint_bundle(tmp_path):
    import torch
    from types import SimpleNamespace

    torch.set_num_threads(1)
    config = dict(
        steps=4000, batch=2, mask_seed=719, learning_rate=0.001, weight_decay=0.01
    )
    registration = dict(config=config, device="cpu")
    for name in ["registration.json", "schedule.npy", "initial.pt"]:
        (tmp_path / name).write_bytes(b"identity")
    identity = {
        key: audit.digest(tmp_path / name)
        for key, name in (
            ("registration", "registration.json"),
            ("schedule", "schedule.npy"),
            ("initial", "initial.pt"),
        )
    }
    module = SimpleNamespace(MaskedCSI=lambda: torch.nn.Linear(2, 2))
    model = module.MaskedCSI().state_dict()
    torch.save(model, tmp_path / "initial.pt")
    identity["initial"] = audit.digest(tmp_path / "initial.pt")
    generator = torch.Generator().manual_seed(config["mask_seed"])
    masks = {}
    for step in range(1, config["steps"] + 1):
        torch.rand(config["batch"] * 8, 95, generator=generator)
        if step % 1000 == 0:
            masks[step] = generator.get_state().clone()
    for arm_index, arm in enumerate(audit.ARMS):
        final_model = {key: value.clone() + arm_index for key, value in model.items()}
        final = tmp_path / (arm + "-final.pt")
        torch.save(final_model, final)
        (tmp_path / (arm + "-complete.json")).write_text(
            json.dumps(dict(identity=identity, model_sha256=audit.digest(final)))
        )
        logs = (
            []
            if arm == "random_frozen"
            else [dict(step=step, loss=1.0) for step in masks]
        )
        (tmp_path / (arm + "-training.json")).write_text(
            json.dumps(dict(steps=0 if arm == "random_frozen" else 4000, logs=logs))
        )
        if arm == "random_frozen":
            continue
        for step in masks:
            optimizer = dict(
                state={
                    0: dict(
                        step=torch.tensor(float(step)),
                        exp_avg=torch.full((2, 2), float(step)),
                    )
                },
                param_groups=[dict(lr=0.001, weight_decay=0.01)],
            )
            state = dict(
                identity=identity,
                arm=arm,
                step=step,
                device="cpu",
                model=final_model if step == 4000 else model,
                optimizer=optimizer,
                logs=logs[: step // 1000],
                mask_generator=masks[step],
                cpu_rng=torch.get_rng_state(),
                mps_rng=None,
            )
            path = tmp_path / f"{arm}-step-{step:05}.pt"
            torch.save(state, path)
            path.with_suffix(".json").write_text(
                json.dumps(
                    dict(
                        checkpoint_sha256=audit.digest(path),
                        identity=identity,
                        arm=arm,
                        step=step,
                    )
                )
            )
        torch.save(optimizer, tmp_path / (arm + "-optimizer.pt"))
    return tmp_path, registration, module


def reseal_checkpoint(path, state):
    import torch

    torch.save(state, path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    receipt.update(
        checkpoint_sha256=audit.digest(path), arm=state["arm"], step=state["step"]
    )
    path.with_suffix(".json").write_text(json.dumps(receipt))


def test_checkpoint_seals_final_states_and_independent_mask_consumption(
    checkpoint_bundle,
):
    import torch

    tmp_path, registration, module = checkpoint_bundle
    result = audit.checkpoint_audit(tmp_path, registration, module)
    assert result == dict(checkpoints=16, shared_mask_generator_states_exact=True)
    path = tmp_path / "masked_only-step-01000.pt"
    state = torch.load(path, weights_only=True)
    state["mask_generator"] = torch.Generator().manual_seed(2).get_state()
    reseal_checkpoint(path, state)
    with pytest.raises(ValueError, match="mask generator state"):
        audit.checkpoint_audit(tmp_path, registration, module)


def test_checkpoint_payload_cannot_relabel_all4000budget_files_as1000(
    checkpoint_bundle,
):
    import torch

    tmp_path, registration, module = checkpoint_bundle
    assert audit.checkpoint_audit(tmp_path, registration, module)["checkpoints"] == 16
    for arm in audit.ARMS[1:]:
        first = torch.load(tmp_path / f"{arm}-step-01000.pt", weights_only=True)
        for path in sorted(tmp_path.glob(arm + "-step-*.pt")):
            reseal_checkpoint(path, copy.deepcopy(first))
        (tmp_path / (arm + "-optimizer.pt")).unlink()
    with pytest.raises(ValueError, match="filename, payload and receipt"):
        audit.checkpoint_audit(tmp_path, registration, module)


@pytest.mark.parametrize("field", ["model", "optimizer", "logs"])
def test_final_checkpoint_comparisons_are_required(checkpoint_bundle, field):
    import torch

    tmp_path, registration, module = checkpoint_bundle
    path = tmp_path / "masked_only-step-04000.pt"
    state = torch.load(path, weights_only=True)
    if field == "model":
        state["model"]["weight"] += 1
    elif field == "optimizer":
        state["optimizer"]["state"][0]["exp_avg"] += 1
    else:
        state["logs"][-1]["loss"] += 1
    reseal_checkpoint(path, state)
    with pytest.raises(
        ValueError, match="final (checkpoint model|optimizer|training logs)"
    ):
        audit.checkpoint_audit(tmp_path, registration, module)


@pytest.mark.parametrize(
    "field",
    [
        "features",
        "feature_mean",
        "feature_scale",
        "weight",
        "offset",
        "bandwidth_squared",
        "bandwidth_sample_indices",
        "rff_mean",
        "rff_scale",
    ],
)
def test_rff_complex_features_and_saved_fields_refused_before_cast(toy, field):
    feature, _, train, _, _, mapping, *_ = toy
    if field == "features":
        feature = feature.astype(np.complex128) + 1e6j
    else:
        mapping = copy.deepcopy(mapping)
        mapping[field] = mapping[field].astype(np.complex128) + 1e6j
    with pytest.raises(ValueError, match="Complex artifact"):
        audit.independent_rff(feature, train, mapping, dimensions=16)


@pytest.mark.parametrize("field", ["weights", "prediction", "x_mean", "y_mean"])
def test_readout_complex_state_fields_refused(toy, field):
    _, pose, train, validation, rooms, _, z, state, selection = toy
    state = copy.deepcopy(state)
    state[field] = state[field].astype(np.complex128) + 1000j
    with pytest.raises(ValueError, match="Complex artifact"):
        audit.independent_readout(
            z, pose[train], pose[validation], train, validation, rooms, state, selection
        )


@pytest.mark.parametrize("field", ["model", "optimizer"])
def test_checkpoint_complex_model_or_optimizer_rejected_after_reseal(
    checkpoint_bundle, field
):
    import torch

    tmp_path, registration, module = checkpoint_bundle
    path = tmp_path / "masked_only-step-04000.pt"
    state = torch.load(path, weights_only=True)
    if field == "model":
        state["model"]["weight"] = state["model"]["weight"].to(torch.complex64) + 1000j
    else:
        state["optimizer"]["state"][0]["exp_avg"] = (
            state["optimizer"]["state"][0]["exp_avg"].to(torch.complex64) + 1000j
        )
    reseal_checkpoint(path, state)
    with pytest.raises(ValueError, match="Complex tensor"):
        audit.checkpoint_audit(tmp_path, registration, module)


@pytest.mark.parametrize("value", [1 + 2j, np.complex128(1 + 2j), np.array([1 + 2j])])
def test_complex_scalar_and_array_trees_rejected(value):
    with pytest.raises(ValueError, match="Complex artifact"):
        audit.tensor_tree({"field": value}, "tree")


def test_raw_pose_complex_rejected_before_endpoint_conversion(tmp_path):
    np.save(tmp_path / "pose.npy", np.ones((1, 17, 2), dtype=np.complex64) + 1000j)
    with pytest.raises(ValueError, match="Complex artifact: raw pose"):
        audit.check_targets(tmp_path, {}, np.array([0]))


def test_complex_matrix_product_refused_before_real_conversion():
    with pytest.raises(ValueError, match="Complex artifact"):
        audit.mm(np.ones((2, 2), dtype=np.complex128) + 1000j, np.eye(2))


def resealed_provenance_fixture(tmp_path, *, omitted_training=None, omitted_cache=None):
    experiment, raw, cache, old = [
        tmp_path / name for name in ("experiment", "raw", "cache", "old")
    ]
    for path in (experiment, raw, cache, old):
        path.mkdir()
    for directory, names in (
        (experiment, audit.REQUIRED_TRAINING_SOURCES),
        (cache, audit.REQUIRED_CACHE_SOURCES),
    ):
        for name in names:
            path = directory / "executed-source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source must never be parsed or executed\n")
    training_hashes = {
        name: audit.digest(experiment / "executed-source" / name)
        for name in audit.REQUIRED_TRAINING_SOURCES
    }
    cache_hashes = {
        name: audit.digest(cache / "executed-source" / name)
        for name in audit.REQUIRED_CACHE_SOURCES
    }
    if omitted_training == "unrelated-only":
        path = experiment / "executed-source/unrelated.txt"
        path.write_text("valid hash does not establish production provenance")
        training_hashes = {"unrelated.txt": audit.digest(path)}
    elif omitted_training is not None:
        del training_hashes[omitted_training]
    if omitted_cache == "unrelated-only":
        path = cache / "executed-source/unrelated.txt"
        path.write_text("valid hash does not establish cache producer provenance")
        cache_hashes = {"unrelated.txt": audit.digest(path)}
    elif omitted_cache is not None:
        del cache_hashes[omitted_cache]
    (raw / "receipt.json").write_text("{}")
    (cache / "receipt.json").write_text(json.dumps(dict(source_sha256=cache_hashes)))
    registration = dict(
        source_sha256=training_hashes,
        raw_receipt_sha256=audit.digest(raw / "receipt.json"),
        cache_receipt_sha256=audit.digest(cache / "receipt.json"),
    )
    (experiment / "registration.json").write_text(json.dumps(registration))
    (experiment / "preoptimization-seal.json").write_text(
        json.dumps(
            dict(
                files_sha256={
                    "registration.json": audit.digest(experiment / "registration.json")
                }
            )
        )
    )
    return experiment, raw, cache, old


def forbid_provenance_before_code_and_arrays(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Required source provenance must be checked before source code or arrays"
        )

    monkeypatch.setattr(audit, "check_source_stage_contract", forbidden)
    monkeypatch.setattr(audit, "load_model_source", forbidden)
    monkeypatch.setattr(np, "load", forbidden)


@pytest.mark.parametrize(
    "omitted",
    [
        "scripts/domain_masked_pretrain_probe.py",
        "ghostmesh/masked_csi.py",
        "scripts/domain_covariance_source_probe.py",
        "scripts/domain_reference_covariance_probe.py",
        "ghostmesh/metrics.py",
        "unrelated-only",
    ],
)
def test_actual_preoptimization_resealed_omitted_training_sources_rejected_before_code_arrays(
    tmp_path, monkeypatch, omitted
):
    paths = resealed_provenance_fixture(tmp_path, omitted_training=omitted)
    forbid_provenance_before_code_and_arrays(monkeypatch)
    with pytest.raises(
        ValueError, match="Training provenance omits required production sources"
    ):
        audit.preoptimization(*paths)


@pytest.mark.parametrize(
    "omitted",
    [
        "scripts/domain_covariance_full_cache.py",
        "scripts/domain_covariance_source_probe.py",
        "ghostmesh/covariance_dynamics.py",
        "unrelated-only",
    ],
)
def test_actual_preoptimization_resealed_omitted_cache_sources_rejected_before_code_arrays(
    tmp_path, monkeypatch, omitted
):
    paths = resealed_provenance_fixture(tmp_path, omitted_cache=omitted)
    forbid_provenance_before_code_and_arrays(monkeypatch)
    with pytest.raises(
        ValueError, match="Cache provenance omits required production sources"
    ):
        audit.preoptimization(*paths)
