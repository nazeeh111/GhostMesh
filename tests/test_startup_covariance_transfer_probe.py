"""SOURCE-only controls, all-person comparator joins and pre-score freeze tests."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import domain_startup_covariance_transfer_probe as study
from scripts.domain_covariance_source_probe import fold_masks


def test_source_priors_use_only_training_labels_and_clock_is_separate():
    rng = np.random.default_rng(141)
    target = rng.normal(size=(8, 17, 2))
    train = np.arange(8) < 4
    endpoint = np.array([39, 39, 47, 47, 39, 47, 55, 55])
    query = np.arange(4, 8)
    first, missing = study.source_priors(target, endpoint, train, query)
    np.testing.assert_array_equal(first["source_body_mean"][0], target[:4].mean(0))
    np.testing.assert_array_equal(first["source_endpoint_clock"][0], target[:2].mean(0))
    assert missing == [55]
    target[~train] = np.nan
    actual, missing2 = study.source_priors(target, endpoint, train, query)
    for key in first:
        np.testing.assert_array_equal(first[key], actual[key])
    assert missing2 == missing


def test_frozen_rff_application_has_no_target_stats_or_batch_dependence():
    rng = np.random.default_rng(142)
    x = rng.normal(size=(5, 6))
    state = dict(
        feature_mean=rng.normal(size=6),
        feature_scale=rng.uniform(0.5, 2, size=6),
        weight=rng.normal(size=(6, 12)),
        offset=rng.normal(size=12),
        rff_mean=rng.normal(size=12),
        rff_scale=rng.uniform(0.5, 2, size=12),
    )
    first = study.transform_frozen(x, state)
    direct = np.sqrt(2 / 12) * np.cos(
        np.einsum(
            "ni,ij->nj",
            (x - state["feature_mean"]) / state["feature_scale"],
            state["weight"],
        )
        + state["offset"]
    )
    direct = (direct - state["rff_mean"]) / state["rff_scale"]
    np.testing.assert_allclose(first, direct, atol=1e-15)
    np.testing.assert_allclose(
        study.transform_frozen(x[:1], state), first[:1], atol=1e-15
    )
    x[1:] += 1e8
    np.testing.assert_array_equal(study.transform_frozen(x, state)[:1], first[:1])


def test_full_gate_requires_every_control_clock_pck_motion_and_shuffle():
    def row(nje, r2=0.1):
        return dict(
            aligned=dict(
                metrics=dict(
                    normalized_joint_error=nje, pck_curve={"0.05": 0.7, "0.2": 0.9}
                ),
                record_articulation_r2=r2,
            ),
            shuffled=dict(metrics=dict(normalized_joint_error=0.4)),
        )

    names = [
        "startup_shape",
        "startup_shape_volume",
        "source_body_mean",
        "source_endpoint_clock",
        "raw_current",
        "startup_frozen_raw",
        "frozen_temporal",
    ]
    good = {
        room: {name: row(0.1 if name == "startup_shape" else 0.2) for name in names}
        for room in ("E01", "E02", "E03")
    }
    assert study.frozen_gate(good)["passed"]
    for control in (
        "source_body_mean",
        "source_endpoint_clock",
        "raw_current",
        "startup_frozen_raw",
    ):
        bad = copy.deepcopy(good)
        bad["E02"][control]["aligned"]["metrics"]["normalized_joint_error"] = 0.09
        assert not study.frozen_gate(bad)["passed"]
    bad = copy.deepcopy(good)
    bad["E01"]["startup_shape"]["aligned"]["record_articulation_r2"] = -0.01
    assert not study.frozen_gate(bad)["passed"]
    bad = copy.deepcopy(good)
    bad["E03"]["startup_shape"]["shuffled"]["metrics"]["normalized_joint_error"] = 0.09
    assert not study.frozen_gate(bad)["passed"]
    bad = copy.deepcopy(good)
    bad["E02"]["startup_shape"]["aligned"]["metrics"]["pck_curve"]["0.05"] = 0.6
    assert not study.frozen_gate(bad)["passed"]
    # A volume success can never replace the declared shape primary.
    assert study.CONFIG["primary"] == "startup_shape"


def test_source_readout_is_unchanged_when_held_teachers_are_nonfinite():
    rng = np.random.default_rng(143)
    z = rng.normal(size=(24, 9))
    y = rng.normal(size=(24, 17, 2))
    train = np.arange(24) < 14
    val = (np.arange(24) >= 14) & (np.arange(24) < 20)
    rooms = np.array(["E01"] * 12 + ["E03"] * 12)
    first, grid = study.fit_readout(z, y, train, val, rooms)
    y[20:] = np.nan
    other, grid2 = study.fit_readout(z, y, train, val, rooms)
    for key in first:
        np.testing.assert_array_equal(first[key], other[key])
    assert grid == grid2


def cohort():
    rows = [
        (f"E{room:02}", f"S{person:02}", f"A{action:02}")
        for room in range(1, 4)
        for person in range((room - 1) * 10 + 1, room * 10 + 1)
        for action in range(1, 28)
    ]
    return {
        key: np.asarray([row[j] for row in rows])
        for j, key in enumerate(("room", "person", "action"))
    } | dict(
        record=np.asarray(["/".join(row) for row in rows]),
        end=np.repeat(39, len(rows)),
        pose=np.zeros((len(rows), 17, 2)),
    )


def make_frozen(tmp_path, data):
    old = tmp_path / "old"
    raw = tmp_path / "raw"
    old.mkdir()
    raw.mkdir()
    (old / "verification.json").write_text(
        json.dumps(
            dict(total_records_feature_verified=810, retained_examples_verified=26556)
        )
    )
    (raw / "verification.json").write_text(
        json.dumps(
            dict(
                feature_windows_verified=26556,
                all_current_reference_features_and_person_disjoint_queries_verified=True,
            )
        )
    )
    roles = {}
    for room in ("E01", "E02", "E03"):
        train, val, held = fold_masks(
            data["room"],
            data["person"],
            room,
            dict(train=432, validation=108, held=270),
        )
        roles[room + "_train"] = np.flatnonzero(train)
        roles[room + "_validation"] = np.flatnonzero(val)
        ids = np.flatnonzero(held)
        keep = ids[~np.isin(data["person"][ids], sorted(set(data["person"][ids]))[:2])]
        np.savez(
            old / (room + "-congruence_dynamics-model-prediction.npz"),
            held_indices=ids,
            pose=np.zeros((len(ids), 17, 2)),
        )
        np.savez(
            raw / (room + "-raw_current_control-model.npz"),
            indices=keep,
            prediction=np.zeros((len(keep), 17, 2)),
            weights=np.zeros((5, 512, 34)),
            selected=np.asarray(0),
            x_mean=np.zeros(512),
            y_mean=np.zeros(34),
        )
        np.savez(
            raw / (room + "-raw_current_control-rff.npz"),
            feature_mean=np.zeros(1368),
            feature_scale=np.ones(1368),
            weight=np.zeros((1368, 512)),
            offset=np.zeros(512),
            rff_mean=np.zeros(512),
            rff_scale=np.ones(512),
        )
    np.savez(raw / "cohort-indices.npz", **roles)
    for folder in (old, raw):
        (folder / "registration.json").write_text("{}\n")
        (folder / "results.json").write_text("{}\n")
        (folder / "artifact_manifest.json").write_text(
            json.dumps({p.name: study.digest(p) for p in folder.iterdir()})
        )
    return old, raw


def test_frozen_controls_include_all_ten_people_and_reject_row_join_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        study,
        "fold_masks",
        lambda room, person, held: fold_masks(
            room, person, held, dict(train=432, validation=108, held=270)
        ),
    )
    data = cohort()
    old, raw = make_frozen(tmp_path, data)
    models, hashes, roles = study.load_frozen_comparators(data, old, raw)
    assert len(models) == 3 and hashes["raw_verification_sha256"] == study.digest(
        raw / "verification.json"
    )
    for info in roles.values():
        assert (
            len(info["held_people"]) == 10
            and info["held"] == 270
            and info["raw_control_old_saved_rows"] == 216
        )
    path = old / "E02-congruence_dynamics-model-prediction.npz"
    with np.load(path) as z:
        ids = z["held_indices"].copy()
    np.savez(path, held_indices=ids[::-1], pose=np.zeros((270, 17, 2)))
    manifest = json.loads((old / "artifact_manifest.json").read_text())
    manifest[path.name] = study.digest(path)
    (old / "artifact_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rows differ"):
        study.load_frozen_comparators(data, old, raw)


def test_permutation_keeps_every_record_and_is_deterministic():
    record = np.repeat(["a", "b", "c"], [5, 7, 9])
    p = study.permutation(record)
    np.testing.assert_array_equal(p, study.permutation(record))
    np.testing.assert_array_equal(record[p], record)
    np.testing.assert_array_equal(np.sort(p), np.arange(len(record)))


def test_existing_output_refused_without_accessing_inputs(tmp_path):
    with pytest.raises(FileExistsError):
        study.run(None, None, None, None, tmp_path)


def test_runtime_all_nine_source_fits_finish_before_any_held_score(
    tmp_path, monkeypatch
):
    data = cohort()
    monkeypatch.setattr(
        study,
        "fold_masks",
        lambda room, person, held: fold_masks(
            room, person, held, dict(train=432, validation=108, held=270)
        ),
    )
    n = len(data["pose"])
    cache = tmp_path / "cache"
    local = tmp_path / "local"
    cache.mkdir()
    local.mkdir()
    for path in [
        cache / "receipt.json",
        cache / "targets-and-metadata.npz",
        local / "results.json",
        local / "verification.json",
        local / "artifact-manifest-final.json",
    ]:
        path.write_text("{}\n")
    rawstate = dict(
        indices=np.flatnonzero(data["room"] == "E01"),
        prediction=np.zeros((270, 17, 2)),
        selected=np.asarray(0),
    )
    comparators = {
        room: dict(
            state={**rawstate, "indices": np.flatnonzero(data["room"] == room)},
            mapping={},
            temporal_prediction=np.zeros((270, 17, 2)),
        )
        for room in ("E01", "E02", "E03")
    }
    monkeypatch.setattr(
        study,
        "input_setup",
        lambda *args: (
            {"arrays_sha256": {}},
            data,
            np.zeros((n, 1)),
            comparators,
            {},
            {},
        ),
    )
    reference = SimpleNamespace(
        anchors=np.zeros((1, 19, 3, 3)), mean_log_volume=np.zeros(19)
    )
    bank = {
        name: SimpleNamespace(end=39, reference=reference, raw_current=np.zeros(1368))
        for name in data["record"]
    }
    monkeypatch.setattr(study, "fit_startups", lambda *args: bank)
    monkeypatch.setattr(
        study,
        "transform_startups",
        lambda x, *args: {
            arm: np.zeros((len(x), width))
            for arm, width in study.CONFIG["dimensions"].items()
        },
    )
    monkeypatch.setattr(
        study, "rff_transform", lambda x, *args: np.zeros((len(x), 512))
    )
    called = []

    def fit(*args):
        called.append(True)
        return {"selected": np.asarray(0)}, [
            {"alpha": 0.001, "source_validation_nje": 0.1}
        ]

    monkeypatch.setattr(study, "fit_readout", fit)
    monkeypatch.setattr(
        study, "transform_frozen", lambda x, *args: np.zeros((len(x), 512))
    )
    monkeypatch.setattr(study, "predict", lambda x, *args: np.zeros((len(x), 17, 2)))
    out = tmp_path / "out"

    def scored(*args):
        assert len(called) == 9
        selections = json.loads((out / "ALL-FOLDS-FROZEN-SELECTIONS.json").read_text())
        assert set(selections) == {"E01", "E02", "E03"}
        assert all(
            set(v) == set(study.CONFIG["fit_arms"]) | {"raw_current"}
            for v in selections.values()
        )
        return dict(
            aligned=dict(
                metrics=dict(
                    normalized_joint_error=0.2, pck_curve={"0.05": 0.7, "0.2": 0.9}
                ),
                record_articulation_r2=0.0,
            ),
            shuffled=dict(metrics=dict(normalized_joint_error=0.2)),
        )

    monkeypatch.setattr(study, "scores", scored)
    study.run(cache, None, None, local, out)
    assert not json.loads((out / "results.json").read_text())["gate"]["passed"]
