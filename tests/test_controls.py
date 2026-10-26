"""SOFTWARE TEST ONLY: tiny invented labels/artifacts, never research results."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ghostmesh import controls as c


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def row(pose, mask, room):
    return {"pose": np.asarray(pose), "valid": np.asarray(mask), "environment": room}


def test_prior_masks_variance_and_unseen_room_fallback():
    training = [
        row([[0, 0], [2, 2]], [1, 1], "E01"),
        row([[2, 0], [999, 999]], [1, 0], "E01"),
        row([[4, 0], [4, 4]], [1, 1], "E02"),
    ]
    prior = c.fit_room_prior(training)
    assert prior["rooms"]["E01"]["mean"] == [[1, 0], [2, 2]]
    assert np.exp(2 * prior["rooms"]["E01"]["log_sigma"][0]) == pytest.approx(0.5)
    prediction, sigma = c.prior_predictions(prior, [{"environment": "E04"}])
    np.testing.assert_allclose(prediction[0], [[2, 0], [3, 3]])
    np.testing.assert_allclose(sigma[0], prior["global"]["log_sigma"])
    assert prior["global"]["joint_counts"] == [3, 2]


def test_no_heldout_or_action_fit_and_missing_joint_fallback():
    training = [
        row([[0, 0], [np.nan, np.nan]], [1, 0], "E01"),
        row([[4, 0], [4, 4]], [1, 1], "E02"),
    ]
    training[0]["action"] = "unused"
    fit = c.fit_room_prior(training)
    assert fit["rooms"]["E01"]["mean"][1] == [4, 4]
    training[0]["action"] = "different"
    assert c.fit_room_prior(training) == fit
    heldout = [{"environment": "E04", "pose": np.full((2, 2), -999)}]
    c.prior_predictions(fit, heldout)
    assert c.fit_room_prior(training) == fit


def test_prior_refuses_joint_without_any_training_labels():
    with pytest.raises(ValueError, match="global training label"):
        c.fit_room_prior([row([[0, 0], [0, 0]], [1, 0], "E01")])


def test_permutation_bijection_room_reproducibility_and_pairing():
    metadata = [
        {"environment": "E01" if i % 2 else "E02", "frame_index": i} for i in range(50)
    ]
    indices = c.within_room_permutation(metadata)
    assert sorted(indices) == list(range(50))
    assert np.array_equal(indices, c.within_room_permutation(metadata, 90210))
    assert not np.array_equal(indices, np.arange(50))
    assert all(
        metadata[i]["environment"] == metadata[j]["environment"]
        for i, j in enumerate(indices)
    )
    pose, sigma = np.arange(100).reshape(50, 1, 2), np.arange(50).reshape(50, 1)
    moved, uncertainty = c.permute_predictions(pose, sigma, indices)
    assert np.array_equal(moved[:, 0, 0], uncertainty[:, 0] * 2)
    np.testing.assert_array_equal(pose, np.arange(100).reshape(50, 1, 2))
    with pytest.raises(ValueError, match="bijection"):
        c.permute_predictions(pose, sigma, np.zeros(50, int))


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "SOFTWARE_TEST_ONLY"
    source = Path(c.__file__).resolve().parents[1]
    for relative in c.FILES + ("scripts/run_suite.py",):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / relative).read_bytes())
    suite = root / "suite"
    manifests = {p: root / f"{p}.json" for p in ("cross_subject", "cross_environment")}
    for p in manifests.values():
        save(p, {"SOFTWARE_TEST_ONLY": True})
    variants = []
    names = c.read(root / "configs/signal-controls-v1.json")["variants"]
    for i, name in enumerate(names + [f"fixture-{i}" for i in range(8)]):
        protocol = "cross_environment" if "environment" in name else "cross_subject"
        cfg = {
            "protocol": protocol,
            "seed": 2027,
            "training": {"epochs": 40},
            "data": {"split_stride": {"train": 8, "validation": 8, "test": 1}},
        }
        variants.append(
            {
                "name": name,
                "config": cfg,
                "manifest_sha256": c.digest(manifests[protocol]),
            }
        )
        save(suite / "configs" / f"{name}.json", cfg)
    registration = {
        "profile": "cohort20",
        "variants": variants,
        "source_sha256": c._source_fingerprints(),
        "runtime_environment": c.runtime(),
        "runner_sha256": c.digest(root / "scripts/run_suite.py"),
    }
    save(suite / "preregistration.json", registration)
    out, config = root / "controls", root / "configs/signal-controls-v1.json"
    c.freeze(root, suite, config, manifests, out)
    return root, suite, config, manifests, out


def test_freeze_records_sources_and_does_not_open_models_or_data(frozen):
    root, suite, config, manifests, out = frozen
    plan = c.read(out / "plan.json")
    assert plan["status"] == "frozen_before_test"
    assert len(plan["files_sha256"]) == 6
    assert plan["config"]["permutation_seed"] == 90210
    assert len(plan["variants"]) == 4
    with pytest.raises(ValueError, match="replace frozen"):
        c.freeze(*frozen)


@pytest.mark.parametrize(
    "artifact", ["cohort20-baseline/test.json", "suite_report.json"]
)
def test_late_freeze_refused(frozen, artifact):
    root, suite, config, manifests, out = frozen
    save(suite / artifact, {"SOFTWARE_TEST_ONLY": True})
    with pytest.raises(ValueError, match="Late freeze"):
        c.freeze(root, suite, config, manifests, out.parent / "late")


@pytest.mark.parametrize(
    "change", ["code", "config", "manifest", "preregistration", "plan"]
)
def test_frozen_changes_refused_before_eval(frozen, change):
    root, suite, config, manifests, out = frozen
    path = {
        "code": root / "ghostmesh/controls.py",
        "config": config,
        "manifest": manifests["cross_subject"],
        "preregistration": suite / "preregistration.json",
        "plan": out / "plan.json",
    }[change]
    if change == "code":
        path.write_text(path.read_text() + "\n# changed\n")
    else:
        value = c.read(path)
        value["tampered"] = True
        save(path, value)
    with pytest.raises(ValueError, match="changed|differs"):
        c.evaluation_guard(*frozen)


def test_all_twelve_training_and_main_eval_guards(frozen):
    root, suite, config, manifests, out = frozen
    with pytest.raises(ValueError, match="twelve training"):
        c.evaluation_guard(*frozen)
    for row in c.read(suite / "preregistration.json")["variants"]:
        save(
            suite / row["name"] / "training_report.json",
            {"status": "trained", "epochs": 40, "config": row["config"]},
        )
    save(suite / "process.json", {"status": "running", "stage": "training"})
    with pytest.raises(ValueError, match="Main suite"):
        c.evaluation_guard(*frozen)


def test_alignment_checks_metric_and_count_drift():
    metric = {
        "pck": 0.5,
        "normalized_joint_error": 0.2,
        "valid_frames": 5,
        "valid_joints": 10,
        "excluded_frames": 1,
    }
    reference = {**metric, "windows": 6}
    c.verify_alignment(metric, reference, 6)
    for key in metric:
        other = copy.deepcopy(metric)
        other[key] += 0.01 if key in ("pck", "normalized_joint_error") else 1
        with pytest.raises(ValueError, match="differs"):
            c.verify_alignment(other, reference, 6)


def test_coverage_ignores_only_cache_reuse_flag():
    a = {"accepted_windows": 3, "storage": {"reused": False, "cache_key": "abc"}}
    b = {"accepted_windows": 3, "storage": {"reused": True, "cache_key": "abc"}}
    assert c.stable_coverage(a) == c.stable_coverage(b)
    b["storage"]["cache_key"] = "changed"
    assert c.stable_coverage(a) != c.stable_coverage(b)
    b["storage"]["cache_key"] = "abc"
    b["accepted_windows"] = 2
    assert c.stable_coverage(a) != c.stable_coverage(b)


def test_completed_main_eval_still_requires_full_report_validator(frozen):
    root, suite, config, manifests, out = frozen
    for row in c.read(suite / "preregistration.json")["variants"]:
        save(
            suite / row["name"] / "training_report.json",
            {"status": "trained", "epochs": 40, "config": row["config"]},
        )
    save(suite / "process.json", {"status": "completed", "stage": "evaluated"})
    with pytest.raises(ValueError, match="Missing required artifact"):
        c.evaluation_guard(*frozen)


def test_software_only_four_model_collection_and_scoring(tmp_path, monkeypatch):
    import torch

    root, suite, out = (
        tmp_path / "SOFTWARE_TEST_ONLY",
        tmp_path / "suite",
        tmp_path / "out",
    )
    out.mkdir()
    cfg = {
        "threads": 1,
        "training": {"batch_size": 2},
        "evaluation": {"pck_threshold": 0.2},
    }
    variants = {}
    names = ["baseline", "temporal", "environment-baseline", "environment-temporal"]
    for name in names:
        protocol = (
            "cross_environment" if name.startswith("environment") else "cross_subject"
        )
        variants[name] = {"config": {**cfg, "protocol": protocol}}
    plan = {"variants": variants, "config": {"permutation_seed": 90210}}
    save(out / "plan.json", plan)
    monkeypatch.setattr(
        c,
        "evaluation_guard",
        lambda *a: (plan, {"inputs_sha256": {"fixture": "software-only"}}),
    )
    monkeypatch.setattr(c, "_registration", lambda *a: plan)
    dataset = []
    for i in range(6):
        target = torch.tensor([[0.1 + 0.01 * i, 0.1], [0.7, 0.9]])
        dataset.append(
            {
                "pose": target,
                "valid": torch.tensor([True, True]),
                "csi": target.clone(),
                "environment": "E01" if i < 3 else "E04",
                "sequence": str(i),
                "subject": "fixture",
                "time": float(i),
                "frame_index": i,
            }
        )
    training = [r for r in dataset if r["environment"] == "E01"]
    monkeypatch.setattr(
        c,
        "_dataset",
        lambda path, split, config: training if split == "train" else dataset,
    )

    class WindowModel(torch.nn.Module):
        def forward(self, x):
            return {"pose": x, "log_sigma": torch.full(x.shape[:-1], -2.0)}

    def checkpoint(path, device):
        config = variants[Path(path).parent.name]["config"]
        return WindowModel(), {
            "config": config,
            "selected_by": "validation_normalized_joint_error",
        }

    monkeypatch.setattr(c, "load_checkpoint", checkpoint)
    metric, arrays, metadata = c._collect(
        WindowModel(), c.DataLoader(dataset, batch_size=2), torch.device("cpu")
    )
    for name in names:
        save(
            suite / name / "test.json",
            {
                **metric,
                "windows": len(dataset),
                "coverage": c._coverage(dataset, metric),
            },
        )
        (suite / name / "best.pt").write_bytes(
            b"SOFTWARE TEST ONLY; no checkpoint execution"
        )
    result = c.evaluate_controls(
        root,
        suite,
        root / "config.json",
        {"cross_subject": "fixture", "cross_environment": "fixture"},
        out,
    )
    assert len(result["models"]) == 4
    assert len(result["protocols"]) == 2
    for state in result["protocols"].values():
        assert state["windows"] == 6
        assert state["unseen_rooms_using_global_training_prior"] == ["E04"]
        assert len(state["permutation_sha256"]) == 64
    for model in result["models"].values():
        assert model["aligned"]["normalized_joint_error"] == 0
        assert model["permuted"]["normalized_joint_error"] > 0
        assert model["permuted"]["valid_joints"] == 12
