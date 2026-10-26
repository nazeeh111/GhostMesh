import json
import random

import numpy as np
import pytest
import torch

from ghostmesh import engine


def manifest(protocol="cross_subject"):
    return {
        "protocol": protocol,
        "provenance": {"kind": "synthetic_fixture"},
        "sequences": [
            {"id": "a", "subject": "S01", "environment": "E01", "split": "train"},
            {"id": "b", "subject": "S02", "environment": "E01", "split": "validation"},
            {"id": "c", "subject": "S31", "environment": "E04", "split": "test"},
        ],
    }


def test_protocol_catches_people_environment_and_declared_protocol_leakage():
    m = manifest()
    engine.validate_protocol(m, "cross_subject")
    m["sequences"][2]["subject"] = "S01"
    with pytest.raises(ValueError, match="Subjects overlap"):
        engine.validate_protocol(m, "cross_subject")
    m = manifest("cross_environment")
    engine.validate_protocol(m, "cross_environment")
    with pytest.raises(ValueError, match="does not match"):
        engine.validate_protocol(m, "cross_subject")
    m["sequences"][2]["environment"] = "E01"
    with pytest.raises(ValueError, match="environments overlap"):
        engine.validate_protocol(m, "cross_environment")


class TinyDataset(torch.utils.data.Dataset):
    input_shape = (3, 16, 4)
    joints = 17

    def __len__(self):
        return 2

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return {
            "csi": torch.rand(4, 3, 16, 4, generator=generator),
            "pose": torch.rand(17, 2, generator=generator),
            "valid": torch.ones(17, dtype=torch.bool),
        }


def test_training_never_constructs_test_dataset_and_binds_selected_checkpoint(
    tmp_path, monkeypatch
):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()))
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "protocol": "cross_subject",
                "model": {"width": 16},
                "training": {"epochs": 2, "batch_size": 2},
            }
        )
    )
    monkeypatch.setattr(engine, "load_manifest", lambda _: manifest())
    constructed = []

    def dataset(_, split, __):
        constructed.append(split)
        assert split != "test", "Test samples must never influence checkpoint selection"
        return TinyDataset()

    monkeypatch.setattr(engine, "_dataset", dataset)
    report = engine.train(path, cfg, tmp_path / "run")
    assert constructed == ["train", "validation"]
    assert report["purpose"] == "software_check_only"
    checkpoint_path = tmp_path / "run" / "best.pt"
    _, checkpoint = engine.load_checkpoint(checkpoint_path)
    assert checkpoint["selected_by"] == "validation_normalized_joint_error"
    assert (
        checkpoint["epoch"]
        == min(report["history"], key=lambda r: r["validation_nje"])["epoch"]
    )
    assert report["source_sha256"] == checkpoint["source_sha256"]
    assert set(report["source_sha256"]) == {
        "engine.py",
        "models.py",
        "data.py",
        "preprocess.py",
        "metrics.py",
    }
    assert all(len(value) == 64 for value in report["source_sha256"].values())
    assert report["runtime_environment"]["python"]
    assert report["runtime_environment"]["machine"]
    assert report["training_coverage"]["accepted_windows"] == 2
    assert report["training_coverage"]["accepted_fraction"] is None
    monkeypatch.setattr(
        engine, "benchmark", lambda *a, **kw: {"scope": "test-only stub"}
    )
    monkeypatch.setattr(engine, "_dataset", lambda *a, **kw: TinyDataset())
    evaluated = engine.evaluate(path, checkpoint_path)
    assert evaluated["source_matches_training"] is True
    assert evaluated["source_changed_since_training"] == []
    assert evaluated["coverage"]["accepted_windows"] == 2
    assert evaluated["coverage"]["metric_eligible_windows"] == 2
    assert evaluated["coverage"]["metric_eligible_fraction_of_candidates"] is None
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Manifest differs"):
        engine.evaluate(path, checkpoint_path)


def test_existing_experiment_is_preserved(tmp_path):
    with pytest.raises(FileExistsError):
        engine.train("missing.json", "missing.json", tmp_path)


def test_dense_test_stride_is_independent_of_training_sampling(monkeypatch):
    calls = []
    monkeypatch.setattr(
        engine,
        "WindowDataset",
        lambda path, split, **kwargs: calls.append((split, kwargs["stride"])),
    )
    config = {
        "data": {
            "window": 8,
            "stride": 8,
            "split_stride": {"train": 8, "validation": 8, "test": 1},
        }
    }
    for split in ["train", "validation", "test"]:
        engine._dataset("unused.json", split, config)
    assert calls == [("train", 8), ("validation", 8), ("test", 1)]


def test_mean_pose_is_fitted_only_on_training_labels(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest()))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"protocol": "cross_subject", "data": {"window": 4}}))
    monkeypatch.setattr(engine, "load_manifest", lambda _: manifest())
    requested = []
    shift = [0.1]

    class BaselineDataset(TinyDataset):
        def __init__(self, split):
            self.split = split

        def __getitem__(self, index):
            row = super().__getitem__(index)
            if self.split != "train":
                row["pose"] = row["pose"] * (1 - shift[0])
            row.update(
                sequence=f"seq{index}",
                subject=f"person{index}",
                environment="room",
                time=float(index),
                frame_index=index,
            )
            return row

    def dataset(_, split, __):
        requested.append(split)
        return BaselineDataset(split)

    monkeypatch.setattr(engine, "_dataset", dataset)
    first = engine.evaluate_mean_pose(manifest_path, config)
    shift[0] = 0.9
    second = engine.evaluate_mean_pose(manifest_path, config)
    assert requested == ["train", "test", "train", "test"]
    np.testing.assert_array_equal(first["train_mean_pose"], second["train_mean_pose"])
    np.testing.assert_array_equal(first["train_sigma"], second["train_sigma"])
    assert first["normalized_joint_error"] != second["normalized_joint_error"]
    assert first["purpose"] == "software_check_only"
    assert first["coverage"]["accepted_windows"] == 2
    assert first["coverage"]["metric_eligible_windows"] == 2
    assert first["training_coverage"]["accepted_windows"] == 2
    assert set(first["source_sha256"]) == {
        "engine.py",
        "models.py",
        "data.py",
        "preprocess.py",
        "metrics.py",
    }


def comparison_report(nje, pck):
    return {
        "manifest_sha256": "same",
        "split": "test",
        "protocol": "cross_subject",
        "pck_threshold": 0.2,
        "purpose": "dataset_experiment",
        "windows": 500,
        "sequence_subjects": {"a": "S01", "b": "S01", "c": "S02", "d": "S02"},
        "domains": {
            "sequence": {
                name: {
                    "normalized_joint_error": nje[index],
                    "pck": pck[index],
                    "valid_joints": [100, 100, 1000, 500][index],
                }
                for index, name in enumerate("abcd")
            }
        },
    }


def test_paired_bootstrap_resamples_sequence_means_not_correlated_frames():
    reference = comparison_report([0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6])
    candidate = comparison_report([0.05, 0.15, 0.25, 0.35], [0.95, 0.85, 0.75, 0.65])
    result = engine.compare_reports(reference, candidate, repeats=500)
    assert result["clusters"] == 4
    assert result["normalized_joint_error"]["delta"] == pytest.approx(-0.05)
    assert result["normalized_joint_error"]["ci95"] == pytest.approx([-0.05, -0.05])
    assert result["pck"]["delta"] == pytest.approx(0.05)
    clustered = engine.compare_reports(
        reference, candidate, repeats=500, cluster="subject"
    )
    assert clustered["clusters"] == 2
    assert clustered["normalized_joint_error"]["delta"] == pytest.approx(-0.05)
    assert engine.compare_reports(reference, candidate, repeats=500) == result


def test_comparison_refuses_unpaired_target_coverage():
    reference = comparison_report([0.1] * 4, [0.9] * 4)
    candidate = comparison_report([0.2] * 4, [0.8] * 4)
    candidate["domains"]["sequence"]["a"]["valid_joints"] += 1
    with pytest.raises(ValueError, match="coverage"):
        engine.compare_reports(reference, candidate)
    candidate = comparison_report([0.2] * 4, [0.8] * 4)
    candidate["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="manifest"):
        engine.compare_reports(reference, candidate)


def test_cached_training_builds_only_development_splits_and_reports_storage(
    tmp_path, monkeypatch
):
    from ghostmesh.prepare import make_fixture

    path = make_fixture(tmp_path / "fixture", frames=16)
    config = tmp_path / "cache-config.json"
    config.write_text(
        json.dumps(
            {
                "protocol": "cross_subject",
                "threads": 2,
                "model": {"width": 8, "dilations": [1]},
                "data": {"window": 4, "cache_windows": True, "stride": 2},
                "training": {"epochs": 1, "batch_size": 8},
            }
        )
    )
    trained = engine.train(path, config, tmp_path / "run")
    assert trained["training_coverage"]["storage"]["mode"] == "readonly_window_memmap"
    assert trained["validation_coverage"]["storage"]["checksums_verified"] is True
    cache_root = path.parent / ".ghostmesh-cache"
    completed = list(cache_root.glob("*/metadata.json"))
    assert {json.loads(p.read_text())["request"]["split"] for p in completed} == {
        "train",
        "validation",
    }
    monkeypatch.setattr(
        engine, "benchmark", lambda *a, **kw: {"scope": "test-only stub"}
    )
    evaluated = engine.evaluate(path, tmp_path / "run" / "best.pt")
    assert evaluated["coverage"]["storage"]["mode"] == "readonly_window_memmap"
    assert evaluated["coverage"]["rejected_windows"] > 0
    assert evaluated["source_matches_training"] is True
    completed = list(cache_root.glob("*/metadata.json"))
    assert {json.loads(p.read_text())["request"]["split"] for p in completed} == {
        "train",
        "validation",
        "test",
    }


class RandomizedTinyDataset(TinyDataset):
    """Use all restored RNG families while retaining deterministic supervision."""

    def __getitem__(self, index):
        row = super().__getitem__(index)
        row["csi"] += 0.01 * (
            random.random() + float(np.random.random()) + float(torch.rand(()))
        )
        return row


def _resume_files(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "protocol": "cross_subject",
                "threads": 2,
                "seed": 123,
                "model": {"width": 8, "dilations": [1]},
                "training": {
                    "epochs": 3,
                    "batch_size": 1,
                    "augmentation_strength": 0.2,
                    "consistency_weight": 0.3,
                    "consistency_mode": "bounded_precision",
                    "ema_decay": 0.8,
                },
            }
        )
    )
    monkeypatch.setattr(engine, "load_manifest", lambda _: manifest())

    def dataset(_, split, __):
        assert split in {"train", "validation"}, "Resume must never decode test samples"
        return RandomizedTinyDataset()

    monkeypatch.setattr(engine, "_dataset", dataset)
    return path, config


def _assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("interruption", ["validation", "best_publication"])
def test_completed_epoch_resume_matches_uninterrupted_rng_weights_history_and_best_hash(
    tmp_path, monkeypatch, interruption
):
    path, config = _resume_files(tmp_path, monkeypatch)
    full = engine.train(path, config, tmp_path / "full")
    if interruption == "validation":
        collect = engine._collect
        calls = [0]

        def interrupted_collect(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise KeyboardInterrupt(
                    "test-owned interruption in second partial epoch"
                )
            return collect(*args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(engine, "_collect", interrupted_collect)
            with pytest.raises(KeyboardInterrupt):
                engine.train(path, config, tmp_path / "resumed")
    else:
        save = engine._atomic_torch_save

        def interrupted_publish(value, path):
            if path.name == "best.pt":
                raise KeyboardInterrupt("test-owned interruption after epoch commit")
            return save(value, path)

        with monkeypatch.context() as scoped:
            scoped.setattr(engine, "_atomic_torch_save", interrupted_publish)
            with pytest.raises(KeyboardInterrupt):
                engine.train(path, config, tmp_path / "resumed")
        assert not (tmp_path / "resumed" / "best.pt").exists()
    committed = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=True)
    assert committed["completed_epoch"] == 1
    resumed = engine.train(path, config, tmp_path / "resumed", resume=True)
    assert resumed["resume"]["from_completed_epoch"] == 1
    assert resumed["history"] == full["history"]
    assert resumed["best_epoch"] == full["best_epoch"]
    assert resumed["checkpoint_sha256"] == full["checkpoint_sha256"]
    a = torch.load(tmp_path / "full" / "latest.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=True)
    for key in ("model", "teacher", "optimizer", "rng", "history", "best_checkpoint"):
        _assert_nested_equal(a[key], b[key])


def test_resume_before_first_epoch_restarts_seed_and_preserves_partial_managed_bytes(
    tmp_path, monkeypatch
):
    path, config = _resume_files(tmp_path, monkeypatch)
    full = engine.train(path, config, tmp_path / "full")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            engine,
            "_collect",
            lambda *a, **kw: (_ for _ in ()).throw(
                KeyboardInterrupt("test-owned first epoch interruption")
            ),
        )
        with pytest.raises(KeyboardInterrupt):
            engine.train(path, config, tmp_path / "resumed")
    assert not (tmp_path / "resumed" / "latest.pt").exists()
    leftovers = {
        "best.pt": b"test-owned incomplete snapshot",
        "history.json": b"test-owned incomplete history",
    }
    for name, content in leftovers.items():
        (tmp_path / "resumed" / name).write_bytes(content)
    resumed = engine.train(path, config, tmp_path / "resumed", resume=True)
    assert resumed["resume"]["seed_restart"] is True
    assert resumed["history"] == full["history"]
    assert resumed["checkpoint_sha256"] == full["checkpoint_sha256"]
    preserved = [
        (tmp_path / "resumed" / p).read_bytes()
        for p in resumed["resume"]["recovery_files"]
    ]
    assert all(content in preserved for content in leftovers.values())


def test_resume_refuses_changed_config_source_manifest_runtime_or_device_without_writing(
    tmp_path, monkeypatch
):
    path, config = _resume_files(tmp_path, monkeypatch)
    engine.train(path, config, tmp_path / "run")
    original_config = config.read_text()
    latest = tmp_path / "run" / "latest.pt"
    original_latest = engine._fingerprint(latest)
    changed = json.loads(original_config)
    changed["training"]["epochs"] = 4
    config.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="config"):
        engine.train(path, config, tmp_path / "run", resume=True)
    config.write_text(original_config)
    actual_sources = engine._source_fingerprints()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            engine,
            "_source_fingerprints",
            lambda: {**actual_sources, "engine.py": "f" * 64},
        )
        with pytest.raises(ValueError, match="source_sha256"):
            engine.train(path, config, tmp_path / "run", resume=True)
    original_manifest = path.read_text()
    path.write_text(original_manifest + "\n")
    with pytest.raises(ValueError, match="manifest_sha256"):
        engine.train(path, config, tmp_path / "run", resume=True)
    path.write_text(original_manifest)
    environment = engine._runtime_environment()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            engine,
            "_runtime_environment",
            lambda: {**environment, "machine": "different"},
        )
        with pytest.raises(ValueError, match="runtime_environment"):
            engine.train(path, config, tmp_path / "run", resume=True)
    with pytest.raises(ValueError, match="device"):
        engine.train(path, config, tmp_path / "run", device="cpu:0", resume=True)
    assert engine._fingerprint(latest) == original_latest


def test_bounded_precision_consistency_has_explicit_units_bounds_and_no_teacher_gradient():
    pose = torch.tensor([[[0.3, 0.4], [0.3, 0.4]]], requires_grad=True)
    teacher_pose = torch.zeros_like(pose, requires_grad=True)
    teacher_sigma = torch.log(torch.tensor([[0.1, 1.0]])).requires_grad_()
    prediction = {"pose": pose}
    teacher = {"pose": teacher_pose, "log_sigma": teacher_sigma}
    assert engine._consistency_loss(prediction, teacher, "raw").item() == pytest.approx(
        0.25
    )
    bounded = engine._consistency_loss(prediction, teacher, "bounded_precision")
    assert bounded.item() == pytest.approx(6.5)
    bounded.backward()
    assert pose.grad is not None and torch.isfinite(pose.grad).all()
    assert teacher_pose.grad is None and teacher_sigma.grad is None
    teacher["log_sigma"] = torch.log(torch.tensor([[1e-6, 10.0]]))
    assert engine._consistency_loss(
        prediction, teacher, "bounded_precision"
    ).item() == pytest.approx(25.25)
    with pytest.raises(ValueError, match="consistency_mode"):
        engine._consistency_loss(prediction, teacher, "unknown")
