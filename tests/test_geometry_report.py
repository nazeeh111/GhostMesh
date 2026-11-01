"""Cross-bind the public scientific report, original gates and actual audit."""

import json
from pathlib import Path

import pytest


def report():
    return json.loads(
        (
            Path(__file__).resolve().parents[1] / "docs/domain-v2-geometry-results.json"
        ).read_text()
    )


def test_all_frozen_readout_configurations_and_negative_gate_remain_visible():
    r = report()["frozen_readout"]
    assert len(r["arms"]) == 4
    assert (
        sum(
            len(rows)
            for arm in r["arms"].values()
            for rows in arm["all_configs"].values()
        )
        == 40
    )
    assert r["passes"] is False and not all(r["gate"].values())
    assert all(
        set(arm["selections"]) == {"ordinary", "fixed_effects"}
        for arm in r["arms"].values()
    )


def test_actual_cpu_audit_is_bound_to_report_and_all_groups():
    r = report()
    audit = r["published_independent_auditor"]
    assert audit["status"] == "qualified"
    assert audit["manifest_sha256"] == r["distillation_manifest_sha256"]
    assert audit["checkpoint_seals"] == 16
    assert audit["source_validation_endpoints_per_arm"] == 3546
    assert audit["independent_pose_and_geometry_groups"] == 136
    assert audit["max_metric_disagreement"] < 2e-12
    for arm, result in audit["arms"].items():
        expected = r["distillation"]["arms"][arm]["pooled"]
        actual = result["independent"]["pooled"]
        assert actual["nje"] == pytest.approx(
            expected["pose"]["normalized_joint_error"], abs=2e-12
        )
        assert actual["motion_r2"] == pytest.approx(
            expected["pose"]["record_centered_articulation_r2"], abs=2e-12
        )
        assert result["cpu_pose_max_abs"] < 3e-6


def test_no_failed_gate_or_source_only_scope_is_presented_as_room_success():
    r = report()
    d = r["distillation"]
    assert d["passes"] is False
    assert d["gate"]["body_motion_useful"] is False
    assert d["registration"]["E02_numeric_values_read"] is False
    assert d["registration"]["config"]["steps"] == 2000
    assert len(d["arms"]) == 4
    assert r["annotation_contract"]["raw_images_acquired"] is False
    assert r["annotation_contract"]["members"] == 540
