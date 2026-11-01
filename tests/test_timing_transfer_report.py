"""Bind reported failed source results to the executed recipes and independent audits."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def report():
    return json.loads(
        (ROOT / "docs/domain-v2-timing-transfer-results.json").read_text()
    )


@pytest.mark.parametrize(
    "study,config,auditor",
    [
        (
            "cross_person",
            "pose-alignment-e03-v1.json",
            "verify_pose_alignment_source.py",
        ),
        ("cross_room", "cross-room-dense-e03-v1.json", "verify_dense_room_source.py"),
    ],
)
def test_every_source_candidate_recipe_and_independent_gate_are_bound(
    study, config, auditor
):
    item = report()[study]
    result = item["results"]
    registration = result["registration"]
    assert (
        sha256(
            (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        ).hexdigest()
        == item["results_sha256"]
    )
    for name, expected in registration["source_sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert (
        sha256((ROOT / "configs" / config).read_bytes()).hexdigest()
        == registration["config_sha256"]
    )
    audit = item["independent_audit"]
    assert (
        sha256((ROOT / "scripts" / auditor).read_bytes()).hexdigest()
        == audit["auditor_sha256"]
    )
    assert audit["source_ready_sha256"] == item["source_ready_sha256"]
    assert audit["status"] == "qualified" and audit["max_metric_disagreement"] < 1e-12
    assert audit["independent_gates"] == result["gates"]
    assert result["passes"] == {"zero": False, "calibrated": False}
    assert registration["held_annotation_values_decoded"] is False
    assert result["held_metrics_computed"] is False
    assert len(result["candidates"]) == 3
    for candidates in result["candidates"].values():
        assert [v["step"] for v in candidates] == list(range(0, 3001, 500))
    if study == "cross_room":
        assert result["source_reference_rows_per_room"] == 120
        assert result["source_reference_rows_total"] == 240
        assert audit["source_target_frame_joins"] == 112391
        assert audit["independent_reference_equations"] == 42
    else:
        assert result["source_reference_rows"] == 120
        assert audit["source_target_frame_joins"] == 8549


def test_complete_timing_curves_use_train_selection_and_preserve_limits():
    r = report()
    timing = r["timing"]
    result = timing["results"]
    assert (
        sha256(
            (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        ).hexdigest()
        == timing["results_sha256"]
    )
    for name, expected in result["registration"]["source_sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert len(result["summary"]) == 10
    for row in result["summary"].values():
        assert set(map(int, row["all_lags"])) == set(range(-24, 25))
        assert row["selected_train_lag"] == -4
        assert max(row["all_lags"], key=lambda k: row["all_lags"][k]["train"]) == "-4"
    assert r["held_room_annotations_decoded"] is False
    assert r["default_models_promoted"] is False


def test_research_figure_is_bound_to_current_report_and_renderer():
    folder = ROOT / "docs/assets/timing-transfer-v3"
    provenance = json.loads((folder / "provenance.json").read_text())
    assert (
        provenance["report_sha256"]
        == sha256(
            (ROOT / "docs/domain-v2-timing-transfer-results.json").read_bytes()
        ).hexdigest()
    )
    assert (
        provenance["renderer_sha256"]
        == sha256((ROOT / "scripts/plot_timing_transfer.py").read_bytes()).hexdigest()
    )
    for name, expected in provenance["files_sha256"].items():
        assert sha256((folder / name).read_bytes()).hexdigest() == expected


def test_portable_cache_and_two_room_timing_keep_executed_sources_and_exact_comparison():
    r = report()
    proof = r["portable_dense_cache_verification"]
    assert proof["status"] == "qualified" and proof["all_output_array_bytes_exact"]
    for name, expected in proof["public_registration"]["source_sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert proof["public_receipt"]["frame_rows"] == 160380
    assert proof["public_receipt"]["logical_dense_endpoints"] == 136213
    assert proof["public_receipt"]["annotation_arrays_decoded"] is False
    timing = r["two_source_room_timing"]
    result = timing["results"]
    assert (
        sha256(
            (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        ).hexdigest()
        == timing["results_sha256"]
    )
    for name, expected in result["registration"]["source_sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert timing["all_preceding_aggregate_values_exact"] is True
    assert result["records"] == 432
    for room in result["summary"].values():
        assert len(room) == 10
        assert all(curve["selected_lag"] == -4 for curve in room.values())
