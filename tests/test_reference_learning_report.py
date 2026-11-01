"""Bind published measurements to actual executed recipes and audit versions."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def report():
    return json.loads((ROOT / "docs/domain-v2-reference-learning-results.json").read_text())


@pytest.mark.parametrize("study", ["episodic", "amplitude"])
def test_published_results_keep_actual_bytes_recipes_and_source_boundaries(study):
    r = report()
    item = r[study]
    encoded = (json.dumps(item["results"], indent=2, allow_nan=False) + "\n").encode()
    assert sha256(encoded).hexdigest() == item["results_sha256"]
    registration = item["results"]["registration"]
    for relative, expected in registration["source_sha256"].items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    config = (
        "episodic-reference-e03-v1.json"
        if study == "episodic"
        else "amplitude-person-e03-v1.json"
    )
    assert sha256((ROOT / "configs" / config).read_bytes()).hexdigest() == registration[
        "config_sha256"
    ]
    assert registration["held_annotation_values_decoded"] is False
    assert item["results"]["passes"] == {"zero": False, "calibrated": False}
    assert item["results"]["source_reference_rows"] == 120
    assert item["results"]["source_query_rows"] == 1773
    assert r["held_room_annotations_decoded"] is False
    assert r["default_models_promoted"] is False


@pytest.mark.parametrize("study", ["episodic", "amplitude"])
def test_public_auditor_version_and_both_independent_gates_match(study):
    item = report()[study]
    audit = item["independent_audit"]
    script = ROOT / "scripts" / f"verify_{study}_source.py"
    assert sha256(script.read_bytes()).hexdigest() == audit["auditor_sha256"]
    assert audit["status"] == "qualified"
    assert audit["source_ready_sha256"] == item["source_ready_sha256"]
    assert audit["source_target_frame_joins"] == 8823
    assert audit["independent_gates"] == item["results"]["gates"]
    if study == "episodic":
        assert audit["independent_metric_groups"] == 341
        assert audit["independent_metric_max_error"] < 2e-12
        assert audit["checkpoint_seals"] == 12
    else:
        assert audit["candidate_checkpoint_seals"] == 27
        assert audit["independent_reference_equations"] == 27
        assert audit["max_metric_disagreement"] < 2e-12


def test_every_amplitude_candidate_remains_visible_and_static_selection_is_explicit():
    item = report()["amplitude"]["results"]
    assert len(item["candidates"]) == 3
    for arm, candidates in item["candidates"].items():
        assert [r["step"] for r in candidates] == list(range(0, 4001, 500))
        assert item["selected"][arm] == {"zero_step": 0, "calibrated_step": 0}
        assert item["zero"][arm]["pooled"]["motion_r2"] == 0
