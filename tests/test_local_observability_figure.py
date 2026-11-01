"""Scientific reporting boundaries; no raw data or plotting dependency needed."""

import json
from pathlib import Path

import pytest

from scripts.plot_local_observability import EXPECTED, digest, extract, row


def test_extraction_refuses_changed_evidence_before_reading_results(tmp_path):
    name = next(iter(EXPECTED))
    (tmp_path / name).mkdir()
    # A plausible but changed result cannot silently enter the historical plot.
    (tmp_path / name / "results.json").write_text('{"results": []}\n')
    with pytest.raises(ValueError, match="Frozen evidence changed"):
        extract(tmp_path)


def test_result_identity_must_be_unique():
    rows = [
        dict(protocol="held_time", feature="quotient", kernel="rbf", value=1),
        dict(protocol="held_action", feature="quotient", kernel="rbf", value=2),
    ]
    assert (
        row(rows, protocol="held_time", feature="quotient", kernel="rbf")["value"] == 1
    )
    with pytest.raises(ValueError, match="Expected one"):
        row(rows, kernel="rbf")
    with pytest.raises(ValueError, match="Expected one"):
        row(rows, feature="absent")


def test_shipped_data_hash_and_person_disjoint_room_scope():
    root = Path(__file__).resolve().parents[1] / "docs/assets/local-observability-v2"
    data = json.loads((root / "figure-data.json").read_text())
    manifest = json.loads((root / "artifact-hashes.json").read_text())
    assert digest(root / "figure-data.json") == manifest["figure-data.json"]
    assert data["evidence_sha256"] == EXPECTED
    assert not set(data["room"]["reference_people"]) & set(data["room"]["query_people"])
    assert all(
        r["windows"] == data["room"]["query_windows"]
        for r in data["room"]["results"].values()
    )
    assert data["local"]["counts"] == {"train": 111, "validation": 45, "test": 104}
    assert "SOURCE DEVELOPMENT" in data["scope"]
