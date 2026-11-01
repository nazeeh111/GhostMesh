"""A freeze must precede scores; fabricated fixtures are software tests only."""

import json

import pytest

from scripts import freeze_masked_evaluation as f


@pytest.mark.parametrize(
    "existing", ["results.json", "query-permutation.npy", "arm-query-prediction.npy"]
)
def test_freeze_refuses_completed_or_partial_scoring_before_seal_reads(
    tmp_path, monkeypatch, existing
):
    directory = tmp_path / "work" / next(iter(f.v.PROTOCOLS.values()))[0]
    directory.mkdir(parents=True)
    (directory / existing).write_bytes(b"SOFTWARE TEST ONLY")
    monkeypatch.setattr(
        f.v.contracts,
        "sealed",
        lambda *a: pytest.fail("Scored protocol must fail before seal reads"),
    )
    with pytest.raises(ValueError, match="already started"):
        f.freeze(tmp_path, tmp_path / "ledger.json")
    assert not (tmp_path / "ledger.json").exists()


def test_freeze_preserves_existing_ledger_before_input_reads(tmp_path):
    out = tmp_path / "ledger.json"
    out.write_bytes(b"original ledger")
    with pytest.raises(FileExistsError):
        f.freeze(tmp_path / "absent", out)
    assert out.read_bytes() == b"original ledger"


def test_all_four_audits_bind_fresh_ledger_and_the_post_score_contract(
    tmp_path, monkeypatch
):
    (tmp_path / "runtime.py").write_text("# SOFTWARE TEST ONLY\n")
    manifest = {"runtime.py": f.v.contracts.digest(tmp_path / "runtime.py")}
    for directory, filename, kind, flag in f.v.PROTOCOLS.values():
        path = tmp_path / "work" / directory
        path.mkdir(parents=True)
        (path / filename).write_text(json.dumps(dict(kind=kind, **{flag: False})))
    for filename in f.v.AUDITS.values():
        (tmp_path / filename).write_text("{}")
    monkeypatch.setattr(
        f.v.contracts,
        "sealed",
        lambda *a: ({}, dict(config=dict(seed=2027), source_sha256=manifest)),
    )
    audits = []
    monkeypatch.setattr(f.v, "audit_receipt", lambda *a: audits.append(a[1]))
    output = tmp_path / "ledger.json"
    sha = f.freeze(tmp_path, output)
    assert set(audits) == {"source", "root", "body", "full"}
    assert f.v.ledger_contract(output, sha, tmp_path)["reference_labels"] == 120
    assert len(json.loads(output.read_text())["frozen_sha256"]) == 8
