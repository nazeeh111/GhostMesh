"""Ordering and failure isolation with no ML, data or network execution."""

from types import SimpleNamespace

import pytest

from scripts import run_masked_study as r


def prerequisites(root):
    for name in (r.RAW, r.CACHE, r.OLD):
        (root / name).mkdir(parents=True)


def test_query_stages_follow_all_fits_audits_and_ledger_in_actual_plan():
    names = [name for name, _, _ in r.plan()]
    first_score = next(i for i, name in enumerate(names) if name.startswith("score_"))
    assert names[:first_score] == [
        "fit_source",
        "audit_source",
        "fit_root",
        "audit_root",
        "fit_calibrated_body",
        "fit_calibrated_full",
        "audit_calibrated_body",
        "audit_calibrated_full",
        "freeze",
    ]
    assert names[first_score:] == [
        "score_zero_body",
        "score_zero_full",
        "score_calibrated_body",
        "score_calibrated_full",
        "audit_scores",
    ]
    for _, _, arguments in r.plan():
        assert "E04" not in " ".join(arguments)


def test_existing_experiment_or_receipt_is_preserved_before_child_launch(
    tmp_path, monkeypatch
):
    prerequisites(tmp_path)
    (tmp_path / r.SOURCE).mkdir()
    sentinel = tmp_path / r.SOURCE / "untouched"
    sentinel.write_bytes(b"existing scientific bytes")
    monkeypatch.setattr(
        r.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("Existing experiment must block children"),
    )
    with pytest.raises(FileExistsError):
        r.run(tmp_path, tmp_path / "runner")
    assert sentinel.read_bytes() == b"existing scientific bytes"
    assert not (tmp_path / "runner").exists()


def test_failed_source_audit_stops_before_any_refitting_or_query_scoring(
    tmp_path, monkeypatch
):
    prerequisites(tmp_path)
    commands = []

    def child(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0 if len(commands) == 1 else 7)

    monkeypatch.setattr(r.subprocess, "run", child)
    with pytest.raises(RuntimeError, match="audit_source"):
        r.run(tmp_path, tmp_path / "runner")
    assert len(commands) == 2
    assert (
        tmp_path / "runner/audit_source-exit.json"
    ).read_text().strip() == '{"returncode": 7}'


def test_hash_placeholders_bind_actual_artifact_bytes_and_missing_files_fail(tmp_path):
    (tmp_path / "ready.json").write_bytes(b"synthetic readiness bytes")
    assert r.resolve(tmp_path, ["--expected", "sha256:ready.json"]) == [
        "--expected",
        r.v.contracts.digest(tmp_path / "ready.json"),
    ]
    with pytest.raises(FileNotFoundError):
        r.resolve(tmp_path, ["sha256:absent.json"])
