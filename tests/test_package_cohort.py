"""SOFTWARE TEST ONLY: tiny ZIP placeholders and invented receipts, no MM-Fi/ML."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ghostmesh_cohort_packager_test", PROJECT / "scripts/package_cohort.py"
)
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)
import package_study as study  # noqa: E402


def data_entry(data):
    return {"data": data, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_archive_default_and_custom_prefix_metadata(tmp_path):
    entries = {"z.txt": data_entry(b"last"), "a.txt": data_entry(b"first")}
    for prefix, level in ((study.PREFIX, 9), (cohort.PREFIX, 1)):
        out = tmp_path / (prefix + ".tgz")
        study.write_archive(out, entries, prefix=prefix, compresslevel=level)
        assert out.read_bytes()[4:8] == b"\0" * 4
        with tarfile.open(out) as archive:
            assert archive.getnames() == [prefix + "/a.txt", prefix + "/z.txt"]
            for member in archive:
                assert member.isfile() and member.mode == 0o644
                assert (
                    member.mtime,
                    member.uid,
                    member.gid,
                    member.uname,
                    member.gname,
                ) == (0, 0, 0, "", "")
        duplicate = tmp_path / (prefix + "-again.tgz")
        study.write_archive(duplicate, entries, prefix=prefix, compresslevel=level)
        assert duplicate.read_bytes() == out.read_bytes()
    default = tmp_path / "default.tgz"
    study.write_archive(default, entries)
    assert default.read_bytes() == (tmp_path / (study.PREFIX + ".tgz")).read_bytes()


@pytest.mark.parametrize(
    "prefix", ["", ".", "..", "../escape", "a/b", "/absolute", "a\\b"]
)
def test_archive_unsafe_prefix_refused_before_writing(tmp_path, prefix):
    with pytest.raises(ValueError, match="prefix"):
        study.write_archive(
            tmp_path / "out.tgz", {"a": data_entry(b"a")}, prefix=prefix
        )
    assert not list(tmp_path.iterdir())


def test_archive_overwrite_partial_input_drift_and_size_cap(tmp_path):
    out = tmp_path / "out.tgz"
    out.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        study.write_archive(out, {})
    assert out.read_bytes() == b"preserve"
    out.unlink()
    partial = tmp_path / "out.tgz.partial"
    partial.write_bytes(b"preserve partial")
    with pytest.raises(FileExistsError):
        study.write_archive(out, {})
    assert partial.read_bytes() == b"preserve partial"
    partial.unlink()
    with pytest.raises(ValueError, match="smaller"):
        study.write_archive(out, {"a": data_entry(b"small fixture")}, max_bytes=1)
    assert partial.is_file() and not out.exists()
    partial.unlink()
    source = tmp_path / "source"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input changed"):
        study.write_archive(
            out, {"source": {"source": source, **data_entry(b"original")}}
        )
    assert not out.exists()


@pytest.fixture
def tiny_cohort(tmp_path, monkeypatch):
    # 540 metadata records satisfy the fixed release shape, but payload totals <1 MiB.
    root, prepared = (
        tmp_path / "SOFTWARE_TEST_ONLY",
        tmp_path / "SOFTWARE_TEST_ONLY/data/mmfi-cohort20",
    )
    prepared.mkdir(parents=True)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name in ("csi.npy", "pose.npy", "timestamps.npy", "valid.npy"):
            archive.writestr(
                name, b"SOFTWARE TEST ONLY: placeholder, not a NumPy array"
            )
    content = payload.getvalue()
    checksum = hashlib.sha256(content).hexdigest()
    rows = []
    subjects = [f"S{base + i:02}" for base in (0, 10, 20, 30) for i in range(1, 6)]
    actions = [f"A{i:02}" for i in range(1, 28)]
    for subject in subjects:
        environment = f"E{int(subject[1:]) // 10 + 1:02}"
        for action in actions:
            name = f"{environment}-{subject}-{action}.npz"
            (prepared / name).write_bytes(content)
            rows.append(
                {
                    "id": f"{environment}/{subject}/{action}",
                    "path": name,
                    "sha256": checksum,
                    "frames": 297,
                    "subject": subject,
                    "environment": environment,
                    "action": action,
                }
            )
    for name, protocol in (
        ("manifest.json", "cross_subject"),
        ("cross-environment.json", "cross_environment"),
    ):
        (prepared / name).write_text(
            json.dumps({"protocol": protocol, "sequences": rows})
        )
    receipt = {
        "manifest_sha256": study.digest(prepared / "manifest.json"),
        "frames": 160380,
        "sequences": 540,
        "subjects": subjects,
        "actions": actions,
        "environments": [f"E{i:02}" for i in range(1, 5)],
        "sequence_records": [
            {"id": r["id"], "frames": r["frames"], "prepared_npz_sha256": r["sha256"]}
            for r in rows
        ],
    }
    (root / "docs").mkdir()
    (root / "docs/cohort-provenance.json").write_text(json.dumps(receipt))
    (root / "docs/mmfi-source.json").write_text('{"SOFTWARE_TEST_ONLY": true}')
    for name in ("NOTICE.md", "LICENSE"):
        (root / name).write_text("SOFTWARE TEST ONLY\n")
    (root / "data/LICENSE.md").write_text(
        "SOFTWARE TEST ONLY [receipt](../docs/cohort-provenance.json)\n"
    )
    hashes = {
        name: study.digest(prepared / name)
        for name in ("manifest.json", "cross-environment.json")
    }
    hashes["docs/cohort-provenance.json"] = study.digest(
        root / "docs/cohort-provenance.json"
    )
    monkeypatch.setattr(cohort, "ROOT", root)
    monkeypatch.setattr(cohort, "HASHES", hashes)
    return root, prepared, tmp_path / "fixture.tgz"


def test_cohort_exact_members_checksums_exclusions_and_license_links(tiny_cohort):
    root, prepared, out = tiny_cohort
    (prepared / "raw.mat").write_bytes(b"EXCLUDED")
    (prepared / "rgb.jpg").write_bytes(b"EXCLUDED")
    (prepared / "cache.npy").write_bytes(b"EXCLUDED")
    cohort.package_cohort(out)
    with tarfile.open(out) as archive:
        names = archive.getnames()
        assert names == sorted(names)
        assert not any(name.endswith((".mat", ".jpg", "cache.npy")) for name in names)
        assert len([name for name in names if name.endswith(".npz")]) == 540
        manifest = json.load(
            archive.extractfile(cohort.PREFIX + "/bundle_manifest.json")
        )
        assert len(manifest["members"]) == len(names) - 1
        for name, record in manifest["members"].items():
            content = archive.extractfile(cohort.PREFIX + "/" + name).read()
            assert len(content) == record["size"]
            assert hashlib.sha256(content).hexdigest() == record["sha256"]
        assert (
            archive.extractfile(
                cohort.PREFIX + "/data/mmfi-cohort20/manifest.json"
            ).read()
            == (prepared / "manifest.json").read_bytes()
        )
        license_text = (
            archive.extractfile(cohort.PREFIX + "/data/LICENSE.md").read().decode()
        )
        assert (
            "../docs" not in license_text
            and cohort.RELEASE + "docs/cohort-provenance.json" in license_text
        )
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        cohort.package_cohort(out)
    assert out.read_bytes() == before


@pytest.mark.parametrize("target", ["manifest", "receipt", "npz"])
def test_cohort_changed_source_refused_before_archive(tiny_cohort, target):
    root, prepared, out = tiny_cohort
    path = {
        "manifest": prepared / "manifest.json",
        "receipt": root / "docs/cohort-provenance.json",
        "npz": next(prepared.glob("*.npz")),
    }[target]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed|hash mismatch"):
        cohort.package_cohort(out)
    assert not out.exists() and not out.with_name(out.name + ".partial").exists()
