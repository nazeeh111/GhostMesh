"""Manifest integrity and refusal boundaries; no dataset required."""

from hashlib import sha256
import json

import pytest

from scripts.seal_research_artifacts import SCOPE, seal


def test_manifest_binds_exact_bytes_and_nested_relative_paths(tmp_path):
    root = tmp_path / "experiment"
    (root / "executed-source").mkdir(parents=True)
    (root / "results.json").write_bytes(b'{"value": 1}\n')
    (root / "executed-source" / "probe.py").write_bytes(b"# utf8 \xc3\xa9\n")
    result = seal(root)
    assert result == {
        "scope": SCOPE,
        "sha256": {
            "executed-source/probe.py": sha256(b"# utf8 \xc3\xa9\n").hexdigest(),
            "results.json": sha256(b'{"value": 1}\n').hexdigest(),
        },
    }
    saved = json.loads((root / "artifact-manifest-final.json").read_text())
    assert saved == result
    assert "artifact-manifest-final.json" not in saved["sha256"]
    (root / "results.json").write_bytes(b'{"value": 2}\n')
    assert (
        saved["sha256"]["results.json"]
        != sha256((root / "results.json").read_bytes()).hexdigest()
    )
    before = (root / "artifact-manifest-final.json").read_bytes()
    with pytest.raises(FileExistsError):
        seal(root)
    assert (root / "artifact-manifest-final.json").read_bytes() == before


@pytest.mark.parametrize("kind", ["file", "directory", "dangling"])
def test_symlinks_are_rejected_without_creating_manifest(tmp_path, kind):
    root = tmp_path / "experiment"
    root.mkdir()
    outside = tmp_path / "outside"
    if kind == "file":
        outside.write_bytes(b"outside")
    elif kind == "directory":
        outside.mkdir()
        (outside / "private.txt").write_bytes(b"outside")
    (root / "link").symlink_to(outside, target_is_directory=kind == "directory")
    with pytest.raises(ValueError, match="Symlink"):
        seal(root)
    assert not (root / "artifact-manifest-final.json").exists()


@pytest.mark.parametrize(
    "filename", ["../elsewhere.json", "/tmp/elsewhere.json", "nested/manifest.json"]
)
def test_output_cannot_leave_selected_root(tmp_path, filename):
    with pytest.raises(ValueError, match="filename"):
        seal(tmp_path, filename)
