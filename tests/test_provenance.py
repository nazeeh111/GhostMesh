import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "compile_provenance",
    Path(__file__).resolve().parents[1] / "scripts" / "compile_provenance.py",
)
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


def source_manifest(tmp_path):
    value = {
        "schema_version": 1,
        "protocol": "cross_subject",
        "provenance": {"kind": "licensed_dataset", "dataset": "MM-Fi"},
        "sequences": [
            {
                "id": "E01/S01/A01",
                "frames": 1,
                "split": "train",
                "sha256": "0" * 64,
                "raw_frame_hashes": [
                    {"frame": 1, "csi_sha256": "1" * 64, "pose_sha256": "2" * 64}
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path, value


def test_receipts_have_exact_path_order_and_manifest_binding(tmp_path):
    path, value = source_manifest(tmp_path)
    out = tmp_path / "provenance.json"
    result = compiler.compile_provenance(path, out)
    expected = (
        "E01/S01/A01/wifi-csi/frame001.mat\t"
        + "1" * 64
        + "\nE01/S01/A01/rgb/frame001.npy\t"
        + "2" * 64
        + "\n"
    ).encode()
    assert (
        result["raw_source_paths_and_sha256_digest"]
        == hashlib.sha256(expected).hexdigest()
    )
    assert result["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["frames"] == 1
    with pytest.raises(FileExistsError):
        compiler.compile_provenance(path, out)


def test_different_label_cohort_is_rejected(tmp_path):
    path, _ = source_manifest(tmp_path)
    inspection = tmp_path / "inspection.json"
    inspection.write_text(json.dumps({"sequence_records": [{"id": "E01/S02/A01"}]}))
    with pytest.raises(ValueError, match="different sequence cohort"):
        compiler.compile_provenance(path, tmp_path / "out.json", inspection)
