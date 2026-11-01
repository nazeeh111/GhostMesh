"""Integrity checks must reject altered bytes and paths before model selection."""

from hashlib import sha256
import importlib.util
from pathlib import Path
import zlib

import pytest

spec = importlib.util.spec_from_file_location(
    "verify_mmfi_inventory",
    Path(__file__).resolve().parents[1] / "scripts/verify_mmfi_inventory.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_byte_integrity_without_array_decoder(tmp_path):
    value = b"arbitrary bytes; intentionally not a MAT file"
    name = "E01/S01/A01/wifi-csi/frame001.mat"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(value)
    row = {"name": name, "size": len(value), "crc": zlib.crc32(value)}
    assert module.verify_entry(tmp_path, row) == sha256(value).digest()
    path.write_bytes(value[:-1] + b"X")
    with pytest.raises(ValueError, match="CRC mismatch"):
        module.verify_entry(tmp_path, row)


@pytest.mark.parametrize("name", ["../secret", "/etc/passwd", "E01/S01/A01/depth/a"])
def test_reject_unexpected_paths(tmp_path, name):
    with pytest.raises(ValueError):
        module.verify_entry(tmp_path, {"name": name, "size": 0, "crc": 0})
