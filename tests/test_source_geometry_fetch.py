"""Selection/CRC/schema boundaries without network access."""

import json
import zlib

import numpy as np
import pytest

from scripts import fetch_source_geometry as f


def receipt():
    return dict(
        records=[
            dict(id=f"{room}/S{person:02}/A{action:02}")
            for room, first in (("E01", 1), ("E03", 21))
            for person in range(first, first + 10)
            for action in range(1, 28)
        ]
    )


def test_selection_never_includes_target_rooms_or_rgb():
    records = f.source_records(receipt())
    catalog = [dict(name=name + "/ground_truth.npy", size=60716) for name in records]
    catalog += [
        dict(name="E02/S11/A01/ground_truth.npy", size=60716),
        dict(name="E01/S01/A01/rgb/frame001.jpg", size=60716),
    ]
    assert len(f.selection(catalog, "E01", records)) == 270
    with pytest.raises(ValueError, match="Source rooms"):
        f.selection(catalog, "E02", records)
    catalog.append(catalog[0])
    room = catalog[0]["name"][:3]
    with pytest.raises(ValueError, match="catalog differs"):
        f.selection(catalog, room, records)


def test_duplicate_or_missing_source_people_are_refused():
    value = receipt()
    value["records"].append(value["records"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        f.source_records(value)
    value = receipt()
    value["records"].pop()
    with pytest.raises(ValueError, match="Complete"):
        f.source_records(value)


def test_crc_is_checked_before_decoding_and_schema_is_explicit(tmp_path):
    path = tmp_path / "E01/S01/A01/ground_truth.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((297, 17, 3), np.float32), allow_pickle=False)
    raw = path.read_bytes()
    row = dict(
        name=str(path.relative_to(tmp_path)),
        size=len(raw),
        crc=zlib.crc32(raw) & 0xFFFFFFFF,
    )
    assert f.verify_member(tmp_path, row)["bytes"] == 60716
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="CRC"):
        f.verify_member(tmp_path, row)
    np.save(path, np.zeros((297, 17, 2), np.float32), allow_pickle=False)
    raw = path.read_bytes()
    row.update(size=len(raw), crc=zlib.crc32(raw) & 0xFFFFFFFF)
    with pytest.raises(ValueError, match="297x17x3"):
        f.verify_member(tmp_path, row)


def test_existing_outputs_are_preserved_before_network_access(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(receipt()))
    out = tmp_path / "raw"
    out.mkdir()
    original = out / "keep"
    original.write_text("preserve")
    monkeypatch.setattr(
        f.provider, "directory", lambda *_: pytest.fail("network called")
    )
    with pytest.raises(FileExistsError):
        f.acquire(source, out, tmp_path / "evidence", tmp_path / "cache")
    assert original.read_text() == "preserve"
