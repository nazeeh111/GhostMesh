"""Range/ZIP safety and integrity tests use a local in-memory archive only."""

import importlib.util
import io
import hashlib
from pathlib import Path
import struct
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_mmfi", Path(__file__).resolve().parents[1] / "scripts" / "fetch_mmfi.py"
)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def archive(name="E01/S01/A02/rgb/frame001.npy", payload=b"keypoints"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, payload)
    raw = stream.getvalue()
    end = raw.rfind(b"PK\x05\x06")
    eocd = struct.unpack("<4s4H2LH", raw[end : end + 22])
    central = raw[eocd[6] : eocd[6] + eocd[5]]
    return raw, fetch.parse_directory(central), central


def install_archive(monkeypatch, raw):
    monkeypatch.setitem(fetch.FILES, "E01", ("fixture", len(raw)))
    monkeypatch.setattr(fetch, "get", lambda env, start, end: raw[start : end + 1])


def test_crc_hash_and_preserve_existing(monkeypatch, tmp_path):
    raw, entries, _ = archive()
    install_archive(monkeypatch, raw)
    records, _ = fetch.extract_group("E01", entries, tmp_path)
    target = tmp_path / entries[0]["name"]
    assert target.read_bytes() == b"keypoints"
    assert records[0]["sha256"] == hashlib.sha256(b"keypoints").hexdigest()
    fetch.extract_group("E01", entries, tmp_path)
    target.write_bytes(b"user work")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        fetch.extract_group("E01", entries, tmp_path)
    assert target.read_bytes() == b"user work"


def test_checksum_mismatch_fails(monkeypatch, tmp_path):
    raw, entries, _ = archive()
    install_archive(monkeypatch, raw)
    entries[0]["crc"] ^= 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch.extract_group("E01", entries, tmp_path)


def test_explicit_resume_verifies_local_bytes_without_network(monkeypatch, tmp_path):
    raw, entries, _ = archive()
    install_archive(monkeypatch, raw)
    initial, _ = fetch.extract_group("E01", entries, tmp_path)
    monkeypatch.setattr(
        fetch,
        "get",
        lambda *args: pytest.fail("complete resumed group must not redownload"),
    )
    resumed, downloaded = fetch.extract_group("E01", entries, tmp_path, resume=True)
    assert resumed == initial and downloaded == 0
    target = tmp_path / entries[0]["name"]
    target.write_bytes(b"changed!!")
    with pytest.raises(ValueError, match="existing member CRC mismatch"):
        fetch.extract_group("E01", entries, tmp_path, resume=True)
    assert target.read_bytes() == b"changed!!"


@pytest.mark.parametrize(
    "name", ["E01/../outside.npy", "/E01/outside.npy", "E02/S01/A02/rgb/frame001.npy"]
)
def test_archive_path_rejected(monkeypatch, tmp_path, name):
    raw, entries, _ = archive(name)
    install_archive(monkeypatch, raw)
    with pytest.raises(ValueError, match="unsafe archive path"):
        fetch.extract_group("E01", entries, tmp_path)


def test_existing_symlink_cannot_escape(monkeypatch, tmp_path):
    raw, entries, _ = archive()
    install_archive(monkeypatch, raw)
    output = tmp_path / "out"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "E01").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes output"):
        fetch.extract_group("E01", entries, output)
    assert list(outside.iterdir()) == []


def test_payload_budget_rejects_before_network(monkeypatch, tmp_path):
    _, entries, _ = archive()
    entries[0]["size"] = fetch.MAX_FILE_BYTES + 1
    monkeypatch.setattr(fetch, "get", lambda *a: pytest.fail("unexpected network"))
    with pytest.raises(ValueError, match="file budget"):
        fetch.extract_group("E01", entries, tmp_path)


def test_truncated_central_directory_rejected():
    _, _, central = archive()
    with pytest.raises(ValueError, match="truncated central"):
        fetch.parse_directory(central[:-1])


def test_declared_size_cannot_expand_without_bound(monkeypatch, tmp_path):
    raw, entries, _ = archive(payload=b"x" * 10000)
    install_archive(monkeypatch, raw)
    entries[0]["size"] = 1
    with pytest.raises(ValueError, match="oversized deflate"):
        fetch.extract_group("E01", entries, tmp_path)


def test_zip64_member_fields_decoded():
    name = b"E01/S01/A02/rgb/frame001.npy"
    extra = struct.pack("<HHQQQ", 1, 24, 999, 123, 2**32 + 50)
    central = (
        struct.pack(
            "<4s6H3L5H2L",
            b"PK\x01\x02",
            45,
            45,
            0,
            8,
            0,
            0,
            7,
            0xFFFFFFFF,
            0xFFFFFFFF,
            len(name),
            len(extra),
            0,
            0,
            0,
            0,
            0xFFFFFFFF,
        )
        + name
        + extra
    )
    entry = fetch.parse_directory(central)[0]
    assert (entry["size"], entry["compressed"], entry["offset"]) == (
        999,
        123,
        2**32 + 50,
    )


@pytest.mark.parametrize(
    "start,end", [(-1, 4), (4, 3), (0, fetch.MAX_RANGE_BYTES), (0, 10**12)]
)
def test_range_budget_rejects_without_request(start, end):
    with pytest.raises(ValueError, match="range budget"):
        fetch.get("E01", start, end)


def test_http_content_range_is_required(monkeypatch):
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, n):
            return b"wrong"

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: Response())
    monkeypatch.setattr(fetch.time, "sleep", lambda *a: None)
    with pytest.raises(ValueError, match="range unsupported"):
        fetch.get("E01", 0, 3)


def test_access_denial_is_terminal(monkeypatch):
    calls = []

    def denied(*args, **kwargs):
        calls.append(1)
        raise fetch.urllib.error.HTTPError(
            "https://example.invalid", 403, "denied", {}, None
        )

    monkeypatch.setattr(fetch.urllib.request, "urlopen", denied)
    monkeypatch.setattr(
        fetch.time, "sleep", lambda *a: pytest.fail("access denial must not retry")
    )
    with pytest.raises(fetch.urllib.error.HTTPError):
        fetch.get("E01", 0, 3)
    assert len(calls) == 1
