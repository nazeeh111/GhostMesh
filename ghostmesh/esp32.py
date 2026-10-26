"""Opt-in ESP-CSI serial acquisition and deterministic amplitude conversion.

Hardware mode records signal only. MM-Fi weights cannot be transferred to an
ESP32's different channel layout by relabeling tensor dimensions.
"""

from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np


def convert_csv(
    source, output, *, timestamp_field="local_timestamp", data_field="data"
):
    """Read header-bearing ESP-CSI CSV or the recorder's unedited serial log.

    Firmware startup/log lines are ignored; CSI rows remain strictly validated.
    Only signed-eight-bit integer I/Q payloads are supported. Wider formats,
    including out-of-range gain-compensated values, need a validated adapter.
    """
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Amplitude output must use the .npz suffix")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    signals = []
    times = []
    wrap = 0
    previous = None
    rejected = 0
    ignored_lines = 0
    headers_seen = 0
    unsupported_payloads = 0
    fieldnames = None
    with open(source, newline="", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            try:
                cells = next(csv.reader([line], strict=True), [])
            except csv.Error:
                if fieldnames is not None and line.lstrip().startswith("CSI_DATA,"):
                    rejected += 1
                else:
                    ignored_lines += 1
                continue
            cells = [cell.strip() for cell in cells]
            if cells:
                cells[0] = cells[0].lstrip("\ufeff")
            if {timestamp_field, data_field} <= set(cells):
                if len(set(cells)) != len(cells):
                    raise ValueError("CSI CSV header contains duplicate field names")
                fieldnames = cells
                headers_seen += 1
                continue
            if fieldnames is None:
                ignored_lines += 1
                continue
            if "type" in fieldnames:
                index = fieldnames.index("type")
                if len(cells) <= index or cells[index] != "CSI_DATA":
                    ignored_lines += 1
                    continue
            elif len(cells) != len(fieldnames) or not cells[
                fieldnames.index(data_field)
            ].startswith("["):
                ignored_lines += 1
                continue
            if len(cells) != len(fieldnames):
                rejected += 1
                continue
            row = dict(zip(fieldnames, cells))
            try:
                payload = json.loads(row[data_field])
                raw_time = int(row[timestamp_field])
                if not 0 <= raw_time < 2**32:
                    raise ValueError("Timestamp outside uint32 range")
                if (
                    not isinstance(payload, list)
                    or len(payload) % 2
                    or len(payload) < 4
                ):
                    raise ValueError("Malformed signed-byte I/Q payload")
                if any(
                    type(value) is not int or not -128 <= value <= 127
                    for value in payload
                ):
                    unsupported_payloads += 1
                    raise ValueError("Unsupported non-signed-eight-bit I/Q payload")
                pairs = np.asarray(payload, dtype=np.float32)
                candidate_wrap = wrap
                if previous is not None and raw_time < previous:
                    if previous - raw_time > 2**31:
                        candidate_wrap += 2**32
                    else:
                        raise ValueError("Out-of-order timestamp")
                timestamp = (raw_time + candidate_wrap) * 1e-6
                if times and timestamp <= times[-1]:
                    raise ValueError("Non-increasing timestamp")
                amplitude = np.hypot(pairs[0::2], pairs[1::2])
                invalid_flags = [
                    row[name].lower()
                    for name in ("first_word_invalid", "first_word")
                    if name in row
                ]
                if any(
                    flag not in {"0", "1", "false", "true"} for flag in invalid_flags
                ):
                    raise ValueError("Malformed first-word validity flag")
                invalid_flags = [flag in {"1", "true"} for flag in invalid_flags]
                if invalid_flags and any(
                    flag != invalid_flags[0] for flag in invalid_flags
                ):
                    raise ValueError(
                        "Conflicting first_word and first_word_invalid flags"
                    )
                if invalid_flags and invalid_flags[0]:
                    amplitude = amplitude.copy()
                    amplitude[:2] = np.nan
                if signals and amplitude.shape != signals[0].shape:
                    raise ValueError("Mixed PHY payload lengths")
                signals.append(amplitude)
                times.append(timestamp)
                previous = raw_time
                wrap = candidate_wrap
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                rejected += 1
    if not headers_seen:
        raise ValueError(
            f"No CSI CSV header found; record a header containing {timestamp_field!r} and {data_field!r}"
        )
    if not signals:
        raise ValueError(
            f"No usable CSI_DATA rows; unsupported signed-eight-bit payload rows: {unsupported_payloads}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        csi=np.stack(signals)[:, None, :, None],
        timestamps=np.asarray(times) - times[0],
    )
    report = {
        "mode": "hardware_signal_only",
        "frames": len(signals),
        "rejected_rows": rejected,
        "ignored_log_lines": ignored_lines,
        "headers_seen": headers_seen,
        "unsupported_payload_rows": unsupported_payloads,
        "channels": 1,
        "subcarriers": len(signals[0]),
        "pair_order": "Espressif signed I/Q pairs; magnitude is independent of pair order",
        "payload_format": "integer pairs in [-128,127] only; wider values, including out-of-range gain-compensated output, are rejected without clipping",
        "clock": "receiver local_timestamp microseconds; uint32 wraps corrected, not synchronized across receivers",
        "pose_inference": "Unavailable until a separately calibrated, labeled ESP32-layout model is trained and validated",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def record_serial(port, output, *, duration=60.0, baud=921600, consent_note):
    if not consent_note.strip() or not 0 < duration <= 3600:
        raise ValueError("Provide a consent note and duration in (0,3600] seconds")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite recording {output}")
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "Install the free hardware extra: pip install '.[hardware]'"
        ) from exc
    output.mkdir(parents=True)
    count = 0
    # Raw bytes preserve firmware startup lines and malformed rows for later audit.
    with (
        serial.Serial(port, baudrate=baud, timeout=0.25) as device,
        open(output / "serial.log", "xb") as stream,
    ):
        start = time.monotonic()
        while time.monotonic() - start < duration:
            line = device.readline()
            stream.write(line)
            count += len(line)
    metadata = {
        "mode": "hardware_signal_only",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "port": port,
        "baud": baud,
        "duration_seconds": duration,
        "bytes": count,
        "consent_note": consent_note,
        "note": "Raw serial capture only. Pass serial.log directly to esp-convert; capture the firmware CSV header. No pose model run.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
