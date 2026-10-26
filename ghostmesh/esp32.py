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
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    signals = []
    times = []
    wrap = 0
    previous = None
    rejected = 0
    with open(source, newline="") as stream:
        reader = csv.DictReader(stream)
        if not {timestamp_field, data_field} <= set(reader.fieldnames or []):
            raise ValueError(
                "CSV needs a header with local_timestamp and data (or explicit field overrides)"
            )
        for row in reader:
            if row.get("type", "CSI_DATA") != "CSI_DATA":
                continue
            try:
                pairs = np.asarray(json.loads(row[data_field]), dtype=np.float32)
                raw_time = int(row[timestamp_field])
                if not 0 <= raw_time < 2**32:
                    raise ValueError("Timestamp outside uint32 range")
                if (
                    pairs.ndim != 1
                    or len(pairs) % 2
                    or len(pairs) < 4
                    or not np.isfinite(pairs).all()
                    or ((pairs < -128) | (pairs > 127)).any()
                ):
                    raise ValueError("Malformed signed-byte I/Q payload")
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
                if row.get("first_word_invalid", "0") in {"1", "true"}:
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
    if not signals:
        raise ValueError("No usable CSI_DATA rows")
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
        "channels": 1,
        "subcarriers": len(signals[0]),
        "pair_order": "Espressif signed I/Q pairs; magnitude is independent of pair order",
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
        "note": "Raw serial capture only. Select the actual CSV header/CSI rows for esp-convert; no pose model run.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
