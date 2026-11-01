"""Source-only RF/annotation motion alignment; not a pose efficacy evaluation.

TRAIN people select offsets; source VAL people only report them. Correlation
cannot establish physical clock error or pose reconstruction. No 3D input,
model fitting, target-room annotation decoding or image decoding is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_moment_cache import digest, require, save  # noqa:E402
from scripts.verify_geometric_distillation import read, bind  # noqa:E402

WINDOWS = (1, 4, 8, 16, 32)
LAGS = tuple(range(-24, 25))
SHIFTS = (41, 59, 83, 107, 131, 157, 181)


def pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(
        a.ndim == b.ndim == 1 and a.shape == b.shape,
        "Matching scalar motion traces required",
    )
    if len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    a = a - a[0]
    b = b - b[0]
    a -= a.mean()
    b -= b.mean()
    aa = np.square(a).sum()
    bb = np.square(b).sum()
    return (
        float(np.einsum("i,i->", a, b, optimize=False) / np.sqrt(aa * bb))
        if aa > 0 and bb > 0
        else None
    )


def rms_motion(x, width):
    x = np.asarray(x, dtype=float)
    require(
        type(width) is int and 0 < width < len(x),
        "Positive within-record motion lag required",
    )
    out = np.full(len(x), np.nan)
    out[width:] = np.sqrt(
        np.square(x[width:] - x[:-width]).reshape(len(x) - width, -1).mean(1)
    )
    return out


def average(values):
    v = [value for value in values if value is not None]
    return float(np.mean(v)) if v else None


def select_lag(curve):
    usable = [int(k) for k, v in curve.items() if v["train"] is not None]
    return (
        min(usable, key=lambda lag: (-curve[str(lag)]["train"], abs(lag), lag))
        if usable
        else None
    )


def radio(raw):
    # Only RF-invalid frames are neutral-filled; full forty-frame guards later
    # exclude them at every evaluated endpoint. No annotation enters this step.
    x = np.asarray(raw, dtype=float)
    require(x.shape == (297, 6, 114, 10), "One complete typed RF record required")
    safe = x[:, :3].copy()
    for frame in safe:
        for column in range(10):
            v = frame[:, :, column]
            valid = np.isfinite(v)
            v[~valid] = v[valid].mean() if valid.any() else 0
    a, ph = x[:, :3], x[:, 3:]
    finite = np.isfinite(a) & np.isfinite(ph)
    good = finite[:, 1:] & finite[:, :1]
    a = np.where(finite, a, 0)
    ph = np.where(finite, ph, 0)
    phase = ph[:, 1:] - ph[:, :1]
    quotient = np.concatenate(
        (
            np.where(good, np.clip((a[:, 1:] - a[:, :1]) / 20, -2, 2), 0),
            np.where(good, np.cos(phase), 0),
            np.where(good, np.sin(phase), 0),
        ),
        axis=1,
    )
    return {"raw_log_magnitude": safe, "interreceiver_log_circular": quotient}


def run(raw, quality_cache, out):
    require(
        not out.exists()
        and not any(
            out.resolve().is_relative_to(p.resolve()) for p in [raw, quality_cache]
        ),
        "Existing/nested diagnostic output preserved",
    )
    receipt = read(raw / "receipt.json")
    qc = read(quality_cache / "receipt.json")
    qr = read(quality_cache / "registration.json")
    require(
        qc["kind"] == qr["kind"] == "E01-label-free-delayed-moments"
        and qc["annotation_arrays_decoded"] is False
        and qr["raw_receipt_sha256"] == digest(raw / "receipt.json")
        and qc["registration_sha256"] == digest(quality_cache / "registration.json"),
        "RF-only source quality provenance differs",
    )
    bind(quality_cache, qc["arrays_sha256"])
    for name in ["rf", "pose"]:
        require(
            digest(raw / f"{name}.npy") == receipt["arrays"][name],
            "Raw input bytes differ",
        )
    require(
        receipt["config"]["reference_canvas"] == [640, 640]
        and receipt["config"]["magnitude_units"] == "log_magnitude_db"
        and receipt["config"]["phase_units"] == "radians",
        "Typed RF/annotation plane required",
    )
    quality = {r["record"]: r for r in read(quality_cache / "quality.json")}
    rows = [r for r in receipt["records"] if r["id"].startswith("E01/")]
    require(
        len(rows) == 270 and len({r["id"] for r in rows}) == 270,
        "Complete E01 source scope required",
    )
    registration = dict(
        kind="E01-source-motion-clock-diagnostic",
        source_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                ROOT / "scripts/build_moment_cache.py",
                ROOT / "scripts/verify_geometric_distillation.py",
            ]
        },
        raw_receipt_sha256=digest(raw / "receipt.json"),
        quality_receipt_sha256=digest(quality_cache / "receipt.json"),
        windows=list(WINDOWS),
        lags=list(LAGS),
        circular_shifts=list(SHIFTS),
        source_validation_people=["S04", "S09"],
        endpoint_grid="63..272 inclusive, same forty-frame RF-only guard at everylag/window",
        selection="Macro TRAIN-record Pearson maximum, smallerabsLag/lowerlag ties; VAL never selects",
        annotation_arrays_decoded="E01 2D only, noE03/E04/3D/images",
        limits="Repeated source development, not untouchedfinal. Circular nulls compress retained-time gaps and are not IID significance tests. Autocorrelation uses paired originalframeindices. Correlation is not physical clock calibration or joint reconstruction.",
    )
    out.mkdir(parents=True)
    save(out / "registration.json", registration)
    rf = np.load(raw / "rf.npy", mmap_mode="r", allow_pickle=False)
    pose = np.load(raw / "pose.npy", mmap_mode="r", allow_pickle=False)
    records = []
    started = time.monotonic()
    for record in rows:
        name = record["id"]
        start = record["offset"]
        y = np.asarray(pose[start : start + 297], dtype=float)
        require(
            y.shape == (297, 17, 2) and np.isfinite(y).all(),
            "Finite source teacher record required",
        )
        y -= y[:, [11, 12]].mean(1, keepdims=True)
        signals = radio(np.asarray(rf[start : start + 297]))
        valid = np.ones(297, bool)
        valid[quality[name]["rejected_frames"]] = False
        ends = np.array(
            [e for e in range(63, 273) if valid[e - 39 : e + 1].all()], dtype=np.int64
        )
        person = name.split("/")[1]
        role = "validation" if person in ["S04", "S09"] else "train"
        correlations = {}
        null = {}
        autocorrelation = {}
        for rname, value in signals.items():
            for width in WINDOWS:
                key = f"{rname}_width{width}"
                r = rms_motion(value, width)
                target = rms_motion(y, width)
                correlations[key] = {
                    str(lag): pearson(r[ends], target[ends + lag]) for lag in LAGS
                }
                null[key] = {
                    str(lag): [
                        pearson(np.roll(r[ends], shift), target[ends + lag])
                        for shift in SHIFTS
                    ]
                    for lag in LAGS
                }
                lookup = set(map(int, ends))
                autocorrelation[key] = {}
                for lag in [1, 2, 4, 8, 16, 24]:
                    first = np.array(
                        [e for e in ends if int(e + lag) in lookup], dtype=np.int64
                    )
                    autocorrelation[key][str(lag)] = dict(
                        RF=pearson(r[first], r[first + lag]),
                        teacher_motion=pearson(target[first], target[first + lag]),
                        original_frame_pairs=len(first),
                    )
        records.append(
            dict(
                record=name,
                person=person,
                action=name.split("/")[-1],
                role=role,
                endpoints=len(ends),
                correlations=correlations,
                shift_controls=null,
                autocorrelation=autocorrelation,
            )
        )
    save(out / "all-records.json", records)
    summary = {}
    for key in records[0]["correlations"]:
        curve = {
            str(lag): {
                role: average(
                    [
                        r["correlations"][key][str(lag)]
                        for r in records
                        if r["role"] == role
                    ]
                )
                for role in ["train", "validation"]
            }
            for lag in LAGS
        }
        selected = select_lag(curve)
        row = dict(all_lags=curve, selected_train_lag=selected)
        if selected is not None:
            row["selected"] = curve[str(selected)]
            row["validation_per_person"] = {
                person: average(
                    [
                        r["correlations"][key][str(selected)]
                        for r in records
                        if r["person"] == person
                    ]
                )
                for person in ["S04", "S09"]
            }
            row["train_per_action"] = {
                f"A{action:02}": average(
                    [
                        r["correlations"][key][str(selected)]
                        for r in records
                        if r["role"] == "train" and r["action"] == f"A{action:02}"
                    ]
                )
                for action in range(1, 28)
            }
            row["shift_control_means"] = {
                role: [
                    average(
                        [
                            r["shift_controls"][key][str(selected)][i]
                            for r in records
                            if r["role"] == role
                        ]
                    )
                    for i in range(len(SHIFTS))
                ]
                for role in ["train", "validation"]
            }
        row["autocorrelation"] = {
            str(lag): {
                role: {
                    kind: average(
                        [
                            r["autocorrelation"][key][str(lag)][kind]
                            for r in records
                            if r["role"] == role
                        ]
                    )
                    for kind in ["RF", "teacher_motion"]
                }
                for role in ["train", "validation"]
            }
            for lag in [1, 2, 4, 8, 16, 24]
        }
        summary[key] = row
    bind(ROOT, registration["source_sha256"])
    result = dict(
        registration=registration,
        records=len(records),
        train_records=sum(r["role"] == "train" for r in records),
        validation_records=sum(r["role"] == "validation" for r in records),
        summary=summary,
        seconds=time.monotonic() - started,
    )
    save(out / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "quality-cache", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    r = run(args.raw, args.quality_cache, args.out)
    print(
        json.dumps(
            dict(
                records=r["records"],
                seconds=r["seconds"],
                selected={k: v.get("selected") for k, v in r["summary"].items()},
            )
        ),
        flush=True,
    )
