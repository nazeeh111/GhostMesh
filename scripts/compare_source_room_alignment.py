"""TRAIN-only timing comparison, two SOURCE rooms, no held-room values."""

import argparse
from pathlib import Path
import sys
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_source_alignment import (  # noqa:E402
    radio,
    rms_motion,
    pearson,
    average,
    select_lag,
)
from scripts.verify_geometric_distillation import read, digest, bind, require  # noqa:E402
from scripts.build_moment_cache import save  # noqa:E402


def run(raw, cache, out):
    RAW, OUT = Path(raw), Path(out)
    require(not OUT.exists(), "Existing diagnostic preserved")
    receipt = read(RAW / "receipt.json")
    cr = read(cache / "receipt.json")
    bind(cache, cr["arrays_sha256"])
    require(
        digest(RAW / "rf.npy") == receipt["arrays"]["rf"]
        and digest(RAW / "pose.npy") == receipt["arrays"]["pose"],
        "Actual typed input bytes differ",
    )
    quality = {r["record"]: r for r in read(cache / "quality.json")}
    started = time.monotonic()
    OUT.mkdir(parents=True)
    reg = dict(
        status="sourceTRAIN-only timingcomparison; previous project development exposure disclosed",
        raw_receipt_sha256=digest(RAW / "receipt.json"),
        quality_receipt_sha256=digest(cache / "receipt.json"),
        source_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in [Path(__file__), ROOT / "scripts/diagnose_source_alignment.py"]
        },
        rooms=["E01", "E02"],
        excluded_people=["S04", "S09", "S14", "S19"],
        lags=list(range(-24, 25)),
        widths=[1, 4, 8, 16, 32],
        scope="TRAIN16sourcepeople/all27actions,noE03/E04values,no neuralfit. Prior E01fixed4delayed outputs failedE02; do notchangeoldoutputs/gates",
        selection="MeanwithinrecordTRAINPearsonmax,smallerabsLag/lowerlag ties; reportroom/width curves separately, no clockcorrection claim",
    )
    save(OUT / "registration.json", reg)
    rf = np.load(RAW / "rf.npy", mmap_mode="r", allow_pickle=False)
    pose = np.load(RAW / "pose.npy", mmap_mode="r", allow_pickle=False)
    records = []
    for row in receipt["records"]:
        name = row["id"]
        room, person, action = name.split("/")
        if room not in reg["rooms"] or person in reg["excluded_people"]:
            continue
        start = row["offset"]
        y = np.asarray(pose[start : start + 297], dtype=float)
        y -= y[:, [11, 12]].mean(1, keepdims=True)
        signals = radio(np.asarray(rf[start : start + 297]))
        valid = np.ones(297, bool)
        valid[quality[name]["rejected_frames"]] = False
        ends = np.array(
            [e for e in range(63, 273) if valid[e - 39 : e + 1].all()], dtype=np.int64
        )
        measured = {}
        for key, x in signals.items():
            for width in reg["widths"]:
                r, t = rms_motion(x, width), rms_motion(y, width)
                measured[f"{key}_width{width}"] = {
                    str(lag): pearson(r[ends], t[ends + lag]) for lag in reg["lags"]
                }
        records.append(
            dict(
                record=name,
                room=room,
                person=person,
                action=action,
                correlations=measured,
                endpoints=len(ends),
            )
        )
    summary = {}
    for room in reg["rooms"]:
        summary[room] = {}
        for key in records[0]["correlations"]:
            curve = {
                str(lag): dict(
                    train=average(
                        [
                            r["correlations"][key][str(lag)]
                            for r in records
                            if r["room"] == room
                        ]
                    )
                )
                for lag in reg["lags"]
            }
            selected = select_lag(curve)
            summary[room][key] = dict(
                all_lags=curve,
                selected_lag=selected,
                selected_correlation=curve[str(selected)]["train"]
                if selected is not None
                else None,
            )
    bind(ROOT, reg["source_sha256"])
    save(OUT / "all-records.json", records)
    save(
        OUT / "summary.json",
        dict(
            registration=reg,
            summary=summary,
            records=len(records),
            seconds=time.monotonic() - started,
        ),
    )
    print(
        json.dumps(
            {
                room: {
                    key: (v["selected_lag"], v["selected_correlation"])
                    for key, v in rows.items()
                }
                for room, rows in summary.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "cache", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.raw, args.cache, args.out)
