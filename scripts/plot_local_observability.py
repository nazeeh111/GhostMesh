"""Bind three completed source-only experiments to a compact scientific figure.

The extraction path reads only completed result/verification JSONs, never raw
data. Rendering a shipped figure-data.json needs only Matplotlib. No fits run.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

EXPECTED = {
    "domain-same-condition-v1": {
        "results.json": "7d0d385edee477d0a4b958ba77f0c50d94a49474f82d399c44683e73388307d7",
        "registration.json": "3a014927d3750165569f4882b06a68859f16c2771a8883386e802a6d4b8b00f7",
        "verification.json": "e3c48097f90d3b690b4afd6d33849bfe630266f495b2b0dbee87dc2acb864ad3",
    },
    "domain-observation-positive-v1": {
        "results.json": "5447e3214eff2afc824d11f0087ddfc5928d7e8858b1cf0fdb81d7d8328a4333",
        "registration.json": "79250abf5f675dc792e484c543608bc5cc71c5912562a52a988e09f4bbb2ce4c",
        "verification.json": "fe817e55c54d3a41817526c849ee80db7700d0b7131c7c99e047ba60d4d86663",
    },
    "domain-meta-e02-v1": {
        "results.json": "4832be23e3b37de23fc9f1f8ce7cce8795831de57f95b7d884a811aba293905c",
        "registration.json": "cc27669562ab98cbddc9f00280c0f23a42b48e0162ab6c414adac7fdaaf44818",
        "verification.json": "12d57678176b811719c87f50704c07b3b134ae8a76b5c7cea4cca0aaaa0290ec",
    },
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def row(rows, **criteria):
    selected = [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]
    require(len(selected) == 1, f"Expected one result matching {criteria}")
    return selected[0]


def summary(value):
    if "metrics" in value:
        metrics = value["metrics"]
        motion = value["motion"]["relative"]["within_record_r2"]
    else:
        metrics = value
        motion = value["relative_motion_r2"]
    return dict(
        nje=metrics["normalized_joint_error"],
        pck_05=metrics["pck_curve"]["0.05"],
        pck_20=metrics["pck_curve"]["0.2"],
        relative_motion_r2=motion,
        windows=metrics["valid_frames"],
    )


def extract(experiments):
    evidence = {}
    for name, bindings in EXPECTED.items():
        evidence[name] = {}
        for filename, expected in bindings.items():
            path = experiments / name / filename
            require(
                digest(path) == expected, f"Frozen evidence changed: {name}/{filename}"
            )
            evidence[name][filename] = json.loads(path.read_text())
    local = evidence["domain-same-condition-v1"]
    compressed = evidence["domain-observation-positive-v1"]
    meta = evidence["domain-meta-e02-v1"]
    lv, cv, mv = (e["verification.json"] for e in (local, compressed, meta))
    require(
        lv["models_verified"] == 8 and lv["raw_supports_disjoint"],
        "Local verification failed",
    )
    require(lv["maximum_saved_prediction_difference"] == 0, "Local replay mismatch")
    require(
        cv["models_verified"] == 4 and cv["maximum_saved_prediction_difference"] == 0,
        "Summary replay mismatch",
    )
    require(
        all(
            v == 0
            for arm in mv["saved_model_prediction_parity"].values()
            for v in arm.values()
        ),
        "Neural replay mismatch",
    )
    lrows, crows = (
        local["results.json"]["results"],
        compressed["results.json"]["results"],
    )
    quotient = row(lrows, protocol="held_time", feature="quotient", kernel="rbf")
    raw = row(lrows, protocol="held_time", feature="raw_db", kernel="rbf")
    full = row(crows, feature="observation_full", kernel="rbf")
    amp = row(crows, feature="observation_amplitude_zero_phase", kernel="rbf")
    mrows = meta["results.json"]["results"]
    room = {}
    for name in ("source_mean", "target_mean", "raw_ridge"):
        room[name] = summary(mrows["controls"][name]["all"])
    for arm in ("erm", "meta"):
        room[arm + "_zero"] = summary(mrows[arm]["zero_common"])
        room[arm + "_120"] = summary(mrows[arm]["adapted"]["all"])
        room[arm + "_120"]["within_record_shuffle_nje"] = mrows[arm]["adapted"]["all"][
            "within_record_shuffle_nje"
        ]
    return dict(
        schema="ghostmesh-local-observability-figure-v1",
        scope="SOURCE DEVELOPMENT only. Local A12 held time; failed A14 action transfer and E02 person-disjoint room calibration. No E04 results.",
        metrics=dict(
            nje="Mean joint error / per-frame teacher bounding-box diagonal; lower is better",
            pck="Fraction of joints inside the stated fraction of that diagonal",
            relative_motion_r2="Hip-root-relative articulation: 1 - centered prediction error energy / centered teacher energy. Predictions and teachers centered independently within each recording for scoring only; higher is better; a static prediction gives0",
        ),
        local=dict(
            record="E01/S01/A12",
            raw_blocks_half_open=dict(
                train=[0, 118], validation=[126, 178], test=[186, 297]
            ),
            context_frames=8,
            counts=dict(train=111, validation=45, test=104),
            reference_span_seconds=11.8,
            earliest_test_seconds_after_last_training_endpoint=7.6,
            quotient_rbf=summary(quotient["test"]),
            raw_db_rbf=summary(raw["test"]),
            quotient_shuffle=summary(quotient["test_controls"]["prediction_shuffle"]),
            quotient_past_lag_permutation=summary(
                quotient["test_controls"]["past_lag_permutation"]
            ),
            reference_mean=summary(
                row(
                    lrows,
                    protocol="held_time",
                    feature="quotient",
                    kernel="training_reference_mean",
                )["test"]
            ),
        ),
        compression=dict(
            record="E01/S01/A12",
            full_rbf=summary(full["test"]),
            amplitude_rbf=summary(amp["test"]),
            full_shuffle=summary(full["test_controls"]["prediction_shuffle"]),
            full_past_lag_permutation=summary(
                full["test_controls"]["past_lag_permutation"]
            ),
        ),
        held_action=dict(
            train=["E01/S01/A03", "E01/S01/A07"],
            validation="E01/S01/A12",
            test="E01/S01/A14",
            counts=dict(train=580, validation=290, test=290),
            raw_db_rbf=summary(
                row(lrows, protocol="held_action", feature="raw_db", kernel="rbf")[
                    "test"
                ]
            ),
            last_reference=summary(
                row(
                    lrows,
                    protocol="held_action",
                    feature="raw_db",
                    kernel="last_training_reference",
                )["test"]
            ),
        ),
        room=dict(
            room="E02",
            seed=2027,
            references=120,
            reference_people=["S11", "S12"],
            query_people=[f"S{i:02}" for i in range(13, 21)],
            query_windows=7992,
            source_train_people=16,
            source_validation_people=4,
            selection="Equal-room source-validation NJE averaged over zero-shot/adapted modes",
            results=room,
        ),
        verification=dict(
            local_models=8,
            local_max_prediction_difference=lv["maximum_saved_prediction_difference"],
            local_max_normal_equation_relative_residual=lv[
                "maximum_recomputed_normal_equation_relative_residual"
            ],
            compressed_models=4,
            compressed_max_prediction_difference=cv[
                "maximum_saved_prediction_difference"
            ],
            compressed_max_normal_equation_relative_residual=cv[
                "maximum_normal_equation_relative_residual"
            ],
            neural_saved_prediction_max_abs=mv["saved_model_prediction_parity"],
        ),
        evidence_sha256=EXPECTED,
        executed_sources_sha256={
            name: e["registration.json"].get(
                "sources_sha256", e["registration.json"].get("source_sha256")
            )
            for name, e in evidence.items()
        },
    )


def render(data, output):
    # Use Matplotlib's supported bundled-font mode. This also avoids optional
    # macOS system_profiler scans, whose font inventory can omit _items.
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MultipleLocator

    require(
        data["schema"] == "ghostmesh-local-observability-figure-v1",
        "Unknown figure schema",
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "text.color": "#172b3b",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(16.8, 5.8), gridspec_kw={"width_ratios": [1, 1, 1.12]}
    )
    fig.subplots_adjust(left=0.115, right=0.97, bottom=0.22, top=0.71, wspace=0.70)
    fig.text(
        0.035,
        0.94,
        "GhostMesh  /  Local observability and transfer",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.035,
        0.875,
        "CSI predicts later motion in one condition. Room transfer remains unresolved.",
        fontsize=13,
    )
    fig.text(
        0.035,
        0.815,
        "Source-development evidence  ·  Supplied MM-Fi 2D teacher labels  ·  No reserved E04 evaluation",
        fontsize=10,
        color="#526578",
    )
    teal, orange, gray, blue = "#087e83", "#c76a30", "#7b8995", "#456c9a"
    local, compressed, room = (
        data["local"],
        data["compression"],
        data["room"]["results"],
    )
    panels = [
        (
            axes[0],
            "A   Later time, same condition",
            [
                "Aligned quotient",
                "Shuffled alignment",
                "Past lags permuted",
                "Static reference mean",
            ],
            [
                local["quotient_rbf"]["relative_motion_r2"],
                local["quotient_shuffle"]["relative_motion_r2"],
                local["quotient_past_lag_permutation"]["relative_motion_r2"],
                local["reference_mean"]["relative_motion_r2"],
            ],
            [teal, orange, blue, gray],
        ),
        (
            axes[1],
            "B   Summary retains motion",
            [
                "Full summary",
                "Amplitude only",
                "Full, shuffled",
                "Static reference mean",
            ],
            [
                compressed["full_rbf"]["relative_motion_r2"],
                compressed["amplitude_rbf"]["relative_motion_r2"],
                compressed["full_shuffle"]["relative_motion_r2"],
                local["reference_mean"]["relative_motion_r2"],
            ],
            [teal, blue, orange, gray],
        ),
    ]
    for ax, title, labels, values, colors in panels:
        y = list(range(len(values)))
        ax.hlines(y, 0, values, colors=colors, linewidth=2)
        ax.scatter(values, y, c=colors, s=58, zorder=3)
        ax.set_yticks(y, labels)
        ax.set_xlim(-1.17, 1.18)
        ax.set_ylim(3.55, -0.55)
        ax.set_xlabel("Relative motion R²  ↑", labelpad=10)
        ax.set_title(title, loc="left", pad=18, fontsize=11)
        ax.axvline(0, color="#b4c0ca", linewidth=0.9)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        for yi, value in zip(y, values):
            ax.annotate(
                f"{value:.3f}",
                (value, yi),
                xytext=(5 if value >= 0 else -5, 9),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                fontsize=9,
            )
        ax.text(
            0,
            -0.30,
            "E01/S01/A12 · 111 refs / 104 test",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#526578",
        )
    ax = axes[2]
    labels = [
        "Source pose mean",
        "Target reference mean",
        "Raw ridge · 120 refs",
        "ERM · zero-shot",
        "ERM · 120 refs",
        "Meta · zero-shot",
        "Meta · 120 refs",
    ]
    keys = [
        "source_mean",
        "target_mean",
        "raw_ridge",
        "erm_zero",
        "erm_120",
        "meta_zero",
        "meta_120",
    ]
    values = [room[key]["nje"] for key in keys]
    colors = [teal, gray, gray, blue, blue, orange, orange]
    y = list(range(len(values)))
    ax.hlines(y, 0, values, colors=colors, linewidth=2)
    ax.scatter(values, y, c=colors, s=45, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.17)
    ax.set_ylim(6.65, -0.65)
    ax.set_xlabel("Normalized joint error  ↓", labelpad=10)
    ax.set_title("C   Held room: controls still win", loc="left", pad=18, fontsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    for yi, value in zip(y, values):
        ax.annotate(
            f"{value:.3f}",
            (value, yi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.text(
        0,
        -0.30,
        "E02 · 8 people / 7,992 query windows",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#526578",
    )
    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#ccd4dc")
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", color="#ccd4dc")
        ax.grid(axis="x", color="#e6ebef", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.text(
        0.035,
        0.04,
        "Local panels reuse one participant/action with dense references; test windows overlap each other. No population confidence intervals are implied.",
        fontsize=9,
        color="#526578",
    )
    for extension in ("png", "svg"):
        fig.savefig(
            output / f"local-observability.{extension}",
            dpi=180,
            facecolor="white",
            metadata={"Creator": "GhostMesh scientific reporting"}
            if extension == "svg"
            else {"Software": "GhostMesh scientific reporting"},
        )
        if extension == "svg":
            path = output / "local-observability.svg"
            # Matplotlib emits spaces before SVG path line breaks. Canonicalize
            # those separators without changing any drawing coordinates.
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiments", type=Path)
    source.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh output directory to preserve prior renders")
    data = (
        extract(args.experiments)
        if args.experiments
        else json.loads(args.data.read_text())
    )
    args.out.mkdir(parents=True)
    (args.out / "figure-data.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    render(data, args.out)
    artifacts = {p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()}
    artifacts["plot_script_sha256"] = digest(Path(__file__))
    (args.out / "artifact-hashes.json").write_text(
        json.dumps(artifacts, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
