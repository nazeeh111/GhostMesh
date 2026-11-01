"""Render a source-bound static figure from the complete published source report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa:E402
import numpy as np  # noqa:E402

ROOT = Path(__file__).resolve().parents[1]


def render(report, out):
    if out.exists():
        raise ValueError("Existing figure preserved; choose a fresh directory")
    r = json.loads(report.read_text())
    fig, axes = plt.subplots(
        1, 3, figsize=(13.4, 5.3), gridspec_kw={"width_ratios": [1.45, 1, 1]}
    )
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    plt.rcParams["svg.hashsalt"] = "ghostmesh-source-timing-transfer"
    fig.patch.set_facecolor("#f8fafc")
    teal, amber, ink = "#087f8c", "#bd6b11", "#172c40"
    for ax in axes:
        ax.set_facecolor("#f8fafc")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c4ccd6")
        ax.tick_params(colors=ink, labelsize=9)
        ax.grid(axis="y", color="#dfe6ec", linewidth=0.7)
        ax.set_axisbelow(True)
    timing = r["timing"]["results"]["summary"]["interreceiver_log_circular_width4"]
    lags = np.array(sorted(map(int, timing["all_lags"])))
    for role, label, color in [
        ("train", "Training people", ink),
        ("validation", "Validation people", teal),
    ]:
        axes[0].plot(
            lags,
            [timing["all_lags"][str(k)][role] for k in lags],
            label=label,
            color=color,
            lw=2,
        )
    axes[0].axvline(timing["selected_train_lag"], color=amber, linestyle="--", lw=1.3)
    axes[0].axhline(0, color="#c4ccd6", lw=0.7)
    axes[0].set(
        title="A  Source timing pattern",
        xlabel="Annotation offset (frames)",
        ylabel="Mean record Pearson correlation",
        xlim=(-24, 24),
        ylim=(-0.25, 0.5),
    )
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].text(
        0.02,
        0.96,
        "−4 selected on training people",
        transform=axes[0].transAxes,
        va="top",
        color=amber,
        fontsize=9,
    )
    for ax, key, title, cal_label in zip(
        axes[1:],
        ["cross_person", "cross_room"],
        ["B  One source room", "C  Two source rooms"],
        ["120 refs", "120 refs/room"],
    ):
        result = r[key]["results"]
        primary = key
        values = [
            result["zero"][primary]["pooled"]["nje"],
            result["zero"]["source_mean"]["pooled"]["nje"],
            result["calibrated"][primary]["pooled"]["nje"],
            result["calibrated"]["reference_mean"]["pooled"]["nje"],
        ]
        ax.bar([0, 1, 3, 4], values, width=0.82, color=[teal, amber, teal, amber])
        ax.set_xticks([0.5, 3.5], ["Zero", cal_label])
        ax.set_ylim(0, 0.092)
        ax.set_title(title)
        ax.set_ylabel("Normalized joint error ↓")
        for x, value in zip([0, 1, 3, 4], values):
            ax.text(
                x, value + 0.002, f"{value:.4f}", ha="center", fontsize=8, color=ink
            )
        ax.text(
            0.5,
            0.95,
            "BOTH GATES FAIL",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#9e3b28",
            weight="bold",
        )
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(color=teal, label="Pose-aligned model"),
            Patch(color=amber, label="Static source/reference mean"),
        ],
        loc="lower right",
        bbox_to_anchor=(0.97, 0.16),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "A timing signal, but no transfer-qualified pose model",
        x=0.055,
        y=0.97,
        ha="left",
        fontsize=17,
        color=ink,
        weight="bold",
    )
    fig.text(
        0.055,
        0.88,
        "Real held-person source validation • hip-relative 2D • complete results retain every candidate",
        fontsize=10,
        color="#4c6075",
    )
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.30, top=0.78, wspace=0.39)
    fig.text(
        0.055,
        0.12,
        "Timing is an empirical correlation, not physical clock calibration. Decoder buffer: nominal 400 ms.",
        fontsize=9,
        color=ink,
    )
    fig.text(
        0.055,
        0.075,
        "Reused development people; no new E03/E04 scores. Lower error is better; no default-model promotion.",
        fontsize=9,
        color=ink,
    )
    out.mkdir(parents=True)
    for suffix in ["png", "svg"]:
        fig.savefig(
            out / f"source-timing-transfer.{suffix}",
            dpi=160,
            facecolor=fig.get_facecolor(),
        )
    svg = out / "source-timing-transfer.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
    provenance = dict(
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
        },
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/domain-v2-timing-transfer-results.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.report, args.out)))
