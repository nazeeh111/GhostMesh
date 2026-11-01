"""Plot measured frozen E02 scores; no model fitting or fabricated intervals."""

import argparse
import hashlib
import json
from pathlib import Path


def plot(report, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out.exists():
        raise FileExistsError(out)
    content = report.read_bytes()
    data = json.loads(content)
    if (
        data["audit"]["prediction_sets_verified"] != 89
        or data["audit"]["metric_findings"]
    ):
        raise ValueError("Expected the qualified89-set measured report")
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#f7f9fb")
    fig.suptitle(
        "Frozen CSI pretraining: transfer remains unresolved",
        x=0.06,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#162b3e",
    )
    fig.text(
        0.06,
        0.92,
        "Real MM-Fi development data · seed2027 · lower normalized joint error is better",
        color="#536576",
    )
    entries = [
        ("zero_full", "Zero-shot · full image", False),
        ("calibrated_full", "120-label calibration · full image", True),
        ("zero_body", "Zero-shot · hip-relative", False),
        ("calibrated_body", "120-label calibration · hip-relative", True),
    ]
    for ax, (protocol, title, calibrated) in zip(axes.flat, entries):
        methods = data["protocols"][protocol]["methods"]
        names = [
            "source_mean",
            *[
                arm + ("/residual_krr" if calibrated else "")
                for arm in (
                    "random_frozen",
                    "masked_temporal",
                    "supervised",
                    "raw_current",
                )
            ],
        ]
        labels = [
            "Static source mean",
            "Random encoder",
            "Masked + temporal",
            "Supervised encoder",
            "Raw-current control",
        ]
        values = [methods[name]["aligned"]["normalized_joint_error"] for name in names]
        colors = ["#6c7e90", "#afbbc6", "#007e83", "#afbbc6", "#afbbc6"]
        ax.set_facecolor("white")
        ax.barh(range(5), values, color=colors, height=0.62)
        ax.set_yticks(range(5), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 0.18 if protocol.endswith("full") else 0.095)
        for i, value in enumerate(values):
            ax.text(
                value + 0.001,
                i,
                f"{value:.5f}",
                va="center",
                fontsize=9,
                color="#203549",
            )
        ax.set_title(title, loc="left", pad=15, fontweight="bold", color="#162b3e")
        ax.set_xlabel("Normalized joint error")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#ccd4dd")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color="#e5eaf0", linewidth=0.6)
        ax.set_axisbelow(True)
        cohort = data["protocols"][protocol]
        ax.text(
            0,
            1.02,
            f"{len(cohort['people'])} people · {cohort['query_rows']:,} endpoints · gate FAILED",
            transform=ax.transAxes,
            fontsize=8,
            color="#963d39",
        )
    fig.text(
        0.06,
        0.035,
        "Calibration and zero-shot use different query cohorts. Compare bars within each panel.\nNo error bars: one development seed. Supplied 2D teacher labels; no 3D or final E04 result.",
        color="#536576",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.18, right=0.97, top=0.84, bottom=0.14, hspace=0.55, wspace=0.65
    )
    out.mkdir(parents=True)
    fig.savefig(out / "masked-transfer.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(out / "masked-transfer.svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "report_sha256": hashlib.sha256(content).hexdigest(),
                "plotter_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "data": "Actual independently verified E02 development scores, no fitted or simulated values",
                "files_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in out.iterdir()
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("docs/domain-v2-masked-results.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.out)
