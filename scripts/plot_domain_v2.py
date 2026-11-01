"""Render measured source-held domain results; never opens dataset arrays."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def render(source, output):
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(source.read_text())
    rooms = ["E01", "E02", "E03"]
    rows = {(row["room"], row["mode"]): row for row in data["results"]}
    prior = {row["room"]: row for row in data["static_priors"]}
    if len(rows) != 6 or len(prior) != 3:
        raise ValueError("Expected complete six-fit/three-prior table")
    output.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "ghostmesh-domain-v2",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    fig.patch.set_facecolor("#f7faf9")
    x = np.arange(3)
    colors = {"legacy": "#147d79", "quotient": "#d57d2a"}
    axes[0].bar(
        x - 0.25,
        [prior[r]["nje"] for r in rooms],
        0.23,
        color="#9ba9ae",
        label="Static source mean",
    )
    for offset, mode in [(0, "legacy"), (0.25, "quotient")]:
        values = [rows[r, mode]["metrics"]["nje"] for r in rooms]
        axes[0].bar(x + offset, values, 0.23, color=colors[mode], label=mode.title())
    for offset, mode in [(-0.13, "legacy"), (0.13, "quotient")]:
        values = [
            rows[r, mode]["motion"]["relative"]["within_record_r2"] for r in rooms
        ]
        axes[1].bar(x + offset, values, 0.24, color=colors[mode])
    for ax in axes:
        ax.set_facecolor("#f7faf9")
        ax.set_xticks(x, rooms)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.17)
    axes[0].set_ylabel("Normalized joint error ↓")
    axes[0].set_ylim(0, 0.175)
    axes[0].set_title("A static pose still has lower error", loc="left", fontsize=12)
    axes[1].axhline(0, color="#647078", lw=1)
    axes[1].set_ylabel("Within-record relative-motion R² ↑")
    axes[1].set_ylim(-0.16, 0.025)
    axes[1].set_title("Predicted motion does not transfer", loc="left", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.suptitle(
        "Signal correction alone does not solve room transfer",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#172d35",
    )
    fig.text(
        0.07,
        0.89,
        "E01–E03 development folds · seed 2027 · source-validation selection · no E04 scoring",
        color="#53666d",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.76, wspace=0.35)
    for suffix in ("svg", "png"):
        fig.savefig(
            output / f"domain-transfer.{suffix}",
            dpi=180,
            facecolor=fig.get_facecolor(),
            metadata={"Creator": "GhostMesh measured-results renderer"},
        )
        if suffix == "svg":
            path = output / "domain-transfer.svg"
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": sha256(source.read_bytes()).hexdigest(),
                "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "outputs": {
                    p.name: sha256(p.read_bytes()).hexdigest()
                    for p in sorted(output.iterdir())
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("docs/domain-v2-dense-rf-results.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.out)
