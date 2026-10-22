#!/usr/bin/env python3
"""Create cluster-wise violin plots of individual WRIME sentiment scores."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


FONT_PATH = Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")
CLUSTER_ORDER = [
    "予算・財政運営",
    "外交・国際関係",
    "感染症・公衆衛生",
    "政治・政党・政治資金",
    "社会保障・子育て支援",
    "経済・物価・賃金",
    "調査・報告・検証",
    "防衛・安全保障",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/speeches_sample_200_wrime.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/wrime_sentiment_violinplot_by_cluster.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[float]] = defaultdict(list)
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["cluster_label"]].append(float(row["wrime_sentiment"]))

    labels = [label for label in CLUSTER_ORDER if label in grouped]
    values = [grouped[label] for label in labels]
    positions = list(range(1, len(labels) + 1))
    font = FontProperties(fname=str(FONT_PATH))

    fig, ax = plt.subplots(figsize=(12, 8), dpi=200, layout="constrained")
    violin = ax.violinplot(
        values,
        positions=positions,
        orientation="horizontal",
        widths=0.82,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        quantiles=[[0.25, 0.75] for _ in values],
        bw_method="scott",
    )
    for body in violin["bodies"]:
        body.set_facecolor("#7db5d8")
        body.set_edgecolor("#216fa6")
        body.set_linewidth(1.4)
        body.set_alpha(0.9)
    violin["cmedians"].set_color("#082f55")
    violin["cmedians"].set_linewidth(2.2)
    violin["cquantiles"].set_color("#216fa6")
    violin["cquantiles"].set_linewidth(1.5)

    ax.axvline(0, color="#172033", linewidth=1.5, linestyle=":", label="中立 0")
    ax.set_yticks(positions, [f"{label}（n={len(grouped[label])}）" for label in labels], fontproperties=font, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-0.75, 0.70)
    ax.set_title("8クラスタ別の個別感情極性スコア分布", fontproperties=font, fontsize=21, pad=18)
    ax.set_xlabel("感情極性スコア（−1：否定的、+1：肯定的）", fontproperties=font, fontsize=13, labelpad=10)
    ax.set_ylabel("クラスタ", fontproperties=font, fontsize=13, labelpad=10)
    ax.grid(axis="x", color="#dce6ef", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(prop=font, frameon=False, loc="lower right")
    ax.text(
        0.01,
        0.015,
        "濃線：中央値　細線：第1・第3四分位",
        transform=ax.transAxes,
        fontproperties=font,
        fontsize=10,
        color="#475569",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
