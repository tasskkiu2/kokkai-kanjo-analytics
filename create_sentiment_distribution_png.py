#!/usr/bin/env python3
"""Plot the distribution of individual WRIME sentiment scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


FONT_PATH = Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/speeches_sample_200_wrime.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/wrime_sentiment_distribution.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        scores = np.array([float(row["wrime_sentiment"]) for row in csv.DictReader(handle)])

    font = FontProperties(fname=str(FONT_PATH))
    mean = float(scores.mean())
    median = float(np.median(scores))
    bins = np.arange(-0.75, 0.701, 0.05)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200, layout="constrained")
    counts, _, _ = ax.hist(scores, bins=bins, color="#4a96c6", edgecolor="white", linewidth=1.2)
    ax.axvline(0, color="#172033", linewidth=1.5, linestyle=":", label="中立 0")
    ax.axvline(mean, color="#0b4f8a", linewidth=2.2, label=f"平均 {mean:+.3f}")
    ax.axvline(median, color="#277db5", linewidth=2.2, linestyle="--", label=f"中央値 {median:+.3f}")

    ax.set_xlim(-0.75, 0.70)
    ax.set_ylim(0, max(counts) * 1.12)
    ax.set_title("国会発言200件の個別感情極性スコア分布", fontproperties=font, fontsize=21, pad=18)
    ax.set_xlabel("感情極性スコア（−1：否定的、+1：肯定的）", fontproperties=font, fontsize=13, labelpad=10)
    ax.set_ylabel("発言数", fontproperties=font, fontsize=13, labelpad=10)
    ax.grid(axis="y", color="#dce6ef", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    legend = ax.legend(prop=font, frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_fontproperties(font)
    ax.text(
        0.01,
        0.96,
        f"負のスコア: {(scores < 0).sum()}件　正のスコア: {(scores > 0).sum()}件",
        transform=ax.transAxes,
        va="top",
        fontproperties=font,
        fontsize=11,
        color="#334155",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
