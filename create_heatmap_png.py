#!/usr/bin/env python3
"""Create PNG heatmaps from the topic-level WRIME summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties


EMOTION_COLUMNS = [
    ("mean_joy", "喜び"),
    ("mean_sadness", "悲しみ"),
    ("mean_anticipation", "期待"),
    ("mean_surprise", "驚き"),
    ("mean_anger", "怒り"),
    ("mean_fear", "恐れ"),
    ("mean_disgust", "嫌悪"),
    ("mean_trust", "信頼"),
]
FONT_PATH = Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/wrime_summary_by_cluster.csv"))
    parser.add_argument("--emotion-output", type=Path, default=Path("outputs/wrime_emotions_heatmap.png"))
    parser.add_argument("--sentiment-output", type=Path, default=Path("outputs/wrime_sentiment_heatmap.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    topics = [row["cluster_label"] for row in rows]
    emotion_values = np.array(
        [[float(row[column]) for column, _ in EMOTION_COLUMNS] for row in rows]
    )
    sentiment_values = np.array([[float(row["mean_sentiment"])] for row in rows])
    font = FontProperties(fname=str(FONT_PATH))

    emotion_fig, emotion_ax = plt.subplots(figsize=(15, 9), dpi=200, layout="constrained")
    emotion_image = emotion_ax.imshow(emotion_values, cmap="Blues", vmin=0, vmax=0.60, aspect="auto")
    emotion_ax.set_title("WRIMEによる国会発言200件の8感情", fontproperties=font, fontsize=21, pad=18)
    emotion_ax.set_xticks(range(len(EMOTION_COLUMNS)), [label for _, label in EMOTION_COLUMNS], fontproperties=font, fontsize=13)
    emotion_ax.set_yticks(range(len(topics)), topics, fontproperties=font, fontsize=12)
    emotion_ax.set_xlabel("感情", fontproperties=font, fontsize=13, labelpad=10)
    emotion_ax.set_ylabel("トピック（各25発言）", fontproperties=font, fontsize=13, labelpad=12)
    for row_index in range(len(topics)):
        for column_index in range(len(EMOTION_COLUMNS)):
            value = emotion_values[row_index, column_index]
            emotion_ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.36 else "#172033",
                fontsize=11,
                fontweight="bold",
            )
    emotion_colorbar = emotion_fig.colorbar(emotion_image, ax=emotion_ax, location="bottom", shrink=0.60, pad=0.08)
    emotion_colorbar.set_label("平均確率", fontproperties=font, fontsize=12)
    style_axis(emotion_ax, len(topics))
    args.emotion_output.parent.mkdir(parents=True, exist_ok=True)
    emotion_fig.savefig(args.emotion_output, bbox_inches="tight", facecolor="white")
    plt.close(emotion_fig)

    sentiment_fig, sentiment_ax = plt.subplots(figsize=(7, 9), dpi=200, layout="constrained")
    lower = min(-0.15, float(sentiment_values.min()))
    sentiment_image = sentiment_ax.imshow(
        sentiment_values, cmap="Blues_r", norm=Normalize(vmin=lower, vmax=0), aspect="auto"
    )
    sentiment_ax.set_title("WRIMEによる国会発言200件の感情極性", fontproperties=font, fontsize=19, pad=18)
    sentiment_ax.set_xticks([0], ["平均極性"], fontproperties=font, fontsize=13)
    sentiment_ax.set_yticks(range(len(topics)), topics, fontproperties=font, fontsize=12)
    sentiment_ax.set_ylabel("トピック（各25発言）", fontproperties=font, fontsize=13, labelpad=12)
    for row_index, value in enumerate(sentiment_values[:, 0]):
        sentiment_ax.text(0, row_index, f"{value:+.3f}", ha="center", va="center", fontsize=12, fontweight="bold")
    sentiment_colorbar = sentiment_fig.colorbar(
        sentiment_image, ax=sentiment_ax, location="bottom", shrink=0.72, pad=0.08
    )
    sentiment_colorbar.set_label("平均極性（濃いほど否定的）", fontproperties=font, fontsize=12)
    style_axis(sentiment_ax, len(topics))
    args.sentiment_output.parent.mkdir(parents=True, exist_ok=True)
    sentiment_fig.savefig(args.sentiment_output, bbox_inches="tight", facecolor="white")
    plt.close(sentiment_fig)

    print(f"wrote {args.emotion_output}")
    print(f"wrote {args.sentiment_output}")


def style_axis(axis, topic_count: int) -> None:
    axis.set_xticks(np.arange(-0.5, axis.images[0].get_array().shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, topic_count, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    main()
