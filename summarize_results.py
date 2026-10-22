#!/usr/bin/env python3
"""Validate and summarize the 200-speech WRIME experiment."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


EMOTIONS = ("joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/speeches_sample_200_wrime.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/wrime_summary_by_cluster.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input contains no rows")

    numeric = [f"wrime_{name}" for name in EMOTIONS] + ["wrime_sentiment"]
    for row in rows:
        for column in numeric:
            row[column] = float(row[column])

    for name in EMOTIONS:
        values = [row[f"wrime_{name}"] for row in rows]
        if min(values) < 0 or max(values) > 1:
            raise ValueError(f"wrime_{name} is outside [0, 1]")
    sentiments = [row["wrime_sentiment"] for row in rows]
    if min(sentiments) < -1 or max(sentiments) > 1:
        raise ValueError("wrime_sentiment is outside [-1, 1]")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["cluster_label"]].append(row)

    fields = ["cluster_label", "n", "mean_sentiment"] + [f"mean_{name}" for name in EMOTIONS]
    summary = []
    for label in sorted(groups):
        group = groups[label]
        item = {
            "cluster_label": label,
            "n": len(group),
            "mean_sentiment": f'{mean(row["wrime_sentiment"] for row in group):.6f}',
        }
        for name in EMOTIONS:
            item[f"mean_{name}"] = f'{mean(row[f"wrime_{name}"] for row in group):.6f}'
        summary.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    dominant = Counter(row["wrime_dominant_emotion"] for row in rows)
    print(f"rows: {len(rows)}")
    print(f"sentiment mean/min/max: {mean(sentiments):.3f} / {min(sentiments):.3f} / {max(sentiments):.3f}")
    print("dominant emotions:", dict(dominant.most_common()))
    print("cluster mean sentiment:")
    for item in sorted(summary, key=lambda x: float(x["mean_sentiment"])):
        print(f'  {item["cluster_label"]}: {item["mean_sentiment"]}')
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
