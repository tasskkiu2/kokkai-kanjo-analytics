#!/usr/bin/env python3
"""Create a reproducible, topic-balanced sample of Diet speeches."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("speech_clusters_excluding_0_4.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/speeches_sample_200.csv"))
    parser.add_argument("--per-cluster", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_cluster: dict[str, list[dict[str, str]]] = defaultdict(list)

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "cluster_label" not in fieldnames or "speech_cleaned" not in fieldnames:
            raise ValueError("input must contain cluster_label and speech_cleaned")
        for row in reader:
            by_cluster[row["cluster_label"]].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for label in sorted(by_cluster):
        rows = by_cluster[label]
        if len(rows) < args.per_cluster:
            raise ValueError(f"cluster {label!r} has only {len(rows)} rows")
        selected.extend(rng.sample(rows, args.per_cluster))

    selected.sort(key=lambda row: (row["cluster_label"], row["speechID"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"wrote {len(selected)} rows to {args.output}")
    for label in sorted(by_cluster):
        print(f"  {label}: {args.per_cluster}")


if __name__ == "__main__":
    main()
