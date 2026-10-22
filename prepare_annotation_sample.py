#!/usr/bin/env python3
"""Prepare a reproducible 50-speech blind annotation sample."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


LABEL_FIELDS = [
    "reference_joy",
    "reference_sadness",
    "reference_anticipation",
    "reference_surprise",
    "reference_anger",
    "reference_fear",
    "reference_disgust",
    "reference_trust",
    "reference_sentiment",
    "annotation_confidence",
    "annotation_evidence",
    "annotator_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/speeches_sample_200.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/annotation_sample_50.csv"))
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields = reader.fieldnames or []
        for row in reader:
            grouped[row["cluster_label"]].append(row)

    rng = random.Random(args.seed)
    selected = []
    labels = sorted(grouped)
    for index, label in enumerate(labels):
        quota = 7 if index < 2 else 6
        selected.extend(rng.sample(grouped[label], quota))
    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fields + LABEL_FIELDS)
        writer.writeheader()
        for row in selected:
            row.update({field: "AI-reference" if field == "annotator_type" else "" for field in LABEL_FIELDS})
            writer.writerow(row)
    print(f"wrote {len(selected)} rows to {args.output}")


if __name__ == "__main__":
    main()
