#!/usr/bin/env python3
"""Kruskal-Wallis test of WRIME polarity across the eight clusters."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kruskal


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
    parser.add_argument("--output", type=Path, default=Path("outputs/kruskal_wallis_polarity.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[float]] = defaultdict(list)
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["cluster_label"]].append(float(row["wrime_sentiment"]))

    labels = [label for label in CLUSTER_ORDER if label in grouped]
    samples = [np.asarray(grouped[label]) for label in labels]
    result = kruskal(*samples, nan_policy="raise")
    k = len(samples)
    n = sum(len(sample) for sample in samples)
    epsilon_squared = max(0.0, (float(result.statistic) - k + 1) / (n - k))

    lines = [
        "# 8クラスタ間の極性分布：Kruskal–Wallis検定",
        "",
        f"- H({k - 1}) = {result.statistic:.6f}",
        f"- p = {result.pvalue:.6f}",
        f"- ε² = {epsilon_squared:.6f}",
        f"- N = {n}（各クラスタ n = {len(samples[0])}）",
        f"- 判定（α = 0.05）: {'有意差あり' if result.pvalue < 0.05 else '有意差なし'}",
        "",
        "| クラスタ | n | 中央値 | 第1四分位 | 第3四分位 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, sample in zip(labels, samples):
        q1, median, q3 = np.quantile(sample, [0.25, 0.50, 0.75])
        lines.append(f"| {label} | {len(sample)} | {median:.6f} | {q1:.6f} | {q3:.6f} |")
    lines.extend(
        [
            "",
            "帰無仮説: 8クラスタの極性スコア分布は同じである。",
            "",
            "注: 極性スコアはWRIME学習済みモデルの推定値であり、人手による正解ラベルではない。",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
