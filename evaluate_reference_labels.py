#!/usr/bin/env python3
"""Evaluate WRIME predictions against the 50 blind AI reference labels."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


LABELS = Path("outputs/annotation_sample_50_labeled.csv")
PREDICTIONS = Path("outputs/speeches_sample_200_wrime.csv")
OUTPUT = Path("outputs/evaluation_50_ai_reference.md")
EMOTIONS = ("joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust")


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int]:
    return (
        int(np.sum((y_true == 1) & (y_pred == 1))),
        int(np.sum((y_true == 0) & (y_pred == 1))),
        int(np.sum((y_true == 1) & (y_pred == 0))),
    )


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    with LABELS.open(encoding="utf-8-sig", newline="") as handle:
        references = list(csv.DictReader(handle))
    with PREDICTIONS.open(encoding="utf-8-sig", newline="") as handle:
        predictions = {row["speechID"]: row for row in csv.DictReader(handle)}
    if any(row["speechID"] not in predictions for row in references):
        raise ValueError("some labeled speechIDs have no prediction")

    lines = [
        "# 50件のAI参照ラベルによる暫定評価",
        "",
        "## 8感情（有無）",
        "",
        "正解は参照強度1以上、予測は確率0.5以上を「あり」とした。",
        "",
        "| 感情 | Precision | Recall | F1 | 強度MAE | Spearman ρ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    all_true_binary = []
    all_pred_binary = []
    emotion_f1s = []
    emotion_maes = []
    for emotion in EMOTIONS:
        true_intensity = np.array([int(row[f"reference_{emotion}"]) / 3 for row in references])
        predicted = np.array([float(predictions[row["speechID"]][f"wrime_{emotion}"]) for row in references])
        true_binary = (true_intensity > 0).astype(int)
        pred_binary = (predicted >= 0.5).astype(int)
        tp, fp, fn = binary_counts(true_binary, pred_binary)
        precision, recall, f1 = prf(tp, fp, fn)
        mae = float(np.mean(np.abs(true_intensity - predicted)))
        rho = float(spearmanr(true_intensity, predicted).statistic)
        lines.append(f"| {emotion} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {mae:.3f} | {rho:.3f} |")
        all_true_binary.extend(true_binary.tolist())
        all_pred_binary.extend(pred_binary.tolist())
        emotion_f1s.append(f1)
        emotion_maes.append(mae)

    micro_tp, micro_fp, micro_fn = binary_counts(np.array(all_true_binary), np.array(all_pred_binary))
    micro = prf(micro_tp, micro_fp, micro_fn)
    lines.extend(
        [
            "",
            f"- Macro-F1: {np.mean(emotion_f1s):.3f}",
            f"- Micro-F1: {micro[2]:.3f}",
            f"- 8感情の平均MAE: {np.mean(emotion_maes):.3f}",
            "",
            "## 感情極性",
            "",
        ]
    )
    true_sentiment = np.array([int(row["reference_sentiment"]) / 2 for row in references])
    pred_sentiment = np.array([float(predictions[row["speechID"]]["wrime_sentiment"]) for row in references])
    sentiment_mae = float(np.mean(np.abs(true_sentiment - pred_sentiment)))
    sentiment_rmse = math.sqrt(float(np.mean((true_sentiment - pred_sentiment) ** 2)))
    sentiment_rho = float(spearmanr(true_sentiment, pred_sentiment).statistic)
    lines.extend(
        [
            f"- MAE: {sentiment_mae:.3f}",
            f"- RMSE: {sentiment_rmse:.3f}",
            f"- Spearman ρ: {sentiment_rho:.3f}",
            "",
            "## 注意",
            "",
            "これらは単独AIが付けた参照ラベルに対する暫定値であり、人手ゴールドラベルに対する正式な性能ではない。",
            "50件で閾値調整は行っていない。感情間のクラス不均衡が大きいため、Macro-F1を主に確認する。",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
