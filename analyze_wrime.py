#!/usr/bin/env python3
"""Run WRIME-trained emotion models over a sample of Diet speeches."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


EMOTION_MODEL = "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime"
POLARITY_MODEL = "katsutaku/wrime-sentiment-analyzer"
EMOTIONS = ("joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?])\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/speeches_sample_200.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/speeches_sample_200_wrime.csv"))
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    return parser.parse_args()


def chunks(text: str, max_chars: int) -> list[str]:
    """Split long speeches at sentence boundaries before tokenization."""
    sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(text) if s.strip()]
    result: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            result.append(current)
            current = ""
        while len(sentence) > max_chars:
            result.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        current += sentence
    if current:
        result.append(current)
    return result or [text[:max_chars]]


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_id: str, target: torch.device, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    ).to(target)
    model.eval()
    return tokenizer, model


def predict_logits(
    tokenizer, model, texts: list[str], target: torch.device, batch_size: int
) -> torch.Tensor:
    values = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            encoded = {key: value.to(target) for key, value in encoded.items()}
            values.append(model(**encoded).logits.float().cpu())
            done = min(start + batch_size, len(texts))
            if done % 20 == 0 or done == len(texts):
                print(f"  inferred {done}/{len(texts)} chunks")
    return torch.cat(values, dim=0)


def emotion_scores(logits: torch.Tensor, id2label: dict) -> dict[str, float]:
    probabilities = torch.sigmoid(logits).mean(dim=0)
    labels = {int(k): str(v).lower() for k, v in id2label.items()}
    result = {emotion: 0.0 for emotion in EMOTIONS}
    for index, probability in enumerate(probabilities.tolist()):
        label = labels.get(index, f"label_{index}")
        if label in result:
            result[label] = probability
    # Some community models expose LABEL_0...LABEL_7 rather than semantic names.
    if not any(result.values()) and len(probabilities) == len(EMOTIONS):
        result = dict(zip(EMOTIONS, probabilities.tolist()))
    return result


def main() -> None:
    args = parse_args()
    target = device()
    print(f"device: {target}")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        source_fields = reader.fieldnames or []
    if args.limit is not None:
        rows = rows[: args.limit]

    row_chunks = [chunks(row["speech_cleaned"], args.max_chars) for row in rows]
    flat_chunks = [part for parts in row_chunks for part in parts]
    offsets = []
    start = 0
    for parts in row_chunks:
        offsets.append((start, start + len(parts)))
        start += len(parts)

    print("loading emotion model")
    emotion_tokenizer, emotion_model = load_model(EMOTION_MODEL, target)
    all_emotion_logits = predict_logits(
        emotion_tokenizer, emotion_model, flat_chunks, target, args.batch_size
    )
    all_scores = [
        emotion_scores(all_emotion_logits[start:end], emotion_model.config.id2label)
        for start, end in offsets
    ]
    del emotion_model
    if target.type == "mps":
        torch.mps.empty_cache()

    print("loading polarity model")
    polarity_tokenizer, polarity_model = load_model(POLARITY_MODEL, target, trust_remote_code=True)
    all_polarity_logits = predict_logits(
        polarity_tokenizer, polarity_model, flat_chunks, target, args.batch_size
    )
    all_polarities = [
        all_polarity_logits[start:end].reshape(end - start, -1).mean().item()
        for start, end in offsets
    ]

    output_fields = source_fields + [f"wrime_{name}" for name in EMOTIONS] + [
        "wrime_dominant_emotion",
        "wrime_sentiment",
        "wrime_chunk_count",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for number, (row, parts, scores, polarity) in enumerate(
            zip(rows, row_chunks, all_scores, all_polarities), start=1
        ):
            for name, score in scores.items():
                row[f"wrime_{name}"] = f"{score:.6f}"
            row["wrime_dominant_emotion"] = max(scores, key=scores.get)
            row["wrime_sentiment"] = f"{polarity:.6f}"
            row["wrime_chunk_count"] = str(len(parts))
            writer.writerow(row)

    print(f"wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
