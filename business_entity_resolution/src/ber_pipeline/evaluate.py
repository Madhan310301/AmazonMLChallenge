from __future__ import annotations

from collections import defaultdict
from typing import Any

from .decision import resolve_matches


def macro_fbeta(
    truth: set[tuple[str, str, str]],
    predictions: list[dict] | set[tuple[str, str, str]],
    source1_ids: set[str],
    beta: float = 0.5,
) -> dict[str, float | int]:
    if beta <= 0:
        raise ValueError("beta must be positive.")
    predicted_pairs = (
        {
            (row["source1_id"], row["target_source"], row["target_id"])
            for row in predictions
        }
        if isinstance(predictions, list)
        else set(predictions)
    )
    true_by_source1: dict[str, set[tuple[str, str]]] = defaultdict(set)
    pred_by_source1: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for source1_id, target_source, target_id in truth:
        true_by_source1[source1_id].add((target_source, target_id))
    for source1_id, target_source, target_id in predicted_pairs:
        pred_by_source1[source1_id].add((target_source, target_id))

    per_entity_f: list[float] = []
    for source1_id in sorted(source1_ids):
        expected = true_by_source1[source1_id]
        actual = pred_by_source1[source1_id]
        true_positive = len(expected & actual)
        false_positive = len(actual - expected)
        false_negative = len(expected - actual)
        if not expected and not actual:
            per_entity_f.append(1.0)
            continue
        precision = true_positive / (true_positive + false_positive) if actual else 0.0
        recall = true_positive / (true_positive + false_negative) if expected else 0.0
        denominator = beta * beta * precision + recall
        per_entity_f.append(
            (1 + beta * beta) * precision * recall / denominator
            if denominator
            else 0.0
        )

    tp = len(truth & predicted_pairs)
    fp = len(predicted_pairs - truth)
    fn = len(truth - predicted_pairs)
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    beta_sq = beta * beta
    micro_f = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if beta_sq * precision + recall
        else 0.0
    )
    return {
        "macro_f0_5": sum(per_entity_f) / len(per_entity_f) if per_entity_f else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f0_5": micro_f,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "unmatched_entities": sum(
            not pred_by_source1[source1_id] for source1_id in source1_ids
        ),
    }


def sweep_thresholds(
    scored_pairs: list[dict],
    truth: set[tuple[str, str, str]],
    source1_ids: set[str],
    cardinality_by_source: dict[str, str],
    lower: float = 0.5,
    upper: float = 0.99,
    steps: int = 50,
) -> list[dict[str, Any]]:
    """
    Evaluate candidate thresholds across the specified range (Upgrade 15).
    For each threshold computes precision, recall, macro F0.5, false merges,
    matched entities, and unmatched entities.
    """
    if steps < 2:
        raise ValueError("Threshold search requires at least two steps.")
    history = []
    for index in range(steps):
        threshold = lower + (upper - lower) * index / (steps - 1)
        predictions = resolve_matches(scored_pairs, threshold, cardinality_by_source)
        metrics = macro_fbeta(truth, predictions, source1_ids)
        pred_source1_matched = {row["source1_id"] for row in predictions}
        history.append(
            {
                "threshold": round(threshold, 4),
                "macro_f0_5": float(metrics["macro_f0_5"]),
                "micro_precision": float(metrics["micro_precision"]),
                "micro_recall": float(metrics["micro_recall"]),
                "false_merges": int(metrics["false_positives"]),
                "matched_entities": len(pred_source1_matched),
                "unmatched_entities": len(source1_ids - pred_source1_matched),
            }
        )
    return history


def tune_threshold(
    scored_pairs: list[dict],
    truth: set[tuple[str, str, str]],
    source1_ids: set[str],
    cardinality_by_source: dict[str, str],
    lower: float = 0.5,
    upper: float = 0.99,
    steps: int = 50,
) -> tuple[float, list[dict]]:
    """
    Dynamic F0.5 threshold sweep (Upgrade 15).
    Sweeps a sensible range and selects the threshold that maximizes the actual
    competition-style macro F0.5 on held-out validation data.
    """
    if steps < 2:
        raise ValueError("Threshold search requires at least two steps.")
    best_threshold = lower
    best_metrics: dict[str, float | int] | None = None
    for index in range(steps):
        threshold = lower + (upper - lower) * index / (steps - 1)
        predictions = resolve_matches(scored_pairs, threshold, cardinality_by_source)
        metrics = macro_fbeta(truth, predictions, source1_ids)
        if best_metrics is None or (
            float(metrics["macro_f0_5"]),
            float(metrics["micro_precision"]),
        ) > (
            float(best_metrics["macro_f0_5"]),
            float(best_metrics["micro_precision"]),
        ):
            best_metrics = metrics
            best_threshold = threshold
    return best_threshold, resolve_matches(
        scored_pairs, best_threshold, cardinality_by_source
    )