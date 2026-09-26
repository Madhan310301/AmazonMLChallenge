from __future__ import annotations

from collections import Counter


def build_diagnostics(
    truth: set[tuple[str, str, str]],
    candidate_rows: list[dict],
    predictions: list[dict],
) -> dict:
    candidate_keys = {
        (row["source1_id"], row["target_source"], row["target_id"])
        for row in candidate_rows
    }
    predicted_keys = {
        (row["source1_id"], row["target_source"], row["target_id"])
        for row in predictions
    }
    false_positives = sorted(predicted_keys - truth)
    false_negatives = sorted(truth - predicted_keys)
    candidate_misses = sorted(truth - candidate_keys)
    same_name = sum(
        float(row.get("name_exact", 0.0)) == 1.0
        and (row["source1_id"], row["target_source"], row["target_id"])
        not in predicted_keys
        for row in candidate_rows
    )
    country_counts = Counter()
    for row in candidate_rows:
        country_counts[row.get("source1_country", "") or "(missing)"] += 1
    by_method = Counter()
    for row in candidate_rows:
        for method in row.get("blocking_methods", "").split(","):
            if method:
                by_method[method] += 1
    return {
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "candidate_miss_count": len(candidate_misses),
        "false_positive_pairs": [
            {"source1_id": a, "target_source": b, "target_id": c}
            for a, b, c in false_positives[:100]
        ],
        "false_negative_pairs": [
            {"source1_id": a, "target_source": b, "target_id": c}
            for a, b, c in false_negatives[:100]
        ],
        "true_pairs_not_generated_by_blocking": [
            {"source1_id": a, "target_source": b, "target_id": c}
            for a, b, c in candidate_misses[:100]
        ],
        "exact_name_candidates_not_selected": same_name,
        "candidate_counts_by_country": dict(country_counts),
        "candidate_counts_by_blocking_method": dict(by_method),
    }