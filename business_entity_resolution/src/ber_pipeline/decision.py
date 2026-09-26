from __future__ import annotations

from collections import defaultdict


CARDINALITIES = {
    "one_to_one",
    "one_to_many",
    "many_to_one",
    "many_to_many",
}


def resolve_matches(
    scored_pairs: list[dict],
    threshold: float,
    cardinality_by_source: dict[str, str],
) -> list[dict]:
    """Apply NO_MATCH threshold and source-specific cardinality rules."""
    accepted = [
        dict(pair)
        for pair in scored_pairs
        if float(pair.get("confidence", 0.0)) > threshold
    ]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for pair in accepted:
        by_source[pair["target_source"]].append(pair)

    selected: list[dict] = []
    for target_source, rows in by_source.items():
        cardinality = cardinality_by_source.get(target_source, "many_to_many")
        if cardinality not in CARDINALITIES:
            raise ValueError(f"Invalid cardinality {cardinality!r} for {target_source}.")
        if cardinality == "one_to_one":
            selected.extend(_maximum_weight_one_to_one(rows, threshold))
        elif cardinality == "one_to_many":
            selected.extend(_unique_target_winners(rows))
        elif cardinality == "many_to_one":
            selected.extend(_unique_source1_winners(rows))
        else:
            selected.extend(rows)

    selected.sort(
        key=lambda row: (
            row["source1_id"],
            row["target_source"],
            row["target_id"],
        )
    )
    return selected


def resolve_conflicts_globally(rows: list[dict]) -> list[dict]:
    """
    Global conflict resolution (Upgrade 16):
    When multiple Source 1 records compete for the same target and uniqueness is required:
    1. Sort all accepted candidates descending by confidence (with deterministic tie-breakers).
    2. Highest-confidence valid assignment first.
    3. Reserve target so no lower-confidence competitor can claim it.
    Removes order-dependence from assignment resolution.
    """
    result = []
    reserved_targets: set[tuple[str, str]] = set()
    for pair in sorted(
        rows,
        key=lambda row: (
            -float(row.get("confidence", 0.0)),
            row.get("source1_id", ""),
            row.get("target_id", ""),
        ),
    ):
        tgt_key = (pair.get("target_source", ""), pair.get("target_id", ""))
        if tgt_key in reserved_targets:
            continue
        result.append(pair)
        reserved_targets.add(tgt_key)
    return result


def _maximum_weight_one_to_one(rows: list[dict], threshold: float) -> list[dict]:
    if not rows:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
    except ImportError:
        # A stable greedy fallback remains deterministic if SciPy is unavailable.
        return _greedy_one_to_one(rows)

    source1_ids = sorted({row["source1_id"] for row in rows})
    target_ids = sorted({row["target_id"] for row in rows})
    row_index = {value: index for index, value in enumerate(source1_ids)}
    target_index = {value: index for index, value in enumerate(target_ids)}
    # One private dummy column per source1 encodes the explicit NO_MATCH option.
    weights = np.zeros((len(source1_ids), len(target_ids) + len(source1_ids)))
    pair_by_position: dict[tuple[int, int], dict] = {}
    for pair in rows:
        i, j = row_index[pair["source1_id"]], target_index[pair["target_id"]]
        gain = float(pair["confidence"]) - threshold
        if gain > weights[i, j]:
            weights[i, j] = gain
            pair_by_position[(i, j)] = pair
    row_indices, column_indices = linear_sum_assignment(weights, maximize=True)
    result = []
    for i, j in zip(row_indices.tolist(), column_indices.tolist()):
        if j < len(target_ids) and weights[i, j] > 0:
            pair = pair_by_position.get((i, j))
            if pair:
                result.append(pair)
    return result


def _greedy_one_to_one(rows: list[dict]) -> list[dict]:
    result = []
    used_source1: set[str] = set()
    used_targets: set[tuple[str, str]] = set()
    for pair in sorted(
        rows,
        key=lambda row: (
            -float(row.get("confidence", 0.0)),
            row.get("source1_id", ""),
            row.get("target_id", ""),
        ),
    ):
        target_key = (pair["target_source"], pair["target_id"])
        if pair["source1_id"] in used_source1 or target_key in used_targets:
            continue
        result.append(pair)
        used_source1.add(pair["source1_id"])
        used_targets.add(target_key)
    return result


def _unique_target_winners(rows: list[dict]) -> list[dict]:
    return resolve_conflicts_globally(rows)


def _unique_source1_winners(rows: list[dict]) -> list[dict]:
    winners: dict[str, dict] = {}
    for pair in sorted(
        rows,
        key=lambda row: (
            -float(row.get("confidence", 0.0)),
            row.get("source1_id", ""),
            row.get("target_id", ""),
        ),
    ):
        winners.setdefault(pair["source1_id"], pair)
    return list(winners.values())