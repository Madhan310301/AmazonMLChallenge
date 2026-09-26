from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ALIASES = {
    "id": (
        "id",
        "record_id",
        "business_id",
        "entity_id",
        "source1_id",
        "source2_id",
        "source3_id",
        "source_1_id",
        "source_2_id",
        "source_3_id",
        "reference_id",
        "s1_id",
        "s2_id",
        "s3_id",
    ),
    "business_name": ("business_name", "name", "company_name", "merchant_name"),
    "business_address": ("business_address", "address", "location"),
    "country": ("country", "country_name", "nation"),
    "city": ("city", "locality", "town"),
    "region": ("region", "state", "province", "administrative_area"),
    "postal_code": ("postal_code", "postcode", "zip", "zip_code", "pin_code"),
}


def normalized_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def resolve_column(columns: Iterable[str], logical_name: str, explicit: str | None = None) -> str | None:
    available = {normalized_column(name): name for name in columns}
    if explicit:
        key = normalized_column(explicit)
        if key not in available:
            raise ValueError(
                f"Configured column '{explicit}' is missing. Available columns: "
                f"{', '.join(columns)}"
            )
        return available[key]
    for alias in ALIASES.get(logical_name, (logical_name,)):
        if alias in available:
            return available[alias]
    return None


def infer_ground_truth_pairs(
    rows: list[dict[str, str]],
    source1_ids: set[str],
    source2_ids: set[str],
    source3_ids: set[str],
) -> set[tuple[str, str, str]]:
    if not rows:
        raise ValueError("Ground-truth file is empty.")
    columns = list(rows[0])
    canonical = {normalized_column(column): column for column in columns}

    def find(*names: str) -> str | None:
        for name in names:
            if name in canonical:
                return canonical[name]
        return None

    source1_column = find(
        "source1_entity_id", "source1_id", "s1_id", "source_1_id", "reference_id", "left_id", "input_id"
    )
    if source1_column is None:
        candidate = resolve_column(columns, "id")
        if candidate:
            source1_column = candidate
    if source1_column is None:
        raise ValueError(
            "Could not identify the Source 1 ID column in ground truth. "
            "Use source1_id or configure the file to include an equivalent name."
        )

    source2_column = find("source2_id", "s2_id", "source_2_id", "matched_source2_ids")
    source3_column = find("source3_id", "s3_id", "source_3_id", "matched_source3_ids")
    target_source_column = find("target_source", "matched_source", "source")
    target_id_column = find("target_id", "matched_id", "candidate_id")
    generic_matches_column = find("match_ids", "matches")
    matched_entity_ids_column = find("matched_entity_ids", "matched_ids", "candidate_entity_ids", "match_id_list")

    pairs: set[tuple[str, str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        source1_id = str(row.get(source1_column, "")).strip()
        if not source1_id:
            continue
        if source1_id not in source1_ids:
            raise ValueError(
                f"Ground truth row {row_number} references unknown Source 1 ID "
                f"{source1_id!r}."
            )
        if source2_column:
            for target_id in _split_ids(row.get(source2_column, "")):
                _add_pair(pairs, source1_id, "source2", target_id, source2_ids, row_number)
        if source3_column:
            for target_id in _split_ids(row.get(source3_column, "")):
                _add_pair(pairs, source1_id, "source3", target_id, source3_ids, row_number)
        if target_source_column and target_id_column:
            target_source = _normalize_target_source(row.get(target_source_column, ""))
            target_ids = _split_ids(row.get(target_id_column, ""))
            target_pool = source2_ids if target_source == "source2" else source3_ids
            for target_id in target_ids:
                _add_pair(
                    pairs, source1_id, target_source, target_id, target_pool, row_number
                )
        elif generic_matches_column:
            # Generic match lists accept explicit source prefixes: source2:id or source3:id.
            for value in _split_ids(row.get(generic_matches_column, "")):
                if ":" not in value:
                    raise ValueError(
                        f"Ground truth row {row_number} has an untyped match ID "
                        f"{value!r}; use source2_id/source3_id columns or source:id values."
                    )
                target_source, target_id = value.split(":", 1)
                target_source = _normalize_target_source(target_source)
                pool = source2_ids if target_source == "source2" else source3_ids
                _add_pair(pairs, source1_id, target_source, target_id, pool, row_number)
        elif matched_entity_ids_column:
            for value in _split_ids(row.get(matched_entity_ids_column, "")):
                if value.startswith("S2-"):
                    _add_pair(pairs, source1_id, "source2", value, source2_ids, row_number)
                elif value.startswith("S3-"):
                    _add_pair(pairs, source1_id, "source3", value, source3_ids, row_number)
                elif value.startswith("S1-"):
                    continue
                else:
                    raise ValueError(f"Ground truth row {row_number} has invalid match ID {value!r}")

    if not any((source2_column, source3_column, target_source_column and target_id_column, generic_matches_column, matched_entity_ids_column)):
        raise ValueError(
            "Could not identify match columns in ground truth. Supported formats are "
            "source2_id/source3_id, target_source+target_id, typed match_ids, or matched_entity_ids."
        )
    return pairs


def infer_cardinality(pairs: set[tuple[str, str, str]]) -> dict[str, str]:
    by_source1: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_target: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source1_id, target_source, target_id in pairs:
        by_source1[(target_source, source1_id)].add(target_id)
        by_target[(target_source, target_id)].add(source1_id)

    result: dict[str, str] = {}
    for target_source in ("source2", "source3"):
        source1_has_many = any(
            len(targets) > 1
            for (source, _), targets in by_source1.items()
            if source == target_source
        )
        target_has_many = any(
            len(source1s) > 1
            for (source, _), source1s in by_target.items()
            if source == target_source
        )
        if not source1_has_many and not target_has_many:
            result[target_source] = "one_to_one"
        elif not target_has_many:
            result[target_source] = "one_to_many"
        elif not source1_has_many:
            result[target_source] = "many_to_one"
        else:
            result[target_source] = "many_to_many"
    return result


def schema_summary(records_by_name: dict[str, list[dict]]) -> dict:
    summary: dict[str, dict] = {}
    for name, records in records_by_name.items():
        if not records:
            summary[name] = {"rows": 0, "columns": [], "missing": {}}
            continue
        raw_keys = sorted(records[0].get("raw_fields", {}).keys())
        missing = Counter()
        for record in records:
            for key in ("business_name", "business_address", "country", "city", "postal_code"):
                if not record.get(key):
                    missing[key] += 1
        summary[name] = {
            "rows": len(records),
            "columns": raw_keys,
            "missing": dict(missing),
        }
    return summary


def _split_ids(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    value = str(raw_value).strip()
    if not value or value.casefold() in {"none", "null", "nan", "no match"}:
        return []
    return [part.strip() for part in re.split(r"[;,|]", value) if part.strip()]


def _normalize_target_source(value: str) -> str:
    normalized = normalized_column(str(value))
    if normalized in {"source2", "s2", "2", "source_2"}:
        return "source2"
    if normalized in {"source3", "s3", "3", "source_3"}:
        return "source3"
    raise ValueError(
        f"Unrecognized target source {value!r}; expected source2 or source3."
    )


def _add_pair(
    pairs: set[tuple[str, str, str]],
    source1_id: str,
    target_source: str,
    target_id: str,
    target_pool: set[str],
    row_number: int,
) -> None:
    if not target_id:
        return
    if target_id not in target_pool:
        raise ValueError(
            f"Ground truth row {row_number} references unknown {target_source} ID "
            f"{target_id!r}."
        )
    pairs.add((source1_id, target_source, target_id))