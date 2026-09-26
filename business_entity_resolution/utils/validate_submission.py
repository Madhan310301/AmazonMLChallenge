#!/usr/bin/env python3
"""Validate the pipeline's canonical outputs against the local test TSV IDs."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ber_pipeline.io import load_source, read_tsv  # noqa: E402


def validate(project_root: Path = PROJECT_ROOT) -> list[str]:
    data_root = project_root / "dataset"
    output_dir = project_root / "output"
    # Honor paths recorded in run_metadata when a custom data/output path was used.
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.is_file():
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        data_root = Path(metadata.get("data_root", data_root))
        output_dir = Path(metadata.get("output_dir", output_dir))
        candidate_cap = int(metadata.get("max_candidates_per_entity", 60))
    else:
        candidate_cap = 60
    test1 = load_source(data_root / "test" / "test_source1.tsv", "test_source1")
    test2 = load_source(data_root / "test" / "test_source2.tsv", "test_source2")
    test3 = load_source(data_root / "test" / "test_source3.tsv", "test_source3")
    candidate_columns, candidate_rows = read_tsv(output_dir / "candidate_pairs.tsv")
    match_columns, match_rows = read_tsv(output_dir / "matching_results.tsv")

    errors: list[str] = []
    required_candidate_columns = {
        "source1_id",
        "target_source",
        "target_id",
        "confidence",
    }
    required_match_columns = {"source1_id", "target_source", "target_id"}
    if not required_candidate_columns.issubset(candidate_columns):
        errors.append(
            "candidate_pairs.tsv is missing required columns: "
            + ", ".join(sorted(required_candidate_columns - set(candidate_columns)))
        )
    if not required_match_columns.issubset(match_columns):
        errors.append(
            "matching_results.tsv is missing required columns: "
            + ", ".join(sorted(required_match_columns - set(match_columns)))
        )
    source1_ids = {record["id"] for record in test1}
    targets = {
        "source2": {record["id"] for record in test2},
        "source3": {record["id"] for record in test3},
    }
    candidate_keys = set()
    per_source1_source_counts: Counter[tuple[str, str]] = Counter()
    for line_number, row in enumerate(candidate_rows, start=2):
        key = (row.get("source1_id", ""), row.get("target_source", ""), row.get("target_id", ""))
        if key in candidate_keys:
            errors.append(f"duplicate candidate pair at row {line_number}: {key}")
        candidate_keys.add(key)
        source1_id, target_source, target_id = key
        if source1_id not in source1_ids:
            errors.append(f"unknown Source 1 ID at candidate row {line_number}: {source1_id}")
        if target_id not in targets.get(target_source, set()):
            errors.append(f"unknown target ID at candidate row {line_number}: {key}")
        per_source1_source_counts[(source1_id, target_source)] += 1
        try:
            score = float(row.get("confidence", ""))
            if not 0 <= score <= 1:
                errors.append(f"confidence outside [0,1] at candidate row {line_number}")
        except ValueError:
            errors.append(f"invalid confidence at candidate row {line_number}")

    for key, count in per_source1_source_counts.items():
        if count > candidate_cap:
            errors.append(
                f"candidate cap exceeded for {key[0]} / {key[1]}: {count} > {candidate_cap}"
            )
    seen_matches = set()
    for line_number, row in enumerate(match_rows, start=2):
        key = (row.get("source1_id", ""), row.get("target_source", ""), row.get("target_id", ""))
        if key not in candidate_keys:
            errors.append(f"match not present in candidates at row {line_number}: {key}")
        if key in seen_matches:
            errors.append(f"duplicate match at row {line_number}: {key}")
        seen_matches.add(key)
        if key[0] not in source1_ids:
            errors.append(f"unknown Source 1 ID at match row {line_number}: {key[0]}")
        if key[2] not in targets.get(key[1], set()):
            errors.append(f"unknown target ID at match row {line_number}: {key}")
    if not errors:
        print(
            f"PASS: {len(candidate_keys)} candidates and {len(seen_matches)} matches; "
            "IDs, uniqueness, score range, and candidate-subset checks passed."
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    errors = validate(args.project_root)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())