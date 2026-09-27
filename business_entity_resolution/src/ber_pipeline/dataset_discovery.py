"""Centralized dataset discovery, validation, and mode detection.

Provides friendly diagnostics when official competition data is absent,
detects demo vs official mode, and validates file schemas before the
pipeline attempts to run.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("entity_resolution")

DEMO_MARKER_FILENAME = "_DEMO_MARKER.json"

EXPECTED_TRAIN_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
]

EXPECTED_TEST_FILES = [
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
]


def bootstrap_directories(project_root: Path) -> None:
    """Create required directories if they don't exist."""
    for subdir in ("dataset/train", "dataset/test", "output", "cache", "models"):
        directory = project_root / subdir
        directory.mkdir(parents=True, exist_ok=True)
        LOGGER.debug("Ensured directory exists: %s", directory)


def discover_dataset(data_root: Path) -> dict[str, Any]:
    """Check which expected files exist and detect dataset mode.

    Args:
        data_root: Path to the dataset root (contains train/ and test/ subdirs).

    Returns:
        Dict with keys: data_root, file_status, all_files_found,
        train_complete, test_complete, dataset_mode, is_demo,
        demo_metadata, official_submission_allowed.
    """
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    file_status: dict[str, dict[str, Any]] = {}

    for filename in EXPECTED_TRAIN_FILES:
        path = train_dir / filename
        exists = path.is_file()
        file_status[f"train/{filename}"] = {
            "path": str(path),
            "exists": exists,
            "size": path.stat().st_size if exists else 0,
        }

    for filename in EXPECTED_TEST_FILES:
        path = test_dir / filename
        exists = path.is_file()
        file_status[f"test/{filename}"] = {
            "path": str(path),
            "exists": exists,
            "size": path.stat().st_size if exists else 0,
        }

    train_complete = all(
        file_status[f"train/{f}"]["exists"] for f in EXPECTED_TRAIN_FILES
    )
    test_complete = all(
        file_status[f"test/{f}"]["exists"] for f in EXPECTED_TEST_FILES
    )
    all_found = train_complete and test_complete

    # Detect demo mode via marker file
    demo_marker_path = data_root / DEMO_MARKER_FILENAME
    is_demo = demo_marker_path.is_file()
    demo_metadata: dict | None = None
    if is_demo:
        try:
            demo_metadata = json.loads(
                demo_marker_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            demo_metadata = None

    if is_demo:
        dataset_mode = "demo"
    elif all_found:
        dataset_mode = "official"
    else:
        dataset_mode = "missing"

    return {
        "data_root": str(data_root),
        "file_status": file_status,
        "all_files_found": all_found,
        "train_complete": train_complete,
        "test_complete": test_complete,
        "dataset_mode": dataset_mode,
        "is_demo": is_demo,
        "demo_metadata": demo_metadata,
        "official_submission_allowed": dataset_mode == "official",
    }


def validate_schemas(data_root: Path) -> list[str]:
    """Validate that existing TSV files have required columns.

    Returns:
        List of error strings (empty if all valid).
    """
    from .schema import resolve_column
    from .io import read_tsv_header

    errors: list[str] = []
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    source_files = [
        ("train/train_source1.tsv", train_dir / "train_source1.tsv", True),
        ("train/train_source2.tsv", train_dir / "train_source2.tsv", True),
        ("train/train_source3.tsv", train_dir / "train_source3.tsv", True),
        ("train/train_ground_truth.tsv", train_dir / "train_ground_truth.tsv", False),
        ("test/test_source1.tsv", test_dir / "test_source1.tsv", True),
        ("test/test_source2.tsv", test_dir / "test_source2.tsv", True),
        ("test/test_source3.tsv", test_dir / "test_source3.tsv", True),
    ]

    for label, path, needs_business_name in source_files:
        if not path.is_file():
            continue
        try:
            columns = read_tsv_header(path)
        except (ValueError, FileNotFoundError) as exc:
            errors.append(f"{label}: {exc}")
            continue

        id_col = resolve_column(columns, "id")
        if id_col is None:
            errors.append(
                f"{label}: missing required ID column. "
                f"Available columns: {', '.join(columns)}"
            )

        if needs_business_name:
            name_col = resolve_column(columns, "business_name")
            if name_col is None:
                errors.append(
                    f"{label}: missing required 'business_name' column. "
                    f"Available columns: {', '.join(columns)}"
                )

    return errors


def get_row_counts(data_root: Path) -> dict[str, int | None]:
    """Get row counts for each expected file via fast streaming line counting."""
    counts: dict[str, int | None] = {}
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    for filename in EXPECTED_TRAIN_FILES:
        path = train_dir / filename
        if path.is_file():
            try:
                with path.open("rb") as f:
                    counts[filename] = max(0, sum(1 for _ in f) - 1)
            except Exception:
                counts[filename] = None
        else:
            counts[filename] = None

    for filename in EXPECTED_TEST_FILES:
        path = test_dir / filename
        if path.is_file():
            try:
                with path.open("rb") as f:
                    counts[filename] = max(0, sum(1 for _ in f) - 1)
            except Exception:
                counts[filename] = None
        else:
            counts[filename] = None

    return counts


def print_missing_data_message(data_root: Path) -> None:
    """Print a friendly message when official data is not found."""
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 60)
    print()
    print("Dataset status: OFFICIAL DATA NOT FOUND")
    print()
    print("Expected files:")
    print()
    for f in EXPECTED_TRAIN_FILES:
        path = train_dir / f
        status = "FOUND" if path.is_file() else "MISSING"
        print(f"  dataset/train/{f:<30s} [{status}]")
    for f in EXPECTED_TEST_FILES:
        path = test_dir / f
        status = "FOUND" if path.is_file() else "MISSING"
        print(f"  dataset/test/{f:<30s}  [{status}]")
    print()
    print("No official competition dataset is installed.")
    print()
    print("To test the complete pipeline locally:")
    print("    python run.py --make-demo")
    print()
    print("To run on the real challenge data:")
    print("    Place the 7 official TSV files in the paths above")
    print("    then run:")
    print("    python run.py")
    print()
    print("IMPORTANT:")
    print("Demo data cannot be used as an official competition submission.")
    print("=" * 60)


def print_check_data_report(data_root: Path) -> int:
    """Print a detailed dataset status report.

    Returns:
        Exit code: 0=OK, 1=missing files, 2=schema errors.
    """
    discovery = discover_dataset(data_root)
    counts = get_row_counts(data_root)
    has_files = discovery["train_complete"] or discovery["test_complete"]
    schema_errors = validate_schemas(data_root) if has_files else []

    print("=" * 60)
    print("DATASET STATUS REPORT")
    print("=" * 60)
    print()
    print(f"Data root:    {data_root}")
    print(f"Dataset mode: {discovery['dataset_mode'].upper()}")
    print()

    # Train files
    for f in EXPECTED_TRAIN_FILES:
        count = counts.get(f)
        if count is not None:
            print(f"  TRAIN  {f:<30s} FOUND   {count:>8,} rows")
        else:
            print(f"  TRAIN  {f:<30s} MISSING")
    print()

    # Test files
    for f in EXPECTED_TEST_FILES:
        count = counts.get(f)
        if count is not None:
            print(f"  TEST   {f:<30s} FOUND   {count:>8,} rows")
        else:
            print(f"  TEST   {f:<30s} MISSING")

    if schema_errors:
        print()
        print("Schema validation errors:")
        for error in schema_errors:
            print(f"  ERROR: {error}")
    elif has_files:
        print()
        print("Schema validation: PASS")

    if discovery["is_demo"]:
        print()
        print("NOTE: This is DEMO data — not valid for competition submission.")

    print()
    print("=" * 60)

    if not discovery["all_files_found"]:
        return 1
    if schema_errors:
        return 2
    return 0
