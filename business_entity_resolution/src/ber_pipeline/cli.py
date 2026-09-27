from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from .config import PROJECT_ROOT, load_settings
from .dataset_discovery import (
    bootstrap_directories,
    discover_dataset,
    print_check_data_report,
    print_missing_data_message,
    validate_schemas,
)
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline business entity-resolution pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file. Paths are relative to the project directory.",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "inspect",
            "preprocess",
            "blocking",
            "features",
            "scoring",
            "decision",
            "evaluate",
            "all",
        ),
        default="all",
        help=(
            "Run schema inspection, preprocessing, blocking, held-out evaluation, "
            "or the complete pipeline."
        ),
    )
    parser.add_argument("--data-root", type=Path, help="Override the dataset directory.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache reads/writes.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear JSON cache artifacts.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    parser.add_argument(
        "--make-demo",
        action="store_true",
        help="Create a small synthetic dataset under dataset/ and run the pipeline.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Report dataset availability, schema validity, and row counts.",
    )
    parser.add_argument(
        "--max-train-records",
        type=int,
        default=None,
        help="Optional maximum number of training Source 1 entities to use for model fitting.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Bootstrap directories
    bootstrap_directories(PROJECT_ROOT)

    # --check-data: report dataset status and exit
    if args.check_data:
        settings = load_settings(args.config)
        if args.data_root:
            settings = replace(settings, data_root=str(args.data_root))
        resolved = settings.resolved(PROJECT_ROOT)
        return print_check_data_report(resolved.data_root)

    # --make-demo: generate synthetic data, then run pipeline
    if args.make_demo:
        from examples.create_demo_dataset import create_demo_dataset

        target = PROJECT_ROOT / "dataset"
        create_demo_dataset(target)
        print(f"Created demo data at {target}")
        print("Running pipeline on demo data...")
        print()

        settings = load_settings(args.config)
        resolved = settings.resolved(PROJECT_ROOT)
        try:
            result = run_pipeline(
                resolved,
                stage="all",
                use_cache=not args.no_cache,
                clear_cache=args.clear_cache,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            logging.error("%s", error)
            return 2

        print()
        print("=" * 60)
        print("DEMO PIPELINE COMPLETE")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        print()
        print("WARNING: This was a DEMO run using synthetic data.")
        print("These results are NOT valid for competition submission.")
        print("=" * 60)
        return 0

    # Normal pipeline execution
    settings = load_settings(args.config)
    if args.data_root:
        settings = replace(settings, data_root=str(args.data_root))
    if args.output_dir:
        settings = replace(settings, output_dir=str(args.output_dir))
    if args.max_train_records:
        settings = replace(settings, max_train_records=args.max_train_records)
    resolved = settings.resolved(PROJECT_ROOT)

    # Check if data is available before attempting to run
    discovery = discover_dataset(resolved.data_root)
    if not discovery["all_files_found"]:
        print_missing_data_message(resolved.data_root)
        return 3

    # Validate schemas before running
    schema_errors = validate_schemas(resolved.data_root)
    if schema_errors:
        print()
        print("Dataset found, but schema validation failed.")
        print()
        for error in schema_errors:
            print(f"  ERROR: {error}")
        print()
        print("No ML pipeline execution was started.")
        return 4

    try:
        result = run_pipeline(
            resolved,
            stage=args.stage,
            use_cache=not args.no_cache,
            clear_cache=args.clear_cache,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0