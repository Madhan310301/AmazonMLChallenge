from __future__ import annotations

import json
import logging
import random
import time
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .blocking import compute_blocking_stats, generate_candidates
from .cache import CacheStore, cache_metadata, file_fingerprint
from .dataset_discovery import DEMO_MARKER_FILENAME, discover_dataset
from .config import ResolvedSettings
from .decision import resolve_matches
from .diagnostics import build_diagnostics
from .evaluate import macro_fbeta, tune_threshold
from .features import FEATURE_NAMES, compute_features, deterministic_rule, fit_idf
from .io import load_ground_truth_rows, load_source, read_tsv, write_tsv
from .preprocess import preprocess_record
from .schema import infer_cardinality, infer_ground_truth_pairs, schema_summary
from .scoring import MatchScorer

LOGGER = logging.getLogger("entity_resolution")

TRAIN_FILES = {
    "source1": ("train_source1.tsv", "train_source1"),
    "source2": ("train_source2.tsv", "train_source2"),
    "source3": ("train_source3.tsv", "train_source3"),
}
TEST_FILES = {
    "source1": ("test_source1.tsv", "test_source1"),
    "source2": ("test_source2.tsv", "test_source2"),
    "source3": ("test_source3.tsv", "test_source3"),
}


def run_pipeline(
    settings: ResolvedSettings,
    *,
    stage: str = "all",
    use_cache: bool = True,
    clear_cache: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    for directory in (
        settings.output_dir,
        settings.cache_dir,
        settings.model_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    cache = CacheStore(settings.cache_dir, enabled=use_cache)
    if clear_cache:
        cache.clear()

    # Detect dataset mode (demo vs official)
    discovery = discover_dataset(settings.data_root)
    dataset_mode = discovery["dataset_mode"]
    official_submission_allowed = discovery["official_submission_allowed"]

    data = _load_and_prepare(settings, cache)
    if stage == "inspect":
        report = _inspect(data)
        _write_json(settings.output_dir / "schema_summary.json", report)
        return report
    if stage == "preprocess":
        _write_json(settings.cache_dir / "preprocessed_records.json", {
            "train": data["train"],
            "test": data["test"],
        })
        report = _inspect(data)
        LOGGER.info("[PREPROCESS] Cached normalized source records")
        return report
    if stage == "blocking":
        train_candidates = _build_candidates(
            data["train"]["source1"],
            {
                "source2": data["train"]["source2"],
                "source3": data["train"]["source3"],
            },
            settings,
            cache=cache,
            cache_name="train_candidates",
            dataset_fingerprint=data["dataset_fingerprint"],
        )
        test_candidates = _build_candidates(
            data["test"]["source1"],
            {
                "source2": data["test"]["source2"],
                "source3": data["test"]["source3"],
            },
            settings,
            cache=cache,
            cache_name="test_candidates",
            dataset_fingerprint=data["dataset_fingerprint"],
        )
        columns = [
            "source1_id",
            "target_source",
            "target_id",
            "blocking_methods",
            "blocking_score",
        ]
        write_tsv(settings.cache_dir / "train_candidate_pairs.tsv", train_candidates, columns)
        write_tsv(settings.cache_dir / "test_candidate_pairs.tsv", test_candidates, columns)
        return {
            "train_candidate_pairs": len(train_candidates),
            "test_candidate_pairs": len(test_candidates),
            "cache_dir": str(settings.cache_dir),
        }
    if stage not in {"features", "scoring", "decision", "all", "evaluate"}:
        raise ValueError(
            "Supported stages are 'inspect', 'preprocess', 'blocking', "
            "'features', 'scoring', 'decision', 'evaluate', and 'all'. "
            "Module functions can also be called separately from Python."
        )

    train1 = data["train"]["source1"]
    train2 = data["train"]["source2"]
    train3 = data["train"]["source3"]
    test1 = data["test"]["source1"]
    test2 = data["test"]["source2"]
    test3 = data["test"]["source3"]
    truth = data["truth"]
    if len(train1) < 2:
        raise ValueError(
            "At least two Source 1 training rows are required for a held-out "
            "validation split."
        )
    if not test1:
        raise ValueError("The test Source 1 file contains no entities.")

    cardinality = (
        infer_cardinality(truth)
        if settings.cardinality == "infer"
        else {
            "source2": settings.cardinality,
            "source3": settings.cardinality,
        }
    )
    fit_ids, validation_ids = _split_source1_ids(
        [record["id"] for record in train1],
        settings.validation_fraction,
        settings.seed,
    )
    train1_by_id = {record["id"]: record for record in train1}
    fit_source1 = [train1_by_id[record_id] for record_id in fit_ids]
    validation_source1 = [train1_by_id[record_id] for record_id in validation_ids]
    validation_truth = {
        pair for pair in truth if pair[0] in set(validation_ids)
    }
    fit_truth = {pair for pair in truth if pair[0] in set(fit_ids)}

    # A target linked only to a held-out entity is withheld from fitting, but
    # remains available when generating validation candidates.
    fit_targets_by_source = {
        "source2": _without_validation_only_targets(
            train2, fit_truth, validation_truth, "source2"
        ),
        "source3": _without_validation_only_targets(
            train3, fit_truth, validation_truth, "source3"
        ),
    }
    fit_records = fit_source1 + fit_targets_by_source["source2"] + fit_targets_by_source["source3"]
    idf_fit = fit_idf(fit_records)

    LOGGER.info("[BLOCKING] Generate fit and held-out candidates")
    fit_candidates = _build_candidates(
        fit_source1,
        fit_targets_by_source,
        settings,
        cache=cache,
        cache_name="fit_candidates",
        dataset_fingerprint=data["dataset_fingerprint"],
    )
    validation_candidates = _build_candidates(
        validation_source1,
        {"source2": train2, "source3": train3},
        settings,
        cache=cache,
        cache_name="validation_candidates",
        dataset_fingerprint=data["dataset_fingerprint"],
    )
    fit_feature_rows = _feature_rows(
        fit_candidates,
        fit_source1,
        fit_targets_by_source,
        idf_fit,
        cache=cache,
        cache_name="fit_features",
    )
    fit_training_features, fit_training_labels, fit_hn_stats = _mine_hard_negatives(
        fit_candidates, fit_feature_rows, fit_truth, seed=settings.seed
    )
    LOGGER.info(
        "[HARD-NEGATIVES] Fit partition: Positives=%d Hard Negatives=%d Random Negatives=%d Ratio=%.2f",
        fit_hn_stats["positive_count"],
        fit_hn_stats["hard_negative_count"],
        fit_hn_stats["random_negative_count"],
        fit_hn_stats["hard_negative_ratio"],
    )
    validation_features = _feature_rows(
        validation_candidates,
        validation_source1,
        {"source2": train2, "source3": train3},
        idf_fit,
        cache=cache,
        cache_name="validation_features",
    )

    validation_model = MatchScorer(
        settings.seed, settings.rf_estimators, settings.random_forest_max_depth
    ).fit(fit_training_features, fit_training_labels)
    validation_scored = _score_candidates(
        validation_candidates, validation_features, validation_model
    )
    threshold, validation_predictions = tune_threshold(
        validation_scored,
        validation_truth,
        set(validation_ids),
        cardinality,
        settings.threshold_min,
        settings.threshold_max,
        settings.threshold_steps,
    )
    validation_metrics = macro_fbeta(
        validation_truth, validation_predictions, set(validation_ids)
    )
    possible_validation_truth = {
        pair for pair in validation_truth if pair[0] in set(validation_ids)
    }
    candidate_key_set = {
        (row["source1_id"], row["target_source"], row["target_id"])
        for row in validation_candidates
    }
    candidate_recall = (
        len(possible_validation_truth & candidate_key_set) / len(possible_validation_truth)
        if possible_validation_truth
        else 1.0
    )
    validation_metrics.update(
        {
            "selected_threshold": threshold,
            "candidate_recall": candidate_recall,
            "false_positive_rate": (
                int(validation_metrics["false_positives"])
                / max(1, len(validation_candidates))
            ),
            "false_merge_rate": (
                int(validation_metrics["false_positives"])
                / max(
                    1,
                    int(validation_metrics["true_positives"])
                    + int(validation_metrics["false_positives"]),
                )
            ),
            "unmatched_rate": (
                int(validation_metrics["unmatched_entities"])
                / max(1, len(validation_ids))
            ),
            "candidate_count": len(validation_candidates),
            "average_candidates_per_source1": (
                len(validation_candidates) / len(validation_ids)
            ),
        }
    )
    LOGGER.info(
        "[EVALUATION] Held-out macro F0.5=%.4f precision=%.4f recall=%.4f "
        "candidate_recall=%.4f threshold=%.3f",
        float(validation_metrics["macro_f0_5"]),
        float(validation_metrics["micro_precision"]),
        float(validation_metrics["micro_recall"]),
        candidate_recall,
        threshold,
    )

    all_train_targets = {"source2": train2, "source3": train3}
    full_idf = fit_idf(train1 + train2 + train3)
    all_train_candidates = _build_candidates(
        train1,
        all_train_targets,
        settings,
        cache=cache,
        cache_name="all_train_candidates",
        dataset_fingerprint=data["dataset_fingerprint"],
    )
    all_train_features = _feature_rows(
        all_train_candidates,
        train1,
        all_train_targets,
        full_idf,
        cache=cache,
        cache_name="all_train_features",
    )
    all_labels = [
        int((row["source1_id"], row["target_source"], row["target_id"]) in truth)
        for row in all_train_candidates
    ]
    LOGGER.info("[BLOCKING] Generate test candidates")
    test_targets = {"source2": test2, "source3": test3}
    test_candidates = _build_candidates(
        test1,
        test_targets,
        settings,
        cache=cache,
        cache_name="test_candidates",
        dataset_fingerprint=data["dataset_fingerprint"],
    )
    test_features = _feature_rows(
        test_candidates,
        test1,
        test_targets,
        full_idf,
        cache=cache,
        cache_name="test_features",
    )
    if stage == "features":
        _write_feature_table(
            settings.cache_dir / "train_candidate_features.tsv",
            all_train_candidates,
            all_train_features,
        )
        _write_feature_table(
            settings.cache_dir / "test_candidate_features.tsv",
            test_candidates,
            test_features,
        )
        return {
            "train_candidate_pairs": len(all_train_candidates),
            "test_candidate_pairs": len(test_candidates),
            "feature_count": len(FEATURE_NAMES),
            "cache_dir": str(settings.cache_dir),
        }

    all_train_training_features, all_train_training_labels, final_hn_stats = _mine_hard_negatives(
        all_train_candidates, all_train_features, truth, seed=settings.seed
    )
    LOGGER.info(
        "[HARD-NEGATIVES] Final model: Positives=%d Hard Negatives=%d Random Negatives=%d Ratio=%.2f",
        final_hn_stats["positive_count"],
        final_hn_stats["hard_negative_count"],
        final_hn_stats["random_negative_count"],
        final_hn_stats["hard_negative_ratio"],
    )

    final_model = MatchScorer(
        settings.seed, settings.rf_estimators, settings.random_forest_max_depth
    ).fit(all_train_training_features, all_train_training_labels)
    final_model.save(
        settings.model_dir,
        optimal_threshold=threshold,
        validation_f0_5=float(validation_metrics["macro_f0_5"]),
    )
    scored_candidates = _score_candidates(test_candidates, test_features, final_model)
    if stage == "scoring":
        score_columns = [
            "source1_id",
            "target_source",
            "target_id",
            "blocking_methods",
            "blocking_score",
            "confidence",
            "rule_applied",
            *FEATURE_NAMES,
        ]
        write_tsv(
            settings.cache_dir / "test_scored_pairs.tsv",
            scored_candidates,
            score_columns,
        )
        return {
            "test_candidate_pairs": len(test_candidates),
            "scored_pairs": len(scored_candidates),
            "model_path": str(settings.model_dir / "scorer.pkl"),
            "threshold": threshold,
        }
    matches = resolve_matches(scored_candidates, threshold, cardinality)

    candidate_columns = [
        "source1_id",
        "target_source",
        "target_id",
        "blocking_methods",
        "blocking_score",
        "confidence",
        "rule_applied",
        *FEATURE_NAMES,
    ]
    matching_columns = [
        "source1_id",
        "target_source",
        "target_id",
        "confidence",
        "decision_reason",
    ]
    write_tsv(settings.output_dir / "candidate_pairs.tsv", scored_candidates, candidate_columns)
    write_tsv(settings.output_dir / "matching_results.tsv", matches, matching_columns)

    # Convert to official submission format (matching_results.tsv & candidate_pairs.tsv)
    from .output_adapter import convert_to_submission
    submission_dir = settings.output_dir / "submission"
    all_test_s1_ids = {record["id"] for record in test1}
    submission_stats = convert_to_submission(
        internal_matches=matches,
        internal_candidates=test_candidates,
        all_test_s1_ids=all_test_s1_ids,
        output_dir=submission_dir,
    )

    test_candidate_counts = Counter(row["source1_id"] for row in test_candidates)
    diagnostics = build_diagnostics(
        validation_truth, validation_scored, validation_predictions
    )
    test_blocking_stats = compute_blocking_stats(test_candidates, None)
    total_possible = len(test1) * (len(test2) + len(test3))
    report = {
        "status": "complete",
        "dataset_mode": dataset_mode,
        "official_submission_allowed": official_submission_allowed,
        "data_root": str(settings.data_root),
        "output_dir": str(settings.output_dir),
        "max_candidates_per_entity": settings.max_candidates_per_entity,
        "configuration": settings.to_dict(),
        "validation": validation_metrics,
        "training": {
            "source1_records": len(train1),
            "source2_records": len(train2),
            "source3_records": len(train3),
            "candidate_pairs": len(all_train_candidates),
            "positive_count": final_hn_stats["positive_count"],
            "hard_negative_count": final_hn_stats["hard_negative_count"],
            "random_negative_count": final_hn_stats["random_negative_count"],
            "hard_negative_ratio": round(final_hn_stats["hard_negative_ratio"], 2),
            "positive_candidate_pairs": sum(all_labels),
            "negative_candidate_pairs": len(all_labels) - sum(all_labels),
        },
        "test": {
            "source1_records": len(test1),
            "source2_records": len(test2),
            "source3_records": len(test3),
            "candidate_pairs": len(test_candidates),
            "possible_pairs": total_possible,
            "reduction_percentage": 100.0
            * (1 - len(test_candidates) / max(1, total_possible)),
            "matches": len(matches),
            "unmatched_source1_records": len(test1)
            - len({row["source1_id"] for row in matches}),
            "average_candidates_per_source1": (
                len(test_candidates) / len(test1) if test1 else 0.0
            ),
            "maximum_candidates_per_source1": max(
                test_candidate_counts.values(), default=0
            ),
        },
        "blocking_stats": test_blocking_stats,
        "submission": submission_stats,
        "threshold": threshold,
        "cardinality": cardinality,
        "diagnostics": diagnostics,
        "output_schema": {
            "candidate_pairs.tsv": candidate_columns,
            "matching_results.tsv": matching_columns,
            "submission_matching_results.tsv": [
                "source1_entity_id",
                "matched_entity_ids",
            ],
            "submission_candidate_pairs.tsv": [
                "source1_entity_id",
                "candidate_entity_ids",
            ],
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "cache_enabled": use_cache,
    }
    _write_json(settings.output_dir / "run_metadata.json", report)
    _write_text_report(settings.output_dir / "validation_report.txt", report)
    _write_json(settings.output_dir / "diagnostics.json", diagnostics)
    _validate_outputs(
        settings.output_dir,
        test1,
        test2,
        test3,
        test_candidates,
        matches,
        settings.max_candidates_per_entity,
    )
    LOGGER.info(
        "[OUTPUT] %d candidates, %d matches; internal integrity checks passed",
        len(test_candidates),
        len(matches),
    )
    if dataset_mode == "demo":
        LOGGER.warning(
            "[OUTPUT] DEMO MODE — outputs are for testing only, "
            "NOT valid for competition submission."
        )
    return report


def _load_and_prepare(settings: ResolvedSettings, cache: CacheStore) -> dict:
    train_dir = settings.data_root / "train"
    test_dir = settings.data_root / "test"
    source_paths = {
        "train": {
            name: train_dir / filename for name, (filename, _) in TRAIN_FILES.items()
        },
        "test": {
            name: test_dir / filename for name, (filename, _) in TEST_FILES.items()
        },
    }
    source_paths["train"]["ground_truth"] = train_dir / "train_ground_truth.tsv"
    raw_records: dict[str, dict[str, list[dict]]] = {"train": {}, "test": {}}
    columns_by_name: dict[str, list[str]] = {}
    all_paths = [
        path
        for collection in source_paths.values()
        for path in collection.values()
    ]
    dataset_fingerprint = file_fingerprint(all_paths)
    LOGGER.info("[LOAD] Read local TSV inputs from %s", settings.data_root)
    for split_name, mapping in (("train", TRAIN_FILES), ("test", TEST_FILES)):
        for logical_source, (filename, cache_name) in mapping.items():
            path = source_paths[split_name][logical_source]
            columns, input_rows = read_tsv(path)
            columns_by_name[cache_name] = columns
            metadata = cache_metadata(
                dataset_fingerprint=file_fingerprint([path]),
                source_name=cache_name,
                row_count=len(input_rows),
                columns=columns,
                configuration={"preprocessing_version": "unicode-name-address-v1"},
                feature_version="features-v1",
            )
            cached = cache.read_json(f"prepared_{cache_name}", metadata)
            if cached is None:
                loaded = load_source(path, cache_name)
                prepared = [preprocess_record(record) for record in loaded]
                cache.write_json(f"prepared_{cache_name}", metadata, prepared)
            else:
                prepared = cached
            raw_records[split_name][logical_source] = prepared
            LOGGER.info(
                "[SCHEMA] %s rows=%d columns=%s",
                cache_name,
                len(prepared),
                ", ".join(columns),
            )

    ground_truth_rows = load_ground_truth_rows(source_paths["train"]["ground_truth"])
    truth = infer_ground_truth_pairs(
        ground_truth_rows,
        {record["id"] for record in raw_records["train"]["source1"]},
        {record["id"] for record in raw_records["train"]["source2"]},
        {record["id"] for record in raw_records["train"]["source3"]},
    )

    # Multi-branch chain detection (Upgrade 14)
    all_loaded_records = [
        rec
        for split_dict in raw_records.values()
        for rec_list in split_dict.values()
        for rec in rec_list
    ]
    brand_counts = Counter(
        rec.get("brand_name")
        for rec in all_loaded_records
        if rec.get("brand_name")
    )
    chain_brands = {brand for brand, count in brand_counts.items() if count >= 3}
    for rec in all_loaded_records:
        rec["is_chain"] = rec.get("brand_name") in chain_brands

    summary = schema_summary(
        {
            f"{split}_{source}": records
            for split, group in raw_records.items()
            for source, records in group.items()
        }
    )
    return {
        **raw_records,
        "truth": truth,
        "summary": summary,
        "columns": columns_by_name,
        "dataset_fingerprint": dataset_fingerprint,
    }


def _inspect(data: dict) -> dict:
    return {
        "sources": data["summary"],
        "ground_truth_pairs": len(data["truth"]),
        "cardinality_inferred_from_training": infer_cardinality(data["truth"]),
        "dataset_fingerprint": data["dataset_fingerprint"],
    }


def _split_source1_ids(
    source1_ids: list[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    ordered = sorted(source1_ids)
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, round(len(ordered) * validation_fraction))
    validation_count = min(validation_count, len(ordered) - 1)
    validation_ids = sorted(ordered[:validation_count])
    fit_ids = sorted(ordered[validation_count:])
    return fit_ids, validation_ids


def _without_validation_only_targets(
    targets: list[dict],
    fit_truth: set[tuple[str, str, str]],
    validation_truth: set[tuple[str, str, str]],
    target_source: str,
) -> list[dict]:
    fit_target_ids = {
        target_id
        for _, source, target_id in fit_truth
        if source == target_source
    }
    validation_target_ids = {
        target_id
        for _, source, target_id in validation_truth
        if source == target_source
    }
    validation_only = validation_target_ids - fit_target_ids
    return [target for target in targets if target["id"] not in validation_only]


def _build_candidates(
    source1_records: list[dict],
    targets_by_source: dict[str, list[dict]],
    settings: ResolvedSettings,
    *,
    cache: CacheStore | None = None,
    cache_name: str = "",
    dataset_fingerprint: str = "",
) -> list[dict]:
    if cache is not None and cache_name:
        metadata = cache_metadata(
            dataset_fingerprint=dataset_fingerprint,
            source_name=cache_name,
            row_count=len(source1_records),
            columns=["source1_id", "target_source", "target_id"],
            configuration={
                "settings": settings.to_dict(),
                "source1_id_fingerprint": _json_fingerprint(
                    [row["id"] for row in source1_records]
                ),
                "target_id_fingerprint": _json_fingerprint(
                    {
                        source: [row["id"] for row in records]
                        for source, records in targets_by_source.items()
                    }
                ),
            },
            feature_version="blocking-v1",
        )
        cached = cache.read_json(cache_name, metadata)
        if cached is not None:
            LOGGER.info("[CACHE] Reusing %s (%d candidates)", cache_name, len(cached))
            return cached
    pairs = []
    for target_source in ("source2", "source3"):
        quota = (
            settings.source2_top_k
            if target_source == "source2"
            else settings.source3_top_k
        )
        pairs.extend(
            generate_candidates(
                source1_records,
                targets_by_source[target_source],
                target_source,
                seed=settings.seed,
                permutations=settings.minhash_permutations,
                bands=settings.minhash_bands,
                ngram_size=settings.ngram_size,
                top_k=quota,
                max_posting_size=settings.max_posting_size,
                dense_model_name=settings.dense_model_name,
                dense_top_k=settings.dense_top_k,
            )
        )
    source1_country = {
        record["id"]: record.get("country_normalized", "")
        for record in source1_records
    }
    for pair in pairs:
        pair["source1_country"] = source1_country.get(pair["source1_id"], "")
    pairs.sort(
        key=lambda row: (
            row["source1_id"],
            row["target_source"],
            -row["blocking_score"],
            row["target_id"],
        )
    )
    if cache is not None and cache_name:
        cache.write_json(cache_name, metadata, pairs)
    return pairs


def _feature_rows(
    candidates: list[dict],
    source1_records: list[dict],
    targets_by_source: dict[str, list[dict]],
    idf: dict[str, float],
    *,
    cache: CacheStore | None = None,
    cache_name: str = "",
) -> list[dict[str, float]]:
    metadata = None
    if cache is not None and cache_name:
        metadata = cache_metadata(
            dataset_fingerprint=_json_fingerprint(
                {"candidates": candidates, "idf": idf}
            ),
            source_name=cache_name,
            row_count=len(candidates),
            columns=FEATURE_NAMES,
            configuration={
                "source1_ids": _json_fingerprint(
                    [row["id"] for row in source1_records]
                ),
                "idf": _json_fingerprint(idf),
                "record_content": _json_fingerprint(
                    {"source1": source1_records, "targets": targets_by_source}
                ),
            },
            feature_version="features-v1",
        )
        cached = cache.read_json(cache_name, metadata)
        if cached is not None:
            LOGGER.info("[CACHE] Reusing %s (%d feature rows)", cache_name, len(cached))
            return cached
    source1_by_id = {record["id"]: record for record in source1_records}
    target_maps = {
        source: {record["id"]: record for record in records}
        for source, records in targets_by_source.items()
    }
    features = []
    for candidate in candidates:
        left = source1_by_id[candidate["source1_id"]]
        right = target_maps[candidate["target_source"]][candidate["target_id"]]
        features.append(
            compute_features(
                left,
                right,
                candidate["blocking_methods"],
                candidate["blocking_score"],
                idf,
            )
        )
    if cache is not None and cache_name and metadata is not None:
        cache.write_json(cache_name, metadata, features)
    return features


def _write_feature_table(
    path: Path,
    candidates: list[dict],
    feature_rows: list[dict[str, float]],
) -> None:
    rows = [
        {
            "source1_id": candidate["source1_id"],
            "target_source": candidate["target_source"],
            "target_id": candidate["target_id"],
            **features,
        }
        for candidate, features in zip(candidates, feature_rows)
    ]
    write_tsv(
        path,
        rows,
        ["source1_id", "target_source", "target_id", *FEATURE_NAMES],
    )


def _json_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _mine_hard_negatives(
    candidates: list[dict],
    features: list[dict[str, float]],
    truth: set[tuple[str, str, str]],
    seed: int = 42,
    max_negative_ratio: float = 10.0,
) -> tuple[list[dict[str, float]], list[int], dict[str, Any]]:
    """
    Hard-negative mining (Upgrade 8):
    Identifies candidate pairs generated by blocking that are not in ground truth,
    prioritizes them by difficulty (high name similarity, token set ratio, postal exact, blocking score),
    and balances them against verified positive pairs.
    """
    positives: list[tuple[dict, dict[str, float], int]] = []
    hard_negatives: list[tuple[dict, dict[str, float], int, float]] = []

    for cand, feat in zip(candidates, features):
        is_pos = (
            cand["source1_id"],
            cand["target_source"],
            cand["target_id"],
        ) in truth
        if is_pos:
            positives.append((cand, feat, 1))
        else:
            diff = (
                feat.get("name_edit_similarity", 0.0) * 0.35
                + feat.get("name_token_set_ratio", 0.0) * 0.25
                + float(feat.get("postal_exact", 0.0)) * 0.2
                + float(cand.get("blocking_score", 0.0)) * 0.2
            )
            hard_negatives.append((cand, feat, 0, diff))

    # Prioritize difficult negatives
    hard_negatives.sort(key=lambda item: -item[3])
    max_negatives = max(int(len(positives) * max_negative_ratio), 50)
    selected_negatives = hard_negatives[:max_negatives]

    training_data = [(c, f, lbl) for c, f, lbl in positives] + [
        (c, f, lbl) for c, f, lbl, _ in selected_negatives
    ]
    random.Random(seed).shuffle(training_data)

    training_features = [f for _, f, _ in training_data]
    training_labels = [lbl for _, _, lbl in training_data]

    stats = {
        "positive_count": len(positives),
        "hard_negative_count": len(selected_negatives),
        "random_negative_count": 0,
        "hard_negative_ratio": (
            len(selected_negatives) / max(1, len(positives))
        ),
    }
    return training_features, training_labels, stats


def _score_candidates(
    candidates: list[dict],
    feature_rows: list[dict[str, float]],
    scorer: MatchScorer,
) -> list[dict]:
    probabilities = scorer.predict_positive_probability(feature_rows)
    result = []
    for candidate, features, probability in zip(candidates, feature_rows, probabilities):
        is_chain = bool(features.get("is_chain")) or bool(candidate.get("is_chain"))
        rule_applied, rule_probability, rule_name = deterministic_rule(features, is_chain=is_chain)
        row = {
            **candidate,
            **features,
            "confidence": rule_probability if rule_applied else probability,
            "rule_applied": rule_name,
            "decision_reason": (
                f"rule:{rule_name}" if rule_applied else "random_forest_probability"
            ),
        }
        result.append(row)
    result.sort(
        key=lambda row: (
            row["source1_id"],
            row["target_source"],
            -row["confidence"],
            row["target_id"],
        )
    )
    return result


def _validate_outputs(
    output_dir: Path,
    source1_records: list[dict],
    source2_records: list[dict],
    source3_records: list[dict],
    candidates: list[dict],
    matches: list[dict],
    candidate_cap: int,
) -> None:
    source1_ids = {record["id"] for record in source1_records}
    target_ids = {
        "source2": {record["id"] for record in source2_records},
        "source3": {record["id"] for record in source3_records},
    }
    candidate_keys = {
        (row["source1_id"], row["target_source"], row["target_id"])
        for row in candidates
    }
    if len(candidate_keys) != len(candidates):
        raise ValueError("Duplicate candidate pair detected.")
    counts = Counter(row["source1_id"] for row in candidates)
    if any(count > candidate_cap * 2 for count in counts.values()):
        raise ValueError(
            "Candidate cap exceeded: the configured cap applies separately to "
            "Source 2 and Source 3, so the combined maximum is twice the cap."
        )
    match_keys = set()
    for row in matches:
        key = (row["source1_id"], row["target_source"], row["target_id"])
        if key not in candidate_keys:
            raise ValueError(f"Match is not in candidate_pairs.tsv: {key}")
        if key in match_keys:
            raise ValueError(f"Duplicate final match: {key}")
        if row["source1_id"] not in source1_ids:
            raise ValueError(f"Invalid Source 1 ID in match: {row['source1_id']}")
        if row["target_id"] not in target_ids.get(row["target_source"], set()):
            raise ValueError(f"Invalid target ID in match: {row['target_id']}")
        match_keys.add(key)
    if not (output_dir / "candidate_pairs.tsv").is_file():
        raise ValueError("candidate_pairs.tsv was not written.")
    if not (output_dir / "matching_results.tsv").is_file():
        raise ValueError("matching_results.tsv was not written.")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _write_text_report(path: Path, report: dict) -> None:
    validation = report["validation"]
    test = report["test"]
    lines = [
        "Offline Business Entity Resolution — Run Report",
        "=" * 53,
        f"Status: {report['status']}",
        f"Selected threshold: {report['threshold']:.4f}",
        f"Validation macro F0.5: {validation['macro_f0_5']:.4f}",
        f"Validation precision: {validation['micro_precision']:.4f}",
        f"Validation recall: {validation['micro_recall']:.4f}",
        f"Validation candidate recall: {validation['candidate_recall']:.4f}",
        f"Validation false positives: {validation['false_positives']}",
        f"Validation false negatives: {validation['false_negatives']}",
        f"Test candidates: {test['candidate_pairs']}",
        f"Test accepted matches: {test['matches']}",
        f"Test unmatched Source 1 records: {test['unmatched_source1_records']}",
        f"Runtime seconds: {report['runtime_seconds']}",
        "",
        "Cardinality inferred from training ground truth:",
    ]
    lines.extend(
        f"  {source}: {cardinality}"
        for source, cardinality in report["cardinality"].items()
    )
    lines += [
        "",
        "The output uses the pipeline's canonical long-form TSV schema. "
        "Challenge-specific validation still requires the official validator.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")