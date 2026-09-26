from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    data_root: str = "dataset"
    output_dir: str = "output"
    cache_dir: str = "cache"
    model_dir: str = "models"
    seed: int = 42
    validation_fraction: float = 0.2
    minhash_permutations: int = 48
    minhash_bands: int = 12
    ngram_size: int = 3
    max_candidates_per_entity: int = 60
    max_posting_size: int = 500
    rf_estimators: int = 240
    threshold_min: float = 0.5
    threshold_max: float = 0.99
    threshold_steps: int = 50
    cardinality: str = "infer"
    random_forest_max_depth: int | None = None
    dense_model_name: str | None = None
    dense_top_k: int = 20
    source2_top_k: int = 8
    source3_top_k: int = 8

    def resolved(self, root: Path = PROJECT_ROOT) -> "ResolvedSettings":
        return ResolvedSettings(
            data_root=(root / self.data_root).resolve(),
            output_dir=(root / self.output_dir).resolve(),
            cache_dir=(root / self.cache_dir).resolve(),
            model_dir=(root / self.model_dir).resolve(),
            seed=self.seed,
            validation_fraction=self.validation_fraction,
            minhash_permutations=self.minhash_permutations,
            minhash_bands=self.minhash_bands,
            ngram_size=self.ngram_size,
            max_candidates_per_entity=self.max_candidates_per_entity,
            max_posting_size=self.max_posting_size,
            rf_estimators=self.rf_estimators,
            threshold_min=self.threshold_min,
            threshold_max=self.threshold_max,
            threshold_steps=self.threshold_steps,
            cardinality=self.cardinality,
            random_forest_max_depth=self.random_forest_max_depth,
            dense_model_name=self.dense_model_name,
            dense_top_k=self.dense_top_k,
            source2_top_k=self.source2_top_k,
            source3_top_k=self.source3_top_k,
        )


@dataclass(frozen=True)
class ResolvedSettings:
    data_root: Path
    output_dir: Path
    cache_dir: Path
    model_dir: Path
    seed: int
    validation_fraction: float
    minhash_permutations: int
    minhash_bands: int
    ngram_size: int
    max_candidates_per_entity: int
    max_posting_size: int
    rf_estimators: int
    threshold_min: float
    threshold_max: float
    threshold_steps: int
    cardinality: str
    random_forest_max_depth: int | None
    dense_model_name: str | None
    dense_top_k: int
    source2_top_k: int
    source3_top_k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


def load_settings(config_path: Path | None = None) -> Settings:
    if config_path is None:
        return Settings()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    allowed = {field.name for field in fields(Settings)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")
    settings = Settings(**raw)
    if not 0 < settings.validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if settings.max_candidates_per_entity < 1:
        raise ValueError("max_candidates_per_entity must be positive.")
    if settings.minhash_permutations % settings.minhash_bands:
        raise ValueError(
            "minhash_permutations must be divisible by minhash_bands."
        )
    if settings.cardinality not in {
        "infer",
        "one_to_one",
        "one_to_many",
        "many_to_one",
        "many_to_many",
    }:
        raise ValueError(f"Unsupported cardinality: {settings.cardinality}")
    return settings