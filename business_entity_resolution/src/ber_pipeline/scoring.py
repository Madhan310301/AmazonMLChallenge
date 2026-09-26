from __future__ import annotations

import json
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from .features import FEATURE_NAMES


class MatchScorer:
    """Precision-oriented random-forest pair classifier."""

    def __init__(self, seed: int = 42, estimators: int = 240, max_depth: int | None = None):
        self.seed = seed
        self.estimators = estimators
        self.max_depth = max_depth
        self.model: RandomForestClassifier | None = None

    def fit(self, feature_rows: list[dict[str, float]], labels: list[int]) -> "MatchScorer":
        classes = set(labels)
        if classes != {0, 1}:
            positive_count = sum(labels)
            negative_count = len(labels) - positive_count
            raise ValueError(
                "The generated training candidates must contain both matches and "
                f"non-matches (got {positive_count} positive, {negative_count} negative). "
                "Inspect candidate blocking recall and ground-truth columns."
            )
        matrix = _matrix(feature_rows)
        self.model = RandomForestClassifier(
            n_estimators=self.estimators,
            max_depth=self.max_depth,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(matrix, labels)
        return self

    def predict_positive_probability(
        self, feature_rows: list[dict[str, float]]
    ) -> list[float]:
        if self.model is None:
            raise RuntimeError("Scorer has not been fitted.")
        if not feature_rows:
            return []
        probabilities = self.model.predict_proba(_matrix(feature_rows))
        positive_column = list(self.model.classes_).index(1)
        return [float(value) for value in probabilities[:, positive_column]]

    def save(
        self,
        model_dir: Path,
        optimal_threshold: float | None = None,
        validation_f0_5: float | None = None,
    ) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted scorer.")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "scorer.pkl"
        temp_path = model_path.with_suffix(".pkl.tmp")
        with temp_path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(model_path)
        metadata = {
            "model": "sklearn.ensemble.RandomForestClassifier",
            "seed": self.seed,
            "n_estimators": self.estimators,
            "max_depth": self.max_depth,
            "features": FEATURE_NAMES,
            "probability_calibration": "not applied; threshold tuned on held-out training split",
            "optimal_threshold": float(optimal_threshold) if optimal_threshold is not None else 0.82,
            "validation_f0_5": float(validation_f0_5) if validation_f0_5 is not None else None,
        }
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, model_path: Path) -> "MatchScorer":
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved scorer not found: {model_path}")
        with model_path.open("rb") as stream:
            scorer = pickle.load(stream)
        if not isinstance(scorer, cls) or scorer.model is None:
            raise ValueError(f"Invalid saved scorer file: {model_path}")
        return scorer


def _matrix(rows: list[dict[str, float]]) -> list[list[float]]:
    return [
        [float(row.get(name, 0.0)) for name in FEATURE_NAMES]
        for row in rows
    ]