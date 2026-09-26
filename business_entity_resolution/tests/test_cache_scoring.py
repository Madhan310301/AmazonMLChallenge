import tempfile
import unittest
from pathlib import Path

from src.ber_pipeline.cache import CacheStore
from src.ber_pipeline.features import FEATURE_NAMES
from src.ber_pipeline.scoring import MatchScorer


class CacheAndScoringTests(unittest.TestCase):
    def test_cache_rejects_stale_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = CacheStore(Path(temp))
            cache.write_json("item", {"fingerprint": "one"}, {"rows": [1]})
            self.assertEqual(
                cache.read_json("item", {"fingerprint": "one"}), {"rows": [1]}
            )
            self.assertIsNone(cache.read_json("item", {"fingerprint": "two"}))

    def test_random_forest_fits_and_returns_probabilities(self):
        positive = {name: 0.0 for name in FEATURE_NAMES}
        positive.update(
            {
                "name_exact": 1.0,
                "brand_exact": 1.0,
                "name_tfidf_cosine": 1.0,
                "postal_exact": 1.0,
                "street_number_exact": 1.0,
            }
        )
        negative = {name: 0.0 for name in FEATURE_NAMES}
        negative.update({"name_edit_similarity": 0.1, "address_token_jaccard": 0.0})
        training_rows = [positive.copy() for _ in range(20)] + [
            negative.copy() for _ in range(20)
        ]
        training_labels = [1] * 20 + [0] * 20
        model = MatchScorer(seed=17, estimators=50).fit(
            training_rows, training_labels
        )
        probabilities = model.predict_positive_probability([positive, negative])
        self.assertEqual(len(probabilities), 2)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))
        self.assertGreater(probabilities[0], probabilities[1])
        with tempfile.TemporaryDirectory() as temp:
            model.save(Path(temp))
            loaded = MatchScorer.load(Path(temp) / "scorer.pkl")
            self.assertEqual(
                loaded.predict_positive_probability([positive, negative]),
                probabilities,
            )


if __name__ == "__main__":
    unittest.main()