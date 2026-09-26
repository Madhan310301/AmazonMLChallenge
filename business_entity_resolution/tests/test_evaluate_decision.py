import unittest

from src.ber_pipeline.decision import resolve_matches
from src.ber_pipeline.evaluate import macro_fbeta, tune_threshold


class EvaluationAndDecisionTests(unittest.TestCase):
    def test_macro_f0_5_hand_calculation(self):
        truth = {("a", "source2", "x")}
        predictions = [
            {"source1_id": "a", "target_source": "source2", "target_id": "x"},
            {"source1_id": "a", "target_source": "source2", "target_id": "y"},
        ]
        metrics = macro_fbeta(truth, predictions, {"a", "b"})
        self.assertAlmostEqual(metrics["macro_f0_5"], 0.7777777777, places=7)
        self.assertAlmostEqual(metrics["micro_precision"], 0.5)
        self.assertAlmostEqual(metrics["micro_recall"], 1.0)

    def test_one_to_one_uses_global_assignment_and_supports_no_match(self):
        rows = [
            {"source1_id": "a", "target_source": "source2", "target_id": "x", "confidence": 0.90},
            {"source1_id": "a", "target_source": "source2", "target_id": "y", "confidence": 0.80},
            {"source1_id": "b", "target_source": "source2", "target_id": "x", "confidence": 0.85},
            {"source1_id": "b", "target_source": "source2", "target_id": "y", "confidence": 0.20},
        ]
        selected = resolve_matches(rows, 0.5, {"source2": "one_to_one"})
        self.assertEqual(
            {(row["source1_id"], row["target_id"]) for row in selected},
            {("a", "y"), ("b", "x")},
        )

    def test_many_to_many_preserves_multiple_matches(self):
        rows = [
            {"source1_id": "a", "target_source": "source3", "target_id": "x", "confidence": 0.9},
            {"source1_id": "a", "target_source": "source3", "target_id": "y", "confidence": 0.8},
            {"source1_id": "b", "target_source": "source3", "target_id": "x", "confidence": 0.7},
        ]
        selected = resolve_matches(rows, 0.5, {"source3": "many_to_many"})
        self.assertEqual(len(selected), 3)

    def test_threshold_tuning_chooses_precision_safe_threshold(self):
        scored = [
            {"source1_id": "a", "target_source": "source2", "target_id": "x", "confidence": 0.95},
            {"source1_id": "b", "target_source": "source2", "target_id": "y", "confidence": 0.55},
        ]
        threshold, predictions = tune_threshold(
            scored,
            {("a", "source2", "x")},
            {"a", "b"},
            {"source2": "one_to_one"},
            lower=0.5,
            upper=0.99,
            steps=10,
        )
        self.assertGreater(threshold, 0.55)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["target_id"], "x")


    def test_sweep_thresholds_tracks_all_metrics(self):
        scored = [
            {"source1_id": "a", "target_source": "source2", "target_id": "x", "confidence": 0.95},
            {"source1_id": "b", "target_source": "source2", "target_id": "y", "confidence": 0.55},
        ]
        from src.ber_pipeline.evaluate import sweep_thresholds
        history = sweep_thresholds(
            scored,
            {("a", "source2", "x")},
            {"a", "b"},
            {"source2": "one_to_one"},
            lower=0.5,
            upper=0.9,
            steps=5,
        )
        self.assertEqual(len(history), 5)
        for entry in history:
            self.assertIn("macro_f0_5", entry)
            self.assertIn("false_merges", entry)
            self.assertIn("matched_entities", entry)
            self.assertIn("unmatched_entities", entry)

    def test_global_conflict_resolution(self):
        from src.ber_pipeline.decision import resolve_conflicts_globally
        # Multiple S1 competing for target 'x'
        rows = [
            {"source1_id": "s1", "target_source": "source2", "target_id": "x", "confidence": 0.70},
            {"source1_id": "s2", "target_source": "source2", "target_id": "x", "confidence": 0.92},
            {"source1_id": "s3", "target_source": "source2", "target_id": "x", "confidence": 0.85},
        ]
        winners = resolve_conflicts_globally(rows)
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]["source1_id"], "s2")
        self.assertEqual(winners[0]["confidence"], 0.92)


if __name__ == "__main__":
    unittest.main()