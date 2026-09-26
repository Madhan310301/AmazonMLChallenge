import unittest

from src.ber_pipeline.schema import infer_cardinality, infer_ground_truth_pairs, resolve_column


class SchemaTests(unittest.TestCase):
    def test_alias_resolution_is_case_and_punctuation_tolerant(self):
        self.assertEqual(
            resolve_column(["Business ID", "Company Name"], "id"), "Business ID"
        )
        self.assertEqual(
            resolve_column(["Business ID", "Company Name"], "business_name"),
            "Company Name",
        )

    def test_wide_ground_truth_and_cardinality_inference(self):
        rows = [
            {"source1_id": "a", "source2_id": "t1", "source3_id": "u1;u2"},
            {"source1_id": "b", "source2_id": "t2", "source3_id": ""},
        ]
        pairs = infer_ground_truth_pairs(
            rows, {"a", "b"}, {"t1", "t2"}, {"u1", "u2"}
        )
        self.assertEqual(
            pairs,
            {
                ("a", "source2", "t1"),
                ("a", "source3", "u1"),
                ("a", "source3", "u2"),
                ("b", "source2", "t2"),
            },
        )
        cardinality = infer_cardinality(pairs)
        self.assertEqual(cardinality["source2"], "one_to_one")
        self.assertEqual(cardinality["source3"], "one_to_many")

    def test_long_ground_truth(self):
        pairs = infer_ground_truth_pairs(
            [
                {"source1_id": "a", "target_source": "Source 2", "target_id": "t1"}
            ],
            {"a"},
            {"t1"},
            set(),
        )
        self.assertEqual(pairs, {("a", "source2", "t1")})

    def test_unknown_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown source2 ID"):
            infer_ground_truth_pairs(
                [{"source1_id": "a", "source2_id": "missing", "source3_id": ""}],
                {"a"},
                set(),
                set(),
            )


if __name__ == "__main__":
    unittest.main()