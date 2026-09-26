import unittest

from src.ber_pipeline.blocking import generate_candidates
from src.ber_pipeline.preprocess import preprocess_record


def record(record_id, name, address, postal, city="Paris", country="France"):
    return preprocess_record(
        {
            "id": record_id,
            "business_name": name,
            "business_address": address,
            "postal_code": postal,
            "city": city,
            "country": country,
        }
    )


class BlockingTests(unittest.TestCase):
    def test_independent_blockers_union_evidence(self):
        source1 = [record("s1", "Atelier Nova", "27 Rue de la Paix", "33000")]
        targets = [
            record("t1", "Atelier Nova", "27 Rue de la Paix", "33000"),
            record("t2", "Unrelated Shop", "99 Other Road", "75000"),
        ]
        pairs = generate_candidates(
            source1,
            targets,
            "source2",
            seed=42,
            permutations=24,
            bands=6,
            top_k=1,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["target_id"], "t1")
        self.assertEqual(
            set(pairs[0]["blocking_methods"].split(",")),
            {"minhash", "token_posting"},
        )

    def test_candidate_generation_is_stable_and_bounded(self):
        source1 = [record(f"s{index}", "Acme Bakery", "14 Market Street", "10001") for index in range(3)]
        targets = [
            record(f"t{index}", f"Acme Bakery {index}", f"{index + 1} Market Street", "10001")
            for index in range(10)
        ]
        first = generate_candidates(
            source1, targets, "source3", permutations=24, bands=6, top_k=2
        )
        second = generate_candidates(
            source1, targets, "source3", permutations=24, bands=6, top_k=2
        )
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), len(source1) * 2)
        self.assertTrue(all(row["target_id"].startswith("t") for row in first))

    def test_directional_target_source_enforcement(self):
        source1 = [record("s1", "Acme Bakery", "14 Market Street", "10001")]
        targets = [record("t1", "Acme Bakery", "14 Market Street", "10001")]
        with self.assertRaises(ValueError):
            generate_candidates(source1, targets, "source1")

    def test_self_match_is_rejected(self):
        source1 = [record("s1", "Acme Bakery", "14 Market Street", "10001")]
        targets = [
            record("s1", "Acme Bakery", "14 Market Street", "10001"),
            record("t1", "Acme Bakery", "14 Market Street", "10001"),
        ]
        pairs = generate_candidates(source1, targets, "source2", permutations=24, bands=6)
        self.assertTrue(all(p["target_id"] != "s1" for p in pairs))


if __name__ == "__main__":
    unittest.main()