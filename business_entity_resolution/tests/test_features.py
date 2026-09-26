import unittest

from src.ber_pipeline.features import FEATURE_NAMES, compute_features, deterministic_rule, fit_idf
from src.ber_pipeline.preprocess import preprocess_record


def prepared(record_id, name, address, postal, city="Paris", country="France"):
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


class FeatureTests(unittest.TestCase):
    def test_exact_pair_features_and_rule(self):
        left = prepared("a", "Maison Lumiere SAS", "18 Rue des Fleurs", "75001")
        right = prepared("b", "Maison Lumiere", "18 Rue des Fleurs", "75001")
        idf = fit_idf([left, right])
        features = compute_features(left, right, ["minhash", "token_posting"], 0.9, idf)
        self.assertEqual(set(features), set(FEATURE_NAMES))
        self.assertEqual(features["brand_exact"], 1.0)
        self.assertEqual(features["postal_exact"], 1.0)
        self.assertEqual(features["street_number_exact"], 1.0)
        self.assertEqual(features["multiple_blockers_agree"], 1.0)
        accepted, confidence, rule_name = deterministic_rule(features)
        self.assertTrue(accepted)
        self.assertGreater(confidence, 0.99)
        self.assertEqual(rule_name, "exact_brand_street_address_country")

    def test_mismatch_and_missing_fields_are_distinct(self):
        left = prepared("a", "Orchid Books", "35 King Road", "")
        right = prepared("b", "Harbor View Inn", "", "")
        features = compute_features(left, right)
        self.assertEqual(features["name_exact"], 0.0)
        self.assertEqual(features["name_char_jaccard"], 0.0)
        self.assertEqual(features["postal_exact"], 0.0)
        self.assertEqual(features["address_missing_either"], 1.0)
        self.assertEqual(features["postal_prefix_match"], 0.0)

    def test_levenshtein_similarity_is_normalized(self):
        left = prepared("a", "Acme Bakery", "14 Market Street", "10001")
        near = prepared("b", "Acme Baker", "14 Market Street", "10001")
        far = prepared("c", "Harbor View", "14 Market Street", "10001")
        self.assertGreater(
            compute_features(left, near)["name_edit_similarity"],
            compute_features(left, far)["name_edit_similarity"],
        )


    def test_tristate_street_number(self):
        # +1 agreement
        p1 = prepared("a", "Alpha", "14 Market St", "10001")
        p2 = prepared("b", "Alpha", "14 Market St", "10001")
        self.assertEqual(compute_features(p1, p2)["street_number_exact"], 1.0)

        # 0 missing/insufficient
        p3 = prepared("c", "Alpha", "Market St", "10001")
        self.assertEqual(compute_features(p1, p3)["street_number_exact"], 0.0)
        self.assertEqual(compute_features(p3, p3)["street_number_exact"], 0.0)

        # -1 conflict
        p4 = prepared("d", "Alpha", "104 Market St", "10001")
        self.assertEqual(compute_features(p1, p4)["street_number_exact"], -1.0)

    def test_phonetic_similarity(self):
        p1 = prepared("a", "Laxmi General Store", "14 Market St", "10001")
        p2 = prepared("b", "Lakshmi General Store", "14 Market St", "10001")
        features = compute_features(p1, p2)
        self.assertGreater(features["name_phonetic_similarity"], 0.5)

    def test_restricted_auto_accept_requires_street_number(self):
        # Exact name + postal, but NO street number -> MUST NOT auto-accept
        p1 = prepared("a", "Alpha Solutions", "Market St", "10001")
        p2 = prepared("b", "Alpha Solutions", "Market St", "10001")
        features = compute_features(p1, p2)
        accepted, _, _ = deterministic_rule(features)
        self.assertFalse(accepted)

        # Exact name + postal + explicit street number agreement -> DOES auto-accept
        p3 = prepared("c", "Alpha Solutions", "14 Market St", "10001")
        p4 = prepared("d", "Alpha Solutions", "14 Market St", "10001")
        features_with_num = compute_features(p3, p4)
        accepted_with_num, conf, rule = deterministic_rule(features_with_num)
        self.assertTrue(accepted_with_num)
        self.assertEqual(rule, "exact_name_postal_number_country")

    def test_chain_guard_suppresses_auto_accept(self):
        # Even with exact name, postal, and street number, chain entities are NOT auto-accepted
        p1 = prepared("a", "Starbucks Coffee", "14 Market St", "10001")
        p2 = prepared("b", "Starbucks Coffee", "14 Market St", "10001")
        p1["is_chain"] = True
        features = compute_features(p1, p2)
        accepted, _, _ = deterministic_rule(features, is_chain=True)
        self.assertFalse(accepted)


if __name__ == "__main__":
    unittest.main()