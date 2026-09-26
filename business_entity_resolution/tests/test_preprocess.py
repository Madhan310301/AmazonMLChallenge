import unittest

from src.ber_pipeline.preprocess import normalize_name, parse_address, preprocess_record


class PreprocessTests(unittest.TestCase):
    def test_legal_suffix_is_separated(self):
        normalized, brand, legal_form, tokens = normalize_name("XYZ Pvt. Ltd.")
        self.assertEqual(normalized, "xyz pvt ltd")
        self.assertEqual(brand, "xyz")
        self.assertEqual(legal_form, "private limited")
        self.assertEqual(tokens, ["xyz"])

    def test_punctuation_case_and_accent_normalization(self):
        self.assertEqual(normalize_name("  Café & Co.  ")[1], "cafe and")
        self.assertEqual(normalize_name("O'Neil—Works")[1], "oneil works")

    def test_unicode_letters_are_preserved(self):
        self.assertEqual(normalize_name("東京商店株式会社")[1], "東京商店株式会社")

    def test_address_parsing_uses_explicit_postal_and_street_number(self):
        result = parse_address(
            {
                "business_address": "14 Market Street, New York",
                "postal_code": "10001",
                "city": "New York",
                "country": "United States",
            }
        )
        self.assertEqual(result["street_number"], "14")
        self.assertEqual(result["postal_code_normalized"], "10001")
        self.assertEqual(result["city_normalized"], "new york")

    def test_address_parser_does_not_treat_street_number_as_postal_code(self):
        result = parse_address(
            {"business_address": "14 Market Street, Paris", "country": "France"}
        )
        self.assertEqual(result["postal_code_normalized"], "")

    def test_original_fields_are_preserved(self):
        record = preprocess_record(
            {
                "id": "x",
                "business_name": "Café Lumière SAS",
                "business_address": "8 Rue Victor Hugo",
                "country": "France",
            }
        )
        self.assertEqual(record["raw_business_name"], "Café Lumière SAS")
        self.assertEqual(record["raw_address"], "8 Rue Victor Hugo")
        self.assertEqual(record["brand_name"], "cafe lumiere")

    def test_unicode_nfkd_french_accents(self):
        self.assertEqual(normalize_name("Société Générale")[0], "societe generale")
        self.assertEqual(normalize_name("Hôtel Mercure")[0], "hotel mercure")

    def test_french_commercial_routing(self):
        record = {"business_address": "75008 Paris Cedex 08"}
        result = parse_address(record)
        self.assertEqual(result["commercial_routing"], "Cedex 08")
        self.assertEqual(result["street_number"], "")

        record_bp = {"business_address": "BP 402, 12 Rue de la Paix"}
        result_bp = parse_address(record_bp)
        self.assertEqual(result_bp["commercial_routing"], "BP 402")
        self.assertEqual(result_bp["street_number"], "12")

    def test_landmark_extraction(self):
        record = {
            "business_address": "Near SBI ATM, Opposite Metro Pillar 42, 104 MG Road",
            "city": "Bengaluru",
        }
        result = parse_address(record)
        self.assertEqual(result["street_number"], "104")
        self.assertIn("sbi", result["landmark_tokens"])
        self.assertIn("pillar", result["landmark_tokens"])
        self.assertIn("104 mg road", result["core_address"])

    def test_corsica_french_postal_handling(self):
        record_2a = {"business_address": "Rue Principale, 2A004 Ajaccio"}
        result_2a = parse_address(record_2a)
        self.assertEqual(result_2a["postal_code_normalized"], "2A004")

        record_2b = {"business_address": "Boulevard Paoli, 2B200 Bastia"}
        result_2b = parse_address(record_2b)
        self.assertEqual(result_2b["postal_code_normalized"], "2B200")


if __name__ == "__main__":
    unittest.main()