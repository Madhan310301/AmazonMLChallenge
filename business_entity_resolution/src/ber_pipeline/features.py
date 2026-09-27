from __future__ import annotations

import math
from collections import Counter

try:
    import rapidfuzz.fuzz as rf_fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    import metaphone
    HAS_METAPHONE = True
except ImportError:
    HAS_METAPHONE = False


FEATURE_NAMES = [
    "name_exact",
    "brand_exact",
    "name_token_jaccard",
    "name_char_jaccard",
    "name_edit_similarity",
    "name_token_set_ratio",
    "name_phonetic_similarity",
    "name_tfidf_cosine",
    "name_rare_token_similarity",
    "name_shared_tokens",
    "name_token_count_gap",
    "name_length_ratio",
    "legal_form_exact",
    "legal_form_present_both",
    "address_token_jaccard",
    "address_edit_similarity",
    "address_tfidf_cosine",
    "street_number_exact",
    "street_number_present_both",
    "postal_exact",
    "postal_prefix_match",
    "city_exact",
    "region_exact",
    "country_exact",
    "country_present_both",
    "street_name_similarity",
    "landmark_similarity",
    "name_postal_interaction",
    "name_street_number_interaction",
    "moderate_name_address_interaction",
    "multiple_blockers_agree",
    "minhash_blocked",
    "token_blocked",
    "dense_blocked",
    "blocking_score",
    "is_chain",
    "name_missing_either",
    "address_missing_either",
]


def fit_idf(records: list[dict]) -> dict[str, float]:
    document_count = 0
    document_frequency: Counter[str] = Counter()
    for record in records:
        seen = set(record.get("name_tokens", []))
        seen.update(record.get("address_tokens", []))
        if not seen:
            continue
        document_count += 1
        document_frequency.update(seen)
    if not document_count:
        return {}
    return {
        token: math.log((document_count + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }


def compute_features(
    left: dict,
    right: dict,
    blocking_methods: list[str] | tuple[str, ...] | str = (),
    blocking_score: float = 0.0,
    idf: dict[str, float] | None = None,
) -> dict[str, float]:
    if isinstance(blocking_methods, str):
        methods = {method for method in blocking_methods.split(",") if method}
    else:
        methods = set(blocking_methods)
    weights = idf or {}
    left_name = left.get("brand_name") or left.get("normalized_business_name", "")
    right_name = right.get("brand_name") or right.get("normalized_business_name", "")
    left_name_tokens = set(left.get("name_tokens", []))
    right_name_tokens = set(right.get("name_tokens", []))
    address_a = set(left.get("address_tokens", []))
    address_b = set(right.get("address_tokens", []))
    name_jaccard = _jaccard(left_name_tokens, right_name_tokens)
    char_jaccard = _jaccard(_ngrams(left_name), _ngrams(right_name))
    name_edit = _similarity(left_name, right_name)
    address_jaccard = _jaccard(address_a, address_b)
    address_edit = _similarity(
        left.get("normalized_address", ""), right.get("normalized_address", "")
    )
    postal_a = left.get("postal_code_normalized", "")
    postal_b = right.get("postal_code_normalized", "")
    street_a = left.get("street_number", "")
    street_b = right.get("street_number", "")
    street_sim = _similarity(
        left.get("street_name", ""), right.get("street_name", "")
    )
    exact_name = bool(
        left.get("normalized_business_name")
        and left.get("normalized_business_name")
        == right.get("normalized_business_name")
    )
    brand_exact = bool(left_name and right_name and left_name == right_name)
    name_cosine = _weighted_cosine(left_name_tokens, right_name_tokens, weights)
    address_cosine = _weighted_cosine(address_a, address_b, weights)
    rarity_similarity = _weighted_jaccard(
        left_name_tokens, right_name_tokens, weights
    )
    postal_match = bool(postal_a and postal_b and postal_a == postal_b)

    # Tri-state street / building number feature (+1 agreement, 0 missing, -1 conflict)
    if street_a and street_b:
        street_tristate = 1.0 if street_a == street_b else -1.0
    else:
        street_tristate = 0.0

    # RapidFuzz token_set_ratio feature
    if HAS_RAPIDFUZZ and left_name and right_name:
        token_set_ratio = float(rf_fuzz.token_set_ratio(left_name, right_name)) / 100.0
    else:
        token_set_ratio = name_jaccard

    # Phonetic similarity feature using Double Metaphone or pure-Python Soundex fallback
    if left_name and right_name:
        phonetic_sim = _phonetic_similarity(
            left.get("name_tokens", []), right.get("name_tokens", [])
        )
    else:
        phonetic_sim = 0.0

    # Landmark similarity feature
    landmark_a = set(left.get("landmark_tokens", []))
    landmark_b = set(right.get("landmark_tokens", []))
    if landmark_a and landmark_b:
        landmark_sim = _jaccard(landmark_a, landmark_b)
    elif left.get("landmark_text") and right.get("landmark_text"):
        landmark_sim = _similarity(left.get("landmark_text", ""), right.get("landmark_text", ""))
    else:
        landmark_sim = 0.0

    country_a = left.get("country_normalized", "")
    country_b = right.get("country_normalized", "")

    # Multi-branch chain flag
    is_chain = float(bool(left.get("is_chain") or right.get("is_chain")))

    features = {
        "name_exact": float(exact_name),
        "brand_exact": float(brand_exact),
        "name_token_jaccard": name_jaccard,
        "name_char_jaccard": char_jaccard,
        "name_edit_similarity": name_edit,
        "name_token_set_ratio": token_set_ratio,
        "name_phonetic_similarity": phonetic_sim,
        "name_tfidf_cosine": name_cosine,
        "name_rare_token_similarity": rarity_similarity,
        "name_shared_tokens": float(len(left_name_tokens & right_name_tokens)),
        "name_token_count_gap": float(abs(len(left_name_tokens) - len(right_name_tokens))),
        "name_length_ratio": _length_ratio(left_name, right_name),
        "legal_form_exact": float(
            bool(left.get("legal_form"))
            and left.get("legal_form") == right.get("legal_form")
        ),
        "legal_form_present_both": float(
            bool(left.get("legal_form")) and bool(right.get("legal_form"))
        ),
        "address_token_jaccard": address_jaccard,
        "address_edit_similarity": address_edit,
        "address_tfidf_cosine": address_cosine,
        "street_number_exact": street_tristate,
        "street_number_present_both": float(bool(street_a) and bool(street_b)),
        "postal_exact": float(postal_match),
        "postal_prefix_match": float(
            bool(postal_a and postal_b and min(len(postal_a), len(postal_b)) >= 2)
            and postal_a[:2] == postal_b[:2]
        ),
        "city_exact": float(
            bool(left.get("city_normalized"))
            and left.get("city_normalized") == right.get("city_normalized")
        ),
        "region_exact": float(
            bool(left.get("region_normalized"))
            and left.get("region_normalized") == right.get("region_normalized")
        ),
        "country_exact": float(bool(country_a and country_b and country_a == country_b)),
        "country_present_both": float(bool(country_a) and bool(country_b)),
        "street_name_similarity": street_sim,
        "landmark_similarity": landmark_sim,
        "name_postal_interaction": name_edit * float(postal_match),
        "name_street_number_interaction": name_edit * float(street_tristate == 1.0),
        "moderate_name_address_interaction": min(name_edit, address_jaccard),
        "multiple_blockers_agree": float(len(methods) > 1),
        "minhash_blocked": float("minhash" in methods),
        "token_blocked": float("token_posting" in methods),
        "dense_blocked": float("dense" in methods),
        "blocking_score": float(blocking_score),
        "is_chain": is_chain,
        "name_missing_either": float(not left_name or not right_name),
        "address_missing_either": float(
            not left.get("normalized_address") or not right.get("normalized_address")
        ),
    }
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def deterministic_rule(
    features: dict[str, float], is_chain: bool = False
) -> tuple[bool, float, str]:
    """
    Precision-safe deterministic auto-accept rule.
    Upgraded with restricted street-number agreement and chain guard.
    """
    # Chain guard (Upgrade 14): multi-branch chain entities must not be auto-accepted
    if is_chain or features.get("is_chain", 0.0) >= 1.0:
        return False, 0.0, ""

    # Restricted auto-accept rule (Upgrade 13):
    # Requires exact name + exact postal + country + EXPLICIT non-empty street number agreement (+1)
    if (
        features.get("name_exact", 0.0) == 1.0
        and features.get("postal_exact", 0.0) == 1.0
        and features.get("country_exact", 0.0) == 1.0
        and features.get("street_number_exact", 0.0) == 1.0
    ):
        return True, 0.999, "exact_name_postal_number_country"

    # Exact brand + explicit street number agreement + street name match + country
    if (
        features.get("brand_exact", 0.0) == 1.0
        and features.get("street_number_exact", 0.0) == 1.0
        and features.get("street_name_similarity", 0.0) >= 0.82
        and features.get("country_exact", 0.0) == 1.0
    ):
        return True, 0.995, "exact_brand_street_address_country"

    return False, 0.0, ""


def _soundex(token: str) -> str:
    """Pure-Python Soundex implementation for phonetic similarity fallback."""
    token = token.upper()
    if not token or not token[0].isalpha():
        return ""
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    first = token[0]
    encoded = [first]
    prev = mapping.get(first, "")
    for char in token[1:]:
        digit = mapping.get(char, "")
        if digit:
            if digit != prev:
                encoded.append(digit)
                prev = digit
        else:
            prev = ""
    res = "".join(encoded) + "000"
    return res[:4]


def _phonetic_similarity(tokens1: list[str], tokens2: list[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    try:
        if HAS_METAPHONE:
            codes1 = [metaphone.doublemetaphone(t)[0] for t in tokens1 if t]
            codes2 = [metaphone.doublemetaphone(t)[0] for t in tokens2 if t]
        else:
            codes1 = [_soundex(t) for t in tokens1 if t]
            codes2 = [_soundex(t) for t in tokens2 if t]
        str1 = " ".join(c for c in codes1 if c)
        str2 = " ".join(c for c in codes2 if c)
        if not str1 or not str2:
            return 0.0
        return _similarity(str1, str2)
    except Exception:
        return 0.0


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _weighted_jaccard(
    left: set[str], right: set[str], idf: dict[str, float]
) -> float:
    union = left | right
    if not union:
        return 0.0
    intersection_weight = sum(idf.get(token, 1.0) for token in left & right)
    union_weight = sum(idf.get(token, 1.0) for token in union)
    return intersection_weight / union_weight if union_weight else 0.0


def _weighted_cosine(left: set[str], right: set[str], idf: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    left_weights = {token: idf.get(token, 1.0) for token in left}
    right_weights = {token: idf.get(token, 1.0) for token in right}
    dot = sum(left_weights[token] * right_weights[token] for token in left & right)
    norm_left = math.sqrt(sum(weight * weight for weight in left_weights.values()))
    norm_right = math.sqrt(sum(weight * weight for weight in right_weights.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def _ngrams(value: str, size: int = 3) -> set[str]:
    if not value:
        return set()
    compact = f"  {value}  "
    return {compact[index : index + size] for index in range(max(1, len(compact) - size + 1))}


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def _length_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))