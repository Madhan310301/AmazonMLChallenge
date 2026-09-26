from __future__ import annotations

import re
import unicodedata


LEGAL_SUFFIXES = {
    ("private", "limited"): "private limited",
    ("private", "ltd"): "private limited",
    ("pvt", "ltd"): "private limited",
    ("pvt", "limited"): "private limited",
    ("limited", "liability", "company"): "limited liability company",
    ("limited", "liability", "corp"): "limited liability company",
    ("l", "l", "c"): "limited liability company",
    ("l", "l", "p"): "limited liability partnership",
    ("p", "l", "l", "c"): "professional limited liability company",
    ("s", "a", "r", "l"): "sarl",
    ("s", "a", "s"): "sas",
    ("g", "m", "b", "h"): "gmbh",
    ("s", "p", "a"): "spa",
    ("s", "a"): "sa",
    ("l", "l", "c"): "llc",
    ("l", "l", "p"): "llp",
    ("p", "l", "c"): "plc",
    ("incorporated",): "inc",
    ("corporation",): "corp",
    ("company",): "co",
    ("co",): "co",
    ("limited",): "ltd",
    ("private",): "private",
    ("pvt",): "pvt",
    ("ltd",): "ltd",
    ("inc",): "inc",
    ("corp",): "corp",
    ("llc",): "llc",
    ("llp",): "llp",
    ("plc",): "plc",
    ("gmbh",): "gmbh",
    ("sarl",): "sarl",
    ("sas",): "sas",
    ("spa",): "spa",
    ("sa",): "sa",
}
SUFFIX_KEYS = sorted(LEGAL_SUFFIXES, key=len, reverse=True)

# Postal code regex covering:
# - Corsica (2Axxx, 2Bxxx)
# - France mainland (5 digits)
# - US (5 digits, 5+4 digits)
# - India (6 digits)
# - UK / international patterns
POSTAL_RE = re.compile(
    r"\b(?:2[ABab]\d{3}|\d{5}(?:-\d{4})?|\d{6}|\d{4,10}|[A-Za-z]{1,2}\d[A-Za-z0-9]?[ -]?\d[A-Za-z]{2}|[A-Z]\d[A-Z][ -]?\d[A-Z]\d|\d{2,4}[A-Z][A-Z0-9]{1,4})\b"
)
STREET_NUMBER_RE = re.compile(r"^\d+[A-Za-z]?(?:[-/]\d+[A-Za-z]?)?$")

# French commercial routing: CEDEX, CEDEX 08, BP 402
CEDEX_RE = re.compile(r"\bcedex(?:\s*\d+)?\b", re.IGNORECASE)
BP_RE = re.compile(r"\bbp\s*\d+\b", re.IGNORECASE)

# Landmark markers (India / international descriptive address patterns)
LANDMARK_MARKER_RE = re.compile(
    r"\b(near(?:\s+to)?|opp(?:\.|\s+to|\s+site)?|opposite|behind|beside|adj(?:\.|\s+to)?|adjacent(?:\s+to)?|in\s+front\s+of)\b",
    re.IGNORECASE,
)


def normalize_text(value: str | None) -> str:
    """Explicit Unicode normalization using NFKD decomposition with safe accent removal."""
    raw = unicodedata.normalize("NFKD", str(value or ""))
    deaccented = "".join(char for char in raw if not unicodedata.combining(char))
    deaccented = (
        deaccented.casefold()
        .replace("&", " and ")
        .replace("'", "")
        .replace("’", "")
        .replace("‘", "")
        .replace("—", " ")
        .replace("-", " ")
    )
    return " ".join(
        "".join(char if char.isalnum() else " " for char in deaccented).split()
    )


def tokenize(value: str | None) -> list[str]:
    return normalize_text(value).split()


def normalize_name(value: str | None) -> tuple[str, str, str, list[str]]:
    normalized = normalize_text(value)
    tokens = normalized.split()
    brand_tokens = tokens[:]
    legal_forms: list[str] = []
    while brand_tokens:
        match = next(
            (
                suffix
                for suffix in SUFFIX_KEYS
                if len(brand_tokens) >= len(suffix)
                and tuple(brand_tokens[-len(suffix) :]) == suffix
            ),
            None,
        )
        if match is None:
            break
        legal_forms.insert(0, LEGAL_SUFFIXES[match])
        del brand_tokens[-len(match) :]
    brand = " ".join(brand_tokens).strip()
    legal_form = " ".join(legal_forms).strip()
    return normalized, brand, legal_form, brand_tokens


def extract_commercial_routing(raw_address: str) -> tuple[str, str, set[str]]:
    """Extract commercial routing (CEDEX, BP) and return cleaned address, routing string, and routing numbers."""
    routing_parts = []
    routing_numbers = set()
    for m in CEDEX_RE.finditer(raw_address):
        routing_parts.append(m.group(0).strip())
        for num in re.findall(r"\d+", m.group(0)):
            routing_numbers.add(num)
            routing_numbers.add(num.lstrip("0") or "0")
    for m in BP_RE.finditer(raw_address):
        routing_parts.append(m.group(0).strip())
        for num in re.findall(r"\d+", m.group(0)):
            routing_numbers.add(num)
            routing_numbers.add(num.lstrip("0") or "0")

    cleaned = CEDEX_RE.sub(" ", raw_address)
    cleaned = BP_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, " ".join(routing_parts), routing_numbers


def extract_landmarks(raw_address: str) -> tuple[str, str, list[str]]:
    """Separate descriptive landmark information from core address."""
    clauses = [c.strip() for c in re.split(r"[,;]+", raw_address) if c.strip()]
    landmark_clauses = []
    core_clauses = []
    for clause in clauses:
        m = LANDMARK_MARKER_RE.search(clause)
        if m:
            start = m.start()
            before = clause[:start].strip()
            landmark_part = clause[start:].strip()
            if before:
                core_clauses.append(before)
            if landmark_part:
                landmark_clauses.append(landmark_part)
        else:
            core_clauses.append(clause)

    landmark_text = ", ".join(landmark_clauses)
    core_text = ", ".join(core_clauses)
    landmark_tokens = normalize_text(landmark_text).split()
    return core_text, landmark_text, landmark_tokens


def parse_address(record: dict) -> dict:
    raw_address = str(record.get("business_address") or "")
    normalized_address = normalize_text(raw_address)
    address_tokens = normalized_address.split()

    # Commercial routing extraction (Cedex, BP)
    no_routing_address, commercial_routing, routing_numbers = extract_commercial_routing(raw_address)

    # Landmark extraction
    core_address_raw, landmark_text, landmark_tokens = extract_landmarks(no_routing_address)
    core_address = normalize_text(core_address_raw)
    core_address_tokens = core_address.split()

    # Postal code extraction: check explicit first, then regex on raw address
    postal_source = str(record.get("postal_code") or "").strip()
    postal = normalize_text(postal_source).replace(" ", "").upper()
    if not postal:
        match = POSTAL_RE.search(raw_address)
        if match:
            postal_str = match.group(0)
            # Avoid picking up routing numbers as postal code
            if not any(postal_str == r_num for r_num in routing_numbers):
                postal = re.sub(r"[^A-Z0-9]", "", postal_str.upper())

    # Street number extraction: search in core address tokens, avoiding routing numbers & postal codes
    street_number = ""
    candidate_tokens = core_address_tokens if core_address_tokens else address_tokens
    for token in candidate_tokens[:4]:
        clean_token = token.strip(",.;:")
        if clean_token in routing_numbers:
            continue
        if postal and clean_token.upper() == postal:
            continue
        if len(clean_token) >= 5 and clean_token.isdigit():
            # Likely a postal code or phone number
            continue
        if STREET_NUMBER_RE.fullmatch(clean_token):
            street_number = clean_token
            break

    explicit_city = normalize_text(record.get("city"))
    explicit_region = normalize_text(record.get("region"))
    country = normalize_text(record.get("country"))

    street_name_tokens = [
        token
        for token in candidate_tokens
        if token != street_number
        and not (postal and token.upper() in {postal, postal.lower()})
        and token not in routing_numbers
    ]

    return {
        "normalized_address": normalized_address,
        "address_tokens": address_tokens,
        "core_address": core_address,
        "core_address_tokens": core_address_tokens,
        "landmark_text": landmark_text,
        "landmark_tokens": landmark_tokens,
        "commercial_routing": commercial_routing,
        "street_number": street_number,
        "street_name": " ".join(street_name_tokens),
        "postal_code_normalized": postal,
        "city_normalized": explicit_city,
        "region_normalized": explicit_region,
        "country_normalized": country,
    }


def preprocess_record(record: dict) -> dict:
    normalized, brand, legal_form, name_tokens = normalize_name(
        record.get("business_name")
    )
    prepared = dict(record)
    prepared.update(
        {
            "raw_business_name": str(record.get("business_name") or ""),
            "raw_address": str(record.get("business_address") or ""),
            "normalized_business_name": normalized,
            "brand_name": brand,
            "legal_form": legal_form,
            "name_tokens": name_tokens,
        }
    )
    prepared.update(parse_address(record))
    return prepared