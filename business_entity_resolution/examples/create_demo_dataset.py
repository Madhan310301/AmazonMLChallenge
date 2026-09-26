"""Comprehensive synthetic dataset generator for pipeline testing.

Generates realistic noisy business-entity data that exercises:
- Positive matching with controlled noise (name, address, format)
- Hard negatives (similar names/cities but DIFFERENT entities)
- Singletons (Source 1 entities with no valid match)
- Multi-source relationships (S1 -> S2 + S3)
- S2-only and S3-only matches

IMPORTANT: This data is for testing only. It must NEVER be presented
as official competition data, leaderboard results, or real performance.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

DEMO_MARKER_FILENAME = "_DEMO_MARKER.json"

# Each tuple: (name, address, postal, city, country, has_s2, has_s3)
TRAIN_ENTITIES = [
    # --- 15 entities with BOTH S2 and S3 matches ---
    ("Zenith Technologies Pvt Ltd", "12 MG Road", "560001", "Bangalore", "India", True, True),
    ("Aarav Pharma Solutions", "45 Anna Salai", "600002", "Chennai", "India", True, True),
    ("Priya Silk Textiles", "81 Hill Road", "400001", "Mumbai", "India", True, True),
    ("Spice Route Trading Co", "23 Chandni Chowk", "110006", "Delhi", "India", True, True),
    ("Sharma & Associates", "9 Jubilee Hills", "500033", "Hyderabad", "India", True, True),
    ("Ironclad Steel Works", "7 Industrial Area", "831001", "Jamshedpur", "India", True, True),
    ("Nexgen Software Solutions Ltd", "42 Electronics City", "560100", "Bangalore", "India", True, True),
    ("Emerald Consumer Care", "16 Bandra West", "400050", "Mumbai", "India", True, True),
    ("Maple Creek Bakery", "14 Market Street", "10001", "New York", "US", True, True),
    ("Blue Ridge Auto Repair", "220 Mountain Drive", "80201", "Denver", "US", True, True),
    ("Pacific Coast Consulting LLC", "88 Market Street", "94102", "San Francisco", "US", True, True),
    ("Maison Lumiere", "18 Rue des Fleurs", "75001", "Paris", "France", True, True),
    ("Atelier Nova Design", "27 Rue de la Paix", "33000", "Bordeaux", "France", True, True),
    ("Berlin Motors Group", "12 Friedrichstrasse", "10117", "Berlin", "Germany", True, True),
    ("Toronto Tech Hub Inc", "100 King Street West", "M5X1C9", "Toronto", "Canada", True, True),
    # --- 3 entities with S2 match ONLY ---
    ("Quantum Analytics India", "11 HITEC City", "500081", "Hyderabad", "India", True, False),
    ("Liberty Hardware Inc", "52 Freedom Trail", "02101", "Boston", "US", True, False),
    ("Boulangerie du Parc", "4 Avenue Victor Hugo", "69001", "Lyon", "France", True, False),
    # --- 2 entities with S3 match ONLY ---
    ("Ocean Breeze Resort", "3 Marine Drive", "682001", "Kochi", "India", False, True),
    ("Cedar Valley Furniture", "29 Oak Lane", "97201", "Portland", "US", False, True),
    # --- 3 singletons (NO match in S2 or S3) ---
    ("Unique Handcraft Exports", "99 Artisan Lane", "700001", "Kolkata", "India", False, False),
    ("Nomad Adventure Travel", "1 Beach Road", "403001", "Goa", "India", False, False),
    ("Heartland Financial Group", "60 LaSalle Street", "60601", "Chicago", "US", False, False),
]

# Hard negatives: DIFFERENT businesses that look similar to canonical entities
HARD_NEGATIVE_DECOYS_S2 = [
    # Similar to Zenith Technologies (same city, similar name)
    ("Zenith Enterprises Ltd", "78 Brigade Road", "560025", "Bangalore", "India"),
    # Similar to Spice Route Trading (same city, similar name)
    ("Spice Garden Restaurant", "15 Karol Bagh", "110005", "Delhi", "India"),
    # Similar to Maple Creek Bakery (same city, similar name)
    ("Maple Ridge Bakery", "22 Broadway", "10002", "New York", "US"),
    # Similar to Pacific Coast Consulting (same city, similar name)
    ("Pacific West Consulting Inc", "100 Mission Street", "94105", "San Francisco", "US"),
]

HARD_NEGATIVE_DECOYS_S3 = [
    # Similar to Nexgen Software (same city, similar name)
    ("Nexgen Solutions Pvt Ltd", "34 Whitefield", "560066", "Bangalore", "India"),
    # Similar to Aarav Pharma (same city, similar name)
    ("Aarav Healthcare Products", "50 Mount Road", "600006", "Chennai", "India"),
    # Similar to Maison Lumiere (same city, similar name)
    ("Maison du Soleil", "22 Rue de Rivoli", "75004", "Paris", "France"),
]

TEST_ENTITIES = [
    ("Cafe des Artistes", "11 Boulevard Saint Germain", "75006", "Paris", "France", True, True),
    ("Jardin Botanique SAS", "6 Canebiere", "13001", "Marseille", "France", True, True),
    ("Metro Bike Works", "18 Central Avenue", "400006", "Mumbai", "India", True, True),
    ("Sunrise Pharmacy", "4 Garden Road", "400007", "Mumbai", "India", True, True),
    ("Golden Gate Electronics", "155 Market Street", "94103", "San Francisco", "US", True, True),
    ("Pine and Oak Hotel", "52 River Street", "400005", "Mumbai", "India", True, True),
    ("Atlas Plumbing Services", "7 Station Road", "400009", "Mumbai", "India", True, True),
    ("Royal Jaipur Textiles", "100 Pink City Road", "302001", "Jaipur", "India", True, True),
]

TEST_DECOYS_S2 = [
    ("Sunrise Medical Supplies", "12 Garden Road", "400007", "Mumbai", "India"),
]

TEST_DECOYS_S3 = [
    ("Atlas Engineering Works", "9 Station Road", "400009", "Mumbai", "India"),
]


# ---------------------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------------------

def _apply_name_noise(name: str, variant: int, rng: random.Random) -> str:
    """Apply deterministic noise to a business name."""
    if variant == 0:
        return name  # Clean
    if variant == 1:
        return name.lower()
    if variant == 2:
        # Change legal suffix
        suffixes = [
            ("Pvt Ltd", "Private Limited"),
            ("Private Limited", "Pvt. Ltd."),
            ("Ltd", "Limited"),
            ("Limited", "Ltd."),
            ("Inc", "Incorporated"),
            ("Incorporated", "Inc."),
            ("LLC", "L.L.C."),
            ("L.L.C.", "LLC"),
            ("Co", "Company"),
            ("Company", "Co."),
            ("SAS", "S.A.S."),
        ]
        for old, new in suffixes:
            if name.endswith(old):
                return name[: -len(old)] + new
        return name.lower()
    if variant == 3:
        # Abbreviate / expand words
        replacements = [
            ("Technologies", "Tech"),
            ("Tech", "Technologies"),
            ("Solutions", "Soln"),
            ("Consulting", "Consult"),
            ("Services", "Svcs"),
            ("Industries", "Ind"),
            ("& ", "and "),
            ("and ", "& "),
        ]
        for old, new in replacements:
            if old in name:
                return name.replace(old, new, 1)
        return name
    if variant == 4:
        # Punctuation / formatting noise
        return name.replace(" ", "  ").rstrip() + "."
    if variant == 5:
        # Truncate to core brand (drop last suffix words)
        words = name.split()
        if len(words) > 2:
            return " ".join(words[: max(2, len(words) - 1)])
        return name
    return name


def _apply_address_noise(
    address: str, city: str, variant: int, rng: random.Random
) -> str:
    """Apply deterministic noise to a business address."""
    if variant == 0:
        return f"{address}, {city}"
    if variant == 1:
        # Abbreviate road names
        for old, new in [
            ("Road", "Rd"),
            ("Street", "St"),
            ("Avenue", "Ave"),
            ("Boulevard", "Blvd"),
            ("Drive", "Dr"),
            ("Lane", "Ln"),
            ("Rue", "R."),
        ]:
            address = address.replace(old, new)
        return address
    if variant == 2:
        # Add "No." prefix
        parts = address.split(" ", 1)
        if len(parts) > 1 and parts[0].replace("-", "").replace("/", "").isdigit():
            return f"No. {address}"
        return address
    if variant == 3:
        return address.lower()
    return address


# ---------------------------------------------------------------------------
# File writing helpers
# ---------------------------------------------------------------------------

def _write_tsv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write rows to a TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def create_demo_dataset(root: Path, seed: int = 42) -> None:
    """Generate a deterministic synthetic dataset for pipeline testing.

    Produces all 7 expected TSV files plus a _DEMO_MARKER.json marker.
    """
    rng = random.Random(seed)

    train_dir = root / "train"
    test_dir = root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train data ----
    _generate_split(
        train_dir,
        prefix="train",
        id_prefix="tr",
        entities=TRAIN_ENTITIES,
        decoys_s2=HARD_NEGATIVE_DECOYS_S2,
        decoys_s3=HARD_NEGATIVE_DECOYS_S3,
        rng=rng,
        include_ground_truth=True,
    )

    # ---- Test data ----
    _generate_split(
        test_dir,
        prefix="test",
        id_prefix="te",
        entities=TEST_ENTITIES,
        decoys_s2=TEST_DECOYS_S2,
        decoys_s3=TEST_DECOYS_S3,
        rng=rng,
        include_ground_truth=False,
    )

    # ---- Demo marker ----
    marker = {
        "mode": "demo",
        "generator": "create_demo_dataset",
        "seed": seed,
        "train_entities": len(TRAIN_ENTITIES),
        "test_entities": len(TEST_ENTITIES),
        "warning": "This is synthetic demo data. NOT for competition submission.",
    }
    (root / DEMO_MARKER_FILENAME).write_text(
        json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _generate_split(
    directory: Path,
    prefix: str,
    id_prefix: str,
    entities: list[tuple],
    decoys_s2: list[tuple],
    decoys_s3: list[tuple],
    rng: random.Random,
    include_ground_truth: bool,
) -> None:
    """Generate source1/2/3 TSV files and optionally ground truth."""
    source_cols = [
        "business_name", "business_address", "postal_code", "city", "country",
    ]
    s1_rows: list[dict] = []
    s2_rows: list[dict] = []
    s3_rows: list[dict] = []
    gt_rows: list[dict] = []

    for idx, (name, address, postal, city, country, has_s2, has_s3) in enumerate(
        entities, start=1
    ):
        s1_id = f"S1-{id_prefix}_{idx:03d}"
        s2_id = f"S2-{id_prefix}_{idx:03d}" if has_s2 else ""
        s3_id = f"S3-{id_prefix}_{idx:03d}" if has_s3 else ""

        # Source 1: clean record
        s1_rows.append({
            "source1_id": s1_id,
            "business_name": _apply_name_noise(name, 0, rng),
            "business_address": address,
            "postal_code": postal,
            "city": city,
            "country": country,
        })

        # Source 2: moderate noise
        if has_s2:
            name_var = (idx % 5) + 1
            addr_var = idx % 4
            s2_rows.append({
                "source2_id": s2_id,
                "business_name": _apply_name_noise(name, name_var, rng),
                "business_address": _apply_address_noise(address, city, addr_var, rng),
                "postal_code": postal,
                "city": city,
                "country": country,
            })

        # Source 3: different noise
        if has_s3:
            name_var = ((idx + 2) % 5) + 1
            addr_var = (idx + 1) % 4
            s3_rows.append({
                "source3_id": s3_id,
                "business_name": _apply_name_noise(name, name_var, rng),
                "business_address": _apply_address_noise(address, city, addr_var, rng),
                "postal_code": postal,
                "city": city,
                "country": country,
            })

        # Ground truth
        if include_ground_truth:
            gt_rows.append({
                "source1_id": s1_id,
                "source2_id": s2_id,
                "source3_id": s3_id,
            })

    # Add hard negative decoys
    for d_idx, (name, address, postal, city, country) in enumerate(decoys_s2, start=1):
        s2_rows.append({
            "source2_id": f"S2-{id_prefix}_decoy_{d_idx:02d}",
            "business_name": name,
            "business_address": f"{address}, {city}",
            "postal_code": postal,
            "city": city,
            "country": country,
        })

    for d_idx, (name, address, postal, city, country) in enumerate(decoys_s3, start=1):
        s3_rows.append({
            "source3_id": f"S3-{id_prefix}_decoy_{d_idx:02d}",
            "business_name": name,
            "business_address": f"{address}, {city}",
            "postal_code": postal,
            "city": city,
            "country": country,
        })

    # Write TSV files with correct filenames
    _write_tsv(
        directory / f"{prefix}_source1.tsv",
        ["source1_id"] + source_cols,
        s1_rows,
    )
    _write_tsv(
        directory / f"{prefix}_source2.tsv",
        ["source2_id"] + source_cols,
        s2_rows,
    )
    _write_tsv(
        directory / f"{prefix}_source3.tsv",
        ["source3_id"] + source_cols,
        s3_rows,
    )

    if include_ground_truth:
        _write_tsv(
            directory / f"{prefix}_ground_truth.tsv",
            ["source1_id", "source2_id", "source3_id"],
            gt_rows,
        )


if __name__ == "__main__":
    create_demo_dataset(Path(__file__).parent / "demo_dataset")