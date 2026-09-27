"""
Fast, memory-safe test set inference script for Amazon ML Challenge 2026.
Produces:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv
  - output/submission/matching_results.tsv
  - output/submission/candidate_pairs.tsv
  - output/submission.zip
"""
from __future__ import annotations

import gc
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

# Distinctive word filter
STOPWORDS = {
    "inc", "corp", "corporation", "llc", "ltd", "limited", "pvt", "private",
    "sarl", "sa", "sas", "gmbh", "co", "company", "services", "solutions",
    "enterprises", "trading", "technologies", "group", "the", "and", "de",
    "la", "du", "des", "les", "en", "null", "none", "of", "in", "for", "at",
    "p", "a", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
}


def clean_tokens(name: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-z0-9]+", name.lower())
        if w not in STOPWORDS and len(w) > 2
    ]


def extract_street_num(address: str) -> str:
    m = re.findall(r"\b\d+\b", address)
    return m[0] if m else ""


def build_index_for_source(path: Path, max_candidates_per_key: int = 4) -> dict[str, list[str]]:
    """Build high-precision candidate lookup index from target TSV."""
    idx = defaultdict(list)
    print(f"Indexing {path.name}...", flush=True)
    t0 = time.perf_counter()
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            c = country.strip().lower()
            tokens = clean_tokens(name)
            num = extract_street_num(addr)

            # Strategy 1: country + first 2 brand tokens
            if len(tokens) >= 2:
                k1 = f"{c}#n2#{tokens[0]}#{tokens[1]}"
                if len(idx[k1]) < max_candidates_per_key:
                    idx[k1].append(eid)
            elif tokens:
                k1 = f"{c}#n1#{tokens[0]}"
                if len(idx[k1]) < max_candidates_per_key:
                    idx[k1].append(eid)

            # Strategy 2: country + first brand token + street number
            if tokens and num:
                k2 = f"{c}#ns#{tokens[0]}#{num}"
                if len(idx[k2]) < max_candidates_per_key:
                    idx[k2].append(eid)

            if count % 1000000 == 0:
                print(f"  {path.name}: {count:,} rows indexed in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"Finished {path.name}: {count:,} rows, {len(idx):,} unique keys in {time.perf_counter()-t0:.1f}s", flush=True)
    return dict(idx)


def run_inference(data_dir: Path, output_dir: Path) -> None:
    t_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    test_s1_path = data_dir / "test_source1.tsv"
    test_s2_path = data_dir / "test_source2.tsv"
    test_s3_path = data_dir / "test_source3.tsv"

    print("Step 1/4: Indexing Source 2 and Source 3...", flush=True)
    s2_idx = build_index_for_source(test_s2_path)
    s3_idx = build_index_for_source(test_s3_path)

    print("\nStep 2/4: Streaming Test Source 1 and generating matches & candidates...", flush=True)
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"
    sub_matching_file = submission_dir / "matching_results.tsv"
    sub_candidate_file = submission_dir / "candidate_pairs.tsv"

    s1_count = 0
    matched_count = 0
    total_matches = 0
    total_candidates = 0

    t_s1 = time.perf_counter()
    with open(test_s1_path, "r", encoding="utf-8") as in_f, \
         open(matching_file, "w", encoding="utf-8", newline="\n") as out_m, \
         open(candidate_file, "w", encoding="utf-8", newline="\n") as out_c, \
         open(sub_matching_file, "w", encoding="utf-8", newline="\n") as sub_m, \
         open(sub_candidate_file, "w", encoding="utf-8", newline="\n") as sub_c:

        # Official Headers
        header_m = "source1_entity_id\tmatched_entity_ids\n"
        header_c = "source1_entity_id\tcandidate_entity_ids\n"
        out_m.write(header_m)
        out_c.write(header_c)
        sub_m.write(header_m)
        sub_c.write(header_c)

        next(in_f)  # skip header
        for line in in_f:
            s1_count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                # Malformed fallback
                eid = parts[0] if parts else f"S1-{s1_count}"
                out_m.write(f"{eid}\t\n")
                out_c.write(f"{eid}\t\n")
                sub_m.write(f"{eid}\t\n")
                sub_c.write(f"{eid}\t\n")
                continue

            eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            c = country.strip().lower()
            tokens = clean_tokens(name)
            num = extract_street_num(addr)

            # Query keys
            q_keys_high = []
            q_keys_med = []
            if tokens and num:
                q_keys_high.append(f"{c}#ns#{tokens[0]}#{num}")
            if len(tokens) >= 2:
                q_keys_high.append(f"{c}#n2#{tokens[0]}#{tokens[1]}")
            elif tokens:
                q_keys_med.append(f"{c}#n1#{tokens[0]}")

            # Collect matches and candidates
            matches_s2 = []
            matches_s3 = []
            cands_s2 = []
            cands_s3 = []

            # 1. High precision keys (name + street# or 2 tokens)
            for k in q_keys_high:
                if k in s2_idx:
                    cands_s2.extend(s2_idx[k])
                if k in s3_idx:
                    cands_s3.extend(s3_idx[k])

            # If no high-precision candidates, check single token
            if not cands_s2 and not cands_s3:
                for k in q_keys_med:
                    if k in s2_idx:
                        cands_s2.extend(s2_idx[k])
                    if k in s3_idx:
                        cands_s3.extend(s3_idx[k])

            # Deduplicate preserving order
            cands_s2 = list(dict.fromkeys(cands_s2))[:4]
            cands_s3 = list(dict.fromkeys(cands_s3))[:4]

            # High precision filtering: top 1-2 per source accepted as matches
            matches_s2 = cands_s2[:2]
            matches_s3 = cands_s3[:2]

            all_matches = matches_s2 + matches_s3
            all_cands = cands_s2 + cands_s3

            # Ensure matching is strictly a subset of candidates
            all_cands_set = set(all_cands)
            for m in all_matches:
                if m not in all_cands_set:
                    all_cands.append(m)

            match_str = ",".join(all_matches)
            cand_str = ",".join(all_cands)

            if all_matches:
                matched_count += 1
                total_matches += len(all_matches)
            total_candidates += len(all_cands)

            line_m = f"{eid}\t{match_str}\n"
            line_c = f"{eid}\t{cand_str}\n"

            out_m.write(line_m)
            out_c.write(line_c)
            sub_m.write(line_m)
            sub_c.write(line_c)

            if s1_count % 500000 == 0:
                print(f"  Processed {s1_count:,} S1 entities... ({time.perf_counter()-t_s1:.1f}s)", flush=True)

    print(f"\nFinished S1 processing: {s1_count:,} entities.", flush=True)
    print(f"  Entities with matches: {matched_count:,} ({matched_count/s1_count*100:.1f}%)")
    print(f"  Total match pairs: {total_matches:,} (avg {total_matches/max(1, matched_count):.2f}/entity)")
    print(f"  Total candidate pairs: {total_candidates:,} (avg {total_candidates/s1_count:.2f}/entity)")

    print("\nStep 3/4: Packaging submission.zip...", flush=True)
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(matching_file, arcname="output/matching_results.tsv")
        zf.write(candidate_file, arcname="output/candidate_pairs.tsv")
    print(f"Created {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

    print(f"\nStep 4/4: Complete in {time.perf_counter()-t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    DATA_DIR = Path("A:/Amazon ML Challenge/Dataset/student_resource/dataset/test")
    OUTPUT_DIR = Path("A:/Amazon ML Challenge/business_entity_resolution/output")
    run_inference(DATA_DIR, OUTPUT_DIR)
