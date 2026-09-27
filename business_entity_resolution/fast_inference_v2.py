"""
Upgraded, high-precision test inference engine for Amazon ML Challenge 2026.
Fixes false merges, normalizes addresses/names, and strictly controls precision for Macro F0.5.
Produces:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv
  - output/submission/matching_results.tsv
  - output/submission/candidate_pairs.tsv
  - output/submission.zip
  - Team_EntitySync_submission.zip
"""
from __future__ import annotations

import gc
import os
import re
import sys
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

LEGAL_SUFFIXES = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'sarl', 'sa', 'sas', 'gmbh', 'co', 'company', 'services', 'solutions',
    'enterprises', 'trading', 'technologies', 'group', 'the', 'and', 'de',
    'la', 'du', 'des', 'les', 'en', 'null', 'none', 'of', 'in', 'for', 'at'
}

US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA',
    'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
    'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
}

IN_STATES = {
    'maharashtra', 'delhi', 'karnataka', 'tamilnadu', 'uttarpradesh', 'gujarat',
    'westbengal', 'rajasthan', 'telangana', 'andhrapradesh', 'kerala', 'madhyapradesh',
    'punjab', 'haryana', 'bihar', 'odisha', 'orissa', 'assam', 'jharkhand', 'chhattisgarh',
    'uttarakhand', 'goa', 'himachalpradesh', 'chandigarh', 'puducherry', 'jammuandkashmir'
}


def nfkd_clean(text: str) -> str:
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = nfkd.encode('ASCII', 'ignore').decode('ASCII')
    cleaned = re.sub(r'(@|www\.|\.com|\.in|\.fr|\.org|\.net|\bthe\b|\bmr\b|\bmrs\b|\bms\b)', ' ', ascii_text.lower())
    return cleaned


def clean_tokens(text: str) -> list[str]:
    c = nfkd_clean(text)
    return [w for w in re.findall(r'[a-z0-9]+', c) if w not in LEGAL_SUFFIXES and len(w) > 1]


def clean_compact_str(text: str) -> str:
    return ''.join(clean_tokens(text))


def extract_street_num(addr: str) -> str:
    m = re.findall(r'\b\d+\b', addr)
    if m:
        num = m[0].lstrip('0')
        return num if num else '0'
    return ''


def extract_state(addr: str, country: str) -> str:
    if country == 'us':
        codes = re.findall(r'\b[A-Z]{2}\b', addr.upper())
        for c in reversed(codes):
            if c in US_STATES:
                return c
    elif country == 'india':
        clean = addr.lower().replace(' ', '').replace('.', '').replace(',', '')
        for s in IN_STATES:
            if s in clean:
                return s
    elif country == 'france':
        m = re.findall(r'\b(0[1-9]|[1-8][0-9]|9[0-8]|2A|2B)\b', addr.upper())
        if m:
            return m[0]
    return ''


def char_jaccard_3gram(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if len(s1) < 3 or len(s2) < 3:
        return 1.0 if s1 == s2 else 0.0
    g1 = {s1[i:i+3] for i in range(len(s1)-2)}
    g2 = {s2[i:i+3] for i in range(len(s2)-2)}
    return len(g1 & g2) / len(g1 | g2)


def build_target_index(path: Path, max_cands_per_key: int = 4):
    """
    Builds both inverted index and fast metadata store for target entities.
    target_meta: dict[str, tuple[str, str, str, tuple[str, ...]]]
      eid -> (compact_str, street_num, state, first_tokens)
    """
    idx = defaultdict(list)
    meta = {}
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
            c_str = clean_compact_str(name)
            num = extract_street_num(addr)
            state = extract_state(addr, c)

            # Store compact metadata
            first_toks = tuple(tokens[:3])
            meta[eid] = (c_str, num, state, first_toks)

            # Strategy 1: Exact compact name
            if c_str:
                k1 = f"{c}#ex#{c_str}"
                if len(idx[k1]) < max_cands_per_key:
                    idx[k1].append(eid)

            # Strategy 2: First 2 tokens
            if len(tokens) >= 2:
                k2 = f"{c}#t2#{tokens[0]}#{tokens[1]}"
                if len(idx[k2]) < max_cands_per_key:
                    idx[k2].append(eid)

            # Strategy 3: Brand token + Street number
            if tokens and num:
                k3 = f"{c}#ns#{tokens[0]}#{num}"
                if len(idx[k3]) < max_cands_per_key:
                    idx[k3].append(eid)

            if count % 1000000 == 0:
                print(f"  {path.name}: {count:,} rows in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"Finished {path.name}: {count:,} rows, {len(idx):,} keys, {len(meta):,} meta in {time.perf_counter()-t0:.1f}s", flush=True)
    return dict(idx), meta


def run_high_precision_inference(data_dir: Path, output_dir: Path) -> None:
    t_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    test_s1_path = data_dir / "test_source1.tsv"
    test_s2_path = data_dir / "test_source2.tsv"
    test_s3_path = data_dir / "test_source3.tsv"

    print("Step 1/4: Indexing Source 2 and Source 3...", flush=True)
    s2_idx, s2_meta = build_target_index(test_s2_path)
    s3_idx, s3_meta = build_target_index(test_s3_path)

    print("\nStep 2/4: Streaming Test Source 1 & Verifying Pairwise Matches...", flush=True)
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
                eid = parts[0] if parts else f"S1-{s1_count}"
                out_m.write(f"{eid}\t\n")
                out_c.write(f"{eid}\t\n")
                sub_m.write(f"{eid}\t\n")
                sub_c.write(f"{eid}\t\n")
                continue

            eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            c = country.strip().lower()

            tokens = clean_tokens(name)
            token_set = set(tokens)
            c_str = clean_compact_str(name)
            num = extract_street_num(addr)
            state = extract_state(addr, c)

            # Query keys
            q_keys = []
            if c_str:
                q_keys.append(f"{c}#ex#{c_str}")
            if tokens and num:
                q_keys.append(f"{c}#ns#{tokens[0]}#{num}")
            if len(tokens) >= 2:
                q_keys.append(f"{c}#t2#{tokens[0]}#{tokens[1]}")

            # Gather raw candidates
            raw_cands_s2 = []
            raw_cands_s3 = []
            for k in q_keys:
                if k in s2_idx:
                    raw_cands_s2.extend(s2_idx[k])
                if k in s3_idx:
                    raw_cands_s3.extend(s3_idx[k])

            # Deduplicate candidate IDs preserving order
            cands_s2 = list(dict.fromkeys(raw_cands_s2))[:5]
            cands_s3 = list(dict.fromkeys(raw_cands_s3))[:5]

            matches_s2 = []
            matches_s3 = []

            # Verify candidates for Source 2
            for cid in cands_s2:
                t_str, t_num, t_state, t_toks = s2_meta[cid]
                # Hard conflict checks
                if state and t_state and state != t_state:
                    continue
                if num and t_num and num != t_num and not (num.startswith(t_num) or t_num.startswith(num)):
                    continue
                # Verification rules
                if c_str and t_str and c_str == t_str:
                    matches_s2.append(cid)
                    continue
                if num and t_num and num == t_num and (state == t_state or not state or not t_state):
                    if tokens and t_toks and (tokens[0] == t_toks[0] or char_jaccard_3gram(c_str, t_str) >= 0.50):
                        matches_s2.append(cid)
                        continue
                t_token_set = set(t_toks)
                if token_set and t_token_set:
                    jacc = len(token_set & t_token_set) / len(token_set | t_token_set)
                    if jacc >= 0.60:
                        matches_s2.append(cid)
                        continue
                if len(c_str) >= 5 and len(t_str) >= 5:
                    if char_jaccard_3gram(c_str, t_str) >= 0.75:
                        matches_s2.append(cid)
                        continue

            # Verify candidates for Source 3
            for cid in cands_s3:
                t_str, t_num, t_state, t_toks = s3_meta[cid]
                # Hard conflict checks
                if state and t_state and state != t_state:
                    continue
                if num and t_num and num != t_num and not (num.startswith(t_num) or t_num.startswith(num)):
                    continue
                # Verification rules
                if c_str and t_str and c_str == t_str:
                    matches_s3.append(cid)
                    continue
                if num and t_num and num == t_num and (state == t_state or not state or not t_state):
                    if tokens and t_toks and (tokens[0] == t_toks[0] or char_jaccard_3gram(c_str, t_str) >= 0.50):
                        matches_s3.append(cid)
                        continue
                t_token_set = set(t_toks)
                if token_set and t_token_set:
                    jacc = len(token_set & t_token_set) / len(token_set | t_token_set)
                    if jacc >= 0.60:
                        matches_s3.append(cid)
                        continue
                if len(c_str) >= 5 and len(t_str) >= 5:
                    if char_jaccard_3gram(c_str, t_str) >= 0.75:
                        matches_s3.append(cid)
                        continue

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
    print(f"  Total match pairs: {total_matches:,} (avg {total_matches/max(1, matched_count):.2f}/matched entity)")
    print(f"  Total candidate pairs: {total_candidates:,} (avg {total_candidates/s1_count:.2f}/entity)")

    print("\nStep 3/4: Packaging submission zips...", flush=True)
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(matching_file, arcname="output/matching_results.tsv")
        zf.write(candidate_file, arcname="output/candidate_pairs.tsv")
    print(f"Created {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

    root = output_dir.parent
    final_zip = root.parent / "Team_EntitySync_submission.zip"
    print(f"Building complete final submission package: {final_zip}...", flush=True)
    with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(matching_file, arcname="output/matching_results.tsv")
        zf.write(candidate_file, arcname="output/candidate_pairs.tsv")
        doc_path = root / "Documentation_template.md"
        if doc_path.is_file():
            zf.write(doc_path, arcname="Documentation_template.md")
        zf.write(root / "README.md", arcname="code/business_entity_resolution/README.md")
        zf.write(root / "requirements.txt", arcname="code/business_entity_resolution/requirements.txt")
        zf.write(root / "run.py", arcname="code/business_entity_resolution/run.py")
        zf.write(root / "fast_inference_v2.py", arcname="code/business_entity_resolution/fast_inference.py")
        src_dir = root / "src"
        for fpath in src_dir.rglob("*"):
            if fpath.is_file() and "__pycache__" not in fpath.parts:
                rel = fpath.relative_to(root)
                zf.write(fpath, arcname=f"code/business_entity_resolution/{rel.as_posix()}")

    print(f"Done! Final submission zip created: {final_zip} ({final_zip.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
    print(f"\nStep 4/4: Total time: {time.perf_counter()-t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    DATA_DIR = Path("A:/Amazon ML Challenge/Dataset/student_resource/dataset/test")
    OUTPUT_DIR = Path("A:/Amazon ML Challenge/business_entity_resolution/output")
    run_high_precision_inference(DATA_DIR, OUTPUT_DIR)
