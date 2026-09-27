"""
Module to convert the pipeline's internal long-form output into the OFFICIAL
Amazon ML Challenge 2026 submission format.
"""
from __future__ import annotations

import csv
import logging
import zipfile
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger('entity_resolution')


def read_internal_tsv(path: str | Path) -> list[dict[str, str]]:
    """
    Reads a pipeline internal TSV file and returns the list of row dicts.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"File {path_obj} does not exist.")
        return []
        
    with open(path_obj, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)


def convert_to_submission(
    internal_matches: list[dict[str, str]],
    internal_candidates: list[dict[str, str]],
    all_test_s1_ids: set[str],
    output_dir: Path | str
) -> dict[str, int]:
    """
    Groups internal matches and candidates and formats them according to the 
    official submission format.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    matches_map: defaultdict[str, set[str]] = defaultdict(set)
    candidates_map: defaultdict[str, set[str]] = defaultdict(set)
    
    for row in internal_candidates:
        s1 = row.get("source1_id")
        tgt = row.get("target_id")
        if s1 and tgt:
            candidates_map[s1].add(tgt)
    
    for row in internal_matches:
        s1 = row.get("source1_id")
        tgt = row.get("target_id")
        if s1 and tgt:
            matches_map[s1].add(tgt)
            # Ensure matches are also in candidates
            candidates_map[s1].add(tgt)
            
    sorted_s1_ids = sorted(list(all_test_s1_ids))
    
    matching_rows = 0
    candidate_rows = 0
    matched_entities_count = 0
    singleton_entities_count = 0
    
    # Write matching_results.tsv
    matching_file = out_dir / 'matching_results.tsv'
    with open(matching_file, mode='w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted_s1_ids:
            targets = sorted(list(matches_map.get(s1_id, set())))
            if targets:
                matched_entities_count += 1
            else:
                singleton_entities_count += 1
                
            targets_str = ",".join(targets)
            f.write(f"{s1_id}\t{targets_str}\n")
            matching_rows += 1
            
    # Write candidate_pairs.tsv
    candidate_file = out_dir / 'candidate_pairs.tsv'
    with open(candidate_file, mode='w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted_s1_ids:
            targets = sorted(list(candidates_map.get(s1_id, set())))
            targets_str = ",".join(targets)
            f.write(f"{s1_id}\t{targets_str}\n")
            candidate_rows += 1
            
    return {
        "matching_rows": matching_rows,
        "candidate_rows": candidate_rows,
        "matched_entities": matched_entities_count,
        "singleton_entities": singleton_entities_count
    }


def validate_submission_format(output_dir: Path | str, test_dir: Path | str = None) -> tuple[list[str], list[str]]:
    """
    Validates the generated submission files for format correctness.
    """
    out_dir = Path(output_dir)
    matching_file = out_dir / 'matching_results.tsv'
    candidate_file = out_dir / 'candidate_pairs.tsv'
    
    errors = []
    warnings = []
    
    if not matching_file.exists():
        errors.append(f"Missing {matching_file.name}")
    if not candidate_file.exists():
        errors.append(f"Missing {candidate_file.name}")
        
    if errors:
        return errors, warnings
        
    def read_tsv(path):
        with open(path, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                yield row
                
    matches = {
        row['source1_entity_id']: row['matched_entity_ids'].split(',') if row.get('matched_entity_ids') else [] 
        for row in read_tsv(matching_file)
    }
    
    candidates = {
        row['source1_entity_id']: row['candidate_entity_ids'].split(',') if row.get('candidate_entity_ids') else [] 
        for row in read_tsv(candidate_file)
    }
                  
    for s1_id, match_list in matches.items():
        cand_list = candidates.get(s1_id, [])
        cand_set = set(cand_list)
        
        for m in match_list:
            if m.startswith('S1-') or m == s1_id:
                errors.append(f"S1 self-match found for {s1_id}: {m}")
            if m not in cand_set:
                errors.append(f"Match {m} for {s1_id} not found in candidates")
                
    return errors, warnings


def create_submission_zip(source_dir: Path | str, zip_path: Path | str) -> Path:
    """Pack matching_results.tsv and candidate_pairs.tsv into a competition submission zip."""
    src = Path(source_dir)
    dest = Path(zip_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
            fpath = src / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
                logger.info("Archived %s into submission zip", fname)
    return dest
