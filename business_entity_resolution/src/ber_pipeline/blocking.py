from __future__ import annotations

import hashlib
import logging
import math
import re
import statistics
from functools import lru_cache
from collections import defaultdict

LOGGER = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    HAS_DENSE = True
except ImportError:
    HAS_DENSE = False


def generate_candidates(
    source1_records: list[dict],
    target_records: list[dict],
    target_source: str,
    *,
    seed: int = 42,
    permutations: int = 48,
    bands: int = 12,
    ngram_size: int = 3,
    top_k: int = 8,
    max_posting_size: int = 500,
    dense_model_name: str | None = None,
    dense_top_k: int = 20,
) -> list[dict]:
    if target_source not in {"source2", "source3"}:
        raise ValueError(
            f"Invalid target_source {target_source!r}. Blocking is directional: Source 1 -> {{'source2', 'source3'}}."
        )
    if not source1_records or not target_records:
        return []
    if permutations % bands:
        raise ValueError("MinHash permutations must be divisible by bands.")

    rows_per_band = permutations // bands
    target_by_id = {record["id"]: record for record in target_records}
    band_index: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for target in target_records:
        signature = _minhash(_shingles(_record_text(target), ngram_size), permutations, seed)
        for band in range(bands):
            start = band * rows_per_band
            band_index[(band, signature[start : start + rows_per_band])].append(
                target["id"]
            )

    token_index: dict[str, set[str]] = defaultdict(set)
    postal_index: dict[str, set[str]] = defaultdict(set)
    street_index: dict[str, set[str]] = defaultdict(set)
    document_frequency: dict[str, int] = defaultdict(int)
    for target in target_records:
        tokens = set(target.get("name_tokens", []))
        tokens.update(target.get("address_tokens", []))
        for token in tokens:
            token_index[token].add(target["id"])
            document_frequency[token] += 1
        postal = target.get("postal_code_normalized", "")
        if len(postal) >= 3:
            postal_index[postal[:3]].add(target["id"])
        street = target.get("street_number", "")
        if street:
            street_index[street].add(target["id"])

    dense_index = None
    target_ids_for_dense = []
    dense_model = None
    
    if dense_model_name:
        if HAS_DENSE:
            dense_model = SentenceTransformer(dense_model_name)
            target_texts = [_record_text(target) for target in target_records]
            target_embeddings = dense_model.encode(target_texts, normalize_embeddings=True)
            dimension = target_embeddings.shape[1]
            dense_index = faiss.IndexFlatIP(dimension)
            dense_index.add(target_embeddings)
            target_ids_for_dense = [target["id"] for target in target_records]
        else:
            LOGGER.warning("dense_model_name provided but sentence-transformers/faiss not installed. Skipping dense retrieval.")

    results: list[dict] = []
    for source1 in source1_records:
        signature = _minhash(
            _shingles(_record_text(source1), ngram_size), permutations, seed
        )
        evidence: dict[str, set[str]] = defaultdict(set)
        for band in range(bands):
            start = band * rows_per_band
            key = (band, signature[start : start + rows_per_band])
            for target_id in band_index.get(key, ()):
                evidence[target_id].add("minhash")

        query_tokens = set(source1.get("name_tokens", []))
        query_tokens.update(source1.get("address_tokens", []))
        retrieval_weight: dict[str, float] = defaultdict(float)
        for token in query_tokens:
            posting = token_index.get(token, set())
            if not posting or len(posting) > max_posting_size:
                continue
            weight = math.log((len(target_records) + 1) / (len(posting) + 1)) + 1.0
            for target_id in posting:
                evidence[target_id].add("token_posting")
                retrieval_weight[target_id] += weight
        postal = source1.get("postal_code_normalized", "")
        if len(postal) >= 3:
            for target_id in postal_index.get(postal[:3], ()):
                evidence[target_id].add("token_posting")
                retrieval_weight[target_id] += 1.5
        street = source1.get("street_number", "")
        if street:
            for target_id in street_index.get(street, ()):
                evidence[target_id].add("token_posting")
                retrieval_weight[target_id] += 0.5
                
        if dense_model is not None and dense_index is not None:
            source1_text = _record_text(source1)
            query_embedding = dense_model.encode([source1_text], normalize_embeddings=True)
            k = min(dense_top_k, len(target_ids_for_dense))
            if k > 0:
                scores, indices = dense_index.search(query_embedding, k)
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1:
                        t_id = target_ids_for_dense[idx]
                        evidence[t_id].add("dense")
                        retrieval_weight[t_id] += float(score)

        ranked = []
        for target_id, methods in evidence.items():
            if target_id == source1["id"]:
                continue
            target = target_by_id[target_id]
            cheap = _coarse_score(source1, target)
            
            multi_blocker_bonus = 0.2 if len(methods) == 3 else (0.1 if len(methods) > 1 else 0.0)
            
            blocker_score = min(
                1.0,
                cheap * 0.65
                + min(retrieval_weight.get(target_id, 0.0) / 10.0, 0.3)
                + multi_blocker_bonus,
            )
            ranked.append(
                (
                    blocker_score,
                    target_id,
                    {
                        "source1_id": source1["id"],
                        "target_source": target_source,
                        "target_id": target_id,
                        "blocking_methods": ",".join(sorted(methods)),
                        "blocking_score": blocker_score,
                    },
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        results.extend(item[2] for item in ranked[:top_k])
    return results


def compute_blocking_stats(
    candidates: list[dict],
    truth: set[tuple[str, str, str]] | None = None,
) -> dict:
    by_method = {"minhash": 0, "token_posting": 0, "dense": 0}
    overlap = {
        "minhash_and_token_posting": 0,
        "minhash_and_dense": 0,
        "token_posting_and_dense": 0,
        "all_three": 0,
    }
    source1_counts = defaultdict(int)
    
    for c in candidates:
        methods = set(c.get("blocking_methods", "").split(","))
        methods.discard("")
        
        for m in methods:
            if m in by_method:
                by_method[m] += 1
                
        has_minhash = "minhash" in methods
        has_token = "token_posting" in methods
        has_dense = "dense" in methods
        
        if has_minhash and has_token:
            overlap["minhash_and_token_posting"] += 1
        if has_minhash and has_dense:
            overlap["minhash_and_dense"] += 1
        if has_token and has_dense:
            overlap["token_posting_and_dense"] += 1
        if has_minhash and has_token and has_dense:
            overlap["all_three"] += 1
            
        source1_counts[c["source1_id"]] += 1
        
    counts = list(source1_counts.values()) if source1_counts else [0]
    counts.sort()
    
    total = len(candidates)
    avg = sum(counts) / len(counts) if counts else 0.0
    med = statistics.median(counts) if counts else 0.0
    
    p95 = 0.0
    if counts:
        idx = int(math.ceil(0.95 * len(counts))) - 1
        idx = max(0, min(idx, len(counts) - 1))
        p95 = float(counts[idx])
        
    res = {
        "total_candidates": total,
        "by_method": by_method,
        "overlap": overlap,
        "average_candidates_per_source1": float(avg),
        "median_candidates_per_source1": float(med),
        "p95_candidates_per_source1": float(p95),
        "max_candidates_per_source1": max(counts) if counts else 0,
    }
    
    if truth is not None:
        found = 0
        for c in candidates:
            if (c.get("source1_id"), c.get("target_source"), c.get("target_id")) in truth:
                found += 1
        res["candidate_recall"] = found / len(truth) if len(truth) > 0 else 1.0
        res["missed_true_pairs"] = len(truth) - found
        
    return res


def _record_text(record: dict) -> str:
    return " ".join(
        (
            record.get("brand_name", ""),
            record.get("normalized_address", ""),
            record.get("city_normalized", ""),
            record.get("region_normalized", ""),
            record.get("country_normalized", ""),
            record.get("postal_code_normalized", ""),
        )
    ).strip()


def _shingles(value: str, size: int) -> set[str]:
    compact = re.sub(r"\s+", " ", value.casefold()).strip()
    if not compact:
        return set()
    padded = f"  {compact}  "
    if len(padded) <= size:
        return {padded}
    return {padded[index : index + size] for index in range(len(padded) - size + 1)}


def _minhash(shingles: set[str], permutations: int, seed: int) -> tuple[int, ...]:
    prime = 2**61 - 1
    if not shingles:
        return tuple([prime] * permutations)
    hashed_shingles = [
        int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
        )
        % prime
        for shingle in shingles
    ]
    signature = []
    for multiplier, offset in _minhash_coefficients(seed, permutations):
        signature.append(
            min((multiplier * value + offset) % prime for value in hashed_shingles)
        )
    return tuple(signature)


@lru_cache(maxsize=32)
def _minhash_coefficients(seed: int, permutations: int) -> tuple[tuple[int, int], ...]:
    prime = 2**61 - 1
    coefficients = []
    for index in range(permutations):
        material = f"{seed}:{index}".encode("ascii")
        digest = hashlib.blake2b(material, digest_size=16).digest()
        multiplier = int.from_bytes(digest[:8], "big") % (prime - 1) + 1
        offset = int.from_bytes(digest[8:], "big") % prime
        coefficients.append((multiplier, offset))
    return tuple(coefficients)


def _coarse_score(left: dict, right: dict) -> float:
    name_left = set(left.get("name_tokens", []))
    name_right = set(right.get("name_tokens", []))
    address_left = set(left.get("address_tokens", []))
    address_right = set(right.get("address_tokens", []))
    name = _jaccard(name_left, name_right)
    address = _jaccard(address_left, address_right)
    postal = bool(
        left.get("postal_code_normalized")
        and left.get("postal_code_normalized") == right.get("postal_code_normalized")
    )
    street = bool(
        left.get("street_number")
        and left.get("street_number") == right.get("street_number")
    )
    return min(1.0, 0.68 * name + 0.2 * address + 0.08 * postal + 0.04 * street)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0