# Methodology: Business Entity Resolution Pipeline

## Team Information

- **Team Leader:** Madhan Kumar T
- **Team Members:** Dharshini K, Allen Xavier K
- **Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution

---

## 1. Problem Definition & Evaluation Metric

In commercial platforms, business entity records arrive from multiple independent sources without shared keys. Given Source 1 (the deduplicated reference catalog), the task is to identify all matching records in Source 2 and Source 3. A Source 1 entity may match zero (singleton), one, or multiple records across sources.

Because false merges are twice as damaging as missed links in entity resolution, submissions are scored using **Macro-averaged \(F_{0.5}\)**:

\[
F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}
\]

Singletons (Source 1 records with no valid matches) receive an \(F_{0.5}\) of 1.0 when correctly left empty and 0.0 if any false match is predicted.

---

## 2. Preprocessing & Normalization Architecture

The preprocessing stage (`preprocess.py`) standardizes noisy, heterogeneous cross-source records while preserving identity-bearing information:

1. **Explicit Unicode NFKD Normalization:**
   Decomposes combined characters (`unicodedata.normalize("NFKD", text)`) followed by safe accent stripping. Enables robust cross-lingual comparison on unseen domains such as the test set's French entities (e.g., *Société Générale* \(\to\) *societe generale*, *Hôtel Mercure* \(\to\) *hotel mercure*).
2. **Commercial Routing Isolation:**
   Recognizes and isolates French commercial routing markers such as `CEDEX`, `CEDEX 08`, and `BP 402`. Routing numbers are prevented from contaminating building/street number comparisons.
3. **Descriptive Landmark Extraction:**
   Isolates landmark expressions (e.g., *Near SBI ATM*, *Opposite Metro Pillar 42*, *Behind Bus Stand*) into dedicated `landmark_tokens` and `landmark_text` fields. Prevents landmark descriptions from diluting core street address similarity.
4. **Open Postal Code Handling (including Corsica):**
   Supports standard international formats (US 5-digit / ZIP+4, Indian 6-digit PIN) as well as French 5-digit codes and Corsican alphanumeric prefixes (`2A`, `2B`).
5. **Legal Suffix Separation:**
   Extracts and isolates legal forms (e.g., *Pvt Ltd*, *LLC*, *GmbH*, *SAS*, *SARL*, *Inc*) into separate tokens while retaining the root brand name.

---

## 3. Directional Blocking & Source-Balanced Quotas

Blocking (`blocking.py`) constructs a high-recall candidate pool while avoiding quadratic pair comparisons:

1. **Directional Indexing:**
   Strictly directional architecture: indices are built over Source 2 and Source 3 records, and queried by Source 1 records. Redundant comparisons (\(S_1 \leftrightarrow S_1\), \(S_2 \leftrightarrow S_3\), \(S_2 \leftrightarrow S_2\), \(S_3 \leftrightarrow S_3\)) are eliminated.
2. **Source-Balanced Blocking Quotas:**
   Maintains independent candidate budgets for each target source (`SOURCE2_TOP_K = 8`, `SOURCE3_TOP_K = 8`). Source 2 and Source 3 candidates are retrieved, ranked, and capped independently before unioning, preventing noisy or oversized sources from crowding out candidates from the other source.
3. **Multi-Blocker Union:**
   - **Character Trigram MinHash:** 48 permutations and 12 bands via BLAKE2 hashes.
   - **Inverted Token Posting Index:** Rarity-weighted postings based on IDF; overly frequent postings are skipped.
   - **Dense Semantic Retrieval (Optional):** FAISS inner-product index over SentenceTransformer embeddings.
4. **Structural Validity Enforcement:**
   Every candidate pair is structurally validated: `source1_id`, `target_source` \(\in \{\text{source2}, \text{source3}\}\), and `target_id`. Self-matches and invalid source relations are strictly filtered out.

---

## 4. Feature Engineering

The feature vector (`features.py`) contains 38 signals combining lexical, semantic, phonetic, and spatial evidence:

- **Name Similarity:** Exact normalized name, brand exact, character trigram Jaccard, token Jaccard, normalized Levenshtein similarity, RapidFuzz `token_set_ratio`, and Double Metaphone phonetic similarity (e.g., *Laxmi* \(\leftrightarrow\) *Lakshmi*, *Shree* \(\leftrightarrow\) *Sri*).
- **Address & Landmark Similarity:** Address token Jaccard, address edit similarity, IDF-weighted cosine, street name similarity, and landmark token Jaccard.
- **Tri-State Building Number Feature:**
  - `+1.0`: Explicit building number agreement (e.g., `14` vs `14`)
  - `0.0`: Missing or insufficient numeric evidence
  - `-1.0`: Explicit numeric conflict (e.g., `104` vs `850`)
- **Postal Agreement:** Exact postal code equality and informative postal prefix agreement (first 2-3 characters).
- **Locality Match:** City exact match, region exact match, and country match.
- **Interactions:** Name edit \(\times\) postal match, name edit \(\times\) street number agreement, and blocker agreement indicators.

---

## 5. Machine Learning & Hard-Negative Mining

1. **Leakage-Free Partitioning:**
   The training dataset is partitioned into a fitting split and a held-out validation split. Model training, IDF computation, and hard-negative mining use only the fitting partition. The validation split is reserved strictly for evaluation.
2. **Hard-Negative Mining (`_mine_hard_negatives`):**
   Negative pairs generated by the blocking stage (candidates absent from ground truth) are mined and prioritized by difficulty: candidates sharing high name similarity, token overlap, or postal codes are sampled to train the classifier against challenging false merges.
3. **Random Forest Classifier:**
   A 240-tree class-balanced `RandomForestClassifier` scores each candidate pair, producing calibrated match probabilities.

---

## 6. Precision-Safe Decision Tier & Global Conflict Resolution

1. **Restricted Auto-Accept Rule:**
   Deterministic auto-acceptance requires exact normalized business name, exact postal code, matching country, AND explicit non-empty building/street number agreement (`street_number_exact == 1.0`). Exact name and postal code alone are not permitted to auto-match.
2. **Multi-Branch Chain Guard:**
   Detects brands occurring \(\ge 3\) times across the dataset. Chain entities are prohibited from deterministic auto-acceptance and must undergo full ML scoring.
3. **Dynamic \(F_{0.5}\) Threshold Sweep:**
   Sweeps validation thresholds across the range \(0.50 \to 0.99\) on the held-out partition, tracking precision, recall, macro \(F_{0.5}\), and false merges to select the operating point that maximizes the official macro \(F_{0.5}\) metric. The optimal threshold is persisted in `models/metadata.json`.
4. **Global Conflict Resolution (`resolve_conflicts_globally`):**
   When target uniqueness is required, candidate pairs are sorted globally in descending order of confidence. The highest-confidence valid assignment reserves the target, preventing order-dependent greedy assignments.

---

## 7. Submission Output Compliance

The output adapter (`output_adapter.py`) formats final matches and blocking candidates to comply with the official submission specifications:

- `matching_results.tsv`: Tab-separated columns `source1_entity_id\tmatched_entity_ids` (comma-separated, sorted, singletons represented as empty strings).
- `candidate_pairs.tsv`: Tab-separated columns `source1_entity_id\tcandidate_entity_ids`.
- Verified compliant using the official challenge validator `validate_submission.py` with `--check-ids` passing with 0 errors.

---

## 8. Reproduction Instructions

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run test suite (all 33 unit tests)
python -m unittest discover -s tests -v

# 3. Run pipeline on demo data
python run.py --make-demo

# 4. Validate output with official validator
python "../Dataset/student_resource/utils/validate_submission.py" \
    --matching "output/submission/matching_results.tsv" \
    --candidate "output/submission/candidate_pairs.tsv" \
    --test-dir "dataset/test" \
    --check-ids
```