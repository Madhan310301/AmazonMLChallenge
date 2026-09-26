# Business Entity Resolution Pipeline

### Team Information
- **Team Leader:** Madhan Kumar T
- **Team Members:** Dharshini K, Allen Xavier K
- **Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution

An offline, CPU-runnable record-linkage pipeline for matching Source 1 business
records to Source 2 and Source 3. It is designed to favor precision (F0.5 macro),
allow no match, support multiple valid links when training ground truth demonstrates
them, and treat country as open text rather than a fixed country list.

### Upgraded Architecture Highlights
1. **Unicode NFKD Normalization:** Safe accent decomposition (Société Générale → societe generale).
2. **French Commercial Routing:** Isolates CEDEX / BP routing numbers so they do not contaminate building numbers.
3. **Landmark Extraction:** Separates descriptive landmarks (Near SBI ATM, Opposite Metro Pillar) from core street address.
4. **Corsica / Open Postal Handling:** Supports Corsican alphanumeric postal codes (2A, 2B) and standard international formats.
5. **Source-Balanced Blocking Quotas:** Dedicated candidate budgets for Source 2 (`source2_top_k = 8`) and Source 3 (`source3_top_k = 8`).
6. **Directional Indexing:** Directional search (Index: S2 + S3, Query: S1) with structural self-match rejection.
7. **Hard-Negative Mining:** Prioritizes challenging negatives from blocking stages based on composite lexical/spatial difficulty.
8. **Tri-State Building Number:** Explicit agreement (+1.0), missing/insufficient (0.0), explicit conflict (-1.0).
9. **Phonetic & Token Similarity:** Double Metaphone transliteration matching and RapidFuzz `token_set_ratio`.
10. **Restricted Auto-Accept Rule:** Requires exact name, exact postal, matching country, AND explicit street number agreement.
11. **Multi-Branch Chain Guard:** Brand frequency detection (≥ 3) suppresses auto-accept for multi-branch chains.
12. **Dynamic F0.5 Threshold Sweep:** Sweeps validation thresholds and selects the macro F0.5-maximizing operating point.
13. **Global Conflict Resolution:** Confidence-descending target reservation eliminates order-dependence.
14. **Official Validator Compliance:** Produces `output/submission/` matching results and candidate pairs that pass the official challenge validator with `--check-ids`.

## 1. Dataset structure

Place the competition data under `dataset/`:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Run the commands below from the `business_entity_resolution/` directory.
From the workspace root, prefix paths with `business_entity_resolution/`.

All files are read as tab-separated UTF-8 text. The pipeline resolves common
column aliases automatically. Source records need an ID and a business-name
column; address, postal code, city, region, and country can be absent or blank.
IDs are retained as strings.

Ground truth supports either:

```text
source1_id    source2_id    source3_id
```

or long form:

```text
source1_id    target_source    target_id
```

Multi-valued target IDs in a wide-format cell may be separated by commas,
semicolons, or `|`. `target_source` values should identify Source 2 or Source 3.
Unknown or duplicated source IDs and malformed rows fail with a useful error.

## 2. Installation

Python 3.10 or newer is required:

```bash
cd business_entity_resolution
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The implementation uses scikit-learn, NumPy, and SciPy. There are no external
API, database, geocoding, or model-service calls. No language model or pretrained
embedding download is used.

## 3. One-command execution

With the competition files in the default `dataset/` directory:

```bash
python run.py
```

Use a different data directory or configuration file with:

```bash
python run.py --data-root /path/to/dataset
python run.py --config config.example.json
```

To disable cache reads and writes or clear existing cache entries:

```bash
python run.py --no-cache
python run.py --clear-cache
```

## 4. Individual stage execution

The modules under `src/ber_pipeline/` expose the main steps independently.
The CLI also offers the common preparation stages:

```bash
python run.py --stage inspect
python run.py --stage preprocess
python run.py --stage blocking
python run.py --stage features
python run.py --stage scoring
python run.py --stage decision
python run.py --stage evaluate
```

Each stage prepares any required upstream artifacts and reuses compatible cache
entries. `features` saves feature tables, `scoring` saves scored candidates and
the fitted model, while `decision`/`evaluate`/`all` produce final outputs and
run integrity checks. The `all` stage is the default.

## 5. Configuration

All tuning values are in `config.example.json` or the `Settings` dataclass:

- MinHash signature and band counts
- Candidate cap per Source 1 record and target source
- Random-forest size and seed
- Held-out fraction
- Threshold search range and resolution
- Match cardinality (`infer`, `one_to_one`, `one_to_many`, `many_to_one`,
  `many_to_many`)

The default cardinality is inferred independently for Source 2 and Source 3
from the training pair structure. To override it, set `cardinality` in the
configuration. The default candidate cap is per target source, so the combined
maximum per Source 1 record is twice the configured value.

## 6. Cache behavior

Normalized records, candidate pairs, and feature matrices are cached as JSON.
Cache metadata includes input fingerprints, row count, input column schema,
preprocessing and feature versions, configuration hash, and Python/pipeline
version. A mismatch invalidates the cached value rather than silently reusing
it. Clear the cache with `--clear-cache`.

The optional embedding cache from the broader design prompt is intentionally
not present: this implementation does not download or depend on an embedding
model.

## 7. Blocking strategy

Two separate local blockers are unioned and deduplicated:

1. Character 3-gram MinHash with deterministic banded lookup. This retrieves
   near-duplicate names and addresses despite punctuation or small spelling
   changes.
2. Rarity-weighted token postings, supplemented by exact street-number and
   postal-prefix postings. Very common postings are skipped to keep candidate
   generation bounded.

The union is ranked with a cheap name/address overlap score and pruned to the
configured cap. Candidate rows retain blocker names and a blocking score.
Candidate recall is measured against held-out ground truth before final
threshold selection.

No dense semantic blocker is enabled. Using a pretrained model would require
shipping its licensed weights and ensuring they are already local; the prompt
provided neither a local model nor permission to download one at runtime. The
chosen blockers are deterministic, lightweight, and fully offline.

## 8. Feature engineering

Features include exact normalized name and brand match, token and character
Jaccard, normalized Levenshtein similarity, training-derived IDF cosine and
weighted Jaccard, name length/token counts, legal-form signals, address token
and edit similarity, street number, postal exact/prefix match, city/region/
country compatibility, cross-field interactions, missing-field indicators,
and blocker agreement.

IDF weights are learned from the training partition only. Raw names and
addresses are retained alongside normalized values. Legal-form stripping uses
a generic suffix dictionary and does not restrict accepted country labels.

## 9. Model training and hard negatives

A deterministic Source 1 split holds out validation entities. Validation-only
positive targets are excluded from the fitting pool when they are not also
linked to a fitting entity. Candidates generated by blocking but absent from
ground truth become realistic hard negatives; random all-pairs negatives are
not added. A class-balanced random forest learns from the candidate features.

The held-out data is used for threshold selection, not model fitting. The final
scorer is refit on all training entities after threshold selection. Native
random-forest probabilities are not separately calibrated; the operating
threshold is selected empirically on held-out training data.

## 10. Decision logic

Two conservative deterministic rules handle exact brand/address combinations.
All other pairs use the random-forest probability. A threshold is searched over
the configured interval to maximize the macro F0.5 score, with higher precision
as the tie-breaker. Any candidate at or below the threshold is a `NO_MATCH`.

Target-side conflict handling uses maximum-weight bipartite assignment when
training data implies one-to-one matching. Other inferred cardinalities enforce
only the observed uniqueness constraint; many-to-many links are preserved.

## 11. Evaluation and diagnostics

The held-out report includes macro F0.5, micro precision/recall, false positives
and negatives, candidate recall, average candidate count, and the selected
threshold. The local macro implementation scores each Source 1 entity's set
of targets; an entity with neither true nor predicted links receives 1.0. The
challenge's official metric/validator should be treated as authoritative if it
uses a different empty-set convention.

`output/diagnostics.json` summarizes held-out false positives, false negatives,
missed candidates, exact-name non-selections, and candidate counts by country
and blocker.

## 12. Output files
 
 The pipeline writes both rich internal diagnostics and official contest submission files:
 
 ```text
 output/
 ├── candidate_pairs.tsv              # Internal rich candidate pairs with feature vectors
 ├── matching_results.tsv             # Internal scored matches with confidence and decision reasons
 ├── validation_report.txt            # Formatted text run report
 ├── run_metadata.json                # Complete pipeline execution metadata
 ├── diagnostics.json                 # Error analysis and blocking diagnostics
 ├── schema_summary.json              # Preprocessing schema report
 └── submission/                      # OFFICIAL SUBMISSION FILES
     ├── matching_results.tsv         # Final matches (source1_entity_id \t matched_entity_ids)
     └── candidate_pairs.tsv          # Blocking candidates (source1_entity_id \t candidate_entity_ids)
 ```
 
 The files in `output/submission/` are formatted strictly according to the Amazon ML Challenge 2026 specification:
 - Tab-separated UTF-8
 - Header: `source1_entity_id \t matched_entity_ids` (and `candidate_entity_ids`)
 - Comma-separated target IDs with `S2-` and `S3-` prefixes
 - Every Source 1 entity in the test set has exactly one row (empty for singletons)
 
 ## 13. Official Submission Validation
 
 Run the official submission validator:
 
 ```bash
 python ../Dataset/student_resource/utils/validate_submission.py \
     --matching output/submission/matching_results.tsv \
     --candidate output/submission/candidate_pairs.tsv \
     --test-dir dataset/test \
     --check-ids
 ```
 
 This validates ID existence, ensures no self-matches, checks formatting rules, and confirms that matches are a strict subset of candidate pairs. Exit code 0 indicates the submission is safe and verified.

## 14. Synthetic data and tests

Create a small synthetic dataset (including unseen-country examples) with:

```bash
python run.py --make-demo
python run.py --data-root examples/demo_dataset
```

Run module tests with:

```bash
python -m unittest discover -s tests -v
```

The generated data is only for sanity checks and is not competition data.

## 15. Limitations and next steps

- The official challenge files and validator were not provided with the build
  prompt, so the output schema and exact scorer convention cannot be confirmed.
- Dense retrieval, probability calibration, and a local LLM tiebreaker are not
  used. They add model-weight distribution or runtime burden and were not
  available for evidence-based validation.
- The included threshold is data-tuned only after a training dataset is placed
  at the expected paths; no competition score is claimed before that run.