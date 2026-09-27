# Amazon ML Challenge 2026 — Business Entity Resolution

### Team Information
- **Team Leader:** Madhan Kumar T
- **Team Members:** Dharshini K, Allen Xavier K
- **Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution
- **Evaluation Metric:** Macro F₀.₅ (Precision-Weighted)
- **Constraints:** 100% Offline, CPU Execution, Strict Submission Schemas

---

## Overview

A production-grade, offline, CPU-runnable entity resolution pipeline that resolves ambiguous business records from **Source 1** into **Source 2** and **Source 3** across 26.4 million records spanning multiple countries and languages.

The pipeline heavily favors **precision over recall** (F₀.₅), supports multi-source matching and singletons, handles noisy multilingual addresses, and strictly complies with the official Amazon ML Challenge submission format.

---

## Architecture

```
Source 1/2/3 TSVs
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 0: Preprocessing                       │
│  • NFKD Unicode normalization                │
│  • French CEDEX/BP routing isolation         │
│  • Landmark extraction (Near/Opposite/...)   │
│  • Corsican & open postal normalization      │
│  • Legal suffix cleanup (GmbH, LLC, Pvt Ltd) │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 1: Multi-Pass Blocking                 │
│  • MinHash LSH (48 perms, 12 bands)          │
│  • Token inverted index with IDF weights     │
│  • Country-adaptive postal prefix indexing   │
│  • Country hard partitioning                 │
│  • Top-K balanced quotas (8 per source)      │
│  • Optional dense FAISS retrieval            │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 2: Feature Engineering (38 signals)    │
│  • Jaro-Winkler, Levenshtein distances       │
│  • Token Jaccard, Set Ratio, Rare Tokens     │
│  • Tri-state street number (+1/0/-1)         │
│  • Double Metaphone / Soundex phonetics      │
│  • Geographic concordance features           │
│  • Interaction terms (name×postal, etc.)     │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 3: Scoring (LightGBM / Random Forest)  │
│  • Hard-negative mining (2:1 ratio)          │
│  • Bounded depth (max_depth=16)              │
│  • Held-out Macro F₀.₅ threshold sweep      │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 4: Decision & Conflict Resolution      │
│  • Restricted auto-accept rule               │
│  • Multi-branch chain guard (≥3)             │
│  • Global 1-to-1 Hungarian assignment        │
│  • Confidence-descending conflict resolution │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 5: Output                              │
│  • matching_results.tsv (official format)    │
│  • candidate_pairs.tsv  (official format)    │
│  • submission.zip (ready for portal upload)  │
│  • Internal diagnostic debug tables          │
└──────────────────────────────────────────────┘
```

---

## Repository Structure

```
├── README.md                                    # This file
├── .gitignore                                   # Keeps datasets & artifacts out of GitHub
└── business_entity_resolution/
    ├── README.md                                # Detailed pipeline documentation
    ├── run.py                                   # CLI entry point
    ├── app.py                                   # Streamlit interactive dashboard
    ├── requirements.txt                         # Python dependencies
    ├── config.example.json                      # Configuration template
    ├── examples/
    │   └── create_demo_dataset.py               # Synthetic data generator (--make-demo)
    ├── src/ber_pipeline/
    │   ├── blocking.py                          # Multi-pass blocking with country partitioning
    │   ├── cache.py                             # Fingerprint-based caching
    │   ├── cli.py                               # Command-line interface
    │   ├── config.py                            # Settings & configuration
    │   ├── dataset_discovery.py                 # Dataset detection & validation
    │   ├── decision.py                          # Decision rules & conflict resolution
    │   ├── diagnostics.py                       # Error analysis
    │   ├── evaluate.py                          # Macro F₀.₅ evaluation & threshold sweep
    │   ├── features.py                          # 38-feature extractor with phonetic fallback
    │   ├── io.py                                # Streaming & buffered TSV I/O
    │   ├── output_adapter.py                    # Official submission format + zip packaging
    │   ├── pipeline.py                          # End-to-end orchestration
    │   ├── preprocess.py                        # Unicode/address/postal normalization
    │   ├── schema.py                            # Column alias resolution & ground truth parsing
    │   └── scoring.py                           # LightGBM + Random Forest dual-engine scorer
    ├── tests/                                   # 33 unit tests (preprocessing, blocking,
    │   ├── test_blocking.py                     #   features, schema, scoring, decision)
    │   ├── test_cache_scoring.py
    │   ├── test_evaluate_decision.py
    │   ├── test_features.py
    │   ├── test_preprocess.py
    │   └── test_schema.py
    └── utils/
        └── validate_submission.py               # Official competition validator
```

---

## Quickstart

### 1. Install Dependencies

```bash
cd business_entity_resolution
pip install -r requirements.txt
```

### 2. Run Unit Tests

```bash
python -m unittest discover -s tests -v
# Expected: 33 tests, 0 failures
```

### 3. Demo Run (Synthetic Data)

```bash
python run.py --make-demo
python run.py --data-root dataset/ --verbose
```

### 4. Official Competition Run

Place official TSVs into `dataset/train/` and `dataset/test/`, then:

```bash
python run.py --data-root dataset/ --verbose
```

### 5. Validate Submission

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

### 6. Launch Interactive Dashboard

```bash
python -m streamlit run app.py
```

---

## Dataset Scale (Official Competition)

| File | Records |
|---|---:|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `train_ground_truth.tsv` | 2,206,821 |
| `test_source1.tsv` | 1,732,544 |
| `test_source2.tsv` | 4,887,273 |
| `test_source3.tsv` | 5,082,316 |
| **Total** | **26,435,994** |

---

## Key Design Decisions

1. **Precision over Recall:** F₀.₅ penalizes precision 4× more heavily than recall. Every architectural choice favors not making false matches over not missing true matches.

2. **Country Hard Partitioning:** Candidates from different countries are rejected at blocking stage, eliminating cross-border false positives (e.g., US Domino's ≠ Indian Domino's).

3. **Hard-Negative Mining:** Training uses blocking-generated near-miss pairs (same city/postal/name tokens but different entities) rather than random negatives, forcing the model to learn subtle discriminative signals.

4. **Dual-Engine Scoring:** LightGBM is preferred when available for faster training; Scikit-Learn Random Forest serves as a universal fallback. Tree depth is bounded to 16 to cap memory usage.

5. **Pure-Python Phonetic Fallback:** A built-in Soundex implementation ensures phonetic similarity is always computed, even without optional C-extension dependencies.
