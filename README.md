# Amazon ML Challenge 2026 — Business Entity Resolution

### Team Information
- **Team Leader:** Madhan Kumar T
- **Team Members:** Dharshini K, Allen Xavier K
- **Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution

---

## 📌 Overview

An offline, CPU-runnable, high-precision record linkage and entity resolution pipeline designed to resolve ambiguous business entities from **Source 1** into **Source 2** and **Source 3**. 

The system is optimized for **Macro $F_{0.5}$** (precision over recall), supports multi-source links when present in ground truth, handles open-text international address/country fields, and strictly complies with the official competition submission schema.

---

## 🚀 Key Upgraded Architecture Highlights

1. **Unicode NFKD Normalization**: Decomposes accents, cleans typographical variations, and normalizes international character sets without destroying meaning (`Société Générale` $\to$ `societe generale`).
2. **French Commercial Routing & CEDEX**: Isolates French commercial routing (`CEDEX`, `BP`, postal box numbers) so that commercial routing digits never contaminate building/street number comparisons.
3. **Descriptive Landmark Extraction**: Automatically detects and separates descriptive landmark markers (`Near`, `Opposite`, `Behind`, `Adjacent to`) from core street addresses.
4. **Corsican Alphanumeric & Open Postal Normalization**: Full support for Corsican postal departments (`2A`, `2B`) and variable-format postal codes worldwide.
5. **Source-Balanced Blocking Quotas**: Dedicated candidate budgets for Source 2 (`source2_top_k = 8`) and Source 3 (`source3_top_k = 8`) preventing dominant sources from starving candidates.
6. **Directional Indexing**: Efficient directional candidate generation ($S_1 \to S_2 \cup S_3$) with structural self-match prevention ($S_1 \cap S_1 = \emptyset$).
7. **Hard-Negative Mining**: Samples challenging negatives during training prioritized by composite lexical and spatial difficulty.
8. **Tri-State Building Number Agreement**: Explicit match (`+1.0`), missing/unspecified (`0.0`), or explicit street number conflict (`-1.0`).
9. **Phonetic & Token Similarity**: Double Metaphone phonetic similarity alongside RapidFuzz `token_set_ratio` to withstand colloquial misspellings and transliteration drift.
10. **Restricted Auto-Accept Rule**: High-confidence fast-path that requires exact name, exact postal code, matching country, and positive street number agreement.
11. **Multi-Branch Chain Guard**: Detects chain brands (frequency $\ge 3$) and suppresses auto-accept to prevent false positives across multi-branch retailers.
12. **Dynamic Macro $F_{0.5}$ Threshold Sweep**: Automatically discovers and sets the optimal precision-favoring decision threshold during validation, persisting it to `models/metadata.json`.
13. **Global Conflict Resolution**: Resolves candidate assignments in descending order of confidence score to eliminate arbitrary processing order bias.
14. **Official Validator Compliant**: Generates both `matching_results.tsv` and `candidate_pairs.tsv` in the exact wide format, passing the official `validate_submission.py` test suite with `--check-ids`.

---

## 📂 Repository Structure

```text
├── app.py                                   # Streamlit Interactive Pipeline Dashboard
├── business_entity_resolution/
│   ├── README.md                            # Detailed Pipeline Documentation
│   ├── run.py                               # Pipeline CLI entry point
│   ├── requirements.txt                     # Dependencies
│   ├── src/ber_pipeline/
│   │   ├── blocking.py                      # Multi-stage balanced blocking
│   │   ├── decision.py                      # Decision rules & global conflict resolution
│   │   ├── evaluate.py                      # Macro F0.5 optimization & evaluation
│   │   ├── features.py                      # 14+ feature extractors (Phonetic, Jaro, Tri-state)
│   │   ├── io.py                            # Streaming & buffered TSV I/O
│   │   ├── output_adapter.py                # Official submission format exporter
│   │   ├── pipeline.py                      # End-to-end orchestration
│   │   ├── preprocess.py                    # Unicode, landmark, and postal cleaning
│   │   ├── schema.py                        # Dataset schema detection & mapping
│   │   └── scoring.py                       # ML & rule-based scoring models
│   ├── tests/                               # Comprehensive unit test suite (33 tests)
│   └── utils/
│       └── validate_submission.py           # Official competition validator
```

---

## ⚡ Quickstart

### 1. Installation

```bash
cd business_entity_resolution
pip install -r requirements.txt
```

### 2. Run All Unit Tests

```bash
python -m unittest discover -s tests -v
```

### 3. Generate Synthetic Demo Data & Run Pipeline

```bash
# Generate synthetic dataset with realistic noise and hard negatives
python run.py --make-demo

# Execute end-to-end pipeline
python run.py --data-root dataset/ --verbose
```

### 4. Validate Official Submission Format

```bash
python utils/validate_submission.py \
    --submission-dir output/submission \
    --test-dir dataset/test \
    --check-ids
```

### 5. Launch the Interactive Dashboard

```bash
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to inspect matches, confidence distributions, blocking recall, and decision breakdowns in real time.
