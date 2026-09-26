# Project Build Prompt: Business Entity Resolution Pipeline
### Amazon ML Challenge 2026

---

## ROLE

You are a senior ML engineer specializing in entity resolution, record linkage, and information retrieval systems. You write clean, modular, production-quality Python — explainable over clever, and precision-conscious over recall-hungry. You think in pipelines: preprocessing → candidate generation → feature scoring → decision → validation.

---

## PROJECT DEFINITION AND GOALS

Build a complete, offline, self-contained entity resolution pipeline that matches business records across three independently sourced, noisy datasets (Source 1 = clean reference, Source 2 and Source 3 = noisy candidates from different origins).

**Primary goal:** for every Source 1 entity in the test set, output the correct set of matching Source 2/Source 3 entity IDs (zero, one, or many), maximizing the macro-averaged F_0.5 score — a metric that penalizes false merges twice as heavily as missed matches.

**Secondary goals:**
- Generalize to an unseen country (France) present only in the test set, with no France examples in training.
- Produce a fully reproducible, documented, runnable pipeline — no external API/database lookups of any kind.
- Stay within a small, permissively licensed model footprint (MIT/Apache-2.0, ≤8B parameters) if any learned/LLM component is used.

---

## REFERENCES AND INSPIRATION

Borrow proven techniques — not runtimes or dependencies — from the following, adapted for a pure local Python environment:
- **Blocking:** MinHash LSH with character 3-gram shingling (`datasketch`-style); dense bi-encoder retrieval (DeepBlocker-style, small sentence-embedding model + approximate nearest neighbors); meta-blocking edge pruning to cap candidates per entity (pyJedAI-style).
- **Normalization:** country-specific legal suffix stripping (`cleanco`-style); lightweight regex address decomposition capturing country-specific street-token ordering (`libpostal`-style, without the heavy dependency).
- **Feature weighting:** IDF-style token rarity weighting so common noise words (`Pvt`, `Road`, `Near`) don't inflate similarity scores (Senzing-style).
- **Scoring architecture:** two-tier rule-then-ML matcher (AWS Entity Resolution-style): high-precision exact/near-exact rule tier first, classical ML (Random Forest/LightGBM) for the rest.
- **Decision logic:** conservative acceptance threshold protecting precision (D&B confidence-tier style); explicit "NONE" option for ambiguous cases so singletons aren't force-matched (ComEM "SELECT"-style); one-to-one bipartite conflict resolution so no Source 2/3 record is claimed by more than one Source 1 entity (FAMER-style).

---

## FUNCTIONAL REQUIREMENTS

1. Read all TSVs correctly (`sep="\t"`), treating `country` as an open string label — never hard-code to a fixed set.
2. Preprocessing module: clean and normalize `business_name` and `business_address`; split legal suffixes from brand names; parse address components (street number, postal code, thoroughfare).
3. Blocking module: generate a candidate shortlist per Source 1 entity using at least two independent blocking strategies, union the results, then prune to a bounded top-K per entity. Output `candidate_pairs.tsv`.
4. Feature engineering module: compute name similarity (Jaccard, Levenshtein, TF-IDF cosine with rarity weighting), legal-form match flag, address token overlap, street-number overlap, postal code match.
5. Scoring module: rule-tier auto-accept for unambiguous exact matches; ML-tier classifier trained on `train_ground_truth.tsv` for everything else; optional small-LLM tiebreaker only on the most ambiguous top slice of borderline scores.
6. Decision module: apply a conservative probability threshold; resolve one-to-one conflicts by highest-confidence assignment; leave true singletons with an empty match list. Output `matching_results.tsv`.
7. Validation: run the provided `utils/validate_submission.py` against generated outputs before considering a run complete; also implement a local F_0.5 self-scoring script against a held-out training split.

---

## ARCHITECTURE

```
src/
  preprocess.py     # normalization, suffix stripping, address parsing
  blocking.py        # multi-pass candidate generation + pruning
  features.py         # similarity + rarity-weighted feature computation
  scoring.py           # rule tier + ML tier (+ optional LLM tiebreaker)
  decision.py           # thresholding + bipartite conflict resolution
  evaluate.py            # local F_0.5 scorer on held-out split
  run_pipeline.py         # end-to-end orchestration, single entry point
```

Data flows one-directionally through these modules with intermediate outputs saved to disk between stages, so any single stage can be re-run or debugged independently.

---

## DATA / STATE REQUIREMENTS

- Input: `dataset/train/*.tsv`, `dataset/test/*.tsv` as specified in the problem statement.
- Intermediate state: cleaned records, candidate pairs, feature matrices — cached to disk (e.g., parquet/CSV) so re-runs don't repeat expensive blocking or embedding steps.
- No external network calls, no external database lookups, no calls to commercial entity-resolution or geocoding APIs at any stage.

---

## TECH STACK

- Python 3.10+, pandas, scikit-learn or LightGBM
- `datasketch` for MinHash LSH
- `sentence-transformers` (small model, e.g. `all-MiniLM-L6-v2`) + FAISS or annoy for dense retrieval
- Optional: a small (≤8B parameter) Apache-2.0/MIT licensed instruction-tuned model for the ambiguous-case tiebreaker
- Pinned versions in `requirements.txt`

---

## QA / TESTING

- Unit-level sanity checks per module (e.g., suffix stripper correctly separates "XYZ Pvt Ltd" → "XYZ" + legal form "Pvt Ltd").
- Held-out validation split from training data, scored locally with the exact F_0.5 macro-average formula from the problem statement.
- Full run of `utils/validate_submission.py` against generated output files before every submission.
- Sanity-check that `matching_results.tsv` is always a strict subset of `candidate_pairs.tsv`.

---

## PERFORMANCE

- Blocking + pruning must keep the candidate pool small enough (bounded top-K per entity) that the ML scoring tier runs on the full test set without a GPU in a reasonable time.
- Any optional LLM tiebreaker step must be restricted to a small percentage of the most ambiguous pairs only — never run indiscriminately across the full candidate set.

---

## NEGATIVE PROMPTS (things to avoid)

- Do NOT call any external API, database, or geocoding service at any stage — this causes disqualification.
- Do NOT hard-code the country field to {US, India} anywhere in the pipeline — France must flow through the same generic logic.
- Do NOT use a model over 8B parameters or with a non-MIT/Apache-2.0 license as the final decision-making model.
- Do NOT force a match for every Source 1 entity — singletons with no valid candidate must be left with an empty match list.
- Do NOT let a Source 2/Source 3 record be claimed by more than one Source 1 entity in the final output.
- Do NOT introduce heavy infrastructure dependencies (Spark, Flink, multi-GB native libraries) that complicate the self-contained submission zip.
- Do NOT include any human-in-the-loop / manual labeling step in the automated pipeline.

---

## CONTENT AND QUALITY RULES

- Code must be clean, modular, and commented enough that someone unfamiliar with the pipeline can follow it via the README alone.
- All thresholds and design decisions (blocking strategies used, threshold value, why rule-tier vs ML-tier) must be easy to extract into the required methodology document.
- Prioritize precision-safe behavior throughout, per the F_0.5 scoring emphasis — when a decision is close, default to "no match."

---

## FINAL DELIVERABLE EXPECTATIONS

Build straight through to a complete, working pipeline (no phased stop-and-check gates) that produces:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `code/business_entity_resolution/src/` (all pipeline code)
- `code/business_entity_resolution/README.md` (exact reproduction steps: data → blocking → matching → output)
- `code/business_entity_resolution/requirements.txt` (pinned dependencies)
- A completed `Documentation_template.md` describing the methodology, blocking strategy, model architecture, and feature engineering used

The end result should be ready to (1) generate a valid leaderboard submission and (2) pass local validation via `utils/validate_submission.py` without errors.
