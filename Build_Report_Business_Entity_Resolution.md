# Build Report: Business Entity Resolution Pipeline
### Amazon ML Challenge 2026 — What We Are Actually Going to Build

Synthesized from `Entity_Resolution_Existing_Solutions.md` (foundational libraries) and `NEW_EXISTING_SYSTEMS.md` (enterprise/research systems) — taking the best technique from each while avoiding every disqualifying or impractical dependency.

---

## 1. Design Principles (why the pipeline looks the way it does)

| Constraint from the challenge | What it rules out | What we do instead |
|---|---|---|
| No external DB/API lookups | D&B, AWS Entity Resolution, OpenCorporates | Everything runs on local, provided data only |
| Model must be MIT/Apache-2.0, ≤8B params | Large closed LLMs, proprietary SDKs (Senzing) | Jellyfish-7B or a small ModernBERT/DeBERTa-v3 cross-encoder, used sparingly |
| F_0.5 punishes false merges hard | Aggressive fuzzy-only matching | High acceptance threshold + explicit "NONE" option for singletons |
| Must run from a self-contained zip, reproducible | Spark-based tools (Zingg, FAMER), 2GB `libpostal` | Pure Python/pandas pipeline; borrow *ideas* from these tools, not their runtime |
| Test set includes unseen France data | Hard-coded US/India-only rules | Country-routed but pattern-generalizable feature design |
| 2–3 day time window | Human-in-the-loop active learning (Dedupe, Tamr) | Fully automated, training-data-driven pipeline, no manual labeling loop |

---

## 2. Pipeline Overview

```
Raw TSVs (Source1/2/3)
        │
        ▼
STAGE 1 — Preprocessing & Normalization
        │
        ▼
STAGE 2 — Multi-Pass Blocking (Candidate Generation)
        │  → output/candidate_pairs.tsv
        ▼
STAGE 3 — Feature Engineering + Scoring Model
        │
        ▼
STAGE 4 — Thresholding + Bipartite Conflict Resolution
        │  → output/matching_results.tsv
        ▼
STAGE 5 — Validation (utils/validate_submission.py)
```

---

## 3. Stage 1 — Preprocessing & Normalization

**Goal:** strip noise that has nothing to do with true identity, before any comparison happens.

- **Legal suffix stripping** (borrowed from `cleanco`): remove country-specific corporate suffixes so brand names compare cleanly.
  - US: `inc, incorporated, llc, corp, corporation, co, company, lp, llp`
  - India: `pvt ltd, private limited, ltd, limited, llp, enterprises, bros`
  - France: `sarl, sas, sa, eurl, snc, sci`
  - Keep the stripped suffix as a *separate feature* (legal-form match/mismatch), don't just discard it — per D&B's "trade name vs legal name" separation.
- **Address regex decomposition** (lightweight version of `libpostal`'s idea, without the 2GB dependency): pull out street numbers, PIN/ZIP/postal codes, and thoroughfare tokens. Handle the fact that French addresses put the road-type word *first* (`Rue de Paris`) while US/India addresses put it *last* (`MG Road`).
- **Lowercasing, punctuation normalization, transliteration-aware cleanup** (e.g., "&" ↔ "and").
- **Country as an open label** — never hard-code the {US, India} set; leave French records to flow through the same generic pipeline.

**Output:** cleaned `name_clean`, `address_clean`, `street_number`, `postal_code`, `country` columns per record.

---

## 4. Stage 2 — Multi-Pass Blocking (Candidate Generation)

Single-technique blocking is fragile — different noise types need different blockers, so we run three in parallel and union the results:

1. **Country-jurisdiction routing** (from OpenCorporates): never generate candidate pairs across different countries. This alone eliminates most impossible pairs immediately.
2. **MinHash LSH with character 3-gram shingling** (from `datasketch`): catches typos and transliteration variants (`Laxmi` ↔ `Lakshmi`) with zero manual rules, and runs fast with no GPU.
3. **Dense bi-encoder retrieval** (DeepBlocker-style, using a small sentence embedding model like `all-MiniLM-L6-v2` + approximate nearest neighbors): catches cases where string tokens have *zero* overlap — e.g., landmark-based Indian addresses (`"Near SBI ATM"` vs `"Opposite State Bank"`).
4. **Meta-blocking edge pruning** (from pyJedAI's WEP/CEP idea): after the union of the above three passes, keep only the top-K (~15) candidates per Source-1 entity by a cheap combined similarity score, so the candidate pool stays lean before the expensive scoring stage.

**Output:** `output/candidate_pairs.tsv` — this is the *last* filtering stage before the model runs, exactly as the challenge spec requires.

---

## 5. Stage 3 — Feature Engineering + Scoring Model

**Features computed per candidate pair:**
- Name similarity: Jaccard, Levenshtein, TF-IDF cosine — but with **IDF-style token rarity weighting** (from Senzing): common noise words (`Pvt`, `Ltd`, `Road`, `Near`, `Opp`) get down-weighted, rare distinctive tokens get up-weighted.
- Legal-form match flag (from the Stage 1 suffix split).
- Address similarity: token overlap on the parsed components, plus a **street-number / suite-number overlap check** (from Tamr) — a strong disambiguating signal (e.g., "Building 104" vs "Building 720").
- Postal code exact/partial match.
- Country match (always true post-blocking, kept as a sanity feature).

**Two-tier scoring model** (from AWS Entity Resolution's architecture):
1. **Rule tier:** exact postal code + normalized name match → auto-accept at very high confidence, zero ambiguity.
2. **ML tier:** everything else goes through a lightweight classical model — Random Forest or LightGBM trained on the labeled training set — to score match probability. This keeps the pipeline explainable (important for the methodology write-up) and fast enough to score every candidate pair without a GPU.
3. **LLM tiebreaker (optional, budget-permitting):** for the small fraction (~top 1–2%) of genuinely ambiguous, borderline-score pairs, use a small permissively-licensed instruction-tuned model (Jellyfish-7B class, ≤8B, Apache-2.0) with the **"SELECT" prompting pattern** (from ComEM): present the Source-1 record with its full candidate shortlist and ask the model to pick the true match *or explicitly output "NONE."* This protects singletons instead of forcing a binary yes/no per pair.

---

## 6. Stage 4 — Thresholding + Conflict Resolution

- **High acceptance threshold** (from D&B's confidence-tier idea): only accept matches above a conservative probability cutoff (τ ≥ ~0.82) — since F_0.5 punishes false merges twice as hard as missed matches, it's better to leave a borderline pair unmatched.
- **Star-topology bipartite constraint** (from FAMER): each Source-2/Source-3 record may be claimed by **at most one** Source-1 entity. If two Source-1 entities both claim the same candidate, assign it to whichever has the higher confidence score (arg-max resolution) and drop the weaker claim.
- **Singleton protection:** entities with no candidate clearing the threshold are correctly output with an empty match list — worth full credit under F_0.5, so we never force a low-confidence guess just to avoid an empty row.

**Output:** `output/matching_results.tsv`, a strict subset of `candidate_pairs.tsv` as required.

---

## 7. Stage 5 — Validation

Run `utils/validate_submission.py` locally before every leaderboard upload to catch formatting rejections (duplicate IDs, unknown entity IDs, missing rows) for free, and hold out part of the training data to self-score with the F_0.5 formula before submitting.

---

## 8. Tech Stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3 + pandas | Matches the TSV/pandas workflow already shown in the problem statement |
| Blocking | `datasketch` (MinHash LSH), `sentence-transformers` (MiniLM) + FAISS/annoy | Lightweight, no Spark/GPU dependency |
| Suffix cleaning | Custom country-keyed suffix dictionaries (cleanco-style) | Avoids adding a whole extra dependency for a small lookup table |
| Feature scoring model | scikit-learn `RandomForestClassifier` or `LightGBM` | Fast, explainable, easy to document in the methodology write-up |
| Optional LLM tiebreaker | Small Apache-2.0 instruction-tuned model, ≤8B params | Satisfies the license/size constraint while adding judgment only where it's cheap to afford |
| Validation | Provided `utils/validate_submission.py` | Required, stdlib-only |

---

## 9. Deliverable Mapping

| Challenge requirement | Where it's produced |
|---|---|
| `output/matching_results.tsv` | Stage 4 |
| `output/candidate_pairs.tsv` | Stage 2 |
| `code/business_entity_resolution/src/` | All Stage 1–4 scripts, modular by stage |
| `README.md` | Run instructions: preprocess → block → score → threshold → validate |
| `requirements.txt` | pandas, scikit-learn/lightgbm, datasketch, sentence-transformers, (optional) small LLM runtime |
| `Documentation_template.md` | Explicitly document each borrowed technique and why (this report is your first draft) |

---

## 10. Key Risks & Mitigations

1. **Blocking misses a true match (recall ceiling).** Mitigated by running three independent blocking passes and unioning them before pruning.
2. **False merges tank F_0.5.** Mitigated by the high threshold, rule-tier auto-accept only on very safe signals, and the "NONE" option in the optional LLM tier.
3. **Overfitting to US/India, failing on unseen France data.** Mitigated by keeping country an open label throughout, and using generalizable similarity/rarity features rather than hard-coded country rules (only the address-parsing regex is country-aware, and it's designed to extend, not restrict).
4. **Time overrun in a 2–3 day window.** Mitigated by skipping every technique that needs human-in-the-loop labeling (Dedupe, Tamr) or heavy infra (Spark, libpostal's 2GB model), keeping the whole pipeline pure-Python and automatable end-to-end.

---

## Bottom Line

We are building a **five-stage, pure-Python, fully local pipeline**: suffix/address normalization → three-way blocking with edge pruning → a two-tier (rule + classical ML) scorer with an optional small-LLM tiebreaker for ambiguous cases → high-threshold bipartite-constrained matching → local validation. It borrows the *proven ideas* from fourteen-plus enterprise and research systems while avoiding every element that would disqualify us (external lookups, non-permissive licenses, unrunnable cloud dependencies) or blow the time budget (active learning, heavy distributed infra).
