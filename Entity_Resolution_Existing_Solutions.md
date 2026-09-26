# Existing Entity Resolution Solutions — Purpose & Limitations

Reference notes for the Business Entity Resolution Challenge (Amazon ML Challenge 2026).

---

## Open-Source Libraries

### 1. Splink (UK Ministry of Justice)
**Purpose:** Probabilistic record linkage using the Fellegi-Sunter statistical model. Estimates a match probability by combining evidence from multiple field comparisons (name similarity, address similarity, etc.). Built-in blocking rules, scales to millions of records via Spark/DuckDB backends.

**Where it lacks:**
- Assumes fields are conditionally independent, which weakens accuracy when name and address errors are correlated (e.g., both fields mangled by the same OCR/transliteration process)
- Probabilistic weights need enough labeled data or careful manual tuning to calibrate well
- Struggles with unstructured, free-text landmark references ("near SBI ATM") since it expects fairly structured fields
- Not designed for the many-to-many matching pattern (one Source 1 entity to multiple Source 2/3 records) without extra custom logic

### 2. Dedupe.io / dedupe (Python)
**Purpose:** Active-learning based deduplication — you label a handful of example pairs, and it learns similarity thresholds and blocking rules automatically.

**Where it lacks:**
- Active learning loop needs a human in the loop labeling pairs, which doesn't scale well to a fixed hackathon dataset with thousands/millions of records
- Performance degrades on very large datasets compared to Splink or Zingg
- Weak on transliteration/script variation (e.g., Indian address romanization inconsistencies)
- Less flexible blocking strategy customization than purpose-built frameworks

### 3. py_entitymatching (Magellan Project, UW-Madison)
**Purpose:** Academic end-to-end ER framework covering the full blocking → matching pipeline, with multiple blocker types (attribute-equivalence, overlap, rule-based) and multiple ML matchers (Random Forest, SVM, Naive Bayes, etc.) built in.

**Where it lacks:**
- Designed primarily for matching two tables, not natively for three independent sources — needs manual adaptation for your Source1/Source2/Source3 structure
- Classical ML matchers depend heavily on hand-engineered similarity features, which requires real tuning effort for noisy multilingual data
- Project has seen limited active maintenance in recent years, so newer noise patterns (e.g. modern transliteration schemes) aren't well handled out of the box

### 4. RecordLinkage Toolkit (Python)
**Purpose:** General-purpose library offering indexing/blocking methods plus a suite of comparison and classification tools for linking records across datasets.

**Where it lacks:**
- Comparison functions (Jaccard, Levenshtein, etc.) are shallow — they measure surface text similarity, not semantic meaning, so "Apple Inc" vs "Pineapple Inc" can score deceptively high
- Blocking indexers are fairly basic; don't scale as efficiently as Splink/Zingg on very large datasets
- No built-in deep learning matcher — purely classical statistics/ML

### 5. Zingg
**Purpose:** Open-source, Apache Spark-based ER engine built for production-scale deduplication and linking, with active learning to reduce labeling effort.

**Where it lacks:**
- Spark dependency adds operational overhead — heavier setup than a pure-Python solution for a time-boxed hackathon
- Active learning step still needs human labeling rounds, which costs time you may not have in a 2–3 day challenge window
- Tuned more for classic "customer/product dedup" use cases than for the specific noise patterns in business names/addresses (legal suffixes, landmark references)

---

## Research-Grade Deep Learning Models

### 6. DeepMatcher (Carnegie Mellon)
**Purpose:** One of the first deep learning frameworks for entity matching — uses attribute-level embeddings plus neural attention to compare record pairs.

**Where it lacks:**
- Needs a fairly large labeled training set to train embeddings well; risk of overfitting on smaller data
- Slower inference than classical ML — not ideal for scoring huge candidate-pair sets without a very strong blocking stage first
- Older architecture (pre-transformer era) — since surpassed in accuracy by transformer-based approaches like Ditto

### 7. Ditto (Megagon Labs)
**Purpose:** Fine-tunes a BERT-style transformer on entity matching, framing it as a text-pair classification task ("do these two serialized records describe the same entity?"). Currently one of the strongest published approaches on ER benchmarks.

**Where it lacks:**
- Computationally heavy — transformer inference over millions of candidate pairs is slow without good blocking and GPU resources
- Larger pretrained models may exceed the challenge's parameter/license constraints, so you'd need a smaller, permissively-licensed model
- Being a black-box neural model, it's harder to explain *why* a match was made — a disadvantage when writing the methodology document the challenge requires
- Still needs quality training data with matching noise patterns represented, or it won't generalize to the unseen France records in the test set

### 8. HierGAT / GNEM (Graph Neural Network-based matchers)
**Purpose:** Model entity matching as a graph problem, useful when relationships are many-to-many (exactly like your Source 1 → multiple Source 2/3 matches).

**Where it lacks:**
- Most complex to implement and tune of all options here — steep learning curve for a time-limited hackathon
- Requires building and maintaining a graph structure over the whole candidate set, adding engineering overhead
- Less mature tooling/community support compared to Splink, Dedupe, or Ditto — fewer ready-made implementations to adapt

---

## Summary: Common Limitations Across All Existing Solutions

1. **Shallow text similarity ≠ true meaning.** Classical methods (Jaccard, Levenshtein, TF-IDF) can be fooled by names that look similar but are different businesses ("Apple" vs "Pineapple") — a serious risk given F_0.5's heavy precision penalty for false merges.
2. **Data hunger.** Every ML/deep learning approach needs enough good labeled examples to learn from; performance drops sharply on noise patterns underrepresented in training (e.g., the unseen France entities in your test set).
3. **Weak handling of transliteration and landmark-based addresses.** None of the libraries above were built with Indian-style landmark references ("near SBI ATM") or heavy transliteration variance as a first-class case — this needs custom feature engineering regardless of which tool you pick.
4. **Two-source assumption.** Most tools (Splink, py_entitymatching, RecordLinkage Toolkit) are designed for linking two tables, not natively for three independent sources feeding one reference table — you'll need to adapt any of them to your specific structure.
5. **Compute vs. speed trade-off.** The more accurate the model (Ditto, GNN-based matchers), the heavier and slower it is — a real constraint in a short hackathon window with a large candidate-pair set to score.
6. **Explainability gap.** Neural approaches (DeepMatcher, Ditto, GNNs) are hard to justify in a methodology write-up compared to classical statistical/ML approaches, where you can point to exact similarity features driving each decision.
7. **Engineering overhead vs. time available.** Active-learning tools (Dedupe, Zingg) and graph-based models (HierGAT/GNEM) cost setup and tuning time that may not be worth it in a 2–3 day event, compared to a leaner classical-features + Random Forest/XGBoost pipeline.

**Practical takeaway:** a hybrid approach — Splink-style or custom blocking for candidate generation, combined with a lightweight classical ML matcher (Random Forest/XGBoost on similarity features) — is likely the most time-efficient, explainable, and precision-controllable route for this specific challenge, given the time constraints and the F_0.5 scoring emphasis on precision.
