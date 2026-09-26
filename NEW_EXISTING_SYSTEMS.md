# New Existing Systems: Analysis, Shortcomings & What to Borrow

> **Reference Guide for the Amazon Business Entity Resolution Challenge**  
> *Detailed analysis of existing systems NOT in your original file, dissecting where each system lacks and the exact components/techniques to adopt into our winning pipeline.*

---

## Quick Reference Summary Table

| System | Primary Paradigm | Where It Lacks in This Challenge | What We Can Borrow / Steal |
| :--- | :--- | :--- | :--- |
| **1. Dun & Bradstreet (D&B)** | Enterprise Master Registry | Relies on proprietary external databases (Disqualification risk) | Multi-tier confidence grading & trade vs. legal name separation |
| **2. AWS Entity Resolution** | Cloud Managed Service | Proprietary AWS cloud API; cannot run offline in self-contained zip | Two-tier architecture: deterministic rule filtering + ML linkage |
| **3. Senzing (G2 Engine)** | Principle-Based Corporate ER | Closed-source enterprise SDK; heavy commercial license | **Dynamic Token Rarity (IDF)**: down-weight common noise words (`Pvt`, `Near`) |
| **4. Tamr B2B Mastering** | Machine-Learned MDM | Requires interactive human feedback loop; proprietary | Feature clustering on procurement-style messy addresses |
| **5. OpenCorporates Reconcile** | W3C Registry Reconciliation | Cloud API; name-centric; lacks address deep-disambiguation | Country jurisdiction isolation & international corporate registry concepts |
| **6. DeepBlocker** | Deep Vector Blocking (ANN) | Computationally heavy; does not do final matching/scoring | **Dense Bi-Encoder Blocking**: catches Indian landmarks that bypass rule blockers |
| **7. pyJedAI** | Graph Meta-Blocking | Academic toolkit; complex hyperparameter tuning | **Cardinality & Weight Edge Pruning (CEP/WEP)**: slashes candidate noise by 90%+ |
| **8. MinHash LSH (`datasketch`)**| Sub-quadratic Probabilistic Hash | Only does candidate generation; no pairwise classification | **Character 3-gram Shingling**: immune to typos, transpositions, and script noise |
| **9. Jellyfish-7B (LLM)** | Instruction-Tuned 7B LLM | High latency; costly inference across large candidate sets | Structured prompt reasoning on ambiguous edge cases |
| **10. ComEM (COLING 2025)** | LLM "Select" Framework | Complex multi-step LLM prompting pipeline | **The "SELECT" Paradigm**: evaluates candidate pools and explicitly picks "NONE" for singletons |
| **11. ModernBERT / DeBERTa-v3** | SOTA Cross-Encoder | Must be fine-tuned; requires strong blocking to avoid slow inference | **Disentangled Attention**: 8k context, catches subtle street number / branch mismatches |
| **12. FAMER (Univ. of Leipzig)**| Multi-Source Graph Resolution | Distributed Spark dependency; complex setup | **Multi-Source Bipartite Constraint**: guarantees S2/S3 records link to at most ONE S1 |
| **13. `cleanco` / `cleanname`** | Business Legal Suffix Engine | Only strips text; no fuzzy comparison or scoring | **Country-Specific Legal Suffix Dictionaries** for US, India, and France (`SARL`, `SAS`) |
| **14. `libpostal`** | Statistical Address Parser | Giant C model (2GB+); installation hurdles | Street prefix ordering logic for French addresses (`Rue`, `Boulevard`, `Allée`) |

---

## Detailed Breakdown of Each New System

---

### 1. Dun & Bradstreet (D&B) D-U-N-S® Matching Engine
* **What It Is**: The global commercial benchmark for business entity resolution. Resolves vendor, supplier, and enterprise records against D&B's master database of 500M+ businesses.
* **Where It Lacks**:
  * **Requires External Databases**: It achieves its high accuracy by matching against D&B's proprietary global registry—which is **strictly prohibited** by the challenge rules (instant disqualification).
  * **Closed & Costly**: Closed enterprise software with no local, runnable code to submit in a hackathon package.
* **What to Borrow**:
  * **Confidence Scoring Tiers**: D&B groups matches into "Confidence Codes" (Grade 10 to Grade 1). Only Grade 8–10 are accepted automatically. We adopt this by setting our acceptance threshold high ($\tau \ge 0.82$) to protect the $F_{0.5}$ precision score.
  * **Trade Name (DBA) vs. Legal Name Separation**: D&B separates corporate legal forms (`Inc`, `LLC`, `Pvt Ltd`) from the operational brand name before matching.

---

### 2. AWS Entity Resolution (Amazon's Native Service)
* **What It Is**: Amazon's proprietary AWS cloud service for linking and deduplicating customer and business records without shared keys.
* **Where It Lacks**:
  * **Cloud-Only Dependency**: It runs as a managed AWS service (`boto3.client('entityresolution')`). You cannot run it offline in the required student container/zip archive.
  * **No Custom $F_{0.5}$ Tuning**: Its ML models optimize for standard accuracy/F1, not macro $F_{0.5}$ with singleton zero-penalties.
* **What to Borrow**:
  * **The Two-Tier Architecture**: AWS separates matching into two stages:
    1. *Rule-Based Tier*: Exact postal code + normalized name (100% precision, zero false merges).
    2. *ML Linkage Tier*: Fuzzy machine learning for ambiguous records that fail rule matching.
  * We borrow this exact two-tier pattern in our local Python pipeline.

---

### 3. Senzing (G2 Enterprise Engine)
* **What It Is**: An enterprise real-time entity resolution engine built by Jeff Jonas, widely used for corporate entity disambiguation, anti-money laundering, and fraud detection.
* **Where It Lacks**:
  * **Proprietary & Native C SDK**: Closed commercial product; cannot be distributed inside our hackathon submission zip.
* **What to Borrow**:
  * **Dynamic Token Rarity (IDF-Style Weighting)**:
    * In business names, words like `"Pvt"`, `"Ltd"`, `"Enterprises"`, `"Company"` appear thousands of times and provide zero proof of identity.
    * In addresses, words like `"Road"`, `"Street"`, `"Near"`, `"Opp"` are ubiquitous noise.
    * Senzing gives low weight to frequent tokens and exponential weight to rare tokens. We adopt this by using **TF-IDF token weighting** in our feature extraction.
  * **Sequence Neutrality**: Decisions are invariant to data input order.

---

### 4. Tamr B2B Supplier & Customer Mastering
* **What It Is**: Created by Turing Award winner Michael Stonebraker. Built specifically to merge messy procurement and supplier records across disparate ERP systems (SAP, Oracle) into a master catalog.
* **Where It Lacks**:
  * **Human-in-the-Loop Dependency**: Tamr relies on active learning with human operators reviewing ambiguous pairs—infeasible for an automated batch hackathon evaluation.
  * **High Enterprise Cost & Heavy Infrastructure**: Requires dedicated clustering and infrastructure.
* **What to Borrow**:
  * **Domain-Specific Comparison Features**: Tamr engineers specific comparison logic for procurement records, such as **Number-Set Overlap** (checking whether building/suite numbers agree), which we incorporate directly.

---

### 5. OpenCorporates Reconciliation Engine
* **What It Is**: The reconciliation engine powering the world's largest open corporate database (200M+ companies across 140+ jurisdictions).
* **Where It Lacks**:
  * **External API Only**: Operates via web HTTP endpoints; using it violates the offline-only rule.
  * **Address Blindness**: Primarily focuses on company name and jurisdiction code; weak on complex, unstructured landmark addresses.
* **What to Borrow**:
  * **Strict Country Jurisdiction Routing**: OpenCorporates never compares companies across different jurisdiction boundaries. We adopt this by strictly partitioning candidate generation by `country`.

---

### 6. DeepBlocker (PVLDB 2021)
* **What It Is**: A deep learning framework dedicated entirely to the **blocking (candidate generation)** stage, developed by Megagon Labs and UW-Madison.
* **Where It Lacks**:
  * **Candidate Generation Only**: It only retrieves top-$K$ candidate pairs; it does not perform pairwise classification, scoring, or decision filtering.
  * **RAM Intensive**: Storing dense embeddings and building FAISS indexes for millions of pairs can strain memory limits.
* **What to Borrow**:
  * **Dense Vector Bi-Encoder Retrieval**: Converts the entire record string into a dense vector (via `all-MiniLM-L6-v2`) and uses Approximate Nearest Neighbors (ANN) to retrieve candidates.
  * *Why this is critical for us*: It catches Indian landmark variations (*"Near SBI ATM"* vs *"Opposite State Bank"*) where string tokens have zero overlap.

---

### 7. pyJedAI / JedAI (SciFY & University of Athens)
* **What It Is**: A comprehensive academic ER toolkit in Python and Java focusing on scalable blocking, block cleaning, and entity clustering.
* **Where It Lacks**:
  * **Hyperparameter Sensitivity**: Requires complex manual configuration across multiple pipeline components (block building, cleaning, comparison cleaning).
  * **Generic Text Models**: Lacks domain-tailored business logic (e.g., French corporate suffix detection or Indian PIN code hierarchy).
* **What to Borrow**:
  * **Meta-Blocking Edge Pruning (WEP / CEP)**:
    * Standard blocking produces millions of redundant pairs.
    * pyJedAI weights edges in a blocking graph and prunes the weakest edges.
    * We borrow the **top-$K$ per entity pruning rule** to guarantee our candidate pool stays lean ($\le 15$ candidates per $S1$).

---

### 8. MinHash LSH (Locality-Sensitive Hashing via `datasketch`)
* **What It Is**: A sub-quadratic algorithm designed to identify similar documents or text strings in massive datasets using randomized hash permutations.
* **Where It Lacks**:
  * **No Final Classification**: Generates candidate sets based on Jaccard set overlap; cannot evaluate semantic importance, token ordering, or numeric coordinates.
* **What to Borrow**:
  * **Character 3-Gram Shingling**: Decomposing business names into character 3-grams (`"apple"` $\to$ `{"app", "ppl", "ple"}`) and hashing them into LSH bands.
  * *Why we borrow it*: It is blazing fast, runs locally without a GPU, and absorbs spelling typos and transliteration noise (`Laxmi` $\leftrightarrow$ `Lakshmi`) with zero manual rules.

---

### 9. Jellyfish-7B (Chang et al., 2024 - Apache 2.0)
* **What It Is**: An instruction-tuned 7-billion parameter open-weights LLM explicitly trained on entity matching and data preprocessing benchmarks.
* **Where It Lacks**:
  * **Inference Throughput**: Running a 7B LLM across 100,000 candidate pairs is computationally prohibitive within hackathon runtimes without an enterprise GPU cluster.
* **What to Borrow**:
  * **Structured Disambiguation Prompt Design**: Jellyfish demonstrates how to prompt a language model to contrast two records:
    * Evaluating Name Similarity + Address Specificity + Country match.
  * We can use Jellyfish as a **final-stage verifier** exclusively on the top 1% hardest, most ambiguous candidate pairs.

---

### 10. ComEM: Compound Entity Matching (COLING 2025)
* **What It Is**: A cutting-edge research framework introducing the **"Match, Compare, Select"** paradigm for LLM-based entity matching.
* **Where It Lacks**:
  * **Computational Cost**: Evaluating multiple records simultaneously consumes significant context window space.
* **What to Borrow**:
  * **The "SELECT" Paradigm**:
    * Instead of asking a model: *"Does (S1, S2) match?"* (isolated binary question), ComEM presents the reference record $S1_i$ alongside its candidate set $\{Cand_1, Cand_2, \dots, Cand_K\}$ and asks:
      > *"Which of these candidates (if any) is the true match? Choose one, or output 'NONE'."*
    * **Why this is golden for us**: The ability to choose **"NONE"** directly protects **singletons**, preventing false merges and maximizing $F_{0.5}$.

---

### 11. ModernBERT (Dec 2024 - Apache 2.0) & DeBERTa-v3
* **What It Is**: Modernized transformer architectures featuring FlashAttention-2, native 8,192-token context windows, and Disentangled Attention.
* **Where It Lacks**:
  * **Inference Latency on Large Datasets**: Evaluating cross-attention over all pairs requires a prior blocking stage.
* **What to Borrow**:
  * **Disentangled Word-Order Cross-Attention**: DeBERTa-v3/ModernBERT separates content from relative position. We use it to ensure differences like *"Building 104"* vs *"Building 720"* heavily penalize match scores even when all other text tokens match.

---

### 12. FAMER (Fast Multi-Source Entity Resolution - Univ. of Leipzig)
* **What It Is**: A specialized research engine engineered specifically for resolving records across **3 or more heterogeneous data sources**.
* **Where It Lacks**:
  * **Heavy Spark Infrastructure**: Built on top of Apache Flink/Spark, introducing operational overhead for a self-contained Python script.
* **What to Borrow**:
  * **The Star-Topology Bipartite Constraint**:
    * In multi-source matching to a master table, each candidate in $S2$ or $S3$ can belong to **at most ONE** entity in $S1$.
    * If multiple $S1$ entities claim the same $S2$ record, FAMER resolves the conflict by assigning it to the link with the highest confidence weight ($\arg\max$). We adopt this directly in our Stage 4 post-processor.

---

### 13. `cleanco` / `cleanname` (Python)
* **What It Is**: An open-source Python utility dedicated exclusively to cleaning, detecting, and stripping corporate legal suffixes across 30+ jurisdictions.
* **Where It Lacks**:
  * **Narrow Scope**: It is solely a string cleaner; it does not do entity resolution or address parsing.
* **What to Borrow**:
  * **Country-Specific Legal Suffix Lists**:
    * **US**: `inc`, `incorporated`, `llc`, `corp`, `corporation`, `co`, `company`, `lp`, `llp`.
    * **India**: `pvt ltd`, `private limited`, `ltd`, `limited`, `llp`, `enterprises`, `bros`.
    * **France**: `sarl`, `sas`, `sa`, `eurl`, `snc`, `sci`.
  * Stripping these suffixes allows computing "true" brand name similarity without suffix noise distorting results.

---

### 14. `libpostal` (C / Python)
* **What It Is**: An open-source statistical international address parser trained on OpenStreetMap data across every postal system in the world.
* **Where It Lacks**:
  * **Heavy Installation & Large Model**: Requires a 2GB compiled C model and complex setup, making it cumbersome for a pinned `requirements.txt` environment.
* **What to Borrow**:
  * **International Address Decomposition Rules**:
    * Notice how French addresses place thoroughfare prefixes first (`Rue`, `Avenue`, `Boulevard`, `Impasse`, `Allée`), whereas English addresses place them last (`St`, `Rd`, `Ave`).
    * We borrow lightweight regex extractors modeling this specific French prefix structure without installing the heavy 2GB library.

---

## Synthesis: How We Combine What We Borrowed

```
┌───────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│ What We Borrowed                      │ Where It Lives in Our Pipeline                              │
├───────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ cleanco / libpostal logic             │ STAGE 1: Offline Preprocessing (Suffixes & Postal regex)   │
│ MinHash LSH + DeepBlocker ANN         │ STAGE 2: Multi-Pass Blocking -> output/candidate_pairs.tsv  │
│ Senzing (Token Rarity) + Tamr Numbers │ STAGE 3: Feature Engineering (IDF weights & digit overlap)  │
│ AWS 2-Tier + ModernBERT / LightGBM    │ STAGE 3: High-Precision Scoring Model                       │
│ D&B Threshold + ComEM / FAMER Assign  │ STAGE 4: Cutoff (tau >= 0.82) & Star-Topology Bipartite     │
└───────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```
