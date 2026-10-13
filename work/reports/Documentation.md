# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** PopTards
**Team Members:** [fill in]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A blocking-then-ranking pipeline. Candidates come from exact dense search over three pre-computed multilingual
embeddings (e5-large, bge-m3, e5-small), an IDF-weighted lexical index and a sibling-hop channel, fused with
reciprocal-rank fusion. A LightGBM stage-1 model over ~100 country-agnostic pair features narrows each Source-1
entity to at most 24 candidates; two fine-tuned pairwise cross-encoders (mDeBERTa-v3, bge-reranker-v2-m3)
and a **listwise cross-encoder** that reads the entity and all its candidates in one transformer pass score that
set; a stage-2 model is calibrated and each entity's match set is chosen by **exact expected-F0.5 maximisation**
followed by one-owner arbitration. Every choice (retrieval channels, cap K, feature set, model variant, decision
rule) is made on a held-out *stack* split; validation is reported only.

**Validation macro F0.5 (US + India, 100,000 held-out Source-1 entities): 0.98739**
with competitor claims (0.98731 without), by country
india: 0.98728, us: 0.98747. Stack out-of-fold F0.5 (selection split): 0.98687.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train: US + India; test adds France (15% of test S1), which has no labels. Country is treated as an open set:
  it only partitions retrieval by string equality and is never a model feature.
- Noise: abbreviations and legal-form variants (`Pvt Ltd` / `Private Limited`, `SARL`, `SAS`), transliteration and
  accent corruption, typos, word-order swaps, DBA/trade names, partial or reordered addresses, house-number decoys.
- Ground truth: 7,638,365 true pairs; no Source-2/3 record belongs to two
  Source-1 entities (0 multi-owner records), which makes
  one-owner arbitration valid. Training singleton rate 5.58%, mean
  3.46 true matches per entity.

### 2.2 Solution Strategy
**Approach Type:** Hybrid — multi-channel blocking + gradient-boosted ranking + pairwise and listwise transformer
cross-encoders + expected-F0.5 set selection.
**Core Innovation:** a listwise cross-encoder (every candidate attends to the query and to every other candidate,
so decoys and sibling copies are resolved jointly), exact expected-F0.5 set selection with a missed-match term,
and label-free, country-invariant features chosen by leave-one-country-out validation for the unseen country.

Splits (seed 13, disjoint Source-1 groups): ce 120,000 (pairwise cross-encoders), lce 150,000
(listwise model), stack 200,000 (all model and rule selection), val 100,000 (reporting only).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** exact cosine top-100 per country over e5-large (1024-d), bge-m3 (1024-d) and
  e5-small (384-d) pre-computed embeddings (no embeddings were computed by this pipeline); an IDF-weighted sparse
  lexical index (name tokens, 5-char name prefix, street tokens, locality tokens, house-number+street composites,
  postcodes, alphanumeric unit tokens) top-100; a sibling hop (each entity's top-5 e5 neighbours'
  10 nearest pool records). Retrieval is exact (GPU matrix products), never approximate.
- **Fusion:** reciprocal-rank fusion (k0=60) decides survival; each retriever's score, rank and gap stay
  separate features. Optional channels are kept only if they add >= 0.1 pt recall on stack entities; kept:
  **e5, bge, e5s, lex**; cap **K = 64** (smallest K reaching 99.7% of recall@100).
- **Recall of the fused list on stack entities (true pairs retrieved / all true pairs):**

| K | recall |
|---|---|
| 16 | 0.9731 |
| 24 | 0.9789 |
| 32 | 0.9821 |
| 48 | 0.9857 |
| 64 | 0.9880 |
| 80 | 0.9895 |
| 100 | 0.9906 |

- **`ename` channel:** character 3-gram TF-IDF (`char_wb`, sublinear tf) name-only search over *empty-address* pool
  records per country, top 5, added **after** the K cut as a guaranteed union and always passed to stage 2 (empty
  addresses are almost never distractors, but name+address retrievers miss their noisy-name copies). Precision of
  its candidates on stack/val: 0.026; true copies found only by it:
  2253.
- **Stage-2 candidate set** (= `candidate_pairs.tsv`, the exact set the final model scores): stage-1 top 24
  with p >= 0.003, plus every `ename` candidate; it keeps 99.93% of blocking's true pairs
  (98.95% of all true pairs). Test: 17,624,889 candidate pairs.
- **How true matches were not lost:** three dense models with different failure modes plus a lexical channel that
  works on empty/short addresses, the hop channel for scrambled-name entities whose sibling records are close, and a
  stage-1 cap chosen from measured recall.

---

## 4. Matching Model

**Features used** (all computed from the provided text only; no dictionaries or external data):
- Name: ratio, Jaro-Winkler, Levenshtein, token-sort/set, partial, full-name ratio, no-space ratio/partial,
  abbreviation- and typo-aware token coverage both ways, token Jaccard, unmatched-token counts, containment (DBA),
  acronym match, first-token match, legal-form state (legal forms discovered per country **without labels**).
- Address: full-address ratio/token-set/partial/Jaccard, street coverage both ways, locality coverage, house-number
  state (equal / suffix / range / conflict) and distance, numeric-set Jaccard, postcode and unit-token states.
- Retrieval/context: per-retriever score/rank/gap, RRF score/rank, support count, exact e5/bge/e5-small cosines,
  within-query rank/gap/z-score, **per-country standardised cosines** (label-free statistics of random same-country
  pairs), S2<->S3 triangle corroboration, candidate count.
- F1 selection (leave-one-country-out US<->India): chosen feature set **full**; scores {"full": {"in_country": 0.968239796012073, "out_of_country": 0.890256464170105}, "invariant": {"in_country": 0.9676194728598464, "out_of_country": 0.8866509880793355}}.

**Model type:**
1. Stage 1: LightGBM (5 grouped folds on stack), test scoring on the GPU with an exact tree evaluator.
2. Pairwise cross-encoders fine-tuned on the ce split: `microsoft/mdeberta-v3-base` (MIT, 278M; dev AUC
   0.99953) and `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M; dev AUC 0.99953).
3. Listwise cross-encoder: the fine-tuned bge-reranker backbone reads `[CLS] entity [SEP] [CAND] c1 [SEP] ...`
   for up to 24 candidates; per-candidate match head + "no listed match" head; source and rank
   biases; candidate order shuffled in training, two orders averaged at inference (267044 training lists).
4. Stage 2: LightGBM over the stage-2 set with the six **name-sharing count features** (how many Source-1 entities in
   the country share this entity's / this candidate's normalised name, exact-name flags, empty-address flag; counted
   over the split's whole S1 table, label-free), **structure/competitor features** (reverse search over *all* S1:
   rank of this S1 among the record's nearest S1, gap to the best other S1; name-is-another-S1's, sibling agreement),
   both cross-encoders, and the listwise scores. A gated ladder chosen on stack (with every stack+val entity
   competing in arbitration, as on test):

| configuration | stack F0.5 (selection) | val F0.5 with claims | val F0.5 plain |
|---|---|---|---|
| noce | 0.97638 | 0.97767 | 0.97754 |
| ce_nox | 0.98403 | 0.98447 | 0.98431 |
| ce | 0.98687 | 0.98739 | 0.98731 |
| ce_s | 0.98715 | 0.98770 | 0.98766 |
| ce_lce:ce | 0.98767 | 0.98794 | 0.98788 |
| lce_only | 0.98153 | 0.98161 | 0.98135 |

Gates: six features +0.00285 (required +0.002); structure
+0.00028 (required +0.0005); listwise model kept only if +0.0015 over its base.
**Chosen:** `ce`, decision `expf05` (param 0.0).

**Threshold selection method:** none fixed — stage-2 scores are isotonic-calibrated (cross-fitted on stack), then for
each entity the subset size maximising the **exact expected F0.5** (Poisson-binomial dynamic programme; optional
Poisson term for true matches outside the candidate set, rate 0.0365) is chosen and
compared with a threshold grid on stack. One-owner arbitration then iterates to a fixpoint (the higher-probability
claim keeps a contested record, the loser re-decides without it).

Top stage-2 features by gain:

| feature | gain |
|---|---|
| ce_m | 10,524,915 |
| ce_b | 2,307,680 |
| p1 | 831,239 |
| ce_mean_qz | 148,860 |
| x_n_s1_own_name | 87,134 |
| rrf_rank | 73,005 |
| rrf | 65,850 |
| ce_m_gap | 28,186 |
| p1_gap | 15,606 |
| e5_r | 15,061 |
| x_n_s1_cand_name_ns | 13,834 |
| e5_s | 13,752 |
| ename_s | 11,908 |
| e5_cos_qz | 11,886 |
| ce_b_gap | 11,484 |

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** 0.98739 (india: 0.98728, us: 0.98747)
- **Test predictions per country** (label-free checks, README F3):

| country | S1 | predicted empty | mean matches | pool records per S1 |
|---|---|---|---|---|
| france | 259,452 | 5.83% | 3.20 | 5.531 |
| india | 809,986 | 5.84% | 3.34 | 5.824 |
| us | 663,106 | 5.74% | 3.36 | 5.756 |

- Shift flags: none
- Model's own expected F0.5 on test by country (calibrated probabilities, label-free): {'france': 0.97272, 'india': 0.99045, 'us': 0.98998};
  on val the same estimate vs the actual score: {'india': {'expected': 0.99083, 'actual': 0.9872819915698673}, 'us': {'expected': 0.98878, 'actual': 0.9874676146346176}}
- Val loss decomposition (F0.5 recovered if that bucket were fixed): {'rejected_in_stage2': 0.0072, 'dropped_before_stage2': 0.00021, 'never_retrieved': 0.00317, 'fp_distractor': 0.00081, 'fp_owner_scored': 0.00012, 'fp_owner_not_scored': 0.00126}
  (`fp_owner_not_scored` is handled by test-time arbitration)
- Count features on a France-sized (~260k) S1 table: val change -0.00482
- **Leave-one-country-out**, leak-free (every component trained on one country; F1 selection): {"full: train india": {"india": 0.9637097199710807, "us": 0.9285630175609085}, "full: train us": {"india": 0.8519499107793013, "us": 0.9727698720530654}, "invariant: train india": {"india": 0.9629152615536233, "us": 0.9174679436885926}, "invariant: train us": {"india": 0.8558340324700784, "us": 0.9723236841660694}}
- Leave-one-country-out with the stage-2 model (**upper bounds**: stage-1 scores and cross-encoders saw both
  countries), train on one country's stack, F0.5 on each country's val:

| model | result |
|---|---|
| s2noce: train india | india: 0.9716 | us: 0.9783 |
| s2noce: train us | india: 0.9703 | us: 0.9798 |
| s2ce: train india | india: 0.9867 | us: 0.9860 |
| s2ce: train us | india: 0.9853 | us: 0.9869 |

---

## 6. Other information

- Models: all MIT / Apache-2.0 and <= 568M parameters (limit 8B). No external data, APIs, geocoding or dictionaries.
- Compute: 2 x RTX PRO 6000 (shared). Every stage checkpoints (sharded atomic outputs, resumable training, claimed
  shard queues across GPUs), so the run survives crashes, OOMs from other users and SSH drops.
- Reproduction: see `code/business_entity_resolution/README.md`.
