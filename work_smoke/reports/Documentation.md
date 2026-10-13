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

**Validation macro F0.5 (US + India, 2,000 held-out Source-1 entities): 0.96602**
with competitor claims (0.96602 without), by country
india: 0.96098, us: 0.96934. Stack out-of-fold F0.5 (selection split): 0.96610.

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

Splits (seed 13, disjoint Source-1 groups): ce 3,000 (pairwise cross-encoders), lce 2,000
(listwise model), stack 3,000 (all model and rule selection), val 2,000 (reporting only).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** exact cosine top-100 per country over e5-large (1024-d), bge-m3 (1024-d) and
  e5-small (384-d) pre-computed embeddings (no embeddings were computed by this pipeline); an IDF-weighted sparse
  lexical index (name tokens, 5-char name prefix, street tokens, locality tokens, house-number+street composites,
  postcodes, alphanumeric unit tokens) top-100; a sibling hop (each entity's top-5 e5 neighbours'
  10 nearest pool records). Retrieval is exact (GPU matrix products), never approximate.
- **Fusion:** reciprocal-rank fusion (k0=60) decides survival; each retriever's score, rank and gap stay
  separate features. Optional channels are kept only if they add >= 0.1 pt recall on stack entities; kept:
  **e5, bge, e5s, lex, hop**; cap **K = 48** (smallest K reaching 99.7% of recall@100).
- **Recall of the fused list on stack entities (true pairs retrieved / all true pairs):**

| K | recall |
|---|---|
| 16 | 0.9740 |
| 24 | 0.9809 |
| 32 | 0.9845 |
| 48 | 0.9882 |
| 64 | 0.9909 |
| 80 | 0.9918 |
| 100 | 0.9927 |

- **`ename` channel:** character 3-gram TF-IDF (`char_wb`, sublinear tf) name-only search over *empty-address* pool
  records per country, top 5, added **after** the K cut as a guaranteed union and always passed to stage 2 (empty
  addresses are almost never distractors, but name+address retrievers miss their noisy-name copies). Precision of
  its candidates on stack/val: 0.027; true copies found only by it:
  47.
- **Stage-2 candidate set** (= `candidate_pairs.tsv`, the exact set the final model scores): stage-1 top 24
  with p >= 0.003, plus every `ename` candidate; it keeps 99.69% of blocking's true pairs
  (98.69% of all true pairs). Test: 56,264 candidate pairs.
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
- F1 selection (leave-one-country-out US<->India): chosen feature set **full**; scores {"full": {"in_country": 0.9389445028560357, "out_of_country": 0.8391374400688996}, "invariant": {"in_country": 0.9353301795660638, "out_of_country": 0.8343014005144584}}.

**Model type:**
1. Stage 1: LightGBM (3 grouped folds on stack), test scoring on the GPU with an exact tree evaluator.
2. Pairwise cross-encoders fine-tuned on the ce split: `microsoft/mdeberta-v3-base` (MIT, 278M; dev AUC
   0.99290) and `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M; dev AUC 0.99194).
3. Listwise cross-encoder: the fine-tuned bge-reranker backbone reads `[CLS] entity [SEP] [CAND] c1 [SEP] ...`
   for up to 24 candidates; per-candidate match head + "no listed match" head; source and rank
   biases; candidate order shuffled in training, two orders averaged at inference (4992 training lists).
4. Stage 2: LightGBM over the stage-2 set with the six **name-sharing count features** (how many Source-1 entities in
   the country share this entity's / this candidate's normalised name, exact-name flags, empty-address flag; counted
   over the split's whole S1 table, label-free), **structure/competitor features** (reverse search over *all* S1:
   rank of this S1 among the record's nearest S1, gap to the best other S1; name-is-another-S1's, sibling agreement),
   both cross-encoders, and the listwise scores. A gated ladder chosen on stack (with every stack+val entity
   competing in arbitration, as on test):

| configuration | stack F0.5 (selection) | val F0.5 with claims | val F0.5 plain |
|---|---|---|---|
| noce | 0.95062 | 0.95493 | 0.95493 |
| ce_nox | 0.95954 | 0.96081 | 0.96081 |
| ce | 0.96278 | 0.96436 | 0.96436 |
| ce_s | 0.96610 | 0.96602 | 0.96602 |
| ce_lce:ce_s | 0.96540 | 0.96660 | 0.96660 |
| lce_only | 0.85390 | 0.85546 | 0.85505 |

Gates: six features +0.00324 (required +0.002); structure
+0.00332 (required +0.0005); listwise model kept only if +0.0015 over its base.
**Chosen:** `ce_s`, decision `expf05` (param 0.04).

**Threshold selection method:** none fixed — stage-2 scores are isotonic-calibrated (cross-fitted on stack), then for
each entity the subset size maximising the **exact expected F0.5** (Poisson-binomial dynamic programme; optional
Poisson term for true matches outside the candidate set, rate 0.0400) is chosen and
compared with a threshold grid on stack. One-owner arbitration then iterates to a fixpoint (the higher-probability
claim keeps a contested record, the loser re-decides without it).

Top stage-2 features by gain:

| feature | gain |
|---|---|
| p1 | 125,100 |
| ce_m | 20,627 |
| ce_mean_qz | 12,886 |
| ce_b | 6,380 |
| rev_gap_best_other | 3,235 |
| rev_rank | 1,162 |
| p1_gap | 1,076 |
| ce_b_gap | 640 |
| tri_max_cos | 563 |
| name_is_other_s1 | 532 |
| ce_m_gap | 527 |
| bge_cos_qz | 436 |
| lex_s | 428 |
| a_tset | 425 |
| x_n_s1_own_name | 388 |

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** 0.96602 (india: 0.96098, us: 0.96934)
- **Test predictions per country** (label-free checks, README F3):

| country | S1 | predicted empty | mean matches | pool records per S1 |
|---|---|---|---|---|
| france | 755 | 5.56% | 3.35 | 1900.653 |
| india | 2,355 | 4.97% | 3.40 | 2003.212 |
| us | 1,890 | 4.92% | 3.41 | 2019.593 |

- Shift flags: ['france: predicted matches per pool record 0.0018 vs 0.0017 (>5%, F3)']
- Model's own expected F0.5 on test by country (calibrated probabilities, label-free): {'france': 0.95268, 'india': 0.9601, 'us': 0.9699};
  on val the same estimate vs the actual score: {'india': {'expected': 0.962, 'actual': 0.9609808929604043}, 'us': {'expected': 0.97011, 'actual': 0.9693399193486429}}
- Val loss decomposition (F0.5 recovered if that bucket were fixed): {'rejected_in_stage2': 0.01235, 'dropped_before_stage2': 0.0013, 'never_retrieved': 0.00522, 'fp_distractor': 0.00955, 'fp_owner_scored': 0.0, 'fp_owner_not_scored': 0.00587}
  (`fp_owner_not_scored` is handled by test-time arbitration)
- Count features on a France-sized (~260k) S1 table: val change -0.00341
- **Leave-one-country-out** (train on one country's stack, F0.5 on each country's val):

| model | result |
|---|---|
| s2noce: train india | india: 0.9400 | us: 0.9547 |
| s2noce: train us | india: 0.9296 | us: 0.9584 |
| s2ce: train india | india: 0.9567 | us: 0.9583 |
| s2ce: train us | india: 0.9546 | us: 0.9635 |

---

## 6. Other information

- Models: all MIT / Apache-2.0 and <= 568M parameters (limit 8B). No external data, APIs, geocoding or dictionaries.
- Compute: 2 x RTX PRO 6000 (shared). Every stage checkpoints (sharded atomic outputs, resumable training, claimed
  shard queues across GPUs), so the run survives crashes, OOMs from other users and SSH drops.
- Reproduction: see `code/business_entity_resolution/README.md`.
