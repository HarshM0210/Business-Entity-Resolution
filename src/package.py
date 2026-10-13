"""Final submission package (problem statement, "Final Submission Package"):

  <team>_submission.zip
    output/matching_results.tsv, output/candidate_pairs.tsv
    code/business_entity_resolution/{src/, scripts/, utils/, README.md, requirements.txt}
    Documentation_template.md        (filled in from this run's reports)
"""
import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from common import get_logger, load_json
from config import CFG, M5, OUTPUT, WORK

log = get_logger("package")
TEAM = os.environ.get("M5_TEAM", "PopTards")
R = WORK / "reports"


def _j(name, default=None):
    p = R / name
    return load_json(p) if p.exists() else default


def _fmt(x, nd=5):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def documentation():
    bl = _j("blocking_recall.json", {})
    k = _j("k_choice.json", {})
    s2 = _j("s2set_train.json", {})
    b = load_json(WORK / "models" / "stage2_bundle.json")
    tr = _j("test_report.json", {})
    f1 = _j("f1_select.json", {})
    loco = _j("loco.json", {})
    ce = {x: _j(f"ce_train_{x}.json", {}) for x in ("m", "b")}
    lce = _j("lce_train.json", {})
    man = _j("manifest.json", {})
    fi = pd.read_csv(R / "feature_importance.tsv", sep="\t").head(15) if (R / "feature_importance.tsv").exists() else None
    sub_rows = "\n".join(f"| {s} | {v:.4f} |" for s, v in sorted(bl.get("subsets", {}).get(
        "+".join(k.get("retr", [])), {}).items(), key=lambda x: int(x[0]))) if bl else ""
    test_rows = "\n".join(f"| {c} | {v['n_s1']:,} | {v['empty_rate']:.2%} | {v['mean_pred']:.2f} | {v.get('pool_per_s1', '')} |"
                          for c, v in tr.get("test_pred_stats", {}).items())
    var_rows = "\n".join(f"| {v} | {x['stack']:.5f} | {x['val_claims']:.5f} | {x['val_plain']:.5f} |"
                         for v, x in b["ladder"].items())
    gates = _j("stage2_gates.json", {})
    six = gates.get("six_features_gate", {})
    st = gates.get("structure_gate", {})
    en = _j("ename_train.json", {})
    dec = _j("val_loss_decomposition.json", {})
    subs = b.get("count_subsample_test") or {}
    fi_rows = "\n".join(f"| {r.feature} | {r.gain:,.0f} |" for r in fi.itertuples()) if fi is not None else ""
    loco_rows = "\n".join(f"| {name} | " + " | ".join(f"{c}: {v:.4f}" for c, v in res.items()) + " |"
                          for name, res in loco.items() if isinstance(res, dict))
    return f"""# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** {TEAM}
**Team Members:** [fill in]
**Submission Date:** {pd.Timestamp.now():%Y-%m-%d}

---

## 1. Executive Summary

A blocking-then-ranking pipeline. Candidates come from exact dense search over three pre-computed multilingual
embeddings (e5-large, bge-m3, e5-small), an IDF-weighted lexical index and a sibling-hop channel, fused with
reciprocal-rank fusion. A LightGBM stage-1 model over ~100 country-agnostic pair features narrows each Source-1
entity to at most {CFG['s2_topn']} candidates; two fine-tuned pairwise cross-encoders (mDeBERTa-v3, bge-reranker-v2-m3)
and a **listwise cross-encoder** that reads the entity and all its candidates in one transformer pass score that
set; a stage-2 model is calibrated and each entity's match set is chosen by **exact expected-F0.5 maximisation**
followed by one-owner arbitration. Every choice (retrieval channels, cap K, feature set, model variant, decision
rule) is made on a held-out *stack* split; validation is reported only.

**Validation macro F0.5 (US + India, {sum(v['n_s1'] for v in b['val_pred_stats'].values()):,} held-out Source-1 entities): {b['val_f05']:.5f}**
with competitor claims ({b['val_f05_plain']:.5f} without), by country
{', '.join(f'{c}: {v:.5f}' for c, v in b['val_f05_by_country'].items())}. Stack out-of-fold F0.5 (selection split): {b['stack_f05']:.5f}.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train: US + India; test adds France (15% of test S1), which has no labels. Country is treated as an open set:
  it only partitions retrieval by string equality and is never a model feature.
- Noise: abbreviations and legal-form variants (`Pvt Ltd` / `Private Limited`, `SARL`, `SAS`), transliteration and
  accent corruption, typos, word-order swaps, DBA/trade names, partial or reordered addresses, house-number decoys.
- Ground truth: {man.get('ground_truth', {}).get('n_matched', 'n/a'):,} true pairs; no Source-2/3 record belongs to two
  Source-1 entities ({man.get('ground_truth', {}).get('multi_owner', 'n/a')} multi-owner records), which makes
  one-owner arbitration valid. Training singleton rate {tr.get('train_singleton_rate', float('nan')):.2%}, mean
  {tr.get('train_mean_true', float('nan')):.2f} true matches per entity.

### 2.2 Solution Strategy
**Approach Type:** Hybrid — multi-channel blocking + gradient-boosted ranking + pairwise and listwise transformer
cross-encoders + expected-F0.5 set selection.
**Core Innovation:** a listwise cross-encoder (every candidate attends to the query and to every other candidate,
so decoys and sibling copies are resolved jointly), exact expected-F0.5 set selection with a missed-match term,
and label-free, country-invariant features chosen by leave-one-country-out validation for the unseen country.

Splits (seed {CFG['seed']}, disjoint Source-1 groups): ce {CFG['n_ce']:,} (pairwise cross-encoders), lce {CFG['n_lce']:,}
(listwise model), stack {CFG['n_stack']:,} (all model and rule selection), val {CFG['n_val']:,} (reporting only).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** exact cosine top-{CFG['dense_k']} per country over e5-large (1024-d), bge-m3 (1024-d) and
  e5-small (384-d) pre-computed embeddings (no embeddings were computed by this pipeline); an IDF-weighted sparse
  lexical index (name tokens, 5-char name prefix, street tokens, locality tokens, house-number+street composites,
  postcodes, alphanumeric unit tokens) top-{CFG['lex_k']}; a sibling hop (each entity's top-{CFG['hop_seed']} e5 neighbours'
  {CFG['hop_nn']} nearest pool records). Retrieval is exact (GPU matrix products), never approximate.
- **Fusion:** reciprocal-rank fusion (k0={CFG['rrf_k0']}) decides survival; each retriever's score, rank and gap stay
  separate features. Optional channels are kept only if they add >= 0.1 pt recall on stack entities; kept:
  **{', '.join(k.get('retr', []))}**; cap **K = {k.get('K')}** (smallest K reaching {CFG['k_recall_frac']:.1%} of recall@{CFG['k_max']}).
- **Recall of the fused list on stack entities (true pairs retrieved / all true pairs):**

| K | recall |
|---|---|
{sub_rows}

- **`ename` channel:** character 3-gram TF-IDF (`char_wb`, sublinear tf) name-only search over *empty-address* pool
  records per country, top 5, added **after** the K cut as a guaranteed union and always passed to stage 2 (empty
  addresses are almost never distractors, but name+address retrievers miss their noisy-name copies). Precision of
  its candidates on stack/val: {en.get('recall', {}).get('precision', float('nan')):.3f}; true copies found only by it:
  {en.get('recall', {}).get('true_not_in_fused_topK', 'n/a')}.
- **Stage-2 candidate set** (= `candidate_pairs.tsv`, the exact set the final model scores): stage-1 top {CFG['s2_topn']}
  with p >= {CFG['s2_min_prob']}, plus every `ename` candidate; it keeps {s2.get('s2_recall_vs_union', float('nan')):.2%} of blocking's true pairs
  ({s2.get('s2_recall_vs_total', float('nan')):.2%} of all true pairs). Test: {tr.get('n_candidate_pairs', 0):,} candidate pairs.
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
- F1 selection (leave-one-country-out US<->India): chosen feature set **{f1.get('choice', 'n/a')}**; scores {json.dumps(f1.get('scores', {}))}.

**Model type:**
1. Stage 1: LightGBM ({CFG['n_folds']} grouped folds on stack), test scoring on the GPU with an exact tree evaluator.
2. Pairwise cross-encoders fine-tuned on the ce split: `microsoft/mdeberta-v3-base` (MIT, 278M; dev AUC
   {_fmt(ce['m'].get('dev_auc'))}) and `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M; dev AUC {_fmt(ce['b'].get('dev_auc'))}).
3. Listwise cross-encoder: the fine-tuned bge-reranker backbone reads `[CLS] entity [SEP] [CAND] c1 [SEP] ...`
   for up to {CFG['lce_max_cand']} candidates; per-candidate match head + "no listed match" head; source and rank
   biases; candidate order shuffled in training, two orders averaged at inference ({lce.get('lists', 'n/a')} training lists).
4. Stage 2: LightGBM over the stage-2 set with the six **name-sharing count features** (how many Source-1 entities in
   the country share this entity's / this candidate's normalised name, exact-name flags, empty-address flag; counted
   over the split's whole S1 table, label-free), **structure/competitor features** (reverse search over *all* S1:
   rank of this S1 among the record's nearest S1, gap to the best other S1; name-is-another-S1's, sibling agreement),
   both cross-encoders, and the listwise scores. A gated ladder chosen on stack (with every stack+val entity
   competing in arbitration, as on test):

| configuration | stack F0.5 (selection) | val F0.5 with claims | val F0.5 plain |
|---|---|---|---|
{var_rows}

Gates: six features {six.get('stack_gain', float('nan')):+.5f} (required +{six.get('required', 0.002)}); structure
{st.get('stack_gain', float('nan')):+.5f} (required +{st.get('required', 0.0005)}); listwise model kept only if +0.0015 over its base.
**Chosen:** `{b['config']}`, decision `{b['family']}` (param {b['thr']}).

**Threshold selection method:** none fixed — stage-2 scores are isotonic-calibrated (cross-fitted on stack), then for
each entity the subset size maximising the **exact expected F0.5** (Poisson-binomial dynamic programme; optional
Poisson term for true matches outside the candidate set, rate {b.get('miss_rate', float('nan')):.4f}) is chosen and
compared with a threshold grid on stack. One-owner arbitration then iterates to a fixpoint (the higher-probability
claim keeps a contested record, the loser re-decides without it).

Top stage-2 features by gain:

| feature | gain |
|---|---|
{fi_rows}

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** {b['val_f05']:.5f} ({', '.join(f'{c}: {v:.5f}' for c, v in b['val_f05_by_country'].items())})
- **Test predictions per country** (label-free checks, README F3):

| country | S1 | predicted empty | mean matches | pool records per S1 |
|---|---|---|---|---|
{test_rows}

- Shift flags: {tr.get('country_shift_flags') or 'none'}
- Model's own expected F0.5 on test by country (calibrated probabilities, label-free): {tr.get('expected_f05_test')};
  on val the same estimate vs the actual score: {b.get('val_expected_vs_actual_by_country')}
- Val loss decomposition (F0.5 recovered if that bucket were fixed): {dec.get('f05_lost_by_bucket')}
  (`fp_owner_not_scored` is handled by test-time arbitration)
- Count features on a France-sized (~260k) S1 table: val change {subs.get('delta', float('nan')):+.5f}
- **Leave-one-country-out**, leak-free (every component trained on one country; F1 selection): {json.dumps(f1.get('detail', {}))}
- Leave-one-country-out with the stage-2 model (**upper bounds**: stage-1 scores and cross-encoders saw both
  countries), train on one country's stack, F0.5 on each country's val:

| model | result |
|---|---|
{loco_rows}

---

## 6. Other information

- Models: all MIT / Apache-2.0 and <= 568M parameters (limit 8B). No external data, APIs, geocoding or dictionaries.
- Compute: 2 x RTX PRO 6000 (shared). Every stage checkpoints (sharded atomic outputs, resumable training, claimed
  shard queues across GPUs), so the run survives crashes, OOMs from other users and SSH drops.
- Reproduction: see `code/business_entity_resolution/README.md`.
"""


def run_package():
    stage = WORK / "package"
    if stage.exists():
        shutil.rmtree(stage)
    code = stage / "code" / "business_entity_resolution"
    (stage / "output").mkdir(parents=True)
    code.mkdir(parents=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(OUTPUT / f, stage / "output" / f)
    shutil.copytree(M5 / "src", code / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(M5 / "scripts", code / "scripts")
    shutil.copytree(M5 / "utils", code / "utils")
    shutil.copy2(M5 / "docs" / "PACKAGE_README.md", code / "README.md")
    shutil.copy2(M5 / "requirements.txt", code / "requirements.txt")
    doc = documentation()
    (stage / "Documentation_template.md").write_text(doc)
    (R / "Documentation.md").write_text(doc)
    zp = M5 / f"{TEAM}_submission.zip"
    tmp = zp.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage))
    os.replace(tmp, zp)
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip()
        names = z.namelist()
    assert bad is None, f"corrupt zip member {bad}"
    for need in ("output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
                 "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt",
                 "code/business_entity_resolution/src/run_pipeline.py"):
        assert need in names, f"{need} missing from the package"
    shutil.rmtree(stage)
    log.info(f"package written: {zp} ({zp.stat().st_size / 1e6:.0f} MB, {len(names)} files)")
