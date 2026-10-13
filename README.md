# model-5 — Business Entity Resolution

Status on 2026-09-27: **finished**. The full run completed at 13:43. The submission in `output/` passes the organisers'
validator with `--check-ids`. Every number below was read from `work/reports/`, `work/models/stage2_bundle.json` and
`work/state/*.done` after the run.

| | |
|---|---|
| Stack out-of-fold macro F0.5 (selection split, 200,000 S1, competitor claims) | **0.98687** |
| Validation macro F0.5 (100,000 held-out S1, US + India) | **0.98739** with competitor claims, 0.98731 plain (India 0.98728, US 0.98747) |
| Chosen configuration | stage-2 LightGBM `ce`: base features + six name-sharing features + mDeBERTa + bge-reranker, exact expected-F0.5 decision |
| Against the v2 line (model-4 + six features, stack 0.98543) | **+0.00144**, above the +0.001 switch rule in `DESIGN.md` §5 |
| Submission | `output/matching_results.tsv` + `output/candidate_pairs.tsv`, validator **PASS** (1,732,544 rows) |
| Package | `PopTards_submission.zip` (148 MB, 28 files), structure as in `ps.pdf` |
| Leaderboard | not submitted yet |

Companion documents:
- `DESIGN.md`: the design this was built from (LIFTOVER v2, the team's plan). It was `README.md` until the run finished.
- `docs/PACKAGE_README.md`: the reproduction guide shipped in the zip (`code/business_entity_resolution/README.md`).
- `work/reports/Documentation.md`: the filled-in methodology template shipped in the zip.

## 1. Pipeline as built

```
dataset/og (HF Jyo-K/PopTards_Machine_Code)   pre-computed embeddings: e5-large, bge-m3, e5-small (no new ones)
  │  fetch (downloads only what is missing) → manifest (fatal: ids, rows, norms, row alignment)
  │  prep: label-free normalisation of all 24.2M records (model-4's text_utils) → splits
  ▼
splits (seed 13, disjoint S1 groups): ce 120k | lce 150k | stack 200k | val 100k
  ▼
retrieval, exact, per country:  e5-large top-100 · bge-m3 top-100 · e5-small top-100 · IDF lexical top-100
  ▼  RRF (k0=60), cap K=64 chosen on stack; hop channel built and dropped by the recall gate
  ▼  + ename: char 3-gram TF-IDF over EMPTY-ADDRESS pool records, top 5, added AFTER the cap (guaranteed union)
exact cosines (3 models) + triangle features + per-country cosine statistics → ~100 pair features
  ▼
stage-1 LightGBM (5 folds on stack) → stage-2 set: top 24 with p1 ≥ 0.003, plus every ename candidate
  ▼                                                              (= candidate_pairs.tsv)
mDeBERTa-v3 CE + bge-reranker-v2-m3 CE (both fine-tuned here on the ce split) + reverse search over all S1
  ▼  listwise CE (trained on ce+lce lists; gated, not adopted)
stage-2 LightGBM ladder chosen on stack with competitor claims → cross-fitted isotonic calibration
  ▼
exact expected-F0.5 per S1 + one-owner arbitration → matching_results.tsv, validator, reports, zip
```

Country is used only as a string key for grouping; no rule depends on a country name.

## 2. Results

### 2.1 Retrieval (stack entities; `work/reports/blocking_recall.json`, `ename_*.json`, `s2set_train.json`)

| | Value |
|---|---|
| Fused recall at K=64: e5-large + bge-m3 + e5-small + lexical | **98.80%** (India 98.15%, US 99.24%; model-4 had 98.50%) |
| + hop channel | 98.86%, i.e. +0.06 pt, below the 0.1 pt gate, so dropped |
| ename: precision of its candidates | 2.59% (the fixes doc measured 2.6%) |
| ename: true copies it finds that the fused top-64 missed | 2,253 |
| Stage-2 set: share of all true pairs | **98.95%** (98.71% before ename), 2.88M rows on stack+val |
| Test stage-2 set | 17.6M pairs, every S1 has candidates; ename gives 5.00 per S1 in each country |

French pool records are 3.0% empty-address (43,078), so ename works on France too. The "0% empty" in `DESIGN.md`
§2.1 is the S1 side, which has no empty addresses in any country.

### 2.2 Models

| Model | Result |
|---|---|
| mDeBERTa-v3-base CE (900k pairs, 1 epoch, 7,032 steps) | dev AUC 0.99953 |
| bge-reranker-v2-m3 CE (900k pairs, 1 epoch, 14,063 steps) | dev AUC 0.99953 |
| Listwise CE (267,044 lists, 1,755 steps, from the fine-tuned bge-reranker) | trained; not adopted (§2.3) |
| F1 feature-set selection (leave-one-country-out, stage-1 style) | **full** kept. Out-of-country 0.8903 vs 0.8867 invariant; US→India 0.852, India→US 0.929 |

### 2.3 Stage-2 ladder (`work/reports/stage2_gates.json`, `work/models/stage2_bundle.json`)

| Configuration | Stack F0.5 | Val F0.5 (claims) | Val F0.5 (plain) | Gate |
|---|---|---|---|---|
| `noce`: base + six features | 0.97638 | 0.97767 | 0.97754 | |
| `ce_nox`: base + both CEs | 0.98403 | 0.98447 | 0.98431 | |
| **`ce`: base + six + both CEs** | **0.98687** | **0.98739** | 0.98731 | six features +0.00285 (≥ +0.002) ✔ |
| `ce_s`: + structure / reverse search | 0.98715 | 0.98770 | 0.98766 | +0.00028 (< +0.0005) ✘ |
| `ce_lce:ce`: + listwise CE | 0.98767 | 0.98794 | 0.98788 | +0.00080 (< +0.0015) ✘ |
| `lce_only` (report only) | 0.98153 | 0.98161 | 0.98135 | |
| `ce` with mDeBERTa only (one-off ablation) | 0.98644 | 0.98716 | | bge-reranker adds +0.00043 |

- Six-feature gain ranks: `x_n_s1_own_name` 5th and `x_n_s1_cand_name_ns` 11th of 108 (the top 20 was required).
- Gain shares in the shipped model: `ce_m` 73.0%, `ce_b` 16.0%, `p1` 5.8%, `ce_mean_qz` 1.0%.
- Decision: expected F0.5, missed-match rate 0.0365 per S1 (fitted on stack; the selection preferred λ = 0).
- Val loss by bucket (F0.5 recovered if that bucket were fixed):
  - rejected in stage 2: 0.0072
  - never retrieved: 0.0032
  - dropped before stage 2: 0.0002
  - false positives: distractor 0.0008, owner scored 0.0001, owner not scored 0.0013 (test-time arbitration handles this last one)
- Calibration: the model's expected val F0.5 is 0.9908 (India) and 0.9888 (US) against actual 0.9873 and 0.9875, so it is slightly overconfident.

### 2.4 Test (`work/reports/test_report.json`)

| Country | S1 | Predicted empty | Mean matches | Matches per pool record | Model's expected F0.5 |
|---|---|---|---|---|---|
| France | 259,452 | 5.83% | 3.20 | 0.578 | 0.973 |
| India | 809,986 | 5.84% | 3.34 | 0.573 | 0.990 |
| US | 663,106 | 5.74% | 3.36 | 0.584 | 0.990 |

- 5,763,551 predicted pairs; 2.3% of matches come from ename candidates. No shift flags.
- France looks like US/India in quantity (the F3 check) but has the lowest expected F0.5. Given the val overconfidence
  above, the France estimate is probably optimistic.
- French legal forms are stripped in 70.4% of names, and none are left inside `name_core`.

### 2.5 Leave-one-country-out (`work/reports/loco.json`, `f1_select.json`)

| | Train India → India / US | Train US → India / US |
|---|---|---|
| stage-2 `noce` | 0.9716 / 0.9783 | 0.9703 / 0.9798 |
| stage-2 `ce` | 0.9867 / 0.9860 | 0.9853 / 0.9869 |
| stage-1 style, leak-free (F1) | 0.9637 / 0.9286 | 0.8519 / 0.9728 |

The stage-2 rows are **upper bounds**. Stage-1 `p1` and both cross-encoders were trained on both countries, so the held-out
country is not unseen (DESIGN §2.6). The leak-free row is the honest transfer estimate.

## 3. Run

```bash
cd /scratch/user4/dai/model-5
scripts/run.sh start            # run / resume everything, detached (supervisor survives SSH drops)
scripts/run.sh status           # per-stage state: done / running / pending / FAILED xN
scripts/run.sh logs             # supervisor + runner log;  scripts/run.sh tail <stage>  for one stage
scripts/run.sh stop             # stop (finished work is kept; start resumes)
M5_SMOKE=1 scripts/run.sh start # end-to-end check on a small sample: work_smoke/, output_smoke/ (~25 min)
cd src && python run_pipeline.py --reset <stage>   # rerun a stage and everything after it
```

- **Python:** `/scratch/user4/dai/model-4/.venv/bin/python` (3.12; torch 2.14 cu130, transformers 5.17). It chains onto another
  user's venv; the pins are in `requirements.txt`. Override with `M5_PY`.
- **Inputs:** `dataset/og`, `dataset/{e5_multilingual_large,bge3,e5-small}`. The HF repos are listed in `src/config.py:HF`;
  e5-large comes from `Jyo-K/PopTard_Qwen_Machine`, not from Big_Boy (which holds bge-m3).
- **Environment:** `M5_GPUS`, `M5_WORK`, `M5_OUTPUT`, `M5_LCE=0`, `M5_TEAM`, `M5_INFER_BF16` (off: measured slower, §5).
- **Time:** the full run took 04:18–13:43 wall-clock. That includes a 20-minute pause to merge DESIGN v2 and about 1.5 h
  lost to external kills (§6).

| Stage (minutes) | | | |
|---|---|---|---|
| dense e5 / bge / e5s, test | 16.3 / 9.6 / 9.8 | feat train / test | 61.0 / 92.7 (2 workers) |
| lexical train / test | 2.5 / 2.2 | stage-1 train | 61.0 |
| ce_train m / b | 48.9 / 95.9 | stage-1 test (CPU workers) | about 60 |
| lce_train | 82.3 | ename train / test | 4.3 / 6.2 |
| featx train / test | 14.0 / 21.4 | rev train / test | 4.6 / 18.0 |
| CE test inference, 17.6M pairs × 2 models | about 4 h on the shared cards | stage2 prelim / train / test | 17.4 / 12.9 / 11.9 |

### Colab lanes (built, not yet used)

`src/remote.py` plus `~/Desktop/ML Challenge/Model5_CE_Colab.ipynb` offload cross-encoder test shards to up to three
Colab notebooks, through a private HF dataset repo:

```bash
# needs a WRITE token in ~/.hf_m5_token (chmod 600)
python src/remote.py export --key b --split test --lanes A,B,C --per-lane 6
python src/remote.py import --key b --split test    # background; imports results as they arrive
python src/remote.py status --key b --split test
```

- The notebook checks parity against server-scored pairs before it uploads anything.
- A lane that stops making progress releases its shards after 2.5 h.
- This path has **not been run end to end yet**.

## 4. Layout

```
src/config.py          paths, HF repos, embeddings, retrievers, cross-encoders, all settings (CFG)
src/run_pipeline.py    DAG runner: process groups, retries, hang watchdog, GPU scheduling, leftover reaping, disk guard
src/stages.py          stage registry + worker entry points
src/common.py          atomic writes, claimed shard queue (stale-claim recovery), GPU workers, adaptive n_workers
src/prep.py            fetch, manifest, normalisation, test TSVs, splits
src/text_utils.py      model-4's label-free normalisation, legal forms, address parsing
src/embeddings.py      e5-large shards / bge-m3 / e5-small matrix loaders
src/retrieve_dense.py  exact dense search, sibling hop, exact cosines + triangle, cosine stats, reverse search (rev)
src/retrieve_lex.py    IDF lexical search
src/blocking.py        RRF fusion, recall gate, K on stack
src/ename.py           empty-address name-only channel
src/features.py        pair features, within-query and per-country context, featx (union with ename)
src/stacker.py         F1 selection, stage 1 (train, CPU test workers, stage1x), stage-2 set
src/cross_encoder.py   CE training (micro-batch on OOM), (q, c)-keyed score cache, text shards, claimed inference
src/lce.py             listwise cross-encoder
src/stage2.py          six features, structure, the ladder and its gates, test decision, submission, reports, LOCO
src/decide.py          metric, calibration, expected F0.5 (with missed-match term), arbitration
src/package.py         zip + generated methodology document
src/remote.py          Colab offload
src/gpu_lgb.py         exact GPU LightGBM evaluator (kept; slower than CPU here, §5)
scripts/run.sh         start | status | stop | logs | tail | gpus  (supervisor loop)
scripts/keepalive.sh   cron keepalive for long runs (installed during the run, removed after)
```

`work/` holds all intermediates and `work/reports/` every report. Shard files are resumable. Model files are named after a hash
of their feature list, so a resumed run never loads a model trained on other columns.

## 5. Measured and rejected

| Idea | Result | Decision |
|---|---|---|
| bf16 weights + SDPA for CE inference | mDeBERTa 1.01×, bge-reranker 0.75×; max \|Δprob\| 0.23 against the scores stage 2 is trained on | off by default |
| GPU LightGBM for stage-1 test | about 6 min per shard per GPU vs 0.9 min on 24 CPU threads | CPU workers |
| Listwise inference including ename-only pairs | lists twice as long, about 6 min per shard | ename-only pairs excluded (NaN in LightGBM) |
| Invariant feature set (F1) | out-of-country −0.0036 | full set kept |
| hop channel | +0.06 pt recall | dropped by the gate |
| Structure / reverse-search features | +0.00028 stack | not adopted |
| Listwise CE | +0.00080 stack | not adopted; test inference skipped automatically |
| Address-only channel, must-assign, unique-name rule | rejected in DESIGN v2 | not built |

## 6. Known issues

1. **The count features depend on S1 table size.**
   - Val drops by −0.0048 when the six counts come from a France-sized (~260k per country) S1 table (§2.5 check, flagged).
   - The test design is pessimistic: it removes S1 but keeps their pool records.
   - This is the main open France risk. Settle it with a leaderboard A/B.
2. **External kills on the shared account.**
   - From 10:32 to 12:08, GPU workers were repeatedly SIGKILLed. It was not the OOM killer: the cgroup shows `oom_kill` 0.
   - The timing matches another Claude session working in `/home/user4/venu/amzn`. That is unproven.
   - Workers now carry `--owner=model-5-pipeline-DO-NOT-KILL`, and there is a notice in `DO_NOT_KILL.md`.
3. **The code changed during the run.**
   - Changes made mid-run: the DESIGN v2 merge, the CPU stage-1 workers, listwise lists without ename-only pairs, and text-shard CE workers.
   - Every change went through the smoke run or was applied before its stage ran, and no output mixes versions.
   - The old train-side CE scores were moved into the (q, c) cache, and their legacy shard folders were renamed.
4. **LOCO overstates transfer** (§2.5).
5. **Not implemented from DESIGN v2:**
   - geo equivalence and a France threshold offset (both need a leaderboard A/B first)
   - the R6 `lex_max_df` sweep
   - the six features in stage 1
   - French pseudo-labels
   - v2's p-based competitor features (reverse search over embeddings was built instead)
6. **Reproduction depends on other users' files:** the venv chain (`/home/user4/Jyo/...`) and the HF cache at
   `/scratch/user4/mtp/cache/huggingface`.
7. **`Documentation_template.md` lists the team members as "[fill in]"**, and the team name is a guess (`PopTards`).
8. **The smoke run's outputs** (`work_smoke/`, `output_smoke/`) are still on disk and can be deleted.

## 7. Compliance

- **Models:** `microsoft/mdeberta-v3-base` (MIT, 278M), `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M), LightGBM (MIT), all
  well under 8B. The pre-computed embeddings are from `intfloat/multilingual-e5-large`, `BAAI/bge-m3` and
  `intfloat/multilingual-e5-small` (MIT); none were computed by this pipeline.
- No external data, dictionaries, APIs or geocoding. Legal forms and all counts come from the provided text, without labels.
- Country is used only as a string key; France goes through the same code path as US and India.
