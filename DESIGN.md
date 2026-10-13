# LIFTOVER — model-5, v2: friend's measured changes merged in, and a review

**Written:** 2026-09-27, as an update to `LIFTOVER_model5.md` (v1, 2026-09-26).
**Merges:** `MODEL5_FALSE_NEGATIVE_FIXES.md`, `AUDIT_V2_CHANGES.md`, the v2 run log (leaderboard result, `ename`, CE budget), and the two independent reviews in the project (`audit-v2-review.md` and the review that followed it).
**Status legend:** **[M]** measured, source named · **[E]** estimate, reasoning given · **[?]** not measured · **[X]** measured and rejected

Only the document was reviewed, not the code on college-gpu. Anything marked *verify* needs a check on the server.

---

## 0. TL;DR

- **The anchor has moved.** Model-4 + the six name-sharing features (friend's fix 1):
  - val **0.98621**, stack OOF **0.98543** [M]
  - leaderboard **0.98253** (+0.00329 over model-4's 0.979241) [M]
  - This is the number model-5 must beat. It is also the **fallback submission** at every gate.
- **Leaderboard top: 0.9911** [M] (v1 quoted an older 0.9886).
- **Main v1 premise confirmed:** the loss is mostly in the **decision stack**, not retrieval.
  - Model-4 val loses 0.0083 to rejected true copies, 0.0053 to never-retrieved copies, and 0.0041 to false positives [M].
  - Fix 1 was a decision-stack fix, and it produced the biggest gain so far.
- **Merged into model-5** (details in §3):
  1. the six features in stage 2 **(do first)**
  2. the `ename` channel (v1's R4) as a guaranteed union after the K cut
  3. the must-assign rule, unique-name rule and address-only channel (R5) are **removed**, because they were measured and rejected
- **New blockers found in review** (§2): the `ename` count on France, and whether the new channels have any training rows.
- **How to reach 0.99 on the leaderboard** (§4): neither France nor US/India alone gets there. Even a perfect France gives only 0.98827.

---

## 1. What changed since v1 (evidence summary)

| Item | v1 said | Now | Source |
|---|---|---|---|
| Base val | 0.98298 | **0.98621** with the six features | [M] friend, same folds, seeds and calibration |
| Leaderboard | 0.979241 | **0.98253** | [M] submission |
| Val → leaderboard transfer | assumed discounted | **val +0.0032 → leaderboard +0.00329**. One data point, and it mixes countries (§2.4) | [M] |
| Oracle / decision loss | oracle ≈ 0.995, decision loss ≈ 1.2 pt | Confirmed: never-retrieved loss is 0.0053, so oracle ≈ 0.9947; decision loss ≈ 0.0124 | [M] friend's decomposition |
| France stand-in | 0.854 (US-trained stage 1 scoring India) | Leaderboard implies **≈ 0.962, *if* US/India holds its val level on test** (§2.3). 0.854 is obsolete. | [E] |
| R4 name-only over empty-address records | proposed | **Built as `ename`**: TF-IDF `char_wb` 3-grams, k=5, recovers **44%** of missed empty-address copies | [M] friend |
| R5 address-only | proposed | **[X]** recovers 6–12% of missed copies, about +0.0002 val | [M] friend |
| Must-assign | — | **[X]** −0.013 plain val; the emulation gain shrinks as more S1 compete (+0.0041 → +0.0032) | [M] friend |
| Unique-name rule | — | **[X]** −0.0001 once the six features are in | [M] friend |
| Distractor false positives (§2.6 worry) | 40% of test pool may be distractors | Distractors cost only **≈ 0.00006** of val loss. False positives are almost all other S1s' copies. | [M] friend |

**Correction to earlier project notes:** Model-3's report claimed a "blocking ceiling 0.9822, model at 99.9% of it". That is inconsistent with the F0.5 maths. With pair recall 0.9834 and perfect precision, F0.5 ≈ 0.995. v1's oracle is the right number. `plan-0.99-sprint.md` built on the wrong premise, so read its "blocking is the lever" framing as superseded.

---

## 2. Review findings (fix before or while merging)

### 🔴 2.1 `ename` produced exactly 5.00 candidates per S1, including France, but France has 0% empty addresses
- `ename` gave 8,662,720 pairs = 1,732,544 × 5 exactly. So **every** S1 got 5 candidates, including France's 259,452.
- v1 §6 measured **0% empty addresses** in France's test data. So France's `ename` pool should be empty and give 0 candidates.
- **Possible causes:**
  - the "empty address" definition changed in v2
  - the channel falls back to the whole country pool
  - a country-key mismatch
- **Why it matters:** if France is getting 5 name-only candidates from the *full* French pool, they are:
  - about 1.3M new, noisy pairs on 15% of the metric
  - of a kind stage 2 was never trained on, since train `ename` is empty-address-only
- ***Verify now:*** the `ename` pair count per country, and the address-empty rate of the France `ename` documents. If France should have 0, drop those pairs before CE scoring.

### 🔴 2.2 New channels need training rows, or stage 2 flies blind
- The CE upload is `--split test`. For any new channel (`ename`, `nameonly`, `e5_small_tail`), stage 2 must have seen that channel's pairs in **stack** (fit) and **val** (read), each with:
  - CE scores
  - `p1`
  - ranks and gaps recomputed over the union
- Otherwise:
  - the new pairs get NaN or out-of-distribution features on test
  - adding candidates silently shifts `p1_rank` and `p1_gap` for existing pairs
- **Rule for model-5:** a channel enters test only if it also exists in stack and val and passes a stack-OOF gate.
- **If time is short:** ship `ename` only (the one channel with a measured recovery rate), after 2.1 is resolved.

### 🔴 2.3 The "France ≈ 0.962" figure is conditional, and the France quantity check already looks normal
- **The implied score:** `0.98253 = 0.8503 × 0.98621 + 0.1497 × F` gives **F ≈ 0.9616**. That is true only if US/India score their val level on test, which is not measured.
- **v1's own label-free F3 check, run on the six-feature test predictions:**

  | Country | Predicted matches / S1 | Pool / S1 | Ratio |
  |---|---|---|---|
  | US | 3.36 | 5.76 | 0.583 |
  | India | 3.32 | 5.82 | 0.570 |
  | **France** | **3.20** | **5.53** | **0.579** |

- France sits *between* US and India, so by v1's F3 rule (flag if more than 5% off) **no France threshold offset is warranted**.
- **So either France loses on *which* records it picks (precision and ranking), not how many, or France isn't as bad as 0.962.**
- **Next check** (label-free, 15 min): compute the model's own **expected F0.5 per country on test** from calibrated probabilities.
  - US/India well below 0.986: US/India loses on test too, and France is less bad than implied.
  - US/India ≈ 0.986 but France well above 0.962: France is **overconfident**. That's a calibration issue, not a language issue.

### 🟠 2.4 "1:1 transfer" is one data point
- The +0.00329 mixes 85% US/India with 15% France. France's matches per S1 also moved (3.15 → 3.20).
- France has no empty addresses, so the six features' main mechanism barely applies there. If France gained ≈ 0, the US/India test gain was ≈ 0.0039, about **1.2× val**.
- Treat the leaderboard as a trustworthy **A/B instrument**. Don't treat "val = leaderboard" as a rule.

### 🟠 2.5 The six count features scale with S1 table size, and France's table is small
- Test S1 per country: US 663,106 · India 809,986 · **France 259,452**. France's table is roughly ⅓ of the US/India size, so French names look far more unique.
- The model learned `x_n_s1_own_name` = 1 → "likely match" from the bigger US/India tables.
- Add these counts to v1's §6 F1 list (country-invariant features).
- **Test:** recompute the counts on train S1 subsampled per country to about 260k (always keeping the entity itself), refit, read val.
  - If val drops noticeably, train with subsampled counts, or use `count / country_S1_size × 1e6`.

### 🟠 2.6 LOCO and the India pseudo-label simulation leak through the cross-encoder
- v1 F0 ("LOCO with the full stack") and F2 step 3 ("pseudo-label India as if unlabelled") both reuse a CE **trained on India text**.
- Hiding India only from stage 2 does not simulate "never seen the language". Both measurements will overstate transfer.
- **Fix:** retrain the CE per direction (about 65 min each), or state clearly that the result is an upper bound.

### 🟠 2.7 The listwise cross-encoder (LCE) cannot see global counts
- The six features count over the **whole S1 table**. An LCE reads one S1 and its 16 candidates, so it cannot learn "this name is shared by 7 other S1s".
- **Consequence:** prefer the `lgbm + lce` variant over `lce_only`. If `lce_only` is tried, add the counts as a text tag (`[name shared by N]`), as friend's doc suggests.

### 🟠 2.8 Cross-encoder parity on Colab
- mDeBERTa-v3 is unstable in fp16, and a T4 has no bf16.
- Before scoring on Colab:
  - use an A100 or L4 in bf16, or run fp32
  - re-score about 10k **existing** pairs, including empty-address ones, and match model-4's stored scores to < 1e-3
  - confirm the same text template and max_len

### 🟡 2.9 Other points
- **Team constraint "No new embeddings" (v1 §1).** Friend's `nameonly` dense channel (C7) and `e5_small_tail` must reuse existing vectors. *Verify* C7 didn't encode new name-only vectors. If it did, it breaks the constraint and can't ship.
- **CE budget:** 13.0M of the 20.8M new pairs go to the unmeasured `nameonly` and `e5_small_tail` channels. Gate them on recall gain, as v1's §4 says for R1, before spending CE time.
- **Must-assign and competitor claims:** v1 §5.5 "competitor claims on val" is the right way to make val test-like, because other S1s' claims join arbitration as they do on test. Keep it high priority. It also fixes the part of the false-positive loss that val overstates.
- **Reproducibility:** v2 reuses model-4's scored test table (C1). The final zip must regenerate everything from its own folder (v1 §1), so whichever pipeline ships must include that path.
- **HF uploads:** keep tokens out of the code (use an environment variable) and keep the payload repo private.

---

## 3. Merged model-5 design (changes to v1 only)

### 3.1 Stage 2: add the six name-sharing features [M +0.0030 stack OOF on model-4]
- **Definitions:** exactly as `MODEL5_FALSE_NEGATIVE_FIXES.md` §3.1: `x_empty`, `x_name_eq`, `x_name_ns_eq`, `x_n_s1_cand_name`, `x_n_s1_cand_name_ns`, `x_n_s1_own_name` (clip 10).
- **Counts** use each split's own S1 table (train: all 2.2M; test: all 1.73M, France included). They are label-free and don't depend on which S1 are scored.
- **Where:**
  - join on the stage-2 rows in `stacker._attach` for train and test
  - add to every LightGBM variant (`noce`, `ce`, `ce_lce`)
  - keep them **out of** the F1 drop list
- **Normalisation:** if model-5 changed `name_core` or `name_ns`, recompute with model-5's version.
- **Re-run:** `stage2_train` and `stage2_test` only. Model-5 has already finished `feat_*`, `stage1_train` and `s2set_train`, so don't touch those shards.
- **Gate:** model-5 stack OOF with vs without the features gains ≥ +0.002, and `x_n_s1_own_name` and `x_n_s1_cand_name_ns` rank in the top 20.
  - A smaller gain means model-5's other changes (F1 features, CE ensemble) overlap with them. That's fine; keep them anyway.
- **Then:** run §2.5's subsample test.

### 3.2 Retrieval: `ename` replaces v1's R4; R5 is dropped

| ID | v1 | v2 decision |
|---|---|---|
| R1 e5-small dense | proposed | Keep behind the recall gate. Friend's `e5_small_tail` is the same idea; **don't build it twice**. |
| R2 sibling hop | proposed | **Keep. Next retrieval priority.** It targets the ≈ 0.38 pt of never-retrieved copies with scrambled, partial or native-script names [M]. Measure its recall on those pairs first. |
| R3 reverse search | proposed | Keep, low cost. Its `rev_rank` also gives competitor information without labels. |
| R4 empty-address name index | proposed | **Replaced by `ename`**: TF-IDF `char_wb` (3,3), `sublinear_tf`, fitted per country on empty-address pool records, k=5. |
| R5 address-only | proposed | **Drop [X].** |
| R6 `lex_max_df` sweep | proposed | Keep, cheap. |

**`ename` has three traps, and all three must be handled:**
1. **RRF + K cut:** add its top 5 **after** the K cut as a guaranteed union, never through RRF.
2. **Recall gate:** `blocking.recall_probe` with GATE = 0.001 would reject it (it adds about 0.2–0.3 pt of recall). **Exempt it**, and gate on stage-2 stack F0.5 instead.
3. **Stage-1 drop:** `p1 < 0.003` can remove its pairs before stage 2. Always pass `ename` hits into the stage-2 set.

- **Also:** build it for stack and val (§2.2), resolve the France count (§2.1), and add a `from_ename` flag feature.
- **Check:** the fraction of `ename` true pairs in val that reach CE scoring. It should be ≈ 100%.

### 3.3 Decision layer

| Change | Status |
|---|---|
| Expected-F0.5 selection | [M] +0.0003 in v2; already in model-4 |
| `miss_mass` / `MissModel` (v1 §5.5) | **Fit on stack.** With miss mass = 0, the rule chooses too few matches, which fits recall being the weak side. |
| **Per-country** `miss_mass` / threshold | **Only if** the expected-F0.5 check in §2.3 flags France. Missed truths have no score, so the France value must come from an assumption (France's true matches per S1 ≈ train's 3.46) or a leaderboard A/B. |
| Competitor claims on val (v1 §5.5) | **Keep, high priority** (see §2.9) |
| Must-assign | **Drop [X]** |
| Unique-name rule | **Drop [X]** (the model learns it) |

### 3.4 Friend's other v2 items

| Item | Decision |
|---|---|
| **24 structure/competitive features (C3)** | Measured +0.0035, but on a **4-feature** baseline. Port them into model-5's stage 2 on top of the six, and keep them only if the marginal gain on stack OOF is ≥ +0.0005. Every competitor-based feature must be computed over the **whole S1 table** without labels (for example from reverse search), never over "scored S1s". |
| **Geo admin equivalence (C5)** | Measured +0.00014 on val, which has no France. Ship it only after a **leaderboard A/B** (with vs without). This is v1 §6 F4 by another name. |
| **Dense name-only (C7) and `e5_small_tail`** | See §2.9: check the embedding constraint, then gate on recall. |
| **False-positive diagnosis (`20_fp_diag.py`)** | Keep, with a **4th bucket**: "true owner exists but isn't scored in this eval". Test-time arbitration already handles that bucket, so don't count it as available gain. |

### 3.5 Unchanged from v1, but gated harder
- **Two-cross-encoder ensemble (§5.2):** still sensible and inference only, *after* P0.6 confirms model-3's `ce` split can be reproduced.
- **LCE (§5.4):** a big bet (5–7 GPU-hours of test inference). Keep the pilot gate (+0.15 pt stack OOF), measured **on top of the six features**, and prefer `lgbm + lce` (§2.7).
- **France pseudo-labels (§6 F2):** only if §2.3 shows a *scoring* problem, not a calibration one. Use the one-owner rule for negatives (a record confidently given to S1-A is a negative for S1-B), not low-p pairs. Run the India simulation with a CE retrained without India (§2.6).

---

## 4. Targets

Leaderboard ≈ 0.8503 × US/India + 0.1497 × France, where the weights are the test S1 shares.

| France | US/India needed for leaderboard 0.990 |
|---|---|
| 0.962 (implied now) | 0.9949 |
| 0.975 | 0.9926 |
| 0.986 | 0.9907 |
| 1.000 | 0.9882 |

- A perfect France alone gives **0.98827**, so 0.99 needs US/India gains too.
- **The US/India val path from 0.98621** [E]:
  - `ename`: +0.001
  - competitor claims / arbitration done properly: +0.000–0.001 on val (much of it is already realised on test)
  - `miss_mass`: +0.0003–0.001
  - CE ensemble or LCE: 0–0.003, unmeasured
  - Realistic total: **val 0.988–0.991**
- **France** is the swing factor, and it can't be sized until the check in §2.3 is done.

---

## 5. Order of work

### Phase A: must do (about 0–3 h)

| # | Task | Lane | Gate / output |
|---|---|---|---|
| A1 | Six features into model-5 stage 2 (§3.1) | CPU | stack OOF ≥ +0.002; the fallback stays at 0.98253 |
| A2 | `ename` France anomaly (§2.1) | CPU | per-country pair counts; drop France pairs if they shouldn't exist |
| A3 | Label-free France check: expected F0.5 per country on test, plus the top `name_core` tokens in France (are SARL, SAS, EURL still in names?) | CPU | decides whether France is a calibration or a scoring problem |
| A4 | CE parity on Colab (§2.8) before any Colab scoring | Colab | 10k re-score within 1e-3 |
| A5 | Loss decomposition redone on model-4 + six val, with the 4th false-positive bucket | CPU | where the remaining 0.0138 is |

### Phase B: build (about 3–7 h)

| # | Task | Gate |
|---|---|---|
| B1 | `ename` train side (stack + val) with CE, `p1` and union-recomputed ranks, then test | stack OOF gain > 0 |
| B2 | Competitor claims on val + `MissModel` on stack | stack OOF |
| B3 | Sibling hop (R2): recall on never-retrieved scrambled-name pairs | recall gain ≥ 0.1 pt |
| B4 | CE ensemble (P0.6 → P0.7) | stack OOF ≥ +0.1 pt |
| B5 | Port the 24 structure features (§3.4) | marginal ≥ +0.0005 |
| B6 | §2.5 subsample test for the count features | robustness |

### Phase C: bets, only if A and B leave time

| # | Task | Gate |
|---|---|---|
| C1 | LCE pilot, on top of the six features | +0.15 pt stack OOF |
| C2 | France pseudo-labels, only if A3 points to scoring | India simulation with a CE retrained without India |
| C3 | Geo, France threshold offset | leaderboard A/B, one change per submission |

### Leaderboard discipline
- Submit **one change at a time** on top of the current best, and keep a log: change → val delta → leaderboard delta.
- 0.98253 stays the fallback until something beats it on the leaderboard.

### Two pipelines, one ship
- v2 (friend's, model-4 base) is the **shipping line** today. Model-5 is the **upgrade line**.
- Switch the ship to model-5 only when model-5's stack OOF (with the six features) beats v2's **0.98543** by ≥ 0.001.
- Don't let both lines spend GPU on the same work (for example CE-scoring the same `ename` pairs twice).

---

## 6. Success criteria (v2)

- **Leaderboard > 0.98253** with every shipped change A/B-logged. Target **≥ 0.988**; 0.99 needs §4.
- **Stack OOF > 0.98543 + 0.001** before model-5 replaces v2.
- **Validator** passes with `--check-ids`. One owner per record, asserted.
- **No leakage:** no stack or val entity in `reserved.json`. `ename` and every other new channel have train-side rows.
- **Package** reproduces from its own folder, with no token in the code.