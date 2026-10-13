"""Paths and settings for model-5. Environment overrides:
  M5_SMOKE=1        small end-to-end run in work_smoke/ + output_smoke/
  M5_GPUS=0,1       GPUs the scheduler may use (default: all visible)
  M5_ROOT, M5_WORK, M5_OUTPUT, M5_DATA, M5_EMB   relocate inputs/outputs
  M5_LCE=0          disable the listwise cross-encoder branch
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("M5_ROOT", "/scratch/user4/dai"))
M5 = Path(__file__).resolve().parent.parent            # the package folder (model-5/ or the zip's code dir)
SMOKE = os.environ.get("M5_SMOKE", "0") == "1"
WORK = Path(os.environ.get("M5_WORK", M5 / ("work_smoke" if SMOKE else "work")))
OUTPUT = Path(os.environ.get("M5_OUTPUT", M5 / ("output_smoke" if SMOKE else "output")))
LOGS = WORK / "logs"
STATE = WORK / "state"
PREP = M5 / "work" / "prep"          # normalised text is config-independent: shared by smoke and full runs

# ------------------------------------------------------------------------------------------------ inputs
# Cleaned records + ground truth: Hugging Face dataset Jyo-K/PopTards_Machine_Code (downloaded to dataset/og).
DATA = Path(os.environ.get("M5_DATA", ROOT / "dataset" / "og"))
EMB_ROOT = Path(os.environ.get("M5_EMB", ROOT / "dataset"))
HF = {   # key -> (dataset repo, local folder, allow_patterns); fetch downloads only what is missing ($HF_TOKEN)
    "data": ("Jyo-K/PopTards_Machine_Code", DATA, None),
    "e5s": ("Jyo-K/PopTards_Smol_Boy", EMB_ROOT / "e5-small", None),
    "e5": ("Jyo-K/PopTard_Qwen_Machine", EMB_ROOT, ["e5_multilingual_large/*"]),
    "bge": ("Jyo-K/PopTards_Big_Boy", EMB_ROOT / "bge3", None),
}
PATHS = {
    "data": {sp: {s: DATA / f"{sp}_p/{sp}_source{s}.parquet" for s in (1, 2, 3)} for sp in ("train", "test")},
    "ground_truth": DATA / "train_p" / "train_ground_truth.tsv",
    "test_tsv": M5 / "work" / "test_tsv",          # test TSVs written from the parquet, for the validator
    "validator": M5 / "utils" / "validate_submission.py",
    "hf_home": Path(os.environ.get("HF_HOME", ROOT.parent / "mtp" / "cache" / "huggingface")),
}

# Pre-computed, unit-normalised embeddings (no new embeddings are computed). Row order == parquet row order;
# "pool" == Source-2 rows then Source-3 rows.
EMB = {
    "e5": {"kind": "shards", "train": EMB_ROOT / "e5_multilingual_large/train",
           "test": EMB_ROOT / "e5_multilingual_large/test"},
    "bge": {"kind": "s1pool", "train": EMB_ROOT / "bge3/emb-bak", "test": EMB_ROOT / "bge3/emb_test_bak"},
    "e5s": {"kind": "matrix", "train": EMB_ROOT / "e5-small/e5-small", "test": EMB_ROOT / "e5-small/e5-small"},
}
EMB_MODELS = ("e5", "bge", "e5s")
# retrievers fused by RRF. "hop" = sibling hop over e5 pool-pool neighbours (README R2)
RETR = ["e5", "bge", "e5s", "lex", "hop"]

# cross-encoders (both fine-tuned here on the ce split, so every downstream score is out-of-sample)
CES = {
    "m": dict(model="microsoft/mdeberta-v3-base", max_len=128, batch=128, lr=3e-5, epochs=1, max_pairs=900_000,
              infer_batch=1024, text="plain", fp32_master=True, mem_train=16_000, mem_infer=8_000),
    "b": dict(model="BAAI/bge-reranker-v2-m3", max_len=160, batch=64, lr=1e-5, epochs=1, max_pairs=900_000,
              infer_batch=1024, text="tagged", fp32_master=True, mem_train=26_000, mem_infer=10_000),
}

LCE_ON = os.environ.get("M5_LCE", "1") == "1"
# inference with bf16 weights + SDPA: MEASURED 2026-09-27 on 20k stack pairs: mDeBERTa 1.01x, bge-reranker 0.75x, max
# |dprob| 0.23 vs the fp32-master/bf16-autocast scores the stage-2 model is trained on -> off by default
INFER_BF16 = os.environ.get("M5_INFER_BF16", "0") == "1"

CFG = dict(
    seed=13, infer_bf16=INFER_BF16,
    # disjoint S1 groups. ce: pairwise cross-encoders; lce: extra lists for the listwise cross-encoder;
    # stack: stage-1/stage-2 training (all selection); val: reporting only
    n_ce=120_000, n_lce=150_000, n_stack=200_000, n_val=100_000,
    test_limit=None,
    query_chunk=25_000,
    # retrieval
    dense_k=100, lex_k=100, rrf_k0=60, k_max=100,
    k_grid=[16, 24, 32, 48, 64, 80, 100], k_recall_frac=0.997,
    dense_qbatch=1024, dense_pchunk=1_000_000,
    lex_max_df=5000, lex_qbatch=500,
    hop_seed=5, hop_nn=10, hop_k=50,         # sibling hop: top-5 fused seeds x 10 pool neighbours each
    tri_top=10, tri_close=0.92,
    rand_pairs=100_000,                      # label-free per-country cosine statistics (F1)
    # label-free legal-form discovery
    legal_min_frac=0.002, legal_boundary_ratio=0.7, legal_max_cooc=0.05,
    legal_bigram_tight=0.8, legal_sample=1_500_000,
    # stage 1 / stage 2
    n_folds=5, lgb_threads=32, n_workers=40,
    s1_neg_keep_rank=32, s1_neg_frac=0.3,
    s2_topn=24, s2_min_prob=0.003,      # reranker K: top-24 by stage-1 p, AND p >= 0.003
    ce_neg_hard=6, ce_neg_rand=2, ce_ckpt_every=1000, ce_max_steps=None, ce_shard_pairs=250_000,
    # listwise cross-encoder
    lce_max_cand=24, lce_cand_tokens=40, lce_q_tokens=64, lce_batch_tokens=24_000, lce_lr=1e-5, lce_epochs=1,
    lce_cls_weight=0.3, lce_hardpos_prob=0.5, lce_passes=1, lce_ckpt_every=1000, lce_max_steps=None,
    lce_shard_q=25_000, lce_max_lists=270_000,
    # decision
    thr_grid=[round(0.05 * i, 2) for i in range(2, 19)], arb_max_iter=30,
    # GPU scheduling (MB of free memory a stage needs before it is started on a card)
    gpu_min_free_mb={"dense": 16_000, "hop": 20_000, "rev": 14_000, "cos": 8_000, "lgb": 6_000, "lce_train": 40_000,
                     "lce_infer": 16_000, **{f"ce_train_{k}": v["mem_train"] for k, v in CES.items()},
                     **{f"ce_infer_{k}": v["mem_infer"] for k, v in CES.items()}},
    max_cpu_stages=3,
    stage_hang_min=60,        # a stage whose log has not grown for this long is killed and retried
    max_tries=5,
    min_free_disk_gb=150,
)

if SMOKE:
    CFG.update(n_ce=3000, n_lce=2000, n_stack=3000, n_val=2000, test_limit=5000, query_chunk=2000,
               ce_max_steps=150, ce_ckpt_every=50, ce_shard_pairs=10_000, n_folds=3, lce_max_steps=100,
               lce_ckpt_every=40, lce_shard_q=1000, rand_pairs=5000, stage_hang_min=30)
