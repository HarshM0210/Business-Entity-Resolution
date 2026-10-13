"""CPU stage: reciprocal-rank fusion of the retrievers. RRF only decides which candidates survive the cap; every
kept retriever's own score/rank/gap stays a separate feature.

Train side, pass 1: for every retriever subset (base = e5+bge+e5s+lex, plus each combination of the optional
channels, i.e. hop), the RRF rank of every true pair of a *stack* entity is recorded. The probe keeps an
optional channel only if it adds >= `gate` recall at the chosen K (README §4 "keep if"), and picks K on stack
entities only (model-4 audit M1). Pass 2 writes the fused shards with the chosen retrievers."""
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import Progress, get_logger, load_json, n_workers, save_json, save_parquet, shard_path
from config import CFG, RETR, WORK
from decide import in_sorted
from prep import load_prep, load_splits
from retrieve_dense import n_shards, queries

log = get_logger("block")
BASE = ["e5", "bge", "e5s", "lex"]          # all three embeddings are always fused; only "hop" is gated
OPTIONAL = [r for r in RETR if r not in BASE]
GATE = 0.001
G = {}


def chosen_retr():
    return load_json(WORK / "reports" / "k_choice.json")["retr"]


def _load(r, split, i):
    p = shard_path("lex", split, i, "npz") if r == "lex" else shard_path("dense", split, i, "npz", r)
    z = np.load(p)
    return z["cand"], z["score"].astype(np.float32)


def _union(split, i, retr):
    frames = []
    for r in retr:
        cand, score = _load(r, split, i)
        nq, K = cand.shape
        f = pd.DataFrame({"qi": np.repeat(np.arange(nq), K).astype(np.int32), "c": cand.ravel(),
                          f"{r}_r": np.tile(np.arange(K), nq).astype(np.float32), f"{r}_s": score.ravel()})
        frames.append(f[f.c >= 0])
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["qi", "c"], how="outer")
    return df


def canon(retr):
    """Canonical subset name (RETR order)."""
    return "+".join(r for r in RETR if r in retr)


def _rrf(df, retr):
    k0 = CFG["rrf_k0"]
    return sum((1.0 / (k0 + 1 + df[f"{r}_r"])).fillna(0) for r in retr)


def _probe_shard(i):
    """Ranks of stack true pairs under every retriever subset (train only)."""
    split, Q = "train", G["Q"]
    lo = i * CFG["query_chunk"]
    df = _union(split, i, RETR)
    df["q"] = Q[lo + df.qi.values]
    df = df[in_sorted(df.q.values.astype(np.int64), G["stack"])].reset_index(drop=True)
    key = df.q.values.astype(np.int64) * G["mult"] + df.c.values
    df["y"] = in_sorted(key, G["gt_key"])
    out = {"q": df.q.values[df.y.values]}
    for sub in G["subsets"]:
        name = canon(sub)
        m = np.zeros(len(df), dtype=bool)
        for r in sub:
            m |= df[f"{r}_r"].notna().values
        s = np.where(m, _rrf(df, sub).values, -1.0)
        t = pd.DataFrame({"qi": df.qi.values, "s": s}).sort_values(["qi", "s"], ascending=[True, False], kind="stable")
        rank = np.empty(len(df), dtype=np.int32)
        rank[t.index.values] = t.groupby("qi").cumcount().values
        rank[~m] = 10 ** 6
        out[name] = rank[df.y.values]
    return out


def _block_shard(i):
    split, Q, retr = G["split"], G["Q"], G["retr"]
    lo = i * CFG["query_chunk"]
    df = _union(split, i, retr)
    df["rrf"] = _rrf(df, retr)
    df["support"] = sum(df[f"{r}_r"].notna().astype(np.int8) for r in retr).astype(np.int8)
    df = df.sort_values(["qi", "rrf"], ascending=[True, False], kind="stable")
    df["rrf_rank"] = df.groupby("qi").cumcount().astype(np.int16)
    df = df[df.rrf_rank < CFG["k_max"]]
    for r in retr:
        df[f"{r}_gap"] = df.groupby("qi")[f"{r}_s"].transform("max") - df[f"{r}_s"]
        df[f"{r}_r"] = df[f"{r}_r"].fillna(CFG["dense_k"]).astype(np.int16)
    df["q"] = Q[lo + df.qi.values].astype(np.int32)
    df["c"] = df.c.astype(np.int32)
    if split == "train":
        key = df.q.values.astype(np.int64) * G["mult"] + df.c.values
        df["y"] = in_sorted(key, G["gt_key"]).astype(np.int8)
    else:
        df["y"] = np.int8(-1)
    cols = ["q", "c", "rrf", "rrf_rank", "support", "y"] + [f"{r}_{x}" for r in retr for x in ("r", "s", "gap")]
    for c in cols:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    save_parquet(df[cols].reset_index(drop=True), shard_path("block", split, i))
    return i


def _gt(sp):
    mult = np.int64(1 << 32)
    G["mult"] = mult
    G["gt_key"] = np.sort(sp["gt_q"] * mult + sp["gt_c"])


def run_block(split):
    ns = n_shards(split)
    Q = queries(split)
    sp = load_splits()
    G.update(split=split, Q=Q)
    if split == "train":
        _gt(sp)
        if not (WORK / "reports" / "k_choice.json").exists():
            recall_probe(ns, sp)
    G["retr"] = chosen_retr()
    todo = [i for i in range(ns) if not shard_path("block", split, i).exists()]
    if todo:
        prog = Progress(log, len(todo), f"block {split} shards")
        with Pool(min(n_workers(), 16)) as pool:
            for _ in pool.imap_unordered(_block_shard, todo):
                prog.step()


def recall_probe(ns, sp):
    G["stack"] = np.sort(sp["g_stack"]).astype(np.int64)
    G["subsets"] = [BASE + list(c) for k in range(len(OPTIONAL) + 1) for c in combinations(OPTIONAL, k)]
    with Pool(min(n_workers(), 16)) as pool:
        parts = list(pool.imap_unordered(_probe_shard, range(ns)))
    tot = int(sp["n_true"][sp["g_stack"]].sum())
    country = load_prep("train", "s1", ["country"]).country.values
    rep = {"true_pairs_stack": tot, "gate": GATE, "subsets": {}}
    for sub in G["subsets"]:
        name = canon(sub)
        rk = np.concatenate([p[name] for p in parts])
        rep["subsets"][name] = {int(K): float((rk < K).sum() / tot) for K in CFG["k_grid"]}
    qpos = np.concatenate([p["q"] for p in parts])
    # greedy gate at K=64 (model-4's cap): drop an optional channel whose removal costs < GATE recall
    ref_k = 64 if 64 in CFG["k_grid"] else CFG["k_grid"][-1]
    keep = list(RETR)
    for r in OPTIONAL:
        without = [x for x in keep if x != r]
        a = rep["subsets"][canon(keep)][ref_k]
        b = rep["subsets"][canon(without)][ref_k]
        rep[f"gain_{r}@{ref_k}"] = a - b
        if a - b < GATE:
            keep = without
    name = canon(keep)
    rec = rep["subsets"][name]
    K = next(k for k in CFG["k_grid"] if rec[k] >= CFG["k_recall_frac"] * rec[CFG["k_max"]])
    rk = np.concatenate([p[name] for p in parts])
    rep["chosen"] = {"retr": keep, "K": int(K), "recall_at_K": rec[K],
                     "recall_at_K_by_country": pd.Series(rk < K).groupby(country[qpos]).sum().div(
                         pd.Series(sp["n_true"][sp["g_stack"]]).groupby(country[sp["g_stack"]]).sum()).round(5).to_dict()}
    save_json(rep, WORK / "reports" / "blocking_recall.json")
    save_json({"K": int(K), "retr": keep}, WORK / "reports" / "k_choice.json")
    log.info(f"blocking probe (stack entities): {rep}")
