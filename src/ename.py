"""`ename` channel (README v2 §3.2; MODEL5_FALSE_NEGATIVE_FIXES.md §4): name-only retrieval over EMPTY-ADDRESS pool
records. The data generator deletes the address of 4-5% of true copies but almost never produces an empty-address
distractor, and the dense/lexical channels (which score name + address together) miss such copies when the name is
noisy: 47% of never-retrieved true copies have an empty address. Measured recipe, reproduced exactly:

  documents  pool records with addr_empty, per country        (train: India 122,794 / US 222,089)
  queries    every S1 of that country
  scoring    TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True) fitted on the documents'
             name_full, applied to the S1 name_full; cosine; top k = 5 (recovers 44% of missed empty-address copies)

The candidates are added AFTER the K cut as a guaranteed union and are always passed to stage 2 (they would
otherwise be cut by RRF and by the stage-1 probability floor). Label-free; country is only a grouping key.
France's pool is 3.0% empty-address (43,078 records), so French S1 do get ename candidates."""
import multiprocessing as mp

import numpy as np
import pandas as pd

from common import Progress, get_logger, n_workers, save_json, save_npz, shard_path
from config import CFG, WORK
from prep import load_prep
from retrieve_dense import n_shards, queries

log = get_logger("ename")
K = 5
_G = {}


def _block(args):
    s, e = args
    sim = (_G["Q"][s:e] @ _G["DT"]).toarray()
    kk = min(K, sim.shape[1])
    part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    rows = np.arange(e - s)[:, None]
    sv = sim[rows, part]
    o = np.argsort(-sv, axis=1)
    return s, e, part[rows, o].astype(np.int32), sv[rows, o].astype(np.float32)


def run_ename(split):
    ns = n_shards(split)
    finals = [shard_path("dense", split, i, "npz", "ename") for i in range(ns)]
    if all(p.exists() for p in finals):
        return
    from sklearn.feature_extraction.text import TfidfVectorizer
    Q = queries(split)
    C = CFG["query_chunk"]
    s1 = load_prep(split, "s1", ["country", "name_full"])
    pool = load_prep(split, "pool", ["country", "name_full", "addr_empty"])
    cand = np.full((len(Q), K), -1, dtype=np.int32)
    score = np.zeros((len(Q), K), dtype=np.float16)
    qc = s1.country.values[Q]
    rep = {}
    for country in sorted(set(qc)):
        qi = np.flatnonzero(qc == country)                           # positions in Q
        docs = np.flatnonzero(pool.addr_empty.values.astype(bool) & (pool.country.values == country))
        rep[country] = {"queries": int(len(qi)), "empty_address_docs": int(len(docs))}
        if len(docs) == 0:
            log.info(f"{split} [{country}]: no empty-address pool records -> no ename candidates")
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32)
        D = vec.fit_transform(pool.name_full.values[docs])
        Qm = vec.transform(s1.name_full.values[Q[qi]])
        _G["Q"], _G["DT"] = Qm, D.T.tocsc()
        tasks = [(a, min(a + 2048, len(qi))) for a in range(0, len(qi), 2048)]
        prog = Progress(log, len(tasks), f"ename {split} [{country}] {len(qi)} queries x {len(docs)} docs")
        with mp.get_context("fork").Pool(max(4, min(32, n_workers()))) as p:
            for a, e, idx, val in p.imap_unordered(_block, tasks):
                ok = val > 0                                         # no shared 3-gram at all -> no candidate
                cand[qi[a:e]] = np.where(ok, docs[idx], -1)
                score[qi[a:e]] = np.where(ok, val, 0)
                prog.step()
        _G.clear()
        n = int((cand[qi] >= 0).sum())
        rep[country].update(pairs=n, pairs_per_s1=round(n / len(qi), 3))
        log.info(f"ename {split} [{country}]: {rep[country]}")
    for i in range(ns):
        save_npz(finals[i], cand=cand[i * C:(i + 1) * C], score=score[i * C:(i + 1) * C])
    if split == "train":                                            # measured recovery on stack+val (report)
        sp = load_splits_safe()
        if sp is not None:
            rep["recall"] = _recall(sp, Q, cand)
    save_json(rep, WORK / "reports" / f"ename_{split}.json")
    log.info(f"ename {split}: {rep}")


def load_splits_safe():
    try:
        from prep import load_splits
        return load_splits()
    except Exception:
        return None


def _recall(sp, Q, cand):
    """True copies found by ename, and how many of those the fused top-K missed (stack+val)."""
    import json
    from decide import in_sorted
    Kc = json.loads((WORK / "reports" / "k_choice.json").read_text())["K"]
    mult = np.int64(1 << 32)
    gt = np.sort(sp["gt_q"] * mult + sp["gt_c"])
    ev = np.concatenate([sp["g_stack"], sp["g_val"]])
    pos = np.searchsorted(Q, ev)
    q = np.repeat(ev, K)
    c = cand[pos].ravel()
    m = c >= 0
    hit = in_sorted(q[m].astype(np.int64) * mult + c[m], gt)
    blk = []
    for i in range(n_shards("train")):
        d = pd.read_parquet(shard_path("block", "train", i), columns=["q", "c", "rrf_rank", "y"])
        blk.append(d[(d.y == 1) & (d.rrf_rank < Kc)])
    b = pd.concat(blk)
    bkey = np.sort(b.q.values.astype(np.int64) * mult + b.c.values)
    hk = q[m][hit].astype(np.int64) * mult + c[m][hit]
    new = ~in_sorted(hk, bkey)
    return {"true_found": int(hit.sum()), "true_not_in_fused_topK": int(new.sum()),
            "true_total_stack_val": int(sp["n_true"][ev].sum()), "precision": float(hit.mean()) if len(hit) else 0.0}
