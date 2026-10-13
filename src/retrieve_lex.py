"""CPU stage: exact lexical retrieval (IDF-weighted cosine over rare name tokens, name prefixes, street tokens,
house-number+street composites, postcodes, alphanumeric unit tokens), per country."""
import os
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

from common import Progress, get_logger, n_workers, save_npz, shard_path
from config import CFG
from prep import load_prep
from retrieve_dense import n_shards, queries

log = get_logger("lex")
COLS = ["name_core", "name_ns", "street", "loc", "house", "postal", "alnum", "country"]


def lex_tokens(core, ns, street, loc, house, postal, alnum):
    out = set()
    for t in core.split():
        if len(t) >= 2 and not t.isdigit():
            out.add("n:" + t)
    if len(ns) >= 5:
        out.add("p:" + ns[:5])
    st = [t for t in street.split() if len(t) >= 3 and not t.isdigit()]
    for t in st:
        out.add("s:" + t)
    for seg in loc.split("|"):
        for t in seg.split():
            if len(t) >= 3 and not t.isdigit():
                out.add("l:" + t)
    if house:
        hb = house.split("|")[0]
        for t in st:
            if len(t) >= 4:
                out.add(f"h:{hb}:{t[:4]}")
    for p in postal.split():
        out.add("z:" + p)
    for a in alnum.split():
        out.add("a:" + a)
    return out


def _hash_chunk(rows):
    lens, hs = [], []
    for r in zip(*rows):
        t = lex_tokens(*r)
        lens.append(len(t))
        hs.extend(hash(x) for x in t)
    return np.asarray(lens, dtype=np.int64), np.asarray(hs, dtype=np.int64)


def _hash_side(pool, df):
    step = 200_000
    args = [tuple(df[c].values[i:i + step] for c in COLS[:-1]) for i in range(0, len(df), step)]
    lens, hs = zip(*pool.map(_hash_chunk, args))
    lens = np.concatenate(lens)
    return np.concatenate([[0], np.cumsum(lens)]), np.concatenate(hs)


G = {}


def _search_shard(i):
    Q, q_ptr, q_h, qcode, split = G["Q"], G["q_ptr"], G["q_h"], G["qcode"], G["split"]
    K, qb = CFG["lex_k"], CFG["lex_qbatch"]
    lo, hi = i * CFG["query_chunk"], min((i + 1) * CFG["query_chunk"], len(Q))
    cand = np.full((hi - lo, K), -1, dtype=np.int32)
    score = np.zeros((hi - lo, K), dtype=np.float16)
    for ci, M in G["mats"].items():
        rows = np.where(qcode[lo:hi] == ci)[0]
        for b0 in range(0, len(rows), qb):
            r = rows[b0:b0 + qb]
            ind, dat, ptr = [], [], [0]
            for j in r:
                h = q_h[q_ptr[lo + j]:q_ptr[lo + j + 1]]
                pos = np.searchsorted(M["vocab"], h)
                pos = np.clip(pos, 0, len(M["vocab"]) - 1)
                ok = M["vocab"][pos] == h
                pos = pos[ok]
                pos = pos[M["keep"][pos]]
                w = M["idf"][pos]
                nrm = np.sqrt((w * w).sum())
                ind.append(pos)
                dat.append(w / nrm if nrm > 0 else w)
                ptr.append(ptr[-1] + len(pos))
            Qm = sp.csr_matrix((np.concatenate(dat), np.concatenate(ind), np.asarray(ptr)),
                               shape=(len(r), len(M["vocab"])))
            S = (Qm @ M["PT"]).tocsr()
            for k, j in enumerate(r):
                a, e = S.indptr[k], S.indptr[k + 1]
                if a == e:
                    continue
                d, c = S.data[a:e], S.indices[a:e]
                if len(d) > K:
                    top = np.argpartition(-d, K)[:K]
                    d, c = d[top], c[top]
                o = np.argsort(-d)
                cand[j, :len(o)] = M["glob"][c[o]]
                score[j, :len(o)] = d[o]
    save_npz(shard_path("lex", split, i, "npz"), cand=cand, score=score)
    return i


def run_lex(split):
    assert os.environ.get("PYTHONHASHSEED") == "0", "lexical hashing needs PYTHONHASHSEED=0"
    ns = n_shards(split)
    todo = [i for i in range(ns) if not shard_path("lex", split, i, "npz").exists()]
    if not todo:
        return
    Q = queries(split)
    with Pool(n_workers()) as pool:
        pdf = load_prep(split, "pool", COLS)
        p_ptr, p_h = _hash_side(pool, pdf)
        s1 = load_prep(split, "s1", COLS).iloc[Q].reset_index(drop=True)
        q_ptr, q_h = _hash_side(pool, s1)
    countries = sorted(set(pdf.country.values) | set(s1.country.values))
    code = {c: i for i, c in enumerate(countries)}
    pcode = pdf.country.map(code).values
    qcode = s1.country.map(code).values
    del pdf, s1
    mats = {}
    doc_of_tok = np.repeat(np.arange(len(p_ptr) - 1), np.diff(p_ptr))
    for ci, c in enumerate(countries):
        docs = np.where(pcode == ci)[0]
        if len(docs) == 0:
            continue
        m = pcode[doc_of_tok] == ci
        h, d = p_h[m], doc_of_tok[m]
        vocab, inv = np.unique(h, return_inverse=True)
        df = np.bincount(inv, minlength=len(vocab))
        keep = df <= CFG["lex_max_df"]
        idf = np.log(len(docs) / df).astype(np.float32)
        local = np.searchsorted(docs, d)
        sel = keep[inv]
        w = idf[inv[sel]]
        P = sp.csr_matrix((w, (local[sel], inv[sel])), shape=(len(docs), len(vocab)))
        nrm = np.sqrt(np.asarray(P.multiply(P).sum(1)).ravel())
        nrm[nrm == 0] = 1
        P = sp.diags(1 / nrm) @ P
        mats[ci] = {"vocab": vocab, "keep": keep, "idf": idf, "PT": P.T.tocsr(), "glob": docs.astype(np.int32)}
        log.info(f"lex index [{c}]: docs={len(docs)} vocab={len(vocab)} kept={int(keep.sum())}")
    G.update(Q=Q, q_ptr=q_ptr, q_h=q_h, qcode=qcode, split=split, mats=mats)
    prog = Progress(log, len(todo), f"lex {split} shards")
    with Pool(n_workers()) as pool:          # fork after the index is built: workers share it read-only
        for _ in pool.imap_unordered(_search_shard, todo):
            prog.step()
