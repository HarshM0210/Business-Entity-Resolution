"""GPU stages: exact dense top-K retrieval per country (e5-large, bge-m3, e5-small), the sibling-hop channel,
and exact cosine + cross-source (triangle) features + label-free per-country cosine statistics."""
import math
import time

import numpy as np
import pandas as pd
import torch

from common import Progress, get_logger, gpu_patient, load_json, save_json, save_npz, save_parquet, shard_path
from config import CFG, EMB_MODELS, WORK
from embeddings import load_emb
from prep import load_prep, load_splits

log = get_logger("dense")


def queries(split):
    sp = load_splits()
    return sp["q_train"] if split == "train" else sp["q_test"]


def n_shards(split):
    return math.ceil(len(queries(split)) / CFG["query_chunk"])


def _to_gpu(arr, dev, rows=None, chunk=500_000):
    n = len(arr) if rows is None else len(rows)
    out = torch.empty((n, arr.shape[1]), dtype=torch.float16, device=dev)
    for i in range(0, n, chunk):
        blk = arr[i:i + chunk] if rows is None else arr[rows[i:i + chunk]]
        out[i:i + len(blk)] = torch.from_numpy(np.ascontiguousarray(blk)).to(dev, non_blocking=False)
    return out


def _topk_search(qv, P, k, qb, pc):
    """Exact top-k of qv @ P.T, chunked over queries and pool."""
    nq = qv.shape[0]
    best_s = torch.full((nq, k), -1e4, dtype=torch.float32, device=qv.device)
    best_i = torch.full((nq, k), -1, dtype=torch.int64, device=qv.device)
    for a in range(0, nq, qb):
        x = qv[a:a + qb]
        bs, bi = best_s[a:a + qb], best_i[a:a + qb]
        for p0 in range(0, P.shape[0], pc):
            s = x @ P[p0:p0 + pc].T
            kk = min(k, s.shape[1])
            ts, ti = torch.topk(s, kk, dim=1)
            del s
            cs = torch.cat([bs, ts.float()], 1)
            ci = torch.cat([bi, ti + p0], 1)
            ts2, pos = torch.topk(cs, k, dim=1)
            bs, bi = ts2, torch.gather(ci, 1, pos)
        best_s[a:a + qb], best_i[a:a + qb] = bs, bi
    return best_s, best_i


def _search_adaptive(qv, P, k):
    qb, pchunk = CFG["dense_qbatch"], CFG["dense_pchunk"]
    t0 = None
    while True:
        try:
            return _topk_search(qv, P, k, qb, pchunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if qb == 64:
                t0 = t0 or time.time()
                if time.time() - t0 > 2 * 3600:
                    raise RuntimeError("dense search: GPU out of memory for 2 h at the minimum batch")
                log.warning("OOM at minimum batch; waiting 60s for the shared GPU")
                time.sleep(60)
            qb, pchunk = max(64, qb // 2), max(50_000, pchunk // 2)
            log.warning(f"OOM -> qbatch={qb} pchunk={pchunk}")


def _countries(split, Q):
    s1c = load_prep(split, "s1", ["country"]).country.values
    pc = load_prep(split, "pool", ["country"]).country.values
    countries = sorted(set(s1c[Q]) | set(pc))          # open set: whatever strings occur
    code = {c: i for i, c in enumerate(countries)}
    return countries, np.array([code[c] for c in pc], dtype=np.int32), np.array([code[c] for c in s1c[Q]], np.int32)


def run_dense(model, split):
    """Country-outer loop: only one country's pool is on the GPU at a time, so this fits next to other users'
    jobs. Each (shard, country) result is checkpointed, then merged into the shard file."""
    Q = queries(split)
    ns = n_shards(split)
    C = CFG["query_chunk"]
    final = lambda i: shard_path("dense", split, i, "npz", model)
    part = lambda i, ci: final(i).with_name(f"part_{i:05d}_{ci}.npz")
    todo = [i for i in range(ns) if not final(i).exists()]
    if not todo:
        return
    dev = torch.device("cuda:0")
    countries, pcode, qcode_all = _countries(split, Q)
    K = CFG["dense_k"]
    pool_emb, qemb = load_emb(model, split, "pool"), load_emb(model, split, "s1")
    for ci, cname in enumerate(countries):
        need = [i for i in todo if (qcode_all[i * C:(i + 1) * C] == ci).any() and not part(i, ci).exists()]
        rows_c = np.where(pcode == ci)[0]
        if not need or len(rows_c) == 0:
            continue
        log.info(f"{model}/{split} [{cname}]: pool {len(rows_c)} rows -> GPU, {len(need)} shards")
        host = pool_emb.take(rows_c)
        P = gpu_patient(lambda: _to_gpu(host, dev), log, f"load {cname} pool")
        del host
        prog = Progress(log, len(need), f"dense {model}/{split} [{cname}] shards")
        for i in need:
            q = Q[i * C:(i + 1) * C]
            rows = np.where(qcode_all[i * C:(i + 1) * C] == ci)[0]
            qv = torch.from_numpy(qemb.take(q[rows])).to(dev)
            s, ix = _search_adaptive(qv, P, min(K, len(rows_c)))
            ix = ix.cpu().numpy()
            valid = ix >= 0
            glob = np.where(valid, rows_c[np.clip(ix, 0, None)], -1).astype(np.int32)
            save_npz(part(i, ci), rows=rows, cand=glob, score=np.where(valid, s.cpu().numpy(), 0).astype(np.float16))
            prog.step()
        del P
        torch.cuda.empty_cache()
    for i in todo:
        q = Q[i * C:(i + 1) * C]
        cand = np.full((len(q), K), -1, dtype=np.int32)
        score = np.zeros((len(q), K), dtype=np.float16)
        parts = [part(i, ci) for ci in np.unique(qcode_all[i * C:(i + 1) * C])]
        for p in parts:
            if p.exists():                      # absent only when that country has no pool records
                z = np.load(p)
                cand[z["rows"], :z["cand"].shape[1]] = z["cand"]
                score[z["rows"], :z["cand"].shape[1]] = z["score"]
        save_npz(final(i), cand=cand, score=score)
        for p in parts:
            p.unlink(missing_ok=True)
    if split == "test":
        _nn_word_check(model, split)


def _nn_word_check(model, split):
    """Label-free sanity (fatal): the top-1 neighbour must share a name word far more often than chance, in
    every country. A misaligned embedding matrix fails this."""
    Q = queries(split)
    s1 = load_prep(split, "s1", ["name_core", "country"])
    pool = load_prep(split, "pool", ["name_core"])
    z = np.load(shard_path("dense", split, 0, "npz", model))
    q = Q[:len(z["cand"])]
    top = z["cand"][:, 0]
    ok = top >= 0
    rnd = np.random.default_rng(0).integers(0, len(pool), ok.sum())
    share = np.array([bool(set(a.split()) & set(b.split())) for a, b in
                      zip(s1.name_core.values[q[ok]], pool.name_core.values[top[ok]])])
    chance = np.array([bool(set(a.split()) & set(b.split())) for a, b in
                       zip(s1.name_core.values[q[ok]], pool.name_core.values[rnd])])
    by_c = pd.Series(share).groupby(s1.country.values[q[ok]]).mean().to_dict()
    rep = {"top1_shares_name_word": float(share.mean()), "random_shares_name_word": float(chance.mean()),
           "by_country": by_c}
    save_json(rep, WORK / "reports" / f"nn_check_{model}_{split}.json")
    log.info(f"{model}/{split} nn check: {rep}")
    floor = max(0.3, 3 * float(chance.mean()))
    bad = {c: v for c, v in by_c.items() if v < floor}
    if bad:
        raise RuntimeError(f"{model}/{split}: top-1 neighbours share a name word below {floor:.2f} in {bad}: "
                           "embedding rows are probably misaligned")


# ----------------------------------------------------------------------------------------------- sibling hop (R2)

def run_hop(split):
    """R2: for each S1, its top-`hop_seed` e5 neighbours ("seeds"); every seed's `hop_nn` nearest pool records
    (same country, e5 cosine) become candidates scored by max_seed cos(seed, cand) - 0.01 * seed_rank.
    Pool-pool neighbours are computed only for records that are seeds."""
    Q = queries(split)
    ns = n_shards(split)
    if split == "test":
        from blocking import chosen_retr
        if "hop" not in chosen_retr():
            log.info("hop channel was dropped by the train-side recall gate; nothing to do for test")
            return
    final = lambda i: shard_path("dense", split, i, "npz", "hop")
    todo = [i for i in range(ns) if not final(i).exists()]
    if not todo:
        return
    S, NN = CFG["hop_seed"], CFG["hop_nn"]
    seeds_all = np.unique(np.concatenate([np.load(shard_path("dense", split, i, "npz", "e5"))["cand"][:, :S].ravel()
                                          for i in range(ns)]))
    seeds_all = seeds_all[seeds_all >= 0]
    nnp = WORK / "hop" / split / "nn.npz"
    if not nnp.exists():
        dev = torch.device("cuda:0")
        countries, pcode, _ = _countries(split, Q)
        pool_emb = load_emb("e5", split, "pool")
        nbr = np.full((len(seeds_all), NN), -1, dtype=np.int32)
        sim = np.zeros((len(seeds_all), NN), dtype=np.float16)
        for ci, cname in enumerate(countries):
            pp = nnp.with_name(f"nn_part_{ci}.npz")
            m = pcode[seeds_all] == ci
            if not m.any():
                continue
            if not pp.exists():
                rows_c = np.where(pcode == ci)[0]
                log.info(f"hop/{split} [{cname}]: {m.sum()} seeds vs pool {len(rows_c)}")
                host = pool_emb.take(rows_c)
                P = gpu_patient(lambda: _to_gpu(host, dev), log, f"hop load {cname}")
                del host
                sr = seeds_all[m]
                out_i = np.full((len(sr), NN), -1, dtype=np.int32)
                out_s = np.zeros((len(sr), NN), dtype=np.float16)
                step = 200_000
                prog = Progress(log, math.ceil(len(sr) / step), f"hop nn {split} [{cname}]")
                for a in range(0, len(sr), step):
                    qv = torch.from_numpy(pool_emb.take(sr[a:a + step])).to(dev)
                    s, ix = _search_adaptive(qv, P, min(NN + 1, len(rows_c)))
                    g = rows_c[ix.clamp(min=0).cpu().numpy()]
                    s = s.cpu().numpy()
                    notself = g != sr[a:a + step, None]            # drop the seed itself
                    for j in range(len(g)):
                        keep = np.where(notself[j])[0][:NN]
                        out_i[a + j, :len(keep)] = g[j, keep]
                        out_s[a + j, :len(keep)] = s[j, keep]
                    prog.step()
                save_npz(pp, nbr=out_i, sim=out_s)
                del P
                torch.cuda.empty_cache()
            z = np.load(pp)
            nbr[m], sim[m] = z["nbr"], z["sim"]
        save_npz(nnp, seeds=seeds_all, nbr=nbr, sim=sim)
        for ci in range(256):
            nnp.with_name(f"nn_part_{ci}.npz").unlink(missing_ok=True)
    z = np.load(nnp)
    seeds, nbr, sim = z["seeds"], z["nbr"], z["sim"].astype(np.float32)
    K = CFG["hop_k"]
    for i in todo:
        e5 = np.load(shard_path("dense", split, i, "npz", "e5"))["cand"][:, :S]
        nq = len(e5)
        pos = np.clip(np.searchsorted(seeds, e5), 0, len(seeds) - 1)
        okseed = (e5 >= 0) & (seeds[pos] == e5)
        cand = nbr[pos]                                               # nq, S, NN
        sc = sim[pos] - 0.01 * np.arange(S)[None, :, None]
        valid = okseed[:, :, None] & (cand >= 0)
        f = pd.DataFrame({"qi": np.repeat(np.arange(nq), S * NN), "c": cand.ravel(), "s": sc.ravel()})[valid.ravel()]
        f = f.sort_values(["qi", "s"], ascending=[True, False]).drop_duplicates(["qi", "c"])
        f["r"] = f.groupby("qi").cumcount()
        f = f[f.r < K]
        oc = np.full((nq, K), -1, dtype=np.int32)
        os_ = np.zeros((nq, K), dtype=np.float16)
        oc[f.qi.values, f.r.values] = f.c.values
        os_[f.qi.values, f.r.values] = f.s.values
        save_npz(final(i), cand=oc, score=os_)
    log.info(f"hop/{split}: {len(todo)} shards written")


# ----------------------------------------------------------------------------------------------- exact cosine + triangle

def feature_queries(split):
    """Queries that need pair features: every train group (ce/lce lists are needed by the cross-encoders'
    hard negatives and the listwise model; stage-1 is out-of-sample for them), all test queries."""
    return queries(split)


def block_rows(split, i):
    """Fused candidates of shard i, capped at the chosen K."""
    K = load_json(WORK / "reports" / "k_choice.json")["K"]
    df = pd.read_parquet(shard_path("block", split, i))
    return df[df.rrf_rank < K].reset_index(drop=True)


def cos_stats(model, split, qemb, P, pool_country, s1_country):
    """Label-free per-country mean/sd of cosine between random same-country (S1, pool) pairs (README F1)."""
    rng = np.random.default_rng(1)
    out = {}
    for c in np.unique(s1_country):
        qi = np.where(s1_country == c)[0]
        pi = np.where(pool_country == c)[0]
        if len(qi) == 0 or len(pi) == 0:
            continue
        a = rng.choice(qi, CFG["rand_pairs"])
        b = rng.choice(pi, CFG["rand_pairs"])
        v = (qemb.take(a).astype(np.float32) * P[b].astype(np.float32)).sum(1)
        out[c] = {"mu": float(v.mean()), "sd": float(v.std() + 1e-6)}
    save_json(out, WORK / "reports" / f"cos_stats_{model}_{split}.json")
    log.info(f"cos stats {model}/{split}: {out}")


def run_cos(model, split):
    ns = n_shards(split)
    todo = [i for i in range(ns) if not shard_path("cos", split, i, "npz", model).exists()]
    stats_p = WORK / "reports" / f"cos_stats_{model}_{split}.json"
    if not todo and stats_p.exists():
        return
    dev = torch.device("cuda:0")
    pool = load_prep(split, "pool", ["src", "country"])
    log.info(f"cos {model}/{split}: loading pool into RAM (candidate rows are gathered per batch)")
    P = load_emb(model, split, "pool").all()
    S = torch.from_numpy(pool.src.values.astype(np.int8)).to(dev)
    qemb = load_emb(model, split, "s1")
    if not stats_p.exists():
        cos_stats(model, split, qemb, P, pool.country.values, load_prep(split, "s1", ["country"]).country.values)
    K = load_json(WORK / "reports" / "k_choice.json")["K"]
    tri = model == "e5"
    prog = Progress(log, len(todo), f"cos {model}/{split} shards")
    for i in todo:
        df = block_rows(split, i)
        n = len(df)
        cos = np.full(n, np.nan, dtype=np.float32)
        tmax = np.full(n, np.nan, dtype=np.float32)
        tclose = np.zeros(n, dtype=np.float32)
        if n:
            uq, start = np.unique(df.q.values, return_index=True)
            qv_all = qemb.take(uq)
            ends = np.append(start[1:], n)
            B = 256
            for b0 in range(0, len(uq), B):
                sl = slice(b0, b0 + B)
                st, en = start[sl], ends[sl]
                idx = np.full((len(st), K), -1, dtype=np.int64)
                rr = np.full((len(st), K), 10 ** 6, dtype=np.int64)
                for j, (a, e) in enumerate(zip(st, en)):
                    idx[j, :e - a] = df.c.values[a:e]
                    rr[j, :e - a] = df.rrf_rank.values[a:e]
                Eh = P[np.clip(idx, 0, None)]                           # b,K,D gathered on CPU

                def compute():
                    I = torch.from_numpy(idx).to(dev)
                    valid = I >= 0
                    E = torch.from_numpy(Eh).to(dev)
                    qv = torch.from_numpy(qv_all[sl]).to(dev)
                    c = torch.einsum("bd,bkd->bk", qv.float(), E.float()).cpu().numpy()
                    if not tri:
                        return c, None, None
                    sim = torch.bmm(E, E.transpose(1, 2)).float()       # b,K,K
                    src = S[I.clamp(min=0)]
                    R = torch.from_numpy(rr).to(dev)
                    eye = torch.eye(K, dtype=torch.bool, device=dev)[None]
                    pair_ok = valid[:, :, None] & valid[:, None, :] & ~eye
                    other = pair_ok & (src[:, :, None] != src[:, None, :]) & (R[:, None, :] < CFG["tri_top"])
                    tm = torch.where(other, sim, torch.full_like(sim, -2)).max(2).values
                    tm = torch.where(tm < -1.5, torch.full_like(tm, float("nan")), tm)
                    tc = ((sim >= CFG["tri_close"]) & pair_ok).sum(2).float()
                    return c, tm.cpu().numpy(), tc.cpu().numpy()

                out_c, tm, tc = gpu_patient(compute, log, "cos batch")
                for j, (a, e) in enumerate(zip(st, en)):
                    cos[a:e] = out_c[j, :e - a]
                    if tri:
                        tmax[a:e] = tm[j, :e - a]
                        tclose[a:e] = tc[j, :e - a]
        arrays = {f"{model}_cos": cos}
        if tri:
            arrays.update(tri_max_cos=tmax, tri_n_close=tclose)
        save_npz(shard_path("cos", split, i, "npz", model), **arrays)
        prog.step()


# ----------------------------------------------------------------------------------------------- reverse search (R3)

REV_K = 10


def run_rev(split):
    """README v2 §3.4 / R3: competitor information over the WHOLE S1 table, label-free. For every record in the
    stage-2 set, its REV_K nearest S1 of the same country (e5-large cosine over all S1 of the split, not only the
    scored ones). Per stage-2 pair: rank of this S1 among them, gap to the best other S1, and how many other S1 are
    within 0.02 of this S1's cosine. Identical computation on train (2.2M S1) and test (1.73M S1)."""
    out = WORK / "rev" / f"{split}.parquet"
    if out.exists():
        return
    s2 = pd.read_parquet(WORK / "s2" / f"{split}.parquet", columns=["q", "c"])
    uc = np.unique(s2.c.values)
    s1c = load_prep(split, "s1", ["country"]).country.values
    pc = load_prep(split, "pool", ["country"]).country.values
    es, ep = load_emb("e5", split, "s1"), load_emb("e5", split, "pool")
    dev = torch.device("cuda:0")
    nbr = np.full((len(uc), REV_K), -1, dtype=np.int32)
    sim = np.full((len(uc), REV_K), -2, dtype=np.float32)
    for cname in sorted(set(pc[uc])):
        part = WORK / "rev" / f"{split}_part_{cname}.npz"
        m = pc[uc] == cname
        if not part.exists():
            rows = np.flatnonzero(s1c == cname)
            host = es.take(rows)
            S = gpu_patient(lambda: _to_gpu(host, dev), log, f"rev load {cname}")
            del host
            idx = np.flatnonzero(m)
            oi = np.full((len(idx), REV_K), -1, dtype=np.int32)
            os_ = np.full((len(idx), REV_K), -2, dtype=np.float32)
            step = 200_000
            prog = Progress(log, math.ceil(len(idx) / step), f"rev {split} [{cname}] {len(idx)} records x {len(rows)} S1")
            for a in range(0, len(idx), step):
                qv = torch.from_numpy(ep.take(uc[idx[a:a + step]])).to(dev)
                s, ix = _search_adaptive(qv, S, min(REV_K, len(rows)))
                ix = ix.cpu().numpy()
                oi[a:a + step, :ix.shape[1]] = np.where(ix >= 0, rows[np.clip(ix, 0, None)], -1)
                os_[a:a + step, :ix.shape[1]] = s.cpu().numpy()
                prog.step()
            save_npz(part, nbr=oi, sim=os_)
            del S
            torch.cuda.empty_cache()
        z = np.load(part)
        nbr[m], sim[m] = z["nbr"], z["sim"]
    pos = np.searchsorted(uc, s2.c.values)
    N, Sm = nbr[pos], sim[pos]                                     # (n_pairs, REV_K)
    own = np.zeros(len(s2), dtype=np.float32)
    for a in range(0, len(s2), 1_000_000):
        sl = slice(a, a + 1_000_000)
        own[sl] = (es.take(s2.q.values[sl]).astype(np.float32) * ep.take(s2.c.values[sl]).astype(np.float32)).sum(1)
    is_self = N == s2.q.values[:, None]
    other = np.where(is_self | (N < 0), -2.0, Sm)
    best_other = other.max(1)
    rep = pd.DataFrame({"q": s2.q.values, "c": s2.c.values})
    rep["rev_rank"] = (other > own[:, None]).sum(1).astype(np.float32)
    rep["rev_in_topk"] = is_self.any(1).astype(np.float32)
    rep["rev_gap_best_other"] = np.where(best_other > -1.5, own - best_other, np.nan).astype(np.float32)
    rep["rev_n_close"] = ((other >= own[:, None] - 0.02) & (other > -1.5)).sum(1).astype(np.float32)
    save_parquet(rep, out)
    for p in (WORK / "rev").glob(f"{split}_part_*.npz"):
        p.unlink()
    log.info(f"rev {split}: {len(rep)} pairs; mean rank {rep.rev_rank.mean():.3f}, "
             f"best-S1 share {(rep.rev_rank == 0).mean():.3f}")
