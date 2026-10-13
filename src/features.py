"""CPU stage: country-agnostic pair features on rule-free text views, plus country-invariant versions of the
scale-dependent ones (README F1)."""
import math
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

import text_utils as tu
from common import Progress, get_logger, load_json, n_workers, save_parquet, shard_path, claim, release
from config import EMB_MODELS, WORK
from prep import load_prep
from retrieve_dense import block_rows, n_shards

log = get_logger("feat")
PCOLS = ["name_full", "name_core", "legal", "name_ns", "nonlatin", "addr_norm", "street", "street_ns", "loc",
         "house", "nums", "postal", "alnum", "addr_empty", "src", "country"]
PY_FEATS = ["n_soft_ab", "n_soft_ba", "n_jacc", "n_extra_a", "n_extra_b", "n_contain", "n_acro", "n_first_eq",
            "legal_state", "st_soft_ab", "st_soft_ba", "loc_cov_a", "loc_cov_b", "a_jacc",
            "house_state", "house_diff", "num_jacc", "num_state", "postal_state", "alnum_state",
            "n_tok_a", "n_tok_b", "a_tok_a", "a_tok_b"]
FUZZ = [("n_ratio", "name_core", fuzz.ratio), ("n_jw", "name_core", JaroWinkler.normalized_similarity),
        ("n_lev", "name_core", Levenshtein.normalized_similarity), ("n_tsort", "name_core", fuzz.token_sort_ratio),
        ("n_tset", "name_core", fuzz.token_set_ratio), ("n_partial", "name_core", fuzz.partial_ratio),
        ("n_full_ratio", "name_full", fuzz.ratio), ("ns_ratio", "name_ns", fuzz.ratio),
        ("ns_partial", "name_ns", fuzz.partial_ratio), ("a_ratio", "addr_norm", fuzz.ratio),
        ("a_tset", "addr_norm", fuzz.token_set_ratio), ("a_partial", "addr_norm", fuzz.partial_ratio),
        ("st_ratio", "street", fuzz.ratio), ("st_ns_ratio", "street_ns", fuzz.ratio)]

# features whose scale depends on a country's vocabulary / pool / language (README F1). The F1 selection stage
# decides under leave-one-country-out whether to drop them; their invariant replacements are always present.
def raw_scale_features(cols):
    raw = {f"{r}_{x}" for r in ("e5", "bge", "e5s", "lex", "hop") for x in ("s", "gap")}
    raw |= {"e5_cos", "bge_cos", "e5s_cos", "q_best_e5", "q_best_bge", "q_best_e5s", "tri_max_cos",
            "n_tok_a", "n_tok_b", "a_tok_a", "a_tok_b"}
    return sorted(c for c in cols if c in raw)


def _jacc(a, b):
    if not a and not b:
        return math.nan
    return len(a & b) / len(a | b)


def _py_chunk(args):
    A, B = args                     # dicts of column -> list for the query side and candidate side
    out = np.empty((len(A["name_core"]), len(PY_FEATS)), dtype=np.float32)
    for i in range(len(out)):
        na, nb = A["name_core"][i].split(), B["name_core"][i].split()
        sa, sb = set(na), set(nb)
        fa, fb = A["name_full"][i].split(), B["name_full"][i].split()
        n_ab, n_ba = tu.soft_cov(na, nb), tu.soft_cov(nb, na)
        extra_a = sum(1 for x in na if x not in sb and not any(tu.tok_sim(x, y) > 0 for y in nb))
        extra_b = sum(1 for x in nb if x not in sa and not any(tu.tok_sim(x, y) > 0 for y in na))
        contain = 1.0 if tu.contains_seq(na, fb) else (2.0 if tu.contains_seq(nb, fa) else 0.0)
        acro = 1.0 if (tu.acronym_match(na, nb) or tu.acronym_match(nb, na)) else 0.0
        first = 1.0 if na and nb and na[0] == nb[0] else 0.0
        leg = tu.legal_state(A["legal"][i], B["legal"][i])
        sta, stb = A["street"][i].split(), B["street"][i].split()
        st_ab, st_ba = tu.soft_cov(sta, stb), tu.soft_cov(stb, sta)
        ta, tb = A["addr_norm"][i].split(), B["addr_norm"][i].split()
        la = [t for seg in A["loc"][i].split("|") for t in seg.split() if not t.isdigit()]
        lb = [t for seg in B["loc"][i].split("|") for t in seg.split() if not t.isdigit()]
        loc_a = tu.soft_cov(la, tb)     # does the query's city/region appear anywhere in the candidate
        loc_b = tu.soft_cov(lb, ta)
        hs, hd = tu.house_compare(A["house"][i], B["house"][i])
        ma, mb = set(A["nums"][i].split()), set(B["nums"][i].split())
        pa, pb = set(A["postal"][i].split()), set(B["postal"][i].split())
        xa, xb = set(A["alnum"][i].split()), set(B["alnum"][i].split())
        out[i] = (n_ab, n_ba, _jacc(sa, sb), extra_a, extra_b, contain, acro, first, leg, st_ab, st_ba,
                  loc_a, loc_b, _jacc(set(ta), set(tb)), hs, hd, _jacc(ma, mb), tu.set_state(ma, mb),
                  tu.set_state(pa, pb), tu.set_state(xa, xb), len(na), len(nb), len(ta), len(tb))
    return out


def _gather(tab, idx, cols):
    return {c: tab[c][idx].tolist() for c in cols}


def pair_features(pool, s1tab, pooltab, df, nthreads):
    q, c = df.q.values, df.c.values
    feats = {}
    for name, col, scorer in FUZZ:
        a, b = s1tab[col][q].tolist(), pooltab[col][c].tolist()   # lists: ~10x faster than object arrays
        v = process.cpdist(a, b, scorer=scorer, workers=nthreads).astype(np.float32)
        empty = (pd.Series(a).str.len().values == 0) | (pd.Series(b).str.len().values == 0)
        v[empty] = np.nan
        feats[name] = v
    str_cols = ["name_core", "name_full", "legal", "street", "addr_norm", "loc", "house", "nums", "postal", "alnum"]
    step = 10_000
    args = [(_gather(s1tab, q[i:i + step], str_cols), _gather(pooltab, c[i:i + step], str_cols))
            for i in range(0, len(df), step)]
    py = np.concatenate(list(pool.imap(_py_chunk, args))) if args else np.empty((0, len(PY_FEATS)), np.float32)
    for j, name in enumerate(PY_FEATS):
        feats[name] = py[:, j]
    feats["nonlatin_a"] = s1tab["nonlatin"][q].astype(np.float32)
    feats["nonlatin_b"] = pooltab["nonlatin"][c].astype(np.float32)
    feats["addr_empty_a"] = s1tab["addr_empty"][q].astype(np.float32)
    feats["addr_empty_b"] = pooltab["addr_empty"][c].astype(np.float32)
    feats["cand_src"] = pooltab["src"][c].astype(np.float32)
    return pd.DataFrame(feats)


def query_context(df, retr, stats, s1_country):
    """Within-query ranks / gaps / z-scores, and per-country standardised cosines (label-free statistics)."""
    g = df.groupby("q")
    out = pd.DataFrame(index=df.index)
    out["q_ncand"] = g.c.transform("size").astype(np.float32)
    ctry = s1_country[df.q.values]
    for m in EMB_MODELS:
        col = f"{m}_cos"
        best = g[col].transform("max")
        mu_q, sd_q = g[col].transform("mean"), g[col].transform("std").fillna(0)
        out[f"{m}_cos_gap"] = (best - df[col]).astype(np.float32)
        out[f"q_best_{m}"] = best.astype(np.float32)
        out[f"{m}_cos_rank"] = g[col].rank(ascending=False, method="first").astype(np.float32)
        out[f"{m}_cos_qz"] = ((df[col] - mu_q) / (sd_q + 1e-3)).astype(np.float32)
        st = stats[m]
        mu = np.array([st.get(c, {"mu": np.nan})["mu"] for c in ctry], dtype=np.float32)
        sd = np.array([st.get(c, {"sd": np.nan})["sd"] for c in ctry], dtype=np.float32)
        out[f"{m}_cos_cz"] = ((df[col].values - mu) / sd).astype(np.float32)
        out[f"q_best_{m}_cz"] = ((best.values - mu) / sd).astype(np.float32)
    if "lex_s" in df:
        mx = g.lex_s.transform("max")
        out["lex_rel"] = (df.lex_s / mx.where(mx > 0)).astype(np.float32)
    out["q_best_n_tset"] = g.n_tset.transform("max").astype(np.float32)
    out["n_tset_gap"] = (out.q_best_n_tset - df.n_tset).astype(np.float32)
    out["tok_ratio_n"] = (df.n_tok_a / df.n_tok_b.clip(lower=1)).astype(np.float32)
    out["tok_ratio_a"] = (df.a_tok_a / df.a_tok_b.clip(lower=1)).astype(np.float32)
    return out


def run_features(split):
    ns = n_shards(split)
    paths = [shard_path("feat", split, i) for i in range(ns)]
    if all(p.exists() for p in paths):
        return
    from config import RETR  # noqa: F401  (the kept retrievers are already in the block shards)
    stats = {m: load_json(WORK / "reports" / f"cos_stats_{m}_{split}.json") for m in EMB_MODELS}
    nw = n_workers()
    with Pool(nw) as pool:              # fork before big tables are loaded
        s1 = load_prep(split, "s1", [c for c in PCOLS if c != "src"])
        pl = load_prep(split, "pool", PCOLS)
        s1tab = {c: s1[c].values for c in s1.columns}
        pooltab = {c: pl[c].values for c in pl.columns}
        s1_country = s1.country.values
        del s1, pl
        prog = Progress(log, sum(not p.exists() for p in paths), f"features {split} shards")
        # several processes may drain this queue (python features.py <split> starts a helper); keep going until
        # every shard exists: shards claimed by a live helper are waited for, stale claims are re-taken
        while True:
            for i, p in enumerate(paths):
                if not claim(p):
                    continue
                try:
                    df = block_rows(split, i)
                    for m in EMB_MODELS:
                        z = np.load(shard_path("cos", split, i, "npz", m))
                        for k in z.files:
                            df[k] = z[k]
                    f = pair_features(pool, s1tab, pooltab, df, nw)
                    df = pd.concat([df, f], axis=1)
                    df = pd.concat([df, query_context(df, None, stats, s1_country)], axis=1)
                    save_parquet(df, p)
                finally:
                    release(p)
                prog.step()
            missing = [p for p in paths if not p.exists()]
            if not missing:
                break
            log.info(f"features {split}: waiting for {len(missing)} shard(s) held by other workers")
            time.sleep(30)


def feature_columns(df):
    return [c for c in df.columns if c not in ("q", "c", "y")]


# ----------------------------------------------------------------------------------------------- union with ename (v2)

CONTEXT_COLS = (["q_ncand", "lex_rel", "q_best_n_tset", "n_tset_gap", "tok_ratio_n", "tok_ratio_a"] +
                [f"{x}{m}{y}" for m in EMB_MODELS for x, y in (("", "_cos_gap"), ("q_best_", ""), ("", "_cos_rank"),
                                                               ("", "_cos_qz"), ("", "_cos_cz"), ("q_best_", "_cz"))])
ENAME_K = 5


def run_featx(split):
    """README v2 §2.2/§3.2: the fused top-K rows plus the ename candidates as a guaranteed union. New (q, c) pairs get
    exact e5/bge/e5s cosines and the full pair features; within-query context (ranks, gaps, z-scores, counts) is
    recomputed over the union; flags from_ename / ename_only and ename_s / ename_r are added. Feature shards from
    run_features are read, never modified."""
    from config import CFG, RETR  # noqa: F401
    from decide import in_sorted
    from embeddings import load_emb
    from prep import load_splits
    from blocking import chosen_retr
    from retrieve_dense import queries
    ns = n_shards(split)
    paths = [shard_path("featx", split, i) for i in range(ns)]
    if all(p.exists() for p in paths):
        return
    retr = chosen_retr()
    Kc = load_json(WORK / "reports" / "k_choice.json")["K"]
    Q = queries(split)
    C = CFG["query_chunk"]
    stats = {m: load_json(WORK / "reports" / f"cos_stats_{m}_{split}.json") for m in EMB_MODELS}
    gt = None
    if split == "train":
        sp = load_splits()
        gt = np.sort(sp["gt_q"] * np.int64(1 << 32) + sp["gt_c"])
    nw = n_workers()
    with Pool(nw) as pool:
        s1 = load_prep(split, "s1", [c for c in PCOLS if c != "src"])
        pl = load_prep(split, "pool", PCOLS)
        s1tab = {c: s1[c].values for c in s1.columns}
        pooltab = {c: pl[c].values for c in pl.columns}
        s1_country = s1.country.values
        del s1, pl
        embs = {m: (load_emb(m, split, "s1"), load_emb(m, split, "pool")) for m in EMB_MODELS}
        prog = Progress(log, sum(not p.exists() for p in paths), f"featx {split} shards")
        stat = {"new_pairs": 0, "ename_pairs": 0, "rows": 0}
        while True:
            for i, p in enumerate(paths):
                if not claim(p):
                    continue
                try:
                    base = pd.read_parquet(shard_path("feat", split, i))
                    z = np.load(shard_path("dense", split, i, "npz", "ename"))
                    q = np.repeat(Q[i * C:(i + 1) * C], z["cand"].shape[1])
                    en = pd.DataFrame({"q": q.astype(np.int32), "c": z["cand"].ravel(),
                                       "ename_s": z["score"].astype(np.float32).ravel(),
                                       "ename_r": np.tile(np.arange(z["cand"].shape[1]), len(z["cand"])).astype(np.float32)})
                    en = en[en.c >= 0].reset_index(drop=True)
                    key = lambda d: d.q.values.astype(np.int64) * (1 << 32) + d.c.values
                    bkey = np.sort(key(base))
                    new = en[~in_sorted(key(en), bkey)][["q", "c"]].reset_index(drop=True)
                    if len(new):
                        new["rrf"] = np.float32(0)
                        new["rrf_rank"] = np.int16(Kc)              # sentinel: outside the fused top-K
                        new["support"] = np.int8(0)
                        new["y"] = (in_sorted(key(new), gt).astype(np.int8) if gt is not None else np.int8(-1))
                        for r in retr:
                            new[f"{r}_r"] = np.int16(CFG["dense_k"])
                            new[f"{r}_s"] = np.float32(np.nan)
                            new[f"{r}_gap"] = np.float32(np.nan)
                        for m, (es, ep) in embs.items():
                            a = es.take(new.q.values).astype(np.float32)
                            b = ep.take(new.c.values).astype(np.float32)
                            new[f"{m}_cos"] = (a * b).sum(1)
                        new["tri_max_cos"] = np.float32(np.nan)
                        new["tri_n_close"] = np.float32(0)
                        f = pair_features(pool, s1tab, pooltab, new, nw)
                        new = pd.concat([new, f], axis=1)
                    df = pd.concat([base.drop(columns=[c for c in CONTEXT_COLS if c in base.columns]),
                                    new[[c for c in base.columns if c not in CONTEXT_COLS]] if len(new) else None],
                                   ignore_index=True)
                    df["ename_only"] = np.r_[np.zeros(len(base), np.int8), np.ones(len(new), np.int8)]
                    df = df.merge(en, on=["q", "c"], how="left")
                    df["from_ename"] = df.ename_s.notna().astype(np.int8)
                    df["ename_r"] = df.ename_r.fillna(ENAME_K).astype(np.float32)
                    df = df.sort_values(["q", "ename_only", "rrf_rank"], kind="stable").reset_index(drop=True)
                    df = pd.concat([df, query_context(df, None, stats, s1_country)], axis=1)
                    save_parquet(df, p)
                    stat["new_pairs"] += len(new); stat["ename_pairs"] += len(en); stat["rows"] += len(df)
                finally:
                    release(p)
                prog.step()
            missing = [p for p in paths if not p.exists()]
            if not missing:
                break
            log.info(f"featx {split}: waiting for {len(missing)} shard(s) held by other workers")
            time.sleep(30)
    log.info(f"featx {split} (this process): {stat}")


if __name__ == "__main__":       # helper: python features.py <feat|featx> <split> — joins that stage's shard queue
    (run_featx if sys.argv[1] == "featx" else run_features)(sys.argv[-1])
