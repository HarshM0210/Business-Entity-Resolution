"""F1 feature-set selection (leave-one-country-out), stage 1 (LightGBM over all fused candidates), stage-2 set,
stage 2 (LightGBM over cross-encoder + listwise scores, or the listwise model alone; calibration; decision rule —
everything selected on the stack split), test inference + submission files, and reports."""
import os
import pickle
import subprocess
import sys
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import drain, get_logger, load_json, n_workers, save_json, save_parquet, shard_path, spawn_gpu_workers
from config import CES, CFG, LCE_ON, OUTPUT, PATHS, WORK
from decide import crossfit_isotonic, decide, f05_selftest, fit_isotonic, fold_of, macro_f05
from features import feature_columns, raw_scale_features
from prep import load_prep, load_splits
from retrieve_dense import n_shards

log = get_logger("stack")
MODELS = WORK / "models"
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1)


def train_lgb(X, y, q, path, seed, rounds=3000):
    """Early stopping on an inner holdout of whole S1 groups; model saved atomically for resume."""
    if path.exists():
        return lgb.Booster(model_file=str(path))
    inner = fold_of(q, 10, salt=1) == 0
    p = dict(PARAMS, seed=seed, num_threads=CFG["lgb_threads"])
    dtr = lgb.Dataset(X[~inner], y[~inner], free_raw_data=True)
    dva = lgb.Dataset(X[inner], y[inner], reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    m.save_model(str(tmp), num_iteration=m.best_iteration)
    tmp.replace(path)
    log.info(f"saved {path.name} best_iter={m.best_iteration}")
    return lgb.Booster(model_file=str(path))


def load_feats(split, qset=None, columns=None):
    parts = []
    for i in range(n_shards(split)):
        d = pd.read_parquet(shard_path("feat", split, i), columns=columns)
        if qset is not None:
            d = d[np.isin(d.q.values, qset)]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _subsample(tr, rng):
    keep = (tr.y.values == 1) | (tr.rrf_rank.values < CFG["s1_neg_keep_rank"]) | (rng.random(len(tr)) < CFG["s1_neg_frac"])
    return tr[keep]


def _pred_stats(sel, qs, country):
    n = pd.Series(0, index=qs).add(sel.groupby("q").size(), fill_value=0).loc[qs]
    c = pd.Series(country[qs], index=qs)
    out = {}
    for cc, g in n.groupby(c):
        out[cc] = {"n_s1": int(len(g)), "empty_rate": round(float((g == 0).mean()), 4),
                   "mean_pred": round(float(g.mean()), 3),
                   "dist": {str(k): round(float(v), 4) for k, v in g.clip(upper=6).value_counts(normalize=True).sort_index().items()}}
    return out


# ----------------------------------------------------------------------------------------------- F1 selection

def _loco(df, feats, sp, country, pool, tag, tr_c, variants_out):
    """Train on one country's stack (half A), calibrate on half B, evaluate on both countries' val."""
    rng = np.random.default_rng(CFG["seed"])
    st = df[np.isin(df.q.values, sp["g_stack"]) & (country[df.q.values] == tr_c)].reset_index(drop=True)
    half = fold_of(st.q.values, 2, salt=2) == 0
    tr = _subsample(st[half], rng)
    import hashlib
    h = hashlib.sha1(",".join(feats).encode()).hexdigest()[:8]      # never reuse a model trained on other columns
    m = train_lgb(tr[feats].values.astype(np.float32), tr.y.values, tr.q.values, MODELS / f"loco_{tag}_{tr_c}_{h}.txt", 300)
    cal = st[~half]
    iso = fit_isotonic(m.predict(cal[feats].values.astype(np.float32)), cal.y.values)
    res = {}
    for ev_c in sorted(set(country[sp["g_val"]])):
        qv = sp["g_val"][country[sp["g_val"]] == ev_c]
        v = df[np.isin(df.q.values, qv)]
        p = iso.predict(m.predict(v[feats].values.astype(np.float32)))
        sel = decide(v[["q", "c"]].assign(p=p), "expf05", None, CFG["arb_max_iter"], pool)
        res[ev_c], _ = macro_f05(sel, qv, sp["n_true"], sp["gt_q"], sp["gt_c"])
    variants_out[f"{tag}: train {tr_c}"] = res
    log.info(f"LOCO {tag} train {tr_c}: {res}")
    return res


def run_f1_select():
    """README F1: keep the country-scale-dependent raw features only if dropping them does not improve the
    out-of-country F0.5 (US<->India) at an in-country cost <= 0.05 pt."""
    out = MODELS / "feature_set.json"
    if out.exists():
        return
    sp = load_splits()
    country = load_prep("train", "s1", ["country"]).country.values
    df = load_feats("train", np.concatenate([sp["g_stack"], sp["g_val"]]))
    full = feature_columns(df)
    inv = [c for c in full if c not in set(raw_scale_features(full))]
    rep = {}
    cs = sorted(set(country[sp["g_stack"]]))
    with Pool(n_workers()) as pool:
        score = {}
        for tag, feats in (("full", full), ("invariant", inv)):
            ins, outs = [], []
            for tr_c in cs:
                r = _loco(df, feats, sp, country, pool, tag, tr_c, rep)
                ins.append(r[tr_c])
                outs += [v for k, v in r.items() if k != tr_c]
            score[tag] = {"in_country": float(np.mean(ins)), "out_of_country": float(np.mean(outs))}
    use_inv = (len(cs) > 1 and score["invariant"]["out_of_country"] > score["full"]["out_of_country"]
               and score["full"]["in_country"] - score["invariant"]["in_country"] <= 0.0005)
    choice = "invariant" if use_inv else "full"
    save_json({"choice": choice, "features": inv if use_inv else full, "dropped": sorted(set(full) - set(inv)) if use_inv else [],
               "scores": score, "detail": rep}, out)
    save_json({"choice": choice, "scores": score, "detail": rep}, WORK / "reports" / "f1_select.json")
    log.info(f"F1 feature set: {choice} {score}")


# ----------------------------------------------------------------------------------------------- stage 1

def run_stage1_train():
    sp = load_splits()
    feats = load_json(MODELS / "feature_set.json")["features"]
    save_json(feats, MODELS / "stage1_features.json")
    df = load_feats("train", columns=["q", "c", "y", "rrf_rank"] + [f for f in feats if f != "rrf_rank"])
    stack = df[np.isin(df.q.values, sp["g_stack"])].reset_index(drop=True)
    rest = df[~np.isin(df.q.values, sp["g_stack"])].reset_index(drop=True)      # val, ce, lce: out-of-sample
    del df
    F = CFG["n_folds"]
    fold = fold_of(stack.q.values, F)
    oof = np.zeros(len(stack))
    rp = np.zeros(len(rest))
    rng = np.random.default_rng(CFG["seed"])
    Xr = rest[feats].values.astype(np.float32)
    for k in range(F):
        tr = _subsample(stack[fold != k], rng)
        m = train_lgb(tr[feats].values.astype(np.float32), tr.y.values, tr.q.values, MODELS / f"s1_fold{k}.txt", k)
        oof[fold == k] = m.predict(stack.loc[fold == k, feats].values.astype(np.float32))
        rp += m.predict(Xr) / F
    out = pd.concat([stack[["q", "c", "y"]].assign(p1=oof), rest[["q", "c", "y"]].assign(p1=rp)], ignore_index=True)
    save_parquet(out, WORK / "stage1" / "train.parquet")
    from sklearn.metrics import roc_auc_score
    vm = np.isin(rest.q.values, sp["g_val"])
    log.info(f"stage1 AUC stack-oof={roc_auc_score(stack.y, oof):.5f} val={roc_auc_score(rest.y[vm], rp[vm]):.5f}")


def run_stage1_test_worker(threads):
    """CPU scoring of stage-1 test shards (LightGBM, `threads` threads); shards claimed dynamically, so several
    workers (and machines sharing the filesystem) drain one queue. Measured: ~0.9 min per 480k-row shard with 24
    threads, ~6x faster than the exact GPU tree evaluator (gpu_lgb.py) on 5 x ~2k deep trees."""
    paths = [shard_path("stage1", "test", i) for i in range(n_shards("test"))]
    if all(p.exists() for p in paths):
        return
    feats = load_json(MODELS / "stage1_features.json")
    boosters = [lgb.Booster(model_file=str(MODELS / f"s1_fold{k}.txt")) for k in range(CFG["n_folds"])]

    def job(i):
        d = pd.read_parquet(shard_path("feat", "test", i), columns=["q", "c"] + feats)
        X = d[feats].values.astype(np.float32)
        p = sum(b.predict(X, num_threads=threads) for b in boosters) / len(boosters)
        save_parquet(d[["q", "c"]].assign(p1=p.astype(np.float32)), paths[i])
    drain(paths, job, log, f"stage1 test (CPU, {threads} threads, pid {os.getpid()})")


def run_stage1_test():
    import time
    nproc = int(max(1, min(4, (os.cpu_count() or 8) // 32)))
    threads = max(8, n_workers() // nproc)
    src = os.path.dirname(os.path.abspath(__file__))
    procs = [subprocess.Popen([sys.executable, "-u", os.path.join(src, "stages.py"), "stage1_test_worker", str(threads)],
                              cwd=src) for _ in range(nproc)]
    log.info(f"stage1 test: {nproc} CPU workers x {threads} threads")
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"stage1 test workers failed: {codes}")
    paths = [shard_path("stage1", "test", i) for i in range(n_shards("test"))]
    while True:                     # shards may still be held by helpers that joined the queue
        missing = [p for p in paths if not p.exists()]
        if not missing:
            return
        if not any(os.path.exists(str(p) + ".claim") for p in missing):
            raise RuntimeError(f"missing stage1 test shards: {[str(p) for p in missing[:5]]}")
        time.sleep(30)


# ----------------------------------------------------------------------------------------------- ename pairs (v2)

def run_stage1x(split):
    """Stage-1 scores for the pairs that only the ename channel found (featx rows with ename_only=1). Stack rows are
    scored out-of-fold (the fold model that did not see that S1); every other group gets the fold average."""
    out = WORK / "stage1x" / f"{split}.parquet"
    if out.exists():
        return
    feats = load_json(MODELS / "stage1_features.json")
    boosters = [lgb.Booster(model_file=str(MODELS / f"s1_fold{k}.txt")) for k in range(CFG["n_folds"])]
    parts = []
    for i in range(n_shards(split)):
        d = pd.read_parquet(shard_path("featx", split, i))
        parts.append(d[d.ename_only == 1][["q", "c", "y"] + feats])
    d = pd.concat(parts, ignore_index=True)
    X = d[feats].values.astype(np.float32)
    P = np.stack([b.predict(X, num_threads=CFG["lgb_threads"]) for b in boosters]) if len(d) else np.zeros((1, 0))
    p = P.mean(0)
    if split == "train" and len(d):
        sp = load_splits()
        st = np.isin(d.q.values, sp["g_stack"])
        f = fold_of(d.q.values, CFG["n_folds"])
        p[st] = P[f[st], np.flatnonzero(st)]
    save_parquet(d[["q", "c", "y"]].assign(p1=p.astype(np.float32)), out)
    log.info(f"stage1x {split}: {len(d)} ename-only pairs scored (true {int((d.y == 1).sum())})")


def _ename_keys(split, qset=None):
    parts = []
    for i in range(n_shards(split)):
        d = pd.read_parquet(shard_path("featx", split, i), columns=["q", "c", "from_ename"])
        d = d[d.from_ename == 1]
        if qset is not None:
            d = d[np.isin(d.q.values, qset)]
        parts.append(d[["q", "c"]])
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------------------------- stage-2 set

def _s2(df, ename):
    """Top `s2_topn` by stage-1 p with p >= s2_min_prob (the reranker K and its probability floor), plus every ename
    candidate regardless of rank or probability (README v2 §3.2 trap 3). Ranks/gaps are over the whole union."""
    df = df.sort_values(["q", "p1"], ascending=[True, False], kind="stable")
    g = df.groupby("q")
    df["p1_rank"] = g.cumcount().astype(np.float32)
    df["p1_gap"] = (g.p1.transform("max") - df.p1).astype(np.float32)
    key = lambda d: d.q.values.astype(np.int64) * (1 << 32) + d.c.values
    from decide import in_sorted
    forced = in_sorted(key(df), np.sort(key(ename)))
    keep = ((df.p1_rank < CFG["s2_topn"]) & (df.p1 >= CFG["s2_min_prob"])).values | forced
    return df[keep].reset_index(drop=True)


def _cache_old_ce_scores(split):
    """Before the stage-2 set is rebuilt, keep the cross-encoder scores already computed for the old set, keyed by
    (q, c), so only new pairs are scored again."""
    from cross_encoder import cache_scores
    old = WORK / "s2" / f"{split}.parquet"
    if old.exists():
        for k in CES:
            cache_scores(k, split, pd.read_parquet(old, columns=["q", "c"]))


def run_s2set(split):
    _cache_old_ce_scores(split)
    if split == "train":
        sp = load_splits()
        allq = np.concatenate([sp["g_stack"], sp["g_val"]])
        s1 = pd.read_parquet(WORK / "stage1" / "train.parquet")
        s1 = pd.concat([s1[np.isin(s1.q.values, allq)], pd.read_parquet(WORK / "stage1x" / "train.parquet")
                        .pipe(lambda d: d[np.isin(d.q.values, allq)])], ignore_index=True)
        s2 = _s2(s1, _ename_keys("train", allq))
        en = _ename_keys("train", allq)
        key = lambda d: d.q.values.astype(np.int64) * (1 << 32) + d.c.values
        from decide import in_sorted
        is_en = in_sorted(key(s2), np.sort(key(en)))
        rep = {"rows": len(s2), "true_in_union": int(s1.y.sum()), "true_in_s2": int(s2.y.sum()),
               "true_total": int(sp["n_true"][allq].sum()), "ename_rows": int(is_en.sum()),
               "ename_true": int(s2.y.values[is_en].sum()),
               "ename_true_only_via_ename": int(pd.read_parquet(WORK / "stage1x" / "train.parquet")
                                                .pipe(lambda d: d[np.isin(d.q.values, allq)]).y.sum())}
        rep["s2_recall_vs_union"] = rep["true_in_s2"] / max(rep["true_in_union"], 1)
        rep["s2_recall_vs_total"] = rep["true_in_s2"] / max(rep["true_total"], 1)
        save_json(rep, WORK / "reports" / "s2set_train.json")
        log.info(f"stage-2 set: {rep}")
    else:
        s1 = pd.concat([pd.read_parquet(shard_path("stage1", "test", i)) for i in range(n_shards("test"))] +
                       [pd.read_parquet(WORK / "stage1x" / "test.parquet").drop(columns="y")], ignore_index=True)
        s2 = _s2(s1, _ename_keys("test"))
    save_parquet(s2, WORK / "s2" / f"{split}.parquet")
