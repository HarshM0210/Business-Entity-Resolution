#!/usr/bin/env python3
"""Experiment 5: learned decision policy on top of model-5 ce_s (one-step RL / contextual bandit, LightGBM).

State  = one S1: its ce_s-calibrated candidate list (top-8 p, stage-1 p1, both cross-encoders) + the current
         expected-F0.5 choice k0 (decide.best_k).
Action = delta in {-2..+3}: take the top-(k0+delta) candidates instead of the top-k0.
Reward = true F0.5(k0+delta) - F0.5(k0) for that S1 (full-information: every action's reward is known on OOF).
Policy = argmax_delta predicted reward, applied only if it exceeds tau (tau tuned on stack OOF for macro F0.5).
Trained cross-fitted on stack, checked on val (never used for fitting or tau), then applied to test using the saved
ce_s test scores. No embeddings / CEs / stage-2 retraining. Writes only inside this policy/ folder.
"""
import json
import pickle
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
M5 = HERE.parent.parent
sys.path.insert(0, str(M5 / "src"))

from config import CFG, PATHS, WORK
from cross_encoder import load_ce_scores
from decide import choose, crossfit_isotonic, decide, fit_isotonic, fold_of, macro_f05
from prep import load_prep, load_splits

REP = HERE / "reports"
CFG_NAME = "ce_s"
TOP = 8
DELTAS = [-2, -1, 1, 2, 3]
N_POOL = 24
THREADS = 24
TAUS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=THREADS, seed=7)
N_ROUNDS = 400
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- state / reward

def sort_groups(d):
    d = d.sort_values(["q", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    d["r"] = d.groupby("q", sort=False).cumcount().values
    return d


def current_k(d, pool):
    """k0 = the expected-F0.5 choice per S1 before arbitration (decide.choose, family expf05, thr 0)."""
    sel = choose(d[["q", "c", "p"]], "expf05", 0.0, pool)
    return sel.groupby("q").size()


def states(d, k0):
    """One row per S1: top-8 p/p1/ce_m/ce_b, list stats, k0 and the values around the k0 boundary."""
    qs, inv = np.unique(d.q.values, return_inverse=True)
    n = len(qs)
    top = d.r.values < TOP
    f = {"n_cand": np.bincount(inv, minlength=n).astype(np.float32),
         "sum_p": np.bincount(inv, weights=d.p.values, minlength=n).astype(np.float32)}
    for col in ("p", "p1", "ce_m", "ce_b"):
        m = np.full((n, TOP), np.nan, dtype=np.float32)
        m[inv[top], d.r.values[top]] = d[col].values[top]
        for j in range(TOP):
            f[f"{col}_{j}"] = m[:, j]
    k0v = pd.Series(k0).reindex(qs).fillna(0).values.astype(np.int64)
    f["k0"] = k0v.astype(np.float32)
    pm = np.column_stack([f[f"p_{j}"] for j in range(TOP)])
    idx = np.arange(n)
    f["p_at_k0"] = np.where(k0v >= 1, pm[idx, np.clip(k0v - 1, 0, TOP - 1)], np.nan)
    f["p_next"] = np.where(k0v < TOP, pm[idx, np.clip(k0v, 0, TOP - 1)], np.nan)
    f["gap_k0"] = f["p_at_k0"] - f["p_next"]
    f["exp_count"] = f["sum_p"]
    f["exp_minus_k0"] = f["sum_p"] - k0v
    return pd.DataFrame(f, index=qs), k0v


def f05_of_prefix(d, n_true):
    """F[q, k] for k = 0..n_cand: true F0.5 of predicting the top-k (by p). Returns dict q -> array."""
    y = d.y.values
    q = d.q.values
    tp = pd.Series(y).groupby(q).cumsum().values
    k = d.r.values + 1
    nt = n_true[q]
    fk = 1.25 * tp / (0.25 * nt + k)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(d)]
    out = {}
    for a, e in zip(starts, ends):
        qq = q[a]
        out[qq] = np.r_[1.0 if n_true[qq] == 0 else 0.0, fk[a:e]]
    return out


def action_rows(S, k0v, fmap=None):
    """Expand states to (S1, delta) rows for every valid delta; reward if fmap (labels) is given."""
    ncand = S.n_cand.values.astype(np.int64)
    rows, deltas, rew = [], [], []
    for dlt in DELTAS:
        kk = k0v + dlt
        ok = (kk >= 0) & (kk <= ncand)
        i = np.flatnonzero(ok)
        rows.append(i)
        deltas.append(np.full(len(i), dlt, dtype=np.float32))
        if fmap is not None:
            qs = S.index.values[i]
            rew.append(np.array([fmap[q][k + dlt] - fmap[q][k] for q, k in zip(qs, k0v[i])], dtype=np.float32))
    rows = np.concatenate(rows)
    X = S.iloc[rows].copy()
    X["delta"] = np.concatenate(deltas)
    X["k_new"] = X.k0 + X.delta
    return X, rows, (np.concatenate(rew) if fmap is not None else None)


def choose_delta(S, rows, X, pred, tau):
    """Best predicted delta per S1, applied only if its predicted gain > tau."""
    df = pd.DataFrame({"i": rows, "delta": X.delta.values, "g": pred})
    best = df.sort_values(["i", "g"], ascending=[True, False]).drop_duplicates("i")
    best = best[best.g > tau]
    out = np.zeros(len(S), dtype=np.int64)
    out[best.i.values] = best.delta.values.astype(np.int64)
    return out


def select(d, qs, k):
    """Top-k prefix per S1 (k aligned with qs), then one owner per record (highest p keeps it)."""
    kk = pd.Series(k, index=qs).reindex(d.q.values).fillna(0).values
    sel = d[d.r.values < kk]
    return sel.sort_values(["c", "p", "q"], ascending=[True, False, True], kind="stable").drop_duplicates("c")


# ----------------------------------------------------------------------------------------------- main

def train_side(pool):
    sp = load_splits()
    s2 = pd.read_parquet(WORK / "s2" / "train.parquet", columns=["q", "c", "y", "p1"])
    for k in ("m", "b"):
        s2 = s2.merge(load_ce_scores(k, "train"), on=["q", "c"], how="left")
    stack = s2[np.isin(s2.q.values, sp["g_stack"])].reset_index(drop=True)
    val = s2[np.isin(s2.q.values, sp["g_val"])].reset_index(drop=True)
    prelim = pickle.loads((WORK / "models" / "s2_prelim.pkl").read_bytes())[CFG_NAME]
    s2raw = pd.read_parquet(WORK / "s2" / "train.parquet", columns=["q", "c"])
    # prelim oof/vp are row-aligned with the stack/val filters of s2/train.parquet; the CE merge above is a left
    # merge (keeps row order), checked here
    assert (stack[["q", "c"]].values == s2raw[np.isin(s2raw.q.values, sp["g_stack"])][["q", "c"]].values).all()
    assert (val[["q", "c"]].values == s2raw[np.isin(s2raw.q.values, sp["g_val"])][["q", "c"]].values).all()
    stack["p"] = crossfit_isotonic(prelim["oof"], stack.y.values, stack.q.values, CFG["n_folds"])
    iso = fit_isotonic(prelim["oof"], stack.y.values)
    val["p"] = iso.predict(prelim["vp"])
    both = pd.concat([stack, val], ignore_index=True)
    return sp, both


def main():
    REP.mkdir(parents=True, exist_ok=True)
    out = {}
    with Pool(N_POOL) as pool:
        sp, both = train_side(pool)
        log(f"train pairs {len(both)}")
        l0 = decide(both[["q", "c", "p"]], "expf05", 0.0, CFG["arb_max_iter"], pool)
        f_l0_st = macro_f05(l0, sp["g_stack"], sp["n_true"], sp["gt_q"], sp["gt_c"])[0]
        f_l0_va = macro_f05(l0, sp["g_val"], sp["n_true"], sp["gt_q"], sp["gt_c"])[0]
        log(f"L0 decide(): stack {f_l0_st:.6f} val {f_l0_va:.6f}")
        d = sort_groups(both)
        k0 = current_k(d, pool)
    S, k0v = states(d, k0)
    fmap = f05_of_prefix(d, sp["n_true"])
    X, rows, R = action_rows(S, k0v, fmap)
    feats = list(X.columns)
    is_stack = np.isin(S.index.values, sp["g_stack"])
    log(f"states {len(S)} ({is_stack.sum()} stack), action rows {len(X)}, mean reward {R.mean():.5f}, "
        f"positive-reward rows {(R > 0).mean():.4f}")

    # cross-fitted on stack; val predicted by the all-stack model
    rs = is_stack[rows]
    fold = fold_of(S.index.values[rows], CFG["n_folds"], salt=11)
    pred = np.zeros(len(X))
    for k in range(CFG["n_folds"]):
        tr = rs & (fold != k)
        te = rs & (fold == k)
        m = lgb.train(LGB_PARAMS, lgb.Dataset(X.values[tr], R[tr], feature_name=feats), N_ROUNDS)
        pred[te] = m.predict(X.values[te])
        log(f"fold {k} done")
    final = lgb.train(LGB_PARAMS, lgb.Dataset(X.values[rs], R[rs], feature_name=feats), N_ROUNDS)
    final.save_model(str(REP / "policy_lgb.txt"))
    pred[~rs] = final.predict(X.values[~rs])
    imp = pd.Series(final.feature_importance("gain"), index=feats).sort_values(ascending=False)
    log(f"top features: {imp.head(12).round(0).to_dict()}")

    base = select(d, S.index.values, k0v)
    f_b_st = macro_f05(base, sp["g_stack"], sp["n_true"], sp["gt_q"], sp["gt_c"])[0]
    f_b_va = macro_f05(base, sp["g_val"], sp["n_true"], sp["gt_q"], sp["gt_c"])[0]
    log(f"baseline delta=0 (drop arbitration): stack {f_b_st:.6f} val {f_b_va:.6f}")
    grid = {}
    for tau in TAUS:
        dl = choose_delta(S, rows, X, pred, tau)
        sel = select(d, S.index.values, k0v + dl)
        st = macro_f05(sel, sp["g_stack"], sp["n_true"], sp["gt_q"], sp["gt_c"])[0]
        va = macro_f05(sel, sp["g_val"], sp["n_true"], sp["gt_q"], sp["gt_c"])[0]
        grid[tau] = {"stack": round(st, 6), "val": round(va, 6), "n_changed": int((dl != 0).sum()),
                     "n_up": int((dl > 0).sum()), "n_down": int((dl < 0).sum())}
        log(f"tau {tau:<6} stack {st:.6f} ({st - f_b_st:+.6f})  val {va:.6f} ({va - f_b_va:+.6f})  {grid[tau]}")
    tau = max(grid, key=lambda t: grid[t]["stack"])
    gain_val = grid[tau]["val"] - f_b_va
    out.update({"L0_decide": {"stack": f_l0_st, "val": f_l0_va},
                "baseline_drop_arb": {"stack": f_b_st, "val": f_b_va}, "tau_grid": grid, "tau": tau,
                "val_gain_vs_baseline": round(gain_val, 6), "val_gain_vs_L0": round(grid[tau]["val"] - f_l0_va, 6),
                "top_features": imp.head(20).round(1).to_dict()})
    log(f"chosen tau {tau} (on stack): val {grid[tau]['val']:.6f}, gain vs baseline {gain_val:+.6f}, "
        f"vs L0 {grid[tau]['val'] - f_l0_va:+.6f}")
    (REP / "policy_results.json").write_text(json.dumps(out, indent=2, default=str))

    # ------------------------------------------------------------------------------------------- test
    log("test: loading saved ce_s scores")
    ts = pd.read_parquet(WORK / "stage2" / "test_scores.parquet", columns=["q", "c", "p", "p1", "ce_m", "ce_b"])
    with Pool(N_POOL) as pool:
        tl0 = decide(ts[["q", "c", "p"]], "expf05", 0.0, CFG["arb_max_iter"], pool)
        log(f"test L0 pairs {len(tl0)} (ce_s submission: 5,776,799)")
        td = sort_groups(ts.assign(y=0))
        tk0 = current_k(td, pool)
    TS, tk0v = states(td, tk0)
    TX, trows, _ = action_rows(TS, tk0v)
    tpred = final.predict(TX[feats].values)
    tdl = choose_delta(TS, trows, TX, tpred, tau)
    s1 = load_prep("test", "s1", ["id", "country"])
    ctry = s1.country.values[TS.index.values]
    out["test"] = {"L0_pairs": int(len(tl0)), "tau": tau}
    pool_ids = load_prep("test", "pool", ["id"]).id.values
    q_all = load_splits()["q_test"]
    for name, dl in (("all", tdl), ("no_france", np.where(ctry == "france", 0, tdl))):
        sel = select(td, TS.index.values, tk0v + dl)
        odir = HERE / f"output_{name}"
        odir.mkdir(exist_ok=True)
        lst = sel.sort_values(["q", "p"], ascending=[True, False])
        lst = lst.assign(cid=pool_ids[lst.c.values]).groupby("q").cid.apply(lambda x: ",".join(dict.fromkeys(x)))
        match = pd.DataFrame({"source1_entity_id": s1.id.values[q_all],
                              "matched_entity_ids": pd.Series(q_all).map(lst).fillna("").values})
        match.to_csv(odir / "matching_results.tsv", sep="\t", index=False)
        shutil.copy(M5 / "output_ce_s" / "candidate_pairs.tsv", odir / "candidate_pairs.tsv")
        r = subprocess.run([sys.executable, str(PATHS["validator"]), "--matching", str(odir / "matching_results.tsv"),
                            "--candidate", str(odir / "candidate_pairs.tsv"), "--test-dir", str(PATHS["test_tsv"]),
                            "--check-ids"], capture_output=True, text=True)
        by = pd.DataFrame({"c": ctry, "d": dl}).groupby("c").d.agg(
            up=lambda x: int((x > 0).sum()), down=lambda x: int((x < 0).sum())).to_dict("index")
        out["test"][name] = {"n_pairs": int(len(sel)), "changed_by_country": by, "validator_rc": r.returncode,
                             "validator_tail": (r.stdout + r.stderr)[-600:]}
        log(f"test[{name}]: {len(sel)} pairs, changes {by}, validator rc={r.returncode}")
    (REP / "policy_results.json").write_text(json.dumps(out, indent=2, default=str))
    log("DONE")


if __name__ == "__main__":
    main()
