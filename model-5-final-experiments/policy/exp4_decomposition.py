#!/usr/bin/env python3
"""Experiment 4: oracle decomposition of the stage-2 F0.5 gap (CPU only, read-only on model-5/work).

exp2's oracle (every y==1 candidate) is a perfect-scorer ceiling, identical for every config. This splits the gap:
  L0 current           decide() exactly as the pipeline (joint stack+val, crossfit isotonic)
  L1 oracle-k          current p-ranking, per-S1 prefix k chosen with labels, one-owner arbitration by p
  L2 oracle-k + truth  same, a contested record goes to its true owner
  L3 perfect scorer    all y==1 in the stage-2 set
L1-L0 = max gain for any k/abstain policy on the current scores; L3-L2 = ranking (scorer) errors.
Writes only to model-5-final-experiments/policy/reports/.
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
M5 = HERE.parent.parent
sys.path.insert(0, str(M5 / "src"))

from config import CFG, WORK
from decide import MULT, crossfit_isotonic, decide, expected_f05, fit_isotonic, macro_f05
from prep import load_prep, load_splits

OUT = HERE / "reports"
CONFIGS = ["ce_s", "ce"]
N_POOL = 24
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def score(sel, q_eval, sp, country):
    f, per = macro_f05(sel[["q", "c"]], q_eval, sp["n_true"], sp["gt_q"], sp["gt_c"])
    by = pd.Series(per.f.values).groupby(country[per.q.values]).mean().round(6).to_dict()
    return round(f, 6), by


def oracle_k(d, n_true):
    """d(q, c, p, y) -> rows of the label-optimal prefix of the p-ranking per q (k=0 allowed; ties -> smaller k)."""
    d = d.sort_values(["q", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    q = d.q.values
    g = d.groupby("q", sort=False)
    k = g.cumcount().values + 1
    tp = g.y.cumsum().values
    nt = n_true[q]
    f = 1.25 * tp / (0.25 * nt + k)
    d["f"] = f
    d["k"] = k
    best = d.loc[d.groupby("q", sort=False).f.idxmax(), ["q", "f", "k"]]
    nt_q = n_true[best.q.values]
    k0 = np.where(nt_q == 0, 1.0, 0.0)                 # score of predicting nothing
    kbest = np.where(best.f.values > k0, best.k.values, 0)
    kmap = pd.Series(kbest, index=best.q.values)
    return d[d.k.values <= kmap.reindex(q).values][["q", "c", "p", "y"]], kmap


def arbitrate(sel, by_truth):
    """One owner per record: keep the claim with y==1 (by_truth) or the highest p; losers just drop the pair."""
    cols = ["c", "y", "p", "q"] if by_truth else ["c", "p", "q"]
    asc = [True, False, False, True] if by_truth else [True, False, True]
    return sel.sort_values(cols, ascending=asc, kind="stable").drop_duplicates("c")


def _exp_chunk(args):
    return [expected_f05(p[:k], p[k:]) for p, k in args]


def expected_by_country(both, sel, q_eval, country, pool):
    d = both[np.isin(both.q.values, q_eval)].sort_values(["q", "p"], ascending=[True, False], kind="stable")
    ks = sel.groupby("q").size()
    qv = d.q.values
    starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]])
    ends = np.r_[starts[1:], len(d)]
    pv = d.p.values
    qs = qv[starts]
    items = [(pv[a:e], int(ks.get(q, 0))) for q, a, e in zip(qs, starts, ends)]
    step = 20_000
    vals = [v for part in pool.map(_exp_chunk, [items[i:i + step] for i in range(0, len(items), step)]) for v in part]
    e = pd.Series(1.0, index=q_eval)                     # S1 with no candidates: E[F]=P(no true match) ~ counted as 1
    e.loc[qs] = vals
    return pd.Series(e.values).groupby(country[e.index.values]).mean().round(6).to_dict()


def run_config(cfg, prelim, stack, val, sp, country, gates, pool):
    log(f"=== {cfg} ===")
    oof, vp = prelim[cfg]["oof"], prelim[cfg]["vp"]
    assert len(oof) == len(stack) and len(vp) == len(val), (cfg, len(oof), len(stack), len(vp), len(val))
    fam, thr = gates[cfg]["family"], gates[cfg]["thr"]
    cal = crossfit_isotonic(oof, stack.y.values, stack.q.values, CFG["n_folds"])
    vcal = fit_isotonic(oof, stack.y.values).predict(vp)
    both = pd.concat([stack[["q", "c", "y"]].assign(p=cal), val[["q", "c", "y"]].assign(p=vcal)], ignore_index=True)

    sel0 = decide(both[["q", "c", "p"]], fam, thr, CFG["arb_max_iter"], pool)
    L = {"L0_current": sel0}
    s0, _ = score(sel0, sp["g_stack"], sp, country)
    log(f"L0 stack {s0:.6f} (gates {gates[cfg]['stack']:.6f})")
    if abs(s0 - gates[cfg]["stack"]) > 1e-5:
        raise SystemExit(f"ALIGNMENT GUARD FAILED for {cfg}: L0 {s0} vs gates {gates[cfg]['stack']}")

    ok, kmap = oracle_k(both, sp["n_true"])
    L["L1_oraclek_arb_p"] = arbitrate(ok, by_truth=False)
    L["L2_oraclek_arb_truth"] = arbitrate(ok, by_truth=True)
    L["L3_perfect_scorer"] = both[both.y == 1]

    res = {"family": fam, "thr": thr, "levels": {}}
    for name, sel in L.items():
        st, st_by = score(sel, sp["g_stack"], sp, country)
        va, va_by = score(sel, sp["g_val"], sp, country)
        res["levels"][name] = {"stack": st, "val": va, "stack_by_country": st_by, "val_by_country": va_by,
                               "n_pairs": int(len(sel))}
        log(f"{name:24s} stack {st:.6f}  val {va:.6f}  {st_by}")

    lv = res["levels"]
    res["gaps"] = {sp_: {"k_policy_L1-L0": round(lv["L1_oraclek_arb_p"][sp_] - lv["L0_current"][sp_], 6),
                         "arbitration_L2-L1": round(lv["L2_oraclek_arb_truth"][sp_] - lv["L1_oraclek_arb_p"][sp_], 6),
                         "ranking_L3-L2": round(lv["L3_perfect_scorer"][sp_] - lv["L2_oraclek_arb_truth"][sp_], 6)}
                   for sp_ in ("stack", "val")}
    res["gaps_by_country_stack"] = {
        c: {"k_policy": round(lv["L1_oraclek_arb_p"]["stack_by_country"][c] - lv["L0_current"]["stack_by_country"][c], 6),
            "ranking": round(lv["L3_perfect_scorer"]["stack_by_country"][c]
                             - lv["L2_oraclek_arb_truth"]["stack_by_country"][c], 6)}
        for c in lv["L0_current"]["stack_by_country"]}

    # k disagreement (oracle k before arbitration vs current k after arbitration), stack+val S1 with candidates
    kcur = sel0.groupby("q").size().reindex(kmap.index).fillna(0).values
    ko = kmap.values
    res["k_disagreement"] = {"n_s1": int(len(ko)), "oracle_more": int((ko > kcur).sum()),
                             "oracle_fewer": int((ko < kcur).sum()), "same": int((ko == kcur).sum()),
                             "oracle_more_by_country": pd.Series(ko > kcur).groupby(country[kmap.index.values]).sum()
                             .astype(int).to_dict(),
                             "oracle_fewer_by_country": pd.Series(ko < kcur).groupby(country[kmap.index.values]).sum()
                             .astype(int).to_dict()}
    log(f"k disagreement: {res['k_disagreement']}")

    if cfg == CONFIGS[0]:
        log("calibration check (expected vs actual F0.5 by country, L0)")
        res["calibration"] = {}
        for sp_, qe in (("stack", sp["g_stack"]), ("val", sp["g_val"])):
            e = expected_by_country(both, sel0, qe, country, pool)
            res["calibration"][sp_] = {c: {"expected": e.get(c), "actual": lv["L0_current"][f"{sp_}_by_country"].get(c)}
                                      for c in e}
        log(f"calibration: {res['calibration']}")
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = load_splits()
    s2 = pd.read_parquet(WORK / "s2" / "train.parquet", columns=["q", "c", "y"])
    stack = s2[np.isin(s2.q.values, sp["g_stack"])].reset_index(drop=True)
    val = s2[np.isin(s2.q.values, sp["g_val"])].reset_index(drop=True)
    country = load_prep("train", "s1", ["country"]).country.values
    prelim = pickle.loads((WORK / "models" / "s2_prelim.pkl").read_bytes())
    gates = json.loads((WORK / "reports" / "stage2_gates.json").read_text())["configs"]
    unavailable = json.loads((WORK / "reports" / "val_loss_decomposition.json").read_text())
    log(f"stack {len(stack)} pairs / {len(sp['g_stack'])} S1; val {len(val)} pairs / {len(sp['g_val'])} S1")

    out = {"unavailable_from_val_loss_decomposition_ce": unavailable, "configs": {}}
    with Pool(N_POOL) as pool:
        for cfg in CONFIGS:
            out["configs"][cfg] = run_config(cfg, prelim, stack, val, sp, country, gates, pool)
            (OUT / "oracle_decomposition.json").write_text(json.dumps(out, indent=2, default=str))

    g = out["configs"]["ce_s"]["gaps"]["stack"]["k_policy_L1-L0"]
    out["decision"] = {"k_policy_headroom_stack": g, "threshold": 0.001,
                       "verdict": "build k-policy" if g >= 0.001 else "stop policy work; submit ce_s"}
    (OUT / "oracle_decomposition.json").write_text(json.dumps(out, indent=2, default=str))
    log(f"DECISION: {out['decision']}")


if __name__ == "__main__":
    main()
