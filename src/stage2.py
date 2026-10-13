"""Stage 2 (README v2): LightGBM over the stage-2 set with
  X  the six name-sharing features (MODEL5_FALSE_NEGATIVE_FIXES.md §3.1, counts over the split's whole S1 table)
  S  structure / competitor features: reverse search over ALL S1 (rev_*) + label-free name-structure flags (§3.4)
  C  both pairwise cross-encoders          L  the listwise cross-encoder
Every choice is made on the stack split with competitor claims (stack and val decided jointly, as all S1 compete on
test, README v2 §2.9/§3.3); val is reported with and without claims. Gated ladder:
  noce = B+X,  ce_nox = B+C (six-feature gate),  ce = B+X+C,  ce_s = B+X+S+C (keep S if >= +0.0005),
  ce_lce = best + L (keep L if >= +0.0015, README §3.5),  lce_only = report only (README §2.7).
Also: test inference and submission files, label-free France checks (§2.3, A3), val loss decomposition with the
"owner not scored" bucket (A5), the count-feature subsample test (§2.5), and the leave-one-country-out report."""
import pickle
import subprocess
import sys
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import get_logger, load_json, n_workers, save_json, save_parquet, shard_path
from config import CES, CFG, LCE_ON, OUTPUT, PATHS, WORK
from decide import (crossfit_isotonic, decide, expected_f05, f05_selftest, fit_isotonic, fold_of, in_sorted,
                    macro_f05)
from prep import load_prep, load_splits
from retrieve_dense import n_shards
from stacker import MODELS, _loco, _pred_stats, train_lgb

log = get_logger("stage2")
X_COLS = ["x_empty", "x_name_eq", "x_name_ns_eq", "x_n_s1_cand_name", "x_n_s1_cand_name_ns", "x_n_s1_own_name"]
S_COLS = ["rev_rank", "rev_in_topk", "rev_gap_best_other", "rev_n_close", "name_is_other_s1",
          "empty_and_unique_name", "empty_and_shared_name", "sibling_same_name", "scrambled"]
CE_COLS = [f"ce_{k}{x}" for k in CES for x in ("", "_rank", "_gap")] + ["ce_mean_qz"]
LCE_COLS = ["lce", "lce_rank", "lce_gap", "lce_cls"]
V2_STACK_OOF = 0.98543              # README v2: model-4 + six features, stack OOF (the line model-5 must beat)
GATE_S, GATE_L, GATE_X = 0.0005, 0.0015, 0.002
MULT = np.int64(1 << 32)


def key(d):
    return d.q.values.astype(np.int64) * MULT + d.c.values


# ----------------------------------------------------------------------------------------------- features

def name_features(d, split, s1_table=None):
    """The six features exactly as MODEL5_FALSE_NEGATIVE_FIXES.md §3.1 (reference fn_tests.py::name_features), plus
    the derived structure flags. Counts are over `s1_table` (default: the split's whole S1 table)."""
    s1 = load_prep(split, "s1", ["country", "name_core", "name_ns"])
    pool = load_prep(split, "pool", ["name_core", "name_ns", "addr_empty"])
    t = s1 if s1_table is None else s1.iloc[s1_table]
    cn = (t.country + "\t" + t.name_core).value_counts()
    cns = (t.country + "\t" + t.name_ns).value_counts()
    ctry = s1.country.values[d.q.values]
    a, b = s1.name_core.values[d.q.values], pool.name_core.values[d.c.values]
    ans, bns = s1.name_ns.values[d.q.values], pool.name_ns.values[d.c.values]
    f = pd.DataFrame(index=d.index)
    f["x_empty"] = pool.addr_empty.values[d.c.values].astype(np.float32)
    f["x_name_eq"] = (a == b).astype(np.float32)
    f["x_name_ns_eq"] = (ans == bns).astype(np.float32)
    f["x_n_s1_cand_name"] = pd.Series(ctry + "\t" + b).map(cn).fillna(0).clip(upper=10).values.astype(np.float32)
    f["x_n_s1_cand_name_ns"] = pd.Series(ctry + "\t" + bns).map(cns).fillna(0).clip(upper=10).values.astype(np.float32)
    f["x_n_s1_own_name"] = pd.Series(ctry + "\t" + a).map(cn).fillna(0).clip(upper=10).values.astype(np.float32)
    # label-free structure flags (v2 berx/structure.py, computed from the same whole-table counts)
    f["name_is_other_s1"] = ((f.x_name_eq == 0) & (f.x_n_s1_cand_name >= 1)).astype(np.float32)
    f["empty_and_unique_name"] = ((f.x_empty == 1) & (f.x_name_eq == 1) & (f.x_n_s1_cand_name == 1)).astype(np.float32)
    f["empty_and_shared_name"] = ((f.x_empty == 1) & (f.x_name_eq == 1) & (f.x_n_s1_cand_name >= 3)).astype(np.float32)
    return f, b


def _sibling(s2, cand_name):
    """Does this S1 already have a strong candidate (p1 >= 0.5) with the same name as this one?"""
    k = pd.Series(s2.q.values.astype(str)) + "\t" + pd.Series(cand_name)
    strong = set(k[s2.p1.values >= 0.5])
    return np.where(s2.p1.values < 0.5, k.isin(strong).values, False).astype(np.float32)


def attach(split, s2, with_lce=False):
    """Join union features (featx), X/S features, both cross-encoders and (with_lce) the listwise scores onto the
    stage-2 rows. The listwise scores must be complete when requested (load_lce_scores fails otherwise)."""
    from cross_encoder import load_ce_scores
    qs = np.unique(s2.q.values)
    kk = s2[["q", "c"]]
    parts = []
    for i in range(n_shards(split)):
        d = pd.read_parquet(shard_path("featx", split, i))
        d = d[np.isin(d.q.values, qs)]
        parts.append(kk[np.isin(kk.q.values, np.unique(d.q.values))].merge(d, on=["q", "c"], how="left"))
    f = pd.concat(parts, ignore_index=True)
    s2 = s2.merge(f.drop(columns=[c for c in ("y",) if c in s2.columns]), on=["q", "c"], how="left")
    xf, cand_name = name_features(s2, split)
    s2 = pd.concat([s2, xf], axis=1)
    s2["sibling_same_name"] = _sibling(s2, cand_name)
    s2["scrambled"] = (s2.n_jacc.fillna(0) == 0).astype(np.float32)
    s2 = s2.merge(pd.read_parquet(WORK / "rev" / f"{split}.parquet"), on=["q", "c"], how="left")
    zs = []
    for k in CES:
        s2 = s2.merge(load_ce_scores(k, split), on=["q", "c"], how="left")
        g = s2.groupby("q")[f"ce_{k}"]
        s2[f"ce_{k}_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
        s2[f"ce_{k}_gap"] = (g.transform("max") - s2[f"ce_{k}"]).astype(np.float32)
        zs.append((s2[f"ce_{k}"] - g.transform("mean")) / (g.transform("std").fillna(0) + 1))
    s2["ce_mean_qz"] = (sum(zs) / len(zs)).astype(np.float32)
    if with_lce:
        from lce import load_lce_scores
        s2 = s2.merge(load_lce_scores(split), on=["q", "c"], how="left")
        g = s2.groupby("q").lce
        s2["lce_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
        s2["lce_gap"] = (g.transform("max") - s2.lce).astype(np.float32)
    return s2


def cols(s2, cfg):
    base = [c for c in s2.columns if c not in ("q", "c", "y") and c not in X_COLS + S_COLS + CE_COLS + LCE_COLS]
    groups = {"noce": base + X_COLS, "ce_nox": base + CE_COLS, "ce": base + X_COLS + CE_COLS,
              "ce_s": base + X_COLS + S_COLS + CE_COLS}
    if cfg in groups:
        return groups[cfg]
    if cfg.startswith("ce_lce"):                        # ce_lce:<base cfg>
        return cols(s2, cfg.split(":")[1]) + LCE_COLS
    raise ValueError(cfg)


# ----------------------------------------------------------------------------------------------- training helpers

def model_tag(cfg, feats):
    """Model files carry a hash of their exact feature list: a saved fold model is reused only for the same features
    (a resumed run never loads a model trained on a different column set)."""
    import hashlib
    return f"{cfg.replace(':', '_')}_{hashlib.sha1(','.join(feats).encode()).hexdigest()[:8]}"


def _fit(cfg, stack, val, feats):
    F = CFG["n_folds"]
    fold = fold_of(stack.q.values, F)
    oof, vp = np.zeros(len(stack)), np.zeros(len(val))
    tag = model_tag(cfg, feats)
    for k in range(F):
        tr = stack[fold != k]
        m = train_lgb(tr[feats].values.astype(np.float32), tr.y.values, tr.q.values, MODELS / f"s2_{tag}_fold{k}.txt",
                      100 + k)
        oof[fold == k] = m.predict(stack.loc[fold == k, feats].values.astype(np.float32))
        vp += m.predict(val[feats].values.astype(np.float32)) / F
    return oof, vp


def _score_cfg(cfg, oof, vp, stack, val, sp, fams, pool):
    """Stack F0.5 (with val's claims in arbitration) per decision family; val F0.5 with and without claims."""
    cal = crossfit_isotonic(oof, stack.y.values, stack.q.values, CFG["n_folds"])
    iso = fit_isotonic(oof, stack.y.values)
    vcal = iso.predict(vp)
    both = pd.concat([stack[["q", "c"]].assign(p=cal), val[["q", "c"]].assign(p=vcal)], ignore_index=True)
    res = {}
    for fam, thr in fams:
        sel = decide(both, fam, thr, CFG["arb_max_iter"], pool)
        res[(fam, thr)] = macro_f05(sel, sp["g_stack"], sp["n_true"], sp["gt_q"], sp["gt_c"])[0]
    fam, thr = max(res, key=res.get)
    sel = decide(both, fam, thr, CFG["arb_max_iter"], pool)
    v_claims = macro_f05(sel, sp["g_val"], sp["n_true"], sp["gt_q"], sp["gt_c"])[0]
    v_plain = macro_f05(decide(val[["q", "c"]].assign(p=vcal), fam, thr, CFG["arb_max_iter"], pool),
                        sp["g_val"], sp["n_true"], sp["gt_q"], sp["gt_c"])[0]
    log.info(f"[{cfg}] stack F0.5 {res[(fam, thr)]:.5f} ({fam}, {thr}); val {v_claims:.5f} with claims, "
             f"{v_plain:.5f} plain")
    return {"stack": res[(fam, thr)], "family": fam, "thr": thr, "val_claims": v_claims, "val_plain": v_plain,
            "all_families": {f"{a}|{b}": v for (a, b), v in res.items()}}


def _load_train(with_lce=False):
    sp = load_splits()
    s2 = attach("train", pd.read_parquet(WORK / "s2" / "train.parquet"), with_lce)
    stack = s2[np.isin(s2.q.values, sp["g_stack"])].reset_index(drop=True)
    val = s2[np.isin(s2.q.values, sp["g_val"])].reset_index(drop=True)
    return sp, s2, stack, val


def _miss_rate(sp, s2_stack):
    return float((sp["n_true"][sp["g_stack"]].sum() - s2_stack.y.sum()) / len(sp["g_stack"]))


PRELIM = ["noce", "ce_nox", "ce", "ce_s"]


def run_stage2_prelim():
    """The ladder without the listwise model (runs while listwise inference is still on the GPUs)."""
    f05_selftest()
    out = MODELS / "s2_prelim.pkl"
    done = pickle.loads(out.read_bytes()) if out.exists() else {}
    sp, s2, stack, val = _load_train()
    lam = _miss_rate(sp, stack)
    fams = [("expf05", 0.0), ("expf05", lam)]
    with Pool(n_workers()) as pool:
        for cfg in PRELIM:
            if cfg in done:
                continue
            feats = cols(s2, cfg)
            oof, vp = _fit(cfg, stack, val, feats)
            done[cfg] = {"feats": feats, "oof": oof, "vp": vp, **_score_cfg(cfg, oof, vp, stack, val, sp, fams, pool)}
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(done))
            tmp.replace(out)
    _gate_report(done)


def _gate_report(done):
    r = {c: {k: v for k, v in d.items() if k not in ("oof", "vp", "feats")} for c, d in done.items()}
    rep = {"configs": r}
    if "ce" in r and "ce_nox" in r:
        imp = _importance("ce", done["ce"]["feats"])
        rank = {f: int(i) + 1 for i, f in enumerate(imp.feature)}
        rep["six_features_gate"] = {"stack_gain": r["ce"]["stack"] - r["ce_nox"]["stack"], "required": GATE_X,
                                    "passes": r["ce"]["stack"] - r["ce_nox"]["stack"] >= GATE_X,
                                    "gain_rank_of": {f: rank.get(f) for f in X_COLS},
                                    "top20_required": ["x_n_s1_own_name", "x_n_s1_cand_name_ns"]}
    if "ce_s" in r and "ce" in r:
        rep["structure_gate"] = {"stack_gain": r["ce_s"]["stack"] - r["ce"]["stack"], "required": GATE_S,
                                 "passes": r["ce_s"]["stack"] - r["ce"]["stack"] >= GATE_S}
    save_json(rep, WORK / "reports" / "stage2_gates.json")
    log.info(f"stage-2 gates: {rep}")


def _importance(cfg, feats):
    tag = model_tag(cfg, feats)
    imp = pd.DataFrame({"feature": feats})
    for k in range(CFG["n_folds"]):
        imp[f"g{k}"] = lgb.Booster(model_file=str(MODELS / f"s2_{tag}_fold{k}.txt")).feature_importance("gain")
    imp["gain"] = imp.filter(regex=r"^g\d").mean(1)
    return imp.sort_values("gain", ascending=False)[["feature", "gain"]].reset_index(drop=True)


def run_stage2_train():
    done = pickle.loads((MODELS / "s2_prelim.pkl").read_bytes())
    sp, s2, stack, val = _load_train(with_lce=LCE_ON)
    country = load_prep("train", "s1", ["country"]).country.values
    lam = _miss_rate(sp, stack)
    base = "ce_s" if done["ce_s"]["stack"] - done["ce"]["stack"] >= GATE_S else "ce"
    chosen = base
    with Pool(n_workers()) as pool:
        if LCE_ON and "lce" in s2:
            cfg = f"ce_lce:{base}"
            feats = cols(s2, cfg)
            oof, vp = _fit(cfg, stack, val, feats)
            done[cfg] = {"feats": feats, "oof": oof, "vp": vp,
                         **_score_cfg(cfg, oof, vp, stack, val, sp, [("expf05", 0.0), ("expf05", lam)], pool)}
            if done[cfg]["stack"] - done[base]["stack"] >= GATE_L:
                chosen = cfg
            o, v = stack.lce.fillna(-20).values, val.lce.fillna(-20).values        # report only (README §2.7)
            done["lce_only"] = {"feats": ["lce"], "oof": o, "vp": v,
                                **_score_cfg("lce_only", o, v, stack, val, sp, [("expf05", 0.0)], pool)}
        # full decision-family grid on the chosen configuration only
        d = done[chosen]
        fams = [("expf05", 0.0), ("expf05", lam)] + [("threshold", t) for t in CFG["thr_grid"]]
        final = _score_cfg(chosen, d["oof"], d["vp"], stack, val, sp, fams, pool)
        iso = fit_isotonic(d["oof"], stack.y.values)
        cal = crossfit_isotonic(d["oof"], stack.y.values, stack.q.values, CFG["n_folds"])
        vcal = iso.predict(d["vp"])
        both = pd.concat([stack[["q", "c"]].assign(p=cal), val[["q", "c"]].assign(p=vcal)], ignore_index=True)
        sel_both = decide(both, final["family"], final["thr"], CFG["arb_max_iter"], pool)
        sel_val = sel_both[np.isin(sel_both.q.values, sp["g_val"])]
        f, per = macro_f05(sel_val, sp["g_val"], sp["n_true"], sp["gt_q"], sp["gt_c"])
        exp_val = _expected_by_country(val[["q", "c"]].assign(p=vcal), sel_val, country, pool)
        subs = _subsample_test(chosen, d["feats"], val, iso, final, sp, pool)
    per["country"] = country[per.q.values]
    by_c = per.groupby("country").f.mean().to_dict()
    val_stats = _pred_stats(sel_val, sp["g_val"], country)
    bundle = {"config": chosen, "family": final["family"], "thr": final["thr"], "features": d["feats"],
              "miss_rate": lam, "stack_f05": final["stack"], "val_f05": f, "val_f05_plain": final["val_plain"],
              "val_f05_by_country": by_c, "val_pred_stats": val_stats,
              "ladder": {c: {k: v for k, v in x.items() if k not in ("oof", "vp", "feats")} for c, x in done.items()},
              "beats_v2_line": {"stack": final["stack"], "v2_stack_oof": V2_STACK_OOF,
                                "switch_rule_passes": final["stack"] >= V2_STACK_OOF + 0.001},
              "val_expected_vs_actual_by_country": {c: {"expected": exp_val.get(c), "actual": by_c.get(c)} for c in by_c},
              "count_subsample_test": subs}
    with open(MODELS / "stage2_iso.pkl", "wb") as fh:
        pickle.dump(iso, fh)
    save_json(bundle, MODELS / "stage2_bundle.json")
    _gate_report(done)
    if chosen != "lce_only":
        _importance(chosen, d["feats"]).to_csv(WORK / "reports" / "feature_importance.tsv", sep="\t", index=False)
    per.sort_values("f").head(2000).to_csv(WORK / "reports" / "val_errors.tsv", sep="\t", index=False)
    decomp = _loss_decomposition(sel_val, sp, stack, val)
    save_json(decomp, WORK / "reports" / "val_loss_decomposition.json")
    lines = ["# Stage-2 validation report (README v2)", "",
             f"- chosen on stack: **{chosen}**, decision **{final['family']}** (param {final['thr']})",
             f"- stack OOF F0.5 (with competitor claims): **{final['stack']:.5f}**  (v2 line: {V2_STACK_OOF}; "
             f"switch rule +0.001: {'PASS' if bundle['beats_v2_line']['switch_rule_passes'] else 'not met'})",
             f"- validation macro F0.5: **{f:.5f}** with competitor claims, {final['val_plain']:.5f} plain",
             f"- by country: {', '.join(f'{k}: {v:.5f}' for k, v in by_c.items())}",
             f"- ladder (stack / val): " + "; ".join(f"{c} {x['stack']:.5f}/{x['val_claims']:.5f}"
                                                   for c, x in bundle['ladder'].items()),
             f"- expected vs actual val F0.5 by country: {bundle['val_expected_vs_actual_by_country']}",
             f"- count features on a ~260k-per-country S1 table (README §2.5): {subs}",
             f"- loss decomposition (val): {decomp['f05_lost_by_bucket']}",
             f"- predicted matches per S1 (val): {val_stats}"]
    (WORK / "reports" / "report.md").write_text("\n".join(lines) + "\n")
    log.info("\n".join(lines))


# ----------------------------------------------------------------------------------------------- reports

def _exp_group(args):
    from decide import _poisson
    rows, lam = args
    miss = _poisson(lam) if lam else None
    return [(q, expected_f05(ps, pr, miss)) for q, ps, pr in rows]


def _expected_by_country(scored, sel, country, pool, lam=0.0):
    """Model's own expected F0.5 of its chosen sets (calibrated probabilities), averaged per country (README §2.3)."""
    k = np.sort(key(sel))
    d = scored.assign(s=in_sorted(key(scored), k)).sort_values("q")
    rows = [(q, g.p.values[g.s.values], g.p.values[~g.s.values]) for q, g in d.groupby("q", sort=False)]
    step = 20_000
    res = [x for part in pool.map(_exp_group, [(rows[i:i + step], lam) for i in range(0, len(rows), step)]) for x in part]
    e = pd.Series(dict(res))
    return e.groupby(country[e.index.values]).mean().round(5).to_dict()


def _subsample_test(cfg, feats, val, iso, final, sp, pool):
    """README §2.5: counts on a per-country ~260k S1 table (France-sized; the val entities themselves always kept),
    same model; the val F0.5 change says how much the count features depend on table size."""
    if cfg == "lce_only":
        return None
    s1c = load_prep("train", "s1", ["country"]).country.values
    rng = np.random.default_rng(7)
    keep = [np.asarray(sp["g_val"])]
    for c in np.unique(s1c):
        rows = np.setdiff1d(np.flatnonzero(s1c == c), sp["g_val"])
        n = max(0, 260_000 - int((s1c[sp["g_val"]] == c).sum()))
        keep.append(rng.choice(rows, min(n, len(rows)), replace=False))
    table = np.sort(np.concatenate(keep))
    v = val.copy()
    xf, _ = name_features(v, "train", table)
    for c in xf.columns:
        v[c] = xf[c].values
    tag = model_tag(cfg, feats)
    ms = [lgb.Booster(model_file=str(MODELS / f"s2_{tag}_fold{k}.txt")) for k in range(CFG["n_folds"])]
    p = iso.predict(sum(m.predict(v[feats].values.astype(np.float32)) for m in ms) / len(ms))
    sel = decide(v[["q", "c"]].assign(p=p), final["family"], final["thr"], CFG["arb_max_iter"], pool)
    fs = macro_f05(sel, sp["g_val"], sp["n_true"], sp["gt_q"], sp["gt_c"])[0]
    out = {"val_plain_full_table": final["val_plain"], "val_plain_260k_table": fs,
           "delta": fs - final["val_plain"], "flag": fs - final["val_plain"] < -0.001}
    log.info(f"count subsample test: {out}")
    return out


def _loss_decomposition(sel, sp, stack, val):
    """Where val F0.5 is lost (README A5), by counterfactual: add back / remove one bucket at a time. False positives
    are split by their true owner: none (distractor) / a scored S1 (stack or val) / an S1 not scored in this eval
    (the 4th bucket: test-time arbitration handles it, so it is not available gain)."""
    q_eval = sp["g_val"]
    gt = pd.DataFrame({"q": sp["gt_q"], "c": sp["gt_c"]})
    gtv = gt[np.isin(gt.q.values, q_eval)]
    owner = pd.Series(sp["gt_q"], index=sp["gt_c"])
    base, _ = macro_f05(sel, q_eval, sp["n_true"], sp["gt_q"], sp["gt_c"])
    ks, kv = np.sort(key(sel)), np.sort(key(val[val.y == 1]))
    un = pd.read_parquet(WORK / "stage1" / "train.parquet", columns=["q", "c", "y"])
    un = pd.concat([un[un.y == 1], pd.read_parquet(WORK / "stage1x" / "train.parquet", columns=["q", "c", "y"])
                    .pipe(lambda d: d[d.y == 1])])
    ku = np.sort(key(un))
    missing = gtv[~in_sorted(key(gtv), ks)]
    rej = missing[in_sorted(key(missing), kv)]                     # in the stage-2 set, not selected
    cut = missing[~in_sorted(key(missing), kv) & in_sorted(key(missing), ku)]   # retrieved, dropped before stage 2
    never = missing[~in_sorted(key(missing), ku)]                  # never retrieved
    fp = sel[~in_sorted(key(sel), np.sort(key(gt)))]
    own = owner.reindex(fp.c.values).values
    scored = np.isin(own, np.concatenate([sp["g_stack"], sp["g_val"]]))
    buckets = {"rejected_in_stage2": (rej, "add"), "dropped_before_stage2": (cut, "add"), "never_retrieved": (never, "add"),
               "fp_distractor": (fp[np.isnan(own)], "remove"), "fp_owner_scored": (fp[~np.isnan(own) & scored], "remove"),
               "fp_owner_not_scored": (fp[~np.isnan(own) & ~scored], "remove")}
    lost, pairs = {}, {}
    for name, (dfb, how) in buckets.items():
        s = pd.concat([sel[["q", "c"]], dfb[["q", "c"]]]) if how == "add" else sel[~in_sorted(key(sel), np.sort(key(dfb)))]
        lost[name] = round(macro_f05(s, q_eval, sp["n_true"], sp["gt_q"], sp["gt_c"])[0] - base, 5)
        pairs[name] = int(len(dfb))
    return {"val_f05": base, "f05_lost_by_bucket": lost, "pairs_by_bucket": pairs,
            "note": "fp_owner_not_scored is handled by test-time arbitration; not available gain"}


# ----------------------------------------------------------------------------------------------- test + submission

def run_stage2_test():
    bundle = load_json(MODELS / "stage2_bundle.json")
    with open(MODELS / "stage2_iso.pkl", "rb") as fh:
        iso = pickle.load(fh)
    cfg = bundle["config"]
    s2 = attach("test", pd.read_parquet(WORK / "s2" / "test.parquet"), with_lce=cfg.startswith("ce_lce") or cfg == "lce_only")
    if cfg == "lce_only":
        raw = s2.lce.fillna(-20).values
    else:
        X = s2[bundle["features"]].values.astype(np.float32)
        tag = model_tag(cfg, bundle["features"])
        ms = [lgb.Booster(model_file=str(MODELS / f"s2_{tag}_fold{k}.txt")) for k in range(CFG["n_folds"])]
        raw = sum(m.predict(X, num_threads=CFG["lgb_threads"]) for m in ms) / len(ms)
    s2["p"] = iso.predict(raw)
    keep = ["q", "c", "p1", "p", "from_ename"] + [c for c in ("ce_m", "ce_b", "lce") if c in s2]
    save_parquet(s2[keep], WORK / "stage2" / "test_scores.parquet")
    with Pool(n_workers()) as pool:
        sel = decide(s2[["q", "c", "p"]], bundle["family"], bundle["thr"], CFG["arb_max_iter"], pool)
        country = load_prep("test", "s1", ["country"]).country.values
        exp_test = _expected_by_country(s2[["q", "c", "p"]], sel, country, pool)
    write_submission(s2, sel, bundle, exp_test)


def _name_token_report(split):
    """README A3: most frequent name_core tokens per country, and how often a discovered legal form is still inside
    name_core (a normalisation miss)."""
    s1 = load_prep(split, "s1", ["country", "name_core", "legal"])
    forms = load_json(WORK.parent / "work" / "prep" / "legal_forms.json") if (WORK.parent / "work" / "prep" / "legal_forms.json").exists() else {}
    out = {}
    for c, g in s1.groupby("country"):
        toks = pd.Series(" ".join(g.name_core.values).split())
        fl = set(" ".join(forms.get(c, [])).split())
        out[c] = {"top_tokens": toks.value_counts().head(25).to_dict(),
                  "legal_form_stripped_rate": round(float((g.legal != "").mean()), 4),
                  "legal_form_token_left_in_core_rate": round(float(g.name_core.str.split().apply(
                      lambda t: bool(fl & set(t))).mean()), 4) if fl else None}
    return out


def write_submission(s2, sel, bundle, exp_test):
    sp = load_splits()
    s1 = load_prep("test", "s1", ["id", "country"])
    pool = load_prep("test", "pool", ["id", "country"])
    pool_ids = pool.id.values
    q_all = sp["q_test"]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    def lists(df):
        d = df.assign(cid=pool_ids[df.c.values]).groupby("q").cid.apply(lambda x: ",".join(dict.fromkeys(x)))
        return pd.Series(q_all).map(d).fillna("").values

    # candidate_pairs.tsv = the stage-2 set: exactly what the final model scores (competition README)
    cand = pd.DataFrame({"source1_entity_id": s1.id.values[q_all], "candidate_entity_ids": lists(s2)})
    match = pd.DataFrame({"source1_entity_id": s1.id.values[q_all], "matched_entity_ids": lists(sel)})
    for df, name in ((match, "matching_results.tsv"), (cand, "candidate_pairs.tsv")):
        tmp = OUTPUT / f".{name}.tmp"
        df.to_csv(tmp, sep="\t", index=False)
        tmp.replace(OUTPUT / name)
    assert match.source1_entity_id.is_unique and len(match) == len(q_all)
    assert not sel.c.duplicated().any(), "one owner per record violated"
    cset = dict(zip(cand.source1_entity_id, cand.candidate_entity_ids))
    for sid, m in zip(match.source1_entity_id, match.matched_entity_ids):
        if m:
            ids = m.split(",")
            assert len(ids) == len(set(ids)) and all(x[:3] in ("S2-", "S3-") for x in ids)
            assert set(ids) <= set(cset[sid].split(",")), f"match outside candidates for {sid}"
    stats = _pred_stats(sel, q_all, s1.country.values)
    pool_per_s1 = (pool.country.value_counts() / pd.Series(s1.country.values[q_all]).value_counts()).to_dict()
    f3 = {c: {**v, "pool_per_s1": round(float(pool_per_s1.get(c, np.nan)), 3),
              "pred_per_pool_record": round(v["mean_pred"] / pool_per_s1.get(c, np.nan), 4),
              "expected_f05": exp_test.get(c)} for c, v in stats.items()}
    val_mean = np.mean([v["mean_pred"] for v in bundle["val_pred_stats"].values()])
    val_empty = np.mean([v["empty_rate"] for v in bundle["val_pred_stats"].values()])
    ratios = [v["pred_per_pool_record"] for c, v in f3.items() if c in bundle["val_pred_stats"]]
    ref_ratio = float(np.mean(ratios)) if ratios else np.nan
    flags = []
    for c, v in f3.items():
        if abs(v["mean_pred"] - val_mean) > 0.35 * val_mean or abs(v["empty_rate"] - val_empty) > 0.05:
            flags.append(f"{c}: mean_pred={v['mean_pred']} empty_rate={v['empty_rate']} vs val {val_mean:.3f}/{val_empty:.3f}")
        if np.isfinite(ref_ratio) and abs(v["pred_per_pool_record"] / ref_ratio - 1) > 0.05:
            flags.append(f"{c}: predicted matches per pool record {v['pred_per_pool_record']} vs {ref_ratio:.4f} (>5%, F3)")
    # README §2.3 reading of the expected-F0.5 check
    lab = [exp_test[c] for c in exp_test if c in bundle["val_f05_by_country"]]
    unl = {c: exp_test[c] for c in exp_test if c not in bundle["val_f05_by_country"]}
    rep = {"test_pred_stats": f3, "val_pred_stats": bundle["val_pred_stats"],
           "expected_f05_test": exp_test, "expected_vs_actual_val": bundle.get("val_expected_vs_actual_by_country"),
           "unlabelled_countries_expected_f05": unl,
           "labelled_countries_expected_f05_mean": float(np.mean(lab)) if lab else None,
           "name_tokens": _name_token_report("test"),
           "ename_share_of_candidates": float(s2.from_ename.mean()),
           "ename_share_of_matches": float(sel.merge(s2[["q", "c", "from_ename"]], on=["q", "c"]).from_ename.mean()),
           "train_singleton_rate": float((sp["n_true"] == 0).mean()), "train_mean_true": float(sp["n_true"].mean()),
           "n_pred_pairs": int(len(sel)), "n_candidate_pairs": int(len(s2)), "country_shift_flags": flags}
    if CFG["test_limit"] is None:
        r = subprocess.run([sys.executable, str(PATHS["validator"]), "--matching", str(OUTPUT / "matching_results.tsv"),
                            "--candidate", str(OUTPUT / "candidate_pairs.tsv"), "--test-dir", str(PATHS["test_tsv"]),
                            "--check-ids"], capture_output=True, text=True)
        rep["validator"] = {"returncode": r.returncode, "stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:]}
        if r.returncode != 0:
            save_json(rep, WORK / "reports" / "test_report.json")
            raise RuntimeError(f"validator failed:\n{r.stdout}\n{r.stderr}")
    save_json(rep, WORK / "reports" / "test_report.json")
    log.info(f"submission written to {OUTPUT}; per-country: {f3}; flags: {flags}")


# ----------------------------------------------------------------------------------------------- LOCO report

def run_loco():
    """Leave-one-country-out with the stage-2 feature sets. BOTH rows are UPPER BOUNDS for an unseen country: 'noce'
    includes the stage-1 score p1 (stage 1 was trained on both countries' stack) and 'ce' adds cross-encoders trained on
    both countries' text (README v2 §2.6). The leak-free transfer estimate is reports/f1_select.json (every component
    trained on one country)."""
    sp, s2, _, _ = _load_train()
    country = load_prep("train", "s1", ["country"]).country.values
    rep = {}
    with Pool(n_workers()) as pool:
        for cfg in ("noce", "ce"):
            for tr_c in sorted(set(country[sp["g_stack"]])):
                _loco(s2, cols(s2, cfg), sp, country, pool, f"s2{cfg}", tr_c, rep)
    rep["note"] = ("train country -> F0.5 on each country's val. ALL rows are upper bounds: 'noce' uses stage-1 p1 and "
                   "'ce' the cross-encoders, both trained on both countries (README v2 §2.6). Leak-free transfer: "
                   "reports/f1_select.json")
    save_json(rep, WORK / "reports" / "loco.json")
