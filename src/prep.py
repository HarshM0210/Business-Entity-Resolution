"""Stages: fetch (Hugging Face inputs), manifest (integrity), prep (normalised views + legal forms), test_tsv,
splits."""
import os
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import text_utils as tu
from common import Progress, get_logger, load_json, n_workers, save_json, save_npz, save_parquet
from config import CFG, EMB, EMB_MODELS, HF, PATHS, PREP, WORK
from embeddings import emb_ids, load_emb

log = get_logger("prep")
RAW_COLS = ["entity_id", "business_name", "business_address", "country"]
SIDES = [("train", "s1"), ("train", "pool"), ("test", "s1"), ("test", "pool")]


# ----------------------------------------------------------------------------------------------- fetch

def _needed_files():
    need = {"data": [p for sp in PATHS["data"].values() for p in sp.values()] + [PATHS["ground_truth"]]}
    for m in EMB_MODELS:
        spec = EMB[m]
        fs = []
        for split in ("train", "test"):
            d = Path(spec[split])
            if spec["kind"] == "shards":
                fs += [d / f"source{s}" / "manifest.json" for s in (1, 2, 3)]
            elif spec["kind"] == "s1pool":
                fs += [d / f"{s}{x}" for s in ("s1", "pool") for x in (".npy", ".ids.txt")]
            else:
                fs += [d / f"{split}_{s}" / x for s in ("s1", "pool") for x in ("matrix.npy", "ids.txt")]
        need[m] = fs
    return need


def run_fetch():
    """Download any missing input from Hugging Face (snapshot_download resumes partial files). Nothing is
    downloaded when every file is already present."""
    missing = {k: [str(f) for f in fs if not Path(f).exists()] for k, fs in _needed_files().items()}
    missing = {k: v for k, v in missing.items() if v}
    if not missing:
        log.info("all inputs present locally; nothing to download")
        return
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    for key, files in missing.items():
        repo, local, allow = HF[key]
        log.info(f"{key}: {len(files)} missing file(s), e.g. {files[0]} -> downloading {repo} into {local}")
        for attempt in range(5):
            try:
                snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(local), token=token,
                                  allow_patterns=allow, max_workers=8)
                break
            except Exception as e:                      # network flake: back off and resume
                log.warning(f"{repo}: download attempt {attempt + 1} failed: {e}")
                if attempt == 4:
                    raise
                import time
                time.sleep(60 * (attempt + 1))
    still = [str(f) for fs in _needed_files().values() for f in fs if not Path(f).exists()]
    if still:
        raise RuntimeError(f"inputs still missing after download: {still[:5]}")


# ----------------------------------------------------------------------------------------------- manifest

def read_raw(split, side):
    files = [PATHS["data"][split][1]] if side == "s1" else [PATHS["data"][split][2], PATHS["data"][split][3]]
    df = pd.concat([pq.read_table(f, columns=RAW_COLS).to_pandas() for f in files], ignore_index=True)
    for c in RAW_COLS:
        df[c] = df[c].fillna("").astype(str)
    return df


def prep_path(split, side):
    return PREP / f"{split}_{side}.parquet"


def load_prep(split, side, columns=None):
    return pd.read_parquet(prep_path(split, side), columns=columns)


def _norm_scan(e, rng, n=50_000):
    rows = np.unique(np.concatenate([rng.choice(len(e), min(n, len(e)), replace=False),
                                     np.arange(min(1000, len(e))), np.arange(max(0, len(e) - 1000), len(e))]))
    x = e.take(rows).astype(np.float32)
    nr = np.linalg.norm(x, axis=1)
    return {"sampled": int(len(rows)), "norm_min": float(np.nanmin(nr)), "norm_max": float(np.nanmax(nr)),
            "norm_bad": int((~np.isfinite(nr) | (np.abs(nr - 1) > 0.01)).sum())}


def run_manifest():
    """Fatal integrity checks: row counts, id order (id_mismatch must be 0), unit norm, and a semantic row
    alignment test on ground-truth pairs (true pair vs the pair shifted by one row)."""
    rep, problems = {}, []
    ids = {}
    rng = np.random.default_rng(0)
    for split, side in SIDES:
        df = read_raw(split, side)
        n = len(df)
        dup = int(df.entity_id.duplicated().sum())
        ok_pref = df.entity_id.str[:3].isin(["S1-"] if side == "s1" else ["S2-", "S3-"]).all()
        r = rep[f"{split}_{side}"] = {"rows": n, "dup_ids": dup, "prefix_ok": bool(ok_pref),
                                      "countries": df.country.value_counts().to_dict(),
                                      "empty_name": int((df.business_name == "").sum()),
                                      "empty_addr": int((df.business_address == "").sum()), "emb": {}}
        if dup or not ok_pref:
            problems.append(f"{split}_{side}: dup={dup} prefix_ok={ok_pref}")
        ids[(split, side)] = df.entity_id.values
        for m in EMB_MODELS:
            e = load_emb(m, split, side)
            er = r["emb"][m] = {"rows": len(e), "dim": e.dim, **_norm_scan(e, rng)}
            if len(e) != n:
                problems.append(f"{m} {split}_{side}: {len(e)} emb rows vs {n} data rows")
            eids = emb_ids(m, split, side)
            if eids is not None:
                mm = int((np.asarray(eids, dtype=object) != df.entity_id.values).sum()) if len(eids) == n else None
                er["id_mismatch"] = mm
                if mm != 0:
                    problems.append(f"{m} {split}_{side}: id_mismatch={mm}")
            if er["norm_bad"]:
                problems.append(f"{m} {split}_{side}: {er['norm_bad']} sampled rows not unit-norm")
        log.info(f"checked {split}_{side}: {r}")
    gt = pd.read_csv(PATHS["ground_truth"], sep="\t", dtype=str, keep_default_na=False)
    s1ids, pool_ids = ids[("train", "s1")], ids[("train", "pool")]
    matched = [x for m in gt.matched_entity_ids if m for x in m.split(",")]
    cnt = Counter(matched)
    pool_set = set(pool_ids)
    rep["ground_truth"] = {"rows": len(gt), "s1_covered": int(gt.source1_entity_id.isin(set(s1ids)).sum()),
                           "n_matched": len(matched), "unknown_ids": int(sum(1 for x in cnt if x not in pool_set)),
                           "multi_owner": int(sum(1 for v in cnt.values() if v > 1))}
    if rep["ground_truth"]["unknown_ids"] or rep["ground_truth"]["s1_covered"] != len(s1ids):
        problems.append(f"ground truth coverage: {rep['ground_truth']}")
    if rep["ground_truth"]["multi_owner"]:
        problems.append("S2/S3 ids owned by >1 S1: one-owner arbitration invalid")
    s1_index = pd.Index(s1ids)
    pool_index = pd.Index(pool_ids)
    sub = gt[gt.matched_entity_ids != ""].sample(20_000, random_state=0)
    a = s1_index.get_indexer(sub.source1_entity_id)
    b = pool_index.get_indexer(sub.matched_entity_ids.str.split(",").str[0])
    ok = (a + 1 < len(s1ids)) & (b + 1 < len(pool_ids))
    a, b = a[ok], b[ok]
    rand = rng.choice(len(pool_ids), len(b))
    for m in EMB_MODELS:
        es, ep = load_emb(m, "train", "s1"), load_emb(m, "train", "pool")
        qa, pb = es.take(a).astype(np.float32), ep.take(b).astype(np.float32)
        t = (qa * pb).sum(1)
        sh_q = (es.take(a + 1).astype(np.float32) * pb).sum(1)
        sh_p = (qa * ep.take(b + 1).astype(np.float32)).sum(1)
        rnd = (qa * ep.take(rand).astype(np.float32)).sum(1)
        al = {"cos_true": float(t.mean()), "cos_random": float(rnd.mean()),
              "frac_true_beats_s1_shift": float((t > sh_q).mean()), "frac_true_beats_pool_shift": float((t > sh_p).mean())}
        rep["ground_truth"][f"{m}_alignment"] = al
        if al["frac_true_beats_s1_shift"] < 0.95 or al["frac_true_beats_pool_shift"] < 0.95 or \
                al["cos_true"] - al["cos_random"] < 0.1:
            problems.append(f"{m}: embedding rows not aligned with parquet rows: {al}")
    rep["problems"] = problems
    save_json(rep, WORK / "reports" / "manifest.json")
    if problems:
        raise RuntimeError("integrity manifest failed: " + "; ".join(problems))
    log.info("manifest OK")


# ----------------------------------------------------------------------------------------------- prep

_FORMS = None


def _init(forms):
    global _FORMS
    _FORMS = forms


def _tok_chunk(names):
    return [tu.name_tokens(x) for x in names]


def _proc_chunk(args):
    names, addrs, countries = args
    rows = []
    for n, a, c in zip(names, addrs, countries):
        forms, long_single = _FORMS.get(c, ([], set()))
        r = tu.process_name(n, forms, long_single)
        r.update(tu.parse_address(a))
        rows.append(r)
    return pd.DataFrame(rows)


def norm_country(s):
    return s.strip().lower() or "unknown"


def run_prep():
    PREP.mkdir(parents=True, exist_ok=True)
    forms_path = PREP / "legal_forms.json"
    raws = {}
    with Pool(n_workers()) as pool:
        if not forms_path.exists():
            by_c = {}
            for split, side in SIDES:
                df = read_raw(split, side)
                df["cn"] = df.country.map(norm_country)
                raws[(split, side)] = df
                for c, g in df.groupby("cn"):
                    by_c.setdefault(c, []).extend(g.business_name.tolist())
            rng = random.Random(CFG["seed"])
            forms = {}
            for c, names in by_c.items():
                if len(names) > CFG["legal_sample"]:
                    names = rng.sample(names, CFG["legal_sample"])
                chunks = [names[i:i + 50_000] for i in range(0, len(names), 50_000)]
                toks = [t for part in pool.map(_tok_chunk, chunks) for t in part]
                f = tu.discover_legal_forms(toks, CFG["legal_min_frac"], CFG["legal_boundary_ratio"],
                                            CFG["legal_max_cooc"], CFG["legal_bigram_tight"])
                forms[c] = [" ".join(x) for x in f]
                log.info(f"legal forms [{c}] from {len(names)} names: {forms[c]}")
            save_json(forms, forms_path)
    forms = load_json(forms_path)
    forms_t = {c: ([tuple(x.split()) for x in v], {x for x in v if " " not in x and len(x) >= 3})
               for c, v in forms.items()}
    with Pool(n_workers(), initializer=_init, initargs=(forms_t,)) as pool:
        for split, side in SIDES:
            out = prep_path(split, side)
            if out.exists():
                continue
            df = raws.pop((split, side), None)
            if df is None:
                df = read_raw(split, side)
                df["cn"] = df.country.map(norm_country)
            step = 100_000
            args = [(df.business_name.values[i:i + step], df.business_address.values[i:i + step],
                     df.cn.values[i:i + step]) for i in range(0, len(df), step)]
            prog = Progress(log, len(args), f"prep {split}_{side}")
            parts = []
            for part in pool.imap(_proc_chunk, args):
                parts.append(part)
                prog.step()
            feat = pd.concat(parts, ignore_index=True)
            res = pd.DataFrame({"id": df.entity_id.values, "country": df.cn.values,
                                "src": df.entity_id.str[1].astype(np.int8).values,
                                "raw_name": df.business_name.values, "raw_addr": df.business_address.values,
                                "raw_country": df.country.values})
            res = pd.concat([res, feat], axis=1)
            save_parquet(res, out)
            log.info(f"wrote {out} rows={len(res)}")


def run_test_tsv():
    """The organisers' validator reads test_source{1,2,3}.tsv; write them from the parquet (same ids/rows)."""
    d = PATHS["test_tsv"]
    for s in (1, 2, 3):
        out = d / f"test_source{s}.tsv"
        if out.exists():
            continue
        df = pq.read_table(PATHS["data"]["test"][s], columns=RAW_COLS).to_pandas()
        tmp = out.with_name(f".{out.name}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp, sep="\t", index=False)
        os.replace(tmp, out)
        log.info(f"wrote {out} rows={len(df)}")


# ----------------------------------------------------------------------------------------------- splits

GROUPS = ("ce", "lce", "stack", "val")


def run_splits():
    s1 = load_prep("train", "s1", ["id", "country"])
    pool_ids = load_prep("train", "pool", ["id"]).id.values
    rng = np.random.default_rng(CFG["seed"])
    perm = rng.permutation(len(s1))
    # ce/stack/val are carved first in model-4's order so that, at the full-size config, ce/stack/val equal
    # model-4's groups; lce comes from the remainder
    sizes = [("ce", CFG["n_ce"]), ("stack", CFG["n_stack"]), ("val", CFG["n_val"]), ("lce", CFG["n_lce"])]
    groups, o = {}, 0
    for g, n in sizes:
        groups[g] = np.sort(perm[o:o + n])
        o += n
    for a in GROUPS:                                   # leakage guard
        for b in GROUPS:
            if a < b:
                assert len(np.intersect1d(groups[a], groups[b])) == 0, f"split groups {a} and {b} overlap"
    q_train = np.sort(np.concatenate(list(groups.values())))
    n_test = len(load_prep("test", "s1", ["id"]))
    q_test = np.arange(n_test)
    if CFG["test_limit"]:
        q_test = np.sort(rng.choice(n_test, CFG["test_limit"], replace=False))
    gt = pd.read_csv(PATHS["ground_truth"], sep="\t", dtype=str, keep_default_na=False)
    s1_index = pd.Series(np.arange(len(s1)), index=s1.id.values)
    pool_index = pd.Series(np.arange(len(pool_ids)), index=pool_ids)
    gt = gt[gt.matched_entity_ids != ""]
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    gq = s1_index.loc[ex.source1_entity_id.values].values.astype(np.int64)
    gc = pool_index.loc[ex.m.values].values.astype(np.int64)
    n_true = np.bincount(gq, minlength=len(s1)).astype(np.int16)
    save_npz(WORK / "splits.npz", q_train=q_train, q_test=q_test, n_true=n_true, gt_q=gq, gt_c=gc,
             **{f"g_{k}": v for k, v in groups.items()})
    rep = {k: {"n": int(len(v)), "countries": pd.Series(s1.country.values[v]).value_counts().to_dict(),
               "singleton_rate": float((n_true[v] == 0).mean())} for k, v in groups.items()}
    rep["q_test"] = int(len(q_test))
    save_json(rep, WORK / "reports" / "splits.json")
    log.info(f"splits: {rep}")


def load_splits():
    z = np.load(WORK / "splits.npz")
    return {k: z[k] for k in z.files}
