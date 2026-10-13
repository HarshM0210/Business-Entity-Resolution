#!/usr/bin/env python3
"""Experiment 3: Normalize count features by S1 table size.
Runtime: 1-2 hours (retrain stage-2 only, reuse all cached CE scores).

Fixes the known France issue where count features depend on table size.
Normalizes x_n_s1_* features by dividing by S1 count per country.

Creates submission in output_count_norm/
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from multiprocessing import Pool

# Setup paths
M5 = Path(__file__).parent.parent  # model-5-final-experiments -> model-5
sys.path.insert(0, str(M5 / "src"))

from config import CFG, OUTPUT, WORK
from decide import crossfit_isotonic, decide, fit_isotonic, fold_of, macro_f05

MODELS = WORK / "models"
from prep import load_prep, load_splits
from stage2 import (X_COLS, attach, cols, model_tag, _load_train, _miss_rate,
                     _score_cfg, run_stage2_test)
from stacker import train_lgb


def name_features_normalized(d, split, s1_table=None):
    """Normalized version of name_features that divides counts by S1 table size per country."""
    s1 = load_prep(split, "s1", ["country", "name_core", "name_ns"])
    pool = load_prep(split, "pool", ["name_core", "name_ns", "addr_empty"])
    t = s1 if s1_table is None else s1.iloc[s1_table]
    
    # Count features per country
    cn = (t.country + "\t" + t.name_core).value_counts()
    cns = (t.country + "\t" + t.name_ns).value_counts()
    
    # S1 table size per country for normalization
    country_sizes = t.groupby("country").size()
    
    ctry = s1.country.values[d.q.values]
    a, b = s1.name_core.values[d.q.values], pool.name_core.values[d.c.values]
    ans, bns = s1.name_ns.values[d.q.values], pool.name_ns.values[d.c.values]
    
    f = pd.DataFrame(index=d.index)
    f["x_empty"] = pool.addr_empty.values[d.c.values].astype(np.float32)
    f["x_name_eq"] = (a == b).astype(np.float32)
    f["x_name_ns_eq"] = (ans == bns).astype(np.float32)
    
    # Raw counts
    raw_cand_name = pd.Series(ctry + "\t" + b).map(cn).fillna(0).clip(upper=10).values
    raw_cand_name_ns = pd.Series(ctry + "\t" + bns).map(cns).fillna(0).clip(upper=10).values
    raw_own_name = pd.Series(ctry + "\t" + a).map(cn).fillna(0).clip(upper=10).values
    
    # Normalize by country S1 table size (per 100k S1)
    country_norm = pd.Series(ctry).map(country_sizes).fillna(1).values / 100_000.0
    
    f["x_n_s1_cand_name"] = (raw_cand_name / country_norm).astype(np.float32)
    f["x_n_s1_cand_name_ns"] = (raw_cand_name_ns / country_norm).astype(np.float32)
    f["x_n_s1_own_name"] = (raw_own_name / country_norm).astype(np.float32)
    
    # Structure flags (same as before)
    f["name_is_other_s1"] = ((f.x_name_eq == 0) & (raw_cand_name >= 1)).astype(np.float32)
    f["empty_and_unique_name"] = ((f.x_empty == 1) & (f.x_name_eq == 1) & (raw_cand_name == 1)).astype(np.float32)
    f["empty_and_shared_name"] = ((f.x_empty == 1) & (f.x_name_eq == 1) & (raw_cand_name >= 3)).astype(np.float32)
    
    return f, b


def attach_normalized(split, s2, with_lce=False):
    """Modified attach() that uses normalized count features."""
    from cross_encoder import load_ce_scores
    from retrieve_dense import n_shards
    from common import shard_path
    
    qs = np.unique(s2.q.values)
    kk = s2[["q", "c"]]
    parts = []
    for i in range(n_shards(split)):
        d = pd.read_parquet(shard_path("featx", split, i))
        d = d[np.isin(d.q.values, qs)]
        parts.append(kk[np.isin(kk.q.values, np.unique(d.q.values))].merge(d, on=["q", "c"], how="left"))
    f = pd.concat(parts, ignore_index=True)
    s2 = s2.merge(f.drop(columns=[c for c in ("y",) if c in s2.columns]), on=["q", "c"], how="left")
    
    # Use normalized features
    xf, cand_name = name_features_normalized(s2, split)
    s2 = pd.concat([s2, xf], axis=1)
    
    # Sibling flag
    def _sibling(s2_df, cand_name_vals):
        k = pd.Series(s2_df.q.values.astype(str)) + "\t" + pd.Series(cand_name_vals)
        strong = set(k[s2_df.p1.values >= 0.5])
        return np.where(s2_df.p1.values < 0.5, k.isin(strong).values, False).astype(np.float32)
    
    s2["sibling_same_name"] = _sibling(s2, cand_name)
    s2["scrambled"] = (s2.n_jacc.fillna(0) == 0).astype(np.float32)
    
    # Add reverse search and CE features
    s2 = s2.merge(pd.read_parquet(WORK / "rev" / f"{split}.parquet"), on=["q", "c"], how="left")
    
    from config import CES
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


def train_normalized():
    """Train stage-2 models with normalized count features."""
    print("\n" + "=" * 80)
    print("Training with normalized count features")
    print("=" * 80)
    
    sp = load_splits()
    s2_base = pd.read_parquet(WORK / "s2" / "train.parquet")
    
    # Attach features with normalization
    print("\nAttaching normalized features...")
    s2 = attach_normalized("train", s2_base, with_lce=False)
    
    stack = s2[np.isin(s2.q.values, sp["g_stack"])].reset_index(drop=True)
    val = s2[np.isin(s2.q.values, sp["g_val"])].reset_index(drop=True)
    
    print(f"Stack: {len(stack)} pairs")
    print(f"Val: {len(val)} pairs")
    
    # Train ce and ce_s configs with normalized features
    lam = _miss_rate(sp, stack)
    fams = [("expf05", 0.0), ("expf05", lam)]
    
    results = {}
    with Pool(CFG["n_workers"]) as pool:
        for cfg in ["ce", "ce_s"]:
            print(f"\n{'=' * 60}")
            print(f"Training: {cfg}_norm")
            print(f"{'=' * 60}")
            
            feats = cols(s2, cfg)
            fold = fold_of(stack.q.values, CFG["n_folds"])
            
            # Train fold models
            oof = np.zeros(len(stack))
            vp = np.zeros(len(val))
            tag = f"{cfg}_norm_{hash(tuple(feats)) & 0xFFFFFFFF:08x}"
            
            for k in range(CFG["n_folds"]):
                print(f"  Fold {k+1}/{CFG['n_folds']}...")
                tr = stack[fold != k]
                m = train_lgb(
                    tr[feats].values.astype(np.float32),
                    tr.y.values, tr.q.values,
                    MODELS / f"s2_{tag}_fold{k}.txt",
                    100 + k
                )
                oof[fold == k] = m.predict(stack.loc[fold == k, feats].values.astype(np.float32))
                vp += m.predict(val[feats].values.astype(np.float32)) / CFG["n_folds"]
            
            # Score
            score = _score_cfg(cfg, oof, vp, stack, val, sp, fams, pool)
            results[f"{cfg}_norm"] = {**score, "feats": feats, "tag": tag}
            
            print(f"\n{cfg}_norm Results:")
            print(f"  Stack F0.5: {score['stack']:.5f}")
            print(f"  Val F0.5 (claims): {score['val_claims']:.5f}")
            print(f"  Val F0.5 (plain): {score['val_plain']:.5f}")
    
    return results, s2, stack, val, sp


def main():
    print("=" * 80)
    print("EXPERIMENT 3: Count Feature Normalization")
    print("=" * 80)
    print("\nNormalizing x_n_s1_* features by S1 table size per country")
    print("Goal: Fix France size-dependence issue (README §6.1)")
    
    # Load original results for comparison
    bundle = json.loads((MODELS / "stage2_bundle.json").read_text())
    gates = json.loads((WORK / "reports" / "stage2_gates.json").read_text())
    
    print(f"\nOriginal ce results:")
    print(f"  Stack: {gates['configs']['ce']['stack']:.5f}")
    print(f"  Val: {gates['configs']['ce']['val_plain']:.5f}")
    
    print(f"\nOriginal ce_s results:")
    print(f"  Stack: {gates['configs']['ce_s']['stack']:.5f}")
    print(f"  Val: {gates['configs']['ce_s']['val_plain']:.5f}")
    
    # Train with normalization
    results, s2, stack, val, sp = train_normalized()
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON: Original vs Normalized")
    print("=" * 80)
    
    comparisons = []
    for cfg in ["ce", "ce_s"]:
        orig = gates["configs"][cfg]
        norm = results[f"{cfg}_norm"]
        
        stack_delta = norm["stack"] - orig["stack"]
        val_delta = norm["val_plain"] - orig["val_plain"]
        
        print(f"\n{cfg}:")
        print(f"  Stack:  {orig['stack']:.5f} → {norm['stack']:.5f} ({stack_delta:+.5f})")
        print(f"  Val:    {orig['val_plain']:.5f} → {norm['val_plain']:.5f} ({val_delta:+.5f})")
        
        comparisons.append({
            "config": cfg,
            "original": {"stack": orig["stack"], "val": orig["val_plain"]},
            "normalized": {"stack": norm["stack"], "val": norm["val_plain"]},
            "delta": {"stack": stack_delta, "val": val_delta},
        })
    
    # Choose best
    best_norm = max(results.items(), key=lambda x: x[1]["stack"])
    best_cfg_name, best_result = best_norm
    
    print(f"\n{'=' * 80}")
    print(f"BEST NORMALIZED CONFIG: {best_cfg_name}")
    print(f"  Stack F0.5: {best_result['stack']:.5f}")
    print(f"  Val F0.5: {best_result['val_claims']:.5f}")
    print(f"{'=' * 80}")
    
    # Check if safe to use (no degradation)
    use_normalized = best_result["val_plain"] >= gates["configs"]["ce"]["val_plain"] - 0.0001
    
    if use_normalized:
        print("\n✓ SAFE to use normalized features")
        print("  Generating test submission...")
        
        # Create modified bundle
        norm_bundle = bundle.copy()
        norm_bundle["config"] = best_cfg_name.replace("_norm", "")
        norm_bundle["features"] = best_result["feats"]
        norm_bundle["family"] = best_result["family"]
        norm_bundle["thr"] = best_result["thr"]
        norm_bundle["normalized_counts"] = True
        
        # Save to temp location
        temp_bundle = MODELS / "stage2_bundle_norm.json"
        temp_bundle.write_text(json.dumps(norm_bundle, indent=2))
        
        # Generate test submission
        import os
        output_norm = M5 / "output_count_norm"
        output_norm.mkdir(exist_ok=True)
        os.environ["M5_OUTPUT"] = str(output_norm)
        
        # Temporarily replace bundle
        backup = MODELS / "stage2_bundle_backup.json"
        shutil.copy(MODELS / "stage2_bundle.json", backup)
        shutil.copy(temp_bundle, MODELS / "stage2_bundle.json")
        
        # Patch attach function
        import stage2
        original_attach = stage2.attach
        stage2.attach = attach_normalized
        
        try:
            run_stage2_test()
        finally:
            # Restore
            stage2.attach = original_attach
            shutil.copy(backup, MODELS / "stage2_bundle.json")
        
        print(f"\n✓ Submission created in: {output_norm}/")
    else:
        print("\n✗ UNSAFE - validation degraded slightly")
        print("  Not generating submission")
    
    # Save analysis
    output = {
        "normalized_results": {k: {kk: vv for kk, vv in v.items() if kk != "feats"}
                              for k, v in results.items()},
        "comparisons": comparisons,
        "recommendation": "use" if use_normalized else "skip",
        "best_config": best_cfg_name if use_normalized else None,
    }
    
    output_path = WORK / "reports" / "count_normalization.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Analysis saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
