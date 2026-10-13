#!/usr/bin/env python3
"""Experiment 2: Oracle analysis - compute label-aware optimal selection.
Runtime: ~30 minutes.

Uses existing OOF predictions from s2_prelim.pkl to compute the theoretical
upper bound by selecting the optimal subset of candidates for each S1.

Compares current vs oracle performance to determine if expensive Transformer
work is justified (threshold: >=0.001 gain).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from multiprocessing import Pool

# Setup paths
M5 = Path(__file__).parent.parent  # model-5-final-experiments -> model-5
sys.path.insert(0, str(M5 / "src"))

from config import CFG, WORK
from decide import best_k, crossfit_isotonic, decide, fold_of, macro_f05, _poisson

MODELS = WORK / "models"
from prep import load_splits
from stage2 import _load_train, attach, MULT, key


def oracle_selection(df, y_true, q_eval, n_true, gt_q, gt_c):
    """Given true labels, select optimal subset for each S1."""
    d = df.sort_values(["q", "y"], ascending=[True, False], kind="stable")
    d = d[np.isin(d.q.values, q_eval)]
    
    # For each S1, select all true matches (y==1)
    oracle_sel = d[d.y == 1][["q", "c"]].copy()
    
    # Compute F0.5
    f05, per_s1 = macro_f05(oracle_sel, q_eval, n_true, gt_q, gt_c)
    return f05, oracle_sel, per_s1


def analyze_current_vs_oracle(cfg_name, oof, vp, stack, val, sp, pool_workers):
    """Compare current selection vs oracle for a given configuration."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {cfg_name}")
    print(f"{'=' * 60}")
    
    # Current performance (calibrated)
    cal = crossfit_isotonic(oof, stack.y.values, stack.q.values, CFG["n_folds"])
    vcal = crossfit_isotonic(vp, val.y.values, val.q.values, CFG["n_folds"])
    
    # Current selection with expected F0.5 policy
    current_stack = decide(
        stack[["q", "c"]].assign(p=cal),
        "expf05", 0.0, CFG["arb_max_iter"], pool_workers
    )
    current_val = decide(
        val[["q", "c"]].assign(p=vcal),
        "expf05", 0.0, CFG["arb_max_iter"], pool_workers
    )
    
    # Oracle selection (label-aware optimal)
    oracle_stack_f05, oracle_stack_sel, _ = oracle_selection(
        stack[["q", "c", "y"]], stack.y.values,
        sp["g_stack"], sp["n_true"], sp["gt_q"], sp["gt_c"]
    )
    oracle_val_f05, oracle_val_sel, _ = oracle_selection(
        val[["q", "c", "y"]], val.y.values,
        sp["g_val"], sp["n_true"], sp["gt_q"], sp["gt_c"]
    )
    
    # Current performance
    current_stack_f05, _ = macro_f05(
        current_stack, sp["g_stack"], sp["n_true"], sp["gt_q"], sp["gt_c"]
    )
    current_val_f05, _ = macro_f05(
        current_val, sp["g_val"], sp["n_true"], sp["gt_q"], sp["gt_c"]
    )
    
    # Recoverable gain
    stack_gain = oracle_stack_f05 - current_stack_f05
    val_gain = oracle_val_f05 - current_val_f05
    
    print(f"\nStack Split:")
    print(f"  Current F0.5:  {current_stack_f05:.5f}")
    print(f"  Oracle F0.5:   {oracle_stack_f05:.5f}")
    print(f"  Recoverable:   {stack_gain:.5f} ({100*stack_gain:.3f}%)")
    
    print(f"\nValidation Split:")
    print(f"  Current F0.5:  {current_val_f05:.5f}")
    print(f"  Oracle F0.5:   {oracle_val_f05:.5f}")
    print(f"  Recoverable:   {val_gain:.5f} ({100*val_gain:.3f}%)")
    
    # Analyze where the gap is
    current_k = key(current_stack)
    oracle_k = key(oracle_stack_sel)
    
    # False negatives: oracle selected but current didn't
    fn_k = np.setdiff1d(oracle_k, current_k)
    fn = stack[np.isin(key(stack), fn_k)]
    
    # False positives: current selected but oracle didn't (i.e., y==0)
    fp_k = np.setdiff1d(current_k, oracle_k)
    fp = stack[np.isin(key(stack), fp_k)]
    
    print(f"\nError Analysis (Stack):")
    print(f"  False Negatives: {len(fn_k)} (missed true matches)")
    print(f"  False Positives: {len(fp_k)} (incorrect matches)")
    
    if len(fn) > 0:
        print(f"\nFalse Negative Statistics:")
        print(f"  Mean rank: {fn.groupby('q').p1.rank(ascending=False).mean():.1f}")
        print(f"  Mean p1: {fn.p1.mean():.4f}")
        if "ce_m" in fn.columns:
            print(f"  Mean ce_m: {fn.ce_m.mean():.4f}")
        if "ce_b" in fn.columns:
            print(f"  Mean ce_b: {fn.ce_b.mean():.4f}")
    
    return {
        "config": cfg_name,
        "stack": {
            "current": float(current_stack_f05),
            "oracle": float(oracle_stack_f05),
            "recoverable": float(stack_gain),
            "fn_count": int(len(fn_k)),
            "fp_count": int(len(fp_k)),
        },
        "val": {
            "current": float(current_val_f05),
            "oracle": float(oracle_val_f05),
            "recoverable": float(val_gain),
        },
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 2: Oracle Analysis (Label-Aware Optimal Selection)")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    sp, s2, stack, val = _load_train(with_lce=False)
    prelim = pickle.loads((MODELS / "s2_prelim.pkl").read_bytes())
    
    print(f"Stack size: {len(stack)} pairs, {len(sp['g_stack'])} S1")
    print(f"Val size: {len(val)} pairs, {len(sp['g_val'])} S1")
    
    # Analyze each configuration
    results = {}
    with Pool(CFG["n_workers"]) as pool_workers:
        for cfg_name in ["noce", "ce_nox", "ce", "ce_s"]:
            if cfg_name not in prelim:
                print(f"\nSkipping {cfg_name} (not in prelim)")
                continue
            
            result = analyze_current_vs_oracle(
                cfg_name,
                prelim[cfg_name]["oof"],
                prelim[cfg_name]["vp"],
                stack, val, sp, pool_workers
            )
            results[cfg_name] = result
    
    # Summary and decision
    print("\n" + "=" * 80)
    print("ORACLE ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("\nRecoverable Gain by Configuration:")
    print(f"{'Config':<12} {'Stack Gain':<15} {'Val Gain':<15} {'Decision'}")
    print("-" * 60)
    
    max_stack_gain = 0
    best_config = None
    
    for cfg_name, result in results.items():
        stack_gain = result["stack"]["recoverable"]
        val_gain = result["val"]["recoverable"]
        
        if stack_gain > max_stack_gain:
            max_stack_gain = stack_gain
            best_config = cfg_name
        
        # Decision thresholds
        if stack_gain >= 0.002:
            decision = "✓ Build Set Transformer"
        elif stack_gain >= 0.001:
            decision = "⚠ Minimal Transformer"
        else:
            decision = "✗ Skip Transformer"
        
        print(f"{cfg_name:<12} {stack_gain:>+.5f} ({100*stack_gain:>5.2f}%)  "
              f"{val_gain:>+.5f} ({100*val_gain:>5.2f}%)  {decision}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    
    if max_stack_gain >= 0.002:
        print(f"\n✓ PROCEED with Set Transformer implementation")
        print(f"  Best config: {best_config}")
        print(f"  Recoverable gain: {max_stack_gain:.5f} (>= 0.002 threshold)")
        print(f"  Expected runtime: 1-2 hours")
        recommendation = "full_transformer"
    elif max_stack_gain >= 0.001:
        print(f"\n⚠ Consider MINIMAL Set Transformer")
        print(f"  Best config: {best_config}")
        print(f"  Recoverable gain: {max_stack_gain:.5f} (0.001-0.002 range)")
        print(f"  Recommendation: Small hidden=128, 3 layers")
        recommendation = "minimal_transformer"
    else:
        print(f"\n✗ STOP Transformer work")
        print(f"  Maximum recoverable gain: {max_stack_gain:.5f} (< 0.001)")
        print(f"  Not worth the training time and risk")
        recommendation = "skip_transformer"
    
    # Save results
    output = {
        "summary": {
            "max_recoverable_gain": float(max_stack_gain),
            "best_config": best_config,
            "recommendation": recommendation,
        },
        "details": results,
        "thresholds": {
            "full_transformer": 0.002,
            "minimal_transformer": 0.001,
            "skip": "< 0.001",
        },
    }
    
    output_path = WORK / "reports" / "oracle_analysis.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Results saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
