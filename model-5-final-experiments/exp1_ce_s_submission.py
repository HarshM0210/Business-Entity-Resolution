#!/usr/bin/env python3
"""Experiment 1: Generate ce_s submission using existing trained models.
Runtime: ~10 minutes. All models already exist in work/models/s2_ce_s_*.txt

This modifies stage2_bundle.json to use ce_s and reruns stage2_test only.
Creates submission in output_ce_s/
"""
import json
import shutil
import sys
from pathlib import Path

# Setup paths
M5 = Path(__file__).parent.parent  # model-5-final-experiments -> model-5
sys.path.insert(0, str(M5 / "src"))

from config import OUTPUT, WORK
from stage2 import run_stage2_test

MODELS = WORK / "models"


def main():
    print("=" * 80)
    print("EXPERIMENT 1: ce_s Submission (using existing trained models)")
    print("=" * 80)
    
    # Load existing bundle
    bundle_path = MODELS / "stage2_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    
    print(f"\nOriginal config: {bundle['config']}")
    print(f"  Stack F0.5: {bundle['stack_f05']:.5f}")
    print(f"  Val F0.5: {bundle['val_f05']:.5f}")
    
    # Load ce_s results from gates report
    gates_path = WORK / "reports" / "stage2_gates.json"
    gates = json.loads(gates_path.read_text())
    ce_s_results = gates["configs"]["ce_s"]
    
    print(f"\nce_s config:")
    print(f"  Stack F0.5: {ce_s_results['stack']:.5f} (+{ce_s_results['stack'] - bundle['stack_f05']:.5f})")
    print(f"  Val F0.5: {ce_s_results['val_claims']:.5f} (+{ce_s_results['val_claims'] - bundle['val_f05']:.5f})")
    
    # Backup original
    backup_path = MODELS / "stage2_bundle_original.json"
    if not backup_path.exists():
        shutil.copy(bundle_path, backup_path)
        print(f"\nBacked up original bundle to {backup_path}")
    
    # Modify bundle to use ce_s
    bundle["config"] = "ce_s"
    bundle["family"] = ce_s_results["family"]
    bundle["thr"] = ce_s_results["thr"]
    
    # Update features to include S_COLS (structure features)
    # Load prelim.pkl to get the actual feature list
    import pickle
    prelim_path = MODELS / "s2_prelim.pkl"
    prelim = pickle.loads(prelim_path.read_bytes())
    bundle["features"] = prelim["ce_s"]["feats"]
    
    print(f"\nUpdating bundle to use ce_s with {len(bundle['features'])} features")
    
    # Save modified bundle
    bundle_path.write_text(json.dumps(bundle, indent=2))
    print(f"Modified {bundle_path}")
    
    # Redirect output to output_ce_s/
    import os
    original_output = str(OUTPUT)
    output_ce_s = M5 / "output_ce_s"
    output_ce_s.mkdir(exist_ok=True)
    os.environ["M5_OUTPUT"] = str(output_ce_s)
    
    print(f"\nOutput directory: {output_ce_s}")
    print("\nRunning stage2_test with ce_s configuration...")
    print("-" * 80)
    
    # Run test
    run_stage2_test()
    
    print("-" * 80)
    print("\n✓ EXPERIMENT 1 COMPLETE")
    print(f"\nSubmission files created in: {output_ce_s}/")
    print(f"  - matching_results.tsv")
    print(f"  - candidate_pairs.tsv")
    
    # Show test report
    test_report_path = WORK / "reports" / "test_report.json"
    if test_report_path.exists():
        report = json.loads(test_report_path.read_text())
        print(f"\nTest statistics:")
        for country, stats in report["test_pred_stats"].items():
            print(f"  {country}: {stats['n_s1']} S1, {stats['mean_pred']:.2f} avg matches, "
                  f"expected F0.5 = {stats['expected_f05']:.3f}")
        print(f"\nTotal predicted pairs: {report['n_pred_pairs']}")
        
        if "validator" in report:
            if report["validator"]["returncode"] == 0:
                print("✓ Validator: PASS")
            else:
                print(f"✗ Validator failed: {report['validator']['stderr']}")
    
    # Restore original bundle
    if backup_path.exists():
        shutil.copy(backup_path, bundle_path)
        print(f"\nRestored original bundle from backup")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
