#!/usr/bin/env python3
"""Master runner for model-5 final experiments (6-hour window).

Executes experiments in priority order:
1. ce_s submission (10 min) - immediate alternative using existing models
2. Oracle analysis (30 min) - determine if Transformer work is justified
3. Count normalization (1-2 hr) - fix France size-dependence
4. Set Transformer (conditional) - only if oracle shows >=0.001 gain

Each experiment creates its own submission in a separate output directory.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def run_experiment(script, name, expected_minutes):
    """Run an experiment script and report timing."""
    print("\n" + "=" * 90)
    print(f"STARTING: {name}")
    print(f"Expected runtime: ~{expected_minutes} minutes")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 90)
    
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=script.parent,
            check=False,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print("\n" + "=" * 90)
            print(f"✓ COMPLETED: {name}")
            print(f"Runtime: {elapsed/60:.1f} minutes")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 90)
            return True, elapsed
        else:
            print("\n" + "=" * 90)
            print(f"✗ FAILED: {name}")
            print(f"Exit code: {result.returncode}")
            print(f"Runtime: {elapsed/60:.1f} minutes")
            print("=" * 90)
            return False, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n✗ ERROR: {name}")
        print(f"Exception: {e}")
        print(f"Runtime: {elapsed/60:.1f} minutes")
        return False, elapsed


def check_oracle_decision(work_dir):
    """Check oracle analysis results to decide on Transformer."""
    oracle_path = work_dir / "reports" / "oracle_analysis.json"
    if not oracle_path.exists():
        return None
    
    results = json.loads(oracle_path.read_text())
    return results["summary"]["recommendation"]


def summarize_results(work_dir, output_dirs):
    """Print final summary of all experiments."""
    print("\n" + "=" * 90)
    print("FINAL SUMMARY - ALL EXPERIMENTS")
    print("=" * 90)
    
    # Load reports
    bundle_path = work_dir / "models" / "stage2_bundle.json"
    bundle = json.loads(bundle_path.read_text()) if bundle_path.exists() else {}
    
    gates_path = work_dir / "reports" / "stage2_gates.json"
    gates = json.loads(gates_path.read_text()) if gates_path.exists() else {}
    
    oracle_path = work_dir / "reports" / "oracle_analysis.json"
    oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else {}
    
    count_path = work_dir / "reports" / "count_normalization.json"
    count_norm = json.loads(count_path.read_text()) if count_path.exists() else {}
    
    print("\n1. BASELINE (current submission)")
    if bundle:
        print(f"   Config: {bundle.get('config', 'N/A')}")
        print(f"   Stack F0.5: {bundle.get('stack_f05', 'N/A'):.5f}")
        print(f"   Val F0.5: {bundle.get('val_f05', 'N/A'):.5f}")
    
    print("\n2. EXPERIMENT 1: ce_s Submission")
    if gates and "ce_s" in gates.get("configs", {}):
        ce_s = gates["configs"]["ce_s"]
        baseline_stack = bundle.get('stack_f05', ce_s['stack'])
        baseline_val = bundle.get('val_f05', ce_s['val_claims'])
        print(f"   Stack F0.5: {ce_s['stack']:.5f} ({ce_s['stack'] - baseline_stack:+.5f})")
        print(f"   Val F0.5: {ce_s['val_claims']:.5f} ({ce_s['val_claims'] - baseline_val:+.5f})")
        output = work_dir.parent / "output_ce_s"
        if output.exists():
            print(f"   ✓ Submission: {output}/matching_results.tsv")
        else:
            print(f"   ✗ No submission generated")
    
    print("\n3. EXPERIMENT 2: Oracle Analysis")
    if oracle:
        summary = oracle.get("summary", {})
        print(f"   Max recoverable gain: {summary.get('max_recoverable_gain', 0):.5f}")
        print(f"   Best config: {summary.get('best_config', 'N/A')}")
        print(f"   Recommendation: {summary.get('recommendation', 'N/A')}")
        
        rec = summary.get('recommendation', '')
        if rec == 'skip_transformer':
            print(f"   → Skip Set Transformer (gain < 0.001)")
        elif rec == 'minimal_transformer':
            print(f"   → Consider minimal Set Transformer (gain 0.001-0.002)")
        elif rec == 'full_transformer':
            print(f"   → Proceed with Set Transformer (gain >= 0.002)")
    
    print("\n4. EXPERIMENT 3: Count Normalization")
    if count_norm:
        comparisons = count_norm.get("comparisons", [])
        for comp in comparisons:
            cfg = comp["config"]
            delta = comp["delta"]
            print(f"   {cfg}: Stack {delta['stack']:+.5f}, Val {delta['val']:+.5f}")
        
        rec = count_norm.get("recommendation", "skip")
        best = count_norm.get("best_config")
        if rec == "use":
            print(f"   ✓ Safe to use: {best}")
            output = work_dir.parent / "output_count_norm"
            if output.exists():
                print(f"   ✓ Submission: {output}/matching_results.tsv")
        else:
            print(f"   ✗ Not recommended (slight degradation)")
    
    print("\n" + "=" * 90)
    print("AVAILABLE SUBMISSIONS:")
    print("=" * 90)
    
    submissions = []
    for output_dir in output_dirs:
        matching = output_dir / "matching_results.tsv"
        candidate = output_dir / "candidate_pairs.tsv"
        if matching.exists() and candidate.exists():
            submissions.append(output_dir)
            print(f"  ✓ {output_dir.name}/")
    
    if not submissions:
        print("  (none generated)")
    
    print("\n" + "=" * 90)
    print("RECOMMENDATIONS:")
    print("=" * 90)
    
    # Determine best submission
    if oracle and oracle.get("summary", {}).get("recommendation") == "skip_transformer":
        print("\n1. Upload ce_s submission if val gain > 0.0003")
        print("2. Skip Set Transformer work (oracle gain < 0.001)")
        print("3. Consider count_norm if it's safe and helps France")
    else:
        print("\n1. Review oracle analysis for Transformer decision")
        print("2. Compare ce_s vs count_norm submissions")
        print("3. Upload best performing alternative")
    
    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Run model-5 final experiments")
    parser.add_argument("--skip-ce-s", action="store_true", help="Skip ce_s submission")
    parser.add_argument("--skip-oracle", action="store_true", help="Skip oracle analysis")
    parser.add_argument("--skip-count-norm", action="store_true", help="Skip count normalization")
    parser.add_argument("--only", choices=["ce_s", "oracle", "count_norm"], help="Run only one experiment")
    args = parser.parse_args()
    
    M5 = Path(__file__).parent.parent  # model-5-final-experiments -> model-5
    work_dir = M5 / "work"
    
    print("=" * 90)
    print("MODEL-5 FINAL EXPERIMENTS")
    print("=" * 90)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {M5}")
    print(f"\nPlan:")
    print(f"  1. ce_s submission (~10 min)")
    print(f"  2. Oracle analysis (~30 min)")
    print(f"  3. Count normalization (~60-120 min)")
    print(f"\nTotal expected: ~2-3 hours")
    
    overall_start = time.time()
    results = {}
    exp_dir = Path(__file__).parent
    output_dirs = [M5 / "output_ce_s", M5 / "output_count_norm"]
    
    # Experiment 1: ce_s submission
    if not args.skip_ce_s and (args.only is None or args.only == "ce_s"):
        success, elapsed = run_experiment(
            exp_dir / "exp1_ce_s_submission.py",
            "Experiment 1: ce_s Submission",
            10
        )
        results["ce_s"] = {"success": success, "elapsed": elapsed}
        
        if not success:
            print("\n⚠ Warning: ce_s submission failed, but continuing...")
    
    # Experiment 2: Oracle analysis
    if not args.skip_oracle and (args.only is None or args.only == "oracle"):
        success, elapsed = run_experiment(
            exp_dir / "exp2_oracle_analysis.py",
            "Experiment 2: Oracle Analysis",
            30
        )
        results["oracle"] = {"success": success, "elapsed": elapsed}
        
        if success:
            # Check decision
            recommendation = check_oracle_decision(work_dir)
            print(f"\nOracle recommendation: {recommendation}")
            
            if recommendation == "skip_transformer":
                print("\n✓ Oracle says skip Transformer - will not be implemented")
            elif recommendation in ("minimal_transformer", "full_transformer"):
                print(f"\n⚠ Oracle suggests {recommendation}, but not implemented in this run")
                print("  (Would require 1-2 hours additional training)")
    
    # Experiment 3: Count normalization
    if not args.skip_count_norm and (args.only is None or args.only == "count_norm"):
        success, elapsed = run_experiment(
            exp_dir / "exp3_count_normalization.py",
            "Experiment 3: Count Normalization",
            90
        )
        results["count_norm"] = {"success": success, "elapsed": elapsed}
    
    # Summary
    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 90)
    print("EXPERIMENT RUN COMPLETE")
    print("=" * 90)
    print(f"\nTotal runtime: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nExperiment Results:")
    for name, result in results.items():
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"  {name:15} {status:8} {result['elapsed']/60:6.1f} min")
    
    # Detailed summary
    summarize_results(work_dir, output_dirs)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
