#!/usr/bin/env python3
"""Fast track experiments: ce_s + oracle only (40 minutes total).

Runs the two fastest experiments that require no training:
1. ce_s submission (10 min) - uses existing models
2. Oracle analysis (30 min) - analyzes OOF predictions

Skips count normalization (1-2 hours training).
"""
import subprocess
import sys
from pathlib import Path


def main():
    exp_dir = Path(__file__).parent
    M5 = exp_dir.parent  # model-5-final-experiments -> model-5
    
    print("=" * 80)
    print("FAST TRACK EXPERIMENTS (40 minutes)")
    print("=" * 80)
    print("\n1. ce_s submission (~10 min)")
    print("2. Oracle analysis (~30 min)")
    print("\nSkipping count normalization to save time.")
    print("\n" + "=" * 80)
    
    # Run master script with skips
    result = subprocess.run(
        [sys.executable, str(exp_dir / "run_experiments.py"), "--skip-count-norm"],
        cwd=M5
    )
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
