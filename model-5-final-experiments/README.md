# Model-5 Final Experiments

**IMPORTANT:** This folder contains experiment scripts that work with the model-5 pipeline **WITHOUT modifying the original model-5 directory**.

## Location
```
/scratch/user4/dai/
├── model-5/                           ← UNTOUCHED (original pipeline)
│   ├── src/                          
│   ├── work/                         
│   ├── output/                       ← Original submission
│   └── ...
└── model-5/model-5-final-experiments/ ← NEW (this folder)
    ├── exp1_ce_s_submission.py       
    ├── exp2_oracle_analysis.py       
    ├── exp3_count_normalization.py   
    ├── run_experiments.py            
    └── run_fast_experiments.py       
```

## Quick Start

```bash
# Navigate to experiment directory
cd /scratch/user4/dai/model-5/model-5-final-experiments

# Activate Python environment
source /scratch/user4/dai/bge-out/.venv/bin/activate

# Option 1: Fast track (40 minutes, no training)
python run_experiments.py --skip-count-norm

# Option 2: Full experiments (2-3 hours, includes training)
python run_experiments.py
```

## What This Does

All scripts read from `../work/` and `../src/` (the original model-5 directory) but create new outputs in separate directories:

- **Reads from:** `model-5/work/`, `model-5/src/`, etc. (never modified)
- **Writes to:** `model-5/output_ce_s/`, `model-5/output_count_norm/`, `model-5/work/reports/*.json`

## Experiments

### 1. ce_s Submission (10 min)
Uses existing trained models, creates `../output_ce_s/matching_results.tsv`

### 2. Oracle Analysis (30 min)
Computes theoretical upper bound, writes `../work/reports/oracle_analysis.json`

### 3. Count Normalization (1-2 hr)
Retrains stage-2 with normalized features, creates `../output_count_norm/matching_results.tsv` if safe

## Files in This Directory

- `exp1_ce_s_submission.py` - Experiment 1 script
- `exp2_oracle_analysis.py` - Experiment 2 script
- `exp3_count_normalization.py` - Experiment 3 script
- `run_experiments.py` - Master runner for all experiments
- `run_fast_experiments.py` - Fast track runner (skip training)
- `EXPERIMENTS.md` - Detailed documentation
- `QUICKSTART.txt` - Quick reference guide
- `README.md` - This file

## Safety

- Original `model-5/` directory is **never modified**
- Original `model-5/output/` submission is **untouched**
- All new outputs go to separate directories
- Backups are created before any modifications
- Validation runs automatically on all submissions

## Support

See `EXPERIMENTS.md` for full documentation or `QUICKSTART.txt` for a condensed guide.
