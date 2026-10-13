# Model-5 Final Experiments

Quick experiments to maximize F0.5 gain in the remaining 6-hour window.

## Quick Start

```bash
cd /scratch/user4/dai/model-5/model-5-final-experiments

# Run all experiments (2-3 hours total)
python run_experiments.py

# Or run specific experiments only
python run_experiments.py --only ce_s        # 10 minutes
python run_experiments.py --only oracle      # 30 minutes
python run_experiments.py --only count_norm  # 1-2 hours

# Fast track: just ce_s + oracle (40 minutes)
python run_fast_experiments.py
```

## Experiments

### 1. ce_s Submission (10 minutes) ✓ FASTEST
- **What**: Generate submission using existing ce_s models
- **Why**: ce_s already trained and scored 0.98715 stack / 0.98770 val
- **Gain**: +0.00028 stack, +0.00031 val vs current
- **Output**: `output_ce_s/matching_results.tsv`
- **Run**: `python exp1_ce_s_submission.py`

### 2. Oracle Analysis (30 minutes)
- **What**: Compute label-aware optimal selection on OOF candidates
- **Why**: Determines if Set Transformer work is justified
- **Decision**: 
  - Gain >= 0.002: Build full Transformer
  - Gain 0.001-0.002: Minimal Transformer
  - Gain < 0.001: Skip Transformer
- **Output**: `work/reports/oracle_analysis.json`
- **Run**: `python exp2_oracle_analysis.py`

### 3. Count Normalization (1-2 hours)
- **What**: Normalize count features by S1 table size per country
- **Why**: Fixes France size-dependence issue (README §6.1)
- **Method**: Divide x_n_s1_* features by country S1 count / 100k
- **Safety**: Only generates submission if val doesn't degrade
- **Output**: `output_count_norm/matching_results.tsv` (if safe)
- **Run**: `python exp3_count_normalization.py`

### 4. Set Transformer (NOT IMPLEMENTED)
- **When**: Only if oracle shows >= 0.001 recoverable gain
- **Time**: Would require 1-2 hours additional training
- **Status**: Deferred - evaluate after oracle analysis

## Current Baseline

- Config: `ce`
- Stack F0.5: 0.98687
- Val F0.5: 0.98739
- Submission: `output/matching_results.tsv`

## Expected Outcomes

### ce_s (10 min)
- Immediate alternative submission
- Small but consistent gain
- Zero training cost (models exist)
- **Recommendation**: Always run this

### Oracle Analysis (30 min)
- Theoretical upper bound
- Informs Transformer decision
- No training required
- **Recommendation**: Run to guide strategy

### Count Normalization (1-2 hr)
- Addresses known France issue
- Retrains stage-2 only (fast)
- May improve generalization
- **Recommendation**: Run if time permits

## Priority Order

1. **ce_s** (10 min) - Immediate win, zero risk
2. **Oracle** (30 min) - Strategic decision point
3. **Count norm** (1-2 hr) - If time allows and oracle says skip Transformer

## Files Created

- `exp1_ce_s_submission.py` - ce_s submission script
- `exp2_oracle_analysis.py` - Oracle analysis script
- `exp3_count_normalization.py` - Count normalization script
- `run_experiments.py` - Master runner (all experiments)
- `run_fast_experiments.py` - Fast track (ce_s + oracle only)

## Results Location

- Submissions: `output_ce_s/`, `output_count_norm/`
- Reports: `work/reports/oracle_analysis.json`, `work/reports/count_normalization.json`
- Models: `work/models/s2_*_norm_*.txt` (if count norm runs)

## Environment

- Python: Use same as main pipeline (`M5_PY` or model-4 venv)
- GPU: Not required (stage-2 only, LightGBM on CPU)
- RAM: ~32GB should be sufficient
- Disk: ~10GB for new models/outputs

## Validation

All submissions automatically run through the validator with `--check-ids`.
Check `work/reports/test_report.json` for validation results.

## Rollback

Original `stage2_bundle.json` is backed up before modifications.
All experiments write to separate output directories.
Original `output/` is never modified.
