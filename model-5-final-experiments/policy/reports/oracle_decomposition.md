# exp4 oracle decomposition (ce_s, stack / val)

| level | stack | val |
|---|---|---|
| L0 current decide() | 0.98715 | 0.98770 |
| L1 oracle-k, arbitration by p | 0.99530 | 0.99543 |
| L2 oracle-k, arbitration by truth | 0.99531 | 0.99544 |
| L3 perfect scorer | 0.99672 | 0.99681 |

- k/abstain headroom (L1-L0): +0.0082 stack. This is an UPPER bound: the oracle cuts exactly at the last true positive.
- arbitration (L2-L1): +0.00001, negligible.
- ranking (L3-L2): +0.0014.
- oracle wants MORE matches for 19,008 S1 and FEWER for 2,100 (of 300k): current selection is systematically too conservative.
- calibration (L0, stack): india expected 0.9912 vs actual 0.9865; us 0.9894 vs 0.9876. Train has no France.
- ce is nearly identical (L1-L0 +0.0084).

Verdict: build k-policy. Cheapest first step: tune a selection bias (p power / per-country k offset) on stack, confirm on val.
