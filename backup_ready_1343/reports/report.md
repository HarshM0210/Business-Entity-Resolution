# Stage-2 validation report (README v2)

- chosen on stack: **ce**, decision **expf05** (param 0.0)
- stack OOF F0.5 (with competitor claims): **0.98687**  (v2 line: 0.98543; switch rule +0.001: PASS)
- validation macro F0.5: **0.98739** with competitor claims, 0.98731 plain
- by country: india: 0.98728, us: 0.98747
- ladder (stack / val): noce 0.97638/0.97767; ce_nox 0.98403/0.98447; ce 0.98687/0.98739; ce_s 0.98715/0.98770; ce_lce:ce 0.98767/0.98794; lce_only 0.98153/0.98161
- expected vs actual val F0.5 by country: {'india': {'expected': 0.99083, 'actual': 0.9872819915698673}, 'us': {'expected': 0.98878, 'actual': 0.9874676146346176}}
- count features on a ~260k-per-country S1 table (README §2.5): {'val_plain_full_table': 0.9873092268011118, 'val_plain_260k_table': 0.9824879387483545, 'delta': -0.004821288052757278, 'flag': True}
- loss decomposition (val): {'rejected_in_stage2': 0.0072, 'dropped_before_stage2': 0.00021, 'never_retrieved': 0.00317, 'fp_distractor': 0.00081, 'fp_owner_scored': 0.00012, 'fp_owner_not_scored': 0.00126}
- predicted matches per S1 (val): {'india': {'n_s1': 40206, 'empty_rate': 0.0583, 'mean_pred': 3.351, 'dist': {'0.0': 0.0583, '1.0': 0.063, '2.0': 0.182, '3.0': 0.2458, '4.0': 0.2124, '5.0': 0.1376, '6.0': 0.1009}}, 'us': {'n_s1': 59794, 'empty_rate': 0.0574, 'mean_pred': 3.34, 'dist': {'0.0': 0.0574, '1.0': 0.0632, '2.0': 0.1823, '3.0': 0.2491, '4.0': 0.2126, '5.0': 0.1391, '6.0': 0.0962}}}
