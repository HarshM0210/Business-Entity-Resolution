# Stage-2 validation report (README v2)

- chosen on stack: **ce_s**, decision **expf05** (param 0.04)
- stack OOF F0.5 (with competitor claims): **0.96610**  (v2 line: 0.98543; switch rule +0.001: not met)
- validation macro F0.5: **0.96602** with competitor claims, 0.96602 plain
- by country: india: 0.96098, us: 0.96934
- ladder (stack / val): noce 0.95062/0.95493; ce_nox 0.95954/0.96081; ce 0.96278/0.96436; ce_s 0.96610/0.96602; ce_lce:ce_s 0.96540/0.96660; lce_only 0.85390/0.85546
- expected vs actual val F0.5 by country: {'india': {'expected': 0.962, 'actual': 0.9609808929604043}, 'us': {'expected': 0.97011, 'actual': 0.9693399193486429}}
- count features on a ~260k-per-country S1 table (README §2.5): {'val_plain_full_table': 0.9660213858725124, 'val_plain_260k_table': 0.9626121133687552, 'delta': -0.0034092725037572214, 'flag': True}
- loss decomposition (val): {'rejected_in_stage2': 0.01235, 'dropped_before_stage2': 0.0013, 'never_retrieved': 0.00522, 'fp_distractor': 0.00955, 'fp_owner_scored': 0.0, 'fp_owner_not_scored': 0.00587}
- predicted matches per S1 (val): {'india': {'n_s1': 794, 'empty_rate': 0.0655, 'mean_pred': 3.262, 'dist': {'0.0': 0.0655, '1.0': 0.0844, '2.0': 0.1914, '3.0': 0.2141, '4.0': 0.204, '5.0': 0.1474, '6.0': 0.0932}}, 'us': {'n_s1': 1206, 'empty_rate': 0.0481, 'mean_pred': 3.283, 'dist': {'0.0': 0.0481, '1.0': 0.0738, '2.0': 0.1816, '3.0': 0.2678, '4.0': 0.2106, '5.0': 0.1368, '6.0': 0.0813}}}
