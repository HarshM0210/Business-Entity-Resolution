# policy/ — exp4 oracle decomposition (2026-09-27)

- exp4_decomposition.py: splits the stage-2 F0.5 gap into k/abstain-policy (L1-L0), arbitration (L2-L1),
  ranking/scorer (L3-L2). Reads model-5/work only; writes reports/ and logs/ here.
- Run: /scratch/user4/dai/model-4/.venv/bin/python -u exp4_decomposition.py > logs/exp4.log 2>&1

## Submission file provenance (exp1 bug)
exp1 set M5_OUTPUT after config.OUTPUT was imported, so the ce_s submission went to model-5/output/ (16:28),
overwriting the ce submission there, and overwrote work/reports/test_report.json.
- model-5/output/            -> ce_s (16:28)
- model-5/output_ce_s/       -> copy of the ce_s files + test_report_ce_s.json (17:12)
- model-5/backup_ready_1343/ -> ce baseline submission (13:41) + its reports
Nothing was deleted or restored.
