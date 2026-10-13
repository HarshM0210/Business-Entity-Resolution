#!/usr/bin/env bash
# One-off: pause the runner as soon as lce_train finishes (README v2 changes the stage-2 set, so nothing
# downstream of s2set may run on the old set), and keep stage-1 test scoring going outside the runner.
cd /scratch/user4/dai/model-5
until [ -f work/state/lce_train.done ]; do sleep 10; done
echo "[$(date "+%F %T")] lce_train done -> pausing the runner for the README-v2 merge"
scripts/run.sh stop
sleep 20
cd src
for t in 16 16 16; do
  setsid -f /scratch/user4/dai/model-4/.venv/bin/python -u stages.py stage1_test_worker $t >> ../work/logs/stage1_test_helper.log 2>&1 < /dev/null
done
echo "[$(date "+%F %T")] started 3 stage1_test helper workers"
