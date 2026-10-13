#!/usr/bin/env bash
# One extra featx_test worker (joins the claimed shard queue) once ename_test is done and featx_test has started.
cd /scratch/user4/dai/model-5/src
until [ -f ../work/state/ename_test.done ] && [ -f ../work/state/running/featx_test ]; do sleep 20; done
sleep 60
[ -f ../work/state/featx_test.done ] && exit 0
echo "[$(date "+%F %T")] starting featx_test helper"
exec /scratch/user4/dai/model-4/.venv/bin/python -u features.py featx test
