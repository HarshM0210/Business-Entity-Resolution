#!/usr/bin/env bash
# Starts one extra feat_test worker (joins the claimed shard queue) once feat_train has released its CPUs.
cd /scratch/user4/dai/model-5/src
until [ -f ../work/state/feat_train.done ]; do sleep 30; done
[ -f ../work/state/feat_test.done ] && exit 0
echo "[$(date "+%F %T")] starting feat_test helper"
exec /scratch/user4/dai/model-4/.venv/bin/python -u features.py test
