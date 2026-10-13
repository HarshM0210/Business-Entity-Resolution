#!/usr/bin/env bash
# Adds an mDeBERTa test-inference worker on GPU 1 (same claimed shard queue) once lce_infer_train frees that card.
cd /scratch/user4/dai/model-5/src
until [ -f ../work/state/lce_infer_train.done ]; do sleep 30; done
[ -f ../work/state/ce_infer_test_m.done ] && exit 0
echo "[$(date "+%F %T")] starting ce_m test helper on GPU 1"
export HF_HOME=/scratch/user4/mtp/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 M5_ASSIGNED_GPUS=1
exec /scratch/user4/dai/model-4/.venv/bin/python -u stages.py ce_infer_worker m test bcbb1a04f1
