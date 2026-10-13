# business_entity_resolution — how to reproduce end to end

Data → blocking → matching → `output/matching_results.tsv` + `output/candidate_pairs.tsv`, with one command.

## Environment

- Python 3.12, CUDA 13 (PyTorch cu130 wheel); pinned versions in `requirements.txt`
  (`pip install -r requirements.txt`).
- 1–2 GPUs with ≥ 40 GB free (tested on 2 × RTX PRO 6000 96 GB), ~500 GB RAM recommended, ~600 GB disk.
- Models (downloaded once into `$HF_HOME`, all MIT / Apache-2.0):
  `microsoft/mdeberta-v3-base`, `BAAI/bge-reranker-v2-m3`.

## Inputs

The `fetch` stage downloads whatever is missing (set `HF_TOKEN` if the repos are private):

| input | Hugging Face dataset | local path (default, see `src/config.py`) |
|---|---|---|
| records + ground truth (parquet) | `Jyo-K/PopTards_Machine_Code` | `$M5_ROOT/dataset/og/{train,test}_p/` |
| e5-large embeddings | `Jyo-K/PopTard_Qwen_Machine` (`e5_multilingual_large/`) | `$M5_ROOT/dataset/e5_multilingual_large/` |
| bge-m3 embeddings | `Jyo-K/PopTards_Big_Boy` | `$M5_ROOT/dataset/bge3/` |
| e5-small embeddings | `Jyo-K/PopTards_Smol_Boy` | `$M5_ROOT/dataset/e5-small/` |

Override the locations with `M5_ROOT`, `M5_DATA`, `M5_EMB`. The embeddings are pre-computed, unit-normalised and in
parquet row order; the `manifest` stage refuses to continue if ids, row counts, norms or row alignment are wrong.

## Run

```bash
export M5_PY=/path/to/python          # the environment above
scripts/run.sh start                  # run / resume everything, detached (survives SSH drops)
scripts/run.sh status                 # per-stage state: done / running / pending / FAILED xN
scripts/run.sh logs                   # supervisor + runner log;  scripts/run.sh tail <stage>  for one stage
scripts/run.sh stop                   # stop (finished work is kept; "start" resumes)
M5_SMOKE=1 scripts/run.sh start       # ~1 h end-to-end check on a small sample (work_smoke/, output_smoke/)
```

Outputs: `output/matching_results.tsv` (leaderboard file), `output/candidate_pairs.tsv` (the stage-2 set the final
model scores), `<team>_submission.zip`, and every report under `work/reports/`.
The official validator (`utils/validate_submission.py --check-ids`) runs automatically; a failure stops the run.

## Stages (DAG, `src/run_pipeline.py`)

1. fetch → manifest → prep → splits (ce / lce / stack / val)
2. dense retrieval e5-large, bge-m3, e5-small (GPU) + lexical (CPU) + sibling hop (GPU)
3. RRF blocking with a recall gate and K chosen on stack
4. `ename` (empty-address name-only channel), added after the K cut
5. exact cosines + triangle features (GPU) → pair features → `featx` (union with ename, context recomputed)
6. F1 feature-set selection (leave-one-country-out) → stage-1 LightGBM (test scored by CPU workers) → `stage1x`
   (ename-only pairs)
7. stage-2 set (top 24 with p ≥ 0.003, plus every ename candidate) = `candidate_pairs.tsv`
8. reverse search over all S1 (GPU)
9. cross-encoders mDeBERTa + bge-reranker: trained on one GPU each, inference on every GPU from a claimed shard queue,
   scores cached by (q, c)
10. listwise cross-encoder: trained here; test inference runs only if stage 2 adopts it
11. stage 2 (`stage2_prelim` then `stage2_train`): the gated ladder chosen on stack with competitor claims →
    `stage2_test` (decision, submission, validator) → `package`

`loco` (leave-one-country-out report) runs alongside. All reports are written to `work/reports/`.

## Fail-proofing

- **Stages and shards:**
  - Every stage is its own process group and writes `work/state/<stage>.done` only on success; restarts skip finished
    stages.
  - Sharded stages write each shard atomically (temp file + rename) and resume from the first missing shard.
- **Shared work queues:**
  - Workers on any GPU (and CPU helpers) claim shards with lock files, so one queue can be drained by any number of
    processes.
  - Claims of dead processes are re-taken; claims of remote (Colab) lanes expire after 2.5 h without progress.
  - A stage waits for shards still held by live helpers before declaring itself done.
- **Training:** cross-encoder and listwise training checkpoint model, optimiser and scheduler every N steps, and resume
  at the exact step with the same data order. Non-finite gradients abort instead of saving a corrupt checkpoint.
- **GPU memory:**
  - On CUDA out-of-memory from other users on a shared card, training switches to 2/4/8/16 micro-batches (same
    effective batch) and inference halves its batch.
  - After 2 h of continuous OOM the stage fails on purpose, so its retry can land on another card.
- **Host memory:** GPU workers hold one shard of pre-built text, not the whole corpus (small processes are not what the
  kernel's OOM killer picks).
- **Stale files:** stage-2 model files are named after a hash of their feature list, so a resumed run never loads a
  model trained on different columns. Cross-encoder scores are keyed by (q, c), so a changed candidate set re-scores
  only the new pairs.
- **Retries and restarts:**
  - Failed stages retry with back-off (5 tries), and a watchdog kills stages whose log is silent for 60 min.
  - The supervisor restarts the runner if it exits, and leftover stage processes of a dead runner are reaped on
    restart.
  - An optional cron keepalive (`scripts/keepalive.sh`) restarts the supervisor after a kill or reboot, unless the run
    was stopped on purpose (`STOPPED`) or gave up (`GAVE_UP`).
- **Disk:** new stages wait while free disk is below 150 GB.

`src/run_pipeline.py --reset <stage>` forces a stage and everything after it to rerun.

## Optional: Colab lanes for cross-encoder inference

`src/remote.py` exports claimed test shards (text pairs, the fine-tuned model, a parity sample) to a private Hugging Face
dataset repo. The notebook `Model5_CE_Colab.ipynb` scores one lane per Colab session (A, B, C), checks parity first,
and uploads per shard; `remote.py import` brings results back into the pipeline. It needs a write token in
`~/.hf_m5_token` or `$HF_TOKEN`; no token is stored in the code.
