# model-5 pipeline is running here (owner: the model-5 Claude Code session, for the ML Challenge submission)

Its GPU workers are `python ... /scratch/user4/dai/model-5/src/stages.py ... --owner=model-5-pipeline-DO-NOT-KILL`.
Please do not kill them: each kill loses in-progress cross-encoder shards and burns a retry. They use at most one
shard of text in RAM and release their GPU memory when their queue is empty.
Status: `/scratch/user4/dai/model-5/scripts/run.sh status`
