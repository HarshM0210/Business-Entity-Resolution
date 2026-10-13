#!/usr/bin/env bash
# Usage: scripts/run.sh start|status|stop|logs|tail <stage>|gpus     (prefix M5_SMOKE=1 for the smoke run)
#
# "start" launches a detached supervisor (setsid: survives SSH disconnects). The supervisor restarts the DAG
# runner if it dies or stops with failed stages (checkpoints make every restart a resume), up to
# M5_MAX_RESTARTS times. "start" again while it runs is a no-op; after "stop", "start" resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${M5_PY:-/scratch/user4/dai/model-4/.venv/bin/python}
W=${M5_WORK:-$ROOT/work}; [ "${M5_SMOKE:-0}" = "1" ] && W=${M5_WORK:-$ROOT/work_smoke}
mkdir -p "$W/logs" "$W/state"
SUP="$W/state/supervisor.pid"
export HF_HOME=${HF_HOME:-/scratch/user4/mtp/cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}          # models are cached; set 0 to allow downloads
export TOKENIZERS_PARALLELISM=false
alive() { [ -f "$SUP" ] && kill -0 "$(cut -d' ' -f1 "$SUP")" 2>/dev/null; }

supervise() {
  local max=${M5_MAX_RESTARTS:-20} n=0 rc
  echo "$$ $(ps -o pgid= $$ | tr -d ' ')" > "$SUP"
  trap 'kill -TERM "$child" 2>/dev/null; wait "$child"; rm -f "$SUP"; exit 143' TERM INT
  while :; do
    echo "[$(date '+%F %T')] supervisor: runner start (restart $n/$max)"
    "$PY" -u src/run_pipeline.py "$@" & child=$!
    set +e; wait "$child"; rc=$?; set -e
    echo "[$(date '+%F %T')] supervisor: runner exited rc=$rc"
    if [ $rc -eq 0 ]; then echo "[$(date '+%F %T')] supervisor: PIPELINE COMPLETE"; break; fi
    if [ $rc -eq 3 ]; then echo "supervisor: another runner holds the lock; exiting"; break; fi
    n=$((n + 1)); if [ $n -gt "$max" ]; then echo "supervisor: GIVING UP after $max restarts (see stage logs)"
      touch "$W/state/GAVE_UP"; break; fi
    sleep $((60 * (n < 10 ? n : 10)))
  done
  rm -f "$SUP"
}

case "${1:-}" in
  start)
    rm -f "$W/state/STOPPED" "$W/state/GAVE_UP"
    if alive; then echo "already running (supervisor pid $(cut -d' ' -f1 "$SUP"))"; exit 0; fi
    setsid -f bash "$0" _supervise "${@:2}" >> "$W/logs/pipeline.out" 2>&1 < /dev/null
    sleep 3; alive && echo "started (supervisor pid $(cut -d' ' -f1 "$SUP")); logs: $W/logs/pipeline.out" \
      || { echo "failed to start"; tail -5 "$W/logs/pipeline.out"; } ;;
  _supervise) shift; supervise "$@" ;;
  status) alive && echo "RUNNING" || echo "not running"; (cd src && "$PY" run_pipeline.py --status) ;;
  stop)
    touch "$W/state/STOPPED"            # tells the cron keepalive not to restart it
    if alive; then kill -TERM "$(cut -d' ' -f1 "$SUP")"; echo "stopping (running stages are terminated; their"\
      "finished shards are kept)"; else echo "not running"; fi ;;
  logs) tail -n "${2:-40}" "$W/logs/pipeline.out"; echo ---; tail -n "${2:-40}" "$W/logs/pipeline.log" 2>/dev/null || true ;;
  tail) tail -n "${3:-40}" "$W/logs/$2.log" ;;
  gpus) nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv ;;
  *) echo "usage: $0 start|status|stop|logs|tail <stage>|gpus"; exit 1 ;;
esac
