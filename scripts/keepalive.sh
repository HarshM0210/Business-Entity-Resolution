#!/usr/bin/env bash
# Cron keepalive (every 10 min): restarts the supervisor if it was killed (OOM killer, pkill, reboot), unless the
# run is finished or was stopped on purpose (scripts/run.sh stop writes state/STOPPED). Logs to logs/keepalive.log.
#   crontab:  */10 * * * * /scratch/user4/dai/model-5/scripts/keepalive.sh
cd "$(dirname "$0")/.."
W=${M5_WORK:-$PWD/work}
[ -f "$W/state/package.done" ] && exit 0
[ -f "$W/state/STOPPED" ] && exit 0
[ -f "$W/state/GAVE_UP" ] && exit 0      # repeated deterministic failure: needs a human, not another restart
SUP="$W/state/supervisor.pid"
if [ -f "$SUP" ] && kill -0 "$(cut -d' ' -f1 "$SUP")" 2>/dev/null; then exit 0; fi
echo "[$(date '+%F %T')] keepalive: supervisor not running -> restarting" >> "$W/logs/keepalive.log"
exec scripts/run.sh start >> "$W/logs/keepalive.log" 2>&1
