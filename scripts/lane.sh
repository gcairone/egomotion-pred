#!/usr/bin/env bash
# Job lane on the GPU host: waits until no process matches WAIT_PATTERN, then runs the train
# jobs of JOBFILE (one "NAME args..." per line) one after another. The file is re-read after
# every job, so jobs can be appended while the lane runs; runs that already have a result.json
# are skipped. Each vision run keeps its features on the GPU (~11.6 GB), so two lanes = two
# parallel runs on a 24 GB card. MAX_MINUTES (default 60) caps each run.
#   usage (detached, from the repo root on the instance):
#     nohup setsid scripts/lane.sh '[-]-name RUN_TO_WAIT_FOR ' lane1.txt > lane1.log 2>&1 < /dev/null &
set -uo pipefail
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
while pgrep -f "$1" >/dev/null; do sleep 20; done
i=1
while :; do
  line=$(sed -n "${i}p" "$2")
  [ -z "$line" ] && { log "LANE DONE"; break; }
  name=${line%% *}; args=${line#* }
  if [ -f "outputs/runs/$name/result.json" ]; then log "skip $name"; else
    log "train $name $args"
    # shellcheck disable=SC2086
    python -m egopred train --name "$name" $args --fraction 1.0 --device cuda --max-minutes "${MAX_MINUTES:-60}" > "$name.log" 2>&1
    log "end $name rc=$?"
  fi
  i=$((i + 1))
done
