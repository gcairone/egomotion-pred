#!/usr/bin/env bash
# F3 on a rented GPU host that is already up (vastai_scripts/create_and_wait.sh): push code,
# tables and manifest, measure the upload speed, start the feature extraction, push the frame
# shards (extraction runs on them as they land), wait, pull the features back and check
# coverage. It does NOT destroy the instance: do that after checking the coverage it prints.
# Run vastai_scripts/watch_cost.sh alongside it.
#   usage: [KNOWN_HOSTS=...] [FEATURE_OPTS="..."] [REMOTE_DIR=...] scripts/f3_session.sh SSH_URL [JOBS] [MIN_MBPS]
#   e.g.   scripts/f3_session.sh "$(vastai ssh-url ID)" 4 10
# Exit codes: 3 = upload slower than MIN_MBPS MB/s (destroy and pick another offer),
#             4 = remote extraction failed or stalled, 7 = host unreachable for 10 minutes.
set -euo pipefail
export LC_ALL=C  # decimal point in awk/printf: comparisons break with comma locales
URL="$1"
JOBS="${2:-4}"
MIN_MBPS="${3:-10}"
DIR="${REMOTE_DIR:-/workspace/egopred}"
FEATURE_OPTS="${FEATURE_OPTS:---device cuda --batch 128 --workers 6}"
hostport="${URL#ssh://}"
host="${hostport%:*}"
port="${hostport##*:}"
# KNOWN_HOSTS: the per-instance file create_and_wait.sh prints (~/.vastai_exp_known_hosts/instance_ID.known_hosts)
KNOWN_HOSTS="${KNOWN_HOSTS:-$HOME/.ssh/known_hosts}"
SSH="ssh -p $port -o ConnectTimeout=20 -o ServerAliveInterval=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS"
export RSYNC_RSH="$SSH"
DATA="$(make -s --no-print-directory data-dir)"
FRAMES="$DATA/openscene/frames"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
# runs a command on the host, drops the login banner, returns the command's own exit code
remote() {
  local out rc
  out=$($SSH "$host" "$@" 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | grep -v "X11\|Welcome\|Have fun" || true
  return $rc
}
# 0 while a process matching $1 runs on the host, 1 once it is gone. ssh drops (exit 255) are
# retried once a minute, 10 times, so a flaky link is not mistaken for the job having ended.
running() {
  local rc tries=0
  while :; do
    remote "pgrep -f '$1' >/dev/null" >/dev/null && return 0 || rc=$?
    [ "$rc" = 255 ] || return 1
    tries=$((tries + 1))
    [ "$tries" -lt 10 ] || { log "HOST UNREACHABLE for 10 minutes"; exit 7; }
    log "ssh dropped, retrying in 60 s"; sleep 60
  done
}

log "hardware"
remote 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; echo "cpus $(nproc)"; free -g | sed -n 2p; df -h /workspace | tail -1'

log "code, tables, manifest"
remote "mkdir -p $DIR/data/openscene/frames/trainval $DIR/data/openscene/frames/test"
rsync -az --filter=':- .gitignore' ./ "$host:$DIR/"  # code only: what .gitignore leaves out stays here
rsync -a "$DATA/samples" "$DATA/manifest.sqlite" "$host:$DIR/data/"
rsync -a "$DATA/openscene/scene_filters" "$host:$DIR/data/openscene/"
remote "cd $DIR && pip install -q pandas pyarrow pyyaml timm==1.0.30 2>&1 | tail -1; python -c 'import torch, timm; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"timm\", timm.__version__)'"

# shards in a fixed order, largest first so the parallel copies finish close together
mapfile -t SHARDS < <(cd "$FRAMES" && ls -S trainval/*.tar test/*.tar | sed 's/\.tar$//')
log "${#SHARDS[@]} shards, $(du -csh "$FRAMES"/trainval "$FRAMES"/test | tail -1 | cut -f1) to send"

log "upload speed probe on ${SHARDS[0]}"
t0=$(date +%s.%N)
rsync -a "$FRAMES/${SHARDS[0]}.tar" "$FRAMES/${SHARDS[0]}.index.parquet" "$host:$DIR/data/openscene/frames/${SHARDS[0]%/*}/"
t1=$(date +%s.%N)
mbps=$(awk -v b="$(stat -c %s "$FRAMES/${SHARDS[0]}.tar")" -v a="$t0" -v z="$t1" 'BEGIN{printf "%.1f", b/(z-a)/1e6}')
log "upload: $mbps MB/s on one stream"
if awk -v m="$mbps" -v t="$MIN_MBPS" 'BEGIN{exit !(m < t)}'; then
  log "SLOW UPLOAD: below $MIN_MBPS MB/s, stopping here (destroy and pick another offer)"
  exit 3
fi

log "remote extraction started (log: $DIR/features.log)"
# not `cd && nohup ... &`: the & would background the whole list in a subshell that keeps the
# ssh stdout open, and this call would never return
remote "cd $DIR || exit 1; nohup setsid scripts/extract_features.sh $FEATURE_OPTS > features.log 2>&1 < /dev/null & echo started"

log "sending the other shards, $JOBS in parallel"
printf '%s\n' "${SHARDS[@]:1}" | xargs -P "$JOBS" -I{} sh -c \
  'rsync -a "'"$FRAMES"'/{}.tar" "'"$FRAMES"'/{}.index.parquet" "'"$host:$DIR"'/data/openscene/frames/$(dirname {})/" || echo "SEND FAILED: {}"'
log "all shards sent"

log "waiting for the extraction"
# [e]: the pattern must not match the remote shell running this very pgrep
while running '[e]xtract_features.sh'; do
  remote "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; grep -E 'without final features|STALLED|PCA|Error' $DIR/features.log | tail -1" || true
  sleep 60
done
remote "tail -20 $DIR/features.log"
if remote "grep -q STALLED $DIR/features.log"; then log "EXTRACTION STALLED"; exit 4; fi

log "pulling results back"
KNOWN_HOSTS="$KNOWN_HOSTS" scripts/pull_results.sh "$URL" "$DIR"
log "done: check that coverage shows 0 missing_features for navtrain and navtest, then destroy"
