#!/usr/bin/env bash
# Gate G4 on the instance F3 just ran on (scripts/f3_session.sh), so the features are already
# there: re-sync the code, start scripts/g4.sh detached, wait (GPU use every minute), pull
# everything back (pull_results.sh, outputs/ included). Does NOT destroy the instance.
# PUSH=1 prepares a fresh instance instead: sample tables, manifest, frame shard indexes (for
# coverage; the .tar files are not needed) and the final features (~5.5 GB, raw/ excluded), deps.
#   usage: [KNOWN_HOSTS=...] [REMOTE_DIR=...] [PUSH=1] [G4_ENV="FRACTION=0.1 G3=0 TRAIN_OPTS=..."] scripts/g4_session.sh SSH_URL
# Exit codes: 4 = g4.sh did not finish (its log is pulled back anyway), 7 = host unreachable
# for 10 minutes while waiting.
set -euo pipefail
export LC_ALL=C  # decimal point in awk/printf: comparisons break with comma locales
URL="$1"
DIR="${REMOTE_DIR:-/workspace/egopred}"
G4_ENV="${G4_ENV:-}"
PUSH="${PUSH:-0}"
hostport="${URL#ssh://}"
host="${hostport%:*}"
port="${hostport##*:}"
KNOWN_HOSTS="${KNOWN_HOSTS:-$HOME/.ssh/known_hosts}"
SSH="ssh -p $port -o ConnectTimeout=20 -o ServerAliveInterval=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS"
export RSYNC_RSH="$SSH"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
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

log "code"
rsync -az --filter=':- .gitignore' ./ "$host:$DIR/"  # code only: what .gitignore leaves out stays here
# rsync again after a dropped link (exit 255 / 12 / 30): every copy is incremental, --partial keeps
# the part of a large file already sent
retry() {
  local i
  for i in 1 2 3 4 5; do
    "$@" && return 0
    log "copy failed (attempt $i/5), retrying in 20 s"; sleep 20
  done
  return 1
}
if [ "$PUSH" = 1 ]; then
  DATA="$(make -s --no-print-directory data-dir)"
  log "tables, manifest, frame indexes"
  remote "mkdir -p $DIR/data/openscene/frames $DIR/data/features"
  retry rsync -a --partial "$DATA/samples" "$DATA/manifest.sqlite" "$host:$DIR/data/"
  retry rsync -a --include '*/' --include '*.index.parquet' --exclude '*' "$DATA/openscene/frames/" "$host:$DIR/data/openscene/frames/"
  log "features ($(du -sh --exclude raw "$DATA/features" | cut -f1))"
  retry rsync -a --partial --info=stats1 --exclude raw "$DATA/features/" "$host:$DIR/data/features/"
  remote "cd $DIR && pip install -q pandas pyarrow pyyaml timm==1.0.30 2>&1 | tail -1; python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'"
fi
log "g4 started (log: $DIR/g4.log)"
remote "cd $DIR || exit 1; nohup setsid env $G4_ENV scripts/g4.sh > g4.log 2>&1 < /dev/null & echo started"
sleep 5
while running '[g]4.sh'; do
  remote "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null; grep -E '^\[|epoch|G3:|Error' $DIR/g4.log | tail -1 | cut -c1-200" || true
  sleep 60
done
remote "tail -25 $DIR/g4.log"
ok=0; remote "grep -q 'G4 DONE' $DIR/g4.log" && ok=1

log "pulling results back"
KNOWN_HOSTS="$KNOWN_HOSTS" scripts/pull_results.sh "$URL" "$DIR"
if [ "$ok" != 1 ]; then log "G4 DID NOT FINISH: see outputs/remote_logs/g4.log"; exit 4; fi
log "done: outputs/g4_val.md, outputs/g4_test.md, outputs/runs/g4_*; then destroy"
