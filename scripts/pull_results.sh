#!/usr/bin/env bash
# Copy the outputs of a remote run (frame shards, features, logs) back to this machine, which
# is the source of truth, and import the remote manifest. Run it, and check its coverage
# report, before destroying the instance: anything left only there is lost.
#   usage: [KNOWN_HOSTS=...] scripts/pull_results.sh SSH_URL [REMOTE_DIR]
#   e.g.   scripts/pull_results.sh "$(vastai ssh-url ID)" /workspace/egopred
set -euo pipefail
export LC_ALL=C  # decimal point in awk/printf: comparisons break with comma locales
URL="$1"
DIR="${2:-/workspace/egopred}"
hostport="${URL#ssh://}"
host="${hostport%:*}"
port="${hostport##*:}"
KNOWN_HOSTS="${KNOWN_HOSTS:-$HOME/.ssh/known_hosts}"
export RSYNC_RSH="ssh -p $port -o ConnectTimeout=20 -o ServerAliveInterval=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS"
DATA="$(make -s --no-print-directory data-dir)"  # local data root (see Makefile)

df -h "$DATA" | tail -1
mkdir -p "$DATA/openscene/frames" "$DATA/features" outputs/remote_logs
# *.partial are files still being written (or left by a crash): never copy them
rsync -a --info=stats1 --exclude '*.partial' "$host:$DIR/data/openscene/frames/" "$DATA/openscene/frames/"
# raw/ = pre-PCA stage-1 cache (~16 GB), only needed on the GPU host
rsync -a --info=stats1 --exclude '*.partial' --exclude 'raw/' "$host:$DIR/data/features/" "$DATA/features/"
rsync -a "$host:$DIR/*.log" outputs/remote_logs/
# reports, run results and checkpoints (G3/G4/training); absent after an F3-only session
if $RSYNC_RSH "$host" test -d "$DIR/outputs" 2>/dev/null; then
  rsync -a --info=stats1 --exclude '*.partial' "$host:$DIR/outputs/" outputs/
fi
rsync -a "$host:$DIR/data/manifest.sqlite" "$DATA/remote_manifest.sqlite"
make run ARGS="merge-manifest $DATA/remote_manifest.sqlite"
make run ARGS="status"
make run ARGS="coverage navtrain navtest"
df -h "$DATA" | tail -1
