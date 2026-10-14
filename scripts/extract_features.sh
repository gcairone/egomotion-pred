#!/usr/bin/env bash
export LC_ALL=C  # decimal point in awk/printf: comparisons break with comma locales
# Feature extraction on a GPU host while the frame shards are still arriving by rsync from the
# local server. The manifest copied from there lists every verified shard; `features` only
# processes shards whose files are fully present, and is idempotent, so it is re-run until
# every shard has final features (stage 1 runs as shards land; the PCA fit and the projection
# wait for the PCA-subset frames). Gives up if nothing progresses for STALL_MIN minutes, so a
# stuck transfer does not keep a paid instance busy forever.
#   usage: STALL_MIN=20 scripts/extract_features.sh [features options...]
set -uo pipefail
STALL_MIN="${STALL_MIN:-20}"
# prints "<shards without final features> <shards present> <shards with raw features>";
# progress = any of the three changing (during the upload only the last two move)
progress() {
  python -c "from egopred import paths; from egopred.features import DEFAULT, frame_shards, pending_shards
from egopred.manifest import Manifest
m = Manifest(paths.MANIFEST)
print(len(pending_shards(DEFAULT, m)), len(frame_shards(m)), len(m.keys('features_raw:' + DEFAULT.name, 'verified')))"
}
prev=""
since=$(date +%s)
while true; do
  python -m egopred features "$@" || echo "features run failed"
  state=$(progress) || { echo "cannot read the manifest"; exit 1; }
  left=${state%% *}
  [ "$left" = 0 ] && break
  if [ "$state" != "$prev" ]; then prev=$state; since=$(date +%s); fi
  if [ $(( $(date +%s) - since )) -ge $(( STALL_MIN * 60 )) ]; then
    echo "STALLED: $left shards still without features after ${STALL_MIN} min without progress"
    python -m egopred coverage navtrain navtest
    exit 2
  fi
  echo "without final features / present / with raw features: $state; next pass in 60 s"
  sleep 60
done
python -m egopred coverage navtrain navtest
