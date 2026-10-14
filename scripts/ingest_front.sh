#!/usr/bin/env bash
# Front-camera ingest of everything the NAVSIM task needs, several archives in parallel:
# navtrain current + history archives (front frames of navtrain samples) and the test-split
# camera archives (only frames of navtest samples). Frames are stored at 960x540.
# Idempotent: verified shards are skipped, so it can simply be re-run after a failure.
# Runs locally (`make ingest JOBS=4`): the archives hold all 8 cameras + LiDAR and ~97% of the
# bytes are discarded, so streaming them into a host that bills inbound traffic is wasted money.
#   usage: scripts/ingest_front.sh [JOBS]   (run where python -m egopred works)
set -euo pipefail
JOBS="${1:-4}"
{
  for i in $(seq 1 32); do echo "--split navtrain --kind current --index $i --only-samples navtrain"; done
  for i in $(seq 1 32); do echo "--split navtrain --kind history --index $i --only-samples navtrain"; done
  for i in $(seq 0 31); do echo "--split test --index $i --only-samples navtest"; done
} | xargs -P "$JOBS" -I{} sh -c 'python -m egopred frames {} --max-width 960 || echo "FAILED: {}"'
python -m egopred status
