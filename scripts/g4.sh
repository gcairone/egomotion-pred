#!/usr/bin/env bash
export LC_ALL=C  # decimal point in awk/printf: comparisons break with comma locales
# Gate G4 on the GPU host, right after F3 on the same instance (features already there):
# coverage must be 0, G3 repeated on real features, then variants A/B/C trained on FRACTION
# of navtrain and compared with B0 by scenario. Results in outputs/ (pulled back by
# pull_results.sh). Run detached from scripts/g4_session.sh.
#   usage: [FRACTION=0.1] [DEVICE=cuda] [G3=1] [G3_OPTS=...] [TRAIN_OPTS=...] scripts/g4.sh
# G3=0 skips the overfit check on real features: neighbouring real frames are near duplicates,
# so even variant A does not reach the minADE target on 256 samples (outputs/g3_diag/); the
# fake-feature G3 remains the bug check.
# Exit codes: 5 = frames/features missing, 6 = G3 failed on real features (stop: bug or
# features problem, to look at locally).
set -euo pipefail
FRACTION="${FRACTION:-0.1}"
DEVICE="${DEVICE:-cuda}"
G3="${G3:-1}"
G3_OPTS="${G3_OPTS:-}"
TRAIN_OPTS="${TRAIN_OPTS:---max-minutes 20}"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
mkdir -p outputs

log "coverage"
python -m egopred coverage navtrain navtest | tee outputs/g4_coverage.json
python - <<'EOF' || { log "MISSING FRAMES/FEATURES"; exit 5; }
import json, sys
c = json.load(open("outputs/g4_coverage.json"))
sys.exit(any(v["missing_features"] for v in c.values()))
EOF

if [ "$G3" = 1 ]; then
  log "G3 on real features"
  # shellcheck disable=SC2086
  python -m egopred g3 --frames real --device "$DEVICE" $G3_OPTS
  python -c 'import json, sys; sys.exit(not json.load(open("outputs/g3/report.json"))["passed"])' \
    || { log "G3 FAILED on real features"; exit 6; }
fi

for v in A B C; do
  log "train g4_$v"
  # shellcheck disable=SC2086
  python -m egopred train --name "g4_$v" --variant "$v" --fraction "$FRACTION" --device "$DEVICE" $TRAIN_OPTS
done
python -m egopred compare g4_A g4_B g4_C --split val > outputs/g4_val.md
python -m egopred compare g4_A g4_B g4_C --split test > outputs/g4_test.md
cat outputs/g4_val.md
log "G4 DONE"
