#!/usr/bin/env bash
# Installs the reel pipeline's tools: FableCut (pinned), faster-whisper.
# Idempotent — safe to run at the start of every session.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FABLECUT_REF="53b268aed0cf209266bfb76dc12762a17b2bd34e"   # FableCut v1.15.0
FC="$HERE/.fablecut"

if [ ! -d "$FC/.git" ]; then
  git clone --quiet https://github.com/ronak-create/FableCut.git "$FC"
fi
if [ "$(git -C "$FC" rev-parse HEAD)" != "$FABLECUT_REF" ]; then
  git -C "$FC" fetch --quiet origin "$FABLECUT_REF" 2>/dev/null || git -C "$FC" fetch --quiet origin
  git -C "$FC" checkout --quiet "$FABLECUT_REF"
fi

python3 -c "import faster_whisper" 2>/dev/null || pip install -q faster-whisper

mkdir -p "$HERE/workspace"
for t in node ffmpeg ffprobe; do command -v "$t" >/dev/null || echo "WARNING: $t not on PATH" >&2; done
echo "reel-pipeline ready: FableCut at $FC, data in $HERE/workspace"
