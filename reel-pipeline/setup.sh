#!/usr/bin/env bash
# Installs the reel pipeline's tools: FableCut (pinned), sherpa-onnx + Parakeet speech model.
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

python3 -c "import sherpa_onnx" 2>/dev/null || pip install -q sherpa-onnx

# Speech model for scripts/transcribe.py (GitHub release; huggingface.co is often blocked)
ASR="parakeet-tdt-0.6b-v2-int8"
if [ ! -f "$HERE/.models/sherpa-onnx-nemo-$ASR/encoder.int8.onnx" ]; then
  mkdir -p "$HERE/.models"
  curl -sSL "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-$ASR.tar.bz2" \
    | tar xj -C "$HERE/.models" || echo "WARNING: speech model download failed; transcription unavailable" >&2
fi

mkdir -p "$HERE/workspace"
for t in node ffmpeg ffprobe; do command -v "$t" >/dev/null || echo "WARNING: $t not on PATH" >&2; done
echo "reel-pipeline ready: FableCut at $FC, data in $HERE/workspace"
