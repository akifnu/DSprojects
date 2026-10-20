#!/usr/bin/env bash
# Installs everything the reel production line needs. Idempotent: re-running only
# fetches what is missing. `setup.sh --check` verifies without installing.
#
#   ffmpeg 9.0 (static BtbN build: libx264/x265/svt-av1, libvmaf, libass, zscale, NVENC)
#     → .tools/ffmpeg, and made the system default via /usr/local/bin when running as root
#   Python packages pinned in requirements.txt
#   Models (GitHub releases; huggingface.co is not needed):
#     Parakeet TDT 0.6B v2 int8 (English ASR), Whisper large-v3-turbo (multilingual ASR),
#     Silero VAD, YuNet face detector
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$HERE/.tools"; MODELS="$HERE/.models"
FFMPEG_SERIES="9.0"
FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n${FFMPEG_SERIES}-latest-linux64-gpl-${FFMPEG_SERIES}.tar.xz"
SHERPA="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

if [ "${1:-}" = "--check" ]; then
  exec "$HERE/reel" check
fi

# --- ffmpeg 9.0 ------------------------------------------------------------------
if ! "$TOOLS/ffmpeg/bin/ffmpeg" -hide_banner -version 2>/dev/null | head -1 | grep -q "version n${FFMPEG_SERIES}"; then
  mkdir -p "$TOOLS"
  tmp="$(mktemp -d)"
  curl -fsSL --retry 4 -o "$tmp/ff.tar.xz" "$FFMPEG_URL"
  tar xJf "$tmp/ff.tar.xz" -C "$TOOLS"
  rm -rf "$tmp"
  dir="$(ls -d "$TOOLS"/ffmpeg-n${FFMPEG_SERIES}* | sort | tail -1)"
  ln -sfn "$dir" "$TOOLS/ffmpeg"
  find "$TOOLS" -maxdepth 1 -name 'ffmpeg-n*' ! -path "$dir" -exec rm -rf {} +
fi
"$TOOLS/ffmpeg/bin/ffmpeg" -hide_banner -version | head -1 > "$TOOLS/ffmpeg.version"
if [ "$(id -u)" = 0 ]; then                       # plain `ffmpeg` = 9.0 everywhere
  for t in ffmpeg ffprobe; do ln -sf "$TOOLS/ffmpeg/bin/$t" /usr/local/bin/$t; done
fi

# --- Python ----------------------------------------------------------------------
if ! python3 - "$HERE/requirements.txt" <<'PY' 2>/dev/null
import sys, importlib.metadata as m
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if line:
        name, ver = line.split("==")
        assert m.version(name) == ver, name
PY
then
  pip install -q --root-user-action=ignore -r "$HERE/requirements.txt"
fi

# --- Models ----------------------------------------------------------------------
mkdir -p "$MODELS/yunet"
fetch_tar() {   # name → $MODELS/name
  [ -d "$MODELS/$1" ] || curl -fsSL --retry 4 "$SHERPA/$1.tar.bz2" | tar xj -C "$MODELS"
}
fetch_tar sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
fetch_tar sherpa-onnx-whisper-turbo
[ -f "$MODELS/silero_vad.onnx" ] || curl -fsSL --retry 4 -o "$MODELS/silero_vad.onnx" "$SHERPA/silero_vad.onnx"
[ -f "$MODELS/yunet/yunet.onnx" ] || curl -fsSL --retry 4 -o "$MODELS/yunet/yunet.onnx" \
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

"$HERE/reel" check >/dev/null && echo "reel-pipeline ready ($(cat "$TOOLS/ffmpeg.version" | cut -d' ' -f1-3))"
