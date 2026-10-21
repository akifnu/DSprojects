#!/usr/bin/env bash
# Small CI/test profile: real ffmpeg 9, face detector, OCR (from requirements), fonts.
# Speech is supplied as fingerprinted fixtures; production ASR models are not downloaded.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p .tools .models/yunet .models/testdata
tmp="$(mktemp -d .tools/test-download.XXXXXX)"
trap 'rm -rf -- "$tmp"' EXIT
if [ ! -x .tools/ffmpeg/bin/ffmpeg ]; then
  archive=ffmpeg-n9.0-latest-linux64-gpl-9.0.tar.xz
  base=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest
  curl -fsSL --retry 3 "$base/$archive" -o "$tmp/$archive"
  curl -fsSL --retry 3 "$base/checksums.sha256" -o "$tmp/checksums.sha256"
  (cd "$tmp" && sha256sum --check --ignore-missing checksums.sha256)
  mkdir -p .tools/ffmpeg
  tar -xJf "$tmp/$archive" --strip-components=1 -C .tools/ffmpeg
fi
.tools/ffmpeg/bin/ffmpeg -version > "$tmp/version.txt"
head -1 "$tmp/version.txt"
if [ ! -s .models/yunet/yunet.onnx ]; then
  curl -fsSL --retry 3 https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx -o .models/yunet/yunet.onnx
fi
if [ ! -s .models/testdata/lena.jpg ]; then
  curl -fsSL --retry 3 https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg -o .models/testdata/lena.jpg
fi
sha256sum --check <<'SUMS'
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  .models/yunet/yunet.onnx
7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07  .models/testdata/lena.jpg
SUMS
python3 - <<'PY'
from pathlib import Path
from reelkit import captions, sample_text
captions.prefetch_fonts(sample_text.CANDIDATES)
# A fallback bank would make font-identity assertions meaningless. Fail setup
# with a useful message rather than silently testing every family as one font.
for family in ("Arimo", "Montserrat", "Anton", "Poppins", "Roboto", "Inter"):
    font = captions.font_file(family, 400 if family == "Anton" else 700)
    if Path(font) in map(Path, captions.FALLBACKS):
        raise SystemExit(f"test font {family} is unavailable: allow Google Fonts during bootstrap")
PY
