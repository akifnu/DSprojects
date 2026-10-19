#!/usr/bin/env bash
# Quality check of a finished reel against its sample.
#   qc.sh <reel.mp4> [sample.mp4]
# Prints format / duration / loudness and writes <reel>.sheet.jpg (and the
# sample's sheet) — 12 evenly spaced frames — to compare look and pacing.
set -euo pipefail
sheet() {
  local d; d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$1")
  ffmpeg -loglevel error -y -i "$1" -vf "fps=12/$d,scale=180:-2,tile=6x2" -frames:v 1 "${1%.*}.sheet.jpg"
  echo "sheet: ${1%.*}.sheet.jpg"
}
report() {
  echo "== $1"
  ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,r_frame_rate -of compact "$1"
  ffmpeg -hide_banner -nostats -i "$1" -vn -af ebur128=peak=true -f null - 2>&1 | grep -A1 -E "^\s+(I:|True peak:)" | grep -E "I:|Peak:" | tail -2 | sed 's/^ */  /'
  sheet "$1"
}
for f in "$@"; do report "$f"; done
