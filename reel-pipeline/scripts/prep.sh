#!/usr/bin/env bash
# Optional: an edit-friendly copy of heavy footage (4K / HEVC / 60 fps), optionally
# trimmed, so renders decode less.  prep.sh <in> <out-name> [start-sec] [end-sec]
# Video → H.264/AAC MP4, long side ≤ 1920, 30 fps, short GOP; audio-only → FLAC.
# Writes into workspace/media/ and prints the /media/ src. fast_render reads the
# originals too, so this is only for speed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
in="$1"; name="$2"; ss="${3:-}"; to="${4:-}"
trim=(); [ -n "$ss" ] && trim+=(-ss "$ss"); [ -n "$to" ] && trim+=(-to "$to")
mkdir -p "$HERE/workspace/media"
has_video=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_type -of csv=p=0 "$in" | head -1)
if [ "$has_video" = "video" ]; then
  out="$HERE/workspace/media/${name%.*}.edit.mp4"
  ffmpeg -loglevel error -y "${trim[@]}" -i "$in" \
    -vf "scale='if(gt(iw,ih),min(1920,iw),-2)':'if(gt(iw,ih),-2,min(1920,ih))',fps=30" \
    -c:v libx264 -preset veryfast -crf 16 -g 15 -pix_fmt yuv420p -c:a aac -b:a 256k -ar 48000 "$out"
else
  out="$HERE/workspace/media/${name%.*}.flac"
  ffmpeg -loglevel error -y "${trim[@]}" -i "$in" -vn -c:a flac -ar 48000 "$out"
fi
echo "/media/$(basename "$out")"
