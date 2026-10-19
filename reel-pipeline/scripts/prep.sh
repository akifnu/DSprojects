#!/usr/bin/env bash
# An edit-friendly copy of heavy footage (4K / HEVC / VP9 / 60 fps), optionally
# trimmed, so renders decode less.  prep.sh <in> <out-name> [start-sec] [end-sec]
# Video → H.264/AAC MP4, short side ≤ $PREP_RES (default 720), 30 fps, short GOP;
# audio-only → FLAC.
# Writes into workspace/media/ and prints the /media/ src. fast_render reads the
# originals too, so this is only for speed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
in="$1"; name="$2"; ss="${3:-}"; to="${4:-}"; R="${PREP_RES:-720}"
trim=(); [ -n "$ss" ] && trim+=(-ss "$ss"); [ -n "$to" ] && trim+=(-to "$to")
mkdir -p "$HERE/workspace/media"
has_video=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_type -of csv=p=0 "$in" | head -1)
if [ "$has_video" = "video" ]; then
  out="$HERE/workspace/media/${name%.*}.edit.mp4"
  ffmpeg -loglevel error -y "${trim[@]}" -i "$in" \
    -vf "scale='if(gt(iw,ih),-2,min($R,iw))':'if(gt(iw,ih),min($R,ih),-2)',fps=30" \
    -c:v libx264 -preset veryfast -crf 16 -g 15 -pix_fmt yuv420p -c:a aac -b:a 256k -ar 48000 "$out"
else
  out="$HERE/workspace/media/${name%.*}.flac"
  ffmpeg -loglevel error -y "${trim[@]}" -i "$in" -vn -c:a flac -ar 48000 "$out"
fi
echo "/media/$(basename "$out")"
