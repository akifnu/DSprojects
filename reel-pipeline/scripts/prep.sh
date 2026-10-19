#!/usr/bin/env bash
# Makes a browser-safe copy of a media file for the editor, optionally trimmed.
#   prep.sh <in> <out-name> [start-sec] [end-sec]
# Video → VP9/Opus WebM (short GOP for frame-accurate seeking, long side ≤ 1920);
# audio-only → FLAC. Writes into workspace/media/ and prints the /media/ src.
# Needed because the container's headless Chromium cannot decode H.264/AAC;
# trim long raw footage to selects first rather than converting whole files.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
in="$1"; name="$2"; ss="${3:-}"; to="${4:-}"
trim=(); [ -n "$ss" ] && trim+=(-ss "$ss"); [ -n "$to" ] && trim+=(-to "$to")
mkdir -p "$HERE/workspace/media"
has_video=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_type -of csv=p=0 "$in" | head -1)
if [ "$has_video" = "video" ]; then
  out="$HERE/workspace/media/${name%.*}.webm"
  ffmpeg -loglevel error -y "${trim[@]}" -i "$in" \
    -vf "scale='if(gt(iw,ih),min(1920,iw),-2)':'if(gt(iw,ih),-2,min(1920,ih))'" \
    -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -row-mt 1 -g 15 -crf 24 -b:v 0 -pix_fmt yuv420p \
    -c:a libopus -b:a 192k -ar 48000 "$out"
else
  out="$HERE/workspace/media/${name%.*}.flac"
  ffmpeg -loglevel error -y "${trim[@]}" -i "$in" -vn -c:a flac -ar 48000 "$out"
fi
echo "/media/$(basename "$out")"
