#!/usr/bin/env bash
# Endless background with no transition: repeat frames 84-160 of the 30 fps ocean clip
# (hard cut 160 -> 84, chosen by scoring every possible cut against the motion around it).
#   build_loop.sh <clip> <frames-out> <out.mp4>
set -euo pipefail
in="$1"; frames="$2"; out="$3"
ffmpeg -v error -y -i "$in" -vf "fps=30,scale=720:1280:flags=lanczos,setsar=1,trim=start_frame=84:end_frame=161,setpts=N/30/TB,loop=loop=$((frames/77+1)):size=77:start=0,trim=end_frame=$frames,setpts=N/30/TB" \
  -r 30 -c:v libx264 -preset veryfast -crf 14 -pix_fmt yuv420p "$out"
