#!/usr/bin/env bash
# Seamless, forward-only background loop from a short clip:  build_loop.sh <clip> <seconds> <out.mp4>
# The clip's last frames are optically blended into its first frames (minterpolate), so the
# wrap has no cut and nothing plays backwards.
set -euo pipefail
in="$1"; secs="$2"; out="$3"; W="$(mktemp -d)"; mkdir "$W/seq"
ffmpeg -v error -y -i "$in" -vf "fps=30,scale=720:1280:flags=lanczos,setsar=1" -c:v libx264 -preset veryfast -crf 10 -pix_fmt yuv420p "$W/fwd.mp4"
N=$(ffprobe -v error -count_frames -select_streams v -show_entries stream=nb_read_frames -of csv=p=0 "$W/fwd.mp4")
ffmpeg -v error -y -i "$W/fwd.mp4" -vf "select='gte(n\,$((N-3)))'" -vsync 0 -start_number 0 "$W/seq/s%02d.png"
ffmpeg -v error -y -i "$W/fwd.mp4" -vf "select='lt(n\,3)'" -vsync 0 -start_number 3 "$W/seq/s%02d.png"
ffmpeg -v error -y -framerate 30 -i "$W/seq/s%02d.png" -vf "minterpolate=fps=360:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:me=umh" \
  -c:v libx264 -crf 10 -pix_fmt yuv420p "$W/morph.mp4"
ffmpeg -v error -y -i "$W/morph.mp4" -vf "select='between(n\,25\,35)',setpts=N/30/TB" -r 30 -c:v libx264 -crf 10 -pix_fmt yuv420p "$W/bridge.mp4"
printf "file '%s'\nfile '%s'\n" "$W/fwd.mp4" "$W/bridge.mp4" > "$W/unit.txt"
U=$((N+11))
ffmpeg -v error -y -f concat -safe 0 -i "$W/unit.txt" -r 30 -frames:v $U -c:v libx264 -crf 10 -pix_fmt yuv420p "$W/unit.mp4"
F=$(python3 -c "print(round($secs*30))")
ffmpeg -v error -y -i "$W/unit.mp4" -vf "loop=loop=$((F/U+1)):size=$U:start=0,trim=end_frame=$F,setpts=N/30/TB" -r 30 \
  -c:v libx264 -preset veryfast -crf 14 -pix_fmt yuv420p "$out"
rm -rf "$W"
