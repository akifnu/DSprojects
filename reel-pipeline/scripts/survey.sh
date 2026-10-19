#!/usr/bin/env bash
# Surveys long raw footage so selects can be chosen without watching it all.
#   survey.sh <video> <out-dir> [seconds-per-thumb]   (default: auto, ~48 thumbs)
# Writes <out-dir>/<name>.scenes.json (shot boundaries from FableCut's analyzer)
# and <name>.sheet-N.jpg contact sheets (4×4, each thumb stamped with its time).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
in="$1"; out="$2"; mkdir -p "$out"
name=$(basename "${in%.*}")
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$in")
step="${3:-$(python3 -c "print(max(1, round($dur/48)))")}"
FABLECUT_DATA_DIR="$HERE/workspace" node "$HERE/.fablecut/analyze.js" "$in" --no-music \
  | python3 -c "
import sys,json; j=json.load(sys.stdin)
json.dump({k:j[k] for k in ('duration','fps','width','height','cuts','shots','avgShotLen') if k in j}, open('$out/$name.scenes.json','w'), indent=1)
print(f\"{len(j.get('shots',[]))} shots over {j['duration']:.0f}s → $out/$name.scenes.json\")"
ffmpeg -loglevel error -y -i "$in" -vf "fps=1/$step,scale=320:-2,drawtext=text='%{pts\:hms\:0}':x=6:y=6:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,setpts=N/TB*$step,tile=4x4" \
  -fps_mode passthrough "$out/$name.sheet-%d.jpg"
echo "contact sheets every ${step}s → $out/$name.sheet-*.jpg"
