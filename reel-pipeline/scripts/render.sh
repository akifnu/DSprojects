#!/usr/bin/env bash
# Renders the current FableCut timeline with ffmpeg (no browser) to <dest-dir>/<name>.mp4.
#   render.sh <dest-dir> [name]                 LUFS=-14 by default
#   render.sh --frame <seconds> <out.jpg>       preview one frame of the edit
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-7777}"; LUFS="${LUFS:--14}"
proj=$(mktemp --suffix=.json); trap 'rm -f "$proj"' EXIT
curl -sf "localhost:$PORT/api/project" > "$proj"
if [ "$1" = "--frame" ]; then
  exec python3 "$HERE/scripts/fast_render.py" "$proj" "$3" --frame "$2"
fi
dest="$1"; mkdir -p "$dest"
name="${2:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('name','reel'))" "$proj")}"
python3 "$HERE/scripts/fast_render.py" "$proj" "$dest/$name.mp4" --lufs "$LUFS"
