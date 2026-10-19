#!/usr/bin/env bash
# Renders the current FableCut timeline headlessly and copies the file out.
#   export.sh <dest-dir> [preset]      (preset default: instagram-reels)
# Audio is normalized to $LUFS (default −14 LUFS, the Reels/TikTok norm).
# Slow in a GPU-less container (~1 frame/s); runs until done or failed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$1"; preset="${2:-instagram-reels}"; PORT="${PORT:-7777}"; LUFS="${LUFS:--14}"
mkdir -p "$dest"
id=$(curl -sf -X POST "localhost:$PORT/api/export/request" -H 'content-type: application/json' \
  -d "{\"where\":\"headless\",\"range\":\"entire\",\"preset\":\"$preset\",\"delivery\":{\"audio\":{\"loudness\":$LUFS}}}" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
echo "export job $id ($preset)"
last=""
while :; do
  j=$(curl -sf "localhost:$PORT/api/export/job?id=$id")
  read -r status progress src err < <(echo "$j" | python3 -c "import sys,json;j=json.load(sys.stdin);print(j['status'],int(100*(j.get('progress') or 0)),j.get('src') or '-',(j.get('error') or '-').replace(' ','_'))")
  [ "$progress" != "$last" ] && [ $((progress % 10)) = 0 ] && echo "  $status $progress%" && last=$progress
  case "$status" in
    done) f="$HERE/workspace${src}"; cp "$f" "$dest/"; echo "$dest/$(basename "$f")"; exit 0;;
    failed|cancelled) echo "export $status: ${err//_/ }" >&2; exit 1;;
  esac
  sleep 5
done
