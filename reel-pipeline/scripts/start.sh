#!/usr/bin/env bash
# Starts the FableCut server in the background if it is not already up.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-7777}"
if curl -sf "localhost:$PORT/api/project" >/dev/null; then echo "FableCut already up on :$PORT"; exit 0; fi
[ -d "$HERE/.fablecut" ] || "$HERE/setup.sh"
nohup "$HERE/fablecut.sh" server.js > "$HERE/workspace/server.log" 2>&1 &
for _ in $(seq 1 40); do curl -sf "localhost:$PORT/api/project" >/dev/null && { echo "FableCut up on :$PORT (log: workspace/server.log)"; exit 0; }; sleep 0.5; done
echo "FableCut failed to start; see $HERE/workspace/server.log" >&2; exit 1
