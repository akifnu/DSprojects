#!/usr/bin/env bash
# Runs a FableCut entry point (server.js, mcp-server.js, analyze.js) with the
# pipeline's data dir. The editor's headless-browser features (export, composed
# timeline frames) are switched off: rendering is ffmpeg-only (scripts/render.sh).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FABLECUT_DATA_DIR="${FABLECUT_DATA_DIR:-$HERE/workspace}"
export FABLECUT_CHROME="/nonexistent/browser-disabled"
exec node "$HERE/.fablecut/${1:-server.js}" "${@:2}"
