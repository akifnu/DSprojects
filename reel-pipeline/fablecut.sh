#!/usr/bin/env bash
# Runs a FableCut entry point (server.js, mcp-server.js, analyze.js) with the
# pipeline's data dir and a headless-capable Chromium for export.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FABLECUT_DATA_DIR="${FABLECUT_DATA_DIR:-$HERE/workspace}"
export FABLECUT_CHROME="${FABLECUT_CHROME:-$HERE/scripts/chrome.sh}"
exec node "$HERE/.fablecut/${1:-server.js}" "${@:2}"
