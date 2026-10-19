#!/usr/bin/env bash
# Chromium wrapper for FableCut's headless export: adds --no-sandbox when
# running as root (cloud containers), which Chromium otherwise refuses.
for c in "${REEL_CHROME:-}" /opt/pw-browsers/chromium-*/chrome-linux/chrome "$(command -v chromium || true)" "$(command -v google-chrome || true)"; do
  [ -n "$c" ] && [ -x "$c" ] && exe="$c" && break
done
[ -z "${exe:-}" ] && { echo "no Chromium found" >&2; exit 1; }
extra=(--enable-unsafe-swiftshader --use-angle=swiftshader --ignore-gpu-blocklist)
[ "$(id -u)" = 0 ] && extra+=(--no-sandbox)
exec "$exe" "${extra[@]}" "$@"
