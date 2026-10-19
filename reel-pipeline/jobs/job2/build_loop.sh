#!/usr/bin/env bash
# Endless background, no transition: hard-cut loop made invisible by spreading the
# end→start correction (optical flow + shading) across the whole loop.
#   build_loop.sh <clip> <seconds> <out.mp4>
exec python3 "$(dirname "$0")/../../scripts/seamless_loop.py" "$@"
