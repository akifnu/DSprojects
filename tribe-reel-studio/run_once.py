"""Analyze one reel or photo from the command line (no web UI).

    python run_once.py path/to/reel.mp4
    python run_once.py path/to/photo.jpg --seconds 10
Results land in ./cache/reel_outputs/<name>_<timestamp>/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--seconds", type=float, default=10, help="how long to show a photo")
ap.add_argument("--start", type=float, default=0, help="start the clip at this second")
ap.add_argument("--clip", type=float, default=0, help="only analyze this many seconds (0 = all)")
ap.add_argument("--scale", type=float, default=app.DEFAULT_SCALE_MAX, help="color scale max")
ap.add_argument("--language", default="english", choices=app.LANGUAGES)
ap.add_argument("--demo", action="store_true")
args = ap.parse_args()
app.DEMO = args.demo
app.CACHE.mkdir(parents=True, exist_ok=True)

suffix = Path(args.path).suffix.lower()
is_image, is_audio = suffix in app.IMAGE_SUFFIXES, suffix in app.AUDIO_SUFFIXES
is_video = not (is_image or is_audio)
out = app.analyze(args.path if is_video else None, args.path if is_audio else None,
                  args.path if is_image else None, args.seconds,
                  args.start, args.clip, args.language, "Activation only (hot)", 55, args.scale, True,
                  progress=lambda frac, desc="": print(f"[{frac:4.0%}] {desc}", flush=True))
print(out[0])
print(out[7].to_string(index=False))
print("\n".join(out[8]))
