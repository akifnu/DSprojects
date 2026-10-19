"""Where things live: tools, models, the data store."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # reel-pipeline/
DATA = Path(os.environ.get("REELKIT_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("REELKIT_MODELS", ROOT / ".models"))
TOOLS = ROOT / ".tools"
LIBRARY = ROOT / "library"                              # committed fonts list, synthesized sfx, …
FONTS = DATA / "fonts"                                  # downloaded Google Fonts cache


def _tool(name):
    """ffmpeg 8.1 static build from setup.sh first: its threaded pipeline renders
    ~30 % faster than distro ffmpeg 6.1 and it ships libvmaf, libass, zscale."""
    env = os.environ.get(f"REELKIT_{name.upper()}")
    if env:
        return env
    local = TOOLS / "ffmpeg" / "bin" / name
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found: run reel-pipeline/setup.sh")
    return found


FFMPEG = _tool("ffmpeg")
FFPROBE = _tool("ffprobe")


def data(*parts):
    p = DATA.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
