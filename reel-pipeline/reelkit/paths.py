"""Where things live: tools, models, the data store.

DATA (REELKIT_DATA) and MODELS (REELKIT_MODELS, the same variable setup.sh installs into)
are plain paths. FFMPEG / FFPROBE are resolved on first use: importing them stops the
importing module with SystemExit when the tool is missing, while `reel check` imports this
module, asks tool() and reports the missing tool itself.
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # reel-pipeline/
DATA = Path(os.environ.get("REELKIT_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("REELKIT_MODELS", ROOT / ".models"))
TOOLS = ROOT / ".tools"
LIBRARY = ROOT / "library"                              # committed fonts list, synthesized sfx, …
FONTS = DATA / "fonts"                                  # downloaded Google Fonts cache


def tool(name):
    """ffmpeg / ffprobe: REELKIT_FFMPEG / REELKIT_FFPROBE, else the static ffmpeg 9 build from
    setup.sh (its threaded pipeline renders ~30 % faster than distro ffmpeg 6.1 and it ships
    libvmaf, zscale and NVENC), else PATH; None when there is none."""
    env = os.environ.get(f"REELKIT_{name.upper()}")
    if env:
        return env
    local = TOOLS / "ffmpeg" / "bin" / name
    if local.exists():
        return str(local)
    return shutil.which(name)


def __getattr__(name):
    if name in ("FFMPEG", "FFPROBE"):
        found = tool(name.lower())
        if not found:
            raise SystemExit(f"{name.lower()} not found: run reel-pipeline/setup.sh")
        globals()[name] = found
        return found
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def data(*parts):
    p = DATA.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
