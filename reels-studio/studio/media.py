"""ffprobe/ffmpeg helpers, proxy generation, transcription (via reelkit) and bundled sounds."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import store

REELKIT = Path(__file__).resolve().parents[2] / "reel-pipeline"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


class MediaError(Exception):
    pass


def run(cmd, timeout=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-6:])
        raise MediaError(f"{Path(cmd[0]).name} failed: {tail}")
    return r.stdout


def probe(path):
    """Duration, dimensions, codecs, rotation-corrected orientation. Raises MediaError."""
    try:
        out = json.loads(run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], 60))
    except (MediaError, json.JSONDecodeError) as e:
        raise MediaError(f"Could not read this file as media ({e}).")
    v = next((s for s in out.get("streams", []) if s["codec_type"] == "video"), None)
    a = next((s for s in out.get("streams", []) if s["codec_type"] == "audio"), None)
    dur = float(out.get("format", {}).get("duration") or (v or a or {}).get("duration") or 0)
    info = {"duration": round(dur, 3), "has_audio": a is not None, "has_video": v is not None,
            "acodec": a and a.get("codec_name"), "vcodec": v and v.get("codec_name")}
    if v:
        w, h = int(v.get("width", 0)), int(v.get("height", 0))
        rot = 0
        for sd in v.get("side_data_list", []):
            rot = int(sd.get("rotation", 0) or 0)
        if abs(rot) in (90, 270):
            w, h = h, w
        fr = v.get("avg_frame_rate", "0/1").split("/")
        fps = float(fr[0]) / float(fr[1]) if len(fr) == 2 and float(fr[1]) else 0
        info.update(width=w, height=h, fps=round(fps, 3),
                    orientation="portrait" if h > w else "landscape" if w > h else "square")
    return info


def validate_source(path):
    if Path(path).suffix.lower() not in VIDEO_EXT:
        raise MediaError("Unsupported file type. Upload an MP4 or MOV video.")
    info = probe(path)
    if not info["has_video"]:
        raise MediaError("This file has no video stream.")
    if info["duration"] <= 0.5:
        raise MediaError("This video is too short or its duration could not be read.")
    if info["duration"] > 60 * 60:
        raise MediaError("Videos longer than 60 minutes are not supported in this draft.")
    return info


def make_proxy(src, dst):
    """Small, seek-friendly H.264 proxy (keyframe every 0.5 s) for browser playback."""
    tmp = Path(dst).with_suffix(".tmp.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", "scale='min(720,iw)':-2", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "27", "-force_key_frames", "expr:gte(t,n_forced*0.5)",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)])
    tmp.replace(dst)


def transcribe(src, cache):
    """Words with stable ids. Uses reelkit's ASR (Parakeet/Whisper via sherpa-onnx)."""
    if str(REELKIT) not in sys.path:
        sys.path.insert(0, str(REELKIT))
    from reelkit import asr  # noqa: E402  (heavy import, only in the worker)
    tr = asr.transcribe(str(src), lang="en", cache=str(cache) + ".asr")
    words = [{"id": f"w{i}", "w": w["w"], "s": round(w["s"], 3), "e": round(w["e"], 3),
              "conf": w.get("c", 1.0)} for i, w in enumerate(tr["words"])]
    # A word whose span is implausible (too short/long for its length) is flagged uncertain.
    for w in words:
        d = w["e"] - w["s"]
        w["uncertain"] = d < 0.03 or d > 0.25 + 0.12 * len(w["w"]) or w["conf"] < 0.5
    sents = [{"id": s["id"], "w0": s["w0"], "w1": s["w1"], "s": s["s"], "e": s["e"]} for s in tr["sentences"]]
    return {"words": words, "sentences": sents, "lang": tr.get("lang"), "timing": tr.get("timing")}


def font_available(family):
    if not shutil.which("fc-list"):
        return True
    out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True).stdout
    fams = {f.strip().lower() for line in out.splitlines() for f in line.split(",")}
    return family.lower() in fams


def thumb(src, dst, at=0.5):
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(at), "-i", str(src), "-frames:v", "1",
         "-vf", "scale=320:-2", str(dst)])


BUNDLED = {
    "Pop": "aevalsrc='0.6*sin(2*PI*t*(900-6000*t))*exp(-25*t)':d=0.18",
    "Whoosh": "anoisesrc=d=0.6:c=pink:a=0.5,afade=t=in:d=0.3,afade=t=out:st=0.3:d=0.3,lowpass=f=2500",
    "Ding": "aevalsrc='0.5*sin(2*PI*1320*t)*exp(-4*t)+0.25*sin(2*PI*1980*t)*exp(-6*t)':d=1.2",
    "Soft bed (loop)": "aevalsrc='0.08*sin(2*PI*220*t)+0.06*sin(2*PI*277*t)+0.05*sin(2*PI*330*t)':d=8,"
                       "afade=t=in:d=0.5,afade=t=out:st=7.5:d=0.5",
}


def ensure_bundled_sounds():
    """Synthesise a few royalty-free bundled sounds once (R1: bundled + uploaded sounds)."""
    have = {a["name"] for a in store.assets() if a.get("bundled")}
    for name, expr in BUNDLED.items():
        if name in have:
            continue
        aid = store.new_id("ast")
        f = f"{aid}.m4a"
        try:
            run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", expr, "-ac", "2", "-ar", "48000",
                 "-c:a", "aac", str(store.asset_dir() / f)])
        except MediaError:
            continue
        info = probe(store.asset_dir() / f)
        store.add_asset({"id": aid, "name": name, "type": "audio", "file": f, "duration": info["duration"],
                         "tags": ["bundled"], "rights": "Bundled, royalty-free", "bundled": True,
                         "created": store.now()})
