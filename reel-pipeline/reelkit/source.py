"""Long-form sources: editing copies, lossless audio, transcript, colour look.

Decoding 4K / HEVC / 60 fps camera files is the single most expensive thing in the
pipeline (≈ 0.5× real time on 4 cores), so it happens once per needed range:

  prepare(asset, ranges)   editing copies ("mezzanine") of the merged ranges plus
                           1 s handles — 30 fps, short side 1080, near-lossless short-GOP
                           H.264 (CRF 12, keyframe every 0.5 s: every seek decodes ≤ 15
                           frames), video only. Cached under data/assets/<id>/mezz/ and
                           reused by any later request a cached copy already covers.
  audio(asset)             the whole soundtrack as 48 kHz FLAC (lossless; ~1 s per hour)
  transcript(asset, lang)  asr.transcribe, cached as transcript.<lang>.json
  look(asset)              colour statistics of the footage (grade.Look), cached

locate(copies, t) maps a source time to (editing copy, offset into it).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

from . import asr, ff, grade
from .paths import FFMPEG

HANDLE = 1.0          # s of footage kept either side of every requested range
MERGE_GAP = 4.0       # ranges closer than this share one editing copy
WHOLE_FILE = 0.6      # if the ranges cover more than this fraction, copy the whole source


def merge_ranges(ranges, duration, handle=HANDLE, gap=MERGE_GAP):
    """Sorted, merged (s, e) spans with handles, clipped to the source."""
    spans = sorted((max(0.0, s - handle), min(duration, e + handle)) for s, e in ranges)
    out = []
    for s, e in spans:
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in out]


def _index_path(asset):
    return asset.cache("mezz/index.json")


def _load_index(asset):
    p = _index_path(asset)
    if p.exists():
        idx = json.loads(p.read_text())
        return [m for m in idx if os.path.exists(m["path"])]
    return []


def _geometry(info, short_side):
    w, h = info["width"], info["height"]
    k = min(1.0, short_side / min(w, h))           # never upscale
    return int(round(w * k / 2)) * 2, int(round(h * k / 2)) * 2


def _encode(src, info, s, e, out, fps, size, threads):
    w, h = size
    vf = [f"fps={fps}", f"scale={w}:{h}:flags=lanczos", "setsar=1"]
    cs = info.get("colorspace")
    if cs == "bt709" or (cs in (None, "", "unknown") and min(info["width"], info["height"]) >= 720):
        vf[1] += ":in_color_matrix=bt709:out_color_matrix=bt709"
    tmp = str(out) + ".part.mp4"
    ff.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-threads", str(threads),
            "-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i", str(src), "-an", "-vf", ",".join(vf),
            "-c:v", "libx264", "-preset", "faster", "-crf", "12", "-g", str(max(1, fps // 2)),
            "-keyint_min", str(max(1, fps // 2)), "-bf", "0", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-threads", str(threads), "-movflags", "+faststart", tmp], name=f"editing copy {s:.1f}-{e:.1f}s")
    os.replace(tmp, out)


def prepare(asset, ranges=None, fps=30, short_side=1080, handle=HANDLE, workers=2):
    """Editing copies covering `ranges` (source seconds; None = whole file).
    Returns [{"s", "e", "path", "width", "height", "fps"}] sorted by s."""
    info = asset.info
    dur = info["duration"]
    want = merge_ranges(ranges or [(0, dur)], dur, handle)
    if sum(e - s for s, e in want) > WHOLE_FILE * dur:
        want = [(0.0, round(dur, 3))]
    size = _geometry(info, short_side)
    have = _load_index(asset)

    def covered(s, e):
        for m in have:
            if m["fps"] == fps and (m["width"], m["height"]) == size and m["s"] <= s + 1e-3 and m["e"] >= e - 1e-3:
                return m
        return None
    todo = [(s, e) for s, e in want if covered(s, e) is None]
    if todo:
        threads = max(1, (os.cpu_count() or 4) // max(1, min(workers, len(todo))))
        jobs = []
        for s, e in todo:
            out = asset.cache(f"mezz/{s:09.3f}_{e:09.3f}_{size[1]}p{fps}.mp4")
            jobs.append({"s": s, "e": e, "path": str(out), "width": size[0], "height": size[1], "fps": fps})
        with ThreadPoolExecutor(max(1, min(workers, len(jobs)))) as pool:
            list(pool.map(lambda j: _encode(asset.path, info, j["s"], j["e"], j["path"], fps, size, threads), jobs))
        have = _load_index(asset) + jobs
        _index_path(asset).write_text(json.dumps(have, indent=1))
    return sorted((covered(s, e) for s, e in want), key=lambda m: m["s"])


def locate(copies, s, e=None):
    """(copy, offset) of the editing copy holding source time s (and e, if given)."""
    e = s if e is None else e
    for m in copies:
        if m["s"] - 1e-3 <= s and e <= m["e"] + 1e-3:
            return m, round(s - m["s"], 4)
    raise KeyError(f"no editing copy covers {s:.3f}-{e:.3f}s")


def audio(asset):
    """Lossless 48 kHz stereo FLAC of the whole source (cached)."""
    out = asset.cache("audio.flac")
    if not out.exists():
        tmp = str(out) + ".part.flac"
        ff.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(asset.path), "-vn",
                "-ac", "2", "-ar", "48000", "-c:a", "flac", tmp], name="source audio")
        os.replace(tmp, out)
    return str(out)


def transcript(asset, lang="en"):
    path = asset.cache(f"transcript.{lang}.json")
    src = audio(asset) if asset.info.get("has_audio") else str(asset.path)
    return asr.transcribe(src, lang, cache=str(path))


def look(asset, copies=None, fps=0.5, width=320):
    """Colour statistics of the footage (for grade transfer), from editing copies if given."""
    out = asset.cache("look.json")
    if out.exists():
        return json.loads(out.read_text())
    L = grade.Look()
    sources = [(m["path"], m["width"], m["height"]) for m in copies] if copies else \
        [(str(asset.path), asset.info["width"], asset.info["height"])]
    for path, w, h in sources:
        hh = int(round(h * width / w / 2)) * 2
        for f in ff.frames(path, width, hh, fps=fps):
            L.add(f)
    stats = L.stats()
    out.write_text(json.dumps(stats))
    return stats
