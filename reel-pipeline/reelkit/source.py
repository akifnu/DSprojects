"""Long-form sources: editing copies, lossless audio, transcript, colour look.

Decoding 4K / HEVC / 60 fps camera files is the single most expensive thing in the
pipeline (≈ 0.5× real time on 4 cores), so it happens once per needed range:

  prepare(asset, ranges)   editing copies ("mezzanine") of the merged ranges plus
                           1 s handles — 30 fps, short side 1080, near-lossless short-GOP
                           H.264 (ultrafast CRF 10: decoding the camera file is ~half the
                           CPU, this keeps encoding cheap; keyframe every 0.5 s so a seek
                           decodes ≤ 15 frames), video only. Cached under data/assets/<id>/mezz/ and
                           reused by any later request a cached copy already covers.
  audio(asset)             the whole soundtrack as 48 kHz FLAC (lossless; ~1 s per hour)
  transcript(asset, lang)  asr.transcribe, cached as transcript.<lang>.json and redone when
                           its ASR fingerprint (models, language, provider, config) is stale
  look(asset, copies)      colour statistics of the footage (grade.Look), cached per measured
                           footage (look.<key>.json: the copies' ranges and geometry)

Every cache is written to a unique temp name and renamed into place, under a lock per
artifact, so concurrent jobs preparing the same source neither duplicate work nor read
a half-written file; a missing or corrupt cache entry is a miss, not an error.

locate(copies, t) maps a source time to (editing copy, offset into it).
"""
import hashlib
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from . import asr, ff, fsutil, grade
from .paths import FFMPEG

HANDLE = 1.0          # s of footage kept either side of every requested range
MERGE_GAP = 4.0       # ranges closer than this share one editing copy
WHOLE_FILE = 0.6      # if the ranges cover more than this fraction, copy the whole source
LOOK_VERSION = 1      # bump when look() samples the footage differently


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


def _temp(final, suffix):
    """A unique, empty temp file beside `final` (ffmpeg picks the format from `suffix`)."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(str(final)), prefix=os.path.basename(str(final)) + ".",
                               suffix=suffix)
    os.close(fd)
    return tmp


def _discard(tmp):
    try:
        os.unlink(tmp)
    except OSError:
        pass


def _sweep(folder, pattern):
    """Delete partial outputs a crashed run left behind; call only under the lock every writer
    of these outputs holds, so none of them is in flight."""
    for p in Path(folder).glob(pattern):
        _discard(p)


# ------------------------------------------------------------------ editing copies

def _index_path(asset):
    return asset.cache("mezz/index.json")


def _load_index(asset):
    """Editing copies on record whose files exist. Entries name their file relative to mezz/
    (older indexes held absolute paths: their basename is looked up there), so a moved data
    volume keeps its copies. A missing or corrupt index is empty."""
    p = _index_path(asset)
    idx = fsutil.read_json(p, default=[]) if p.exists() else []
    out = []
    for m in idx if isinstance(idx, list) else []:
        try:
            f = os.path.basename(str(m.get("file") or m["path"]))
            ent = {"s": float(m["s"]), "e": float(m["e"]), "path": str(p.parent / f), "width": int(m["width"]),
                   "height": int(m["height"]), "fps": m["fps"]}
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if f and os.path.exists(ent["path"]):
            out.append(ent)
    return out


def _save_index(asset, entries):
    seen, rows = set(), []
    for m in entries:
        f = os.path.basename(m["path"])
        if f not in seen:
            seen.add(f)
            rows.append({"s": m["s"], "e": m["e"], "file": f, "width": m["width"], "height": m["height"],
                         "fps": m["fps"]})
    fsutil.atomic_write_json(_index_path(asset), rows)


def _covering(have, s, e, fps, size):
    for m in have:
        if m["fps"] == fps and (m["width"], m["height"]) == tuple(size) and m["s"] <= s + 1e-3 and m["e"] >= e - 1e-3:
            return m
    return None


def _cover(have, want, fps, size):
    """The copies covering every wanted span (each once, by start), or None if one is missing."""
    got = {}
    for s, e in want:
        m = _covering(have, s, e, fps, size)
        if m is None:
            return None
        got[m["path"]] = m
    return sorted(got.values(), key=lambda m: m["s"])


def _geometry(info, short_side):
    w, h = info["width"], info["height"]
    k = min(1.0, short_side / min(w, h))           # never upscale
    return int(round(w * k / 2)) * 2, int(round(h * k / 2)) * 2


def _encode(src, info, s, e, out, fps, size, threads):
    w, h = size
    # explicit colour interpretation: an untagged source would otherwise be converted
    # implicitly when the copy is tagged BT.709 (measured: luma shifted by 1 level)
    cs = info.get("colorspace")
    hd = min(info["width"], info["height"]) >= 720
    src_matrix = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "bt2020nc": "bt2020"}.get(
        cs, "bt709" if hd else "bt601")
    src_range = "pc" if info.get("color_range") in ("pc", "jpeg") else "tv"
    vf = [f"fps={fps}", f"scale={w}:{h}:flags=lanczos+accurate_rnd+full_chroma_int:in_color_matrix={src_matrix}:"
                        f"out_color_matrix=bt709:"
                        f"in_range={src_range}:out_range=tv", "setsar=1"]
    tmp = _temp(out, ".part.mp4")
    try:
        ff.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-threads", str(threads),
                "-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i", str(src), "-an", "-vf", ",".join(vf),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "10", "-g", str(max(1, fps // 2)),
                "-keyint_min", str(max(1, fps // 2)), "-bf", "0", "-pix_fmt", "yuv420p",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
                "-threads", str(threads), "-movflags", "+faststart", tmp], name=f"editing copy {s:.1f}-{e:.1f}s")
        os.replace(tmp, out)
    except BaseException:
        _discard(tmp)
        raise


def prepare(asset, ranges=None, fps=30, short_side=1080, handle=HANDLE, workers=2):
    """Editing copies covering `ranges` (source seconds; None = whole file).
    Returns [{"s", "e", "path", "width", "height", "fps"}] sorted by s."""
    info = asset.info
    dur = info["duration"]
    if not dur or dur <= 0:
        raise ValueError(f"asset {asset.id}: unknown duration, no editing copies can be cut")
    want = merge_ranges(ranges or [(0, dur)], dur, handle)
    if sum(e - s for s, e in want) > WHOLE_FILE * dur:
        want = [(0.0, round(dur, 3))]
    size = _geometry(info, short_side)
    got = _cover(_load_index(asset), want, fps, size)
    if got is not None:
        return got
    # one preparer per source at a time: whoever waited finds the copies it needs already made
    with fsutil.file_lock(_index_path(asset)):
        have = _load_index(asset)
        todo = [(s, e) for s, e in want if _covering(have, s, e, fps, size) is None]
        if todo:
            _sweep(_index_path(asset).parent, "*.part.mp4")
            threads = max(1, (os.cpu_count() or 4) // max(1, min(workers, len(todo))))
            jobs = [{"s": s, "e": e, "path": str(asset.cache(f"mezz/{s:09.3f}_{e:09.3f}_{size[1]}p{fps}.mp4")),
                     "width": size[0], "height": size[1], "fps": fps} for s, e in todo]
            with ThreadPoolExecutor(max(1, min(workers, len(jobs)))) as pool:
                futs = [(j, pool.submit(_encode, asset.path, info, j["s"], j["e"], j["path"], fps, size, threads))
                        for j in jobs]
            done, err = [], None
            for j, f in futs:
                try:
                    f.result()
                    done.append(j)
                except BaseException as e:      # keep the copies that did finish on record
                    err = err or e
            if done:
                have = [m for m in have if m["path"] not in {j["path"] for j in done}] + done
                _save_index(asset, have)
            if err is not None:
                raise err
        got = _cover(have, want, fps, size)
    if got is None:
        raise RuntimeError(f"asset {asset.id}: editing copies do not cover {want}")
    return got


def locate(copies, s, e=None):
    """(copy, offset) of the editing copy holding source time s (and e, if given)."""
    e = s if e is None else e
    for m in copies:
        if m["s"] - 1e-3 <= s and e <= m["e"] + 1e-3:
            return m, round(s - m["s"], 4)
    raise KeyError(f"no editing copy covers {s:.3f}-{e:.3f}s")


# ------------------------------------------------------------------ audio, transcript

def audio(asset):
    """Lossless 48 kHz stereo FLAC of the whole source (cached)."""
    out = asset.cache("audio.flac")
    if out.exists():
        return str(out)
    with fsutil.file_lock(out):
        if not out.exists():
            _sweep(out.parent, "audio.flac*.part.flac")
            tmp = _temp(out, ".part.flac")
            try:
                ff.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(asset.path), "-vn",
                        "-ac", "2", "-ar", "48000", "-c:a", "flac", tmp], name="source audio")
                os.replace(tmp, out)
            except BaseException:
                _discard(tmp)
                raise
    return str(out)


def transcript(asset, lang="en", provider=None):
    """Words + sentences of the source's speech (asr.transcribe, cached per language; see
    asr.transcribe for `provider`). A source without an audio stream has none."""
    if not asset.info.get("has_audio"):
        return asr.empty(lang, asset.info.get("duration"), provider)
    return asr.transcribe(audio(asset), lang, cache=str(asset.cache(f"transcript.{lang}.json")), provider=provider)


# ------------------------------------------------------------------ look

@lru_cache(maxsize=1)
def _grade_code():
    try:
        return hashlib.sha256(Path(grade.__file__).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def look(asset, copies=None, fps=0.5, width=320):
    """Colour statistics of the footage (for grade transfer), from editing copies if given.
    Cached per measured footage: the same copies (by range and geometry) and sampling reuse
    it, other copies of the same source are measured on their own."""
    seen, sources, spans = set(), [], []
    for m in sorted(copies or [], key=lambda m: (m["s"], m["e"])):
        if m["path"] not in seen:
            seen.add(m["path"])
            sources.append((m["path"], m["width"], m["height"]))
            spans.append([round(float(m["s"]), 3), round(float(m["e"]), 3), m["width"], m["height"], m.get("fps")])
    if not sources:
        sources = [(str(asset.path), asset.info["width"], asset.info["height"])]
    key = fsutil.sha256_json({"spans": spans or "original", "fps": fps, "width": width, "grade": _grade_code(),
                              "v": LOOK_VERSION})[:16]
    out = asset.cache(f"look.{key}.json")

    def cached():
        st = fsutil.read_json(out) if out.exists() else None
        return st if isinstance(st, dict) and st else None
    stats = cached()
    if stats is not None:
        return stats
    with fsutil.file_lock(out):
        stats = cached()
        if stats is None:
            L = grade.Look()
            for path, w, h in sources:
                hh = int(round(h * width / w / 2)) * 2
                for f in ff.frames(path, width, hh, fps=fps):
                    L.add(f)
            stats = L.stats()
            fsutil.atomic_write_json(out, stats)
    return stats
