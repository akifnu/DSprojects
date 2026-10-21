"""Bounded, cached still-frame previews before any editing-copy or video render.

Reuse the timeline compiler and compositor with original media, cached transcripts,
no audio preparation, no tracking pass and no whole-source colour measurement.
Approval binds the plan, source identities, fonts, code and every preview artifact.
"""
import copy
import html
import math
import secrets
import shutil
import time
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw

from . import asr, compile as compiler, ff, fsutil, plan as planlib, render, store

DEFAULTS = {"width": 360, "frames_per_reel": 3, "max_frames": 240, "timeout_s": 120}
VERSION = 1
LIMITS = "Static layout preview; audio, animation, smooth subject tracking and final colour matching are reviewed after rendering."


class StoryboardError(ValueError):
    pass


def options(value=None):
    o = dict(DEFAULTS, **(value or {}))
    for k, lo, hi in (("width", 160, 720), ("frames_per_reel", 1, 8), ("max_frames", 1, 480)):
        if isinstance(o[k], bool) or not isinstance(o[k], int) or not lo <= o[k] <= hi:
            raise StoryboardError(f"storyboard.{k} must be an integer {lo}–{hi}")
    if not isinstance(o["timeout_s"], (int, float)) or not math.isfinite(o["timeout_s"]) or not 1 <= o["timeout_s"] <= 600:
        raise StoryboardError("storyboard.timeout_s must be 1–600 seconds")
    return o


def _identity(a):
    s = a.path.stat()
    return {"sha256": a.meta["sha256"], "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def _inputs(plan_path, config=None, provider="cpu"):
    plan = planlib.load(plan_path)
    style = compiler.load_style(plan["style"])
    lang = compiler._lang(plan, style)
    assets = {k: store.get(v) for k, v in plan["sources"].items()}
    transcripts = {}
    for k, a in assets.items():
        if not a.info.get("has_audio"):
            transcripts[k] = asr.empty(lang, a.info.get("duration"), provider)
            continue
        path = a.cache(f"transcript.{lang}.json")
        tr = fsutil.read_json(path)
        if not isinstance(tr, dict) or not asr.current(tr.get("fingerprint"), lang, provider):
            raise StoryboardError(f"source {k} needs a current transcript before storyboarding")
        transcripts[k] = tr
    extra = {}
    if isinstance(plan.get("music"), dict) and plan["music"].get("asset"):
        a = store.get(plan["music"]["asset"])
        extra["music"] = _identity(a)
    for k, ref in (plan.get("sfx") or {}).items():
        extra[f"sfx:{k}"] = _identity(store.get(ref))
    caption_style = (style.get("captions") or {}).get("style") or {}
    title_style = (style.get("titles") or {}).get("style") or caption_style
    fonts = compiler._font_deps({"styles": {"caption": caption_style, "title": title_style},
                                "captions": [{"style": "caption"}], "text": [{"style": "title"}]})
    code = {name: fsutil.sha256_file(Path(__file__).with_name(name)) for name in
            ("storyboard.py", "compile.py", "plan.py", "render.py", "captions.py", "grade.py", "reframe.py", "ff.py")}
    shared = {"v": VERSION, "style": fsutil.sha256_json(style), "fonts": fonts, "code": code,
              "options": options(config), "provider": provider, "lang": lang, "extra": extra,
              "output": dict(plan.get("output") or {}, lang=lang)}
    source_ids = {k: {"media": _identity(a), "transcript": fsutil.sha256_json(transcripts[k])}
                  for k, a in assets.items()}
    key = fsutil.sha256_json({"plan": fsutil.sha256_json(plan), "sources": source_ids, "shared": shared})
    return plan, style, assets, transcripts, source_ids, shared, key


def fingerprint(plan_path, config=None, provider="cpu"):
    return _inputs(plan_path, config, provider)[-1]


def intact(record):
    """A digest alone never approves changed, missing or corrupt preview files."""
    if not isinstance(record, dict) or not record.get("files"):
        return False
    try:
        return all(Path(p).is_file() and fsutil.sha256_file(p) == sha for p, sha in record["files"].items())
    except OSError:
        return False


def _times(tl, n):
    dur, fps = tl["duration"], tl["fps"]
    # Hook, middle and payoff; more panels sample actual shot centres as well.
    want = [min(0.5, dur * 0.15)] if n == 1 else [min(0.5, dur * 0.15), max(0, dur - 0.3)]
    shots = tl.get("video") or []
    centres = [c["start"] + c["dur"] / 2 for c in shots]
    while len(want) < n:
        candidates = centres + [dur * i / (n + 1) for i in range(1, n + 1)]
        candidate = max(candidates, key=lambda t: min(abs(t - x) for x in want))
        if min(abs(candidate - x) for x in want) < 1 / fps:
            break
        want.append(candidate)
    return sorted({max(0, min(round(t * fps) / fps, max(0, dur - 1 / fps))) for t in want})


def _frame(tl, t, path, timeout):
    # Sparse subject framing on the selected still only, never a 5-fps tracking pass.
    from . import reframe
    frame_tl = copy.deepcopy(tl)
    for clip in frame_tl.get("video") or []:
        if not clip["start"] <= t < clip["start"] + clip["dur"]:
            continue
        media = frame_tl["media"][clip["media"]]
        info = ff.probe(media, timeout=min(timeout, 10))
        w = min(640, info["width"])
        h = max(2, round(info["height"] * w / info["width"] / 2) * 2)
        at = clip["in"] + t - clip["start"]
        # A missing detector is a labelled centred preview, not a model download.
        if reframe.YUNET.exists():
            cmd = ff.frames_cmd(media, w, h, start=at, duration=0.05)
            raw, _ = ff._capture(cmd, "storyboard face sample", timeout=min(timeout, 10))
            size = w * h * 3
            if len(raw) < size:
                raise StoryboardError(f"could not decode a storyboard frame at {at:.2f}s")
            img = np.frombuffer(raw[:size], np.uint8).reshape(h, w, 3)
            faces = reframe.detect_image(img)
            if faces:
                face = max(faces, key=lambda f: f[2] * f[3] * f[4])
                cam = [(0.0, face[0], 0.5), (clip["dur"], face[0], 0.5)]
                sc = lambda tt: render.kf(clip.get("scale"), tt, 1)
                x, _ = reframe.offsets(cam, info["width"], info["height"], (tl["width"], tl["height"]), sc,
                                     anchor=(0.5, 0.5))
                clip["x"] = x
    render.still(frame_tl, t, path, timeout_s=timeout)


def _sheet(reels, path):
    count = sum(len(r["frames"]) for r in reels)
    thumb_w, thumb_h, cell_w, cell_h = 180, 320, 208, 360
    cols = min(6, max(1, count))
    sheet = Image.new("RGB", (cols * cell_w, math.ceil(count / cols) * cell_h), "#111827")
    draw = ImageDraw.Draw(sheet)
    i = 0
    for r in reels:
        for f in r["frames"]:
            with Image.open(f["file"]) as img:
                img.thumbnail((thumb_w, thumb_h))
                x, y = (i % cols) * cell_w + 14, (i // cols) * cell_h + 8
                sheet.paste(img, (x + (thumb_w - img.width) // 2, y))
                draw.text((x, y + thumb_h + 6), f"{r['id']}  {f['t']:.2f}s", fill="white")
            i += 1
    sheet.save(path, quality=85)


def _page(reels, path, estimated, frames):
    esc = html.escape
    blocks = []
    for r in reels:
        cards = "".join(f'<figure><img loading="lazy" src="{esc(Path(f["file"]).relative_to(path.parent).as_posix())}" '
                        f'alt="{esc(r["id"])} at {f["t"]:.2f} seconds"><figcaption>{f["t"]:.2f}s</figcaption></figure>'
                        for f in r["frames"])
        blocks.append(f'<section><h2>{esc(r["id"])} · {esc(r["title"])}</h2>'
                      f'<p>{r["duration"]:.1f}s · {r["shots"]} planned shots · {len(r["frames"])} preview frames</p>'
                      f'<div class="frames">{cards}</div><details><summary>Read the planned speech</summary>'
                      f'<p>{esc(r["speech"])}</p></details></section>')
    fsutil.atomic_write_text(path, '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Review your storyboard</title>'
        '<style>body{margin:0 auto;padding:32px;max-width:1100px;background:#0b1020;color:#eef2ff;'
        'font:17px/1.6 system-ui}h1,h2{line-height:1.2}.frames{display:flex;gap:16px;overflow-x:auto}'
        'figure{margin:0;flex:0 0 240px}img{width:100%;border-radius:12px}figcaption{color:#a5b4fc}'
        'section{padding:28px 0;border-top:1px solid #334155}aside{padding:18px;background:#172033;'
        'border-radius:12px}summary{cursor:pointer}</style><h1>Review your storyboard</h1>'
        '<aside>Full rendering is paused. Approve these frames or describe the changes you want.</aside>'
        f'<p>{frames} stills generated instead of approximately {estimated:,} full-render frames.</p>'
        f'<p>{esc(LIMITS)}</p>' + ''.join(blocks) + '</html>')


def build(plan_path, root, config=None, provider="cpu"):
    started = time.monotonic()
    plan, style, assets, transcripts, source_ids, shared, key = _inputs(plan_path, config, provider)
    o = shared["options"]
    if len(plan["reels"]) * o["frames_per_reel"] > o["max_frames"]:
        raise StoryboardError(f"storyboard exceeds {o['max_frames']} stills; split the plan into smaller jobs")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    board = root / f"board-{key[:12]}-{secrets.token_hex(3)}"
    board.mkdir()
    reels, files, estimated, hits = [], {}, 0, 0
    for reel in plan["reels"]:
        if time.monotonic() - started >= o["timeout_s"]:
            raise StoryboardError("storyboard time budget exhausted; use a smaller plan")
        used = compiler._reel_sources(reel)
        rk = fsutil.sha256_json({"shared": shared, "reel": reel, "sources": {k: source_ids[k] for k in used}})
        cache = root / "cache" / rk
        record = fsutil.read_json(cache / "cache.json")
        if intact(record):
            r = record["reel"]
            hits += 1
        else:
            cache.mkdir(parents=True, exist_ok=True)
            small = copy.deepcopy(plan)
            out = small.setdefault("output", {})
            width, height = out.get("width", 720), out.get("height", 1280)
            scale = min(1, o["width"] / width, 640 / height)
            out.update(width=max(2, round(width * scale / 2) * 2), height=max(2, round(height * scale / 2) * 2))
            copies = {k: [{"s": 0, "e": a.info["duration"], "path": str(a.path),
                           "width": a.info["width"], "height": a.info["height"], "fps": a.info["fps"]}]
                      for k, a in assets.items()}
            tl_path = compiler.compile_reel(small, reel, style, cache, transcripts=transcripts, copies=copies,
                       envelopes={}, reframing=False, preview=True, provider=provider)
            tl = fsutil.read_json(tl_path)
            frames = []
            for i, t in enumerate(_times(tl, o["frames_per_reel"])):
                left = o["timeout_s"] - (time.monotonic() - started)
                if left <= 0:
                    raise StoryboardError("storyboard time budget exhausted; use a smaller plan")
                f = cache / f"frame-{i:02d}.jpg"
                _frame(tl, t, f, left)
                frames.append({"file": str(f.resolve()), "t": t})
            r = {"id": reel["id"], "title": reel.get("title") or reel.get("title_text") or reel["id"],
                 "duration": tl["duration"], "fps": tl["fps"], "shots": len(tl.get("video") or []),
                 "speech": " ".join(tl["meta"]["words"]), "frames": frames}
            hashes = {str(Path(tl_path).resolve()): fsutil.sha256_file(tl_path)}
            hashes.update({f["file"]: fsutil.sha256_file(f["file"]) for f in frames})
            fsutil.atomic_write_json(cache / "cache.json", {"files": hashes, "reel": r})
        # Board files are independent copies, so a later cache repair cannot alter
        # the frames a user is currently reviewing.
        r = copy.deepcopy(r)
        for i, f in enumerate(r["frames"]):
            dest = board / f"{reel['id']}-{i:02d}.jpg"
            shutil.copyfile(f["file"], dest)
            f["file"] = str(dest.resolve())
            files[f["file"]] = fsutil.sha256_file(dest)
        reels.append(r)
        estimated += round(r["duration"] * r["fps"])
    total = sum(len(r["frames"]) for r in reels)
    sheet, page, manifest = board / "contact-sheet.jpg", board / "index.html", board / "manifest.json"
    _sheet(reels, sheet)
    _page(reels, page, estimated, total)
    fsutil.atomic_write_json(manifest, {"version": VERSION, "key": key, "reels": reels, "limitations": LIMITS,
                           "preview_frames": total, "estimated_render_frames": estimated, "cache_hits": hits})
    for p in (sheet, page, manifest):
        files[str(p.resolve())] = fsutil.sha256_file(p)
    return {"key": key, "sha256": fsutil.sha256_file(manifest), "manifest": str(manifest.resolve()),
            "page": str(page.resolve()), "sheet": str(sheet.resolve()), "files": files,
            "frames": total, "cache_hits": hits, "estimated_render_frames": estimated,
            "elapsed_s": round(time.monotonic() - started, 2), "limitations": LIMITS}
