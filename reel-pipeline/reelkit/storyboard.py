"""Bounded, cached storyboard stills and GIFs before full media preparation/rendering.

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

DEFAULTS = {"width": 360, "frames_per_reel": 3, "max_frames": 240, "timeout_s": 120,
            "gif_width": 240, "gif_fps": 6, "gif_seconds": 3, "max_gif_frames": 1440,
            "max_gif_bytes": 2 * 1024 * 1024}
VERSION = 2
LIMITS = ("Silent, low-resolution preview. GIFs show selected moments at reduced frame rate, "
          "with a fixed subject crop within each excerpt. Audio, full-speed motion, smooth "
          "subject tracking and final colour matching need the approved render.")


class StoryboardError(ValueError):
    pass


def options(value=None):
    o = dict(DEFAULTS, **(value or {}))
    for k, lo, hi in (("width", 160, 720), ("frames_per_reel", 1, 8), ("max_frames", 1, 480),
                     ("gif_width", 160, 360), ("gif_fps", 2, 10), ("max_gif_frames", 1, 4800),
                     ("max_gif_bytes", 1024, 8 * 1024 * 1024)):
        if isinstance(o[k], bool) or not isinstance(o[k], int) or not lo <= o[k] <= hi:
            raise StoryboardError(f"storyboard.{k} must be an integer {lo}–{hi}")
    for k, lo, hi in (("timeout_s", 1, 600), ("gif_seconds", 1, 6)):
        if isinstance(o[k], bool) or not isinstance(o[k], (int, float)) or not math.isfinite(o[k]) or not lo <= o[k] <= hi:
            raise StoryboardError(f"storyboard.{k} must be {lo}–{hi} seconds")
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


def _framed_timeline(tl, t, timeout):
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
    return frame_tl


def _frame(tl, t, path, timeout):
    deadline = time.monotonic() + timeout
    framed = _framed_timeline(tl, t, timeout)
    render.still(framed, t, path, timeout_s=_remaining(deadline))


def _gif_windows(tl, o):
    """Nonoverlapping real-time excerpts, bounded by the GIF frame allowance."""
    fps = o["gif_fps"]
    available = max(1, math.ceil(tl["duration"] * fps - 1e-6))
    count = min(available, int(o["gif_seconds"] * fps))
    if available <= count:
        return [list(range(available))]
    # Sample hook, middle and payoff; each excerpt advances at normal speed.
    parts = min(3, count)
    sizes = [count // parts + (i < count % parts) for i in range(parts)]
    windows = []
    for i, size in enumerate(sizes):
        if size:
            start = round((available - size) * i / max(1, parts - 1))
            windows.append(list(range(start, start + size)))
    return windows


def _remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 0:
        raise StoryboardError("storyboard time budget exhausted; use a smaller plan")
    return left


def _gif(tl, path, o, deadline):
    """Compose a few low-fps excerpts, never encode a full reel or prepare audio."""
    frames, ranges = [], []
    fps = o["gif_fps"]
    width = min(o["gif_width"], tl["width"])
    height = max(1, round(tl["height"] * width / tl["width"]))
    windows = _gif_windows(tl, o)
    for window in windows:
        left = _remaining(deadline)
        t = min((window[0] + window[-1]) / (2 * fps), tl["duration"] - 1e-6)
        small = _framed_timeline(tl, t, left)
        small["fps"] = fps
        guard = render._Guard(_remaining(deadline), min(10, left))
        comp = render.Compositor(small, ".", guard)
        try:
            comp.from_t = window[0] / fps
            for f in window:
                _remaining(deadline)
                guard.check(stall=False)
                rgb = comp.frame(f)[:, :, ::-1]
                img = Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
                # Timestamp belongs to the preview, not the production edit.
                draw = ImageDraw.Draw(img)
                draw.rectangle((0, 0, 95, 16), fill="black")
                draw.text((4, 2), f"Preview {f / fps:.2f}s", fill="white")
                frames.append(img.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
        finally:
            comp.close()
        ranges.append({"s": window[0] / fps, "e": min(tl["duration"], (window[-1] + 1) / fps)})
    _remaining(deadline)
    # GIF delays use centiseconds. Distribute rounding so a 6fps excerpt lasts 1s.
    delays = [10 * (round((i + 1) * 100 / fps) - round(i * 100 / fps)) for i in range(len(frames))]
    frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:],
                   duration=delays, loop=0, disposal=2, optimize=False)
    if path.stat().st_size > o["max_gif_bytes"]:
        path.unlink()
        raise StoryboardError(f"storyboard GIF exceeds {o['max_gif_bytes']} bytes; reduce GIF width, duration or fps")
    _remaining(deadline)
    return {"file": str(path.resolve()), "frames": len(frames), "fps": fps, "width": width, "height": height,
            "duration": sum(delays) / 1000, "sampled_ranges": ranges, "bytes": path.stat().st_size}


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


def _page(reels, path, estimated, frames, gif_frames):
    esc = html.escape
    blocks = []
    for r in reels:
        gif = r["gif"]
        gif_src = esc(Path(gif["file"]).relative_to(path.parent).as_posix())
        ranges = ", ".join(f'{w["s"]:.2f}–{w["e"]:.2f}s' for w in gif["sampled_ranges"])
        shot_rows = "".join(f'<tr><td>{s["start"]:.2f}s</td><td>{s["duration"]:.2f}s</td>'
                            f'<td>{esc(s["source"])}</td><td>{s["source_in"]:.2f}–{s["source_out"]:.2f}s</td></tr>'
                            for s in r["shot_list"])
        cards = "".join(f'<figure><img loading="lazy" src="{esc(Path(f["file"]).relative_to(path.parent).as_posix())}" '
                        f'alt="{esc(r["id"])} at {f["t"]:.2f} seconds"><figcaption>{f["t"]:.2f}s</figcaption></figure>'
                        for f in r["frames"])
        blocks.append(f'<section><h2>{esc(r["id"])} · {esc(r["title"])}</h2>'
                      f'<p>{r["duration"]:.1f}s · {r["shots"]} planned shots · {len(r["frames"])} preview frames</p>'
                      f'<div class="frames">{cards}</div>'
                      f'<details class="motion"><summary>Play silent GIF · {gif["duration"]:.2f}s of selected moments</summary>'
                      f'<img class="gif" loading="lazy" src="{gif_src}" alt="{esc(r["id"])} silent motion preview">'
                      f'<p>Sampled reel times: {ranges}. <a href="{gif_src}" download>Download GIF</a></p></details>'
                      f'<details><summary>Read the planned speech and shot list</summary>'
                      f'<p>{esc(r["speech"])}</p><table><thead><tr><th>Reel time</th><th>Length</th>'
                      f'<th>Source</th><th>Source range</th></tr></thead><tbody>{shot_rows}</tbody></table></details></section>')
    fsutil.atomic_write_text(path, '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Review your storyboard</title>'
        '<style>body{margin:0 auto;padding:32px;max-width:1100px;background:#0b1020;color:#eef2ff;'
        'font:17px/1.6 system-ui}h1,h2{line-height:1.2}.frames{display:flex;gap:16px;overflow-x:auto}'
        'figure{margin:0;flex:0 0 240px}img{width:100%;border-radius:12px}figcaption{color:#a5b4fc}'
        'section{padding:28px 0;border-top:1px solid #334155}aside{padding:18px;background:#172033;'
        'border-radius:12px}summary{cursor:pointer}details{margin-top:18px}.gif{width:240px;max-width:100%;margin-top:16px}'
        'table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #334155}'
        'a{color:#a5b4fc}@media(prefers-reduced-motion:reduce){.motion img{display:none}.motion[open] img{display:block}}'
        '</style><h1>Review your storyboard</h1>'
        '<aside>Full rendering is paused. Review the stills and silent GIFs, then approve or describe changes.</aside>'
        f'<p>{frames} stills + {gif_frames} GIF frames composed instead of approximately {estimated:,} full-render frames. '
        'This is a frame-count comparison, not a time or cost estimate.</p>'
        f'<p>{esc(LIMITS)}</p>' + ''.join(blocks) + '</html>')


def build(plan_path, root, config=None, provider="cpu"):
    started = time.monotonic()
    plan, style, assets, transcripts, source_ids, shared, key = _inputs(plan_path, config, provider)
    o = shared["options"]
    deadline = started + o["timeout_s"]
    if len(plan["reels"]) * o["frames_per_reel"] > o["max_frames"]:
        raise StoryboardError(f"storyboard exceeds {o['max_frames']} stills; split the plan into smaller jobs")
    if len(plan["reels"]) * int(o["gif_fps"] * o["gif_seconds"]) > o["max_gif_frames"]:
        raise StoryboardError(f"storyboard exceeds {o['max_gif_frames']} GIF frames; reduce GIF fps/duration or split the job")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    board = root / f"board-{key[:12]}-{secrets.token_hex(3)}"
    board.mkdir()
    reels, files, estimated, hits = [], {}, 0, 0
    for reel in plan["reels"]:
        _remaining(deadline)
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
                left = _remaining(deadline)
                f = cache / f"frame-{i:02d}.jpg"
                _frame(tl, t, f, left)
                frames.append({"file": str(f.resolve()), "t": t})
            gif = _gif(tl, cache / "preview.gif", o, deadline)
            shot_list = [{"start": c["start"], "duration": c["dur"], "source_in": c["in"],
                          "source_out": c["in"] + c["dur"],
                          "source": next((k for k, a in assets.items() if str(a.path) == tl["media"][c["media"]]), c["media"])}
                         for c in tl.get("video") or []]
            r = {"id": reel["id"], "title": reel.get("title") or reel.get("title_text") or reel["id"],
                 "duration": tl["duration"], "fps": tl["fps"], "shots": len(tl.get("video") or []),
                 "speech": " ".join(tl["meta"]["words"]), "frames": frames, "gif": gif, "shot_list": shot_list}
            hashes = {str(Path(tl_path).resolve()): fsutil.sha256_file(tl_path)}
            hashes.update({f["file"]: fsutil.sha256_file(f["file"]) for f in frames})
            hashes[gif["file"]] = fsutil.sha256_file(gif["file"])
            fsutil.atomic_write_json(cache / "cache.json", {"files": hashes, "reel": r})
        # Board files are independent copies, so a later cache repair cannot alter
        # the frames a user is currently reviewing.
        r = copy.deepcopy(r)
        for i, f in enumerate(r["frames"]):
            dest = board / f"{reel['id']}-{i:02d}.jpg"
            shutil.copyfile(f["file"], dest)
            f["file"] = str(dest.resolve())
            files[f["file"]] = fsutil.sha256_file(dest)
        dest = board / f"{reel['id']}-preview.gif"
        shutil.copyfile(r["gif"]["file"], dest)
        r["gif"]["file"] = str(dest.resolve())
        files[r["gif"]["file"]] = fsutil.sha256_file(dest)
        reels.append(r)
        estimated += round(r["duration"] * r["fps"])
    total = sum(len(r["frames"]) for r in reels)
    gif_total = sum(r["gif"]["frames"] for r in reels)
    gifs = [{"reel_id": r["id"], **r["gif"]} for r in reels]
    _remaining(deadline)
    sheet, page, manifest = board / "contact-sheet.jpg", board / "index.html", board / "manifest.json"
    _sheet(reels, sheet)
    _page(reels, page, estimated, total, gif_total)
    fsutil.atomic_write_json(manifest, {"version": VERSION, "key": key, "reels": reels, "limitations": LIMITS,
                           "preview_frames": total, "gif_frames": gif_total, "estimated_render_frames": estimated,
                           "cache_hits": hits, "options": o})
    for p in (sheet, page, manifest):
        files[str(p.resolve())] = fsutil.sha256_file(p)
    _remaining(deadline)
    return {"key": key, "sha256": fsutil.sha256_file(manifest), "manifest": str(manifest.resolve()),
            "page": str(page.resolve()), "sheet": str(sheet.resolve()), "files": files,
            "frames": total, "gif_frames": gif_total, "gifs": gifs, "cache_hits": hits, "estimated_render_frames": estimated,
            "elapsed_s": round(time.monotonic() - started, 2), "limitations": LIMITS}
