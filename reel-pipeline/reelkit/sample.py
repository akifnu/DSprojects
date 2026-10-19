"""Sample reel → measured style spec ("reel DNA") + review sheet.

Runs the text, picture and audio analysers, merges them into SPEC.md's style.json,
and draws a review sheet that puts every measured element next to the real one
(real caption crop beside the same words re-rendered from the measured style), so
the spec can be verified by eye before a single reel is cut.
"""
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import asr, captions, ff, store

SHEET_W = 1600


def _norm(w):
    return re.sub(r"[^\w]+", "", w.lower())


def caption_lead(events, words):
    """Median (caption start − start of the spoken word it begins with)."""
    leads = []
    for ev in events:
        first = _norm(ev["text"].split()[0]) if ev["text"].split() else ""
        if not first:
            continue
        near = [w for w in words if abs(w["s"] - ev["s"]) < 1.5 and _norm(w["w"]) == first]
        if near:
            w = min(near, key=lambda w: abs(w["s"] - ev["s"]))
            leads.append(ev["s"] - w["s"])
    return round(statistics.median(leads), 3) if leads else None


def _merge_audio(res):
    """Accept the audio analyser's result as {'audio': …, 'pacing': …, 'structure': …}
    or as the bare audio section with pacing/structure extras inside it."""
    res = dict(res or {})
    audio = res.pop("audio", None)
    pacing = res.pop("pacing", {})
    structure = res.pop("structure", {})
    if audio is None:
        audio = res
        for k in ("speech_wpm", "max_pause", "cut_on"):
            if k in audio:
                pacing[k] = audio.pop(k)
        if "first_word_at" in audio:
            structure["first_word_at"] = audio.pop("first_word_at")
    return audio, pacing, structure


def analyze(ref, lang="en", force=False):
    """Analyse a sample asset (id or path; ingested if needed) → style dict (cached)."""
    from . import sample_audio, sample_text, sample_video
    try:
        a = store.get(ref)
    except KeyError:
        a = store.ingest(ref, "sample")
    out = a.cache("style.json")
    if out.exists() and not force:
        return json.loads(out.read_text())
    info, path = a.info, str(a.path)
    started = time.monotonic()
    with ThreadPoolExecutor(1) as pool:
        tr_f = pool.submit(asr.transcribe, path, lang, str(a.cache(f"transcript.{lang}.json"))) \
            if info.get("has_audio") else None
        gray, _ = sample_video.decode_gray(path, info)
        text = sample_text.analyze_text(path, info, gray)
        ff.log("sample: text done")
        video = sample_video.analyze_video(path, info, gray, text.get("boxes", []))
        ff.log("sample: picture done")
        tr = tr_f.result() if tr_f else {"words": [], "sentences": [], "lang": lang}
    cuts = [s["s"] for s in video["shots"][1:]]
    caps = text.get("captions", {}).get("events", [])
    audio, a_pacing, a_struct = _merge_audio(
        sample_audio.analyze_audio(path, info, tr, cuts, [e["s"] for e in caps]) if info.get("has_audio") else {})
    ff.log("sample: audio done")
    captions_sec = text.get("captions", {"present": False})
    if captions_sec.get("present") and tr["words"]:
        captions_sec["lead"] = caption_lead(caps, tr["words"])
    titles = text.get("titles", {"present": False, "events": []})
    overlays = text.get("overlays", {"persistent": []})
    all_text = sorted(caps + titles.get("events", []), key=lambda e: e["s"])
    hook = None
    if all_text and all_text[0]["s"] < 3.0:
        e0 = all_text[0]
        hook = {"dur": round(e0["e"] - e0["s"], 3), "text": e0["text"],
                "type": "title" if e0 in titles.get("events", []) else "caption"}
    elif tr["sentences"]:
        s0 = tr["sentences"][0]
        hook = {"dur": round(s0["e"] - s0["s"], 3), "text": s0["text"], "type": "spoken"}
    late = [e for e in all_text if e["s"] > 0.7 * info["duration"]]
    structure = {"duration": round(info["duration"], 3), "hook": hook,
                 "cta": {"text": late[-1]["text"], "s": late[-1]["s"]} if late else None,
                 "loop": video.get("loop", False), **a_struct}
    style = {"version": 2,
             "sample": {"id": a.id, "name": a.meta["name"], "duration": round(info["duration"], 3),
                        "width": info["width"], "height": info["height"], "fps": info["fps"]},
             "pacing": {**video.get("pacing", {}), **a_pacing},
             "shots": video["shots"], "motion": video.get("motion", {}), "transitions": video.get("transitions", {}),
             "captions": captions_sec, "titles": titles, "overlays": overlays,
             "grade": video.get("grade", {}), "audio": audio, "structure": structure,
             "transcript": tr, "review": {"verified": False, "notes": []},
             "analysis_seconds": round(time.monotonic() - started, 1)}
    out.write_text(json.dumps(style, indent=1, ensure_ascii=False))
    review_sheet(a, style, a.cache("review.jpg"))
    ff.log(f"sample analysed in {style['analysis_seconds']}s → {out}")
    return style


# ------------------------------------------------------------------ review sheet

def _frame_at(path, info, t, w):
    h = int(round(info["height"] * w / info["width"] / 2)) * 2
    for f in ff.frames(path, w, h, start=max(0.0, t), duration=0.2):
        return f
    return np.zeros((h, w, 3), np.uint8)


def _label_font(size=18):
    try:
        return ImageFont.truetype(captions.FALLBACKS[0], size)
    except OSError:
        return ImageFont.load_default()


def review_sheet(asset, style, out):
    """One image: header numbers, shots strip, every caption (real | re-rendered), grade."""
    path, info = str(asset.path), asset.info
    f_small, f_big = _label_font(16), _label_font(22)
    rows = []
    # header
    p, au = style["pacing"], style.get("audio", {})
    cap = style.get("captions", {})
    cs = cap.get("style", {}) if cap.get("present") else {}
    lines = [f"{style['sample']['name']}  {style['sample']['width']}x{style['sample']['height']}@{style['sample']['fps']}  "
             f"{style['sample']['duration']}s   shots {p.get('shot_count')}  avg {p.get('avg_shot')}s  "
             f"cuts/min {p.get('cuts_per_min')}  wpm {p.get('speech_wpm')}  max pause {p.get('max_pause')}",
             f"captions: {cs.get('font')} {cs.get('weight')} {cs.get('size')}px fill {cs.get('fill')} "
             f"box {(cs.get('box') or {}).get('color')} stroke {(cs.get('stroke') or {}).get('color')} "
             f"anim {(cs.get('anim') or {}).get('in')}  case {cap.get('case')}  words/cap {cap.get('words_per_caption')}  "
             f"lead {cap.get('lead')}",
             f"audio: {au.get('loudness')}  music {au.get('music', {}).get('present')} bpm {au.get('music', {}).get('bpm')}  "
             f"sfx {au.get('sfx', {}).get('types')}   transitions {style.get('transitions', {}).get('counts')}"]
    head = Image.new("RGB", (SHEET_W, 30 * len(lines) + 20), (20, 20, 24))
    d = ImageDraw.Draw(head)
    for i, ln in enumerate(lines):
        d.text((14, 10 + 30 * i), ln[:190], font=f_small, fill=(235, 235, 235))
    rows.append(head)
    # shots
    shots = style["shots"][:18]
    if shots:
        tw = SHEET_W // max(6, len(shots))
        strip = Image.new("RGB", (SHEET_W, int(tw * info["height"] / info["width"]) + 44), (12, 12, 14))
        d = ImageDraw.Draw(strip)
        for i, s in enumerate(shots):
            fr = _frame_at(path, info, s["s"] + min(0.1, s["dur"] / 2), tw - 6)
            strip.paste(Image.fromarray(fr[..., ::-1]), (i * tw + 3, 3))
            tr = s.get("transition_in", "")
            tr = tr if isinstance(tr, str) else tr.get("type", "")
            mv = s.get("move", {}).get("type", "")
            tag = f"{s['s']:.2f} {tr[:6]} {'P' + str(s['punch']) if s.get('punch') else ''}"
            d.text((i * tw + 4, strip.height - 40), tag, font=f_small, fill=(255, 220, 120))
            d.text((i * tw + 4, strip.height - 20), mv, font=f_small, fill=(170, 200, 255))
        rows.append(strip)
    # captions: real crop | re-rendered from measured style
    evs = (cap.get("events") or [])[:10] if cap.get("present") else []
    for ev in evs:
        fr = _frame_at(path, info, (ev["s"] + ev["e"]) / 2, info["width"])
        x0, y0, x1, y1 = [int(v) for v in ev["box"]]
        pad = int(0.6 * (y1 - y0) / max(1, ev["text"].count("\n") + 1))
        crop = fr[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        real = Image.fromarray(crop[..., ::-1]) if crop.size else Image.new("RGB", (10, 10))
        bg = cv2.GaussianBlur(fr, (0, 0), 3)[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        mine_bg = Image.fromarray(bg[..., ::-1]).convert("RGBA") if bg.size else Image.new("RGBA", (10, 10))
        try:
            patch, _ = captions.render(ev["text"], {**cs, "max_width": 10 ** 6}, info["width"])
            canvas = Image.new("RGBA", (max(mine_bg.width, patch.width), max(mine_bg.height, patch.height)), (0, 0, 0, 255))
            canvas.alpha_composite(mine_bg.resize(canvas.size))
            canvas.alpha_composite(patch, ((canvas.width - patch.width) // 2, (canvas.height - patch.height) // 2))
            mine = canvas.convert("RGB")
        except Exception as e:  # a style the renderer cannot draw is itself a finding
            mine = Image.new("RGB", real.size, (60, 0, 0))
            ImageDraw.Draw(mine).text((4, 4), str(e)[:60], font=f_small, fill="white")
        h = max(real.height, mine.height) + 26
        row = Image.new("RGB", (SHEET_W, h), (30, 30, 34))
        row.paste(real, (10, 22))
        row.paste(mine, (SHEET_W // 2 + 10, 22))
        ImageDraw.Draw(row).text((10, 2), f"{ev['s']:.2f}-{ev['e']:.2f}  real", font=f_small, fill=(200, 200, 200))
        ImageDraw.Draw(row).text((SHEET_W // 2 + 10, 2), "re-rendered from style.json", font=f_small,
                                 fill=(140, 230, 140))
        rows.append(row)
    sheet = Image.new("RGB", (SHEET_W, sum(r.height for r in rows) + 4 * len(rows)), (0, 0, 0))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 4
    sheet.save(out, quality=88)
    return out
