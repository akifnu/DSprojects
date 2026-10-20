"""Sample reel → measured style spec ("reel DNA") + review sheet.

Runs the text, picture and audio analysers, merges them into SPEC.md's style.json,
and draws a review sheet that puts every measured element next to the real one
(real caption crop beside the same words re-rendered from the measured style), so
the spec can be verified by eye before a single reel is cut.

style.json carries analysis.fingerprint: the analysis code, the language, the ASR
fingerprint of the sample's transcript and the sample's content. A cached style that is
unreadable or whose fingerprint is no longer current is measured again (a style verified
in place by an earlier version, which has none, is kept); the orchestrator reuses a
verified style only for the measurement it was verified against. style.json and
review.jpg are written to unique temp names and renamed into place, one analysis per
sample at a time.
"""
import hashlib
import os
import re
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import asr, captions, ff, fsutil, store

SHEET_W = 1600
ANALYSIS_CODE = ("sample.py", "sample_text.py", "sample_video.py", "sample_audio.py", "grade.py")
LEGACY = "it predates analysis fingerprints"


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


@lru_cache(maxsize=1)
def code_hash():
    """The analysis code (analysers and merge), as one digest."""
    h = hashlib.sha256()
    for name in ANALYSIS_CODE:
        h.update(name.encode())
        try:
            h.update(Path(__file__).with_name(name).read_bytes())
        except OSError:
            h.update(b"-")
    return h.hexdigest()[:16]


def fingerprint(asset, lang="en", asr_fp=None):
    """What a measured style depends on; asr_fp is its transcript's fingerprint (None without audio)."""
    return fsutil.sha256_json({"code": code_hash(), "lang": lang, "asr": asr_fp,
                               "sample": asset.meta.get("sha256")})[:24]


def _stale(style, a, lang, provider):
    """Why a cached style must be measured again; None when it is current."""
    an = style.get("analysis")
    if not isinstance(an, dict) or not an.get("fingerprint"):
        return LEGACY
    if an.get("code") != code_hash():
        return "the analysis code changed"
    if an.get("lang") != lang:
        return f"it was measured for language {an.get('lang')!r}"
    if a.info.get("has_audio") and not asr.current(an.get("asr"), lang, provider):
        return "its transcript's ASR models or settings changed"
    if an["fingerprint"] != fingerprint(a, lang, an.get("asr")):
        return "its fingerprint does not match"
    return None


def _cached(a, out, lang, provider, quiet=False):
    """The cached style when readable and current (or verified in place by an earlier version)."""
    if not out.exists():
        return None
    st = fsutil.read_json(out)
    if not isinstance(st, dict) or not isinstance(st.get("sample"), dict):
        if not quiet:
            ff.log(f"sample {a.id}: style.json is unreadable; analysing again")
        return None
    why = _stale(st, a, lang, provider)
    if why == LEGACY and isinstance(st.get("review"), dict) and st["review"].get("verified") is True:
        if not quiet:
            ff.log(f"sample {a.id}: keeping style.json verified in place by an earlier version ({why})")
        return st
    if why:
        if not quiet:
            ff.log(f"sample {a.id}: cached style is stale ({why}); analysing again")
        return None
    return st


def analyze(ref, lang="en", force=False, provider=None):
    """Analyse a sample asset (id or path; ingested if needed) → style dict (cached while its
    fingerprint is current). provider: the ASR provider for its transcript (asr.transcribe)."""
    try:
        a = store.get(ref)
    except KeyError:
        a = store.ingest(ref, "sample")
    out, sheet = a.cache("style.json"), a.cache("review.jpg")
    style = None if force else _cached(a, out, lang, provider)
    if style is None:
        with fsutil.file_lock(out):
            style = None if force else _cached(a, out, lang, provider, quiet=True)
            if style is None:
                style = _measure(a, lang, provider)
                fsutil.atomic_write_json(out, style)
                review_sheet(a, style, sheet)
                ff.log(f"sample analysed in {style['analysis_seconds']}s → {out}")
                return style
    if not sheet.exists():
        try:
            review_sheet(a, style, sheet)
        except Exception as e:      # the style stands; its review aid is missing
            ff.log(f"sample {a.id}: review sheet could not be drawn ({ff.describe(e)})")
    return style


def _measure(a, lang, provider):
    from . import sample_audio, sample_text, sample_video
    info, path = a.info, str(a.path)
    started = time.monotonic()
    with ThreadPoolExecutor(1) as pool:
        tr_f = pool.submit(asr.transcribe, path, lang, str(a.cache(f"transcript.{lang}.json")), provider) \
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
    asr_fp = tr.get("fingerprint") if tr_f else None
    seconds = round(time.monotonic() - started, 1)
    return {"version": 2,
            "sample": {"id": a.id, "name": a.meta["name"], "duration": round(info["duration"], 3),
                       "width": info["width"], "height": info["height"], "fps": info["fps"]},
            "pacing": {**video.get("pacing", {}), **a_pacing},
            "shots": video["shots"], "motion": video.get("motion", {}), "transitions": video.get("transitions", {}),
            "captions": captions_sec, "titles": titles, "overlays": overlays,
            "grade": video.get("grade", {}), "audio": audio, "structure": structure,
            "transcript": tr, "review": {"verified": False, "notes": []},
            "analysis": {"fingerprint": fingerprint(a, lang, asr_fp), "code": code_hash(), "lang": lang,
                         "asr": asr_fp, "sample_sha256": a.meta.get("sha256"), "seconds": seconds},
            "analysis_seconds": seconds}


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
    f_small = _label_font(16)
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
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(str(out))), prefix=os.path.basename(str(out)) + ".",
                               suffix=".jpg")
    os.close(fd)
    try:
        sheet.save(tmp, "JPEG", quality=88)
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return out
