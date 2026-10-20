"""Quality gate for a rendered reel.

check(reel.mp4, timeline, style) runs every check below and returns
{"verdict": "pass|warn|fail", "issues": [...], "metrics": {...}}; an issue is
{"check", "severity": "warn|fail", "msg", "t": [s, e] (optional)}.

  format      codec / size / fps / pixel format, duration ±1 frame of the timeline,
              audio and video streams the same length
  loudness    integrated loudness ±0.5 LU of the target, true peak ≤ −1 dBTP
  picture     black frames and frozen runs inside moving footage (outside what the
              timeline intends), uncovered canvas edges (pixels and timeline geometry),
              clips that run past the end of their source
  audio       clicks / discontinuities at every edit point, pauses longer than the
              style allows, slow first word
  captions    spoken words (re-transcribed) vs caption words: WER and timing offset;
              caption visible, inside platform safe zones, not over a face, contrast
  style       shot length, cuts/min, caption size / position / words per caption
              against the sample's style spec

sheet() draws the review sheet: verdict + metrics, before/after frames at every cut,
a frame per caption, and the waveform with edit marks.
"""
import difflib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from . import ff
from .paths import FFPROBE, MODELS

# UI-free zones per platform at 1080×1920 (px of margin), scaled to the canvas.
SAFE = {"reels": {"top": 220, "bottom": 420, "left": 60, "right": 150},
        "tiktok": {"top": 160, "bottom": 480, "left": 60, "right": 160},
        "shorts": {"top": 200, "bottom": 380, "left": 60, "right": 140}}
PRIMARY_PLATFORM = "reels"


# ------------------------------------------------------------------ helpers

def _load_timeline(timeline, base_dir=None):
    if timeline is None:
        return None, base_dir or os.getcwd()
    if isinstance(timeline, (str, Path)):
        return json.loads(Path(timeline).read_text()), str(Path(timeline).resolve().parent)
    return timeline, base_dir or os.getcwd()


def _issue(issues, check, severity, msg, t=None):
    it = {"check": check, "severity": severity, "msg": msg}
    if t is not None:
        it["t"] = [round(float(t[0]), 3), round(float(t[1]), 3)]
    issues.append(it)


def _streams(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _norm(w):
    return re.sub(r"[^\w']+", "", w.lower()).strip("'")


def _tokens(words):
    return [t for t in (_norm(w) for w in words) if t]


def _runs(mask, min_len=1):
    """(start, end_exclusive) runs of True."""
    out, k, n = [], 0, len(mask)
    while k < n:
        if mask[k]:
            j = k
            while j < n and mask[j]:
                j += 1
            if j - k >= min_len:
                out.append((k, j))
            k = j
        else:
            k += 1
    return out


def _base_clips(tl):
    return sorted([c for c in tl.get("video", []) if c.get("track", 1) == 1], key=lambda c: c["start"])


def _media_paths(tl, base_dir):
    from .render import _resolve
    return {k: _resolve(v, base_dir) for k, v in tl.get("media", {}).items()}


def _intended_dark(tl, fps, n):
    """Frames where the timeline itself shows background (gaps) or dips to black."""
    out = np.zeros(n, bool)
    if tl is None:
        return out
    base = _base_clips(tl)
    bg = tl.get("background", "#000000").lstrip("#").lower()
    dark_bg = bg in ("000", "000000") or sum(int(bg[i:i + 2], 16) for i in (0, 2, 4)) < 60 if len(bg) == 6 else True
    for k in range(n):
        t = (k + 0.5) / fps
        covered = any(c["start"] <= t < c["start"] + c["dur"] for c in base)
        if not covered and dark_bg:
            out[k] = True
    for i, c in enumerate(base):
        tr = c.get("transition_in") or {}
        if isinstance(tr, dict) and tr.get("type") in ("dip_black", "dissolve", "zoom", "flash") and tr.get("dur"):
            half = tr["dur"] / 2 + 1 / fps
            if tr.get("type") == "dip_black" or i == 0:
                a, b = int((c["start"] - half) * fps), int(math.ceil((c["start"] + half) * fps))
                out[max(0, a):min(n, b)] = True
    for c in tl.get("video", []):          # fully transparent moments of the base clip
        if c.get("track", 1) == 1 and isinstance(c.get("opacity"), (list, int, float)):
            pass
    return out


def caption_boxes(tl):
    """Pixel box (x0, y0, x1, y1) of every caption / text event at its settled size,
    with all highlight states unioned: [(event, box, mid_t)]."""
    from . import captions as C
    if not tl:
        return []
    W, H = tl["width"], tl["height"]
    styles = tl.get("styles", {})
    out = []
    for ev in tl.get("captions", []) + tl.get("text", []):
        style = ev["style"] if isinstance(ev.get("style"), dict) else styles.get(ev.get("style", "caption"), {})
        te = C.TextEvent(ev, style, (W, H))
        a = te.style.get("anim") or {}
        settle = ev["s"] + a.get("in_dur", 0.15 if a.get("in") else 0) + 0.02
        times = [min(ev["e"] - 1e-3, max(settle, ev["s"]))]
        for w in ev.get("words") or []:
            if ev["s"] <= w["s"] < ev["e"]:
                times.append(min(ev["e"] - 1e-3, max(settle, (w["s"] + min(w["e"], ev["e"])) / 2)))
        box = None
        for t in times:
            p = te.patch(t)
            if p is None:
                continue
            _, alpha, x0, y0 = p
            ys, xs = np.where(alpha > 0.1)
            if not len(ys):
                continue
            b = [x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1]
            box = b if box is None else [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
        if box is not None:
            mid = min(ev["e"] - 1e-3, max(settle, (ev["s"] + ev["e"]) / 2))
            out.append((ev, [int(v) for v in box], mid, te.style))
    return out


def _safe_rect(name, W, H):
    m = SAFE[name]
    return (m["left"] * W / 1080, m["top"] * H / 1920, W - m["right"] * W / 1080, H - m["bottom"] * H / 1920)


def _lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def contrast_ratio(rgb1, rgb2):
    l1 = float(np.dot(_lin(np.asarray(rgb1, float)), [0.2126, 0.7152, 0.0722]))
    l2 = float(np.dot(_lin(np.asarray(rgb2, float)), [0.2126, 0.7152, 0.0722]))
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _hex_rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


_face_det = {}


def _faces(img):
    model = MODELS / "yunet" / "yunet.onnx"
    if not model.exists():
        return []
    h, w = img.shape[:2]
    scale = 320.0 / w if w > 320 else 1.0
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale != 1.0 else img
    key = small.shape[:2]
    if key not in _face_det:
        _face_det[key] = cv2.FaceDetectorYN.create(str(model), "", (small.shape[1], small.shape[0]), 0.75, 0.3, 50)
    _, faces = _face_det[key].detect(small)
    if faces is None:
        return []
    return [[f[0] / scale, f[1] / scale, (f[0] + f[2]) / scale, (f[1] + f[3]) / scale] for f in faces]


# ------------------------------------------------------------------ scanning

class Scan:
    """One sequential decode of the reel: per-frame black/freeze/border signals,
    and full-size copies of the frames asked for (cuts, captions)."""

    def __init__(self, path, info, want=()):
        self.fps = info["fps"] or 30
        W, H = info["width"], info["height"]
        self.size = (W, H)
        want = set(int(k) for k in want)
        self.frames = {}
        means, stds, diffs, borders, thumbs = [], [], [], [], []
        prev = None
        for k, f in enumerate(ff.frames(path, W, H)):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            sm = cv2.resize(g, (64, max(2, int(64 * H / W))), interpolation=cv2.INTER_AREA).astype(np.float32)
            means.append(float(sm.mean()))
            stds.append(float(sm.std()))
            diffs.append(0.0 if prev is None else float(np.abs(sm - prev).mean()))
            prev = sm
            borders.append(self._border(f))
            if k in want:
                self.frames[k] = f
            if k % max(1, int(self.fps // 2)) == 0:
                thumbs.append((k, cv2.resize(f, (90, max(2, int(90 * H / W))), interpolation=cv2.INTER_AREA)))
        self.n = len(means)
        self.mean, self.std, self.diff = np.array(means), np.array(stds), np.array(diffs)
        self.border = borders
        self.thumbs = thumbs

    @staticmethod
    def _border(f):
        """Width (px) of a uniform near-black band on each edge that ends in content."""
        H, W = f.shape[:2]
        mx = f.max(axis=2)
        out = {}
        for edge, line in (("left", lambda i: mx[:, i]), ("right", lambda i: mx[:, W - 1 - i]),
                           ("top", lambda i: mx[i, :]), ("bottom", lambda i: mx[H - 1 - i, :])):
            limit = (W if edge in ("left", "right") else H) // 4
            width = 0
            while width < limit and (line(width) < 20).mean() > 0.97:
                width += 1
            if width >= 2 and width < limit and line(min(limit, width + 2)).mean() > 25:
                out[edge] = width
        return out


def _audio_edits(tl):
    if not tl:
        return []
    pts = set()
    for c in (tl.get("audio") or {}).get("clips", []):
        if c.get("role", "voice") == "voice":
            pts.add(round(c["start"], 4))
            pts.add(round(c["start"] + c["dur"], 4))
    for c in _base_clips(tl)[1:]:
        pts.add(round(c["start"], 4))
    dur = tl["duration"]
    return sorted(p for p in pts if 0.02 < p < dur - 0.02)


def find_clicks(a, sr, edits, win=0.008):
    """Edit points with a broadband burst: the second difference (a steep high-pass)
    peaks far above its level in the surrounding ±150 ms. A waveform cut without a
    fade puts a step there, which the encoder turns into exactly such a burst."""
    out = []
    if len(a) < sr // 10:
        return out
    d = np.abs(np.diff(a, 2))
    for t in edits:
        k = int(t * sr)
        lo, hi = max(0, k - int(win * sr)), min(len(d), k + int(win * sr))
        if hi <= lo:
            continue
        ref = np.concatenate([d[max(0, k - int(0.15 * sr)):max(0, k - int(0.015 * sr))],
                              d[min(len(d), k + int(0.015 * sr)):min(len(d), k + int(0.15 * sr))]])
        if not len(ref):
            continue
        peak = float(d[lo:hi].max())
        base = float(np.percentile(ref, 99)) + 1e-4
        if peak > 0.004 and peak > 6.0 * base:
            out.append({"t": round(t, 3), "jump": round(peak, 4), "ratio": round(peak / base, 1)})
    return out


def _align(cap_words, spoken):
    """WER and timing offsets between caption words [(token, s)] and spoken [(token, s)]."""
    ct, st = [w[0] for w in cap_words], [w[0] for w in spoken]
    sm = difflib.SequenceMatcher(None, ct, st, autojunk=False)
    edits = 0
    offsets = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if cap_words[i][1] is not None:
                    offsets.append(cap_words[i][1] - spoken[j][1])
        else:
            edits += max(i2 - i1, j2 - j1)
    wer = edits / max(1, len(st))
    return wer, offsets


# ------------------------------------------------------------------ main check

def check(reel_mp4, timeline=None, style=None, transcript_words=None, asr=True, lang=None, platforms=None,
          base_dir=None, sheet_path=None):
    """Run every check on a rendered reel. See the module docstring."""
    tl, base_dir = _load_timeline(timeline, base_dir)
    if isinstance(style, (str, Path)):
        style = json.loads(Path(style).read_text())
    issues, metrics = [], {}
    reel_mp4 = str(reel_mp4)
    info = ff.probe(reel_mp4)
    st = _streams(reel_mp4)
    fps = info.get("fps") or (tl or {}).get("fps") or 30
    metrics.update(width=info.get("width"), height=info.get("height"), fps=fps, duration=round(info["duration"], 3))

    # ---- format
    v = next((s for s in st["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in st["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        _issue(issues, "format", "fail", "no video stream")
        return _finish(issues, metrics)
    if v.get("codec_name") != "h264":
        _issue(issues, "format", "warn", f"video codec {v.get('codec_name')} (expected h264)")
    if v.get("pix_fmt") != "yuv420p":
        _issue(issues, "format", "fail", f"pixel format {v.get('pix_fmt')} (platforms need yuv420p)")
    if tl:
        if (info["width"], info["height"]) != (tl["width"], tl["height"]):
            _issue(issues, "format", "fail", f"size {info['width']}x{info['height']} ≠ timeline {tl['width']}x{tl['height']}")
        if abs(fps - tl["fps"]) > 0.01:
            _issue(issues, "format", "fail", f"fps {fps} ≠ timeline {tl['fps']}")
        vd = float(v.get("duration") or info["duration"])
        if abs(vd - tl["duration"]) > 1.0 / tl["fps"] + 1e-3:
            _issue(issues, "format", "fail", f"video lasts {vd:.3f}s, timeline {tl['duration']:.3f}s")
    expect_audio = bool(tl and (tl.get("audio") or {}).get("clips")) or (tl is None and a is not None)
    if expect_audio and a is None:
        _issue(issues, "format", "fail", "no audio stream")
    if a is not None:
        if a.get("codec_name") != "aac":
            _issue(issues, "format", "warn", f"audio codec {a.get('codec_name')} (expected aac)")
        ad, vd = float(a.get("duration") or 0), float(v.get("duration") or 0)
        metrics["av_gap"] = round(ad - vd, 3)
        if ad and vd and abs(ad - vd) > max(1.0 / fps, 0.05):
            _issue(issues, "format", "fail", f"audio {ad:.3f}s vs video {vd:.3f}s")

    # ---- loudness
    if a is not None:
        target = ((tl or {}).get("audio") or {}).get("loudness", -14.0)
        tp_max = ((tl or {}).get("audio") or {}).get("true_peak", -1.0)
        loud = ff.loudness(reel_mp4)
        metrics["loudness"] = loud
        if "I" in loud and abs(loud["I"] - target) > 0.5:
            _issue(issues, "loudness", "fail", f"integrated {loud['I']:.1f} LUFS, target {target} ±0.5")
        if "TP" in loud and loud["TP"] > tp_max:
            sev = "fail" if loud["TP"] > tp_max + 0.5 else "warn"
            _issue(issues, "loudness", sev, f"true peak {loud['TP']:.1f} dBTP > {tp_max}")

    # ---- picture: one decode pass
    caps = caption_boxes(tl)
    base = _base_clips(tl) if tl else []
    cut_frames = [int(round(c["start"] * fps)) for c in base[1:]]
    want = set()
    for k in cut_frames:
        want.update((k - 1, k))
    for _, _, mid, _ in caps:
        want.add(int(mid * fps))
    scan = Scan(reel_mp4, info, want)
    n = scan.n
    intended = _intended_dark(tl, fps, n)
    black = (scan.mean < 14) & (scan.std < 5)
    for s, e in _runs(black & ~intended, 2):
        _issue(issues, "black", "fail", f"{e - s} black frames", (s / fps, e / fps))
    metrics["black_frames"] = int((black & ~intended).sum())
    still = scan.diff < 0.12
    still[0] = False
    for s, e in _runs(still, max(3, int(0.4 * fps))):
        ctx = np.concatenate([scan.diff[max(1, s - int(fps)):s], scan.diff[e:min(n, e + int(fps))]])
        if len(ctx) and np.median(ctx) > 0.8:
            _issue(issues, "frozen", "fail", f"picture frozen for {(e - s) / fps:.2f}s inside moving footage",
                   (s / fps, e / fps))
    bad_border = np.array([bool(b) for b in scan.border])
    for s, e in _runs(bad_border & ~intended, 3):
        edges = sorted({k for b in scan.border[s:e] for k in b})
        _issue(issues, "border", "fail", f"uncovered canvas edge ({', '.join(edges)})", (s / fps, e / fps))
    if tl:
        _geometry(tl, base_dir, issues)

    # ---- audio edits / pauses
    if a is not None:
        samples = ff.audio(reel_mp4, 48000)
        edits = _audio_edits(tl) if tl else []
        clicks = find_clicks(samples, 48000, edits)
        metrics["edits_checked"] = len(edits)
        for c in clicks:
            _issue(issues, "click", "fail", f"click at edit {c['t']:.3f}s (jump {c['ratio']}× local slope)",
                   (c["t"], c["t"]))

    # ---- captions vs speech
    spoken = transcript_words
    if spoken is None and asr and a is not None and (caps or tl is None):
        from . import asr as asr_mod
        lang = lang or ((style or {}).get("audio") or {}).get("speech", {}).get("lang") or "en"
        try:
            spoken = asr_mod.transcribe(reel_mp4, lang)["words"]
        except Exception as e:      # missing model etc.: report, don't crash the batch
            _issue(issues, "captions", "warn", f"could not re-transcribe: {e}")
    if spoken:
        max_pause = ((style or {}).get("pacing") or {}).get("max_pause")
        limit = (max_pause if max_pause is not None else 0.8) + 0.15
        gaps = [(w1["s"] - w0["e"], w0["e"], w1["s"]) for w0, w1 in zip(spoken, spoken[1:])]
        long_gaps = [g for g in gaps if g[0] > limit]
        metrics["max_pause"] = round(max((g[0] for g in gaps), default=0.0), 3)
        for g, s, e in long_gaps:
            _issue(issues, "pause", "warn", f"{g:.2f}s pause (style allows {limit - 0.15:.2f}s)", (s, e))
        if spoken[0]["s"] > 0.6:
            _issue(issues, "pause", "warn", f"first word at {spoken[0]['s']:.2f}s (hook should start immediately)")
    if spoken and caps:
        cap_words = []
        for ev, _, _, _ in caps:
            ws = ev.get("words") or []
            if ws:
                cap_words += [(_norm(w["w"]), w["s"]) for w in ws if _norm(w["w"])]
            else:
                toks = _tokens(ev["text"].split())
                cap_words += [(t, ev["s"] if i == 0 else None) for i, t in enumerate(toks)]
        sp = [(_norm(w["w"]), w["s"]) for w in spoken if _norm(w["w"])]
        cap_words = [c for c in cap_words if c[0]]
        roles = [ev.get("role") for ev, _, _, _ in caps]
        if cap_words and sp and not all(r == "title" for r in roles):
            wer, offs = _align(cap_words, sp)
            metrics["caption_wer"] = round(wer, 3)
            if wer > 0.35:
                _issue(issues, "captions", "fail", f"captions don't match the speech (WER {wer:.0%})")
            elif wer > 0.15:
                _issue(issues, "captions", "warn", f"captions differ from the speech (WER {wer:.0%})")
            if offs:
                med = float(np.median(offs))
                p90 = float(np.percentile(np.abs(np.array(offs) - med), 90))
                metrics["caption_offset"] = round(med, 3)
                lead = ((style or {}).get("captions") or {}).get("lead", 0.0) or 0.0
                if abs(med - lead) > 0.15:
                    _issue(issues, "captions", "warn", f"captions {'late' if med > lead else 'early'} by "
                                                        f"{abs(med - lead):.2f}s vs speech")
                if p90 > 0.35:
                    _issue(issues, "captions", "warn", f"caption timing uneven (p90 spread {p90:.2f}s)")

    # ---- caption placement / visibility / faces / contrast
    W, H = info["width"], info["height"]
    plats = platforms or list(SAFE)
    for ev, box, mid, sty in caps:
        tspan = (ev["s"], ev["e"])
        label = ev["text"].replace("\n", " ")[:40]
        if H > W:
            out_of = []
            for name in plats:
                x0, y0, x1, y1 = _safe_rect(name, W, H)
                if box[0] < x0 - 1 or box[1] < y0 - 1 or box[2] > x1 + 1 or box[3] > y1 + 1:
                    out_of.append(name)
            if out_of:
                sev = "fail" if PRIMARY_PLATFORM in out_of else "warn"
                _issue(issues, "safe_zone", sev, f"'{label}' outside the {', '.join(out_of)} safe zone", tspan)
        frame = scan.frames.get(int(mid * fps))
        if frame is None:
            continue
        x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
        region = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        if not len(region):
            continue
        fill = np.array(_hex_rgb(sty.get("fill", "#ffffff"))[::-1], np.float32)
        near = np.linalg.norm(region - fill, axis=1) < 60
        if near.mean() < 0.005:
            _issue(issues, "captions", "fail", f"'{label}' not visible on screen", tspan)
            continue
        box_style = sty.get("box")
        if box_style and box_style.get("opacity", 1) >= 0.85:
            behind = _hex_rgb(box_style.get("color", "#000000"))
        else:
            other = region[~near & (np.linalg.norm(region - fill, axis=1) > 90)]
            behind = np.median(other, axis=0)[::-1] if len(other) else _hex_rgb("#000000")
        cr = contrast_ratio(fill[::-1], behind)
        if cr < 3.0:
            _issue(issues, "contrast", "warn", f"'{label}' contrast {cr:.1f}:1 (< 3:1)", tspan)
        for fb in _faces(frame):
            ix = max(0, min(fb[2], box[2]) - max(fb[0], box[0]))
            iy = max(0, min(fb[3], box[3]) - max(fb[1], box[1]))
            area = max(1.0, (fb[2] - fb[0]) * (fb[3] - fb[1]))
            if ix * iy / area > 0.25:
                _issue(issues, "face", "warn", f"'{label}' covers a face", tspan)
                break

    # ---- style conformance
    if style and tl:
        _style_checks(tl, style, issues, metrics)

    metrics["cuts"] = len(cut_frames)
    report = _finish(issues, metrics)
    if sheet_path:
        sheet(reel_mp4, report, tl, sheet_path, _scan=scan, _caps=caps, _cuts=cut_frames)
    return report


def _finish(issues, metrics):
    verdict = "fail" if any(i["severity"] == "fail" for i in issues) else \
        "warn" if issues else "pass"
    return {"verdict": verdict, "issues": issues, "metrics": metrics}


def _geometry(tl, base_dir, issues):
    """Does each base clip's picture cover the canvas (fit cover / stretch), and does
    its source last long enough? Checked from the timeline alone, every 0.1 s."""
    from .render import _matrix, kf
    W, H, fps = tl["width"], tl["height"], tl["fps"]
    media = _media_paths(tl, base_dir)
    probes = {}
    base = _base_clips(tl)
    for i, c in enumerate(base):
        path = media.get(c.get("media"))
        if not path or not os.path.exists(path) or path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        info = probes.setdefault(path, ff.probe(path))
        if not info.get("has_video"):
            continue
        speed = c.get("speed", 1.0)
        nxt = base[i + 1] if i + 1 < len(base) else None
        tout = ((nxt.get("transition_in") or {}).get("dur", 0) / 2) if nxt and isinstance(nxt.get("transition_in"), dict) else 0
        need = c.get("in", 0) + (c["dur"] + tout) * speed
        if need > info["duration"] + 1.5 / fps:
            _issue(issues, "source", "fail", f"clip {i} needs {need:.2f}s of a {info['duration']:.2f}s source "
                                             f"(last frame would be held)", (c["start"], c["start"] + c["dur"]))
        if c.get("fit", "cover") == "contain":
            continue
        sw, sh = info["width"], info["height"]
        if c.get("fit") == "stretch":
            fx, fy = W / sw, H / sh
        else:
            fx = fy = max(W / sw, H / sh)
        r = SimpleNamespace(fx=fx, fy=fy, work=1.0, ww=sw, wh=sh)
        steps = max(2, int(c["dur"] * 10))
        bad = None
        for j in range(steps + 1):
            t_rel = c["dur"] * j / steps
            M, _ = _matrix(r, c, t_rel, W, H)
            corners = M @ np.array([[-0.5, sw - 0.5, -0.5, sw - 0.5], [-0.5, -0.5, sh - 0.5, sh - 0.5], [1, 1, 1, 1]])
            if corners[0].min() > 0.5 or corners[0].max() < W - 1.5 or corners[1].min() > 0.5 or corners[1].max() < H - 1.5:
                bad = c["start"] + t_rel
                break
        if bad is not None:
            _issue(issues, "border", "fail", f"clip {i}'s picture leaves the canvas uncovered at {bad:.2f}s "
                                             f"(scale {kf(c.get('scale'), bad - c['start'], 1.0):.2f}, "
                                             f"x {kf(c.get('x'), bad - c['start'], 0):.0f}, y {kf(c.get('y'), bad - c['start'], 0):.0f})",
                   (bad, bad))


def _style_checks(tl, style, issues, metrics):
    base = _base_clips(tl)
    pacing = style.get("pacing") or {}
    if len(base) > 1:
        durs = [c["dur"] for c in base]
        med = float(np.median(durs))
        metrics["median_shot"] = round(med, 3)
        want = pacing.get("median_shot") or pacing.get("avg_shot")
        if want and not (0.5 <= med / want <= 2.0):
            _issue(issues, "style", "warn", f"median shot {med:.2f}s vs sample {want:.2f}s")
        cpm = (len(base) - 1) / max(1e-3, tl["duration"]) * 60
        metrics["cuts_per_min"] = round(cpm, 1)
        if pacing.get("cuts_per_min") and not (0.5 <= cpm / pacing["cuts_per_min"] <= 2.0):
            _issue(issues, "style", "warn", f"{cpm:.0f} cuts/min vs sample {pacing['cuts_per_min']:.0f}")
    capspec = style.get("captions") or {}
    sstyle = capspec.get("style") or {}
    scale = tl["height"] / ((style.get("sample") or {}).get("height") or tl["height"])
    tstyle = (tl.get("styles") or {}).get("caption") or {}
    if sstyle.get("size") and tstyle.get("size"):
        want = sstyle["size"] * scale
        if abs(tstyle["size"] - want) / want > 0.15:
            _issue(issues, "style", "warn", f"caption size {tstyle['size']:.0f}px vs sample {want:.0f}px")
    if "y" in sstyle and "y" in tstyle and abs(tstyle["y"] - sstyle["y"] * scale) > 0.05 * tl["height"]:
        _issue(issues, "style", "warn", f"caption position y={tstyle['y']:.0f} vs sample {sstyle['y'] * scale:.0f}")
    wpc = (capspec.get("words_per_caption") or {}).get("median")
    caps = [c for c in tl.get("captions", []) if c.get("role") != "title"]
    if wpc and caps:
        med = float(np.median([len(c["text"].split()) for c in caps]))
        metrics["words_per_caption"] = med
        if abs(med - wpc) > 1.5:
            _issue(issues, "style", "warn", f"{med:.1f} words per caption vs sample {wpc}")


# ------------------------------------------------------------------ review sheet

def _font(size):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


VERDICT_COLOR = {"pass": (40, 170, 80), "warn": (230, 150, 20), "fail": (210, 50, 50)}


def sheet(reel_mp4, report, timeline=None, out=None, base_dir=None, _scan=None, _caps=None, _cuts=None):
    """Review sheet image: verdict + metrics + issues, before/after frames at every cut,
    one frame per caption, waveform with edit (blue) and issue (red) marks."""
    from PIL import Image, ImageDraw
    tl, base_dir = _load_timeline(timeline, base_dir)
    info = ff.probe(reel_mp4)
    fps = info["fps"] or 30
    caps = _caps if _caps is not None else caption_boxes(tl)
    cuts = _cuts if _cuts is not None else [int(round(c["start"] * fps)) for c in _base_clips(tl)[1:]] if tl else []
    if _scan is None:
        want = set()
        for k in cuts:
            want.update((k - 1, k))
        for _, _, mid, _ in caps:
            want.add(int(mid * fps))
        _scan = Scan(reel_mp4, info, want)
    W, H = info["width"], info["height"]
    tw = 120
    th = int(tw * H / W)
    cols = 12
    SW = cols * (tw + 8) + 16
    f_big, f_small = _font(20), _font(12)

    def thumb(k):
        f = _scan.frames.get(k)
        if f is None:
            return Image.new("RGB", (tw, th), (40, 40, 40))
        return Image.fromarray(cv2.cvtColor(cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

    cut_rows = math.ceil(len(cuts) * 2 / cols) if cuts else 0
    cap_rows = math.ceil(len(caps) / cols) if caps else 0
    issues = report.get("issues", [])
    header_h = 70 + 16 * min(12, len(issues))
    wave_h = 110
    Ht = header_h + (cut_rows * (th + 26) + 30 if cuts else 0) + (cap_rows * (th + 26) + 30 if caps else 0) + wave_h + 40
    img = Image.new("RGB", (SW, Ht), (18, 18, 22))
    d = ImageDraw.Draw(img)
    v = report.get("verdict", "?")
    d.rectangle([0, 0, SW, 40], fill=VERDICT_COLOR.get(v, (90, 90, 90)))
    d.text((12, 9), f"{Path(reel_mp4).name}  —  {v.upper()}", font=f_big, fill=(255, 255, 255))
    m = report.get("metrics", {})
    loud = m.get("loudness", {})
    line = (f"{m.get('width')}x{m.get('height')} @{m.get('fps')}  {m.get('duration')}s  cuts {m.get('cuts', 0)}  "
            f"I {loud.get('I', '-')} LUFS  TP {loud.get('TP', '-')}  WER {m.get('caption_wer', '-')}  "
            f"offset {m.get('caption_offset', '-')}s  max pause {m.get('max_pause', '-')}s")
    d.text((12, 46), line, font=f_small, fill=(220, 220, 220))
    y = 66
    for it in issues[:12]:
        col = VERDICT_COLOR[it["severity"]]
        t = f" @{it['t'][0]:.2f}s" if it.get("t") else ""
        d.text((12, y), f"[{it['severity']}] {it['check']}: {it['msg']}{t}", font=f_small, fill=col)
        y += 16
    y = header_h
    if cuts:
        d.text((12, y + 6), "cuts (before | after)", font=f_small, fill=(160, 160, 170))
        y += 26
        x = 8
        for i, k in enumerate(cuts):
            for j, kk in enumerate((k - 1, k)):
                pos = i * 2 + j
                cx, cy = 8 + (pos % cols) * (tw + 8), y + (pos // cols) * (th + 26)
                img.paste(thumb(kk), (cx, cy))
                d.text((cx, cy + th + 3), f"{kk / fps:.2f}s" + (" ▸" if j else ""), font=f_small, fill=(200, 200, 200))
        y += cut_rows * (th + 26) + 4
    if caps:
        d.text((12, y + 6), "captions", font=f_small, fill=(160, 160, 170))
        y += 26
        for i, (ev, box, mid, _) in enumerate(caps):
            cx, cy = 8 + (i % cols) * (tw + 8), y + (i // cols) * (th + 26)
            img.paste(thumb(int(mid * fps)), (cx, cy))
            d.text((cx, cy + th + 3), ev["text"].replace("\n", " ")[:18], font=f_small, fill=(200, 200, 200))
        y += cap_rows * (th + 26) + 4
    # waveform
    y += 10
    d.text((12, y), "audio (blue = edits, red = issues)", font=f_small, fill=(160, 160, 170))
    y0 = y + 18
    try:
        a = ff.audio(reel_mp4, 8000)
    except Exception:
        a = np.zeros(1, np.float32)
    dur = max(1e-3, info["duration"])
    cols_px = SW - 24
    if len(a) > cols_px:
        env = np.abs(a[: len(a) // cols_px * cols_px]).reshape(cols_px, -1).max(axis=1)
        mid_y = y0 + (wave_h - 30) // 2
        for i, e in enumerate(env):
            h = int(e * (wave_h - 30) / 2)
            d.line([(12 + i, mid_y - h), (12 + i, mid_y + h)], fill=(120, 200, 240))
    for t in _audio_edits(tl) if tl else []:
        x = 12 + int(t / dur * cols_px)
        d.line([(x, y0), (x, y0 + wave_h - 30)], fill=(60, 110, 255))
    for it in issues:
        if it.get("t"):
            x = 12 + int(it["t"][0] / dur * cols_px)
            d.line([(x, y0), (x, y0 + wave_h - 30)], fill=(230, 50, 50), width=2)
    out = out or str(Path(reel_mp4).with_suffix(".sheet.jpg"))
    img.save(out, quality=88)
    return out


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="QC a rendered reel")
    ap.add_argument("reel")
    ap.add_argument("--timeline")
    ap.add_argument("--style")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--sheet")
    a = ap.parse_args()
    rep = check(a.reel, a.timeline, a.style, asr=not a.no_asr, sheet_path=a.sheet)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    _main()
