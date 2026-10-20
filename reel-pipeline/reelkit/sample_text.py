"""On-screen text of a sample reel: every editor-added overlay (caption, hook title,
watermark) as a frame-exact timed event, and the typography that draws it — font,
size, colours, box, outline, shadow, active-word highlight, in/out animation —
measured from the pixels and expressed as a captions.py style.

Pipeline
  1. OCR at 5 samples/s (RapidOCR PP-OCRv3 ONNX): detection on a downscaled frame,
     boxes whose glyphs are too small to be an overlay dropped by measured ink
     height, recognition only for the rest (cached while unchanged). Time-chunked
     worker processes with one ONNX thread each; unchanged frames are skipped.
  2. Detections are tracked into events; start/end snapped to the exact frame on
     the full-frame-rate grey stack.
  3. Overlay vs in-picture text (UI text of a screen recording, signs, …): decoration
     (box / outline / shadow), glyph size, centring, whether it stays put while the
     picture moves, persistence, recurrence as a caption track.
  4. Typography per event at full resolution; animation from the frames around its
     start and end; font by rendering Google Fonts candidates over the glyph mask.
  5. Aggregation into the SPEC.md "captions" / "titles" / "overlays" shapes, plus
     "boxes" (all overlay regions over time) for masking the picture analysis.
"""
import difflib
import math
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import captions as C
from . import ff

OCR_FPS = 5
OCR_WIDTH = 720            # OCR runs on frames at most this wide
DET_SIDE = 576             # detector input: longest side (px)
MIN_INK = 0.011            # smallest glyph height that can be an overlay (fraction of frame height)
BIG_INK = 0.022            # glyphs at least this tall are always read; smaller ones only on a box / at the edges
WORKERS = max(1, min(4, os.cpu_count() or 4))

# ------------------------------------------------------------------ OCR engine

_eng = None


def _engine(threads=1):
    """(detector, recogniser) with a fixed ONNX thread count per process."""
    global _eng
    if _eng is None:
        import rapidocr_onnxruntime.utils as U
        if not getattr(U, "_reelkit_threads", None):
            orig = U.SessionOptions

            def so():
                s = orig()
                s.intra_op_num_threads = U._reelkit_threads
                s.inter_op_num_threads = 1
                return s
            U.SessionOptions = so
        U._reelkit_threads = threads
        from rapidocr_onnxruntime.ch_ppocr_v3_det import TextDetector
        from rapidocr_onnxruntime.ch_ppocr_v3_rec import TextRecognizer
        cfg = U.read_yaml(str(U.root_dir / "config.yaml"))
        det = dict(cfg["Det"], model_path=str(U.root_dir / cfg["Det"]["model_path"]),
                   limit_side_len=DET_SIDE, limit_type="max")
        rec = dict(cfg["Rec"], model_path=str(U.root_dir / cfg["Rec"]["model_path"]))
        _eng = (TextDetector(det), TextRecognizer(rec))
    return _eng


def _ink_extent(gray):
    """Glyph rows/cols inside a text crop: Otsu split, text = the minority class."""
    if gray.size < 16:
        return None
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = th > 0
    if fg.mean() > 0.5:
        fg = ~fg
    # rules / gridlines / box edges run across the whole crop: they are not glyphs
    fg[fg.mean(axis=1) > 0.7] = False
    fg[:, fg.mean(axis=0) > 0.7] = False
    prof = fg.mean(axis=1)
    rows = np.where(prof > 0.03)[0]
    cols = np.where(fg.mean(axis=0) > 0.02)[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    # the glyph band is the longest run of inked rows (ignores stray specks above/below)
    runs = _segments(prof > 0.03, max(2, len(prof) // 12))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return a, b, cols[0], cols[-1] + 1


def _boxed(f, box, ih):
    """Text sitting on a uniform panel that differs from what is around it."""
    H, W = f.shape[:2]
    near = _ring(f, box, 0, max(2, int(0.25 * ih)))
    far = _ring(f, box, int(1.2 * ih), int(2.0 * ih))
    if near is None or far is None:
        return False
    nl, fl = _lab(near), _lab(far)
    if float(np.std(nl, axis=0).mean()) > 7:
        return False
    return float(np.linalg.norm(np.median(nl, axis=0) - np.median(fl, axis=0))) > 14


def _ocr_chunk(args):
    path, t0, n, W, H = args
    det, rec = _engine(1)
    out, last, last_small, cache = [], None, None, {}
    for i, f in enumerate(ff.frames(path, W, H, fps=OCR_FPS, start=t0 if t0 > 0 else None,
                                    duration=n / OCR_FPS + 0.5)):
        if i >= n:
            break
        t = round(t0 + i / OCR_FPS, 3)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, (W // 8, H // 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        if last_small is not None and np.abs(small - last_small).mean() < 0.6:
            out.append((t, last))
            continue
        boxes, _ = det(f)
        cands = []
        for q in (boxes if boxes is not None else []):
            q = np.asarray(q)
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            ext = _ink_extent(g[y0:y1, x0:x1])
            if ext is None:
                continue
            ih = ext[1] - ext[0]
            if ih < MIN_INK * H:
                continue
            if ih < BIG_INK * H and not _boxed(f, (x0, y0, x1, y1), ih):
                edge = (y1 < 0.14 * H or y0 > 0.84 * H)
                if not (edge and i % OCR_FPS == 0):       # small edge text (handles): once a second
                    continue
            cands.append([x0, y0, x1, y1, ih])
        dets, crops, keys = [], [], []
        for x0, y0, x1, y1, ih in cands:
            crop = f[y0:y1, x0:x1]
            key = (round(x0 / 4), round(y0 / 4), round(x1 / 4), round(y1 / 4),
                   cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (16, 4)).tobytes())
            if key in cache:
                text, conf = cache[key]
                if text:
                    dets.append({"text": text, "conf": conf, "box": [x0, y0, x1, y1], "ink_h": int(ih)})
                continue
            crops.append(crop)
            keys.append((key, [x0, y0, x1, y1], ih))
        if crops:
            res, _ = rec(crops)
            for (key, box, ih), (text, conf) in zip(keys, res):
                text, conf = text.strip(), float(conf)
                cache[key] = (text, conf) if conf >= 0.6 and text else ("", 0.0)
                if conf >= 0.6 and text:
                    dets.append({"text": text, "conf": conf, "box": box, "ink_h": int(ih)})
        out.append((t, dets))
        last, last_small = dets, small
    return out


def ocr_samples(path, info):
    """[(t, [{text, conf, box (px at OCR size), ink_h}])] at OCR_FPS, and the OCR frame size."""
    W = min(OCR_WIDTH, info["width"])
    W -= W % 2
    H = int(round(info["height"] * W / info["width"] / 2)) * 2
    total = int(math.floor(info["duration"] * OCR_FPS + 1e-6))
    per = int(math.ceil(total / WORKERS))
    jobs = [(str(path), k / OCR_FPS, min(per, total - k), W, H) for k in range(0, total, per)]
    with ProcessPoolExecutor(len(jobs)) as pool:
        parts = list(pool.map(_ocr_chunk, jobs))
    return [s for p in parts for s in p], (W, H)


# ------------------------------------------------------------------ tracking

def _norm(s):
    return re.sub(r"[^\w]+", "", s.lower())


def _blocks(dets):
    """Merge one frame's OCR lines into text blocks (multi-line captions), and
    detections split along one line back together."""
    dets = sorted(dets, key=lambda d: (d["box"][1], d["box"][0]))
    lines = []
    for d in dets:                                     # same line, horizontally adjacent
        for l in lines:
            ly0, ly1 = l["box"][1], l["box"][3]
            y0, y1 = d["box"][1], d["box"][3]
            ov = min(ly1, y1) - max(ly0, y0)
            gap = max(d["box"][0] - l["box"][2], l["box"][0] - d["box"][2])
            if ov > 0.6 * min(ly1 - ly0, y1 - y0) and gap < 1.2 * max(l["ink_h"], d["ink_h"]):
                first, second = (l, d) if l["box"][0] <= d["box"][0] else (d, l)
                l["text"] = first["text"] + " " + second["text"]
                l["box"] = [min(l["box"][0], d["box"][0]), min(ly0, y0), max(l["box"][2], d["box"][2]), max(ly1, y1)]
                l["conf"] = min(l["conf"], d["conf"])
                l["ink_h"] = max(l["ink_h"], d["ink_h"])
                break
        else:
            lines.append(dict(d, box=list(d["box"])))
    blocks = []
    for d in sorted(lines, key=lambda d: d["box"][1]):
        x0, y0, x1, y1 = d["box"]
        for b in blocks:
            bx0, by0, bx1, by1 = b["box"]
            last = b["lines"][-1]
            gap = y0 - by1
            cx, bcx = (x0 + x1) / 2, (bx0 + bx1) / 2
            similar = 0.7 < d["ink_h"] / max(1, last["ink_h"]) < 1.45
            if -0.4 * d["ink_h"] < gap < 1.2 * d["ink_h"] and similar and \
                    abs(cx - bcx) < 0.12 * (bx1 - bx0 + x1 - x0) + 2 * d["ink_h"]:
                b["lines"].append(d)
                b["box"] = [min(bx0, x0), by0, max(bx1, x1), y1]
                b["text"] += "\n" + d["text"]
                b["ink_h"] = max(b["ink_h"], d["ink_h"])
                break
        else:
            blocks.append({"lines": [d], "box": list(d["box"]), "text": d["text"], "ink_h": d["ink_h"],
                           "conf": d["conf"]})
    return blocks


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def events(samples):
    """Track OCR blocks across samples into timed text events (times on the sample grid)."""
    dt = 1.0 / OCR_FPS
    active, done = [], []
    for t, dets in samples:
        blocks = _blocks(dets)
        used = set()
        for ev in active:
            best = None
            for i, b in enumerate(blocks):
                if i in used:
                    continue
                a, bn = _norm(ev["text"]), _norm(b["text"])
                sim = difflib.SequenceMatcher(None, a, bn).ratio()
                grows = len(a) >= 2 and bn.startswith(a[:max(2, len(a) - 1)]) and len(bn) > len(a)
                shrinks = len(bn) >= 2 and a.startswith(bn) and len(bn) < len(a) and ev.get("growth")
                if (sim > 0.75 or grows or shrinks) and _iou(ev["box"], b["box"]) > 0.1:
                    if best is None or sim > best[1]:
                        best = (i, sim, grows)
            if best is None:
                ev["miss"] += 1
                continue
            i, _, grows = best
            used.add(i)
            b = blocks[i]
            if grows:
                ev.setdefault("growth", []).append((t, b["text"]))
            if len(_norm(b["text"])) >= len(_norm(ev["text"])):
                ev["text"], ev["box"], ev["lines"], ev["ink_h"] = b["text"], b["box"], [l["box"] for l in b["lines"]], \
                    b["ink_h"]
            ev["last"], ev["miss"] = t, 0
            ev["seen"].append((t, list(b["box"])))
        still = []
        for ev in active:
            (done if ev["miss"] > 1 else still).append(ev)
        active = still
        for i, b in enumerate(blocks):
            if i not in used:
                active.append({"text": b["text"], "box": b["box"], "lines": [l["box"] for l in b["lines"]],
                               "ink_h": b["ink_h"], "first": t, "last": t, "miss": 0, "seen": [(t, list(b["box"]))],
                               "conf": b["conf"]})
    done += active
    out = []
    for ev in done:
        if len(ev["seen"]) < 2 and ev["conf"] < 0.9:
            continue                                    # one low-confidence glimpse: OCR noise
        out.append({"text": ev["text"], "s": ev["first"], "e": round(ev["last"] + dt, 3), "box": ev["box"],
                    "lines": ev["lines"], "ink_h": ev["ink_h"], "seen": ev["seen"], "growth": ev.get("growth", [])})
    return sorted(out, key=lambda e: (e["s"], e["box"][1]))


# ------------------------------------------------------------------ frame-exact timing

def refine_times(evs, gray, fps, size, duration):
    """Snap start/end to the frame where the event's glyph pixels change.
    Start = first frame after the last sample without the event whose glyph region
    has moved 25 % of the way from that 'before' state to the 'present' state; end
    likewise towards the 'after' state."""
    W, H = size
    n, gh, gw = gray.shape
    sx, sy = gw / W, gh / H
    dt = 1.0 / OCR_FPS
    for ev in evs:
        x0, y0, x1, y1 = ev["box"]
        r = (slice(max(0, int(y0 * sy) - 1), min(gh, int(math.ceil(y1 * sy)) + 1)),
             slice(max(0, int(x0 * sx) - 1), min(gw, int(math.ceil(x1 * sx)) + 1)))

        def reg(k):
            return gray[max(0, min(n - 1, k))][r].astype(np.float32)

        def onset(k_from, k_ref, frac):
            a, ref = reg(k_from), reg(k_ref)
            change = np.abs(ref - a)
            sel = change >= np.percentile(change, 70)
            total = change[sel].mean()
            if total < 2:
                return None
            for k in range(k_from + 1, k_ref + 1):
                if np.abs(reg(k) - a)[sel].mean() > frac * total:
                    return k
            return None
        k_before = int(round((ev["s"] - dt) * fps))
        k_first = int(round(ev["s"] * fps))
        if k_before >= 0:
            k = onset(k_before, k_first, 0.25)
            if k is not None:
                ev["s"] = round(k / fps, 3)
        else:
            ev["s"] = 0.0
        k_last = int(round((ev["e"] - dt) * fps))
        k_after = int(round(ev["e"] * fps))
        if k_after < n:
            k = onset(k_last, k_after, 0.5)
            if k is not None:
                ev["e"] = round(k / fps, 3)
        else:
            ev["e"] = round(duration, 3)
        ev["e"] = min(ev["e"], round(duration, 3))
    return evs


# ------------------------------------------------------------------ colour helpers

def _lab(bgr):
    return cv2.cvtColor(np.asarray(bgr, np.float32).reshape(-1, 1, 3) / 255.0, cv2.COLOR_BGR2LAB).reshape(-1, 3)


def _de(c1, c2):
    return float(np.linalg.norm(_lab(c1)[0] - _lab(c2)[0]))


def _de_img(img, color):
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return np.linalg.norm(lab - _lab(color)[0], axis=-1)


def _hex(bgr):
    b, g, r = [int(round(min(255, max(0, v)))) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def _ring(frame, box, r0, r1):
    """Pixels between r0 and r1 px outside a box."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    X0, Y0, X1, Y1 = max(0, x0 - r1), max(0, y0 - r1), min(W, x1 + r1), min(H, y1 + r1)
    if X1 <= X0 or Y1 <= Y0:
        return None
    m = np.ones((Y1 - Y0, X1 - X0), bool)
    m[max(0, y0 - r0) - Y0:min(H, y1 + r0) - Y0, max(0, x0 - r0) - X0:min(W, x1 + r0) - X0] = False
    px = frame[Y0:Y1, X0:X1][m]
    return px if len(px) > 20 else None


# ------------------------------------------------------------------ typography of one frame

def _segments(cols, min_gap):
    """Runs of True in a boolean column profile, merging gaps shorter than min_gap."""
    runs, start, gap = [], None, 0
    for i, on in enumerate(cols):
        if on:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                runs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, len(cols) - gap))
    return runs


def measure(frame, ev):
    """Colours, box, outline, shadow, glyph masks and geometry of a text block in one
    frame (frame at the sample's full resolution, ev boxes already in its pixels).
    Glyphs = thin strokes found by a morphological top-hat (bright-on-dark or
    dark-on-bright, whichever is stronger); what hugs them is read from rings at
    increasing distance: outline right next to the glyphs, box or picture further out."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in ev["box"]]
    lh = float(ev["ink_h"])
    pad = int(max(6, 0.6 * lh))
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop = frame[Y0:Y1, X0:X1]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    k = max(5, int(0.45 * lh)) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = np.zeros(g.shape, bool)
    inner[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = True
    best = None
    for op in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
        th = cv2.morphologyEx(g, op, ker)
        v = th[inner]
        if v.size == 0 or v.max() < 25:
            continue
        t, _ = cv2.threshold(v.astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (th > max(t, 25)) & inner
        if mask.sum() < 30:
            continue
        # the true glyph mask leaves a uniform surround (box / picture) a little way out;
        # the inverse mask (letter gaps) would put glyphs in that ring
        dd = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        ringm = (dd > 0.22 * lh) & (dd <= 0.4 * lh) & inner
        sd = float(np.std(_lab(crop[ringm]), axis=0).mean()) if ringm.sum() > 12 else 99.0
        if best is None or sd < best[0] - 1.0:
            best = (sd, mask)
    if best is None or best[1].sum() < 30:
        return None
    glyph = cv2.morphologyEx(best[1].astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    if glyph.sum() < 30:
        return None
    core = cv2.erode(glyph.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    fill = np.median(crop[core if core.sum() > 15 else glyph], axis=0)
    # rings outside the glyphs, by distance
    dist = cv2.distanceTransform((~glyph).astype(np.uint8), cv2.DIST_L2, 3)
    def ring(a, b):
        m = (dist > a) & (dist <= b) & inner
        return crop[m] if m.sum() > 12 else None
    r1 = ring(0, max(1.5, 0.06 * lh))
    r2 = ring(0.22 * lh, 0.4 * lh)
    if r1 is None or r2 is None:
        return None
    c1, c2 = np.median(r1, axis=0), np.median(r2, axis=0)
    sd2 = float(np.std(_lab(r2), axis=0).mean())
    stroke = None
    ink = glyph.copy()
    if _de(c1, fill) > 25 and _de(c1, c2) > 20:
        # outline width: distance over which the ring colour stays the outline colour
        w = 0
        for d in np.arange(0.5, 0.3 * lh, 0.5):
            r = ring(d, d + 0.5)
            if r is None or _de(np.median(r, axis=0), c1) > 18:
                break
            w = d + 0.5
        if w >= 1.0:
            stroke = {"color": _hex(c1), "width": round(float(w), 1)}
            ink = glyph | (dist <= w)
            ink &= inner | (dist <= w)
    near = c2
    ys, xs = np.where(ink)
    ink_box = [X0 + xs.min(), Y0 + ys.min(), X0 + xs.max() + 1, Y0 + ys.max() + 1]
    box = None
    far = _ring(frame, ink_box, int(1.2 * lh), int(2.0 * lh))
    if sd2 < 7 and far is not None:
        far_c = np.median(far, axis=0)
        far_sd = float(np.std(_lab(far), axis=0).mean())
        if _de(far_c, near) > 14 or far_sd > 3 * max(1.0, sd2):
            box = _box_extent(frame, ev, ink_box, near, lh)
            if box is not None and (box["bounds"][2] - box["bounds"][0]) > 0.98 * W:
                box = None                                 # a full-width band is the picture, not a box
    shadow = None
    if box is None and stroke is None:
        shadow = _shadow(frame, ink_box, near, glyph, (X0, Y0), lh)
    return {"fill": fill, "near": near, "near_sd": sd2, "stroke": stroke, "box": box, "shadow": shadow,
            "ink": ink, "fill_mask": glyph, "origin": (X0, Y0), "ink_box": [int(v) for v in ink_box]}


def _box_extent(frame, ev, ink_box, color, lh):
    """Box bounds per line, padding around the glyphs, corner radius, line/block mode."""
    H, W = frame.shape[:2]

    def run(y, x, dy, dx):
        n = 0
        while 0 <= y < H and 0 <= x < W and _de(frame[y, x], color) < 16:
            y, x, n = y + dy, x + dx, n + 1
        return n
    lines = ev["lines"]
    rows = []
    for lb in lines:
        cy = int((lb[1] + lb[3]) / 2)
        # start just outside the glyphs of this line, walk outwards
        lx0, lx1 = int(lb[0]), int(lb[2])
        left = run(cy, max(0, min(lx0, ink_box[0]) - 1), 0, -1)
        right = run(cy, min(W - 1, max(lx1, ink_box[2])), 0, 1)
        rows.append((min(lx0, ink_box[0]) - left, max(lx1, ink_box[2]) + right))
    cx = int((ink_box[0] + ink_box[2]) / 2)
    top = run(max(0, ink_box[1] - 1), cx, -1, 0)
    bottom = run(min(H - 1, ink_box[3]), cx, 1, 0)
    bx0, bx1 = min(r[0] for r in rows), max(r[1] for r in rows)
    by0, by1 = ink_box[1] - top, ink_box[3] + bottom
    # corner radius: inset along the diagonal before the box colour starts
    rad = 0.0
    for i in range(int(min(by1 - by0, bx1 - bx0) / 2)):
        y, x = by0 + i, bx0 + i
        if 0 <= y < H and 0 <= x < W and _de(frame[y, x], color) < 16:
            rad = i / (1 - 1 / math.sqrt(2))
            break
    widths = [r[1] - r[0] for r in rows]
    mode = "line" if len(rows) > 1 and (max(widths) - min(widths)) > 0.04 * max(widths) else "block"
    return {"color": _hex(color), "bounds": [int(bx0), int(by0), int(bx1), int(by1)], "rows": rows,
            "radius": round(rad, 1), "mode": mode}


def _shadow(frame, ink_box, near, ink, origin, lh):
    """Darker halo offset from the glyphs (soft drop shadow)."""
    X0, Y0 = origin
    H, W = frame.shape[:2]
    m = ink.astype(np.uint8)
    k = max(3, int(0.12 * lh))
    halo = cv2.dilate(m, np.ones((k, k), np.uint8)) & (1 - m)
    if halo.sum() < 30:
        return None
    crop = frame[Y0:Y0 + ink.shape[0], X0:X0 + ink.shape[1]]
    hL = _lab(np.median(crop[halo > 0], axis=0))[0][0]
    nL = _lab(near)[0][0]
    if nL - hL < 6:
        return None
    # offset: centre of the darkened halo relative to the glyphs
    dark = (cv2.cvtColor(crop.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0] < nL - 6) & (halo > 0)
    ys, xs = np.where(dark)
    gy, gx = np.where(ink)
    dy = float(ys.mean() - gy.mean()) if len(ys) else 0.0
    dx = float(xs.mean() - gx.mean()) if len(xs) else 0.0
    return {"color": "#000000", "opacity": round(float(min(1.0, (nL - hL) / max(1.0, nL) * 1.6)), 2),
            "blur": round(0.06 * lh, 1), "dx": round(dx, 1), "dy": round(dy, 1)}


def word_fills(frame, m, ev):
    """Per word (ink split at gaps): median glyph colour, glyph height, and the colour
    right around it (a per-word highlight box)."""
    X0, Y0 = m["origin"]
    out = []
    for lb in ev["lines"]:
        ly0, ly1 = int(lb[1]) - Y0, int(math.ceil(lb[3])) - Y0
        sub = m["ink"][max(0, ly0):max(0, ly1)]
        if not sub.size:
            continue
        cols = sub.any(axis=0)
        h = max(1, int(sub.any(axis=1).sum()))
        for a, b in _segments(cols, max(3, int(0.22 * h))):
            region = m["ink"][max(0, ly0):max(0, ly1), a:b]
            px = frame[Y0 + max(0, ly0):Y0 + max(0, ly1), X0 + a:X0 + b][region]
            if len(px) < 6:
                continue
            core = m["fill_mask"][max(0, ly0):max(0, ly1), a:b]
            cp = frame[Y0 + max(0, ly0):Y0 + max(0, ly1), X0 + a:X0 + b][core]
            rows = np.where(region.any(axis=1))[0]
            pad = max(2, int(0.25 * h))
            ring = frame[max(0, Y0 + ly0 - pad):Y0 + ly1 + pad, max(0, X0 + a - pad):X0 + b + pad]
            out.append({"x": (X0 + a, X0 + b), "color": np.median(cp if len(cp) > 6 else px, axis=0),
                        "h": int(rows[-1] - rows[0] + 1) if len(rows) else 0,
                        "around": np.median(ring.reshape(-1, 3), axis=0)})
    return out


# ------------------------------------------------------------------ font matching

CANDIDATES = ["Montserrat", "Poppins", "Inter", "Roboto", "Open Sans", "Lato", "Arimo", "Bebas Neue", "Anton",
              "Oswald", "Archivo Black", "Archivo", "Rubik", "Nunito", "Raleway", "Work Sans", "DM Sans", "Barlow",
              "Barlow Condensed", "League Spartan", "Kanit", "Bangers", "Luckiest Guy", "Titan One", "Lilita One",
              "Fredoka", "Noto Sans", "Manrope", "Outfit", "Sora", "Plus Jakarta Sans", "Roboto Condensed", "Teko",
              "Righteous", "Bungee", "Playfair Display", "Merriweather", "Lora", "Permanent Marker", "Caveat",
              "Source Sans 3", "Ubuntu", "Mulish", "Heebo", "IBM Plex Sans", "Josefin Sans", "Quicksand",
              "Exo 2", "Titillium Web", "Libre Franklin"]
WEIGHTS = [400, 500, 600, 700, 800, 900]


def _squeeze(mask):
    """Drop empty columns (word/letter gaps) so OCR spacing errors do not matter."""
    cols = mask.any(axis=0)
    rows = np.where(mask.any(axis=1))[0]
    if not cols.any() or not len(rows):
        return None
    return mask[rows[0]:rows[-1] + 1][:, cols]


_ref_cache = {}


def _render_ref(text, family, weight):
    key = (text, family, weight)
    if key not in _ref_cache:
        path = C.font_file(family, weight)
        font = C._font(family, weight, 100)
        w = int(font.getlength(text)) + 40
        img = Image.new("L", (w, 200), 0)
        ImageDraw.Draw(img).text((20, 100), text, font=font, fill=255, anchor="lm")
        a = np.array(img) > 127
        ys, xs = np.where(a)
        if not len(ys):
            _ref_cache[key] = None
        else:
            bb = font.getbbox(text, anchor="lm")
            _ref_cache[key] = {"mask": _squeeze(a), "ink_top": ys.min() - 100, "ink_bot": ys.max() + 1 - 100,
                               "ink_left": xs.min() - 20, "advance": font.getlength(text), "path": path,
                               "bbox": bb}
    return _ref_cache[key]


def _font_weights(family):
    """Distinct font files for a family (weights that fall back to the same file collapse)."""
    seen, out = set(), []
    for w in WEIGHTS:
        try:
            p = C.font_file(family, w)
        except SystemExit:
            continue
        if p not in seen and "liberation" not in p.lower() and "dejavu" not in p.lower():
            seen.add(p)
            out.append(w)
    return out


def match_font(glyph_masks, texts, families=None):
    """Rank (family, weight) by how well rendered text overlays the measured glyph masks.
    glyph_masks / texts: one per text line (same order)."""
    pairs = []
    for m, t in zip(glyph_masks, texts):
        sq = _squeeze(m)
        t = t.strip()
        if sq is not None and sq.shape[0] >= 8 and len(re.sub(r"\s", "", t)) >= 3:
            pairs.append((sq, t))
    if not pairs:
        return []

    def score(fam, wgt):
        tot, n = 0.0, 0
        for sq, t in pairs:
            ref = _render_ref(t, fam, wgt)
            if ref is None or ref["mask"] is None:
                return None
            r = ref["mask"]
            mh, mw = sq.shape
            rh, rw = r.shape
            ratio = (rw * mh / rh) / mw
            fit = cv2.resize(r.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_AREA) > 0
            iou = (fit & sq).sum() / max(1, (fit | sq).sum())
            tot += iou - 1.5 * abs(math.log(ratio))
            n += 1
        return tot / n
    fams = families or CANDIDATES
    first = []
    for fam in fams:
        ws = _font_weights(fam)
        if not ws:
            continue
        w = min(ws, key=lambda x: abs(x - 700))
        s = score(fam, w)
        if s is not None:
            first.append((s, fam))
    first.sort(reverse=True)
    results = []
    for _, fam in first[:8]:
        for w in _font_weights(fam):
            s = score(fam, w)
            if s is not None:
                results.append({"font": fam, "weight": w, "score": round(s, 4)})
    return sorted(results, key=lambda r: -r["score"])


def font_size(glyph_mask, text, family, weight):
    """Font size (px) at which `text` in this font has the measured glyph height, and the
    line centre y relative to the mask's top (for placing the anchor)."""
    ref = _render_ref(text, family, weight)
    sq = _squeeze(glyph_mask)
    if ref is None or sq is None:
        return None, None
    ih = ref["ink_bot"] - ref["ink_top"]
    size = 100.0 * sq.shape[0] / max(1, ih)
    centre_from_top = -ref["ink_top"] * size / 100.0      # anchor 'lm' middle sits this far below the ink top
    return size, centre_from_top


# ------------------------------------------------------------------ overlay decision & roles

def _features(ev, m, size, duration):
    W, H = size
    x0, y0, x1, y1 = ev["box"]
    seen = ev["seen"][1:] if len(ev["seen"]) > 2 else ev["seen"]     # skip the entrance
    cxs = [(b[0] + b[2]) / 2 for _, b in seen]
    cys = [(b[1] + b[3]) / 2 for _, b in seen]
    hs = [b[3] - b[1] for _, b in seen]
    moving = (max(cys) - min(cys)) > 0.03 * H or (max(cxs) - min(cxs)) > 0.04 * W or \
        (max(hs) / max(1, min(hs)) > 1.3 and len(seen) > 2)
    return {"ink_rel": ev["ink_h"] / H, "decorated": bool(m and (m["box"] or m["stroke"] or m["shadow"])),
            "centred": abs((x0 + x1) / 2 - W / 2) < 0.06 * W, "moving": moving,
            "dur": ev["e"] - ev["s"], "persistent": (ev["e"] - ev["s"]) > 0.6 * duration}


def _wordy(text):
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 2 and letters / max(1, len(text.replace(" ", "").replace("\n", ""))) > 0.6


def _is_candidate(f, text):
    """Could be an editor overlay: not moving with the picture, big enough, real words."""
    if f["moving"]:
        return False
    if f["persistent"] and ("@" in text or ".com" in text.lower()) and f["ink_rel"] >= 0.009:
        return True
    return f["ink_rel"] >= 0.02 and _wordy(text)


# ------------------------------------------------------------------ animation

def _anim_in(path, ev, info, ref):
    """Classify the entrance from the frames at its start against a settled frame."""
    fps = info["fps"] or 30
    W, H = info["width"], info["height"]
    frames = list(ff.frames(path, W, H, fps=fps, start=max(0.0, ev["s"]), duration=0.7))
    return _anim_from_frames(frames, ref, fps)


def _glyph_state(frame, ref):
    """Width/height/centre of fill-coloured glyph pixels near the settled text, and the
    fraction of the settled fill's colour present at the settled glyph positions."""
    bx0, by0, bx1, by1 = ref["region"]
    crop = frame[by0:by1, bx0:bx1]
    d = _de_img(crop, ref["fill"])
    mask = d < ref["tol"]
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    out = {"area": int(mask.sum())}
    if mask.sum() > 0.08 * ref["area"]:
        ys, xs = np.where(mask)
        out.update(w=xs.max() - xs.min() + 1, h=ys.max() - ys.min() + 1, cy=ys.mean(), cx=xs.mean())
    # alpha along the settled glyph pixels: projection between surround and fill colour
    lab = cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)[ref["pix"]]
    v = ref["fill_lab"] - ref["near_lab"]
    out["alpha"] = float(np.clip(((lab - ref["near_lab"]) @ v) / max(1e-6, v @ v), -0.5, 1.5).mean())
    return out


def _anim_from_frames(frames, ref, fps):
    if not frames:
        return {"in": "none"}
    st = [_glyph_state(f, ref) for f in frames[:int(0.6 * fps) + 1]]
    W0, H0 = ref["w"], ref["h"]
    ws = [s.get("w", 0) / W0 for s in st]
    hs = [s.get("h", 0) / H0 for s in st]
    al = [s["alpha"] for s in st]
    first = st[0]

    def settle(vals, target=1.0, tol=0.025):
        for i in range(len(vals)):
            if all(abs(v - target) <= tol for v in vals[i:i + 3]):
                return i
        return len(vals)
    if "w" in first and hs[0] < 0.93 and ws[0] < 0.95:
        k = settle(ws)
        peak = max(ws[:k + 1] or [1])
        return {"in": "pop", "in_dur": round(max(1, k) / fps, 3), "from": round(ws[0], 2),
                "overshoot": round(max(0.0, peak - 1), 3)}
    if "w" in first and ws[0] > 1.06:
        k = settle(ws)
        return {"in": "zoom", "in_dur": round(max(1, k) / fps, 3), "from": round(ws[0], 2)}
    if "w" in first and ws[0] < 0.85 and hs[0] > 0.9:
        k = settle(ws, tol=0.03)
        return {"in": "word_reveal", "in_dur": round(max(1, k) / fps, 3)}
    if "cy" in first and abs(first["cy"] - ref["cy"]) > 0.12 * H0:
        k = next((i for i, s in enumerate(st) if "cy" in s and abs(s["cy"] - ref["cy"]) < 0.03 * H0), len(st))
        return {"in": "slide_up" if first["cy"] > ref["cy"] else "slide_down", "in_dur": round(max(1, k) / fps, 3),
                "slide_px": round(abs(first["cy"] - ref["cy"]), 1)}
    if al[0] < 0.75:
        k = settle(al, tol=0.08)
        return {"in": "fade", "in_dur": round(max(1, k) / fps, 3)}
    return {"in": "none"}


def _anim_out(path, ev, info, ref):
    fps = info["fps"] or 30
    W, H = info["width"], info["height"]
    t0 = max(0.0, ev["e"] - 0.5)
    frames = list(ff.frames(path, W, H, fps=fps, start=t0, duration=ev["e"] - t0))
    if len(frames) < 3:
        return {"out": "none"}
    st = [_glyph_state(f, ref) for f in frames]
    al = [s["alpha"] for s in st]
    ws = [s.get("w", 0) / ref["w"] for s in st]
    if al[-1] < 0.75 and al[0] > 0.85:
        k = next((i for i in range(len(al)) if al[i] < 0.92), len(al))
        dur = (len(al) - k) / fps
        shrink = ws[-1] < 0.9 and ws[-1] > 0
        return {"out": "pop" if shrink else "fade", "out_dur": round(max(1, len(al) - k) / fps, 3)}
    return {"out": "none"}


# ------------------------------------------------------------------ main

def _frames_at(path, info, times):
    """Full-resolution frames at the given sample times (one sequential decode)."""
    want = sorted(set(round(t * OCR_FPS) for t in times))
    out, wi = {}, 0
    if not want:
        return out
    for i, f in enumerate(ff.frames(path, info["width"], info["height"], fps=OCR_FPS)):
        if wi < len(want) and i == want[wi]:
            out[round(i / OCR_FPS, 3)] = f
            wi += 1
            if wi == len(want):
                break
    return out


def _scale_ev(ev, k):
    ev = dict(ev)
    ev["box"] = [v * k for v in ev["box"]]
    ev["lines"] = [[v * k for v in b] for b in ev["lines"]]
    ev["ink_h"] = ev["ink_h"] * k
    ev["seen"] = [(t, [v * k for v in b]) for t, b in ev["seen"]]
    return ev


def _case(texts):
    letters = "".join(re.sub(r"[^A-Za-z]", "", t) for t in texts)
    if not letters:
        return "none"
    if sum(c.isupper() for c in letters) / len(letters) > 0.9:
        return "upper"
    words = [w for t in texts for w in re.findall(r"[A-Za-z]+", t)]
    if words and sum(w[0].isupper() for w in words) / len(words) > 0.85:
        return "title"
    return "sentence"


def _style_of(group, size, transcript=None):
    """Aggregate measured events into one captions.py style + stats."""
    W, H = size
    mids = [g for g in group if g.get("m")]
    if not mids:
        return None
    pick = lambda key: [g["m"][key] for g in mids if g["m"].get(key)]
    fills = np.array([g["m"]["fill"] for g in mids])
    style = {"fill": _hex(np.median(fills, axis=0)), "align": "center"}
    fonts = Counter((g["font"]["font"], g["font"]["weight"]) for g in group if g.get("font"))
    if fonts:
        (fam, wgt), _ = fonts.most_common(1)[0]
        style["font"], style["weight"] = fam, wgt
    sizes = [g["size"] for g in group if g.get("size")]
    if sizes:
        style["size"] = round(float(np.median(sizes)), 1)
    strokes = pick("stroke")
    if len(strokes) > len(mids) / 2:
        style["stroke"] = {"color": _hex(np.median([np.array(C.rgba(s["color"])[:3][::-1]) for s in strokes], axis=0)),
                           "width": round(float(np.median([s["width"] for s in strokes])), 1)}
    shadows = pick("shadow")
    if len(shadows) > len(mids) / 2:
        style["shadow"] = {"color": "#000000",
                           "opacity": round(float(np.median([s["opacity"] for s in shadows])), 2),
                           "blur": round(float(np.median([s["blur"] for s in shadows])), 1),
                           "dx": round(float(np.median([s["dx"] for s in shadows])), 1),
                           "dy": round(float(np.median([s["dy"] for s in shadows])), 1)}
    boxes = [g for g in mids if g["m"].get("box")]
    if len(boxes) > len(mids) / 2:
        cols = np.array([C.rgba(g["m"]["box"]["color"])[:3][::-1] for g in boxes])
        modes = Counter(g["m"]["box"]["mode"] for g in boxes if len(g["lines"]) > 1)
        style["box"] = {"color": _hex(np.median(cols, axis=0)), "opacity": 1.0,
                        "radius": round(float(np.median([g["m"]["box"]["radius"] for g in boxes])), 1),
                        "pad_x": round(float(np.median([g["pad_x"] for g in boxes if g.get("pad_x") is not None]
                                                       or [0])), 1),
                        "pad_y": round(float(np.median([g["pad_y"] for g in boxes if g.get("pad_y") is not None]
                                                       or [0])), 1),
                        "mode": modes.most_common(1)[0][0] if modes else "line"}
    hl = [g["highlight"] for g in group if g.get("highlight")]
    multi = [g for g in group if len(g["text"].split()) > 1]
    if hl and len(hl) >= max(1, 0.3 * len(multi)):
        h = {}
        fills_h = [x["fill"] for x in hl if x.get("fill")]
        if fills_h:
            h["fill"] = Counter(fills_h).most_common(1)[0][0]
        bx = [x["box"] for x in hl if x.get("box")]
        if bx and len(bx) > len(hl) / 2:
            h["box"] = {"color": Counter(bx).most_common(1)[0][0]}
        sc = [x["scale"] for x in hl if x.get("scale")]
        if sc and np.median(sc) > 1.03:
            h["scale"] = round(float(np.median(sc)), 2)
        if h:
            style["highlight"] = h
    anims = Counter(g["anim"]["in"] for g in group if g.get("anim"))
    if anims:
        kind = anims.most_common(1)[0][0]
        a = {"in": kind}
        same = [g["anim"] for g in group if g.get("anim", {}).get("in") == kind]
        for key in ("in_dur", "overshoot", "from", "slide_px"):
            vals = [x[key] for x in same if key in x]
            if vals:
                a[key] = round(float(np.median(vals)), 3)
        outs = Counter(g["anim_out"]["out"] for g in group if g.get("anim_out"))
        if outs and outs.most_common(1)[0][0] != "none":
            a["out"] = outs.most_common(1)[0][0]
            a["out_dur"] = round(float(np.median([g["anim_out"]["out_dur"] for g in group
                                                  if g.get("anim_out", {}).get("out") == a["out"]])), 3)
        style["anim"] = a
    texts = [g["text"] for g in group]
    style["case"] = _case(texts) if _case(texts) != "sentence" else "none"
    lhs = [g["line_height"] for g in group if g.get("line_height")]
    style["line_height"] = round(float(np.median(lhs)), 3) if lhs else 1.2
    centres = [g["centre"] for g in group if g.get("centre")]
    if centres:
        style["x"] = round(float(np.median([c[0] for c in centres])) - W / 2, 1)
        style["y"] = round(float(np.median([c[1] for c in centres])) - H / 2, 1)
    widths = [g["line_w"] for g in group if g.get("line_w")]
    if widths:
        style["max_width"] = round(min(0.96, max(widths) / W + 0.03), 3)
    return style


def _lead(group, transcript):
    if not transcript:
        return None
    words = transcript.get("words", [])
    leads = []
    for g in group:
        first = _norm(g["text"].split()[0]) if g["text"].split() else ""
        cands = [w for w in words if _norm(w["w"]) == first and abs(w["s"] - g["s"]) < 1.0]
        if cands:
            w = min(cands, key=lambda w: abs(w["s"] - g["s"]))
            leads.append(g["s"] - w["s"])
    return round(float(np.median(leads)), 3) if leads else None


def analyze_text(path, info, gray=None, transcript=None):
    """Overlay text of a sample reel → SPEC.md captions / titles / overlays + boxes."""
    path = str(path)
    W0, H0, fps = info["width"], info["height"], info["fps"] or 30
    samples, ocr_size = ocr_samples(path, info)
    k = W0 / ocr_size[0]
    evs = [_scale_ev(e, k) for e in events(samples)]
    if gray is None:
        from .sample_video import decode_gray
        gray, _ = decode_gray(path, info)
    evs = refine_times(evs, gray, fps, (W0, H0), info["duration"])
    evs = [e for e in evs if e["e"] - e["s"] >= 1.5 / fps]
    # one full-resolution frame per event (a sample time inside it, past any entrance)
    for ev in evs:
        ts = [t for t, _ in ev["seen"]]
        ev["t_mid"] = ts[len(ts) // 2]
    mid = _frames_at(path, info, [ev["t_mid"] for ev in evs])
    kept = []
    for ev in evs:
        f = mid.get(round(ev["t_mid"], 3))
        m = measure(f, ev) if f is not None else None
        if m is None:
            continue
        feat = _features(ev, m, (W0, H0), info["duration"])
        if _is_candidate(feat, ev["text"]):
            ev["m"], ev["feat"] = m, feat
            kept.append(ev)
    # captions = the largest recurring group with one look and position; then titles / watermarks
    for ev in kept:
        _typography(path, info, ev, mid)
    groups = _group([e for e in kept if not e["feat"]["persistent"]], (W0, H0))
    groups = [g for g in groups if len(g) >= 3 and sum(e["e"] - e["s"] for e in g) >= 2.0]
    caps = max(groups, key=lambda g: sum(e["e"] - e["s"] for e in g)) if groups else []
    persistent = [e for e in kept if e["feat"]["persistent"] and e not in caps]
    # a title: on screen ≥ 1 s, centred, styled (box / outline / shadow) or large
    titles = [e for e in kept if e not in caps and e not in persistent and e["e"] - e["s"] >= 1.0
              and e["feat"]["centred"] and (e["feat"]["decorated"] or e["feat"]["ink_rel"] >= 0.03)]
    kept = caps + titles + persistent
    for e in caps:
        e["role"] = "caption"
    for e in titles:
        e["role"] = "title"
    for e in persistent:
        e["role"] = "persistent"

    def ev_out(e):
        return {"s": round(e["s"], 3), "e": round(e["e"], 3), "text": e["text"],
                "box": [int(round(v)) for v in (e["m"]["box"]["bounds"] if e.get("m") and e["m"].get("box")
                                                  else e["box"])], "role": e["role"]}
    cap_style = _style_of(caps, (W0, H0)) if caps else None
    wpc = [len(e["text"].split()) for e in caps]
    durs = [e["e"] - e["s"] for e in caps]
    lines = [len(e["lines"]) for e in caps]
    cpl = [max(len(l) for l in e["text"].split("\n")) for e in caps]
    captions = {"present": bool(caps), "style": cap_style or {},
                "words_per_caption": {"median": float(np.median(wpc)) if wpc else 0, "max": max(wpc) if wpc else 0},
                "max_lines": max(lines) if lines else 0, "chars_per_line": max(cpl) if cpl else 0,
                "duration": {"median": round(float(np.median(durs)), 3) if durs else 0,
                             "min": round(min(durs), 3) if durs else 0},
                "lead": _lead(caps, transcript), "case": (cap_style or {}).get("case", "none"),
                "events": [ev_out(e) for e in caps]}
    title_style = _style_of(titles, (W0, H0)) if titles else None
    titles_out = {"present": bool(titles), "style": title_style or {}, "events": [ev_out(e) for e in titles]}
    overlays = {"persistent": [{"text": e["text"], "box": ev_out(e)["box"], "s": round(e["s"], 3),
                                "e": round(e["e"], 3)} for e in persistent]}
    boxes = []
    for e in kept:
        b = e["m"]["box"]["bounds"] if e.get("m") and e["m"].get("box") else e["box"]
        pad = 0.25 * e["ink_h"]
        boxes.append({"s": round(e["s"], 3), "e": round(e["e"], 3),
                      "box": [int(max(0, b[0] - pad)), int(max(0, b[1] - pad)), int(min(W0, b[2] + pad)),
                              int(min(H0, b[3] + pad))]})
    return {"captions": captions, "titles": titles_out, "overlays": overlays, "boxes": boxes}


def _group(evs, size):
    """Cluster events into tracks: same height band, similar glyph size and fill colour."""
    W, H = size
    groups = []
    for e in sorted(evs, key=lambda e: e["s"]):
        cy = (e["box"][1] + e["box"][3]) / 2
        for g in groups:
            gcy = np.median([(x["box"][1] + x["box"][3]) / 2 for x in g])
            gh = np.median([x["ink_h"] for x in g])
            if abs(cy - gcy) < 0.07 * H and abs(e["ink_h"] / gh - 1) < 0.35 and \
                    _de(e["m"]["fill"], g[0]["m"]["fill"]) < 30:
                g.append(e)
                break
        else:
            groups.append([e])
    return groups


def _typography(path, info, ev, mid):
    """Per-event: font, size, centre, padding, line height, highlight, animation."""
    m = ev["m"]
    if m is None:
        return
    f = mid.get(round(ev["t_mid"], 3))
    X0, Y0 = m["origin"]
    texts = [t for t in ev["text"].split("\n")]
    glyph = m["ink"].copy()
    if m.get("stroke"):
        sd = _de_img(f[Y0:Y0 + glyph.shape[0], X0:X0 + glyph.shape[1]], C.rgba(m["stroke"]["color"])[:3][::-1])
        glyph &= sd > 20
    masks, line_tops, line_ws = [], [], []
    for lb in ev["lines"]:
        ly0, ly1 = int(lb[1]) - Y0, int(math.ceil(lb[3])) - Y0
        sub = glyph[max(0, ly0):max(0, ly1)]
        masks.append(sub)
        rows = np.where(sub.any(axis=1))[0]
        cols = np.where(sub.any(axis=0))[0]
        line_tops.append(Y0 + max(0, ly0) + (rows[0] if len(rows) else 0))
        line_ws.append((X0 + cols[0], X0 + cols[-1] + 1) if len(cols) else None)
    ranked = match_font(masks, texts)
    if not ranked:
        return
    best = ranked[0]
    ev["font"], ev["font_ranked"] = best, ranked[:5]
    sizes, centres = [], []
    for msk, t, top in zip(masks, texts, line_tops):
        s, c = font_size(msk, t, best["font"], best["weight"])
        if s:
            sizes.append(s)
            centres.append(top + c)
    if not sizes:
        return
    size = float(np.median(sizes))
    ev["size"] = size
    font = C._font(best["font"], best["weight"], size)
    advs = [font.getlength(t) for t in texts]
    ev["line_w"] = max(advs)
    if len(centres) > 1:
        pitch = float(np.median(np.diff(centres)))
        ev["line_height"] = pitch / size
    lh = size * ev.get("line_height", 1.2)
    # block centre (what captions.py positions): middle of the stack of line boxes
    bcy = (centres[0] + centres[-1]) / 2
    xs = [w for w in line_ws if w]
    bcx = float(np.median([(a + b) / 2 for a, b in xs])) if xs else (ev["box"][0] + ev["box"][2]) / 2
    ev["centre"] = (bcx, bcy)
    if m.get("box"):
        b = m["box"]["bounds"]
        ev["pad_y"] = max(0.0, ((b[3] - b[1]) - lh * len(texts)) / 2)
        widest = max(advs)
        rows = m["box"]["rows"]
        row_w = [r[1] - r[0] for r in rows]
        ev["pad_x"] = max(0.0, (max(row_w) - widest) / 2)
        ev["centre"] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    # active-word highlight
    words = word_fills(f, m, ev)
    if len(words) >= 2:
        base = np.median(np.array([w["color"] for w in words]), axis=0)
        odd = [w for w in words if _de(w["color"], base) > 25]
        hl = None
        if len(odd) == 1:
            hl = {"fill": _hex(odd[0]["color"])}
        boxy = [w for w in words if _de(w["around"], m["near"]) > 25]
        if len(boxy) == 1:
            hl = dict(hl or {}, box=_hex(boxy[0]["around"]))
        if hl:
            idx = words.index(odd[0] if odd else boxy[0])
            hs = [w["h"] for i, w in enumerate(words) if i != idx and w["h"]]
            if hs and words[idx]["h"]:
                hl["scale"] = round(words[idx]["h"] / float(np.median(hs)), 3)
            ev["highlight"] = hl
    # animation, against the settled look of this frame
    bx0, by0, bx1, by1 = m["ink_box"]
    pad_w, pad_h = int(0.6 * (bx1 - bx0)), int(1.2 * (by1 - by0)) + int(0.8 * ev["ink_h"])
    H0, W0 = f.shape[:2]
    region = [max(0, bx0 - pad_w), max(0, by0 - pad_h), min(W0, bx1 + pad_w), min(H0, by1 + pad_h)]
    crop = f[region[1]:region[3], region[0]:region[2]]
    d = _de_img(crop, m["fill"])
    tol = 30
    smask = d < tol
    pix = np.zeros_like(smask)
    fm = m["fill_mask"]
    oy, ox = Y0 - region[1], X0 - region[0]
    pix[max(0, oy):max(0, oy) + fm.shape[0], max(0, ox):max(0, ox) + fm.shape[1]] = \
        fm[:pix.shape[0] - max(0, oy), :pix.shape[1] - max(0, ox)]
    pix &= smask
    ys, xs = np.where(smask)
    if len(ys) < 20 or pix.sum() < 20:
        return
    ref = {"region": region, "fill": m["fill"], "tol": tol, "area": int(smask.sum()),
           "w": xs.max() - xs.min() + 1, "h": ys.max() - ys.min() + 1, "cy": ys.mean(), "cx": xs.mean(),
           "pix": pix, "fill_lab": _lab(m["fill"])[0], "near_lab": _lab(m["near"])[0]}
    try:
        ev["anim"] = _anim_in(path, ev, info, ref)
        ev["anim_out"] = _anim_out(path, ev, info, ref)
    except Exception as exc:          # a broken decode must not lose the event
        ff.log(f"animation of {ev['text'][:20]!r}: {exc}")
