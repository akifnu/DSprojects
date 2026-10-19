"""On-screen text of a sample reel: every caption / title / watermark as a timed
event, and the typography that draws it (font, size, colours, box, outline,
highlight, animation) measured from the pixels.

OCR: RapidOCR (PP-OCR ONNX models bundled in the wheel), 5 samples/s in a process pool.
Event start/end are refined to the exact frame on the full-frame-rate grey pass.
"""
import difflib
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import captions as C
from . import ff

OCR_FPS = 5
_ocr = None


def _init():
    global _ocr
    from rapidocr_onnxruntime import RapidOCR
    _ocr = RapidOCR()


def _run(item):
    t, frame = item
    res, _ = _ocr(frame)
    out = []
    for box, text, conf in res or []:
        conf = float(conf)
        if conf < 0.6 or not text.strip():
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append({"text": text.strip(), "conf": conf, "box": [min(xs), min(ys), max(xs), max(ys)]})
    return t, out


def ocr_samples(path, info, width=720):
    w = min(width, info["width"])
    h = int(round(info["height"] * w / info["width"] / 2)) * 2
    items = [(round(i / OCR_FPS, 3), f) for i, f in enumerate(ff.frames(path, w, h, fps=OCR_FPS))]
    with ProcessPoolExecutor(max(1, min(4, os.cpu_count() or 4)), initializer=_init) as pool:
        results = list(pool.map(_run, items, chunksize=4))
    return results, (w, h)


def _norm(s):
    return re.sub(r"[^\w]+", "", s.lower())


def _blocks(dets, H):
    """Merge the OCR lines of one frame into text blocks (multi-line captions)."""
    dets = sorted(dets, key=lambda d: d["box"][1])
    blocks = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        h = y1 - y0
        for b in blocks:
            bx0, by0, bx1, by1 = b["box"]
            lh = b["lines"][-1]["box"][3] - b["lines"][-1]["box"][1]
            gap = y0 - by1
            cx, bcx = (x0 + x1) / 2, (bx0 + bx1) / 2
            overlap = min(x1, bx1) - max(x0, bx0)
            if -0.3 * h < gap < 0.6 * min(h, lh) and 0.6 < h / lh < 1.6 and \
                    (abs(cx - bcx) < 0.15 * (bx1 - bx0 + x1 - x0) or overlap > 0.3 * min(x1 - x0, bx1 - bx0)):
                b["lines"].append(d)
                b["box"] = [min(bx0, x0), by0, max(bx1, x1), y1]
                b["text"] += "\n" + d["text"]
                break
        else:
            blocks.append({"lines": [d], "box": list(d["box"]), "text": d["text"]})
    return blocks


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def events(samples, size):
    """Track OCR blocks across samples into timed text events."""
    W, H = size
    dt = 1.0 / OCR_FPS
    active, done = [], []
    for t, dets in samples:
        blocks = _blocks(dets, H)
        used = set()
        for ev in active:
            best = None
            for i, b in enumerate(blocks):
                if i in used:
                    continue
                sim = difflib.SequenceMatcher(None, _norm(ev["text"]), _norm(b["text"])).ratio()
                grows = _norm(b["text"]).startswith(_norm(ev["text"])) and len(_norm(b["text"])) > len(_norm(ev["text"]))
                if (sim > 0.8 or grows) and _iou(ev["box"], b["box"]) > 0.15:
                    if best is None or sim > best[1]:
                        best = (i, sim, grows)
            if best is not None:
                i, _, grows = best
                used.add(i)
                b = blocks[i]
                if grows:
                    ev.setdefault("growth", []).append((t, b["text"]))
                if len(b["text"]) >= len(ev["text"]) or b["lines"][0]["conf"] > ev["conf"]:
                    ev["text"], ev["box"], ev["lines"] = b["text"], b["box"], [l["box"] for l in b["lines"]]
                ev["last"], ev["miss"] = t, 0
                ev["seen"].append(t)
            else:
                ev["miss"] += 1
        still = []
        for ev in active:
            (done if ev["miss"] > 1 else still).append(ev)
        active = still
        for i, b in enumerate(blocks):
            if i not in used:
                active.append({"text": b["text"], "box": b["box"], "lines": [l["box"] for l in b["lines"]],
                               "first": t, "last": t, "miss": 0, "seen": [t],
                               "conf": min(l["conf"] for l in b["lines"])})
    done += active
    out = []
    for ev in done:
        if len(ev["seen"]) < 2 and ev["conf"] < 0.85:
            continue                      # one low-confidence glimpse: OCR noise
        out.append({"text": ev["text"], "s": ev["first"], "e": round(ev["last"] + dt, 3),
                    "box": [round(v, 1) for v in ev["box"]], "lines": ev["lines"],
                    "growth": ev.get("growth", [])})
    return sorted(out, key=lambda e: (e["s"], e["box"][1]))


def refine_times(evs, gray, fps, size, duration):
    """Snap each event's start/end to the frame where its region changes most."""
    W, H = size
    n, gh, gw = gray.shape
    sx, sy = gw / W, gh / H
    dt = 1.0 / OCR_FPS
    for ev in evs:
        x0, y0, x1, y1 = ev["box"]
        r = (slice(max(0, int(y0 * sy) - 1), min(gh, int(math.ceil(y1 * sy)) + 1)),
             slice(max(0, int(x0 * sx) - 1), min(gw, int(math.ceil(x1 * sx)) + 1)))

        def jump(a, b):
            a, b = max(1, a), min(n - 1, b)
            if b <= a:
                return None
            diffs = [np.abs(gray[k][r].astype(np.float32) - gray[k - 1][r]).mean() for k in range(a, b + 1)]
            return a + int(np.argmax(diffs))
        k = jump(int((ev["s"] - dt) * fps) + 1, int(ev["s"] * fps) + 1)
        if k is not None:
            ev["s"] = round(k / fps, 3)
        k = jump(int((ev["e"] - dt) * fps), int(ev["e"] * fps) + 1)
        if k is not None:
            ev["e"] = round(k / fps, 3)
        ev["e"] = min(ev["e"], round(duration, 3))
    return evs


def classify(evs, duration, size):
    W, H = size
    for ev in evs:
        d = ev["e"] - ev["s"]
        cy = (ev["box"][1] + ev["box"][3]) / 2 / H
        if d > 0.7 * duration:
            ev["role"] = "persistent"
        elif cy < 0.3 and d > 2.0:
            ev["role"] = "title"
        else:
            ev["role"] = "caption"
    return evs


# ------------------------------------------------------------- typography

def _lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _hex(bgr):
    b, g, r = [int(round(v)) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def _de(c1, c2):
    a = cv2.cvtColor(np.float32([[c1]]) / 255, cv2.COLOR_BGR2LAB)[0, 0]
    b = cv2.cvtColor(np.float32([[c2]]) / 255, cv2.COLOR_BGR2LAB)[0, 0]
    return float(np.linalg.norm(a - b))


def measure(frame, ev):
    """Colours, box, outline, size and ink geometry of one text event in one frame."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in ev["box"]]
    lh = float(np.median([b[3] - b[1] for b in ev["lines"]]))
    core = frame[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if core.size == 0:
        return None
    lab = _lab(core).reshape(-1, 3)
    # background inside the text box = the most common colour among its pixels
    Z = np.float32(lab)
    k = 3 if len(Z) > 50 else 1
    _, labels, centers = cv2.kmeans(Z, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3,
                                    cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    counts = np.bincount(labels, minlength=k)
    # border pixels of the OCR box are mostly background
    hh, ww = core.shape[:2]
    edge = np.zeros((hh, ww), bool)
    edge[:2] = edge[-2:] = True
    edge[:, :2] = edge[:, -2:] = True
    bg_k = int(np.bincount(labels[edge.reshape(-1)], minlength=k).argmax())
    bg_lab = centers[bg_k]
    dist = np.linalg.norm(lab - bg_lab, axis=1).reshape(hh, ww)
    thr, _ = cv2.threshold(np.clip(dist * 2, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (dist * 2 > max(thr, 30))
    if ink.sum() < 20:
        return None
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    interior = dt >= max(1.5, dt[ink].max() * 0.5)
    core_bgr = core.reshape(-1, 3)
    fill = np.median(core_bgr[interior.reshape(-1)], axis=0) if interior.sum() > 10 else np.median(core_bgr[ink.reshape(-1)], axis=0)
    rim = ink & (dt <= 1.5)
    stroke = None
    if rim.sum() > 20:
        rim_c = np.median(core_bgr[rim.reshape(-1)], axis=0)
        if _de(rim_c, fill) > 25 and _de(rim_c, _from_lab(bg_lab)) > 15:
            outer = ink & ~interior
            width = float(outer.sum() / max(1, _perimeter(interior)))
            stroke = {"color": _hex(rim_c), "width_ratio": round(width / lh, 3)}
    # box: background inside is uniform and differs from what is around it
    bg_bgr = _from_lab(bg_lab)
    inside_bg = core_bgr[(labels == bg_k)]
    uniform = float(np.std(_lab(inside_bg[None]).reshape(-1, 3), axis=0).mean()) < 6 if len(inside_bg) else False
    far = _ring(frame, (x0, y0, x1, y1), int(0.9 * lh), int(1.4 * lh))
    box = None
    if uniform and far is not None and _de(np.median(far, axis=0), bg_bgr) > 12:
        bx = _box_extent(frame, (x0, y0, x1, y1), bg_bgr, lh)
        box = {"color": _hex(bg_bgr), "opacity": 1.0, **bx}
    ys, xs = np.where(ink)
    ink_box = [x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1]
    return {"fill": _hex(fill), "stroke": stroke, "box": box, "line_px": lh, "ink_box": [int(v) for v in ink_box],
            "ink": ink, "core_origin": (x0, y0), "bg": _hex(bg_bgr)}


def _from_lab(lab):
    return cv2.cvtColor(np.float32([[lab]]), cv2.COLOR_LAB2BGR)[0, 0] * 255


def _perimeter(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return sum(len(c) for c in cs)


def _ring(frame, box, r0, r1):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box
    X0, Y0, X1, Y1 = max(0, x0 - r1), max(0, y0 - r1), min(W, x1 + r1), min(H, y1 + r1)
    m = np.zeros((Y1 - Y0, X1 - X0), bool)
    m[:] = True
    ix0, iy0, ix1, iy1 = max(0, x0 - r0) - X0, max(0, y0 - r0) - Y0, min(W, x1 + r0) - X0, min(H, y1 + r0) - Y0
    m[iy0:iy1, ix0:ix1] = False
    px = frame[Y0:Y1, X0:X1][m]
    return px if len(px) > 20 else None


def _box_extent(frame, box, color, lh):
    """Padding and corner radius of a solid box around the text, measured along its edges."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2

    def run(points):
        n = 0
        for (y, x) in points:
            if not (0 <= y < H and 0 <= x < W) or _de(frame[y, x], color) > 18:
                break
            n += 1
        return n
    left = run([(cy, x) for x in range(x0, -1, -1)])
    right = run([(cy, x) for x in range(x1 - 1, W)])
    top = run([(y, cx) for y in range(y0, -1, -1)])
    bottom = run([(y, cx) for y in range(y1 - 1, H)])
    bx0, bx1, by0, by1 = x0 - left, x1 - 1 + right, y0 - top, y1 - 1 + bottom
    # corner radius: how far along the diagonal from the corner the box colour starts
    rad = 0
    for i in range(int(lh)):
        y, x = by0 + i, bx0 + i
        if 0 <= y < H and 0 <= x < W and _de(frame[y, x], color) < 18:
            rad = i * 2 / (2 - math.sqrt(2))     # diagonal inset of a quarter circle
            break
    return {"pad_x": round(max(0, left - 1), 1), "pad_y": round(max(0, top - 1), 1), "radius": round(rad, 1),
            "x0": bx0, "x1": bx1, "y0": by0, "y1": by1}


# ------------------------------------------------------------- font matching

CANDIDATES = ["Montserrat", "Poppins", "Inter", "Roboto", "Open Sans", "Lato", "Arimo", "Bebas Neue", "Anton",
              "Oswald", "Archivo Black", "Archivo", "Rubik", "Nunito", "Raleway", "Work Sans", "DM Sans", "Barlow",
              "Barlow Condensed", "League Spartan", "Kanit", "Bangers", "Luckiest Guy", "Titan One", "Lilita One",
              "Fredoka", "Noto Sans", "Manrope", "Outfit", "Sora", "Plus Jakarta Sans", "Roboto Condensed", "Teko",
              "Righteous", "Bungee", "Playfair Display", "Merriweather", "Lora", "Permanent Marker", "Caveat"]
WEIGHTS = [400, 600, 700, 800, 900]


def _ink_of(text, family, weight, size):
    font = C._font(family, weight, size)
    asc, desc = font.getmetrics()
    w = int(font.getlength(text)) + 8
    img = Image.new("L", (w, asc + desc + 8), 0)
    ImageDraw.Draw(img).text((4, 4), text, font=font, fill=255)
    a = np.array(img) > 127
    ys, xs = np.where(a)
    if not len(ys):
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def match_font(ink_line, text, families=None, weights=None):
    """Rank (family, weight, size_px) by how well the rendered text overlays the measured ink."""
    ys, xs = np.where(ink_line)
    if len(ys) < 30:
        return []
    meas = ink_line[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    mh, mw = meas.shape
    results = []
    for fam in families or CANDIDATES:
        for wgt in weights or WEIGHTS:
            try:
                ref = _ink_of(text, fam, wgt, 100)
            except Exception:
                continue
            if ref is None:
                continue
            rh, rw = ref.shape
            size = 100 * mh / rh
            ratio = (rw * mh / rh) / mw                   # width at matched height vs measured
            fit = cv2.resize(ref.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_AREA) > 0
            iou = (fit & meas).sum() / max(1, (fit | meas).sum())
            results.append({"font": fam, "weight": wgt, "size": round(size, 1), "iou": round(float(iou), 3),
                            "width_ratio": round(ratio, 3), "score": round(float(iou) - 1.5 * abs(math.log(ratio)), 3)})
    return sorted(results, key=lambda r: -r["score"])


def line_inks(m, ev):
    """Per OCR line: the ink mask clipped to that line."""
    x0, y0 = m["core_origin"]
    out = []
    for lb in ev["lines"]:
        ly0, ly1 = int(lb[1]) - y0, int(math.ceil(lb[3])) - y0
        lx0, lx1 = int(lb[0]) - x0, int(math.ceil(lb[2])) - x0
        sub = m["ink"][max(0, ly0):max(0, ly1), max(0, lx0):max(0, lx1)]
        out.append(sub)
    return out


def word_colors(frame, m, ev):
    """Median fill colour of each word (ink split at horizontal gaps), per line."""
    x0, y0 = m["core_origin"]
    res = []
    for lb, sub in zip(ev["lines"], line_inks(m, ev)):
        if sub.size == 0:
            continue
        cols = sub.any(axis=0)
        gap_min = max(2, int(0.18 * sub.shape[0]))
        words, start, gap = [], None, 0
        for i, on in enumerate(cols):
            if on:
                if start is None:
                    start = i
                gap = 0
            elif start is not None:
                gap += 1
                if gap >= gap_min:
                    words.append((start, i - gap + 1))
                    start, gap = None, 0
        if start is not None:
            words.append((start, len(cols)))
        oy, ox = max(0, int(lb[1])), max(0, int(lb[0]))
        for a, b in words:
            mask = sub[:, a:b]
            px = frame[oy:oy + mask.shape[0], ox + a:ox + b][mask[:frame[oy:oy + mask.shape[0], ox + a:ox + b].shape[0],
                                                                   :frame[oy:oy + mask.shape[0], ox + a:ox + b].shape[1]]]
            if len(px) > 5:
                res.append(np.median(px, axis=0))
    return res
