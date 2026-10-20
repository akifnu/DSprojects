"""Picture structure of a sample reel: shots, transitions, punch-ins, camera moves,
pacing and the colour look.

Everything runs on every frame decoded at 180 px width (grey). For each frame k the
previous frame is aligned to it with a similarity transform (LK flow + RANSAC on
features outside text overlays), and the frame is split into a grid of cells:
a cell has *changed* when, after alignment, its normalised correlation collapses
(textured cells) or its mean jumps (flat cells). An edit changes most of the frame
at once (high spread, isolated in time); typing, menus, cursors and moving people
change a few cells or change steadily, so they are not cuts. Multi-frame events
(zoom / whip / flash / dip) are runs of abnormal frames classified by scale ramps,
blur, blur direction and luma; dissolves are found by a blend fit between frames
far apart, since each dissolve frame differs only slightly from the last.

Text overlays (captions, titles, watermarks) are masked out of every measurement.
Positions are fractions of the frame; scales are factors — they transfer to any
output size.
"""
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import ff, grade
from .render import ease

AW = 180
ROWS, COLS = 16, 9
FEATURES = dict(maxCorners=200, qualityLevel=0.01, minDistance=6, blockSize=5)
EASE_NAMES = ["linear", "inout", "in", "out", "inout_cubic", "in_cubic", "out_cubic"]
# ffmpeg vignette (render.grade_filters: angle = pi/5 * v): corner/centre luma ratio per v
_VIGNETTE = [(0.0, 1.0), (0.25, 0.956), (0.5, 0.837), (1.0, 0.46), (1.5, 0.10), (2.0, 0.0)]
# ffmpeg noise c0s (render grain) → added noise std (8-bit levels)
_GRAIN = [(0, 0.0), (2, 0.5), (4, 1.09), (8, 2.62), (12, 4.0), (20, 6.59)]


def decode_gray(path, info):
    ah = int(round(info["height"] * AW / info["width"] / 2)) * 2
    # area pre-scale: much cheaper than lanczos from full size, and anti-aliased
    frames = np.stack(list(ff.frames(path, AW, ah, fps=info["fps"] or 30, pix="gray",
                                     vf=f"scale={AW}:{ah}:flags=area")))
    return frames, ah


# ----------------------------------------------------------------- masks

def _mask_fn(text_boxes, info, shape, fps):
    """Per-frame uint8 mask (255 = usable) excluding text overlays (with a margin)."""
    H, W = shape
    sx, sy = W / info["width"], H / info["height"]
    boxes = []
    for tb in text_boxes or ():
        x0, y0, x1, y1 = tb["box"]
        mx, my = 0.12 * (x1 - x0) + 3 / sx, 0.25 * (y1 - y0) + 3 / sy
        boxes.append((int(math.floor(tb["s"] * fps)) - 2, int(math.ceil(tb["e"] * fps)) + 2,
                      max(0, int((x0 - mx) * sx)), max(0, int((y0 - my) * sy)),
                      min(W, int(math.ceil((x1 + mx) * sx))), min(H, int(math.ceil((y1 + my) * sy)))))
    full = np.full((H, W), 255, np.uint8)
    cache = {}

    def at(k):
        active = tuple(i for i, b in enumerate(boxes) if b[0] <= k <= b[1])
        if active not in cache:
            m = full.copy()
            for i in active:
                _, _, x0, y0, x1, y1 = boxes[i]
                m[y0:y1, x0:x1] = 0
            cache[active] = m
        return cache[active]
    return at


# ----------------------------------------------------------------- geometry

def _similarity(a, b, mask=None, level=3):
    """Similarity transform taking frame a to frame b: (scale, tx, ty, inlier_ratio, M, n_inliers)."""
    pts = cv2.goodFeaturesToTrack(a, mask=mask, **FEATURES)
    if pts is None or len(pts) < 12:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=level)
    ok = st.reshape(-1) == 1
    if ok.sum() < 10:
        return None
    M, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                         maxIters=2000, confidence=0.995)
    if M is None:
        return None
    s = math.hypot(M[0, 0], M[1, 0])
    return s, M[0, 2], M[1, 2], float(inl.mean()) * ok.mean(), M, int(inl.sum())


def _match(a, b, mask=None):
    """Similarity between two frames across a cut: ORB + RANSAC, refined with ECC.
    Returns (scale, M, inliers) or None when they do not show the same scene."""
    orb = cv2.ORB_create(2000, scaleFactor=1.15, nlevels=10, edgeThreshold=9, patchSize=15, fastThreshold=8)
    ka, da = orb.detectAndCompute(a, mask)
    kb, db = orb.detectAndCompute(b, mask)
    if da is None or db is None or len(ka) < 15 or len(kb) < 15:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 12:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in good])
    pb = np.float32([kb[x.trainIdx].pt for x in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=4000)
    if M is None or inl.sum() < 12 or inl.mean() < 0.3:
        return None
    try:   # sub-pixel refinement of the whole-image alignment
        warp = M.astype(np.float32)
        _, warp = cv2.findTransformECC(b.astype(np.float32), a.astype(np.float32), cv2.invertAffineTransform(warp),
                                       cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
                                       mask, 3)
        M2 = cv2.invertAffineTransform(warp)
        s2 = math.sqrt(abs(np.linalg.det(M2[:, :2])))
        if abs(math.log(s2 / math.hypot(M[0, 0], M[1, 0]))) < 0.05:
            M = M2
    except cv2.error:
        pass
    return math.sqrt(abs(np.linalg.det(M[:, :2]))), M, int(inl.sum())


def _cell_stats(img, rows=ROWS, cols=COLS):
    H, W = img.shape
    ch, cw = H // rows, W // cols
    c = img[:ch * rows, :cw * cols].reshape(rows, ch, cols, cw).transpose(0, 2, 1, 3).reshape(rows, cols, -1)
    return c


def _change(a, b, valid=None):
    """(spread, informative cells, changed-cell map) between two aligned grey frames."""
    A, B = _cell_stats(a.astype(np.float32)), _cell_stats(b.astype(np.float32))
    ma, mb = A.mean(-1), B.mean(-1)
    sa, sb = A.std(-1), B.std(-1)
    cov = ((A - ma[..., None]) * (B - mb[..., None])).mean(-1)
    ncc = cov / (sa * sb + 1e-6)
    tex = (sa > 4) & (sb > 4)
    one_tex = (sa > 4) ^ (sb > 4)
    jump = np.abs(ma - mb) > 14
    inf = tex | one_tex | jump
    if valid is not None:
        inf &= valid
    changed = inf & ((tex & (ncc < 0.5)) | (one_tex & (np.abs(sa - sb) > 6)) | jump)
    n = int(inf.sum())
    return (changed.sum() / n if n else 0.0), n, changed


def _valid_cells(mask, M=None):
    m = mask
    if M is not None:
        H, W = mask.shape
        m = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    c = _cell_stats(m.astype(np.float32))
    return c.mean(-1) > 0.7 * 255


def _sharp(f, mask=None):
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    if mask is not None:
        sel = mask > 0
        return float(np.abs(gx)[sel].mean()), float(np.abs(gy)[sel].mean())
    return float(np.abs(gx).mean()), float(np.abs(gy).mean())


# ----------------------------------------------------------------- per-frame signals

def frame_signals(gray, mask_at):
    """Short-range signals for every frame k (relative to k-1)."""
    n, H, W = gray.shape
    sig = {k: np.zeros(n) for k in ("spread", "raw", "logs", "tx", "ty", "inl", "luma", "gx", "gy")}
    sig["tracked"] = np.zeros(n, bool)
    sig["cmap"] = np.zeros((n, ROWS, COLS), bool)
    sig["energy"] = np.zeros((n, ROWS, COLS), np.float32)     # per-cell mean |Δ| after alignment
    for k in range(n):
        m = mask_at(k)
        sel = m > 0
        sig["luma"][k] = float(gray[k][sel].mean()) if sel.any() else float(gray[k].mean())
        sig["gx"][k], sig["gy"][k] = _sharp(gray[k], m)
        if k == 0:
            continue
        a, b = gray[k - 1], gray[k]
        ad = np.abs(a.astype(np.int16) - b)
        if not (ad[sel] > 6).any():                 # unchanged frame (static screen): skip tracking
            sig["tracked"][k], sig["inl"][k] = True, 1.0
            sig["energy"][k] = _cell_stats(ad.astype(np.float32)).mean(-1)
            continue
        valid = _valid_cells(m) & _valid_cells(mask_at(k - 1))
        raw, _, _ = _change(a, b, valid)
        sig["raw"][k] = raw
        r = _similarity(a, b, mask_at(k - 1) & m)
        if r is not None and r[3] > 0.45 and r[5] >= 10:
            s, tx, ty, q, M, _ = r
            warped = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            sp, _, cm = _change(warped, b, valid & _valid_cells(np.full_like(m, 255), M))
            sig["spread"][k], sig["logs"][k], sig["tx"][k], sig["ty"][k], sig["inl"][k] = \
                min(sp, raw), math.log(s), tx / W, ty / H, q
            sig["tracked"][k] = True
            sig["cmap"][k] = cm
            sig["energy"][k] = np.abs(_cell_stats(warped.astype(np.float32)) - _cell_stats(b.astype(np.float32))).mean(-1)
        else:
            sig["spread"][k] = raw
            sig["cmap"][k] = _change(a, b, valid)[2]
            sig["energy"][k] = np.abs(_cell_stats(a.astype(np.float32)) - _cell_stats(b.astype(np.float32))).mean(-1)
    return sig


# ----------------------------------------------------------------- boundaries

def _isolated_peak(x, k, lo):
    nb = [x[j] for j in (k - 2, k - 1, k + 1, k + 2) if 0 < j < len(x)]
    return x[k] - (max(nb) if nb else 0) > 0.3 or (x[k] > 0.5 and (max(nb) if nb else 0) < lo)


def _luma_events(luma, fps):
    """Flashes / dips: luma excursions towards white or black that return."""
    n = len(luma)
    out = []
    w = max(4, int(round(0.25 * fps)))
    k = 1
    while k < n - 1:
        pre = luma[max(0, k - 2 * w):max(1, k - w)]
        post = luma[min(n - 1, k + w):min(n, k + 2 * w)]
        if not len(pre) or not len(post):
            k += 1
            continue
        lo_b, hi_b = min(np.median(pre), np.median(post)), max(np.median(pre), np.median(post))
        base = (np.median(pre) + np.median(post)) / 2
        if luma[k] > hi_b + 45 and luma[k] >= luma[k - 1] and luma[k] >= luma[k + 1]:
            kind, peak = ("white", luma[k])
        elif luma[k] < lo_b - 45 and luma[k] <= luma[k - 1] and luma[k] <= luma[k + 1] and luma[k] < 40:
            kind, peak = ("black", luma[k])
        else:
            k += 1
            continue
        # extent: frames whose excursion exceeds 5 % of the peak excursion
        exc = np.abs(luma - base) / max(1e-6, abs(peak - base))
        a = k
        while a > 0 and exc[a - 1] > 0.05 and k - a < 3 * w:
            a -= 1
        b = k
        while b < n - 1 and exc[b + 1] > 0.05 and b - k < 3 * w:
            b += 1
        seg = exc[a:b + 1]
        u = np.linspace(-1, 1, len(seg) + 2)[1:-1]
        tri = 1 - np.abs(u)
        sharp = np.mean((seg - tri ** 2) ** 2) < np.mean((seg - tri) ** 2)
        out.append({"centre": k, "a": a, "b": b,
                    "type": ("flash" if sharp else "dip_white") if kind == "white" else "dip_black"})
        k = b + 1
    return out


def _dissolves(gray, mask_at, fps, busy, sig=None):
    """Dissolves: windows where middle frames are a linear blend of the frames before
    and after, with the blend weight rising monotonically from 0 to 1."""
    n = len(gray)
    w = max(3, int(round(0.13 * fps)))
    small = gray[:, ::2, ::2].astype(np.float32)
    lr = np.zeros(n)
    for k in range(w, n - w):
        valid = _valid_cells(mask_at(k - w)[::2, ::2]) & _valid_cells(mask_at(k + w)[::2, ::2])
        lr[k] = _change(small[k - w], small[k + w], valid)[0] if valid.any() else 0
        if lr[k] >= 0.45:      # camera/digital motion is not a change: align and look again
            r = _similarity(gray[k - w], gray[k + w], mask_at(k - w) & mask_at(k + w), level=4)
            if r is not None and r[3] > 0.6:
                M = r[4].copy()
                M[:, 2] /= 2
                H2, W2 = small.shape[1:]
                warped = cv2.warpAffine(small[k - w], M, (W2, H2), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
                lr[k] = min(lr[k], _change(warped, small[k + w], valid & _valid_cells(
                    np.full(small.shape[1:], 255, np.uint8), M))[0])
    out, k = [], w
    while k < n - w:
        if lr[k] < 0.45 or busy[k]:
            k += 1
            continue
        j = k
        while j + 1 < n - w and lr[j + 1] >= 0.45:
            j += 1
        d0, d1 = k + w - 1, j - w + 1             # first / last blended frame if a dissolve
        moving = sig is not None and any(sig["tracked"][q] and (abs(sig["logs"][q]) > 0.012 or
                                         math.hypot(sig["tx"][q], sig["ty"][q]) > 0.015)
                                         for q in range(max(1, d0 - 1), min(n, d1 + 2)))
        if d1 - d0 >= 1 and d0 >= 1 and d1 < n - 1 and not busy[d0:d1 + 1].any() and not moving:
            A, B = small[d0 - 1], small[d1 + 1]
            diff = (B - A).reshape(-1)
            den = float(diff @ diff)
            if den > 0 and np.abs(diff).mean() > 6:
                alphas, resid = [], []
                for f in range(d0, d1 + 1):
                    x = (small[f] - A).reshape(-1)
                    al = float(x @ diff) / den
                    alphas.append(al)
                    resid.append(np.abs(x - al * diff).mean())
                alphas = np.array(alphas)
                end_err = np.abs(diff).mean()
                mono = np.all(np.diff(alphas) > -0.05)
                vis = int(((alphas > 0.03) & (alphas < 0.97)).sum())
                if vis >= 2 and mono and alphas[0] < 0.4 and alphas[-1] > 0.6 and np.mean(resid) < 0.3 * end_err:
                    # widen to the frames that still blend
                    a, b = d0, d1
                    cen = d0 + int(np.argmin(np.abs(alphas - 0.5)))
                    if alphas[np.argmin(np.abs(alphas - 0.5))] > 0.5 and cen > d0:
                        cen = cen if abs(alphas[cen - d0] - 0.5) < abs(alphas[cen - d0 - 1] - 0.5) else cen - 1
                    cen = int(round(d0 + np.interp(0.5, alphas, np.arange(len(alphas)))))
                    out.append({"centre": cen, "a": a, "b": b, "type": "dissolve",
                                "frames": _dissolve_frames(small, cen, max(vis, d1 - d0 + 1))})
        k = j + 1
    return out


def _dissolve_frames(small, c, seen, hmax=30):
    """Number of visibly blended frames of the dissolve centred on c: the widest window
    whose ends are pure A / pure B with the blend weight rising monotonically between
    (an eased dissolve barely changes near its ends, so the core alone undercounts)."""
    n = len(small)
    best = seen
    for h in range(max(3, seen // 2 + 1), hmax + 1):
        if c - h < 0 or c + h >= n:
            break
        A, B = small[c - h], small[c + h]
        d = (B - A).reshape(-1)
        den = float(d @ d)
        if den <= 0:
            break
        al = np.array([float((small[f] - A).reshape(-1) @ d) / den for f in range(c - h, c + h + 1)])
        res = np.mean([np.abs((small[f] - A).reshape(-1) - a_ * d).mean() for f, a_ in zip(range(c - h, c + h + 1), al)])
        if not np.all(np.diff(al) > -0.04) or res > 0.3 * np.abs(d).mean():
            break
        inner = int(((al > 0.02) & (al < 0.98)).sum())
        if al[1] < 0.02 and al[-2] > 0.98:       # flat ends reached: the dissolve is inside
            return inner
        best = max(best, inner)
    return best


def boundaries(gray, sig, mask_at, fps):
    """Edits as [(frame, transition)] where frame is the first frame of the new shot
    (the switch frame for transitions) and transition is "cut" or {"type", "dur"}."""
    n = len(gray)
    sp, logs, tr = sig["spread"], sig["logs"], sig["tracked"]
    gsum = sig["gx"] + sig["gy"]
    dl = np.zeros(n)
    dl[1:] = np.abs(np.diff(sig["luma"]))
    flag = np.zeros(n, bool)
    for k in range(1, n):
        flag[k] = sp[k] > 0.45 or (tr[k] and abs(logs[k]) > 0.035) or (tr[k] and math.hypot(sig["tx"][k], sig["ty"][k]) > 0.1)
    events = []
    used = np.zeros(n, bool)
    # flashes / dips first (they also trip the change detector)
    for ev in _luma_events(sig["luma"], fps):
        a, b = ev["a"], ev["b"]
        events.append((ev["centre"], {"type": ev["type"], "dur": round((b - a + 2) / fps, 3)}))
        used[max(0, a - 1):b + 2] = True
    # runs of abnormal frames
    k = 1
    while k < n:
        if not flag[k] or used[k]:
            k += 1
            continue
        j = k
        while j + 1 < n and (flag[j + 1] or (j + 2 < n and flag[j + 2])) and not used[j + 1]:
            j += 1
        L = j - k + 1
        if L == 1:
            if _isolated_peak(sp, k, 0.3) or (tr[k] and abs(logs[k]) > 0.035):
                events.append((k, "cut"))
        elif L <= 15:
            ev = _classify_run(gray, sig, k, j, fps, mask_at)
            if ev is not None:
                events.append(ev)
        else:   # sustained change: motion, but a cut may sit inside it
            for q in range(k, j + 1):
                if sp[q] > 0.6 and _isolated_peak(sp, q, 0.25):
                    events.append((q, "cut"))
        used[k:j + 1] = True
        k = j + 1
    for k in _jump_cuts(sig, used | flag):
        events.append((k, "cut"))
    for ev in _dissolves(gray, mask_at, fps, used | flag, sig):
        events.append((ev["centre"], {"type": "dissolve", "dur": round((ev["frames"] + 1) / fps, 3)}))
    events.sort(key=lambda e: e[0])
    merged = []
    for e in events:          # one edit per 3 frames at most
        if merged and e[0] - merged[-1][0] < 3:
            if isinstance(e[1], dict) and not isinstance(merged[-1][1], dict):
                merged[-1] = e
            continue
        merged.append(e)
    return [e for e in merged if 0 < e[0] < n]


def _jump_cuts(sig, busy, look=15):
    """Jump cuts in footage that moves (a talking head): an isolated, modest spike of
    change located where the picture was already moving. A menu or window that pops
    up in a still screen changes cells that were static, so it is not one."""
    sp, cm = sig["spread"], sig["cmap"]
    n = len(sp)
    out = []
    for k in range(2, n - 2):
        if busy[k] or sp[k] < 0.15:
            continue
        loc = np.concatenate([sp[max(1, k - look):k - 1], sp[k + 2:k + look]])
        if not len(loc):
            continue
        med = float(np.median(loc))
        nb = max(sp[k - 1], sp[k + 1])
        if med < 0.002 or sp[k] < 4 * med or nb > 0.5 * sp[k]:
            continue
        moving = sig["energy"][max(1, k - 6):k].mean(axis=0) > 0.5
        ch = cm[k]
        if ch.sum() and (ch & moving).sum() / ch.sum() >= 0.15:
            out.append(k)
    return out


def _classify_run(gray, sig, a, b, fps, mask_at):
    sp, logs, tr = sig["spread"], sig["logs"], sig["tracked"]
    n = len(gray)
    pre, post = max(0, a - 2), min(n - 1, b + 2)
    content = [q for q in range(a, b + 1) if sp[q] > 0.45]
    gsum = sig["gx"] + sig["gy"]
    ends = (gsum[pre] + gsum[post]) / 2
    centre = a + int(np.argmin(gsum[a:b + 1]))
    blurred = gsum[centre] < 0.6 * ends
    scale_frames = [q for q in range(a, b + 1) if tr[q] and abs(logs[q]) > 0.02]
    all_tracked = all(tr[q] and sp[q] < 0.3 for q in range(a, b + 1))
    if all_tracked and not blurred:
        return None                                   # a fast camera/digital move, not an edit
    # switch frame: biggest content change (or the blur minimum)
    sw = max(content, key=lambda q: sp[q]) if content else centre
    if blurred:
        # a whip slides the picture a long way each frame; a zoom blurs in place
        dx, dy = _shift(gray, a, b)
        H, W = gray.shape[1:]
        if max(abs(dx) / W, abs(dy) / H) > 0.04:
            if abs(dx) / W >= abs(dy) / H:
                kind = "whip_left" if dx < 0 else "whip_right"
            else:
                kind = "whip_up" if dy < 0 else "whip_down"
        else:
            kind = "zoom"
        span = _extent(sig, a, b)
    elif scale_frames and content:
        kind = "zoom"
        span = _extent(sig, a, b)
    elif content:
        return (sw, "cut")
    else:
        return None
    return (sw, {"type": kind, "dur": round((span[1] - span[0] + 2) / fps, 3)})


def _extent(sig, a, b):
    """First/last frame of a transition: grow the run while frames still change,
    zoom or slide clearly."""
    n = len(sig["raw"])

    def busy(q):
        return sig["raw"][q] > 0.3 or (sig["tracked"][q] and (abs(sig["logs"][q]) > 0.01 or
                                       math.hypot(sig["tx"][q], sig["ty"][q]) > 0.02))
    lo, hi = a, b
    while lo > 1 and busy(lo - 1):
        lo -= 1
    while hi < n - 1 and busy(hi + 1):
        hi += 1
    return lo, hi


def _shift(gray, a, b):
    """Median per-frame translation (px) across a run, by phase correlation."""
    win = cv2.createHanningWindow(gray.shape[1:][::-1], cv2.CV_32F)
    ds = []
    for q in range(max(1, a), b + 1):
        (dx, dy), resp = cv2.phaseCorrelate(gray[q - 1].astype(np.float32), gray[q].astype(np.float32), win)
        ds.append((dx, dy))
    ds = np.array(ds) if ds else np.zeros((1, 2))
    return float(np.median(ds[:, 0])), float(np.median(ds[:, 1]))


# ----------------------------------------------------------------- per-shot motion

def _fit_ease(progress):
    u = np.linspace(0, 1, len(progress))
    best = None
    for name in EASE_NAMES:
        err = float(np.sqrt(np.mean((np.array([ease(name, x) for x in u]) - progress) ** 2)))
        if best is None or err < best[1]:
            best = (name, err)
    return best


def _chain(gray, a, b, mask_at, step):
    """Cumulative log-scale / pan of frames a..b-1 relative to frame a (chained
    estimates `step` frames apart: a slow move is far above per-frame noise)."""
    idx = list(range(a, b, step))
    if idx[-1] != b - 1:
        idx.append(b - 1)
    H, W = gray.shape[1:]
    logs, xs, ys, ts = [0.0], [0.0], [0.0], [idx[0]]
    jit = []
    for i0, i1 in zip(idx, idx[1:]):
        r = _similarity(gray[i0], gray[i1], mask_at(i0) & mask_at(i1), level=4)
        if r is None or r[3] < 0.5:
            logs.append(logs[-1]); xs.append(xs[-1]); ys.append(ys[-1])
        else:
            s, tx, ty, _, M, _ = r
            dl = math.log(s)
            logs.append(logs[-1] + (dl if abs(dl) > 0.0015 else 0.0))
            # translation of the frame centre (independent of the zoom's anchor)
            cx, cy = W / 2, H / 2
            mx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
            my = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
            xs.append(xs[-1] + (mx / W if abs(mx) > 0.25 else 0.0))
            ys.append(ys[-1] + (my / H if abs(my) > 0.25 else 0.0))
        ts.append(i1)
    return np.array(ts), np.array(logs), np.array(xs), np.array(ys)


def shot_motion(gray, sig, a, b, fps, mask_at, skip_in=0, skip_out=0):
    a2, b2 = min(b - 1, a + skip_in), max(a + 1, b - skip_out)
    if b2 - a2 < 3:
        return {"type": "static"}
    step = int(max(2, min(6, (b2 - a2) // 8)))
    ts, logs, xs, ys = _chain(gray, a2, b2, mask_at, step)
    # moves are measured away from transitions; scale them to the whole shot
    full = (b - a) / max(1, ts[-1] - ts[0])
    if full > 1.0:
        logs, xs, ys = logs * full, xs * full, ys * full
    zl, pan = logs[-1], math.hypot(xs[-1], ys[-1])
    move = {"type": "static"}
    if abs(zl) > 0.02 and len(logs) >= 3:
        # the moving part only: trim leading/trailing holds
        prog = logs / zl
        name, err = _fit_ease(prog)
        move = {"type": "zoom_in" if zl > 0 else "zoom_out", "scale": round(math.exp(zl), 4), "ease": name,
                "fit_rmse": round(err, 3)}
    elif pan > 0.03 and len(xs) >= 3:
        prog = (xs * xs[-1] + ys * ys[-1]) / (pan * pan)
        name, err = _fit_ease(prog)
        move = {"type": "pan", "dx": round(float(xs[-1]), 4), "dy": round(float(ys[-1]), 4), "ease": name,
                "fit_rmse": round(err, 3)}
    # shake: high-frequency residual of the per-frame centre translation
    tx = sig["tx"][a2 + 1:b2][sig["tracked"][a2 + 1:b2]]
    ty = sig["ty"][a2 + 1:b2][sig["tracked"][a2 + 1:b2]]
    if len(tx) > 8:
        hf = np.hypot(tx - np.convolve(tx, np.ones(5) / 5, "same"), ty - np.convolve(ty, np.ones(5) / 5, "same"))
        sh = float(np.median(hf[2:-2])) if len(hf) > 4 else 0.0
        if sh > 0.002:
            move["shake"] = round(sh, 4)
    return move


def _cut_kind(gray, k, mask_at):
    """At a hard cut: punch (same scene re-framed, with its scale), jump cut (same
    framing, time skipped) or a new scene."""
    H, W = gray.shape[1:]
    m = mask_at(k - 1) & mask_at(k)
    for r in (_match(gray[k - 1], gray[k], m), _similarity(gray[k - 1], gray[k], m, level=5)):
        if r is None:
            continue
        s, M = r[0], r[1] if len(r) == 3 else r[4]
        if len(r) != 3 and r[3] < 0.5:
            continue
        warped = cv2.warpAffine(gray[k - 1], M.astype(np.float64), (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        valid = _valid_cells(m) & _valid_cells(np.full_like(m, 255), M.astype(np.float64))
        if valid.sum() < 6 or _change(warped, gray[k], valid)[0] > 0.4:
            continue                       # matched features, but the pictures differ: not the same scene
        if abs(math.log(s)) > 0.03:
            return {"punch": round(s, 3)}
        return {"jump_cut": True}
    return {"new_scene": True}


def fast_moves(sig, a, b, fps, min_len=3):
    """Quick zooms / pans inside a shot (per-frame motion well above noise), with timing,
    amount and ease — e.g. a 0.3 s push-in between holds."""
    out = []
    logs, tx, ty, tr = sig["logs"], sig["tx"], sig["ty"], sig["tracked"]
    active = np.array([tr[k] and (abs(logs[k]) > 0.004 or math.hypot(tx[k], ty[k]) > 0.004) for k in range(a + 1, b)])
    k = 0
    while k < len(active):
        if not active[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(active) and (active[j + 1] or (j + 2 < len(active) and active[j + 2])):
            j += 1
        if j - k + 1 >= min_len:
            f0, f1 = a + 1 + k, a + 1 + j
            seg = range(f0, f1 + 1)
            zl = float(sum(logs[q] for q in seg))
            dx, dy = float(sum(tx[q] for q in seg)), float(sum(ty[q] for q in seg))
            if abs(zl) > 0.02:
                prog = np.concatenate([[0], np.cumsum([logs[q] for q in seg])]) / zl
                name, err = _fit_ease(prog)
                out.append({"type": "zoom_in" if zl > 0 else "zoom_out", "t0": (f0 - 1) / fps, "t1": f1 / fps,
                            "scale": round(math.exp(zl), 4), "ease": name})
            elif math.hypot(dx, dy) > 0.02:
                pan = math.hypot(dx, dy)
                cum = np.cumsum([(tx[q] * dx + ty[q] * dy) / pan for q in seg])
                name, err = _fit_ease(np.concatenate([[0], cum]) / pan)
                out.append({"type": "pan", "t0": (f0 - 1) / fps, "t1": f1 / fps, "dx": round(dx, 4),
                            "dy": round(dy, 4), "ease": name})
        k = j + 1
    return out


# ----------------------------------------------------------------- look

def _look(path, info, text_boxes):
    w = 360
    h = int(round(info["height"] * w / info["width"] / 2)) * 2
    look = grade.Look()
    corners, centres, noise, acut = [], [], [], []
    sx, sy = w / info["width"], h / info["height"]
    for i, f in enumerate(ff.frames(path, w, h, fps=2)):
        t = i / 2
        ex = np.zeros((h, w), bool)
        for tb in text_boxes or ():
            if tb["s"] - 0.1 <= t <= tb["e"] + 0.1:
                x0, y0, x1, y1 = tb["box"]
                ex[max(0, int(y0 * sy) - 4):int(y1 * sy) + 4, max(0, int(x0 * sx) - 4):int(x1 * sx) + 4] = True
        look.add(f, ex)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ch, cw = h // 8, w // 8
        corners.append(np.mean([g[:ch, :cw].mean(), g[:ch, -cw:].mean(), g[-ch:, :cw].mean(), g[-ch:, -cw:].mean()]))
        centres.append(g[h // 2 - ch // 2:h // 2 + ch // 2, w // 2 - cw // 2:w // 2 + cw // 2].mean())
        sm = cv2.GaussianBlur(g, (0, 0), 1.2)
        res = g - sm
        grad = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1))
        # noise = residual std of the flattest 8×8 blocks (texture and edges excluded)
        bh, bw = h // 8, w // 8
        rb = res[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).std(axis=(1, 3))
        gb = grad[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).mean(axis=(1, 3))
        eb = ex[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).any(axis=(1, 3))
        mb = g[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).mean(axis=(1, 3))
        ok = ~eb & (mb > 12) & (mb < 243)
        if ok.sum() > 20:
            sel = gb[ok] <= np.percentile(gb[ok], 25)
            # blocks still textured after picking the flattest quarter mean there is no flat area
            if np.median(gb[ok][sel]) < 10:
                noise.append(float(np.median(rb[ok][sel])))
            else:
                noise.append(0.0)
        acut.append(float(grad[~ex].mean()))
    ratio = float(np.median(np.array(corners) / np.maximum(1, np.array(centres)))) if corners else 1.0
    vx, vy = zip(*_VIGNETTE)
    vig = float(np.interp(-ratio, [-r for r in vy], vx)) if ratio < 0.93 else 0.0
    nstd = float(np.median(noise)) if noise else 0.0
    gx, gy = zip(*_GRAIN)
    # Gaussian-residual std underestimates uniform temporal noise by ~0.8
    grain = float(np.interp(nstd / 0.8, gy, gx)) if nstd > 0.6 else 0.0
    return {"look": look.stats(), "sharpen": 0.0, "grain": round(grain, 2), "vignette": round(vig, 2),
            "noise_std": round(nstd, 2), "corner_ratio": round(ratio, 3),
            "acutance": round(float(np.median(acut)) if acut else 0.0, 2)}


# ----------------------------------------------------------------- main entry

def analyze_video(path, info, gray=None, text_boxes=()):
    """Shots, transitions, motion, pacing (shot fields), look and loop flag of a sample.
    text_boxes: [{"s", "e", "box": [x0, y0, x1, y1]}] in px at the sample's size —
    caption / title / watermark overlays only (not text that is part of the footage)."""
    fps = info.get("fps") or 30
    pool = ThreadPoolExecutor(1)
    look = pool.submit(_look, path, info, text_boxes)          # colour pass runs alongside
    if gray is None:
        gray, _ = decode_gray(path, info)
    n, H, W = gray.shape
    mask_at = _mask_fn(text_boxes, info, (H, W), fps)
    sig = frame_signals(gray, mask_at)
    bounds = boundaries(gray, sig, mask_at, fps)
    cuts = [b for b, _ in bounds]
    kinds = dict(bounds)
    edges = [0] + cuts + [n]
    shots = []
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        shot = {"index": i, "s": round(a / fps, 3), "e": round(b / fps, 3), "dur": round((b - a) / fps, 3)}
        tin = kinds.get(a, "cut") if i else None
        tout = kinds.get(b) if b < n else None
        skip_in = int(round(tin["dur"] * fps / 2)) + 1 if isinstance(tin, dict) else 0
        skip_out = int(round(tout["dur"] * fps / 2)) + 1 if isinstance(tout, dict) else 0
        if i:
            shot["transition_in"] = tin
        shot["move"] = shot_motion(gray, sig, a, b, fps, mask_at, skip_in, skip_out)
        mv = fast_moves(sig, a + skip_in, b - skip_out, fps)
        whole = shot["move"]
        if mv and whole["type"] in ("zoom_in", "zoom_out") and \
                abs(math.log(whole["scale"]) - sum(math.log(m_["scale"]) for m_ in mv if "scale" in m_)) < 0.025:
            shot["move"] = {"type": "static"}       # the shot's zoom is just its quick moves
        if mv:
            for m_ in mv:
                m_["t0"], m_["t1"] = round(m_["t0"] - a / fps, 3), round(m_["t1"] - a / fps, 3)
            shot["moves"] = mv
        if i and tin == "cut":
            shot.update(_cut_kind(gray, a, mask_at))
        shots.append(shot)
    durs = [s["dur"] for s in shots]
    dur = n / fps
    counts = Counter(("cut" if t == "cut" else t["type"]) for _, t in bounds)
    special = None
    sp = [t for _, t in bounds if isinstance(t, dict)]
    if sp:
        top = Counter(t["type"] for t in sp).most_common(1)[0][0]
        special = {"type": top, "dur": round(float(np.median([t["dur"] for t in sp if t["type"] == top])), 3),
                   "rate": round(sum(1 for t in sp if t["type"] == top) / max(1, len(bounds)), 3)}
    punches = [s for s in shots if "punch" in s]
    levels, lvl = [1.0], 1.0
    for s in shots[1:]:
        if "punch" in s:
            lvl *= s["punch"]
        elif s.get("new_scene") or isinstance(s.get("transition_in"), dict):
            lvl = 1.0
        levels.append(round(lvl, 2))
    zooms = [s for s in shots if s["move"]["type"] in ("zoom_in", "zoom_out")]
    shakes = [s["move"].get("shake", 0.0) for s in shots]
    motion = {"punch_rate": round(len(punches) / max(1, len(cuts)), 3),
              "punch_scales": sorted(set(round(x, 2) for x in levels)),
              "push_rate": round(len(zooms) / max(1, len(shots)), 3),
              "push_scale_per_s": round(float(np.median([math.exp(abs(math.log(s["move"]["scale"])) / max(0.2, s["dur"])) - 1
                                                         for s in zooms])), 4) if zooms else 0.0,
              "push_ease": Counter(s["move"]["ease"] for s in zooms).most_common(1)[0][0] if zooms else None,
              "shake": round(float(np.median(shakes)), 4) if shakes else 0.0}
    a0, a1 = gray[0], gray[-1]
    loop = _change(a0, a1, _valid_cells(mask_at(0)) & _valid_cells(mask_at(n - 1)))[0] < 0.2
    return {
        "shots": shots,
        "motion": motion,
        "transitions": {"counts": dict(counts), "default": counts.most_common(1)[0][0] if counts else "cut",
                        "special": special},
        "pacing": {"shot_count": len(shots), "avg_shot": round(float(np.mean(durs)), 3),
                   "median_shot": round(float(np.median(durs)), 3), "min_shot": round(min(durs), 3),
                   "max_shot": round(max(durs), 3), "shot_durations": durs,
                   "cuts_per_min": round(len(cuts) / dur * 60, 2) if dur else 0.0},
        "grade": look.result(),
        "loop": bool(loop),
        "frames": n, "fps": fps,
    }


# kept for callers of the first draft
def shots(gray, fps, mask=None):
    m = (lambda k: mask) if mask is not None else (lambda k: np.full(gray.shape[1:], 255, np.uint8))
    sig = frame_signals(gray, m)
    return boundaries(gray, sig, m, fps), sig["spread"]
