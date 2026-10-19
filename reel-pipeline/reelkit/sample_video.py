"""Picture structure of a sample reel: shots, transitions, punch-ins, camera moves.

Works on every frame at 180-px width (grey). Measurements are in canvas units
(fractions of the frame or scale factors) so they transfer to any output size.
"""
import math

import cv2
import numpy as np

from . import ff
from .render import EASES, ease

AW = 180


def decode_gray(path, info):
    ah = int(round(info["height"] * AW / info["width"] / 2)) * 2
    frames = np.stack(list(ff.frames(path, AW, ah, fps=info["fps"] or 30, pix="gray")))
    return frames, ah


FEATURES = dict(maxCorners=300, qualityLevel=0.01, minDistance=6, blockSize=5)


def _similarity(a, b, mask=None):
    """Similarity transform taking frame a's points to b's: (scale, tx, ty, inlier_ratio)."""
    pts = cv2.goodFeaturesToTrack(a, mask=mask, **FEATURES)
    if pts is None or len(pts) < 12:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.reshape(-1) == 1
    if ok.sum() < 10:
        return None
    M, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                         maxIters=2000, confidence=0.995)
    if M is None:
        return None
    s = math.hypot(M[0, 0], M[1, 0])
    return s, M[0, 2], M[1, 2], float(inl.mean()), M


def _match_scale(a, b, mask=None):
    """Scale between two frames of the same scene across a cut (ORB + RANSAC)."""
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, mask)
    kb, db = orb.detectAndCompute(b, mask)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 20:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl.sum() < 18 or inl.mean() < 0.35:
        return None
    s = math.hypot(M[0, 0], M[1, 0])
    return s, M


def _fit_ease(progress):
    """Best-fitting named ease for a 0→1 progress curve."""
    u = np.linspace(0, 1, len(progress))
    best = None
    for name in ["linear", "inout", "in", "out", "inout_cubic", "in_cubic", "out_cubic"]:
        err = float(np.sqrt(np.mean((np.array([ease(name, x) for x in u]) - progress) ** 2)))
        if best is None or err < best[1]:
            best = (name, err)
    return best


def _aligned_error(a, b, M):
    """Mean |b - a warped by M| over the area both cover (prediction error)."""
    H, W = a.shape
    pred = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(np.full_like(a, 255), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    if valid.sum() < 0.3 * H * W:
        return float(np.abs(a.astype(np.float32) - b).mean())
    return float(np.abs(pred.astype(np.float32) - b)[valid].mean())


def _anisotropy(f):
    gx = np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0)).mean()
    gy = np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1)).mean()
    return gx / (gy + 1e-6)


def shots(gray, fps, mask=None):
    """Shot boundaries with their transition, from how well each frame is predicted
    by the previous one moved/zoomed (natural motion predicts well; cuts,
    punch-ins and transitions do not). Returns [(frame, transition), ...]."""
    n, H, W = gray.shape
    if mask is None:
        mask = np.full((H, W), 255, np.uint8)
    e, ls, q, luma = np.zeros(n), np.zeros(n), np.ones(n), gray.reshape(n, -1).mean(1)
    for k in range(1, n):
        r = _similarity(gray[k - 1], gray[k], mask)
        if r is None:
            e[k] = float(np.abs(gray[k - 1].astype(np.float32) - gray[k]).mean())
            q[k] = 0.0
            continue
        ls[k], q[k] = math.log(r[0]), r[3]
        e[k] = _aligned_error(gray[k - 1], gray[k], r[4])
    base = np.array([np.median(np.concatenate([e[max(1, k - 15):k], e[k + 1:k + 16]]) if n > 2 else [0])
                     for k in range(n)])
    flag = np.zeros(n, bool)
    for k in range(1, n):
        flag[k] = (e[k] > max(3.0, 4.0 * base[k] + 1.0)) or abs(ls[k]) > 0.04 or (q[k] < 0.3 and e[k] > 2.0 * base[k] + 2.0)
    # flashes: a few frames much brighter than both neighbours
    for k in range(1, n - 1):
        for w in (1, 2, 3):
            if k + w < n and luma[k:k + w].min() > max(luma[k - 1], luma[k + w]) + 40:
                flag[k:k + w + 1] = True
    runs, k = [], 1
    while k < n:
        if flag[k]:
            j = k
            while j + 1 < n and (flag[j + 1] or (j + 2 < n and flag[j + 2] and j + 2 - k < 20)):
                j += 1
            runs.append((k, j))
            k = j + 1
        else:
            k += 1
    out = []
    for a, b in runs:
        L = b - a + 1
        pre, post = gray[a - 1].astype(np.float32), gray[min(n - 1, b + 1)].astype(np.float32)
        if L == 1:
            out.append((a, "cut"))
            continue
        mids = gray[a:b + 1].astype(np.float32)
        if luma[a:b + 1].max() > max(luma[a - 1], luma[min(n - 1, b + 1)]) + 40:
            peak = a + int(np.argmax(luma[a:b + 1]))
            out.append((peak, {"type": "flash", "dur": round(L / fps, 3)}))
            continue
        m = mids[L // 2]
        blend = np.abs(m - 0.5 * (pre + post)).mean()
        ends = min(np.abs(m - pre).mean(), np.abs(m - post).mean())
        if blend < 0.6 * ends and np.abs(pre - post).mean() > 6:
            out.append((a + L // 2, {"type": "dissolve", "dur": round((L + 1) / fps, 3)}))
            continue
        an = np.mean([_anisotropy(f) for f in mids])
        an0 = (_anisotropy(pre) + _anisotropy(post)) / 2
        if an > 1.6 * an0 or an < an0 / 1.6:
            horiz = an > an0
            out.append((a + L // 2, {"type": "whip_left" if horiz else "whip_up", "dur": round((L + 1) / fps, 3)}))
            continue
        if np.abs(ls[a:b + 1]).max() > 0.04 and L <= 15:
            out.append((a + L // 2, {"type": "zoom", "dur": round((L + 1) / fps, 3)}))
            continue
        out.append((a + L // 2, "cut"))
    return out, e


def _chain_motion(gray, a, b, mask, step=6, dead=0.002):
    """Zoom (log scale) and pan curves over a shot, measured between frames `step`
    apart (a slow zoom is far above per-frame noise over 0.2 s), chained."""
    idx = list(range(a, b, step)) + ([b - 1] if (b - 1 - a) % step else [])
    logs, xs, ys = [0.0], [0.0], [0.0]
    H, W = gray.shape[1:]
    for i0, i1 in zip(idx, idx[1:]):
        r = _similarity(gray[i0], gray[i1], mask)
        if r is None or r[3] < 0.5:
            logs.append(logs[-1]); xs.append(xs[-1]); ys.append(ys[-1])
            continue
        dl = math.log(r[0])
        logs.append(logs[-1] + (dl if abs(dl) > dead else 0.0))
        xs.append(xs[-1] + (r[1] / W if abs(r[1]) > 0.3 else 0.0))
        ys.append(ys[-1] + (r[2] / H if abs(r[2]) > 0.3 else 0.0))
    return idx, logs, xs, ys


def motion(gray, bounds_list, fps, mask=None):
    """Per shot: zoom / pan fitted to an ease; punch-in scale or jump cut at its start."""
    n, H, W = gray.shape
    if mask is None:
        mask = np.full((H, W), 255, np.uint8)
    border = mask.copy()
    border[:int(H * 0.03)] = 0
    border[-int(H * 0.03):] = 0
    cuts = [c for c, _ in bounds_list]
    kinds = dict(bounds_list)
    bounds = [0] + cuts + [n]
    out = []
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        dur = (b - a) / fps
        shot = {"index": i, "s": round(a / fps, 3), "e": round(b / fps, 3), "dur": round(dur, 3)}
        if i > 0:
            shot["transition_in"] = kinds[a]
        trim = 3 if isinstance(kinds.get(a), dict) else 0       # skip transition frames
        idx, logs, xs, ys = _chain_motion(gray, min(b - 1, a + trim), b, border)
        zl, pan = logs[-1], math.hypot(xs[-1], ys[-1])
        if abs(zl) > 0.02 and len(logs) >= 3:
            prog = np.array(logs) / zl
            name, err = _fit_ease(prog)
            shot["move"] = {"type": "zoom_in" if zl > 0 else "zoom_out", "scale": round(math.exp(zl), 4),
                            "ease": name, "fit_rmse": round(err, 3)}
        elif pan > 0.03 and len(xs) >= 3:
            prog = np.hypot(xs, ys) / max(pan, 1e-6)
            name, err = _fit_ease(prog)
            shot["move"] = {"type": "pan", "dx": round(xs[-1], 4), "dy": round(ys[-1], 4), "ease": name}
        else:
            shot["move"] = {"type": "static"}
        if i > 0 and kinds[a] == "cut":
            m = _match_scale(gray[a - 1], gray[a], border)
            if m is not None and abs(math.log(m[0])) > 0.04:
                shot["punch"] = round(m[0], 3)       # scale relative to the previous shot (same scene)
            elif m is not None:
                shot["jump_cut"] = True              # same framing, time skipped
        out.append(shot)
    return out
