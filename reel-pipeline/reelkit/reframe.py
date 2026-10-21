"""Subject-tracked reframing: keep the speaker in a 9:16 crop of 16:9 footage
(or centred under a punch-in on vertical footage).

detect()      YuNet faces at ~5 fps on an editing copy (detector run at 640 px wide)
subject()     one face per sample: start on the largest, most central face, then follow
              that identity (nearest box, similar size); switch only if another face is
              clearly dominant for a sustained stretch
path()        camera centre over time: a dead zone (the frame holds while the subject
              moves a little), then a critically damped spring (no overshoot, no jitter)
              towards the subject; a cut resets the camera onto the subject
keyframes()   the path as x/y offset keyframes (timeline px) for one clip, clamped so
              the scaled picture always covers the canvas; collinear keys dropped.
No face → the centre of the frame (and the clip is flagged "subject": None).
"""
import math

import cv2
import numpy as np

from . import ff
from .paths import MODELS

YUNET = MODELS / "yunet" / "yunet.onnx"
DET_W = 640
_det = {}


def _detector(w, h):
    key = (w, h)
    if key not in _det:
        _det[key] = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.7, 0.3, 50)
    return _det[key]


def detect_image(img, min_rel=0.03):
    """Faces in one BGR image: [(cx, cy, w, h, score)] normalised to 0..1."""
    h, w = img.shape[:2]
    _, faces = _detector(w, h).detect(img)
    out = []
    for f in faces if faces is not None else []:
        # YuNet: box (4), five x/y landmarks (10), then confidence.
        # f[4] is the first eye's x coordinate, not a probability.
        x, y, fw, fh = f[:4]
        score = f[14]
        if fw / w < min_rel:
            continue
        out.append((float((x + fw / 2) / w), float((y + fh / 2) / h), float(fw / w), float(fh / h), float(score)))
    return out


def detect(path, start, dur, fps=5.0, info=None):
    """[(t_rel, faces)] over [start, start+dur) of a video, t_rel from `start`."""
    info = info or ff.probe(path)
    w = min(DET_W, info["width"])
    h = int(round(info["height"] * w / info["width"] / 2)) * 2
    out = []
    for i, f in enumerate(ff.frames(path, w, h, fps=fps, start=start, duration=dur)):
        out.append((round(i / fps, 3), detect_image(f)))
    return out


def subject(dets, switch_after=1.5, dominance=1.8):
    """One (cx, cy, w, h) or None per sample, following a stable identity."""
    def score(f):
        cx, cy, w, h, s = f
        return w * h * (1.2 - abs(cx - 0.5)) * s
    track, cur, rival_since = [], None, None
    for t, faces in dets:
        if not faces:
            track.append((t, None))
            continue
        if cur is None:
            cur = max(faces, key=score)
        else:
            same = min(faces, key=lambda f: math.hypot(f[0] - cur[0], f[1] - cur[1]) + abs(math.log(f[2] / cur[2])))
            near = math.hypot(same[0] - cur[0], same[1] - cur[1]) < max(0.25, 2 * cur[2]) and \
                0.5 < same[2] / cur[2] < 2.0
            best = max(faces, key=score)
            if near:
                cur = same
            if best is not same and score(best) > dominance * score(same if near else cur):
                rival_since = rival_since if rival_since is not None else t
                if t - rival_since >= switch_after or not near:
                    cur, rival_since = best, None
            else:
                rival_since = None
        track.append((t, cur[:4]))
    return track


def _fill(track):
    """Hold the last known position through detection gaps (first known back-filled)."""
    known = [p for _, p in track if p is not None]
    if not known:
        return None
    first, out, last = known[0], [], None
    for t, p in track:
        last = p if p is not None else last
        out.append((t, (last or first)[:2]))
    return out


def path(track, fps=30.0, dead_zone=0.06, omega=5.0, dur=None):
    """Camera centre (cx, cy) in 0..1 source coordinates at every output frame.
    dead_zone is a fraction of the source width/height."""
    filled = _fill(track)
    if filled is None:
        return None
    ts = np.array([t for t, _ in filled])
    xs = np.array([p[0] for _, p in filled])
    ys = np.array([p[1] for _, p in filled])
    n = int(round((dur if dur is not None else ts[-1] + 1e-3) * fps)) + 1
    out = []
    pos = np.array([xs[0], ys[0]])
    vel = np.zeros(2)
    dt = 1.0 / fps
    goal = pos.copy()
    for k in range(n):
        t = k / fps
        target = np.array([np.interp(t, ts, xs), np.interp(t, ts, ys)])
        off = target - goal
        for i in range(2):                       # the goal follows only beyond the dead zone
            if abs(off[i]) > dead_zone:
                goal[i] = target[i] - math.copysign(dead_zone, off[i])
        acc = omega * omega * (goal - pos) - 2 * omega * vel   # critically damped spring
        vel += acc * dt
        pos = pos + vel * dt
        out.append((round(t, 4), float(pos[0]), float(pos[1])))
    return out


def _rdp(points, eps):
    """Ramer–Douglas–Peucker on (t, x, y): keep the keys where linear interpolation
    between kept keys would stray more than eps px (iterative)."""
    if len(points) < 3:
        return list(points)
    P = np.array(points, np.float64)
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = P[i], P[j]
        u = (P[i + 1:j, 0] - a[0]) / (b[0] - a[0]) if b[0] != a[0] else np.zeros(j - i - 1)
        q = a[1:] + (b[1:] - a[1:]) * u[:, None]
        d = np.abs(P[i + 1:j, 1:] - q).max(axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return [tuple(p) for p in P[keep]]


def offsets(cam, src_w, src_h, canvas, scale_at, fit="cover", anchor=(0.5, 0.42), eps=0.4):
    """x/y keyframes (clip-relative, px) placing the camera centre at the canvas
    anchor, clamped so the picture covers the canvas. scale_at(t) is the clip's scale."""
    W, H = canvas
    f = (max if fit == "cover" else min)(W / src_w, H / src_h)
    pts = []
    for t, cx, cy in cam:
        s = scale_at(t)
        pw, ph = src_w * f * s, src_h * f * s
        x = (anchor[0] - 0.5) * W - (cx - 0.5) * pw
        y = (anchor[1] - 0.5) * H - (cy - 0.5) * ph
        mx, my = max(0.0, (pw - W) / 2), max(0.0, (ph - H) / 2)
        pts.append((t, float(np.clip(x, -mx, mx)), float(np.clip(y, -my, my))))
    thin = pts[::3] + ([pts[-1]] if (len(pts) - 1) % 3 else [])   # 10 key candidates/s at 30 fps
    keys = _rdp(thin, eps)
    return [[round(t, 3), round(x, 2)] for t, x, _ in keys], [[round(t, 3), round(y, 2)] for t, _, y in keys]


def clip_keyframes(path_, start, dur, canvas, scale_at, info=None, fps=30.0, det_fps=5.0, horizontal_only=None):
    """Detect, track and convert for one clip of an editing copy.
    Returns {"x": keys|0, "y": keys|0, "subject": bool}."""
    info = info or ff.probe(path_)
    dets = detect(path_, start, dur, det_fps, info)
    track = subject(dets)
    cam = path(track, fps, dur=dur)
    W, H = canvas
    landscape_to_vertical = info["width"] / info["height"] > W / H + 0.05
    if horizontal_only is None:
        horizontal_only = landscape_to_vertical
    if cam is None:
        return {"x": 0, "y": 0, "subject": False}
    if horizontal_only:
        cam = [(t, cx, 0.5) for t, cx, _ in cam]
        xk, _ = offsets(cam, info["width"], info["height"], canvas, scale_at, anchor=(0.5, 0.5))
        return {"x": xk, "y": 0, "subject": True}
    xk, yk = offsets(cam, info["width"], info["height"], canvas, scale_at)
    return {"x": xk, "y": yk, "subject": True}

