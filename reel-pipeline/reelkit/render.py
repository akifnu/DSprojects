"""Timeline → finished MP4.

Video is composited frame by frame in OpenCV — sub-pixel affine transforms (smooth
zooms and pans, no integer-crop jitter), transitions, image overlays, animated
captions — and piped as raw BGR to ffmpeg, which applies the grade (3D LUT,
sharpening, grain, vignette) and encodes. Audio is mixed in a separate ffmpeg graph
at the same time: voice / music / sfx buses, micro-fades at every edit, voice chain,
music ducked under the voice, two-pass loudness normalisation. Then the two are muxed.

Timeline (JSON) — times in seconds, positions in px at canvas size, (x, y) = offset
of the picture centre from the canvas centre:

  {"width": 720, "height": 1280, "fps": 30, "duration": 28.2, "background": "#000000",
   "media": {"m1": "/abs/or/relative/path.mp4" | "asset:<id>"},
   "video": [{"media": "m1", "start": 0, "in": 12.3, "dur": 2.1, "fit": "cover",
              "scale": 1.0 | [[t, v, ease], ...], "x": 0, "y": 0, "rotation": 0, "opacity": 1,
              "speed": 1.0, "track": 1, "transition_in": {"type": "dissolve", "dur": 0.3}}],
   "captions": [{"s": 0.2, "e": 1.4, "text": "...", "words": [{"w","s","e"}], "style": "caption"}],
   "styles": {"caption": {...captions.py style...}},
   "grade": {"lut": "x.cube", "sharpen": 0.3, "grain": 4, "vignette": 0.2},
   "audio": {"clips": [{"media": "m1", "start": 0, "in": 12.3, "dur": 2.1, "role": "voice",
                        "gain_db": 0, "fade_in": 0.008, "fade_out": 0.008, "mono": true}],
             "voice_chain": true, "duck_db": 12, "loudness": -14, "true_peak": -1},
             (music is keyframe-ducked by duck_db wherever voice clips play)
   "encode": {"preset": "medium", "crf": 17}}

Keyframes: [t, value] or [t, value, ease] with t relative to the clip's start; ease is
the curve INTO that key: linear, in, out, inout, in_cubic, out_cubic, inout_cubic,
out_back, hold, or [x1, y1, x2, y2] (CSS cubic-bezier).
Transitions (transition_in, centred on the cut): cut, dissolve, dip_black, dip_white,
flash, zoom, whip_left, whip_right, whip_up, whip_down. Video tracks > 1 are overlays.
"""
import json
import math
import os
import queue
import subprocess
import threading
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from . import captions, ff
from .paths import FFMPEG

# ---------------------------------------------------------------- keyframes

EASES = {"in": (0.42, 0, 1, 1), "out": (0, 0, 0.58, 1), "inout": (0.42, 0, 0.58, 1),
         "in_cubic": (0.32, 0, 0.67, 0), "out_cubic": (0.33, 1, 0.68, 1), "inout_cubic": (0.65, 0, 0.35, 1),
         "out_back": (0.34, 1.56, 0.64, 1), "in_quart": (0.5, 0, 0.75, 0), "out_quart": (0.25, 1, 0.5, 1)}


def bezier(x1, y1, x2, y2, u):
    """CSS cubic-bezier y at x=u (Newton, then bisection fallback)."""
    if u <= 0 or u >= 1:
        return min(1.0, max(0.0, u))
    def bx(s):
        return 3 * (1 - s) ** 2 * s * x1 + 3 * (1 - s) * s ** 2 * x2 + s ** 3
    def by(s):
        return 3 * (1 - s) ** 2 * s * y1 + 3 * (1 - s) * s ** 2 * y2 + s ** 3
    s = u
    for _ in range(8):
        d = 3 * (1 - s) ** 2 * x1 + 6 * (1 - s) * s * (x2 - x1) + 3 * s ** 2 * (1 - x2)
        if abs(d) < 1e-6:
            break
        s -= (bx(s) - u) / d
        s = min(1.0, max(0.0, s))
    if abs(bx(s) - u) > 1e-4:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            s = (lo + hi) / 2
            lo, hi = (s, hi) if bx(s) < u else (lo, s)
    return by(s)


def ease(name, u):
    if name in (None, "linear"):
        return u
    if name == "hold":
        return 0.0 if u < 1 else 1.0
    return bezier(*(EASES[name] if isinstance(name, str) else name), u)


def kf(spec, t, default=0.0):
    """Value of a number-or-keyframes spec at time t (clip-relative)."""
    if spec is None:
        return default
    if isinstance(spec, (int, float)):
        return float(spec)
    keys = [k if isinstance(k, (list, tuple)) else (k["t"], k["v"], k.get("ease")) for k in spec]
    if t <= keys[0][0]:
        return float(keys[0][1])
    for a, b in zip(keys, keys[1:]):
        if t < b[0]:
            u = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 1.0
            return a[1] + (b[1] - a[1]) * ease(b[2] if len(b) > 2 else None, u)
    return float(keys[-1][1])


def kf_max(spec, dur, default=1.0, samples=48):
    if spec is None or isinstance(spec, (int, float)):
        return kf(spec, 0, default)
    return max(kf(spec, dur * i / samples, default) for i in range(samples + 1))


# ---------------------------------------------------------------- sources

class Reader:
    """Sequential frames of one clip window, decoded by ffmpeg at a working size
    chosen so the per-frame warp rarely has to shrink the picture by much."""

    def __init__(self, path, info, clip, canvas, fps, pre, post, from_t=None):
        W, H = canvas
        sw, sh = info["width"], info["height"]
        fit = clip.get("fit", "cover")
        if fit == "stretch":
            self.fx, self.fy = W / sw, H / sh
        else:
            f = (max if fit == "cover" else min)(W / sw, H / sh)
            self.fx = self.fy = f
        smax = kf_max(clip.get("scale"), clip["dur"], 1.0) * (1.6 if clip.get("_zoom_transition") else 1.0)
        self.work = min(self.fx * smax, max(self.fx, 1.0))          # px of working image per px of source
        self.ww = max(2, int(round(sw * self.work / 2)) * 2)
        self.wh = max(2, int(round(sh * self.work * (self.fy / self.fx) / 2)) * 2)
        self.speed = clip.get("speed", 1.0)
        self.first = clip["start"] - pre                             # timeline time of frame 0
        if from_t is not None and from_t > self.first:                # a still: start decoding near t
            pre -= from_t - self.first
            self.first = from_t
        src_start = clip.get("in", 0.0) - pre * self.speed
        self.hold_head = max(0, int(round(-src_start / self.speed * fps))) if src_start < 0 else 0
        vf = [f"setpts=PTS/{self.speed}"] if self.speed != 1 else []
        cs = info.get("colorspace")
        if cs == "bt709" or (cs in (None, "", "unknown") and min(sw, sh) >= 720):
            vf.append("scale=in_color_matrix=bt709")   # HD without a tag is BT.709, not ffmpeg's BT.601 default
        self.gen = ff.frames(path, self.ww, self.wh, fps=fps, start=max(0.0, src_start),
                             duration=(pre + clip["dur"] + post) * self.speed + 1.0, vf=",".join(vf) or None)
        self.idx = -1
        self.cur = None
        self.fps = fps
        # decode ahead on a thread: the OS pipe holds only 64 KB, so without this
        # ffmpeg would wait for the compositor and the compositor for ffmpeg
        self.q = queue.Queue(maxsize=6)
        self.stop = False
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self):
        try:
            for fr in self.gen:
                while not self.stop:
                    try:
                        self.q.put(fr, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self.stop:
                    return
        finally:
            self.q.put(None)

    def _next(self):
        return None if self.stop else self.q.get()

    def at(self, t):
        k = int(round((t - self.first) * self.fps)) - self.hold_head
        while self.idx < k:
            nxt = self._next()
            if nxt is None:
                self.stop = True
                break                       # source ran out: hold the last frame
            self.cur, self.idx = nxt, self.idx + 1
        if self.cur is None and not self.stop:
            self.cur = self._next()
            self.idx = 0
        return self.cur

    def close(self):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.thread.join(timeout=2)
        self.gen.close()


def _resolve(ref, base):
    ref = str(ref)
    if ref.startswith("asset:"):
        from . import store
        return str(store.get(ref[6:]).path)
    p = Path(ref)
    return str(p if p.is_absolute() else (Path(base) / p).resolve())


# ---------------------------------------------------------------- compositor

def _matrix(r, clip, t_rel, W, H, extra_scale=1.0, dx=0.0, dy=0.0):
    s = kf(clip.get("scale"), t_rel, 1.0) * extra_scale
    x = kf(clip.get("x"), t_rel, 0.0) + dx
    y = kf(clip.get("y"), t_rel, 0.0) + dy
    rot = math.radians(kf(clip.get("rotation"), t_rel, 0.0))
    kx, ky = r.fx * s / r.work, r.fy * s / r.work
    c, si = math.cos(rot), math.sin(rot)
    a, b, d, e = kx * c, -ky * si, kx * si, ky * c
    cx, cy = (r.ww - 1) / 2, (r.wh - 1) / 2
    return np.array([[a, b, (W - 1) / 2 + x - (a * cx + b * cy)], [d, e, (H - 1) / 2 + y - (d * cx + e * cy)]],
                    np.float64), min(kx, ky)


def _warp(img, M, k, W, H, dst=None, border=cv2.BORDER_TRANSPARENT):
    if dst is not None and abs(M[0, 0] - 1) < 1e-9 and abs(M[1, 1] - 1) < 1e-9 and M[0, 1] == 0 and M[1, 0] == 0 \
            and abs(M[0, 2] - round(M[0, 2])) < 1e-6 and abs(M[1, 2] - round(M[1, 2])) < 1e-6:
        ox, oy = int(round(M[0, 2])), int(round(M[1, 2]))        # whole-pixel move: plain copy
        h, w = img.shape[:2]
        xa, ya, xb, yb = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
        if xa < xb and ya < yb:
            dst[ya:yb, xa:xb] = img[ya - oy:yb - oy, xa - ox:xb - ox]
        return dst
    if k < 0.85:   # shrink with area averaging first: warpAffine alone would alias
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        M = M.copy()
        M[:, :2] /= k
    flags = cv2.INTER_CUBIC if k >= 1.0 else cv2.INTER_LINEAR
    if dst is None:
        return cv2.warpAffine(img, M, (W, H), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.warpAffine(img, M, (W, H), dst=dst, flags=flags, borderMode=border)
    return dst


def _motion_blur(img, dx, dy):
    """Directional (whip) blur: a box filter along the motion axis."""
    n = int(min(96, max(abs(dx), abs(dy))))
    if n < 3:
        return img
    return cv2.blur(img, (n, 1) if abs(dx) >= abs(dy) else (1, n))


class Compositor:
    def __init__(self, tl, base_dir):
        self.tl = tl
        self.W, self.H, self.fps = tl["width"], tl["height"], tl["fps"]
        self.N = int(round(tl["duration"] * self.fps))
        self.bg = np.array(captions.rgba(tl.get("background", "#000000"))[:3][::-1], np.uint8)
        self.from_t = None
        self.media = {k: _resolve(v, base_dir) for k, v in tl.get("media", {}).items()}
        self.info = {}
        clips = sorted(tl.get("video", []), key=lambda c: (c.get("track", 1), c["start"]))
        self.base = [c for c in clips if c.get("track", 1) == 1]
        self.over = [c for c in clips if c.get("track", 1) > 1]
        for i, c in enumerate(self.base):
            tr = c.get("transition_in") or {}
            c["_tin"] = tr.get("dur", 0.0) / 2 if tr.get("type", "cut") != "cut" else 0.0
            c["_zoom_transition"] = tr.get("type") == "zoom" or (
                i + 1 < len(self.base) and (self.base[i + 1].get("transition_in") or {}).get("type") == "zoom")
        for i, c in enumerate(self.base):
            c["_tout"] = self.base[i + 1]["_tin"] if i + 1 < len(self.base) else 0.0
        for c in self.over:
            c["_tin"] = c["_tout"] = 0.0
        styles = tl.get("styles", {})
        self.text = [captions.TextEvent(ev, ev["style"] if isinstance(ev.get("style"), dict)
                                        else styles.get(ev.get("style", "caption"), {}), (self.W, self.H))
                     for ev in tl.get("captions", []) + tl.get("text", [])]
        self.images = {}
        self.readers = {}

    def _info(self, path):
        if path not in self.info:
            self.info[path] = ff.probe(path)
        return self.info[path]

    def _reader(self, c):
        key = id(c)
        if key not in self.readers:
            path = self.media[c["media"]]
            self.readers[key] = Reader(path, self._info(path), c, (self.W, self.H), self.fps, c["_tin"], c["_tout"],
                                       from_t=self.from_t)
        return self.readers[key]

    def _retire(self, t):
        for c in self.base + self.over:
            if id(c) in self.readers and t > c["start"] + c["dur"] + c["_tout"] + 1.0 / self.fps:
                self.readers.pop(id(c)).close()

    def _image(self, c):
        path = self.media[c["media"]]
        if path not in self.images:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            self.images[path] = img
        return self.images[path]

    def _draw(self, canvas, c, t, alpha=1.0, extra_scale=1.0, dx=0.0, dy=0.0, blur=(0, 0)):
        t_rel = t - c["start"]
        alpha *= kf(c.get("opacity"), t_rel, 1.0)
        if c.get("kind") == "image" or self.media[c["media"]].lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            img = self._image(c)
            fit = c.get("fit", "none")
            f = 1.0 if fit == "none" else (max if fit == "cover" else min)(self.W / img.shape[1], self.H / img.shape[0])
            r = SimpleNamespace(fx=f, fy=f, work=1.0, ww=img.shape[1], wh=img.shape[0])  # an image: a one-frame reader
            M, k = _matrix(r, c, t_rel, self.W, self.H, extra_scale, dx, dy)
            h, w = img.shape[:2]
            pts = M @ np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]], np.float64)
            x0, y0 = max(0, int(pts[0].min()) - 2), max(0, int(pts[1].min()) - 2)
            x1, y1 = min(self.W, int(pts[0].max()) + 3), min(self.H, int(pts[1].max()) + 3)
            if x0 >= x1 or y0 >= y1:
                return
            M2 = M.copy()
            M2[:, 2] -= (x0, y0)                                   # warp only the sticker's box
            warped = _warp(img, M2, k, x1 - x0, y1 - y0)
            a = warped[..., 3:4].astype(np.float32) / 255.0 * alpha
            region = canvas[y0:y1, x0:x1]
            region[:] = np.clip(region * (1 - a) + warped[..., :3] * a, 0, 255).astype(np.uint8)
            return
        r = self._reader(c)
        frame = r.at(t)
        if frame is None:
            return
        if blur != (0, 0):
            frame = _motion_blur(frame, *blur)
        M, k = _matrix(r, c, t_rel, self.W, self.H, extra_scale, dx, dy)
        if alpha >= 0.999:
            _warp(frame, M, k, self.W, self.H, dst=canvas)
        else:
            layer = canvas.copy()
            _warp(frame, M, k, self.W, self.H, dst=layer)
            cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0, dst=canvas)

    def _base(self, canvas, t):
        for i, c in enumerate(self.base):
            if not (c["start"] - c["_tin"] <= t < c["start"] + c["dur"] + c["_tout"]):
                continue
            tr = c.get("transition_in") or {}
            in_window = c["_tin"] and t < c["start"] + c["_tin"]
            prev = self.base[i - 1] if i else None
            if in_window and prev is not None:
                self._transition(canvas, prev, c, tr, (t - (c["start"] - c["_tin"])) / (2 * c["_tin"]), t)
                return
            nxt = self.base[i + 1] if i + 1 < len(self.base) else None
            if nxt is not None and nxt["_tin"] and t >= nxt["start"] - nxt["_tin"]:
                continue            # the next clip's transition window draws this frame
            if in_window and prev is None:           # transition into the first clip: from the background
                self._draw(canvas, c, t, alpha=ease("inout", (t - (c["start"] - c["_tin"])) / (2 * c["_tin"])))
                return
            if c["start"] <= t < c["start"] + c["dur"]:
                self._draw(canvas, c, t)
                return

    def _transition(self, canvas, a, b, tr, u, t):
        kind = tr.get("type", "dissolve")
        W, H = self.W, self.H
        if kind == "dissolve":
            self._draw(canvas, a, t)
            self._draw(canvas, b, t, alpha=ease("inout", u))
        elif kind in ("dip_black", "dip_white", "flash"):
            col = (255, 255, 255) if kind != "dip_black" else (0, 0, 0)
            (self._draw(canvas, a, t) if u < 0.5 else self._draw(canvas, b, t))
            v = 1 - abs(2 * u - 1)
            if kind == "flash":
                v = v ** 2
            overlay = np.empty_like(canvas)
            overlay[:] = col
            cv2.addWeighted(overlay, v, canvas, 1 - v, 0, dst=canvas)
        elif kind == "zoom":
            if u < 0.5:
                self._draw(canvas, a, t, extra_scale=1 + 0.5 * ease("in_cubic", 2 * u))
            else:
                self._draw(canvas, b, t, extra_scale=1 + 0.35 * (1 - ease("out_cubic", 2 * u - 1)))
        elif kind.startswith("whip"):
            d = {"whip_left": (-1, 0), "whip_right": (1, 0), "whip_up": (0, -1), "whip_down": (0, 1)}[kind]
            e = ease("inout_cubic", u)
            vel = 3 * (1 - abs(2 * u - 1)) * (W if d[0] else H) / max(1, tr.get("dur", 0.3) * self.fps)
            self._draw(canvas, a, t, dx=d[0] * W * e, dy=d[1] * H * e, blur=(d[0] * vel, d[1] * vel))
            self._draw(canvas, b, t, dx=-d[0] * W * (1 - e), dy=-d[1] * H * (1 - e), blur=(d[0] * vel, d[1] * vel))
        else:
            (self._draw(canvas, a, t) if u < 0.5 else self._draw(canvas, b, t))

    def frame(self, f):
        t = f / self.fps
        canvas = np.empty((self.H, self.W, 3), np.uint8)
        canvas[:] = self.bg
        self._base(canvas, t)
        for c in self.over:
            if c["start"] <= t < c["start"] + c["dur"]:
                self._draw(canvas, c, t)
        for ev in self.text:
            p = ev.patch(t)
            if p is not None:
                captions.blend(canvas, *p)
        self._retire(t)
        return canvas

    def close(self):
        for r in self.readers.values():
            r.close()
        self.readers.clear()


# ---------------------------------------------------------------- encode / audio / mux

def write_cube(lut, path):
    n = lut.shape[0]
    with open(path, "w") as f:
        f.write(f"LUT_3D_SIZE {n}\n")
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    f.write("%.6f %.6f %.6f\n" % tuple(lut[r, g, b]))


def grade_filters(g, base_dir):
    out = []
    if g.get("lut"):
        out.append(f"lut3d=file='{_resolve(g['lut'], base_dir)}':interp=tetrahedral")
    if g.get("sharpen"):
        out.append(f"unsharp=5:5:{g['sharpen']}:5:5:0")
    if g.get("grain"):
        out.append(f"noise=c0s={g['grain']}:c0f=t+u")
    if g.get("vignette"):
        out.append(f"vignette=angle={math.pi / 5 * g['vignette']:.4f}")
    return out


def _encoder_args(enc, draft, encoder):
    if encoder == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5" if not draft else "p2", "-rc", "vbr", "-cq",
                str(enc.get("cq", 19)), "-b:v", "0"]
    preset = "veryfast" if draft else enc.get("preset", "medium")
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(enc.get("crf", 17)), "-profile:v", "high"]


def render_video(tl, base_dir, out, draft=False, encoder="cpu", threads=0):
    comp = Compositor(tl, base_dir)
    W, H, fps = comp.W, comp.H, comp.fps
    vf = grade_filters(tl.get("grade") or {}, base_dir) + [
        "scale=out_color_matrix=bt709:out_range=tv", "format=yuv420p"]
    enc = tl.get("encode") or {}
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-vf", ",".join(vf),
           *_encoder_args(enc, draft, encoder), "-g", str(int(round(fps))), "-bf", "2",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-maxrate", "25M", "-bufsize", "50M", "-an", str(out)]
    if threads:
        cmd[1:1] = ["-threads", str(threads)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    started = time.monotonic()
    out_q = queue.Queue(maxsize=8)

    def writer():
        while True:
            buf = out_q.get()
            if buf is None:
                return
            proc.stdin.write(buf)

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()
    try:
        for f in range(comp.N):
            out_q.put(comp.frame(f).tobytes())
    finally:
        out_q.put(None)
        wt.join()
        comp.close()
        proc.stdin.close()
        rc = proc.wait()
    if rc:
        raise RuntimeError(f"video encode failed ({rc})")
    ff.log(f"video: {comp.N} frames in {time.monotonic() - started:.1f}s")


def _duck_expr(spans, depth_db, ramp=0.15):
    """volume expression: -depth_db while the voice speaks (spans merged across gaps
    < 0.4 s), with `ramp`-second ramps — keyframed ducking, exact and predictable."""
    merged = []
    for a, b in sorted(spans):
        if merged and a - merged[-1][1] < 0.4:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if not merged:
        return None
    env = "0"
    for a, b in merged:
        term = f"clip((t-{a - ramp:.3f})/{ramp},0,1)*clip(({b + ramp:.3f}-t)/{ramp},0,1)"
        env = term if env == "0" else f"max({env},{term})"
    return f"volume='pow(10,-{depth_db}*({env})/20)':eval=frame"


def render_audio(tl, base_dir, out, tmp):
    a = tl.get("audio") or {}
    clips = [c for c in a.get("clips", []) if c.get("gain_db", 0) > -60]
    dur = tl["duration"]
    if not clips:
        return None
    inputs, graph, buses = [], [], {"voice": [], "music": [], "sfx": []}
    probe_cache, voice_spans = {}, []
    for n, c in enumerate(clips):
        path = _resolve(tl["media"][c["media"]], base_dir)
        info = probe_cache.setdefault(path, ff.probe(path))
        role = c.get("role", "voice")
        if not info["has_audio"] or role not in buses:
            continue
        inputs += ["-ss", f"{c.get('in', 0):.4f}", "-t", f"{c['dur']:.4f}", "-i", path]
        k = inputs.count("-i") - 1
        fi = c.get("fade_in", 0.008 if role == "voice" else 0.0)
        fo = c.get("fade_out", 0.008 if role == "voice" else 0.0)
        if info.get("channels", 2) == 1:
            chans = ["pan=stereo|c0=c0|c1=c0"]
        elif c.get("mono"):
            chans = ["pan=mono|c0=0.5*c0+0.5*c1", "pan=stereo|c0=c0|c1=c0"]
        else:
            chans = ["aformat=channel_layouts=stereo"]
        chain = ["aresample=48000", "aformat=sample_fmts=fltp", *chans, f"volume={c.get('gain_db', 0)}dB"]
        if fi:
            chain.append(f"afade=t=in:d={fi}")
        if fo:
            chain.append(f"afade=t=out:st={max(0.0, c['dur'] - fo):.4f}:d={fo}")
        ms = int(round(c["start"] * 1000))
        chain.append(f"adelay={ms}|{ms}")
        graph.append(f"[{k}:a]{','.join(chain)}[c{n}]")
        buses[role].append(f"[c{n}]")
        if role == "voice":
            voice_spans.append((c["start"], c["start"] + c["dur"]))
    finals = []
    for role, labels in buses.items():
        if labels:
            graph.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest[{role}0]")
    if buses["voice"]:
        vc = "highpass=f=70,acompressor=threshold=0.08:ratio=2.5:attack=8:release=120:makeup=1.6" \
            if a.get("voice_chain", True) else "anull"
        graph.append(f"[voice0]{vc}[voice]")
        finals.append("[voice]")
    if buses["music"]:
        duck = _duck_expr(voice_spans, a.get("duck_db", 12)) if a.get("duck_db", 12) else None
        graph.append(f"[music0]{duck or 'anull'}[music]")
        finals.append("[music]")
    if buses["sfx"]:
        graph.append("[sfx0]anull[sfx]")
        finals.append("[sfx]")
    if not finals:
        return None
    graph.append(f"{''.join(finals)}amix=inputs={len(finals)}:normalize=0:duration=longest,"
                 f"apad=whole_dur={dur:.4f},atrim=0:{dur:.4f}[mix]")
    wav = os.path.join(tmp, "mix.wav")
    ff.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(graph),
            "-map", "[mix]", "-c:a", "pcm_f32le", wav], name="audio mix")
    target, tp = a.get("loudness", -14.0), a.get("true_peak", -1.0)
    m = ff.loudnorm_measure(wav, target, tp)
    ff.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", wav, "-af", ff.loudnorm_filter(m, target, tp),
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-t", f"{dur:.4f}", out], name="audio master")
    return out


def render(timeline, out, draft=False, encoder="cpu", threads=0):
    """Render a timeline (dict or JSON path) to an MP4."""
    if isinstance(timeline, (str, Path)):
        base_dir = str(Path(timeline).resolve().parent)
        tl = json.loads(Path(timeline).read_text())
    else:
        tl, base_dir = timeline, os.getcwd()
    started = time.monotonic()
    out = str(out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reel-") as tmp:
        video = os.path.join(tmp, "video.mp4")
        sound = os.path.join(tmp, "audio.m4a")
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(render_audio, tl, base_dir, sound, tmp)
            render_video(tl, base_dir, video, draft, encoder, threads)
            has_audio = fut.result() is not None
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", video]
        if has_audio:
            cmd += ["-i", sound, "-map", "0:v", "-map", "1:a"]
        ff.run(cmd + ["-c", "copy", "-movflags", "+faststart", "-t", f"{tl['duration']:.4f}", out], name="mux")
    ff.log(f"rendered {out} in {time.monotonic() - started:.1f}s")
    return out


def still(timeline, t, out):
    """One composed frame at time t as an image (no encode)."""
    if isinstance(timeline, (str, Path)):
        base_dir = str(Path(timeline).resolve().parent)
        tl = json.loads(Path(timeline).read_text())
    else:
        tl, base_dir = timeline, os.getcwd()
    comp = Compositor(tl, base_dir)
    f = int(round(t * comp.fps))
    comp.from_t = f / comp.fps
    img = comp.frame(f)
    comp.close()
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92] if str(out).lower().endswith((".jpg", ".jpeg")) else [])
    return out
