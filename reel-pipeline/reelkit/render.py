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
              "speed": 1.0, "track": 1, "transition_in": {"type": "dissolve", "dur": 0.3},
              "hold_last": false}],
             (a source that runs out before its clip ends fails the render, beyond a couple
              of frames of rate rounding, unless the clip sets hold_last: then its last
              frame is held)
   "captions": [{"s": 0.2, "e": 1.4, "text": "...", "words": [{"w","s","e"}], "style": "caption"}],
   "styles": {"caption": {...captions.py style...}},
   "grade": {"lut": "x.cube", "sharpen": 0.3, "grain": 4, "vignette": 0.2},
   "audio": {"clips": [{"media": "m1", "start": 0, "in": 12.3, "dur": 2.1, "role": "voice",
                        "gain_db": 0, "fade_in": 0.008, "fade_out": 0.008, "mono": true}],
             "voice_chain": true, "duck_db": 12, "loudness": -14, "true_peak": -1},
             (music is keyframe-ducked by duck_db wherever voice clips play; true_peak is the
              delivery ceiling: loudnorm aims 0.5 dB under it so AAC encoding cannot overshoot)
   "encode": {"preset": "medium", "crf": 17}}

Keyframes: [t, value] or [t, value, ease] with t relative to the clip's start; ease is
the curve INTO that key: linear, in, out, inout, in_cubic, out_cubic, inout_cubic,
out_back, hold, or [x1, y1, x2, y2] (CSS cubic-bezier).
Transitions (transition_in, centred on the cut): cut, dissolve, dip_black, dip_white,
flash, zoom, whip_left, whip_right, whip_up, whip_down. Video tracks > 1 are overlays.

Failures are bounded: one cancel flag and two deadlines (timeout_s for the whole render,
stall_s without an encoded frame) govern every wait of the frame producer, the encoder
writer, the decoders and the audio thread. On any failure the decoders, the encoder and
the audio ffmpegs are stopped (SIGTERM, SIGKILL after a grace), threads are joined with
bounded waits and RenderError is raised with .record = {"stage": video | audio | mux |
compose | decode | timeout, "cmd", "returncode", "stderr_tail", "cause", "transient"}.
The MP4 is written to a unique temp file beside `out` and renamed only when complete.
Child processes stay in the caller's process group.
"""
import json
import math
import os
import queue
import secrets
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from . import captions, ff
from .paths import FFMPEG

RENDERER_VERSION = 1     # bump whenever the same timeline can render to different pixels or audio
STOP_GRACE = 2.0         # s from SIGTERM to SIGKILL, and for joining a thread once its process is gone
HOLD_TOLERANCE = 2       # frames a source may fall short by (frame-rate rounding) before it is an error
DECODE_STALL = 120.0     # s a decoder may go without a frame when no render deadline governs it
PROBE_TIMEOUT = 120.0

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


# ---------------------------------------------------------------- failures and deadlines

class RenderError(RuntimeError):
    """A failed render. .record has the keys of ff.FFError.record; its stage is one of
    video | audio | mux | compose | decode | timeout."""

    def __init__(self, stage, message, cmd=None, returncode=None, stderr_tail="", cause=None, transient=False):
        if cause is not None:
            message = f"{message}: {ff.describe(cause) if isinstance(cause, BaseException) else cause}"
        self.record = ff.error_record(stage, cmd, returncode, stderr_tail, message, transient)
        super().__init__(f"{stage}: {message}")

    def __reduce__(self):
        return ff.rebuild_error, (type(self), self.args, self.__dict__)

    @classmethod
    def wrap(cls, stage, message, exc):
        """`exc` as a RenderError, keeping an ff record's command, exit code and stderr; an
        ff timeout becomes stage timeout. A RenderError is returned as it is."""
        if isinstance(exc, RenderError):
            return exc
        rec = getattr(exc, "record", None) or {}
        if getattr(exc, "reason", None) == "timeout":
            stage = "timeout"
        return cls(stage, message, rec.get("cmd"), rec.get("returncode"), rec.get("stderr_tail", ""), exc,
                   rec.get("transient", False))


def _attach(err, p, rc):
    """Give a deadline error the command, exit code and stderr of the process it caught."""
    rec = ff.error_record(err.record["stage"], p.cmd, rc, p.tail())
    err.record.update(cmd=rec["cmd"], returncode=rec["returncode"], stderr_tail=rec["stderr_tail"])


class _EncoderGone(Exception):
    """The encoder exited, or its pipe broke, while frames were still due."""


class _Guard:
    """What one render's threads share: the cancel flag, the wall and no-progress
    deadlines and the first failure. Only the frame producer raises from check()."""

    def __init__(self, timeout_s=None, stall_s=120):
        self.cancel = threading.Event()
        self.timeout_s = float(timeout_s) if timeout_s else None
        self.stall_s = float(stall_s) if stall_s else None
        now = time.monotonic()
        self.deadline = now + self.timeout_s if self.timeout_s else None
        self.progress = now          # when the encoder last took a frame
        self.failure = None
        self.hooks = []              # callables that raise when something they watch has died
        self._lock = threading.Lock()

    def note(self, err):
        """Record a failure (the first one wins) and cancel everything."""
        with self._lock:
            if self.failure is None:
                self.failure = err
        self.cancel.set()

    def remaining(self):
        return None if self.deadline is None else max(0.0, self.deadline - time.monotonic())

    def check(self, where=None, stall=True):
        now, at = time.monotonic(), f" while {where}" if where else ""
        if self.failure is None:
            if self.deadline is not None and now > self.deadline:
                self.note(RenderError("timeout", f"the render exceeded its {self.timeout_s:g}s limit{at}",
                                      transient=True))
            elif stall and self.stall_s and now - self.progress > self.stall_s:
                self.note(RenderError("timeout", f"no frame encoded for {self.stall_s:g}s{at}", transient=True))
        if self.failure is not None:
            raise self.failure
        for hook in self.hooks:
            hook()
        if self.cancel.is_set():
            raise RenderError("video", "the render was cancelled")


def _step_timeout(guard, dur):
    """A bound for one ffmpeg step: what is left of the wall time or, without a wall
    limit, stall_s plus 20× the reel's length (generous, but never unbounded)."""
    if guard is None:
        return None
    left = guard.remaining()
    if left is not None:
        return max(1.0, left)
    return guard.stall_s + 20.0 * float(dur) if guard.stall_s else None


def _unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _quietly(fn, what):
    try:
        fn()
    except Exception as e:      # cleanup must not hide the error being reported
        ff.log(f"{what}: {ff.describe(e)}")


# ---------------------------------------------------------------- sources

_END = object()                 # a clean end of decode, in a Reader's queue


class Reader:
    """Sequential frames of one clip window, decoded by ffmpeg at a working size
    chosen so the per-frame warp rarely has to shrink the picture by much.
    A decoder failure, or a source that runs out more than HOLD_TOLERANCE frames before
    the clip ends (unless the clip sets hold_last), raises RenderError (stage decode).
    close() before the end is a clean cancel."""

    def __init__(self, path, info, clip, canvas, fps, pre, post, from_t=None, guard=None):
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
        self.label = f"clip at {clip['start']:.2f}s ({os.path.basename(path)})"
        self.hold_last = bool(clip.get("hold_last"))
        self.guard = guard
        self.stall_s = guard.stall_s if guard is not None else DECODE_STALL
        self.proc = ff.Proc(ff.frames_cmd(path, self.ww, self.wh, fps=fps, start=max(0.0, src_start),
                                          duration=(pre + clip["dur"] + post) * self.speed + 1.0,
                                          vf=",".join(vf) or None), "decode", stdout=subprocess.PIPE)
        self.idx = -1
        self.cur = None
        self.fps = fps
        self.eof = False
        # decode ahead on a thread: the OS pipe holds only 64 KB, so without this
        # ffmpeg would wait for the compositor and the compositor for ffmpeg
        self.q = queue.Queue(maxsize=6)
        self.stop = False                # set by close(): the frames are no longer wanted
        self.thread = threading.Thread(target=self._pump, daemon=True, name="reel-decode")
        self.thread.start()

    def _pump(self):
        end = _END
        try:
            for fr in ff.read_frames(self.proc, self.ww, self.wh):
                if not self._put(fr):
                    return
        except Exception as e:           # nonzero exit, a cut-off frame (or close() killing it)
            end = e
        self._put(end)

    def _put(self, item):
        while not self.stop:
            try:
                self.q.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def _next(self):
        """The next decoded frame; None once the decode has ended cleanly."""
        waited = time.monotonic()
        while True:
            try:
                item = self.q.get(timeout=0.25)
                break
            except queue.Empty:
                pass
            if self.guard is not None:
                try:
                    self.guard.check(f"waiting for the decoder of the {self.label}")
                except RenderError as e:
                    if e.record["stage"] == "timeout" and not e.record["cmd"]:
                        _attach(e, self.proc, self.proc.proc.poll())
                    raise
            if self.stall_s and time.monotonic() - waited > self.stall_s:
                raise RenderError("timeout", f"the decoder of the {self.label} gave no frame for {self.stall_s:g}s",
                                  self.proc.cmd, self.proc.proc.poll(), self.proc.tail(), transient=True)
            if not self.thread.is_alive() and self.q.empty():
                raise RenderError("decode", f"the decoder thread of the {self.label} ended without a result",
                                  self.proc.cmd, self.proc.proc.poll(), self.proc.tail())
        if item is _END:
            return None
        if isinstance(item, BaseException):
            raise RenderError.wrap("decode", f"decoding the {self.label} failed", item) from item
        return item

    def at(self, t):
        k = int(round((t - self.first) * self.fps)) - self.hold_head
        while self.idx < k and not self.eof:
            nxt = self._next()
            if nxt is None:
                self.eof = True
                break
            self.cur, self.idx = nxt, self.idx + 1
        if self.cur is None and not self.eof:      # inside the held head: the first frame
            nxt = self._next()
            self.eof = nxt is None
            if nxt is not None:
                self.cur, self.idx = nxt, 0
        if self.cur is None:
            raise RenderError("decode", f"no frames decoded for the {self.label}", self.proc.cmd,
                              self.proc.proc.poll(), self.proc.tail())
        if self.eof and k - self.idx > HOLD_TOLERANCE and not self.hold_last:
            raise RenderError("decode", f"the source of the {self.label} ran out {k - self.idx} frames early "
                                        f"(a clip may hold its last frame only with \"hold_last\": true)",
                              self.proc.cmd, self.proc.proc.poll(), self.proc.tail())
        return self.cur

    def close(self):
        self.stop = True
        self.proc.terminate(kill=True)             # a decoder mid-frame or blocked on the pipe ends now
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.thread.join(timeout=STOP_GRACE)
        self.proc.close(pipes=not self.thread.is_alive())


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
    def __init__(self, tl, base_dir, guard=None):
        self.tl = tl
        self.guard = guard
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
            try:
                timeout = PROBE_TIMEOUT
                if self.guard is not None:
                    self.guard.check("probing preview media", stall=False)
                    left = self.guard.remaining()
                    if left is not None:
                        timeout = min(timeout, max(0.001, left))
                self.info[path] = ff.probe(path, timeout=timeout)
            except ff.FFError as e:
                raise RenderError.wrap("decode", f"cannot probe {path}", e) from e
        return self.info[path]

    def _reader(self, c):
        key = id(c)
        if key not in self.readers:
            path = self.media[c["media"]]
            info = self._info(path)
            try:
                self.readers[key] = Reader(path, info, c, (self.W, self.H), self.fps, c["_tin"], c["_tout"],
                                           from_t=self.from_t, guard=self.guard)
            except ff.FFError as e:
                raise RenderError.wrap("decode", f"cannot start decoding {path}", e) from e
        return self.readers[key]

    def _retire(self, t):
        for c in self.base + self.over:
            if id(c) in self.readers and t > c["start"] + c["dur"] + c["_tout"] + 1.0 / self.fps:
                self.readers.pop(id(c)).close()

    def _image(self, c):
        path = self.media[c["media"]]
        if path not in self.images:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RenderError("compose", f"cannot read the image {path}")
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
            d = {"whip_left": (-1, 0), "whip_right": (1, 0), "whip_up": (0, -1), "whip_down": (0, 1)}.get(kind, (-1, 0))
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
        readers, self.readers = list(self.readers.values()), {}
        for r in readers:
            _quietly(r.close, "closing a decoder")


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
    if enc.get("lossless"):              # references for quality measurement (VMAF)
        return ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"]
    if encoder == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5" if not draft else "p2", "-rc", "vbr", "-cq",
                str(enc.get("cq", 19)), "-b:v", "0"]
    preset = "veryfast" if draft else enc.get("preset", "medium")
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(enc.get("crf", 17)), "-profile:v", "high"]


def _put(q, item, guard, where):
    """A queue put that keeps checking the guard while the queue stays full."""
    while True:
        try:
            q.put(item, timeout=0.25)
            return
        except queue.Full:
            guard.check(where)


def render_video(tl, base_dir, out, draft=False, encoder="cpu", threads=0, guard=None):
    """Composite every frame and encode it (no audio) to `out`. Raises RenderError and
    leaves no `out` behind on failure."""
    guard = guard or _Guard()
    try:
        comp = Compositor(tl, base_dir, guard)
        W, H, fps = comp.W, comp.H, comp.fps
        vf = grade_filters(tl.get("grade") or {}, base_dir) + [
            "scale=out_color_matrix=bt709:out_range=tv", "format=yuv420p"]
    except RenderError:
        raise
    except Exception as e:
        raise RenderError.wrap("compose", "the timeline could not be set up", e) from e
    if comp.N <= 0:
        raise RenderError("compose", f"the timeline has no frames (duration {tl.get('duration')})")
    settings = tl.get("encode") or {}
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-vf", ",".join(vf),
           *_encoder_args(settings, draft, encoder), "-g", str(int(round(fps))), "-bf", "2",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-maxrate", "25M", "-bufsize", "50M", "-an", str(out)]
    if threads:                        # an output option: limits the encoder, not the raw input
        cmd[-1:-1] = ["-threads", str(threads)]
    try:
        enc = ff.Proc(cmd, "video", stdin=subprocess.PIPE, bufsize=0)   # unbuffered: close() never blocks
    except ff.FFError as e:
        raise RenderError.wrap("video", "could not start the encoder", e) from e
    out_q = queue.Queue(maxsize=8)
    werr, written = [], [0]

    def writer():
        try:
            while not guard.cancel.is_set():
                try:
                    buf = out_q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if buf is None:
                    return
                ff.write_all(enc.proc.stdin, buf)
                written[0] += 1
                guard.progress = time.monotonic()
        except Exception as e:         # BrokenPipeError once the encoder is gone: the producer reports it
            werr.append(e)

    def encoder_alive():
        if werr or enc.proc.poll() is not None:
            raise _EncoderGone()

    wt = threading.Thread(target=writer, daemon=True, name="reel-encode")
    wt.start()
    guard.hooks.append(encoder_alive)
    started = guard.progress = time.monotonic()
    try:
        for f in range(comp.N):
            guard.check()
            try:
                buf = comp.frame(f).tobytes()
            except (RenderError, _EncoderGone):
                raise
            except Exception as e:
                raise RenderError.wrap("compose", f"compositing frame {f} failed", e) from e
            _put(out_q, buf, guard, f"encoding frame {f}")
        _put(out_q, None, guard, "finishing the encode")
        while wt.is_alive():
            wt.join(0.25)
            guard.check("encoding the last frames")
        guard.hooks.remove(encoder_alive)
        if werr or written[0] != comp.N:
            raise _EncoderGone()
        enc.proc.stdin.close()
        while True:
            try:
                rc = enc.proc.wait(0.25)
                break
            except subprocess.TimeoutExpired:
                guard.check("finishing the encode")
        if rc != 0:
            raise _EncoderGone()
    except BaseException as e:
        rc0 = enc.proc.poll()                      # before we stop it ourselves
        guard.cancel.set()
        _quietly(comp.close, "closing the decoders")
        enc.terminate()
        wt.join(STOP_GRACE)
        enc.close(pipes=not wt.is_alive())
        _unlink(out)
        if not isinstance(e, Exception):
            raise
        raise _video_error(e, guard, enc, rc0, werr, written[0], comp.N) from e
    finally:
        if encoder_alive in guard.hooks:
            guard.hooks.remove(encoder_alive)
    _quietly(comp.close, "closing the decoders")
    enc.close()
    ff.log(f"video: {comp.N} frames in {time.monotonic() - started:.1f}s")


def _video_error(e, guard, enc, rc, werr, written, total):
    """The most telling RenderError for a failed video pass: a recorded failure (deadline,
    audio) first, then the encoder's own exit, then whatever was raised."""
    if guard.failure is not None:
        err = guard.failure
        if err.record["stage"] == "timeout" and not err.record["cmd"]:
            _attach(err, enc, rc)
        return err
    if isinstance(e, _EncoderGone):
        if rc is None:
            msg = f"the encoder stopped taking frames after {written} of {total}"
        else:
            msg = f"the encoder exited with {rc} after {written} of {total} frames"
        return RenderError("video", msg, enc.cmd, rc, enc.tail(), werr[0] if werr else None,
                           transient=rc is not None and rc < 0)
    return RenderError.wrap("compose", "rendering the video failed", e)


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


def render_audio(tl, base_dir, out, tmp, guard=None):
    """Mix and master the audio to an AAC file at `out`; None if the timeline has none.
    With a guard every ffmpeg step is bounded and stops when the render is cancelled."""
    a = tl.get("audio") or {}
    clips = [c for c in a.get("clips", []) if c.get("gain_db", 0) > -60]
    dur = tl["duration"]
    if not clips:
        return None
    cancel = guard.cancel if guard is not None else None
    inputs, graph, buses = [], [], {"voice": [], "music": [], "sfx": []}
    probe_cache, voice_spans = {}, []
    for n, c in enumerate(clips):
        path = _resolve(tl["media"][c["media"]], base_dir)
        if path not in probe_cache:
            probe_cache[path] = ff.probe(path, timeout=PROBE_TIMEOUT)
        info = probe_cache[path]
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
            "-map", "[mix]", "-c:a", "pcm_f32le", wav], name="audio mix", timeout=_step_timeout(guard, dur),
           cancel=cancel)
    target, tp = a.get("loudness", -14.0), a.get("true_peak", -1.0) - 0.5
    m = ff.loudnorm_measure(wav, target, tp, timeout=_step_timeout(guard, dur), cancel=cancel)
    ff.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", wav, "-af", ff.loudnorm_filter(m, target, tp),
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-t", f"{dur:.4f}", out], name="audio master",
           timeout=_step_timeout(guard, dur), cancel=cancel)
    return out


def _audio_job(tl, base_dir, out, tmp, guard, res):
    """render_audio on its own thread: the result or error goes into `res`; a failure of
    its own (not one caused by a cancel) also stops the video."""
    try:
        res["path"] = render_audio(tl, base_dir, out, tmp, guard)
    except BaseException as e:
        err = RenderError.wrap("audio", "the audio mix failed", e)
        res["error"] = err
        if not guard.cancel.is_set():
            guard.note(err)


def _load(timeline):
    """(timeline dict, the directory its relative paths are resolved against)."""
    if isinstance(timeline, (str, Path)):
        return json.loads(Path(timeline).read_text()), str(Path(timeline).resolve().parent)
    return timeline, os.getcwd()


def _fsync(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _encoder(encoder):
    """cpu | nvenc as given; "auto" is resolved here (logged) if a caller did not."""
    if encoder in (None, "cpu", "nvenc"):
        return encoder or "cpu"
    if encoder == "auto":
        from . import readiness
        try:
            r = readiness.resolve_encoder("auto")
        except Exception as e:
            raise RenderError("video", "could not resolve the encoder", cause=e) from e
        ff.log(f"encoder auto -> {r['actual']} ({r.get('reason')})")
        return r["actual"]
    raise RenderError("video", f"unknown encoder {encoder!r} (cpu | nvenc | auto)")


def render(timeline, out, draft=False, encoder="cpu", threads=0, timeout_s=None, stall_s=120):
    """Render a timeline (dict or JSON path) to an MP4. encoder: "cpu" | "nvenc" (resolve
    "auto" before fanning out; it is resolved here otherwise). timeout_s bounds the whole
    render, stall_s the time without an encoded frame (None disables either).
    Raises RenderError; `out` is replaced only by a complete file."""
    encoder = _encoder(encoder)
    try:
        tl, base_dir = _load(timeline)
    except (OSError, ValueError) as e:
        raise RenderError("compose", f"cannot read the timeline {timeline}", cause=e) from e
    started = time.monotonic()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    guard = _Guard(timeout_s, stall_s)
    # unique, hidden, and with the real extension (ffmpeg picks the container from it)
    part = out.with_name(f".{out.stem}.{os.getpid()}-{secrets.token_hex(4)}.tmp{out.suffix or '.mp4'}")
    res = {}
    try:
        with tempfile.TemporaryDirectory(prefix="reel-", ignore_cleanup_errors=True) as tmp:
            video, sound = os.path.join(tmp, "video.mp4"), os.path.join(tmp, "audio.m4a")
            at = threading.Thread(target=_audio_job, args=(tl, base_dir, sound, tmp, guard, res), daemon=True,
                                  name="reel-audio")
            at.start()
            try:
                render_video(tl, base_dir, video, draft, encoder, threads, guard)
                while at.is_alive():
                    at.join(0.25)
                    guard.check("mixing the audio", stall=False)
            except BaseException:
                guard.cancel.set()                 # stops the audio ffmpeg as well
                at.join(STOP_GRACE * 3)
                raise
            if "error" in res:
                raise res["error"]
            cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", video]
            if res.get("path") is not None:
                cmd += ["-i", sound, "-map", "0:v", "-map", "1:a"]
            try:
                ff.run(cmd + ["-c", "copy", "-movflags", "+faststart", "-t", f"{tl['duration']:.4f}", str(part)],
                       name="mux", timeout=_step_timeout(guard, tl["duration"]), cancel=guard.cancel)
                _fsync(part)
                os.replace(part, out)
            except Exception as e:
                raise RenderError.wrap("mux", "muxing failed", e) from e
    finally:
        _unlink(part)
    ff.log(f"rendered {out} in {time.monotonic() - started:.1f}s")
    return str(out)


def still(timeline, t, out, timeout_s=None):
    """One composed frame at time t as an image (no encode)."""
    tl, base_dir = _load(timeline)
    guard = _Guard(timeout_s, min(10, timeout_s) if timeout_s else 10)
    comp = Compositor(tl, base_dir, guard)
    try:
        f = int(round(t * comp.fps))
        comp.from_t = f / comp.fps
        guard.check(stall=False)
        img = comp.frame(f)
    finally:
        comp.close()
    if not cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92] if str(out).lower().endswith((".jpg", ".jpeg"))
                       else []):
        raise RenderError("compose", f"could not write {out}")
    return out
