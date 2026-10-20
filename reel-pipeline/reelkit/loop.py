"""Endless background from a short clip: no transition, no blend, nothing reversed.

    reel loop <clip> <seconds> <out.mp4> [--start F --end F]

The wrap from the loop's last frame back to its first is a plain cut. What makes
it invisible: the dense optical flow from the loop's last frame to its first is
measured, and frame k of the loop is warped by k/L of that flow. The correction
grows by a fraction of a pixel per frame (never visible) and by the last frame
the picture is geometrically where the first frame is, so the cut has no jump of
horizon, boat or arm left in it. Without --start/--end the loop span is chosen
automatically as the one whose wrap remains smallest after this correction.
The leftover difference in light and shading (not fine texture) is spread over
the loop the same way, 1/L per frame.
"""
import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import ff
from .paths import FFMPEG

W, H, FPS = 720, 1280, 30


def read(path):
    frames = list(ff.frames(path, W, H, fps=FPS))      # a decoder failure raises ff.FFError
    if not frames:
        raise SystemExit(f"reel loop: no frames could be decoded from {path}")
    return np.stack(frames)


def encode(unit, total, out):
    """total frames cycling through `unit`, encoded to `out` (written to a temp file
    beside it and renamed only when the encoder finished cleanly)."""
    out = Path(out)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp{out.suffix or '.mp4'}")
    enc = ff.Proc([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                   "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                   "-pix_fmt", "yuv420p", str(tmp)], "loop encode", stdin=subprocess.PIPE, bufsize=0)
    try:
        broken = None
        try:
            for k in range(total):
                ff.write_all(enc.proc.stdin, unit[k % len(unit)].tobytes())
            enc.proc.stdin.close()
        except BrokenPipeError as e:    # the encoder died: its exit code and stderr say why
            broken = e
        rc = enc.proc.wait()
        enc.finish()
        if rc or broken:
            raise enc.error(f"exited with {rc}" if rc else "stopped reading frames", cause=broken)
        os.replace(tmp, out)
    finally:
        enc.close()
        try:
            os.unlink(tmp)
        except OSError:
            pass


def flow(a, b, scale=0.5):
    """Dense flow such that b(x + f(x)) ≈ a(x), at full resolution."""
    ga = cv2.cvtColor(cv2.resize(a, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    f = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 5, 41, 5, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    f = cv2.GaussianBlur(f, (0, 0), 6)  # smooth: follow the scene's geometry, not the wave texture
    return cv2.resize(f, (a.shape[1], a.shape[0])) / scale


def warp(img, f, t):
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    return cv2.remap(img, xs + t * f[..., 0], ys + t * f[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def wrap_error(frames, s, e, f, sigma=3.0):
    nxt = warp(frames[e], f, 1.0).astype(np.float32)  # the loop's last frame after full correction
    nxt += cv2.GaussianBlur(frames[s].astype(np.float32) - nxt, (0, 0), sigma)
    step = np.median([np.abs(frames[k + 1].astype(np.float32) - frames[k]).mean() for k in range(s, e)])
    return np.abs(nxt.astype(np.float32) - frames[s]).mean() / step


def main(argv=None):
    ap = argparse.ArgumentParser(prog="reel loop", description="seamless endless loop from a short clip")
    ap.add_argument("clip")
    ap.add_argument("seconds", type=float)
    ap.add_argument("out")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--texture", type=float, default=3.0, help="px: residual detail finer than this is left alone")
    a = ap.parse_args(argv)
    frames = read(a.clip)
    n = len(frames)
    if a.start is None:
        # candidate spans: long ones first; the wrap goes from frame e to frame s
        cands = [(s, e) for s in range(0, n // 3, 6) for e in range(n - 1, s + int(2.5 * FPS), -6)]
        if not cands:
            ap.error(f"the clip has {n} frames: too short to choose a loop span (or pass --start/--end)")
        small = [cv2.resize(f, (90, 160)).astype(np.float32) for f in frames]
        cands.sort(key=lambda se: np.abs(small[se[1]] - small[se[0]]).mean() - 0.02 * (se[1] - se[0]))
        scored = []
        for s, e in cands[:12]:
            f = flow(frames[s], frames[e])
            scored.append((wrap_error(frames, s, e, f), s, e, f))
        scored.sort(key=lambda x: x[0])
        err, s, e, f = scored[0]
        print(f"loop {s}..{e} ({(e - s) / FPS:.2f}s), wrap = {err:.2f}x a normal frame step after correction")
    else:
        s, e = a.start, a.end
        if e is None or not 0 <= s < e < n:
            ap.error(f"--start and --end need 0 <= start < end < {n} (the clip's frame count)")
        f = flow(frames[s], frames[e])
        print(f"loop {s}..{e}: wrap = {wrap_error(frames, s, e, f):.2f}x a normal frame step after correction")
    L = e - s  # frames s .. e-1 play; frame e is never shown, its corrected version ≈ frame s
    # what the warp leaves (light, cloud and hull shading; not the fine wave texture,
    # which changes every frame anyway) is also spread over the loop, 1/L per frame
    resid = frames[s].astype(np.float32) - warp(frames[e], f, 1.0).astype(np.float32)
    resid = cv2.GaussianBlur(resid, (0, 0), a.texture)
    unit = [np.clip(warp(frames[s + k], f, k / L).astype(np.float32) + resid * (k / L), 0, 255).astype(np.uint8)
            for k in range(L)]
    encode(unit, round(a.seconds * FPS), a.out)


if __name__ == "__main__":
    main()
