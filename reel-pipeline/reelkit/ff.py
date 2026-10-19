"""ffmpeg / ffprobe helpers: probing, frame and audio pipes, loudness."""
import json
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np

from .paths import FFMPEG, FFPROBE

_T0 = time.monotonic()


def log(msg):
    print(f"[reel {time.monotonic() - _T0:6.1f}s] {msg}", file=sys.stderr, flush=True)


def run(cmd, name=None, **kw):
    """subprocess.run with check=True and an elapsed-time log line."""
    started = time.monotonic()
    try:
        return subprocess.run(cmd, check=True, **kw)
    finally:
        if name:
            log(f"{name}: {time.monotonic() - started:.2f}s")


def ffmpeg(*args, name=None, **kw):
    return run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], name=name, **kw)


def probe(path):
    """Display geometry (rotation applied), fps, duration, codecs, audio presence."""
    out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next((s for s in j["streams"] if s["codec_type"] == "video" and
              not s.get("disposition", {}).get("attached_pic")), None)
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    info = {"duration": float(j["format"].get("duration", 0) or 0), "size": int(j["format"].get("size", 0) or 0),
            "has_video": v is not None, "has_audio": a is not None}
    if v:
        w, h = int(v["width"]), int(v["height"])
        rot = 0
        for sd in v.get("side_data_list", []):
            if "rotation" in sd:
                rot = int(sd["rotation"])
        rot = int(v.get("tags", {}).get("rotate", rot))
        if abs(rot) % 180 == 90:
            w, h = h, w
        rate = v.get("avg_frame_rate") if v.get("avg_frame_rate", "0/0") != "0/0" else v.get("r_frame_rate", "0/1")
        fps = float(Fraction(rate)) if rate and rate != "0/0" else 0.0
        info.update(width=w, height=h, fps=round(fps, 3), vcodec=v.get("codec_name"), pix_fmt=v.get("pix_fmt"),
                    colorspace=v.get("color_space"), color_range=v.get("color_range"),
                    vbitrate=int(v.get("bit_rate", 0) or 0), nb_frames=int(v.get("nb_frames", 0) or 0))
    if a:
        info.update(acodec=a.get("codec_name"), sample_rate=int(a.get("sample_rate", 0) or 0),
                    channels=int(a.get("channels", 0) or 0))
    return info


def frames(path, w, h, fps=None, start=None, duration=None, vf=None, pix="bgr24", skip_nokey=False):
    """Yield decoded frames as uint8 arrays (h, w, 3) — or (h, w) for pix='gray'.
    `vf` is a filter chain run before the final scale to w×h (e.g. a crop)."""
    chans = 1 if pix == "gray" else 3
    chain = [f for f in (vf, f"fps={fps}" if fps else None, f"scale={w}:{h}:flags=lanczos", "setsar=1") if f]
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
    if skip_nokey:
        cmd += ["-skip_frame", "nokey"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-an", "-vf", ",".join(chain), "-f", "rawvideo", "-pix_fmt", pix, "-"]
    if skip_nokey:
        cmd[cmd.index("-f"):cmd.index("-f")] = ["-fps_mode", "passthrough"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)  # closed early on purpose
    size = w * h * chans
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            arr = np.frombuffer(buf, np.uint8)
            yield arr.reshape(h, w) if chans == 1 else arr.reshape(h, w, chans)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def keyframe_times(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries",
                          "frame=pts_time", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return [float(x) for x in out.split() if x and x != "N/A"]


def audio(path, sr=16000, mono=True, start=None, duration=None):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1" if mono else "2", "-ar", str(sr), "-f", "f32le", "-"]
    pcm = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(pcm, np.float32)
    return a if mono else a.reshape(-1, 2)


def _last_json(text):
    """The last top-level {...} block in ffmpeg's stderr (loudnorm prints more after it in 8.x)."""
    end = text.rfind("}")
    depth = 0
    for i in range(end, -1, -1):
        depth += text[i] == "}"
        depth -= text[i] == "{"
        if depth == 0:
            return json.loads(text[i:end + 1])
    raise ValueError("no JSON block in ffmpeg output")


def loudnorm_measure(path, target=-14.0, tp=-1.0, lra=11):
    err = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af",
                          f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return _last_json(err)


def loudnorm_filter(m, target=-14.0, tp=-1.0, lra=11):
    return (f"loudnorm=I={target}:TP={tp}:LRA={lra}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
            f"linear=true,aresample=48000")


def loudness(path):
    """Integrated loudness (LUFS), true peak (dBTP), LRA — BS.1770 via ebur128."""
    err = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    summary = err[err.rfind("Summary:"):]
    out = {}
    for line in summary.splitlines():
        line = line.strip()
        for key, name in (("I:", "I"), ("LRA:", "LRA"), ("Peak:", "TP")):
            if line.startswith(key):
                try:
                    out[name] = float(line.split()[1])
                except (IndexError, ValueError):
                    pass
    return out
