"""ffmpeg / ffprobe helpers: probing, frame and audio pipes, loudness.

Every child stays in the caller's process group (a job cancel that signals the group
reaches it), has /dev/null for stdin and stdout unless they are piped, and has its
stderr drained by a thread into a bounded tail, so an unread pipe can never stall it.
Failures raise FFError; its .record is
{"stage", "cmd", "returncode", "stderr_tail", "cause", "transient"}.
"""
import collections
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
from fractions import Fraction

import numpy as np

from .paths import FFMPEG, FFPROBE

_T0 = time.monotonic()
TAIL_LINES = 40           # stderr lines kept per process
TAIL_CHARS = 4000         # ... and at most this much of them in a record
CMD_CHARS = 2000
STOP_GRACE = 2.0          # s between SIGTERM and SIGKILL
DECODE_EXIT_WAIT = 30.0   # s a decoder may take to exit once its output has ended


def log(msg):
    print(f"[reel {time.monotonic() - _T0:6.1f}s] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ errors

def describe(exc):
    """'Type: message', plus where it was raised for an error without an ff record."""
    text = f"{type(exc).__name__}: {exc}"
    if getattr(exc, "record", None) is None and exc.__traceback__ is not None:
        tb = traceback.extract_tb(exc.__traceback__)
        if tb:
            text += f" ({os.path.basename(tb[-1].filename)}:{tb[-1].lineno} in {tb[-1].name})"
    return text


def error_record(stage, cmd=None, returncode=None, stderr_tail="", cause=None, transient=False):
    if cmd is not None and not isinstance(cmd, str):
        cmd = shlex.join(str(c) for c in cmd)
    if cmd and len(cmd) > CMD_CHARS:
        cmd = cmd[:CMD_CHARS] + " ..."
    if isinstance(cause, BaseException):
        cause = describe(cause)
    return {"stage": stage, "cmd": cmd, "returncode": returncode, "stderr_tail": (stderr_tail or "")[-TAIL_CHARS:],
            "cause": cause, "transient": bool(transient)}


def rebuild_error(cls, args, state):
    """Unpickle an error whose __init__ takes more than its message (it crosses from a
    worker process to the batch)."""
    e = cls.__new__(cls, *args)
    e.__dict__.update(state)
    return e


class FFError(RuntimeError):
    """A failed ffmpeg / ffprobe run. .record: see error_record(). .reason: exit | timeout |
    cancelled | short | start | output."""

    def __init__(self, message, record, reason="exit"):
        last = (record.get("stderr_tail") or "").strip().splitlines()[-1:]
        super().__init__(f"{record['stage']}: {message}" + (f" ({last[0].strip()[:300]})" if last else ""))
        self.record, self.reason = record, reason

    def __reduce__(self):
        return rebuild_error, (type(self), self.args, self.__dict__)


def _fail(stage, cmd, message, returncode=None, stderr="", cause=None, reason="exit", transient=None):
    if transient is None:
        transient = reason == "timeout" or (reason == "exit" and returncode is not None and returncode < 0)
    if isinstance(stderr, bytes):
        stderr = stderr[-TAIL_CHARS * 2:].decode("utf-8", "replace")
    return FFError(message, error_record(stage, cmd, returncode, (stderr or "").strip(),
                                         message if cause is None else cause, transient), reason)


# ------------------------------------------------------------------ processes

class Proc:
    """A child process whose stderr a thread drains into a bounded tail (a PIPE nobody
    reads fills at 64 KB and stalls the child). It stays in the caller's process group;
    stdin and stdout default to /dev/null (the CLI's own stdout carries JSON)."""

    def __init__(self, cmd, stage, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, keep=TAIL_LINES, bufsize=-1,
                 **popen):
        self.cmd, self.stage = [str(c) for c in cmd], stage
        self._lines = collections.deque(maxlen=keep)
        self._lock = threading.Lock()
        try:
            self.proc = subprocess.Popen(self.cmd, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE,
                                         bufsize=bufsize, **popen)
        except OSError as e:
            raise _fail(stage, self.cmd, f"could not start {os.path.basename(self.cmd[0])}", cause=e,
                        reason="start") from e
        self._drain = threading.Thread(target=self._read_stderr, daemon=True, name=f"stderr-{stage}")
        self._drain.start()

    def _read_stderr(self):
        try:
            for line in iter(lambda: self.proc.stderr.readline(4096), b""):
                text = line.decode("utf-8", "replace").rstrip()
                with self._lock:
                    self._lines.append(text)
        except (OSError, ValueError):
            pass

    def text(self):
        """Every kept stderr line."""
        with self._lock:
            return "\n".join(self._lines)

    def tail(self):
        return self.text().strip()[-TAIL_CHARS:]

    def finish(self, grace=STOP_GRACE):
        """Once the child has exited: let the drain thread read the rest of its stderr."""
        self._drain.join(grace)

    def error(self, message, reason="exit", cause=None, transient=None):
        return _fail(self.stage, self.cmd, message, self.proc.returncode, self.tail(), cause, reason, transient)

    def terminate(self, grace=STOP_GRACE, kill=False):
        """SIGTERM (SIGKILL straight away with kill=True), SIGKILL after `grace` s, reap.
        Returns the exit code, None if the child would not die."""
        p = self.proc
        if p.poll() is None:
            try:
                p.kill() if kill else p.terminate()
            except OSError:
                pass
            try:
                p.wait(grace if not kill else max(grace, 5.0))
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except OSError:
                    pass
                try:
                    p.wait(5.0)
                except subprocess.TimeoutExpired:
                    log(f"{self.stage}: pid {p.pid} did not exit after SIGKILL")
        return p.returncode

    def close(self, grace=STOP_GRACE, kill=False, pipes=True):
        """Stop the child if it still runs, then close its pipes (pipes=False when another
        thread may still be using one). Safe to call more than once."""
        if self.terminate(grace, kill) is None:
            return                       # a pipe of a live child: closing could block forever
        self._drain.join(grace)
        p = self.proc
        for f in ((p.stdin, p.stdout) if pipes else ()) + ((p.stderr,) if not self._drain.is_alive() else ()):
            if f is not None:
                try:
                    f.close()
                except (OSError, ValueError):
                    pass


def write_all(f, buf):
    """All of `buf` into an unbuffered pipe (one raw write may take only part of it)."""
    mv = memoryview(buf)
    while len(mv):
        n = f.write(mv)
        if not n:
            raise BrokenPipeError("pipe accepted no data")
        mv = mv[n:]


def run(cmd, name=None, timeout=None, cancel=None, stage=None, **kw):
    """Run a command to completion, stderr drained into a bounded tail, stdout discarded
    unless captured. Raises FFError on a nonzero exit, after `timeout` s, or as soon as
    `cancel` (a threading.Event) is set. Returns a CompletedProcess (stderr = the kept
    tail; stdout if capture_output=True)."""
    return _run(cmd, stage or name or os.path.basename(str(cmd[0])), name=name, timeout=timeout, cancel=cancel, **kw)


def _run(cmd, stage, name=None, timeout=None, cancel=None, keep=TAIL_LINES, check=True, capture_output=False,
         stdout=subprocess.DEVNULL, **popen):
    started = time.monotonic()
    out = []
    p = Proc(cmd, stage, stdout=subprocess.PIPE if capture_output else stdout, keep=keep, **popen)
    reader = None
    if capture_output:
        reader = threading.Thread(target=lambda: out.append(p.proc.stdout.read()), daemon=True)
        reader.start()
    deadline = started + timeout if timeout else None
    poll = 0.2 if cancel is not None or deadline is not None else None
    try:
        while True:
            try:
                p.proc.wait(poll)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.is_set():
                p.terminate()
                raise p.error("cancelled", reason="cancelled")
            if deadline is not None and time.monotonic() > deadline:
                p.terminate()
                raise p.error(f"timed out after {timeout:g}s", reason="timeout")
        p.finish()
        if reader is not None:
            reader.join(STOP_GRACE)
        rc = p.proc.returncode
        if rc and check:
            raise p.error(f"exited with {rc}")
        return subprocess.CompletedProcess(p.cmd, rc, b"".join(out) if capture_output else None, p.text())
    finally:
        p.close(pipes=reader is None or not reader.is_alive())
        if name:
            log(f"{name}: {time.monotonic() - started:.2f}s")


def ffmpeg(*args, name=None, **kw):
    return run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], name=name, **kw)


def _capture(cmd, stage, timeout=None, text=False):
    """subprocess.run with both pipes captured; FFError on failure. Returns (stdout, stderr)."""
    cmd = [str(c) for c in cmd]
    try:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                           **({"text": True, "errors": "replace"} if text else {}))
    except subprocess.TimeoutExpired as e:
        raise _fail(stage, cmd, f"timed out after {timeout:g}s", stderr=e.stderr or "", reason="timeout") from e
    except OSError as e:
        raise _fail(stage, cmd, f"could not start {os.path.basename(cmd[0])}", cause=e, reason="start") from e
    if r.returncode:
        raise _fail(stage, cmd, f"exited with {r.returncode}", r.returncode, r.stderr)
    return r.stdout, r.stderr


# ------------------------------------------------------------------ probing

def probe_json(path, timeout=None):
    """ffprobe's streams + format JSON."""
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    out, err = _capture(cmd, "probe", timeout, text=True)
    try:
        return json.loads(out)
    except ValueError as e:
        raise _fail("probe", cmd, "unreadable ffprobe output", 0, err, cause=e, reason="output") from e


def probe(path, timeout=None):
    """Display geometry (rotation applied), fps, duration, codecs, audio presence."""
    j = probe_json(path, timeout)
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


# ------------------------------------------------------------------ frames

# ffmpeg >= 6.1 exits nonzero when a seek lands past the last frame and nothing reaches
# the output; that is an empty decode, not a broken one (unless the input itself failed)
_NO_OUTPUT = ("Nothing was written into output file", "received no packets", "Output file is empty",
              "Could not open encoder before EOF")
_INPUT_FAILED = ("No such file", "Invalid data found", "Error opening input", "Permission denied",
                 "moov atom not found", "does not contain any stream", "matches no streams")


def _empty_decode(stderr):
    return any(m in stderr for m in _NO_OUTPUT) and not any(m in stderr for m in _INPUT_FAILED)


def frames_cmd(path, w, h, fps=None, start=None, duration=None, vf=None, pix="bgr24", skip_nokey=False):
    """The decoder command behind frames()."""
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
    return cmd


def read_frames(p, w, h, pix="bgr24"):
    """Frames from a decoder Proc started with stdout=PIPE. Raises FFError when it exits
    nonzero or its output ends inside a frame; reaching EOF before any frame (a start
    past the end) is an empty decode. Stopping early is the caller's business: close p."""
    chans = 1 if pix == "gray" else 3
    size = w * h * chans
    if size <= 0:
        raise ValueError(f"bad frame size {w}x{h}")
    n, buf = 0, b""
    while True:
        buf = p.proc.stdout.read(size)
        if len(buf) < size:
            break
        n += 1
        arr = np.frombuffer(buf, np.uint8)
        yield arr.reshape(h, w) if chans == 1 else arr.reshape(h, w, chans)
    try:
        rc = p.proc.wait(DECODE_EXIT_WAIT)
    except subprocess.TimeoutExpired:
        p.terminate()
        raise p.error(f"did not exit {DECODE_EXIT_WAIT:g}s after its output ended", reason="timeout") from None
    p.finish()
    if rc and not (n == 0 and _empty_decode(p.text())):
        raise p.error(f"exited with {rc} after {n} frames")
    if buf:
        raise p.error(f"output ended inside frame {n + 1} ({len(buf)} of {size} bytes)", reason="short")


def frames(path, w, h, fps=None, start=None, duration=None, vf=None, pix="bgr24", skip_nokey=False):
    """Yield decoded frames as uint8 arrays (h, w, 3) — or (h, w) for pix='gray'.
    `vf` is a filter chain run before the final scale to w×h (e.g. a crop).
    Raises FFError if the decoder fails (nonzero exit, a cut-off frame). Closing the
    generator early (break, next(), garbage collection) stops the decoder: not an error."""
    p = Proc(frames_cmd(path, w, h, fps, start, duration, vf, pix, skip_nokey), "decode", stdout=subprocess.PIPE)
    try:
        yield from read_frames(p, w, h, pix)
    finally:
        p.close(kill=True)


def keyframe_times(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries",
                          "frame=pts_time", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL).stdout
    return [float(x) for x in out.split() if x and x != "N/A"]


# ------------------------------------------------------------------ audio

def audio(path, sr=16000, mono=True, start=None, duration=None, timeout=None):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1" if mono else "2", "-ar", str(sr), "-f", "f32le", "-"]
    pcm, _ = _capture(cmd, "audio decode", timeout)
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


def loudnorm_measure(path, target=-14.0, tp=-1.0, lra=11, timeout=None, cancel=None):
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af",
           f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"]
    err = _run(cmd, "loudness measure", timeout=timeout, cancel=cancel, keep=400).stderr
    try:
        return _last_json(err)
    except ValueError as e:
        raise _fail("loudness measure", cmd, "no loudnorm measurement in the output", 0, err, cause=e,
                    reason="output") from e


def loudnorm_filter(m, target=-14.0, tp=-1.0, lra=11):
    return (f"loudnorm=I={target}:TP={tp}:LRA={lra}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
            f"linear=true,aresample=48000")


def loudness(path, timeout=None):
    """Integrated loudness (LUFS), true peak (dBTP), LRA — BS.1770 via ebur128."""
    _, err = _capture([FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "ebur128=peak=true",
                       "-f", "null", "-"], "loudness", timeout, text=True)
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
