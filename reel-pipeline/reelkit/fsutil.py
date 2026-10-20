"""Small filesystem primitives every module shares: atomic writes, tolerant JSON reads,
advisory locks and content hashes. Standard library only (readiness imports it before
anything else is known to work).

atomic_write_*  unique temp file in the target's directory, fsync, os.replace: a reader sees
                the old file or the new one, never a torn one, and two writers never share a
                temp name. The new file gets the usual umask permissions.
read_json       the default (with the reason on stderr) when the file is unreadable or not
                JSON; a missing file is a silent miss. Never raises.
file_lock       fcntl.flock on "<path>.lock" (parent directories created). Exclusive or
                shared; timeout None blocks, a number raises TimeoutError. Re-entrant within
                one thread (a nested exclusive lock is the same owner, not a deadlock); the
                descriptor is O_CLOEXEC so ffmpeg and other children never inherit the lock.
sha256_file / sha256_json   full hex digests; JSON canonically encoded (sorted keys, no
                whitespace, UTF-8).
"""
import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path

_held = threading.local()       # {lock path: [shared?, depth, fd]} of this thread


def _log(msg):
    try:
        from .ff import log
    except Exception:           # ff needs numpy and the ffmpeg paths; this must work without them
        print(f"[reel] {msg}", file=sys.stderr, flush=True)
        return
    log(msg)


def _plain(o):
    """json default=: numpy scalars / arrays and paths become plain JSON; anything else is an error."""
    if isinstance(o, os.PathLike):
        return os.fspath(o)
    for attr in ("tolist", "item"):         # numpy arrays, numpy scalars
        f = getattr(o, attr, None)
        if callable(f):
            return f()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


# ------------------------------------------------------------------ writes

def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    with contextlib.suppress(OSError):      # the rename itself survives a crash only once the directory is synced
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    return path


def atomic_write_text(path, text):
    return atomic_write_bytes(path, str(text).encode("utf-8", "surrogatepass"))


def atomic_write_json(path, obj, indent=1):
    return atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False, default=_plain) + "\n")


# ------------------------------------------------------------------ reads

def read_json(path, default=None):
    try:
        with open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError, RecursionError) as e:     # UnicodeDecodeError and JSONDecodeError are ValueErrors
        _log(f"{path}: unreadable ({type(e).__name__}: {str(e)[:200]}); treated as missing")
        return default


# ------------------------------------------------------------------ locks

def _holder(fd):
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        return os.read(fd, 64).decode("ascii", "replace").strip() or "?"
    except OSError:
        return "?"


@contextlib.contextmanager
def file_lock(path, timeout=None, shared=False):
    lp = os.path.abspath(os.fspath(path)) + ".lock"
    held = getattr(_held, "locks", None)
    if held is None:
        held = _held.locks = {}
    cur = held.get(lp)
    if cur is not None:
        if cur[0] and not shared:
            raise RuntimeError(f"{lp}: an exclusive lock cannot be taken while this thread holds it shared")
        cur[1] += 1
        try:
            yield
        finally:
            cur[1] -= 1
        return
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
    try:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if timeout is None:
            fcntl.flock(fd, mode)
        else:
            deadline, pause = time.monotonic() + max(0.0, float(timeout)), 0.02
            while True:
                try:
                    fcntl.flock(fd, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError(f"{lp}: still locked after {timeout:g}s (holder pid {_holder(fd)})") \
                            from None
                    time.sleep(min(pause, left))
                    pause = min(pause * 2, 0.5)
        if not shared:          # who holds it, for the timeout message of the next waiter
            with contextlib.suppress(OSError):
                os.ftruncate(fd, 0)
                os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
    except BaseException:
        os.close(fd)
        raise
    held[lp] = [shared, 1, fd]
    try:
        yield
    finally:
        held.pop(lp, None)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ------------------------------------------------------------------ hashes

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_plain)


def sha256_json(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8", "surrogatepass")).hexdigest()
