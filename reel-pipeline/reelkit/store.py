"""Content-addressed asset store.

Every input (sample reel, long-form source, music, sfx) is stored once under
data/assets/<id>/ where id is the first 16 hex digits of its sha-256. Uploading
the same file twice (or under another name) resolves to the same asset, and every
derived artifact (transcript, style spec, editing copies) is cached next to it.

The store owns its bytes: ingest copies the file (never a hardlink, which would leave
the uploader able to change "stored" content) into a unique temp file inside
data/assets, hashing exactly the bytes it writes, checks and fsyncs the copy, and
publishes the asset directory (original + meta.json) with one rename under a lock on
the asset id, so simultaneous ingests of the same content yield one complete asset
and a failure (unreadable media, full disk, crash) leaves no half asset. An existing
id is accepted only when its full sha-256 matches. An asset's identity is its content:
its kind is the role it was first stored for; later ingests for other roles add to
meta["roles"]. Where the file came from is metadata only (name, origins).
"""
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from . import ff, fsutil
from .paths import DATA

KINDS = ("sample", "source", "music", "sfx", "image", "other")
STORE_VERSION = 2          # 2: private verified copies (1: possibly hardlinked to the upload)
CHUNK = 8 << 20
PROBE_TIMEOUT = 120
MAX_ORIGINS = 20
STALE_TEMP = 12 * 3600     # s after which an ingest's leftover temp file is swept


class IngestError(ValueError):
    """A file that cannot become an asset (unreadable media, id collision)."""


def sha256(path, chunk=CHUNK):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Asset:
    def __init__(self, dir_):
        self.dir = Path(dir_)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.id = self.meta["id"]
        self.kind = self.meta["kind"]
        self.roles = list(self.meta.get("roles") or [self.kind])
        self.path = self.dir / self.meta["file"]
        self.info = self.meta["probe"]

    def cache(self, name):
        """Path for a derived artifact of this asset (created lazily)."""
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __repr__(self):
        return f"<{self.kind} {self.id} {self.meta['name']}>"


# ------------------------------------------------------------------ helpers

def _root():
    return DATA / "assets"


def _meta(d):
    """meta.json of an asset dir as a dict; None when absent or unreadable."""
    p = Path(d) / "meta.json"
    if not p.exists():
        return None
    m = fsutil.read_json(p)
    return m if isinstance(m, dict) and m.get("id") and m.get("file") and isinstance(m.get("probe"), dict) else None


def _fsync_dir(p):
    try:
        fd = os.open(p, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _unlink(p):
    if p is not None:
        try:
            os.unlink(p)
        except OSError:
            pass


def _stat_key(path):
    """Identity of the file as it sits on disk now (path, inode, size, times): a hint that it
    was ingested before, never proof of its content."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    sig = [os.path.realpath(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:32]


def _remember(root, key, digest):
    if key:
        try:
            (root / ".seen").mkdir(parents=True, exist_ok=True)
            fsutil.atomic_write_text(root / ".seen" / key, digest)
        except OSError:
            pass


def _recall(root, key):
    try:
        d = (root / ".seen" / key).read_text().strip() if key else ""
    except OSError:
        return None
    return d if re.fullmatch(r"[0-9a-f]{64}", d) else None


def _copy_in(path, root):
    """Copy `path` into a unique temp file under data/assets/.tmp, hashing exactly the bytes
    written (a source changed mid-copy still yields a copy that matches its digest).
    Returns (temp path, sha-256, size, stat key when the source did not change meanwhile)."""
    tmpdir = root / ".tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    key0 = _stat_key(path)
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as src:
        fd, tmp = tempfile.mkstemp(dir=tmpdir, prefix="ingest.", suffix=path.suffix.lower())
        try:
            with os.fdopen(fd, "wb") as dst:
                while True:
                    b = src.read(CHUNK)
                    if not b:
                        break
                    h.update(b)
                    dst.write(b)
                    n += len(b)
                dst.flush()
                os.fsync(dst.fileno())
                if os.fstat(dst.fileno()).st_size != n:
                    raise OSError(f"{path.name}: stored copy has {os.fstat(dst.fileno()).st_size} bytes, read {n}")
        except BaseException:
            _unlink(tmp)
            raise
    key1 = _stat_key(path)
    return Path(tmp), h.hexdigest(), n, key0 if key0 == key1 else None


def _sweep(root, age=STALE_TEMP):
    """Remove temp copies and trashed dirs that a crashed ingest left behind (an ingest in
    progress keeps touching its temp file)."""
    now = time.time()
    for sub in (".tmp", ".trash"):
        for p in (root / sub).glob("*") if (root / sub).is_dir() else []:
            try:
                if now - p.lstat().st_mtime > age:
                    shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
            except OSError:
                pass


def _trash(root, d):
    """Move a broken asset dir out of the way (one rename), then delete it."""
    junk = root / ".trash" / f"{d.name}.{uuid.uuid4().hex[:8]}"
    junk.parent.mkdir(parents=True, exist_ok=True)
    os.rename(d, junk)
    shutil.rmtree(junk, ignore_errors=True)


def _intact(d, meta, size):
    """The stored original exists as this asset's own file of the expected size, written by this
    store version (an older one may have hardlinked the upload)."""
    try:
        st = os.stat(d / meta["file"])
    except OSError:
        return False
    return meta.get("store") == STORE_VERSION and st.st_nlink == 1 and st.st_size == size


def _with_role(d, meta, kind, path):
    """Record the role and origin of a later ingest (the kind itself never changes)."""
    roles = list(meta.get("roles") or [meta["kind"]])
    origins = [o for o in meta.get("origins") or [] if isinstance(o, str)]
    origin = os.path.realpath(path)
    changed = False
    if kind not in roles:
        roles.append(kind)
        changed = True
    if origin not in origins:
        origins = (origins + [origin])[-MAX_ORIGINS:]
        changed = True
    if changed or meta.get("roles") is None:
        meta = {**meta, "roles": roles, "origins": origins}
        fsutil.atomic_write_json(d / "meta.json", meta)
    return Asset(d)


def _adopt(root, d, digest, size, tmp, kind, path):
    """An existing asset with this content, checked (and repaired from `tmp` when given);
    None when there is none to adopt. Call under the asset id's lock."""
    meta = _meta(d)
    if meta is None:
        if d.exists() and tmp is not None:
            ff.log(f"{d.name}: asset dir without a readable meta.json; replacing it")
            _trash(root, d)
        return None
    if meta.get("sha256") and meta["sha256"] != digest:
        raise IngestError(f"{path.name}: asset id {d.name} is taken by different content "
                          f"(sha-256 {meta['sha256'][:24]}… ≠ {digest[:24]}…)")
    if not _intact(d, meta, size):
        stored = d / meta["file"]
        if tmp is None:
            return None              # the caller copies the file and comes back with bytes to repair from
        ok = stored.is_file() and sha256(stored) == digest
        if not ok:
            ff.log(f"{d.name}: stored original is missing or does not match its sha-256; re-storing the asset")
            _trash(root, d)
            return None
        os.chmod(tmp, 0o444)
        os.replace(tmp, stored)          # this store's own copy from now on
        _fsync_dir(d)
        meta = {**meta, "sha256": digest, "bytes": size, "store": STORE_VERSION}
        fsutil.atomic_write_json(d / "meta.json", meta)
        ff.log(f"{d.name}: stored original verified and replaced by a private copy")
    a = _with_role(d, meta, kind, path)
    ff.log(f"{path.name}: already stored as {a.kind} {a.id} ({a.meta['name']})")
    return a


def _publish(root, d, digest, size, tmp, kind, name, path):
    """Probe the copy and publish original + meta.json as the asset dir in one rename."""
    if d.exists():
        _trash(root, d)
    stage = Path(tempfile.mkdtemp(dir=root / ".tmp", prefix=f"{d.name}."))
    try:
        stored = stage / ("original" + path.suffix.lower())
        os.replace(tmp, stored)
        try:
            info = ff.probe(stored, timeout=PROBE_TIMEOUT)
        except Exception as e:
            raise IngestError(f"{path.name}: not readable as media ({ff.describe(e)[:400]})") from e
        if not (info.get("has_video") or info.get("has_audio")):
            raise IngestError(f"{path.name}: no audio or video stream")
        os.chmod(stored, 0o444)
        meta = {"id": d.name, "kind": kind, "roles": [kind], "name": name or path.name, "file": stored.name,
                "sha256": digest, "bytes": size, "probe": info, "ingested": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "origins": [os.path.realpath(path)], "store": STORE_VERSION}
        fsutil.atomic_write_json(stage / "meta.json", meta)
        _fsync_dir(stage)
        os.rename(stage, d)
        _fsync_dir(root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    ff.log(f"{path.name}: stored as {kind} {d.name}")
    return Asset(d)


# ------------------------------------------------------------------ API

def ingest(path, kind="other", name=None):
    """The asset holding this file's content, stored now if new. kind: the role it is ingested
    for (an existing asset keeps its kind and gains the role in meta["roles"]). Raises
    IngestError (a ValueError) for a file that is not readable media or whose id is taken."""
    path = Path(path)
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such file")
    if not path.is_file():
        raise IsADirectoryError(f"{path}: not a regular file")
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    _sweep(root)
    # the same unchanged file again: hash it to confirm, but skip copying it
    key = _stat_key(path)
    hint = _recall(root, key)
    if hint:
        m = _meta(root / hint[:16])
        if m and m.get("sha256") == hint and _intact(root / hint[:16], m, path.stat().st_size) and sha256(path) == hint:
            with fsutil.file_lock(root / ".locks" / hint[:16]):
                a = _adopt(root, root / hint[:16], hint, path.stat().st_size, None, kind, path)
            if a is not None:
                return a
    tmp, digest, size, key = _copy_in(path, root)
    d = root / digest[:16]
    try:
        # one lock per asset id: it guards everything that can publish into that directory
        with fsutil.file_lock(root / ".locks" / d.name):
            a = _adopt(root, d, digest, size, tmp, kind, path)
            if a is None:
                a = _publish(root, d, digest, size, tmp, kind, name, path)
    finally:
        _unlink(tmp)                     # already gone when it was published
    _remember(root, key, digest)
    return a


def get(ref):
    """An asset by id (or unique id prefix), or by a file path (hashed)."""
    ref = str(ref)
    root = _root()
    if os.path.exists(ref) and not (root / ref).exists():
        digest = sha256(ref)
        m = _meta(root / digest[:16])
        if m and m.get("sha256", digest) == digest:
            return Asset(root / digest[:16])
        raise KeyError(f"{ref} is not ingested")
    hits = [p for p in root.glob(f"{ref}*") if (p / "meta.json").exists()] if root.exists() and ref else []
    if len(hits) != 1:
        raise KeyError(f"asset {ref!r}: {'no match' if not hits else 'ambiguous'}")
    return Asset(hits[0])


def all_assets(kind=None):
    """Every asset (of a kind, or ever ingested for that role), unreadable ones skipped."""
    root = _root()
    out = []
    for p in sorted(root.glob("*")) if root.exists() else []:
        if p.name.startswith(".") or not (p / "meta.json").exists():
            continue
        try:
            out.append(Asset(p))
        except (OSError, ValueError, KeyError) as e:
            ff.log(f"asset {p.name}: unreadable meta.json ({e}); skipped")
    return [a for a in out if kind is None or a.kind == kind or kind in a.roles]
