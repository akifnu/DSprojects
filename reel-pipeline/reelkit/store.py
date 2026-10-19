"""Content-addressed asset store.

Every input (sample reel, long-form source, music, sfx) is stored once under
data/assets/<id>/ where id is the first 16 hex digits of its sha-256. Uploading
the same file twice (or under another name) resolves to the same asset, and every
derived artifact (transcript, style spec, editing copies) is cached next to it.
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from . import ff
from .paths import DATA

KINDS = ("sample", "source", "music", "sfx", "image", "other")


def sha256(path, chunk=8 << 20):
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
        self.path = self.dir / self.meta["file"]
        self.info = self.meta["probe"]

    def cache(self, name):
        """Path for a derived artifact of this asset (created lazily)."""
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __repr__(self):
        return f"<{self.kind} {self.id} {self.meta['name']}>"


def ingest(path, kind="other", name=None):
    path = Path(path)
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    digest = sha256(path)
    aid = digest[:16]
    d = DATA / "assets" / aid
    if (d / "meta.json").exists():
        a = Asset(d)
        ff.log(f"{path.name}: already stored as {a.kind} {aid} ({a.meta['name']})")
        if kind != a.kind and kind != "other":
            a.meta["kind"] = a.kind = kind
            (d / "meta.json").write_text(json.dumps(a.meta, indent=1))
        return a
    d.mkdir(parents=True, exist_ok=True)
    stored = d / ("original" + path.suffix.lower())
    try:
        os.link(path, stored)
    except OSError:
        shutil.copy2(path, stored)
    meta = {"id": aid, "kind": kind, "name": name or path.name, "file": stored.name, "sha256": digest,
            "probe": ff.probe(stored), "ingested": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    ff.log(f"{path.name}: stored as {kind} {aid}")
    return Asset(d)


def get(ref):
    """An asset by id (or unique id prefix), or by a file path (hashed)."""
    ref = str(ref)
    root = DATA / "assets"
    if os.path.exists(ref) and not (root / ref).exists():
        aid = sha256(ref)[:16]
        if (root / aid / "meta.json").exists():
            return Asset(root / aid)
        raise KeyError(f"{ref} is not ingested")
    hits = [p for p in root.glob(f"{ref}*") if (p / "meta.json").exists()] if root.exists() else []
    if len(hits) != 1:
        raise KeyError(f"asset {ref!r}: {'no match' if not hits else 'ambiguous'}")
    return Asset(hits[0])


def all_assets(kind=None):
    root = DATA / "assets"
    out = [Asset(p) for p in sorted(root.glob("*")) if (p / "meta.json").exists()] if root.exists() else []
    return [a for a in out if kind is None or a.kind == kind]
