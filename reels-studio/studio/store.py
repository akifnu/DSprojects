"""File-backed storage: projects, immutable revisions, jobs, presets and the media library.

Layout (under $STUDIO_DATA, default reels-studio/data):
  projects/<pid>/project.json        metadata + current revision pointer
  projects/<pid>/source/<name>       immutable original (read-only after import)
  projects/<pid>/proxy.mp4           browser playback proxy
  projects/<pid>/transcript.json     words with stable ids + sentences
  projects/<pid>/revisions/<n>.json  immutable composition snapshots
  projects/<pid>/exports/            rendered MP4s
  jobs/<jid>.json                    durable job records
  library/presets.json               lookbook (cross-project)
  library/assets.json + assets/      reusable media and sounds
"""
import json
import os
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(os.environ.get("STUDIO_DATA", Path(__file__).resolve().parent.parent / "data"))
_lock = threading.RLock()


def now():
    return round(time.time(), 3)


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


# ------------------------------------------------------------------ projects

def pdir(pid):
    if not pid.startswith("prj_") or "/" in pid or ".." in pid:
        raise KeyError(pid)
    return ROOT / "projects" / pid


def list_projects():
    out = []
    for f in sorted((ROOT / "projects").glob("*/project.json")):
        p = read(f)
        if p:
            out.append(p)
    return sorted(out, key=lambda p: -p["updated"])


def get_project(pid):
    p = read(pdir(pid) / "project.json")
    if p is None:
        raise KeyError(pid)
    return p


def save_project(p):
    p["updated"] = now()
    write(pdir(p["id"]) / "project.json", p)
    return p


def create_project(title):
    p = {"id": new_id("prj"), "title": title or "Untitled reel", "status": "empty", "current": None,
         "source": None, "error": None, "created": now(), "updated": now()}
    pdir(p["id"]).mkdir(parents=True, exist_ok=True)
    return save_project(p)


def get_revision(pid, n):
    r = read(pdir(pid) / "revisions" / f"{int(n)}.json")
    if r is None:
        raise KeyError(n)
    return r


def list_revisions(pid):
    revs = [read(f) for f in (pdir(pid) / "revisions").glob("*.json")]
    return sorted(({k: r[k] for k in ("id", "parent", "actor", "note", "created")} for r in revs if r),
                  key=lambda r: r["id"])


class Conflict(Exception):
    def __init__(self, current):
        self.current = current


def commit_revision(pid, data, base, actor="user", note=""):
    """Append an immutable revision. `base` must equal the project's current revision
    (stale writes raise Conflict); base=None skips the guard (system writes)."""
    with _lock:
        p = get_project(pid)
        if base is not None and base != p["current"]:
            raise Conflict(p["current"])
        n = 0 if p["current"] is None else max(r["id"] for r in list_revisions(pid)) + 1
        rev = dict(data, id=n, parent=p["current"], actor=actor, note=note, created=now())
        write(pdir(pid) / "revisions" / f"{n}.json", rev)
        p["current"] = n
        save_project(p)
        return rev


def transcript(pid):
    return read(pdir(pid) / "transcript.json")


# ------------------------------------------------------------------ jobs

def job_path(jid):
    if not jid.startswith("job_") or "/" in jid:
        raise KeyError(jid)
    return ROOT / "jobs" / f"{jid}.json"


def save_job(j):
    j["updated"] = now()
    write(job_path(j["id"]), j)
    return j


def get_job(jid):
    j = read(job_path(jid))
    if j is None:
        raise KeyError(jid)
    return j


def list_jobs(pid=None):
    jobs = [read(f) for f in (ROOT / "jobs").glob("*.json")]
    return sorted((j for j in jobs if j and (pid is None or j.get("project") == pid)), key=lambda j: -j["created"])


# ------------------------------------------------------------------ library

def presets():
    return read(ROOT / "library" / "presets.json", [])


def save_preset(preset):
    with _lock:
        items = presets()
        old = next((x for x in items if x["id"] == preset.get("id")), None)
        if old:
            old.update(name=preset.get("name", old["name"]), params=preset["params"],
                       version=old["version"] + 1, updated=now())
            preset = old
        else:
            preset = {"id": new_id("pre"), "name": preset.get("name") or "Untitled look",
                      "category": preset.get("category", "caption"), "params": preset["params"],
                      "version": 1, "created": now(), "updated": now()}
            items.append(preset)
        write(ROOT / "library" / "presets.json", items)
        return preset


def delete_preset(pid):
    with _lock:
        write(ROOT / "library" / "presets.json", [x for x in presets() if x["id"] != pid])


def assets():
    return read(ROOT / "library" / "assets.json", [])


def asset_dir():
    d = ROOT / "library" / "assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_asset(meta):
    with _lock:
        items = assets()
        items.append(meta)
        write(ROOT / "library" / "assets.json", items)
        return meta


def get_asset(aid):
    a = next((x for x in assets() if x["id"] == aid), None)
    if a is None:
        raise KeyError(aid)
    return a


def asset_file(a):
    return asset_dir() / a["file"]


def delete_asset(aid):
    with _lock:
        items = assets()
        a = next((x for x in items if x["id"] == aid), None)
        if a:
            asset_file(a).unlink(missing_ok=True)
        write(ROOT / "library" / "assets.json", [x for x in items if x["id"] != aid])
