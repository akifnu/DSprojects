"""Durable background jobs: source processing and final export. Records live on disk, so
status survives a page refresh; jobs interrupted by a server restart are marked failed and
can be retried (retry never duplicates the source or applied edits)."""
import shutil
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from . import edit, media, render, store

_pool = ThreadPoolExecutor(max_workers=2)
_procs = {}
_cancel = set()


def recover():
    for j in store.list_jobs():
        if j["status"] in ("queued", "working"):
            j.update(status="failed", error="Interrupted by a server restart — retry to continue.")
            store.save_job(j)


def submit(kind, project, params=None):
    j = store.save_job({"id": store.new_id("job"), "kind": kind, "project": project, "params": params or {},
                        "status": "queued", "progress": 0, "stage": "queued", "error": None, "result": None,
                        "created": store.now()})
    _pool.submit(_run, j["id"])
    return j


def retry(jid):
    j = store.get_job(jid)
    if j["status"] not in ("failed", "cancelled"):
        return j
    j.update(status="queued", error=None, progress=0, stage="queued")
    store.save_job(j)
    _pool.submit(_run, jid)
    return j


def cancel(jid):
    j = store.get_job(jid)
    if j["status"] in ("queued", "working"):
        _cancel.add(jid)
        p = _procs.get(jid)
        if p:
            p.terminate()
    return j


def _update(j, **kw):
    j.update(kw)
    store.save_job(j)


def _run(jid):
    j = store.get_job(jid)
    if jid in _cancel:
        _update(j, status="cancelled", stage="cancelled")
        return
    _update(j, status="working")
    try:
        {"process": _process, "export": _export}[j["kind"]](j)
        if jid in _cancel:
            raise InterruptedError
        _update(j, status="ready", progress=1, stage="done")
    except InterruptedError:
        _update(j, status="cancelled", stage="cancelled")
    except Exception as e:  # noqa: BLE001 — every failure becomes a visible, retryable state
        traceback.print_exc()
        msg = str(e) if isinstance(e, media.MediaError) else f"{type(e).__name__}: {e}"
        _update(j, status="failed", error=msg[:600])
        if j["kind"] == "process":
            p = store.get_project(j["project"])
            p.update(status="failed", error=msg[:600])
            store.save_project(p)
    finally:
        _cancel.discard(jid)
        _procs.pop(jid, None)


def _process(j):
    pid = j["project"]
    p = store.get_project(pid)
    d = store.pdir(pid)
    src = d / "source" / p["source"]["file"]
    p.update(status="processing", error=None)
    store.save_project(p)

    if not (d / "proxy.mp4").exists():
        _update(j, stage="making playback proxy", progress=0.1)
        media.make_proxy(src, d / "proxy.mp4")
        media.thumb(d / "proxy.mp4", d / "thumb.jpg", min(1.0, p["source"]["duration"] / 2))
    if not (d / "transcript.json").exists():
        _update(j, stage="transcribing speech", progress=0.35)
        if p["source"]["has_audio"]:
            tr = media.transcribe(src, d / "transcript.json")
        else:
            tr = {"words": [], "sentences": [], "lang": None, "timing": None}
        if not tr["words"]:
            tr["note"] = "No speech was detected. You can still trim and style the video manually."
        store.write(d / "transcript.json", tr)
    tr = store.transcript(pid)

    if p["current"] is None:
        _update(j, stage="proposing rough cut", progress=0.85)
        base = edit.default_revision()
        store.commit_revision(pid, base, None, actor="system", note="Imported source (nothing removed)")
        store.commit_revision(pid, edit.rough_cut(tr, base), None, actor="ai",
                              note="AI rough cut: fillers and repeated takes removed, pauses tightened")
    p = store.get_project(pid)
    p.update(status="ready")
    store.save_project(p)


def _export(j):
    pid, n = j["project"], j["params"]["revision"]
    d = store.pdir(pid)
    p = store.get_project(pid)
    rev = store.get_revision(pid, n)  # frozen: later edits create newer revisions
    tr = store.transcript(pid) or {"words": []}
    assets = store.assets()
    comp = edit.compile(tr, rev, p["source"]["duration"], assets)
    blocking, warnings = render.preflight(rev, comp, assets)
    if blocking:
        raise media.MediaError("Export blocked: " + "; ".join(blocking))
    out_dir = d / "exports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{j['id']}.mp4"
    tmp = out_dir / f"{j['id']}.part.mp4"
    ass = out_dir / f"{j['id']}.ass"
    ass.write_text(render.build_ass(comp, rev["caption"], rev["settings"]["w"], rev["settings"]["h"]))
    cmd = render.build_command(d / "source" / p["source"]["file"], p["source"], rev, comp, assets, ass, tmp)
    _update(j, stage="rendering", progress=0.02, result={"warnings": warnings})
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _procs[j["id"]] = proc
    err = []
    threading.Thread(target=lambda: err.extend(proc.stderr), daemon=True).start()
    for line in proc.stdout:
        if line.startswith("out_time_us=") and comp["duration"]:
            try:
                done = int(line.split("=")[1]) / 1e6
                _update(j, progress=round(min(0.97, done / comp["duration"]), 3))
            except ValueError:
                pass
    proc.wait()
    if j["id"] in _cancel:
        tmp.unlink(missing_ok=True)
        raise InterruptedError
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise media.MediaError("Render failed: " + "".join(err[-6:]).strip()[-500:])
    _update(j, stage="validating", progress=0.98)
    info, problems = render.validate_output(tmp, comp, rev["settings"])
    if problems:
        raise media.MediaError("Exported file failed validation: " + "; ".join(problems))
    shutil.move(tmp, out)
    j["result"] = {"file": out.name, "revision": n, "duration": info["duration"], "warnings": warnings,
                   "size": out.stat().st_size}
