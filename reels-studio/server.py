"""Reels Studio API + static editor. Run: python3 server.py  (http://localhost:8765)"""
import contextlib
import os
import shutil
import subprocess
from pathlib import Path

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from studio import edit, jobs, media, store

HERE = Path(__file__).resolve().parent
MAX_UPLOAD = int(os.environ.get("STUDIO_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
_idem = {}


def err(status, msg, **extra):
    return JSONResponse({"error": msg, **extra}, status_code=status)


def project_or_404(pid):
    try:
        return store.get_project(pid)
    except KeyError:
        raise HTTPException(404, "Project not found")


def word_ids(tr):
    return [w["id"] for w in (tr or {}).get("words", [])]


def compiled(pid, p, rev):
    tr = store.transcript(pid) or {"words": []}
    return edit.compile(tr, rev, p["source"]["duration"], store.assets())


async def save_upload(upload, dest):
    size = 0
    with open(dest, "wb") as f:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                Path(dest).unlink(missing_ok=True)
                raise HTTPException(413, "File is too large")
            f.write(chunk)
    return size


# ------------------------------------------------------------------ projects

async def projects(request: Request):
    if request.method == "POST":
        body = await request.json()
        return JSONResponse(store.create_project(str(body.get("title", ""))[:120]))
    return JSONResponse(store.list_projects())


async def project(request: Request):
    pid = request.path_params["pid"]
    p = project_or_404(pid)
    if request.method == "PATCH":
        body = await request.json()
        if "title" in body:
            p["title"] = str(body["title"])[:120]
            store.save_project(p)
        return JSONResponse(p)
    if request.method == "DELETE":
        shutil.rmtree(store.pdir(pid), ignore_errors=True)
        return JSONResponse({"deleted": pid})
    out = {"project": p, "transcript": store.transcript(pid), "revisions": store.list_revisions(pid),
           "jobs": store.list_jobs(pid)[:20]}
    if p["current"] is not None:
        rev = store.get_revision(pid, p["current"])
        out.update(revision=rev, compiled=compiled(pid, p, rev))
    return JSONResponse(out)


async def upload_source(request: Request):
    pid = request.path_params["pid"]
    p = project_or_404(pid)
    if p["current"] is not None:
        return err(409, "This project already has an edited source. Start a new project for new footage.")
    form = await request.form()
    up = form.get("file")
    if up is None or not getattr(up, "filename", None):
        return err(400, "No file uploaded")
    name = Path(up.filename).name
    src_dir = store.pdir(pid) / "source"
    shutil.rmtree(src_dir, ignore_errors=True)  # only a never-processed source can be replaced
    for f in ("proxy.mp4", "transcript.json", "transcript.json.asr", "thumb.jpg"):
        (store.pdir(pid) / f).unlink(missing_ok=True)
    src_dir.mkdir(parents=True)
    dest = src_dir / name
    await save_upload(up, dest)
    try:
        info = media.validate_source(dest)
    except media.MediaError as e:
        shutil.rmtree(src_dir, ignore_errors=True)
        p.update(status="empty", error=str(e), source=None)
        store.save_project(p)
        return err(422, str(e))
    os.chmod(dest, 0o444)  # immutable original
    p.update(source=dict(info, file=name, size=dest.stat().st_size), status="processing", error=None)
    if p["title"] == "Untitled reel":
        p["title"] = Path(name).stem[:120]
    store.save_project(p)
    j = jobs.submit("process", pid)
    return JSONResponse({"project": p, "job": j})


async def project_file(request: Request):
    pid, kind = request.path_params["pid"], request.path_params["kind"]
    project_or_404(pid)
    d = store.pdir(pid)
    f = {"proxy": d / "proxy.mp4", "thumb": d / "thumb.jpg"}.get(kind)
    if f is None or not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


async def export_file(request: Request):
    pid, name = request.path_params["pid"], request.path_params["name"]
    project_or_404(pid)
    f = store.pdir(pid) / "exports" / Path(name).name
    if not f.exists() or f.suffix != ".mp4" or ".part" in f.name:
        raise HTTPException(404)
    p = store.get_project(pid)
    return FileResponse(f, filename=f"{p['title'][:60] or 'reel'}-{name}", media_type="video/mp4")


# ------------------------------------------------------------------ revisions

async def commit(request: Request):
    pid = request.path_params["pid"]
    p = project_or_404(pid)
    key = request.headers.get("idempotency-key")
    if key and (pid, key) in _idem:
        return JSONResponse(_idem[(pid, key)])
    body = await request.json()
    tr = store.transcript(pid)
    data = edit.normalize(body.get("data") or {}, word_ids(tr))
    try:
        rev = store.commit_revision(pid, data, body.get("base"), actor="user", note=str(body.get("note", ""))[:120])
    except store.Conflict as c:
        return err(409, "This project changed elsewhere. Reload to continue from the latest revision.", current=c.current)
    out = {"revision": rev, "compiled": compiled(pid, p, rev)}
    if key:
        _idem[(pid, key)] = out
    return JSONResponse(out)


async def compile_only(request: Request):
    pid = request.path_params["pid"]
    p = project_or_404(pid)
    body = await request.json()
    data = edit.normalize(body.get("data") or {}, word_ids(store.transcript(pid)))
    return JSONResponse({"compiled": compiled(pid, p, data)})


async def revision(request: Request):
    pid, n = request.path_params["pid"], request.path_params["n"]
    project_or_404(pid)
    try:
        return JSONResponse(store.get_revision(pid, n))
    except KeyError:
        raise HTTPException(404)


async def roughcut(request: Request):
    """A fresh AI proposal on top of the current revision — a new revision, never an overwrite."""
    pid = request.path_params["pid"]
    p = project_or_404(pid)
    body = await request.json()
    cur = store.get_revision(pid, p["current"])
    data = edit.rough_cut(store.transcript(pid), edit.normalize(cur, word_ids(store.transcript(pid))))
    try:
        rev = store.commit_revision(pid, data, body.get("base"), actor="ai", note="AI rough cut re-run")
    except store.Conflict as c:
        return err(409, "Reload first: the project changed.", current=c.current)
    return JSONResponse({"revision": rev, "compiled": compiled(pid, p, rev)})


async def start_export(request: Request):
    pid = request.path_params["pid"]
    p = project_or_404(pid)
    body = await request.json()
    n = body.get("revision", p["current"])
    store.get_revision(pid, n)
    running = [j for j in store.list_jobs(pid) if j["kind"] == "export" and j["status"] in ("queued", "working")
               and j["params"]["revision"] == n]
    if running:
        return JSONResponse(running[0])
    return JSONResponse(jobs.submit("export", pid, {"revision": n}))


# ------------------------------------------------------------------ jobs

async def job(request: Request):
    try:
        return JSONResponse(store.get_job(request.path_params["jid"]))
    except KeyError:
        raise HTTPException(404)


async def job_action(request: Request):
    jid, action = request.path_params["jid"], request.path_params["action"]
    try:
        return JSONResponse({"retry": jobs.retry, "cancel": jobs.cancel}[action](jid))
    except KeyError:
        raise HTTPException(404)


# ------------------------------------------------------------------ library

async def presets(request: Request):
    if request.method == "POST":
        body = await request.json()
        if not isinstance(body.get("params"), dict):
            return err(400, "params required")
        return JSONResponse(store.save_preset(body))
    return JSONResponse(store.presets())


async def preset_delete(request: Request):
    store.delete_preset(request.path_params["id"])
    return JSONResponse({"ok": True})


async def assets(request: Request):
    if request.method == "GET":
        return JSONResponse(store.assets())
    form = await request.form()
    up = form.get("file")
    if up is None or not getattr(up, "filename", None):
        return err(400, "No file uploaded")
    ext = Path(up.filename).suffix.lower()
    kind = ("gif" if ext == ".gif" else "image" if ext in media.IMAGE_EXT else "video" if ext in media.VIDEO_EXT
            else "audio" if ext in media.AUDIO_EXT else None)
    if kind is None:
        return err(422, "Unsupported media type. Use PNG/JPG/WEBP/GIF images, MP4/MOV video or MP3/WAV/M4A audio.")
    aid = store.new_id("ast")
    fname = f"{aid}{ext}"
    dest = store.asset_dir() / fname
    await save_upload(up, dest)
    try:
        info = media.probe(dest)
    except media.MediaError as e:
        dest.unlink(missing_ok=True)
        return err(422, str(e))
    meta = {"id": aid, "name": str(form.get("name") or Path(up.filename).stem)[:80], "type": kind, "file": fname,
            "duration": info["duration"] if kind in ("video", "audio", "gif") else None,
            "width": info.get("width"), "height": info.get("height"), "has_audio": info["has_audio"],
            "tags": [t.strip() for t in str(form.get("tags") or "").split(",") if t.strip()][:10],
            "rights": str(form.get("rights") or "")[:200], "source": "upload", "created": store.now()}
    return JSONResponse(store.add_asset(meta))


async def asset_item(request: Request):
    aid = request.path_params["id"]
    if request.method == "DELETE":
        store.delete_asset(aid)
        return JSONResponse({"ok": True})
    try:
        a = store.get_asset(aid)
    except KeyError:
        raise HTTPException(404)
    f = store.asset_file(a)
    if not f.exists():
        raise HTTPException(404, "Asset file missing")
    return FileResponse(f)


async def fonts(request: Request):
    out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True).stdout if shutil.which("fc-list") else ""
    fams = sorted({line.split(",")[0].strip() for line in out.splitlines() if line.strip()})
    return JSONResponse([f for f in fams if not f.startswith(("Unifont", "Noto Color"))])


async def index(request: Request):
    return FileResponse(HERE / "static" / "index.html")


routes = [
    Route("/", index),
    Route("/api/projects", projects, methods=["GET", "POST"]),
    Route("/api/projects/{pid}", project, methods=["GET", "PATCH", "DELETE"]),
    Route("/api/projects/{pid}/upload", upload_source, methods=["POST"]),
    Route("/api/projects/{pid}/files/{kind}", project_file),
    Route("/api/projects/{pid}/exports/{name}", export_file),
    Route("/api/projects/{pid}/revisions", commit, methods=["POST"]),
    Route("/api/projects/{pid}/revisions/{n:int}", revision),
    Route("/api/projects/{pid}/compile", compile_only, methods=["POST"]),
    Route("/api/projects/{pid}/roughcut", roughcut, methods=["POST"]),
    Route("/api/projects/{pid}/export", start_export, methods=["POST"]),
    Route("/api/jobs/{jid}", job),
    Route("/api/jobs/{jid}/{action}", job_action, methods=["POST"]),
    Route("/api/presets", presets, methods=["GET", "POST"]),
    Route("/api/presets/{id}", preset_delete, methods=["DELETE"]),
    Route("/api/assets", assets, methods=["GET", "POST"]),
    Route("/api/assets/{id}", asset_item, methods=["GET", "DELETE"]),
    Route("/api/fonts", fonts),
    Mount("/static", StaticFiles(directory=HERE / "static"), name="static"),
]


@contextlib.asynccontextmanager
async def lifespan(app):
    (store.ROOT / "projects").mkdir(parents=True, exist_ok=True)
    jobs.recover()
    media.ensure_bundled_sounds()
    yield


ACCESS_KEY = os.environ.get("STUDIO_ACCESS_KEY", "")


class AccessKey:
    """Optional shared-link protection: with STUDIO_ACCESS_KEY set, the first visit must carry
    ?key=<key>; it is then kept in a cookie. Without the key every request gets 401."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not ACCESS_KEY or scope["type"] != "http":
            return await self.app(scope, receive, send)
        req = Request(scope)
        if req.cookies.get("studio_key") == ACCESS_KEY:
            return await self.app(scope, receive, send)
        if req.query_params.get("key") == ACCESS_KEY:
            rest = req.url.remove_query_params("key").query
            resp = RedirectResponse(req.url.path + (f"?{rest}" if rest else ""), status_code=303)
            resp.set_cookie("studio_key", ACCESS_KEY, httponly=True, samesite="lax", secure=True, max_age=7 * 86400)
            return await resp(scope, receive, send)
        return await PlainTextResponse("Access key required — use the full link you were given.", 401)(scope, receive, send)


app = AccessKey(Starlette(routes=routes, lifespan=lifespan))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("STUDIO_HOST", "127.0.0.1"), port=int(os.environ.get("STUDIO_PORT", "8765")))
