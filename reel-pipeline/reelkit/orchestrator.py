"""Top orchestration layer: one job = uploads in → finished reels out.

A job is a small state machine on disk (data/jobs/<id>/job.json). `advance()` runs
every mechanical step itself (ingest, sample analysis, transcription, editing
copies, batch render + QC) and stops only where judgment is needed, returning a
`needs` description of which agent must act and what it must produce:

    intake ─► analyze ─► VERIFY_STYLE ─► transcribe ─► PLAN ─► prepare ─► render ─► REVIEW ─┬─► deliver
                         (sample-analyst)              (planner)                  (qc-reviewer) │
                                                          ▲                                    │
                                                          └──── fixes (≤ max_rounds) ◄──────────┘

Agents report back with `complete(job, step, ...)`; the orchestrator (an agent or
the `reel job` CLI) just calls `advance()` again. Efficiency policy lives here, in
one place (`POLICY`): what is cached and reused, what is decoded, how many renders
run at once, how much review each batch gets, how many fix rounds before escalating,
and which model tier each role uses.
"""
import fcntl
import json
import math
import time
from pathlib import Path

from . import ff, store
from .paths import DATA

POLICY = {
    "sample_max_seconds": 120,      # vertical/square video up to this long = a sample reel
    "max_rounds": 2,                # fix-and-re-render rounds per reel before escalating
    "spot_check": "sqrt",           # passed reels reviewed: ceil(sqrt(n)) of them (+ all warn/fail)
    "prepare": "planned_ranges",    # decode only what the plan uses (4K60 HEVC ≈ 1× real time)
    "render_quality": "final",      # draft renders are not made first: QC catches problems
    "jobs": None,                   # parallel renders: None = batch.concurrency() (calibrated)
    "models": {"orchestrator": "inherit", "sample-analyst": "inherit", "reel-planner": "inherit",
               "qc-reviewer": "sonnet"},
}

JUDGMENT = {
    "verify_style": "sample-analyst",
    "plan": "reel-planner",
    "review": "qc-reviewer",
}
ORDER = ["intake", "analyze", "verify_style", "transcribe", "plan", "prepare", "render", "review", "deliver"]


def _jobs_dir():
    d = DATA / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(job_id):
    return _jobs_dir() / job_id / "job.json"


def load(job_id):
    if job_id == "latest":
        ids = sorted(p.parent.name for p in _jobs_dir().glob("*/job.json"))
        if not ids:
            raise SystemExit("no jobs")
        job_id = ids[-1]
    return json.loads(_path(job_id).read_text())


def save(job):
    job["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p = _path(job["id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=1, ensure_ascii=False))
    tmp.replace(p)
    return job


def _log(job, msg):
    job.setdefault("log", []).append({"at": time.strftime("%H:%M:%S"), "msg": msg})
    ff.log(f"job {job['id']}: {msg}")


def classify(path):
    """sample (short vertical/square reel) | source (long-form) | music (audio only) | image."""
    p = Path(path)
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        return "image"
    info = ff.probe(p)
    if not info["has_video"]:
        return "music"
    vertical = info["height"] >= info["width"]
    return "sample" if vertical and info["duration"] <= POLICY["sample_max_seconds"] else "source"


def create(files, count=None, lang="en", duration=None, notes="", kinds=None, name=None):
    """New job from uploaded files. kinds: optional {path: kind} overrides."""
    jid = name or time.strftime("j%Y%m%d-%H%M%S")
    job = {"id": jid, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "intake",
           "request": {"count": count, "lang": lang, "duration": duration, "notes": notes},
           "files": [{"path": str(Path(f).resolve()), "kind": (kinds or {}).get(f) or None} for f in files],
           "inputs": {"samples": [], "sources": [], "music": [], "sfx": [], "images": []},
           "steps": {}, "rounds": 0, "policy": POLICY, "log": []}
    _log(job, f"created with {len(files)} file(s)")
    return save(job)


def _done(job, step, **data):
    job["steps"][step] = {"done": True, "at": time.strftime("%H:%M:%S"), **data}


def _need(job, step, **detail):
    agent = JUDGMENT[step]
    job["status"] = f"needs:{step}"
    job["needs"] = {"step": step, "agent": agent, "model": POLICY["models"].get(agent, "inherit"), **detail}
    save(job)
    return job["needs"]


def complete(job_id, step, note="", **data):
    """An agent finished a judgment step."""
    job = load(job_id)
    if step not in JUDGMENT:
        raise SystemExit(f"{step} is not a judgment step ({', '.join(JUDGMENT)})")
    if step == "review":
        fixes = data.get("fixes") or []
        job["rounds"] += 1 if fixes else 0
        job["fixes"] = fixes
        if fixes and job["rounds"] <= POLICY["max_rounds"]:
            for k in ("plan", "prepare", "render"):   # planner applies the fixes, then re-render those reels
                job["steps"].pop(k, None)
            job["only"] = sorted({f["reel"] for f in fixes if f.get("reel")}) or None
            _log(job, f"review round {job['rounds']}: {len(fixes)} fix(es) → re-plan {job['only'] or 'all'}")
            job.pop("needs", None)
            return save(job)
    _done(job, step, note=note, **{k: v for k, v in data.items() if k != "fixes"})
    job.pop("needs", None)
    _log(job, f"{step} done{': ' + note if note else ''}")
    return save(job)


def _style_path(job):
    return str(store.get(job["inputs"]["samples"][0]).cache("style.json"))


def _plan_path(job):
    return DATA / "batches" / job["id"] / "plan.json"


def _planned_ranges(job, plan):
    """Source seconds each source needs, from the plan's sentence ids / explicit times."""
    from . import source
    ranges = {}
    for reel in plan.get("reels", []):
        segs = list(reel.get("segments", [])) + ([reel["hook"]] if reel.get("hook") else [])
        for seg in segs:
            key = seg.get("src", "S")
            aid = plan["sources"][key]
            if "sentences" in seg:
                tr = source.transcript(store.get(aid), job["request"]["lang"])
                a, b = seg["sentences"][0], seg["sentences"][-1]
                ranges.setdefault(aid, []).append((tr["sentences"][a]["s"], tr["sentences"][b]["e"]))
            else:
                ranges.setdefault(aid, []).append((seg["s"], seg["e"]))
    return ranges


def review_set(manifest):
    """Which reels the reviewer must look at: every fail/warn/error + a spot check of passes."""
    reels = manifest.get("reels", [])
    bad = [r for r in reels if r.get("verdict") != "pass"]
    good = [r for r in reels if r.get("verdict") == "pass"]
    k = min(len(good), math.ceil(math.sqrt(len(reels)))) if POLICY["spot_check"] == "sqrt" else len(good)
    step = max(1, len(good) // max(1, k)) if k else 1
    return bad + good[::step][:k]


def advance(job_id):
    """Run mechanical steps until a judgment step (or the end). Returns the job.
    One advance per job at a time: a second caller gets the job back unchanged."""
    job = load(job_id)
    lock = open(_path(job["id"]).with_name(".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        job["busy"] = True
        return job
    try:
        return _advance(job)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _advance(job):
    while True:
        st = job["steps"]
        if "intake" not in st:
            for f in job["files"]:
                kind = f["kind"] or classify(f["path"])
                a = store.ingest(f["path"], {"image": "image"}.get(kind, kind))
                f.update(kind=kind, asset=a.id)
                bucket = {"sample": "samples", "source": "sources", "music": "music", "sfx": "sfx", "image": "images"}[kind]
                if a.id not in job["inputs"][bucket]:
                    job["inputs"][bucket].append(a.id)
            _done(job, "intake", **{k: len(v) for k, v in job["inputs"].items()})
            _log(job, f"intake: {', '.join(f'{len(v)} {k}' for k, v in job['inputs'].items() if v)}")
            if not job["inputs"]["samples"]:
                job["status"] = "blocked: no sample reel uploaded"
                return save(job)
            save(job)
        elif "analyze" not in st:
            from . import sample
            for sid in job["inputs"]["samples"]:
                sample.analyze(sid, job["request"]["lang"])        # cached per sample hash
            _done(job, "analyze")
            save(job)
        elif "verify_style" not in st:
            s = store.get(job["inputs"]["samples"][0])
            style = json.loads(s.cache("style.json").read_text())
            if style.get("review", {}).get("verified"):          # verified in an earlier job: reuse
                _done(job, "verify_style", note="already verified")
                save(job)
                continue
            _need(job, "verify_style", style=str(s.cache("style.json")), review=str(s.cache("review.jpg")),
                  extra_samples=job["inputs"]["samples"][1:])
            return job
        elif "transcribe" not in st:
            if not job["inputs"]["sources"]:
                job["status"] = "blocked: no long-form source uploaded"
                return save(job)
            from . import source
            for aid in job["inputs"]["sources"]:
                tr = source.transcript(store.get(aid), job["request"]["lang"])
                from . import asr
                store.get(aid).cache(f"transcript.{job['request']['lang']}.txt").write_text(asr.compact(tr))
            _done(job, "transcribe")
            save(job)
        elif "plan" not in st:
            plan_p = _plan_path(job)
            if not plan_p.exists():
                plan_p.parent.mkdir(parents=True, exist_ok=True)
                keys = [chr(ord("A") + i) for i in range(len(job["inputs"]["sources"]))]
                plan_p.write_text(json.dumps({
                    "name": job["id"], "style": job["inputs"]["samples"][0],
                    "output": {"width": 720, "height": 1280, "fps": 30, "quality": POLICY["render_quality"],
                               "lang": job["request"]["lang"]},          # loudness: the sample's, unless set
                    "sources": dict(zip(keys, job["inputs"]["sources"])),
                    "music": {"asset": job["inputs"]["music"][0], "gain_db": -18} if job["inputs"]["music"] else None,
                    "sfx": None, "reels": []}, indent=1))
            transcripts = [str(store.get(a).cache(f"transcript.{job['request']['lang']}.txt"))
                           for a in job["inputs"]["sources"]]
            _need(job, "plan", plan=str(plan_p), style=_style_path(job), transcripts=transcripts,
                  count=job["request"]["count"], target_duration=job["request"]["duration"],
                  fixes=job.get("fixes"), only=job.get("only"))
            return job
        elif "prepare" not in st:
            from . import source
            plan = json.loads(_plan_path(job).read_text())
            if not plan.get("reels"):
                job["steps"].pop("plan", None)
                _log(job, "plan has no reels; back to planning")
                continue
            for aid, rngs in _planned_ranges(job, plan).items():
                source.prepare(store.get(aid), rngs if POLICY["prepare"] == "planned_ranges" else None)
            _done(job, "prepare")
            save(job)
        elif "render" not in st:
            from . import batch
            job["status"] = "rendering"
            save(job)
            manifest = batch.run(str(_plan_path(job)), jobs=POLICY["jobs"], draft=POLICY["render_quality"] == "draft",
                                 only=job.get("only"))
            counts = {}
            for r in manifest.get("reels", []):
                counts[r.get("verdict")] = counts.get(r.get("verdict"), 0) + 1
            _done(job, "render", counts=counts)
            job["manifest"] = str(DATA / "batches" / job["id"] / "manifest.json")
            _log(job, f"render: {counts}")
            save(job)
        elif "review" not in st:
            manifest = json.loads(Path(job["manifest"]).read_text())
            todo = review_set(manifest)
            if job["rounds"] >= POLICY["max_rounds"] and any(r.get("verdict") == "fail" for r in todo):
                _log(job, "fix rounds exhausted: failing reels are escalated to the user")
            _need(job, "review", manifest=job["manifest"],
                  reels=[{"id": r["id"], "verdict": r.get("verdict"), "issues": r.get("issues"),
                          "file": r.get("file"), "sheet": r.get("sheet")} for r in todo],
                  rounds_left=max(0, POLICY["max_rounds"] - job["rounds"]))
            return job
        else:
            manifest = json.loads(Path(job["manifest"]).read_text())
            out = DATA / "batches" / job["id"] / "out"
            job["deliver"] = {"files": sorted(str(p) for p in out.glob("*.mp4")),
                              "review": sorted(str(p) for p in (out / "_review").glob("*.mp4")) if (out / "_review").exists() else [],
                              "contact_sheet": str(out / "contact_sheet.jpg"), "summary": manifest.get("summary")}
            _done(job, "deliver")
            job["status"] = "done"
            job.pop("needs", None)
            _log(job, f"done: {len(job['deliver']['files'])} reel(s) ready")
            return save(job)


def status(job_id):
    job = load(job_id)
    done = [s for s in ORDER if s in job["steps"]]
    return {"id": job["id"], "status": job["status"], "done": done, "needs": job.get("needs"),
            "busy": _busy(job["id"]),
            "rounds": job["rounds"], "inputs": job["inputs"], "deliver": job.get("deliver"),
            "log": job["log"][-8:]}


def _busy(job_id):
    """True while another process is advancing this job (e.g. a long batch render)."""
    p = _path(job_id).with_name(".lock")
    if not p.exists():
        return False
    with open(p, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True
