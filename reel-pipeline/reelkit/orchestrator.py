"""Top orchestration layer: one job = uploads in → finished reels out.

A job is a state machine on disk (data/jobs/<id>/job.json). A worker runs every mechanical
step itself (ingest, sample analysis, transcription, storyboard, editing copies, batch
render + QC, delivery) and stops where an agent's judgment or a user's storyboard
approval is needed, publishing a `needs` record and the `need_id` the answer must quote.

    intake → analyze → VERIFY_STYLE → transcribe → PLAN → storyboard → APPROVE_STORYBOARD
                       (analyst)                 (planner)            (user)
      → prepare → render → REVIEW → deliver (or fixes → new plan → new storyboard approval)

Status while open: "intake" (new) | "waiting:sample" | "waiting:source" (inputs missing; `add`
resumes, keeping unaffected work) | "waiting:worker" (ready: `start` continues) | "needs:<step>"
(an agent or user must answer) | "running:<step>". Terminal:
  succeeded        every planned reel delivered, reviewed where required, no open blocking
                   finding, no shortfall against the requested count
  partial          policy allow_partial and ≥ 1 reel accepted; blocked reels, unresolved
                   findings and the shortfall are listed in deliver
  needs_attention  a blocking finding or missing judgment outlived its budget, or nothing
                   was accepted (an empty delivery never succeeds)
  failed           infrastructure / validation error; typed record in job["error"]
  cancelled        cancellation acknowledged, worker stopped

Every mutation (create, add, a worker's step commit, agent completion, cancel, retry) loads
job.json only after taking fsutil.file_lock(<job>/state) and bumps "revision". A worker attempt
is fenced: it commits only while job["attempt"] is still its id and no cancel flag is up.
Completions must quote the open need (need_id, and its revision when given); a repeat of an
accepted answer is a no-op, anything stale or out of order is rejected. Answers are validated
here, not by prompts:
  verify_style  review.verified + contracts.validate_style → immutable styles/style.v<k>.json,
                pinned in the job and in plan.json "style", kept on the sample asset as
                style.verified.json for later jobs of the same measurement
  plan          contracts.validate_plan against the transcripts and the request; in a repair
                round, entries outside `only` must keep their contracts.plan_entry_hash
  approve_storyboard  only an explicit user decision, bound to the current plan and preview;
                      no generic agent completion and no automatic approval timeout
  review        contracts.normalize_review against the assigned reels and render hashes;
                omitted / null reels stay unreviewed and are re-requested (review_retries),
                every finding is kept whatever the budget, repairs are counted per reel
                (max_repairs_per_reel) and per job (max_review_rounds)
Delivery reads only manifest["deliverables"] of the current batch generation (hash-checked),
never a directory listing.

Workers: `start` spawns `python -m reelkit job _work ID --attempt A` in its own session and
returns once attempts/<A>.json is acknowledged (pid, pgid). The worker heartbeats every 10 s,
checks the cancel flag between steps and writes a terminal record (exit, error_record). `wait`
trusts that record, not a busy flag, and reports a worker that died without one; `cancel`
raises the flag, SIGTERMs the worker's process group and SIGKILLs it after 10 s.
events.jsonl gets one line per transition.

Policy (`POLICY`) is snapshotted into each job at creation together with the request's
execution settings (encoder, GPU requirement, CPU fallback, ASR provider); only the snapshot
is used. Jobs written by earlier versions are migrated explicitly on load (_migrate).
"""
import contextlib
import copy
import fcntl
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from . import contracts, ff, fsutil, store
from .paths import DATA, ROOT

POLICY = {
    "version": 3,
    "storyboard": {"width": 360, "frames_per_reel": 3, "max_frames": 240, "timeout_s": 120},
    "sample_max_seconds": 120,      # bare files: vertical/square video up to this long is guessed to be a sample
    "spot_check": "sqrt",           # passed reels reviewed: ceil(sqrt(n)) of them (+ all warn/fail/error); all | none
    "review_retries": 2,            # re-requests of a reel whose review came back null / omitted
    "max_repairs_per_reel": 2,      # fix → re-plan → re-render attempts per reel before it is blocked
    "max_review_rounds": 6,         # repair rounds per job, across reels (cost / time cap)
    "max_rejections": 5,            # invalid answers to one need before the job moves on without it
    "stage_retries": 1,             # re-runs of a mechanical step after a transient failure
    "allow_partial": True,          # deliver the accepted reels when others stay blocked
    "prepare": "planned_ranges",    # decode only what the plan uses (4K60 HEVC ≈ 1× real time)
    "render_quality": "final",      # draft renders are not made first: QC catches problems
    "jobs": None,                   # parallel renders: None = batch.concurrency() (calibrated)
    "encoder": "auto",              # auto | cpu | nvenc, resolved by readiness before rendering
    "require_gpu": False,           # true: no NVENC → the job fails before expensive work
    "allow_cpu_fallback": True,     # auto / nvenc may fall back to x264 (recorded with the reason)
    "asr_provider": "cpu",          # cpu | cuda (sherpa-onnx provider)
    "reel_timeout_s": 1800,         # backstop per reel render + QC inside the batch
    "models": {"orchestrator": "inherit", "sample-analyst": "inherit", "reel-planner": "inherit",
               "qc-reviewer": "sonnet"},
}

JUDGMENT = {
    "verify_style": "sample-analyst",
    "plan": "reel-planner",
    "review": "qc-reviewer",
}
ORDER = ["intake", "analyze", "verify_style", "transcribe", "plan", "storyboard", "approve_storyboard",
         "prepare", "render", "review", "deliver"]
TERMINAL = ("succeeded", "partial", "needs_attention", "failed", "cancelled")
ROLES = {"sample": "samples", "source": "sources", "music": "music", "sfx": "sfx", "image": "images"}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
SCHEMA = 2
LOCK_TIMEOUT = 60       # s to wait for the state lock (only ever held for short transactions)
ACK_TIMEOUT = 60        # s for a spawned worker to acknowledge ownership
HEARTBEAT_S = 10
CANCEL_GRACE_S = 10     # SIGTERM → SIGKILL
# an input that cannot be ingested is recorded on its file entry; anything else fails the step
_BAD_INPUT = (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, ValueError, KeyError,
              subprocess.CalledProcessError, getattr(ff, "FFError", ValueError))


class JobError(RuntimeError):
    """A request the job cannot honour. .errors lists every reason, .code is machine-readable."""

    def __init__(self, msg, errors=None, code="error"):
        super().__init__(msg)
        self.errors = [e for e in (errors or []) if e is not None]
        self.code = code


class Rejected(JobError):
    """An agent's answer was refused: stale, out of order or invalid. The need stays open."""


class _Interrupted(BaseException):
    """Cancel flag or signal. BaseException so library `except Exception` blocks let it through."""

    def __init__(self, why):
        super().__init__(why)
        self.why = why


class _Superseded(BaseException):
    """Another attempt owns the job now: this worker must not commit anything."""


# ------------------------------------------------------------------ paths and small helpers

def _jobs_dir():
    d = DATA / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_id(x, what="job id"):
    if not isinstance(x, str) or x == "latest" or not ID_RE.fullmatch(x):
        raise JobError(f"invalid {what} {x!r}", code="invalid_id")
    return x


def _dir(jid):
    return _jobs_dir() / _check_id(jid)


def _path(jid):
    return _dir(jid) / "job.json"


def _att_path(jid, att):
    return _dir(jid) / "attempts" / f"{_check_id(att, 'attempt id')}.json"


def _cancel_flag(jid):
    return _dir(jid) / "cancel"


def _plan_path(job):
    return DATA / "batches" / job["id"] / "plan.json"


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _rj(path):
    """JSON at path; None when absent or unreadable (never raises)."""
    return fsutil.read_json(path) if path and Path(path).exists() else None


def _rq(job):
    return (job.get("request") or {}).get("request") or {}


def _lang(job):
    return _rq(job).get("lang") or "en"


def _compact(f):
    return {k: f.get(k) for k in ("id", "reel_id", "code", "severity", "what", "how", "status")}


def _log_tail(rec, n=2000):
    p = (rec or {}).get("log")
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except (OSError, TypeError):
        return None


# ------------------------------------------------------------------ events and log

def _append_event(jid, rec):
    try:
        with open(_dir(jid) / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        ff.log(f"job {jid}: event not recorded ({e})")


def _emit(jid, event, attempt=None, stage=None, **kw):
    """An event outside a state transaction (worker lifecycle): no revision of its own."""
    _append_event(jid, {"at": _now(), "ts": round(time.time(), 3), "job": jid, "stage": stage, "attempt": attempt,
                        "revision": None, "event": event, **kw})


def _ev(job, event, **kw):
    """Queue an event; it is written with the new revision once the transaction commits."""
    job.setdefault("_events", []).append({"event": event, **kw})


def _log(job, msg, event="log", **kw):
    job.setdefault("log", []).append({"at": _now(), "msg": msg})
    ff.log(f"job {job['id']}: {msg}")
    _ev(job, event, msg=msg, **kw)


# ------------------------------------------------------------------ state: load, save, transactions

def _blank(jid):
    return {"id": jid, "schema": SCHEMA, "created": _now(), "created_ts": round(time.time(), 3), "updated": None,
            "revision": 0, "status": "intake", "terminal": False, "policy": copy.deepcopy(POLICY),
            "request": None, "request_hash": None, "files": [], "inputs": {b: [] for b in ROLES.values()},
            "inputs_rev": 0, "steps": {}, "stage_attempts": {}, "waiting": [], "needs": None, "need_seq": 0,
            "completions": {}, "style_version": None, "style_seq": 0, "style_work": None, "plan": None,
            "plan_generation": 0, "render": None, "render_seq": 0, "encoder": None,
            "review": {"ledger": {}, "attempts_by_reel": {}, "rounds": 0, "notes": [], "errors": []},
            "findings": [], "repair": None, "repairs": [], "attempt": None, "attempt_seq": 0,
            "storyboard": None, "storyboard_approval": None, "storyboard_feedback": [],
            "error": None, "errors": [], "attention": None, "deliver": None, "log": []}


def _migrate_policy(p):
    """A policy snapshot in the current version. New defaults reach an old job only through this
    function, explicitly, and the snapshot records that they did."""
    p = copy.deepcopy(p) if isinstance(p, dict) else {}
    if "version" not in p:          # version 1: the module POLICY as schema-1 jobs saved it
        new = copy.deepcopy(POLICY)
        for k in ("sample_max_seconds", "spot_check", "prepare", "render_quality", "jobs"):
            if k in p:
                new[k] = p[k]
        if isinstance(p.get("models"), dict):
            new["models"] = {**new["models"], **p["models"]}
        if isinstance(p.get("max_rounds"), int):
            new["max_repairs_per_reel"] = p["max_rounds"]
        new["encoder"] = "cpu"      # what version 1 executed
        new["migrated_from"] = 1
        return new
    missing = [k for k in POLICY if k not in p]
    for k in missing:
        p[k] = copy.deepcopy(POLICY[k])
    if missing:
        p["filled_defaults"] = sorted(set(p.get("filled_defaults") or []) | set(missing))
    if p.get("version", 0) < POLICY["version"]:
        p["migrated_from"] = p.get("version")
        p["version"] = POLICY["version"]
    return p


def _migrate(job):
    """A job.json written by an earlier version, brought to SCHEMA in memory (the next
    transaction saves it). Old steps are dropped except a completed intake: every later step
    re-validates (cached work is reused) and judgment steps are asked again."""
    if int(job.get("schema") or 1) >= SCHEMA:
        job["policy"] = _migrate_policy(job.get("policy"))
        for k, default in (("storyboard", None), ("storyboard_approval", None), ("storyboard_feedback", [])):
            job.setdefault(k, copy.deepcopy(default))
        return job
    jid = job["id"]
    old_req = job.get("request") if isinstance(job.get("request"), dict) else {}
    files = [f for f in job.get("files") or [] if isinstance(f, dict) and f.get("path")]
    for f in files:
        role = f.pop("kind", None) or f.get("role")
        f["role"] = role if role in ROLES else None
        f["role_source"] = f.get("role_source") or "legacy"
    pol = _migrate_policy(job.get("policy"))
    ex = {k: pol[k] for k in ("encoder", "require_gpu", "allow_cpu_fallback", "asr_provider")}
    req = {"schema_version": 1, "assets": [{"path": f["path"], "role": f["role"]} for f in files if f["role"]],
           "request": {"count": old_req.get("count"), "duration_seconds": old_req.get("duration"),
                       "lang": old_req.get("lang") or "en", "notes": old_req.get("notes") or ""},
           "execution": ex}
    old_status = str(job.get("status") or "")
    steps = job.get("steps") if isinstance(job.get("steps"), dict) else {}
    new = _blank(jid)
    for k in ("created", "updated", "log", "deliver"):
        if k in job:
            new[k] = job[k]
    try:
        new["created_ts"] = time.mktime(time.strptime(str(job.get("created")), "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, OverflowError):
        new["created_ts"] = 0
    inputs = job.get("inputs") if isinstance(job.get("inputs"), dict) else {}
    new["inputs"] = {b: list(inputs.get(b) or []) for b in ROLES.values()}
    new.update(files=files, request=req, request_hash=fsutil.sha256_json(req), policy=pol,
               revision=int(job.get("revision") or 0),
               legacy={k: job[k] for k in ("status", "rounds", "fixes", "only", "manifest", "needs", "steps") if k in job})
    if "intake" in steps and files and all(f.get("asset") for f in files):
        new["steps"]["intake"] = steps["intake"]
    if old_status == "done":
        ok = bool((job.get("deliver") or {}).get("files"))
        new.update(status="succeeded" if ok else "needs_attention", terminal=True,
                   attention=None if ok else "migrated: the earlier version reported done without files")
    _log(new, f"migrated from schema 1 (status was {old_status!r})", "migrated")
    return new


def _load(jid):
    p = _path(jid)
    job = _rj(p)
    if not isinstance(job, dict):
        if not p.exists():
            raise JobError(f"no job {jid}", code="no_job")
        job = _rj(p.with_name("job.prev.json"))
        if not isinstance(job, dict) or job.get("id") != jid:
            raise JobError(f"job {jid}: job.json is unreadable and no previous revision survives "
                           f"({p.with_name('job.prev.json')})", code="corrupt_state")
        _log(job, "job.json was unreadable: continuing from the previous revision (job.prev.json)", "state_recovered")
    if job.get("id") != jid:
        raise JobError(f"job {jid}: job.json belongs to {job.get('id')!r}", code="corrupt_state")
    return _migrate(job)


def _save(job):
    jid = job["id"]
    p = _path(jid)
    cur = _rj(p)
    if isinstance(cur, dict):           # kept so a damaged job.json is recoverable
        fsutil.atomic_write_json(p.with_name("job.prev.json"), cur)
    events = job.pop("_events", [])
    job.pop("_nosave", None)
    job["revision"] = int(job.get("revision") or 0) + 1
    job["updated"] = _now()
    if job.get("needs") and job["needs"].get("revision") is None:
        job["needs"]["revision"] = job["revision"]
    job["log"] = (job.get("log") or [])[-300:]
    fsutil.atomic_write_json(p, {k: v for k, v in job.items() if not k.startswith("_")})
    stage = job["status"].split(":", 1)[-1]
    for ev in events:
        _append_event(jid, {"at": _now(), "ts": round(time.time(), 3), "job": jid, "stage": ev.pop("stage", None) or stage,
                            "attempt": job.get("attempt"), "revision": job["revision"], **ev})
    return job


@contextlib.contextmanager
def _txn(jid, attempt=None, save=True):
    """Load → mutate → save under the job's state lock. With `attempt` the caller must still own
    the job (fencing) and it must not have been cancelled. Nothing is saved if the body raises
    or sets job["_nosave"]."""
    if not _dir(jid).is_dir():                  # never let the lock file create a job directory
        raise JobError(f"no job {jid}", code="no_job")
    with fsutil.file_lock(_dir(jid) / "state", timeout=LOCK_TIMEOUT):
        job = _load(jid)
        if attempt is not None:
            if job.get("attempt") != attempt:
                raise _Superseded(f"job {jid} now belongs to attempt {job.get('attempt')}")
            if not job["terminal"] and _cancel_flag(jid).exists():
                raise _Interrupted("cancel")
        yield job
        if save and not job.get("_nosave"):
            _save(job)


def resolve(ref):
    """A job id; "latest" = the most recently created job (resolve once, at the CLI boundary)."""
    if ref != "latest":
        _check_id(ref)
        if not _path(ref).exists():
            raise JobError(f"no job {ref}", code="no_job")
        return ref
    jobs = list_jobs()
    if not jobs:
        raise JobError("no jobs", code="no_job")
    return jobs[-1]["id"]


def list_jobs():
    out = []
    for p in _jobs_dir().glob("*/job.json"):
        j = _rj(p)
        if isinstance(j, dict) and j.get("id") == p.parent.name:
            out.append({"id": j["id"], "status": j.get("status"),
                        "terminal": bool(j.get("terminal")) or j.get("status") == "done",
                        "revision": j.get("revision"), "created": j.get("created"), "updated": j.get("updated"),
                        "created_ts": j.get("created_ts") or 0})
    return sorted(out, key=lambda r: (str(r["created"] or ""), r["created_ts"], r["id"]))


# ------------------------------------------------------------------ requests and inputs

def classify(path, sample_max_seconds=POLICY["sample_max_seconds"]):
    """Role guess for a file given without one: sample (short vertical/square video) | source
    (long-form) | music (audio only) | image. Only a suggestion: explicit roles always win, and a
    short vertical *source* has to be declared as one."""
    p = Path(path)
    if p.suffix.lower() in IMAGE_EXT:
        return "image"
    info = ff.probe(p)
    if not info.get("has_video"):
        if not info.get("has_audio"):
            raise ValueError(f"{p.name}: no video or audio stream")
        return "music"
    vertical = info["height"] >= info["width"]
    return "sample" if vertical and info["duration"] <= sample_max_seconds else "source"


def _default_execution():
    return {"encoder": POLICY["encoder"], "require_gpu": POLICY["require_gpu"],
            "allow_cpu_fallback": POLICY["allow_cpu_fallback"],
            "asr_provider": os.environ.get("REELKIT_ASR_PROVIDER") or POLICY["asr_provider"]}


def _assets(entries, sample_max):
    """([{path, role, role_source, suggested_role?}], problems). Paths are made absolute; a file
    without a role gets the heuristic guess, recorded as such; a file also given with an explicit
    role is not guessed again."""
    rows, errors = [], []
    for i, a in enumerate(entries):
        a = {"path": a} if isinstance(a, (str, os.PathLike)) else a
        if not isinstance(a, dict) or not a.get("path"):
            errors.append(f"assets[{i}]: an object with a path is required")
            continue
        rows.append((Path(str(a["path"])).expanduser().resolve(), a.get("role")))
    explicit = {str(p) for p, role in rows if role}
    out, seen = [], set()
    for p, role in rows:
        if not role and str(p) in explicit:
            continue
        e = {"path": str(p), "role": role, "role_source": "explicit" if role else "heuristic"}
        if role and role not in ROLES:
            errors.append(f"{p}: role {role!r} is not one of {', '.join(ROLES)}")
        if not p.is_file():
            errors.append(f"{p}: no such file")
        elif not role:
            try:
                e["role"] = e["suggested_role"] = classify(p, sample_max)
            except Exception as ex:
                errors.append(f"{p}: no role given and none can be guessed ({type(ex).__name__}: {ex}); "
                              "declare it with --sample/--source/--music/--sfx/--image")
        if (e["path"], e["role"]) not in seen:
            seen.add((e["path"], e["role"]))
            out.append(e)
    return out, errors


def _build_request(base, files, roles, count, duration, lang, notes, execution, sample_max):
    """Merge request.json, role-tagged files, bare files and CLI overrides into a validated,
    normalised request. Returns (request, assets, errors, warnings)."""
    if base is not None and not isinstance(base, dict):
        return None, [], ["request: a JSON object is required"], []
    base = copy.deepcopy(base or {})
    errors, entries = [], []
    if isinstance(base.get("assets", []), list):
        entries = list(base.get("assets") or [])
    else:
        errors.append("assets: a list is required")
    for role, paths in (roles or {}).items():
        entries += [{"path": p, "role": role} for p in paths or []]
    entries += [{"path": p} for p in files or []]
    assets, errs = _assets(entries, sample_max)
    errors += errs
    r = base.get("request") if isinstance(base.get("request"), dict) else {}
    r = {"count": r.get("count"), "duration_seconds": r.get("duration_seconds"), "lang": r.get("lang") or "en",
         "notes": r.get("notes") if r.get("notes") is not None else ""}
    for k, v in (("count", count), ("duration_seconds", duration), ("lang", lang), ("notes", notes)):
        if v is not None:
            r[k] = v
    ex = _default_execution()
    if isinstance(base.get("execution"), dict):
        ex.update({k: v for k, v in base["execution"].items() if k in ex})
    ex.update({k: v for k, v in (execution or {}).items() if k in ex and v is not None})
    req = {"schema_version": base.get("schema_version", 1),
           "assets": [{"path": a["path"], "role": a["role"]} for a in assets if a.get("role")],
           "request": r, "execution": ex}
    warnings = []
    try:
        v = contracts.validate_request(req)
        errors += list(v.get("errors") or [])
        warnings += list(v.get("warnings") or [])
    except Exception as e:
        errors.append(f"request validation failed: {type(e).__name__}: {e}")
    if ex.get("require_gpu") and ex.get("encoder") == "cpu":
        errors.append("execution: require_gpu needs encoder nvenc or auto")
    return req, assets, errors, warnings


def _new_id(name=None):
    root = _jobs_dir()
    if name is not None:
        _check_id(name, "job name")
        try:
            os.mkdir(root / name)
        except FileExistsError:
            raise JobError(f"job {name} already exists", code="exists") from None
        return name
    for _ in range(50):
        jid = time.strftime("j%Y%m%d-%H%M%S-") + secrets.token_hex(3)
        try:
            os.mkdir(root / jid)
            return jid
        except FileExistsError:
            continue
    raise JobError("could not allocate a job id", code="exists")


def create(files=(), count=None, lang=None, duration=None, notes=None, kinds=None, name=None, roles=None,
           request=None, execution=None):
    """New job. files: paths without a role (guessed, recorded as a suggestion); roles:
    {role: [paths]}; request: a request.json dict {schema_version, assets, request, execution};
    the keyword arguments override its fields. Everything is validated before any expensive
    work; missing inputs just make the job wait. Returns the job."""
    files = list(files or [])
    roles = {k: list(v or []) for k, v in (roles or {}).items()}
    for path, kind in (kinds or {}).items():        # legacy {path: kind} overrides
        roles.setdefault(kind, []).append(path)
        files = [f for f in files if f != path]
    req, assets, errors, warnings = _build_request(request, files, roles, count, duration, lang, notes, execution,
                                                   POLICY["sample_max_seconds"])
    if name is not None and (not isinstance(name, str) or name == "latest" or not ID_RE.fullmatch(name)):
        errors.append(f"name {name!r}: letters, digits, '.', '_', '-' (max 64), not 'latest'")
    if errors:
        raise JobError(f"invalid request ({len(errors)} problem(s))", errors, code="invalid_request")
    jid = _new_id(name)
    try:
        job = _blank(jid)
        job["policy"].update(req["execution"])
        job.update(request=req, request_hash=fsutil.sha256_json(req), request_warnings=warnings,
                   files=[{**a, "added_rev": 1} for a in assets])
        with fsutil.file_lock(_dir(jid) / "state", timeout=LOCK_TIMEOUT):
            fsutil.atomic_write_json(_dir(jid) / "request.json", req)
            _log(job, f"created with {len(assets)} file(s)" +
                 (": " + ", ".join(f"{Path(a['path']).name} ({a['role']})" for a in assets) if assets else ""),
                 "created", request_hash=job["request_hash"])
            for a in assets:
                if a["role_source"] == "heuristic":
                    _log(job, f"{Path(a['path']).name}: role guessed as {a['role']} (a suggestion: declare it to "
                              "override)", "role_suggested", path=a["path"], role=a["role"])
            _save(job)
    except BaseException:
        if not _path(jid).exists():
            shutil.rmtree(_dir(jid), ignore_errors=True)
        raise
    return job


def add(jid, files=(), roles=None):
    """More inputs for an open job. Completed work is kept: a step re-runs only where its inputs
    changed (per-asset caches make that incremental), and a pending need the new inputs make
    stale (plan / review after new footage or music) is withdrawn. Re-adding a file whose
    intake failed (same path and role) retries it."""
    snap = _load(jid)
    entries = [{"path": p, "role": r} for r, ps in (roles or {}).items() for p in ps or []]
    entries += [{"path": p} for p in files or []]
    if not entries:
        raise JobError("nothing to add: give files or --sample/--source/--music/--sfx/--image", code="invalid_request")
    assets, errors = _assets(entries, snap["policy"]["sample_max_seconds"])
    if errors:
        raise JobError(f"invalid input ({len(errors)} problem(s))", errors, code="invalid_request")
    with _txn(jid) as job:
        if job["terminal"]:
            raise JobError(f"job {jid} is {job['status']}: start a new job for new work", code="terminal")
        have = {(f["path"], f.get("role")): f for f in job["files"]}
        paths = {f["path"] for f in job["files"]}
        new, again = [], []
        for a in assets:
            f = have.get((a["path"], a["role"]))
            if f is not None:
                if f.get("error") and not f.get("asset") and f not in again:
                    again.append(f)
            elif not (a["role_source"] == "heuristic" and a["path"] in paths):
                new.append(a)
        if not new and not again:
            job["_nosave"] = True
        else:
            for f in again:                 # intake re-runs for it (_intake_ok)
                f["retried"] = int(f.get("retried") or 0) + 1
                f.pop("error", None)
            job["files"] += [{**a, "added_rev": job["revision"] + 1} for a in new]
            job["request"]["assets"] += [{"path": a["path"], "role": a["role"]} for a in new]
            job["request_hash"] = fsutil.sha256_json(job["request"])
            fsutil.atomic_write_json(_dir(jid) / "request.json", job["request"])
            job["inputs_rev"] = int(job.get("inputs_rev") or 0) + 1
            new += [{"path": f["path"], "role": f["role"], "role_source": f.get("role_source")} for f in again]
            added, n = {a["role"] for a in new}, job.get("needs")
            if job["status"].startswith("waiting:"):
                job["status"], job["waiting"] = "waiting:worker", []
            elif n and n["step"] in ("plan", "review", "approve_storyboard") and added & {"source", "music", "sfx", "sample"}:
                _log(job, f"need {n['need_id']} withdrawn: the new {'/'.join(sorted(added))} changes the plan",
                     "need_withdrawn", need_id=n["need_id"])
                job["needs"], job["status"] = None, "waiting:worker"
            _log(job, "added " + ", ".join(f"{Path(a['path']).name} ({a['role']})" for a in new), "inputs_added",
                 files=[{"path": a["path"], "role": a["role"], "role_source": a["role_source"]} for a in new])
    return {**status(jid), "added": [{"path": a["path"], "role": a["role"]} for a in new]}


# ------------------------------------------------------------------ step bookkeeping

def _intake_ok(job):
    return all(f.get("asset") or f.get("error") for f in job["files"])


def _key(job, step):
    """What a completed step depended on; a step whose recorded key differs is redone."""
    inp, pol = job["inputs"], job["policy"]
    sv, pl = job.get("style_version") or {}, job.get("plan") or {}
    return {"analyze": lambda: [inp["samples"], _lang(job)],
            "verify_style": lambda: [inp["samples"][:1]],
            "transcribe": lambda: [inp["sources"], _lang(job), pol["asr_provider"]],
            "plan": lambda: [inp["sources"], inp["music"], sv.get("sha256")],
            "storyboard": lambda: [job["plan_generation"], job.get("inputs_rev"), _storyboard_fingerprint(job)],
            "approve_storyboard": lambda: [_key(job, "storyboard"), (job.get("storyboard") or {}).get("sha256")],
            "prepare": lambda: [pl.get("sha256"), pol["prepare"], (job.get("storyboard") or {}).get("key")],
            "render": lambda: [pl.get("sha256"), pol["encoder"], pol["require_gpu"], pol["allow_cpu_fallback"],
                               pol["render_quality"], (job.get("storyboard") or {}).get("key")],
            "review": lambda: [pl.get("sha256"), (job.get("render") or {}).get("seq")],
            }.get(step, lambda: None)()


def _valid(job, step):
    rec = job["steps"].get(step)
    if step == "intake":
        return bool(rec) and _intake_ok(job)
    if step == "deliver":
        return bool(rec)
    if not rec or rec.get("reopened") or rec.get("key") != _key(job, step):
        return False
    if step == "verify_style":
        return bool(job.get("style_version")) and Path(job["style_version"]["path"]).exists()
    if step == "plan":
        return bool(job.get("plan")) and Path(job["plan"]["path"]).exists()
    if step in ("storyboard", "approve_storyboard"):
        from . import storyboard
        if step == "approve_storyboard":
            a, b = job.get("storyboard_approval") or {}, job.get("storyboard") or {}
            if a.get("by") != "user" or a.get("sha256") != b.get("sha256") or a.get("plan_generation") != job["plan_generation"]:
                return False
        return bool(job.get("storyboard")) and storyboard.intact(job["storyboard"])
    if step == "render":
        return isinstance(_manifest(job), dict)
    return True


def _next_step(job):
    """First step that is not valid and can run; None = waiting for inputs."""
    inp = job["inputs"]
    if not _valid(job, "intake"):
        return "intake"
    if inp["samples"]:
        for s in ("analyze", "verify_style"):
            if not _valid(job, s):
                return s
    if inp["sources"] and not _valid(job, "transcribe"):
        return "transcribe"
    if not inp["samples"] or not inp["sources"]:
        return None
    for s in ("plan", "storyboard", "approve_storyboard", "prepare", "render", "review"):
        if not _valid(job, s):
            return s
    return "deliver"


def _set_waiting(job):
    missing = [r for r, b in (("sample", "samples"), ("source", "sources")) if not job["inputs"][b]]
    st = f"waiting:{missing[0]}"
    if job["status"] == st and job.get("waiting") == missing:
        job["_nosave"] = True
        return
    job["status"], job["waiting"] = st, missing
    _log(job, f"waiting for {' and '.join(missing)} input: `reel job add {job['id']} --{missing[0]} FILE` resumes",
         "waiting", missing=missing)


def _mark_running(job, step):
    sa = job["stage_attempts"].setdefault(step, {"n": 0, "failures": 0})
    sa.update(n=int(sa.get("n") or 0) + 1, attempt=job.get("attempt"), started=_now())
    job["status"], job["waiting"] = f"running:{step}", []
    _ev(job, "step_started", stage=step)


def _step_ok(job, step, elapsed):
    sa = job["stage_attempts"].setdefault(step, {"n": 1, "failures": 0})
    sa.update(failures=0, last_ok=_now(), elapsed_s=round(elapsed, 1))
    _ev(job, "step_done", stage=step, elapsed_s=round(elapsed, 1))


def _error_record(e, step):
    """Typed error record (FFError / RenderError carry one; anything else gets one). Call inside
    the except block."""
    rec = getattr(e, "record", None)
    out = dict(rec) if isinstance(rec, dict) else {}
    for k in ("cmd", "returncode", "stderr_tail"):
        out.setdefault(k, None)
    out.setdefault("stage", step)
    out.setdefault("cause", f"{type(e).__name__}: {e}")
    name = type(e).__name__
    if "transient" not in out:
        out["transient"] = name in ("TimeoutError", "TimeoutExpired", "BrokenProcessPool")
    if name in ("NotReady", "PlanError"):
        out["transient"] = False
    if name == "PlanError":
        out["plan_error"] = True
    out.update(step=step, type=name, at=_now(), traceback=traceback.format_exc(limit=6)[-3000:])
    return json.loads(json.dumps(out, default=str))


def _fail(job, err):
    job.update(status="failed", terminal=True, needs=None, error=err)
    if not job.get("errors") or job["errors"][-1] is not err:
        job["errors"] = (job.get("errors") or [])[-29:] + [err]
    _log(job, f"failed in {err.get('stage')}: {err.get('cause')}", "failed", error=err)


def _step_failed(job, step, err):
    """Retry a transient failure (stage_retries), send a plan the batch rejects back to the
    planner, fail the job otherwise."""
    pol = job["policy"]
    sa = job["stage_attempts"].setdefault(step, {"n": 1, "failures": 0})
    sa["failures"] = int(sa.get("failures") or 0) + 1
    sa["last_error"] = err
    job["errors"] = (job.get("errors") or [])[-29:] + [err]
    if err.get("plan_error") and job["steps"].get("plan") and sa["failures"] <= pol["max_rejections"]:
        job["steps"]["plan"]["reopened"] = True
        job["plan_errors"] = [str(err.get("cause"))[:4000]]
        job["status"] = "waiting:worker"
        _log(job, f"{step}: the batch rejected the plan; back to the planner", "plan_reopened", error=err)
    elif err.get("transient") and sa["failures"] <= pol["stage_retries"]:
        job["status"] = "waiting:worker"
        _log(job, f"{step} failed ({err.get('cause')}); retrying", "step_retry", stage=step, error=err)
    else:
        _fail(job, err)


def _cancelled(job):
    job.update(status="cancelled", terminal=True, needs=None)
    _log(job, "cancelled", "cancelled")


# ------------------------------------------------------------------ needs (judgment steps)

def _need_id(job, step):
    job["need_seq"] = int(job.get("need_seq") or 0) + 1
    return f"{step}-{job['need_seq']}"


def _issue(job, step, need_id=None, **detail):
    need_id = need_id or _need_id(job, step)
    agent = JUDGMENT[step]
    job["needs"] = {"step": step, "need_id": need_id, "revision": None, "agent": agent,
                    "model": (job["policy"].get("models") or {}).get(agent, "inherit"),
                    "rejections": 0, "errors": [], "issued": _now(), **detail}
    job["status"], job["waiting"] = f"needs:{step}", []
    _log(job, f"needs {step} from {agent} ({need_id})", "need_issued", stage=step, need_id=need_id)
    return True


def _open_need(job, step):
    """Publish the need for a judgment step. False when code settled it (a verified style reused,
    a new style re-pinned into an unchanged plan, nothing left to review): the loop goes on."""
    return {"verify_style": _need_style, "plan": _need_plan, "review": _need_review}[step](job)


def _storyboard_fingerprint(job):
    from . import storyboard
    pl = job.get("plan") or {}
    if not pl.get("path"):
        return None
    try:
        if fsutil.sha256_file(pl["path"]) != pl["sha256"]:
            return None
        return storyboard.fingerprint(pl["path"], job["policy"]["storyboard"], job["policy"]["asr_provider"])
    except (OSError, ValueError, KeyError):
        return None


def _need_storyboard_approval(job):
    board = job["storyboard"]
    nid = _need_id(job, "approve_storyboard")
    job["needs"] = {"step": "approve_storyboard", "agent": "user", "need_id": nid, "revision": None,
                    "issued": _now(), "storyboard_sha256": board["sha256"], "page": board["page"],
                    "sheet": board["sheet"], "manifest": board["manifest"], "frames": board["frames"],
                    "estimated_render_frames": board["estimated_render_frames"], "limitations": board["limitations"],
                    "plan_generation": job["plan_generation"], "errors": []}
    job["status"] = "needs:approve_storyboard"
    _log(job, "storyboard ready: full preparation and rendering await explicit user approval", "storyboard_ready")
    return True


def decide_storyboard(jid, decision, need_id, revision, sha256, notes=""):
    """User-facing command only. Agents cannot complete this via `job done`.
    Approval is for this need, revision and exact storyboard, never a future plan."""
    if decision not in ("approve", "revise"):
        raise Rejected("storyboard decision must be approve or revise", code="bad_decision")
    if decision == "revise" and not str(notes).strip():
        raise Rejected("revision feedback is required", code="no_feedback")
    with _txn(jid) as job:
        n, board = job.get("needs") or {}, job.get("storyboard") or {}
        if (job["terminal"] or n.get("step") != "approve_storyboard" or n.get("need_id") != need_id
                or n.get("revision") != revision or not sha256 or sha256 != board.get("sha256")
                or sha256 != n.get("storyboard_sha256")):
            raise Rejected("stale storyboard decision: quote the current need, revision and storyboard hash", code="stale")
        if not _valid(job, "storyboard"):
            raise Rejected("the storyboard or its inputs changed; run the worker to create a fresh preview", code="stale")
        if decision == "approve":
            job["storyboard_approval"] = {"need_id": need_id, "revision": revision, "sha256": sha256,
                                          "plan_generation": job["plan_generation"], "by": "user", "at": _now(),
                                          "notes": str(notes)[:4000]}
            job["steps"]["approve_storyboard"] = {"done": True, "key": _key(job, "approve_storyboard"), "at": _now()}
        else:
            job["storyboard_feedback"] = (job.get("storyboard_feedback") or [])[-19:] + [
                {"plan_generation": job["plan_generation"], "notes": str(notes)[:4000], "at": _now()}]
            job["steps"]["plan"]["reopened"] = True
            job["storyboard_approval"] = None
            job["steps"].pop("approve_storyboard", None)
        job["needs"], job["status"] = None, "waiting:worker"
        _log(job, f"user storyboard decision: {decision}", f"storyboard_{decision}", storyboard_sha256=sha256)
    return status(jid)


def _require_storyboard_approval(job):
    if not _valid(job, "storyboard") or not _valid(job, "approve_storyboard"):
        raise JobError("full preparation/rendering requires approval of the current storyboard", code="approval_required")


def _analysis_fp(style):
    a = style.get("analysis")
    if isinstance(a, dict) and a.get("fingerprint"):
        return str(a["fingerprint"])
    return fsutil.sha256_json({k: v for k, v in style.items() if k != "review"})


def _review_of(style):
    return style["review"] if isinstance(style, dict) and isinstance(style.get("review"), dict) else {}


def _style_errors(style, sid):
    if not isinstance(style, dict):
        return ["the style must be a JSON object"]
    errs = []
    if _review_of(style).get("verified") is not True:
        errs.append("review.verified must be true once every element was checked against the sample")
    sample = style.get("sample") if isinstance(style.get("sample"), dict) else {}
    if sample.get("id") not in (None, sid):
        errs.append(f"sample.id {sample.get('id')!r} is not this job's sample {sid}")
    try:
        errs += list(contracts.validate_style(style).get("errors") or [])
    except Exception as e:
        errs.append(f"style validation failed: {type(e).__name__}: {e}")
    return errs


def _publish_style(job, style, sid, fp, note):
    """Immutable styles/style.v<k>.json pinned in the job; also saved on the sample asset as
    style.verified.json so later jobs of the same measurement reuse it."""
    d = _dir(job["id"]) / "styles"
    d.mkdir(parents=True, exist_ok=True)
    k = int(job.get("style_seq") or 0) + 1
    while (d / f"style.v{k}.json").exists():
        k += 1
    style = copy.deepcopy(style)
    style["review"] = {**_review_of(style), "verified": True, "analysis_fingerprint": fp, "job": job["id"], "version": k}
    p = d / f"style.v{k}.json"
    fsutil.atomic_write_json(p, style)
    with contextlib.suppress(OSError):
        os.chmod(p, 0o444)
    sha = fsutil.sha256_file(p)
    job["style_seq"] = k
    job["style_version"] = {"version": k, "path": str(p), "sha256": sha, "sample": sid, "fingerprint": fp,
                            "published": _now()}
    try:
        fsutil.atomic_write_json(store.get(sid).cache("style.verified.json"), style)
    except Exception as e:
        _log(job, f"verified style not saved on sample {sid} for reuse ({e})", "warning")
    job["steps"]["verify_style"] = {"done": True, "at": _now(), "key": _key(job, "verify_style"), "version": k,
                                    "note": note}
    if (job.get("repair") or {}).get("style"):
        job["repair"]["style_done"] = True
    _log(job, f"style v{k} pinned ({note})", "style_published", version=k, sha256=sha)


def _need_style(job):
    sid = job["inputs"]["samples"][0]
    a = store.get(sid)
    measured = _rj(a.cache("style.json"))
    if not isinstance(measured, dict):
        _fail(job, {"stage": "analyze", "cause": f"sample {sid}: style.json is missing or unreadable; re-run "
                                                 f"`reel sample {sid} --force`", "transient": False, "at": _now()})
        return True
    fp = _analysis_fp(measured)
    rep = job.get("repair") or {}
    fixing = bool(rep.get("style")) and not rep.get("style_done")
    if not fixing:
        verified = _rj(a.cache("style.verified.json"))
        for cand, how in ((verified, "reused the verified style of this sample"),
                          (measured, "reused a style verified in place by an earlier version")):
            if not isinstance(cand, dict) or _review_of(cand).get("verified") is not True:
                continue
            if cand is verified and _review_of(cand).get("analysis_fingerprint") != fp:
                continue
            if not _style_errors(cand, sid):
                _publish_style(job, cand, sid, fp, how)
                return False
    work = _dir(job["id"]) / "style.work.json"
    mark = {"sample": sid, "fingerprint": fp, "round": rep.get("round")}
    cur = job.get("style_work") or {}
    if not work.exists() or any(cur.get(k) != v for k, v in mark.items()):
        sv = job.get("style_version") or {}
        seed, src = (_rj(sv.get("path")), "pinned") if sv.get("sample") == sid else (None, None)
        if not isinstance(seed, dict):
            v = _rj(a.cache("style.verified.json"))
            ok = isinstance(v, dict) and _review_of(v).get("analysis_fingerprint") == fp
            seed, src = (v, "verified") if ok else (measured, "measured")
        seed = copy.deepcopy(seed)
        seed["review"] = {**_review_of(seed), "verified": False}
        fsutil.atomic_write_json(work, seed)
        job["style_work"] = {**mark, "seeded_from": src}
    return _issue(job, "verify_style", style=str(work), review=str(a.cache("review.jpg")), sample=str(a.path),
                  sample_id=sid, extra_samples=job["inputs"]["samples"][1:],
                  review_out=str(_dir(job["id"]) / "style.review.jpg"),
                  fixes=rep.get("style_fixes") if fixing else None)


def _free_key(srcs):
    for i in range(100000):
        k = chr(ord("A") + i) if i < 26 else f"S{i + 1}"
        if k not in srcs:
            return k


def _scaffold_plan(job):
    """plan.json with the orchestrator's fields current (name, pinned style, sources by key,
    output language, music); the reels are the planner's."""
    p = _plan_path(job)
    p.parent.mkdir(parents=True, exist_ok=True)
    plan = _rj(p)
    if p.exists() and not isinstance(plan, dict):
        bad = p.with_name(f"plan.unreadable.{int(time.time())}.json")
        with contextlib.suppress(OSError):
            os.replace(p, bad)
        _log(job, f"plan.json was unreadable: moved aside to {bad.name}", "warning")
    if not isinstance(plan, dict):
        plan = {"name": job["id"], "style": None,
                "output": {"width": 720, "height": 1280, "fps": 30, "quality": job["policy"]["render_quality"]},
                "sources": {}, "music": None, "sfx": None, "reels": []}   # loudness: the sample's, unless set
    before = json.dumps(plan, sort_keys=True)
    plan["name"], plan["style"] = job["id"], job["style_version"]["path"]
    out = plan["output"] if isinstance(plan.get("output"), dict) else {}
    out["lang"] = _lang(job)
    plan["output"] = out
    srcs = plan["sources"] if isinstance(plan.get("sources"), dict) else {}
    have = set(srcs.values())
    for aid in job["inputs"]["sources"]:
        if aid not in have:
            srcs[_free_key(srcs)] = aid
    plan["sources"] = srcs
    if plan.get("music") is None and job["inputs"]["music"]:
        plan["music"] = {"asset": job["inputs"]["music"][0], "gain_db": -18}
    for k, v in (("music", None), ("sfx", None), ("reels", [])):
        plan.setdefault(k, v)
    if json.dumps(plan, sort_keys=True) != before:
        fsutil.atomic_write_text(p, json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
    return p


def _check_plan(job, plan, repair):
    """(errors, accepted-plan record) for a submitted plan: orchestrator fields re-pinned,
    sources from this job, non-empty, contracts.validate_plan against the transcripts and the
    request, and in a restricted repair round only the reels under repair may differ."""
    from . import source
    if not isinstance(plan, dict):
        return ["plan.json must be a JSON object"], None
    plan, errors, warnings = copy.deepcopy(plan), [], []
    pinned = job["style_version"]["path"]
    if plan.get("name") != job["id"]:
        warnings.append(f"name set to the job id {job['id']}")
        plan["name"] = job["id"]
    if plan.get("style") != pinned:
        warnings.append("style re-pinned to the verified style version")
        plan["style"] = pinned
    srcs = plan.get("sources") if isinstance(plan.get("sources"), dict) else {}
    known = set(job["inputs"]["sources"])
    if not srcs:
        errors.append("sources: the orchestrator's {key: asset} map is required")
    for k, aid in srcs.items():
        if aid not in known:
            errors.append(f"sources.{k}: {aid!r} is not one of this job's sources")
    reels = plan.get("reels") if isinstance(plan.get("reels"), list) else []
    if not reels:
        errors.append("reels: the plan has no reels")
    transcripts = {}
    for k, aid in srcs.items():
        if aid in known:
            try:
                transcripts[k] = source.transcript(store.get(aid), _lang(job), provider=job["policy"]["asr_provider"])
            except Exception as e:
                errors.append(f"sources.{k}: transcript unavailable ({type(e).__name__}: {e})")
    # the request in both shapes: the full normalised request and its "request" fields at top level
    req = {**(job.get("request") or {}), **_rq(job)}
    try:
        v = contracts.validate_plan(plan, transcripts, req)
        errors += list(v.get("errors") or [])
        warnings += list(v.get("warnings") or [])
    except Exception as e:
        errors.append(f"plan validation failed: {type(e).__name__}: {e}")
    hashes = {}
    for r in reels:
        if isinstance(r, dict) and isinstance(r.get("id"), str):
            try:
                hashes[r["id"]] = contracts.plan_entry_hash(r)
            except Exception as e:
                errors.append(f"reel {r['id']}: cannot hash its entry ({e})")
    prev = (job.get("plan") or {}).get("entry_hashes")
    if repair and prev is not None and not repair.get("style"):
        only = set(repair.get("only") or [])
        for rid, h in prev.items():
            if rid in only:
                continue
            if rid not in hashes:
                errors.append(f"reel {rid} was removed but is not under repair (only {sorted(only)})")
            elif hashes[rid] != h:
                errors.append(f"reel {rid} changed but is not under repair (only {sorted(only)}): restore it exactly")
        errors += [f"reel {rid} is new: this repair round may only change {sorted(only)}"
                   for rid in hashes if rid not in prev and rid not in only]
        if only and not any(rid not in hashes or hashes[rid] != prev.get(rid) for rid in only):
            errors.append(f"none of the reels under repair ({sorted(only)}) changed: apply the fixes (or drop a reel "
                          "that cannot be fixed)")
    if errors:
        return errors, None
    return [], {"plan": plan, "text": json.dumps(plan, indent=1, ensure_ascii=False) + "\n", "warnings": warnings,
                "entry_hashes": hashes}


def _accept_plan(job, prep, need_id, note):
    k = int(job.get("plan_generation") or 0) + 1
    d = _dir(job["id"]) / "plans"
    d.mkdir(parents=True, exist_ok=True)
    snap = d / f"plan.g{k}.json"
    fsutil.atomic_write_text(snap, prep["text"])
    with contextlib.suppress(OSError):
        os.chmod(snap, 0o444)
    fsutil.atomic_write_text(_plan_path(job), prep["text"])
    sha = fsutil.sha256_file(snap)
    reels = [r.get("id") for r in prep["plan"]["reels"] if isinstance(r, dict)]
    job["plan"] = {"generation": k, "path": str(snap), "working": str(_plan_path(job)), "sha256": sha, "reels": reels,
                   "entry_hashes": prep["entry_hashes"], "need_id": need_id, "accepted": _now(),
                   "warnings": prep["warnings"][:50]}
    job["plan_generation"] = k
    job["storyboard_approval"] = None
    job["steps"].pop("approve_storyboard", None)
    job["steps"]["plan"] = {"done": True, "at": _now(), "key": _key(job, "plan"), "generation": k, "reels": len(reels),
                            "note": note}
    rep = job.get("repair")
    if rep:
        job["repairs"] = (job.get("repairs") or [])[-49:] + [{**rep, "plan_generation": k}]
        job["repair"] = None
    _log(job, f"plan g{k} accepted: {len(reels)} reel(s)" + (f" ({note})" if note else "") +
         (f"; {len(prep['warnings'])} warning(s)" if prep["warnings"] else ""), "plan_accepted",
         generation=k, sha256=sha, warnings=prep["warnings"][:20])


def _auto_plan(job):
    """Re-pin a new style version into the accepted plan without a planner round when nothing
    else changed (a style-only repair, or a repair abandoned after its style was re-verified)."""
    pl, rec, rep = job.get("plan"), job["steps"].get("plan") or {}, job.get("repair")
    if not pl or (rep and (rep.get("fixes") or not rep.get("style"))):
        return False
    key, cur = rec.get("key") or [], _key(job, "plan")
    if key[:2] != cur[:2] or key == cur:
        return False
    errors, prep = _check_plan(job, _rj(pl["path"]), None)
    if errors:
        _log(job, f"new style version needs the planner: {errors[0]}", "warning")
        return False
    _accept_plan(job, prep, None, "style re-pinned into the unchanged plan")
    return True


def _need_plan(job):
    if _auto_plan(job):
        return False
    plan_p = _scaffold_plan(job)
    lang, rq, rep = _lang(job), _rq(job), job.get("repair") or {}
    return _issue(job, "plan", plan=str(plan_p), style=job["style_version"]["path"],
                  transcripts=[str(store.get(a).cache(f"transcript.{lang}.txt")) for a in job["inputs"]["sources"]],
                  sources=(_rj(plan_p) or {}).get("sources") or {}, count=rq.get("count"),
                  target_duration=rq.get("duration_seconds"), lang=lang, notes=rq.get("notes") or "",
                  request=job["request"], request_hash=job["request_hash"], fixes=rep.get("fixes") or None,
                  only=None if rep.get("style") else (rep.get("only") or None), repair_round=rep.get("round"),
                  plan_generation=int(job.get("plan_generation") or 0) + 1,
                  previous_plan=(job.get("plan") or {}).get("path"), errors=job.pop("plan_errors", None) or [],
                  storyboard_feedback=job.get("storyboard_feedback") or [])


# ------------------------------------------------------------------ review bookkeeping

def _manifest(job):
    p = (job.get("render") or {}).get("manifest")
    return _rj(p) if p else None


def _manifest_path(m, jid):
    base = DATA / "batches" / str(m.get("name") or jid)
    g = m.get("generation")
    p = base / "generations" / str(g) / "manifest.json"
    return p if g is not None and p.exists() else base / "manifest.json"


def _qc(e):
    return str(e.get("qc_verdict") or e.get("verdict") or "error")


def _entry_ok(e):
    """Technically acceptable: rendered, QC pass or warn (old manifests carry no status)."""
    return e.get("status", "accepted") == "accepted" and _qc(e) in ("pass", "warn")


def _entries(job, manifest):
    """Manifest entries, in plan order, for every reel of the accepted plan; a reel the manifest
    lacks is an error entry, never silently skipped."""
    by_id = {e.get("id"): e for e in manifest.get("reels") or [] if isinstance(e, dict)}
    ids = (job.get("plan") or {}).get("reels") or list(by_id)
    return {rid: by_id.get(rid) or {"id": rid, "status": "failed", "qc_verdict": "error", "verdict": "error",
                                    "issues": [], "error": "missing from the batch manifest"} for rid in ids}


def _spot_check(entries, mode="sqrt"):
    good = [e.get("id") for e in entries if _qc(e) == "pass" and _entry_ok(e)]
    if mode == "all":
        return set(good)
    if mode != "sqrt" or not good:
        return set()
    k = min(len(good), math.ceil(math.sqrt(len(entries))))
    step = max(1, len(good) // k)
    return set(good[::step][:k])


def review_set(manifest, spot_check="sqrt"):
    """Reels a reviewer must look at: every fail/warn/error + a spot check of passes."""
    reels = [e for e in manifest.get("reels") or [] if isinstance(e, dict)]
    spot = _spot_check(reels, spot_check)
    return [e for e in reels if not _entry_ok(e) or _qc(e) != "pass" or e.get("id") in spot]


def _required(e, L, spotted):
    if not _entry_ok(e) or _qc(e) != "pass" or spotted:
        return True
    return (L or {}).get("decision") in ("repairing", "unreviewed", "fix_required")


def _decided(e, L):
    L = L or {}
    return L.get("decision") in ("accept", "fix_required") and L.get("render_sha256") == e.get("render_sha256")


def _exhausted(e, L, pol):
    L = L or {}
    return (L.get("decision") == "unreviewed" and L.get("render_sha256") == e.get("render_sha256")
            and int(L.get("requests") or 0) > pol["review_retries"])


def _assignment(job, entries):
    pol, ledger = job["policy"], job["review"]["ledger"]
    spot = _spot_check(list(entries.values()), pol["spot_check"])
    return [rid for rid, e in entries.items() if _required(e, ledger.get(rid), rid in spot)
            and not _decided(e, ledger.get(rid)) and not _exhausted(e, ledger.get(rid), pol)]


def _need_review(job):
    manifest = _manifest(job)
    if not isinstance(manifest, dict):
        job["steps"].pop("render", None)
        _log(job, "batch manifest unreadable: rendering again", "warning")
        return False
    entries = _entries(job, manifest)
    todo = _assignment(job, entries)
    rv = job["review"]
    if not todo:
        job["steps"]["review"] = {"done": True, "at": _now(), "key": _key(job, "review")}
        _log(job, "review complete for this render", "review_complete")
        return False
    pol, ledger, attempts = job["policy"], rv["ledger"], rv["attempts_by_reel"]
    nid = _need_id(job, "review")
    inbox = _dir(job["id"]) / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    assigned = []
    for rid in todo:
        e = entries[rid]
        L = ledger.setdefault(rid, {})
        if L.get("render_sha256") != e.get("render_sha256"):
            L["requests"] = 0               # retries are counted per render
        L.update(requests=int(L.get("requests") or 0) + 1, assigned=nid)
        assigned.append({"id": rid, "render_sha256": e.get("render_sha256"), "qc_verdict": _qc(e),
                         "status": e.get("status"), "issues": e.get("issues") or [], "error": e.get("error"),
                         "file": e.get("file"), "sheet": e.get("sheet"), "qc": e.get("qc"),
                         "previous_findings": [_compact(f) for f in job["findings"]
                                               if f.get("reel_id") == rid and f.get("status") == "open"]})
    return _issue(job, "review", need_id=nid, plan_generation=job["plan_generation"],
                  manifest=job["render"]["manifest"], render_generation=manifest.get("generation"),
                  assigned=assigned,
                  rounds_left_by_reel={rid: max(0, pol["max_repairs_per_reel"] - int(attempts.get(rid, 0)))
                                       for rid in todo},
                  result_file=str(inbox / f"{nid}.json"), style=job["style_version"]["path"],
                  plan=job["plan"]["path"], schema_version=getattr(contracts, "REVIEW_SCHEMA_VERSION", 1))


def _is_style(f):
    return str(f.get("scope") or "").lower() == "style" or str(f.get("code") or "").upper().startswith("STYLE")


def _schedule_repair(job, rids, nid):
    """Reels with blocking findings and budget left go back to the planner (and, for style-level
    findings, the sample-analyst first: a new style version re-renders every reel)."""
    rv = job["review"]
    rv["rounds"] = int(rv.get("rounds") or 0) + 1
    for rid in rids:
        rv["attempts_by_reel"][rid] = int(rv["attempts_by_reel"].get(rid, 0)) + 1
        rv["ledger"][rid]["decision"] = "repairing"
    opened = [f for f in job["findings"] if f.get("reel_id") in rids and f.get("status") == "open"]

    def fix(f):
        return {"reel": f["reel_id"], "what": f.get("what"), "how": f.get("how"), "code": f.get("code"),
                "severity": f.get("severity"), "finding": f["id"]}
    prev = job.get("repair") or {}
    style = [fix(f) for f in opened if _is_style(f)]
    rep = {"round": rv["rounds"], "need_id": nid, "only": sorted(set(rids) | set(prev.get("only") or [])),
           "fixes": (prev.get("fixes") or []) + [fix(f) for f in opened if not _is_style(f)],
           "style_fixes": (prev.get("style_fixes") or []) + style,
           "style": bool(style) or bool(prev.get("style")), "style_done": False, "at": _now()}
    job["repair"] = rep
    for s in ("plan", "verify_style") if rep["style"] else ("plan",):
        if s in job["steps"]:
            job["steps"][s]["reopened"] = True
    _log(job, f"repair round {rep['round']} for {rep['only']}: {len(rep['fixes'])} fix(es)" +
         (f" + {len(rep['style_fixes'])} style-level" if rep["style"] else ""), "repair_scheduled",
         reels=rep["only"], style=rep["style"])


def _abandon_repair(job, why):
    rep = job.get("repair") or {}
    for rid in rep.get("only") or []:
        L = job["review"]["ledger"].get(rid)
        if L and L.get("decision") == "repairing":
            L["decision"] = "fix_required"
    for s in ("plan", "verify_style"):
        if s in job["steps"]:
            job["steps"][s].pop("reopened", None)
    job["repairs"] = (job.get("repairs") or [])[-49:] + [{**rep, "abandoned": why}]
    job["repair"] = None
    _log(job, f"repair round {rep.get('round')} abandoned ({why}): its reels stay blocked", "repair_abandoned")


def _check_review(job, need, raw):
    if not isinstance(raw, dict):
        return ["the review result must be a JSON object"], None
    errors = [f"{k} {raw[k]!r} does not match {want!r}: stale result"
              for k, want in (("need_id", need["need_id"]), ("job_id", job["id"]),
                              ("plan_generation", job["plan_generation"]), ("expected_revision", need.get("revision")))
              if raw.get(k) is not None and raw[k] != want]
    if errors:
        return errors, None
    assigned = [{"id": a["id"], "render_sha256": a.get("render_sha256"), "qc_verdict": a.get("qc_verdict")}
                for a in need.get("assigned") or []]
    try:
        norm = contracts.normalize_review(raw, assigned, job["plan_generation"])
    except Exception as e:
        return [f"review result could not be normalised: {type(e).__name__}: {e}"], None
    ids = {a["id"] for a in assigned}
    results = [r for r in norm.get("results") or [] if isinstance(r, dict) and (r.get("reel_id") or r.get("id")) in ids]
    if not results and norm.get("errors"):      # nothing usable: let the caller fix the envelope
        return [str(x) for x in norm["errors"]], None
    return [], {"results": results, "errors": [str(x) for x in norm.get("errors") or []], "notes": raw.get("notes")}


def _apply_review(job, need, prep, note):
    """Record every decision and finding; omitted reels stay unreviewed (re-requested within
    review_retries); fix_required reels with budget left are scheduled for repair."""
    pol, rv = job["policy"], job["review"]
    ledger, attempts = rv["ledger"], rv["attempts_by_reel"]
    nid, gen = need["need_id"], job["plan_generation"]
    assigned = {a["id"]: a for a in need.get("assigned") or []}
    decided = {}
    for r in prep["results"]:
        rid = r.get("reel_id") or r.get("id")
        if rid in assigned and rid not in decided and r.get("decision") in ("accept", "fix_required"):
            decided[rid] = r
    for rid, r in decided.items():
        a, dec = assigned[rid], r["decision"]
        for f in job["findings"]:       # the new decision supersedes what the reel had open
            if f.get("reel_id") == rid and f.get("status") == "open":
                f["status"], f["closed_by"] = ("resolved" if dec == "accept" else "superseded"), nid
        found = [x for x in r.get("findings") or [] if isinstance(x, dict)]
        if dec == "fix_required" and not found:
            found = [{"code": "UNSPECIFIED", "severity": "blocking", "what": "fix required without details", "how": ""}]
        ids = []
        for x in found:
            fid = f"F{len(job['findings']) + 1:04d}"
            job["findings"].append({"id": fid, "reel_id": rid, "code": x.get("code"),
                                    "severity": x.get("severity") or "blocking", "what": x.get("what"),
                                    "how": x.get("how"), "scope": x.get("scope"),
                                    "status": "open" if dec == "fix_required" else "noted", "need_id": nid,
                                    "plan_generation": gen, "render_sha256": a.get("render_sha256"), "at": _now()})
            ids.append(fid)
        ledger[rid] = {"decision": dec, "render_sha256": a.get("render_sha256"), "qc_verdict": a.get("qc_verdict"),
                       "findings": ids, "need_id": nid, "plan_generation": gen, "requests": 0, "at": _now()}
    unreviewed = [rid for rid in assigned if rid not in decided]
    for rid in unreviewed:
        ledger.setdefault(rid, {}).update(decision="unreviewed", render_sha256=assigned[rid].get("render_sha256"),
                                          qc_verdict=assigned[rid].get("qc_verdict"), need_id=nid, at=_now())
    if prep["errors"]:
        rv["errors"] = (rv.get("errors") or [])[-19:] + [{"need_id": nid, "errors": prep["errors"][:50], "at": _now()}]
    if prep.get("notes") or note:
        rv["notes"] = (rv.get("notes") or [])[-49:] + [{"need_id": nid, "notes": str(prep.get("notes") or note)[:2000]}]
    fix = [rid for rid, r in decided.items() if r["decision"] == "fix_required"]
    can = [rid for rid in fix if int(attempts.get(rid, 0)) < pol["max_repairs_per_reel"]]
    if can and int(rv.get("rounds") or 0) >= pol["max_review_rounds"]:
        _log(job, f"repair budget of {pol['max_review_rounds']} rounds spent: {can} stay blocked", "budget_exhausted")
        can = []
    if can:
        _schedule_repair(job, can, nid)
    blocked = [rid for rid in fix if rid not in can]
    again = [rid for rid in unreviewed if int(ledger[rid].get("requests") or 0) <= pol["review_retries"]]
    _log(job, f"review {nid}: {len(decided) - len(fix)} accepted, {len(fix)} fix required ({len(can)} to repair, "
              f"{len(blocked)} blocked), {len(unreviewed)} unreviewed ({len(again)} to re-request)" +
         (f", {len(prep['errors'])} invalid result(s)" if prep["errors"] else ""), "review_recorded",
         accepted=[rid for rid in decided if rid not in fix], fix_required=fix, repairing=can, blocked=blocked,
         unreviewed=unreviewed, errors=prep["errors"][:20])


# ------------------------------------------------------------------ agent completions

def _read_result(job, step, need_id, result_file, fixes, ok):
    """(answer, sha256 of its canonical JSON)."""
    if step == "review" and (fixes is not None or ok is not None):     # legacy --fixes / --ok
        raw = fixes if isinstance(fixes, dict) else {"fixes": fixes or [], "ok": list(ok or [])}
        return raw, fsutil.sha256_json(raw)
    if result_file:
        p = Path(result_file)
    elif step == "review":
        p = _dir(job["id"]) / "inbox" / f"{need_id}.json"
    elif step == "plan":
        p = _plan_path(job)
    else:
        p = _dir(job["id"]) / "style.work.json"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise Rejected(f"cannot read the {step} result {p}", [f"{p}: {e.strerror or e}"], code="no_result") from None
    except ValueError as e:
        raise Rejected(f"{p} is not valid JSON", [f"{p}: {e}"], code="bad_json") from None
    return obj, fsutil.sha256_json(obj)


def _check_need(job, step, need_id, revision, sha):
    """True = this exact answer was already accepted (idempotent). Raises Rejected when the job is
    not waiting for exactly this step and need (stale or out of order)."""
    done = (job.get("completions") or {}).get(need_id)
    if done:
        # the submitted answer, or (plan) the plan.json the orchestrator rewrote on acceptance
        if done.get("step") == step and sha is not None and sha in (done.get("sha256"), done.get("accepted_sha256")):
            return True
        raise Rejected(f"{need_id} was already completed with a different result",
                       [f"{need_id}: completed at {done.get('at')}"], code="already_completed")
    if job["terminal"]:
        raise Rejected(f"job {job['id']} is {job['status']}", code="terminal")
    n = job.get("needs") or {}
    if n.get("step") != step or n.get("need_id") != need_id:
        now = f"{n['step']} {n['need_id']}" if n else job["status"]
        raise Rejected(f"stale or out-of-order completion: the job is not waiting for {step} {need_id} (now: {now})",
                       code="stale")
    if revision is not None and int(revision) != n.get("revision"):
        raise Rejected(f"stale completion: revision {revision} is not the need's revision {n.get('revision')}",
                       code="stale")
    return False


def _rejections_exhausted(job, step, n):
    pol, ledger = job["policy"], job["review"]["ledger"]
    why = f"{step} answers rejected {n['rejections']} times"
    if step == "review":
        for a in n.get("assigned") or []:
            L = ledger.setdefault(a["id"], {})
            L.update(decision="unreviewed", render_sha256=a.get("render_sha256"), qc_verdict=a.get("qc_verdict"),
                     requests=max(int(L.get("requests") or 0), pol["review_retries"] + 1), need_id=n["need_id"],
                     at=_now())
    elif job.get("repair") and job.get("plan"):
        _abandon_repair(job, why)
    else:
        job.update(status="needs_attention", terminal=True, needs=None, attention=why)
        _log(job, f"needs attention: {why}", "needs_attention")
        return
    job["needs"], job["status"] = None, "waiting:worker"
    _log(job, f"{why}: going on without it", "budget_exhausted", stage=step)


def _reject(jid, step, need_id, errors):
    """Record a refused answer on its need (when still current) and raise Rejected."""
    errors = [str(e) for e in errors][:100] or ["rejected"]
    msg = f"{step} result rejected"
    with _txn(jid) as job:
        n = job.get("needs") or {}
        if n.get("need_id") != need_id or job["terminal"]:
            job["_nosave"] = True
        else:
            n["rejections"] = int(n.get("rejections") or 0) + 1
            n["errors"] = errors[:50]
            _log(job, f"{step} answer for {need_id} rejected ({len(errors)} problem(s)): {errors[0][:300]}",
                 "completion_rejected", stage=step, need_id=need_id, errors=errors[:20])
            if n["rejections"] >= job["policy"]["max_rejections"]:
                msg += f"; {n['rejections']} rejections used up the budget"
                _rejections_exhausted(job, step, n)
    raise Rejected(msg, errors, code="rejected")


def _validate_style(job, need, style):
    sid = job["inputs"]["samples"][0] if job["inputs"]["samples"] else None
    errs = _style_errors(style, sid)
    return errs, None if errs else {"style": style, "sample": sid}


def _commit_style(job, need, prep, note):
    sid = prep["sample"]
    fp = (job.get("style_work") or {}).get("fingerprint") or _analysis_fp(_rj(store.get(sid).cache("style.json")) or {})
    _publish_style(job, prep["style"], sid, fp, note or "verified by the sample-analyst")


_VALIDATE = {"verify_style": _validate_style,
             "plan": lambda job, need, plan: _check_plan(job, plan, job.get("repair")),
             "review": _check_review}
_COMMIT_DONE = {"verify_style": _commit_style,
                "plan": lambda job, need, prep, note: _accept_plan(job, prep, need["need_id"], note),
                "review": _apply_review}


def complete(jid, step, need_id=None, revision=None, note="", result_file=None, fixes=None, ok=None):
    """An agent answered a need. The answer must quote the open need (need_id, and its revision
    when given) and is validated here before anything advances: verify_style reads
    style.work.json (or result_file), plan reads plan.json (or result_file), review reads
    inbox/<need_id>.json (or result_file; legacy fixes/ok accepted). A repeat of an accepted
    answer is a no-op; stale, out-of-order or invalid answers raise Rejected."""
    if step not in JUDGMENT:
        raise Rejected(f"{step} is not a judgment step ({', '.join(JUDGMENT)})", code="bad_step")
    if not need_id:
        raise Rejected("need_id is required: quote needs.need_id from `reel job status`", code="no_need_id")
    _check_id(need_id, "need id")
    note = str(note or "")[:2000]
    job = _load(jid)
    finished = need_id in (job.get("completions") or {})
    if not finished:
        _check_need(job, step, need_id, revision, None)     # stale or out of order: refused before reading
    try:
        result, sha = _read_result(job, step, need_id, result_file, fixes, ok)
    except Rejected as e:
        if finished:
            raise
        _reject(jid, step, need_id, e.errors or [str(e)])
    if _check_need(job, step, need_id, revision, sha):
        return {**status(jid), "idempotent": True}
    errors, prep = _VALIDATE[step](job, job["needs"], result)
    if errors:
        _reject(jid, step, need_id, errors)
    with _txn(jid) as job:
        if _check_need(job, step, need_id, revision, sha):
            job["_nosave"] = True
        else:
            need = job["needs"]
            _COMMIT_DONE[step](job, need, prep, note)
            done = job.setdefault("completions", {})
            done[need_id] = {"step": step, "sha256": sha, "at": _now(), "revision": need.get("revision"),
                             "accepted_sha256": fsutil.sha256_json(prep["plan"]) if step == "plan" else None}
            for k in list(done)[:-100]:
                del done[k]
            job["needs"] = None
            if not job["terminal"]:
                job["status"] = "waiting:worker"
            _log(job, f"{step} done ({need_id})" + (f": {note[:300]}" if note else ""), "completed", stage=step,
                 need_id=need_id, sha256=sha)
    return status(jid)


# ------------------------------------------------------------------ mechanical steps

def _transcripts(plan, lang, provider):
    from . import source
    return {k: source.transcript(store.get(aid), lang, provider=provider) for k, aid in (plan.get("sources") or {}).items()}


def _planned_ranges(plan, transcripts):
    """Source seconds each source needs, from the plan's sentence ids / explicit times."""
    ranges = {}
    for reel in plan.get("reels", []):
        for seg in list(reel.get("segments") or []) + ([reel["hook"]] if reel.get("hook") else []):
            key = seg.get("src", "S")
            aid = plan["sources"][key]
            if "sentences" in seg:
                sents = transcripts[key]["sentences"]
                a, b = seg["sentences"][0], seg["sentences"][-1]
                ranges.setdefault(aid, []).append((sents[a]["s"], sents[b]["e"]))
            else:
                ranges.setdefault(aid, []).append((seg["s"], seg["e"]))
    return ranges


def _render_only(job):
    """Reels the last restricted repair changed (the others were proven unchanged at plan
    acceptance); None = let the batch check every reel's cache."""
    last = (job.get("repairs") or [None])[-1]
    if (not last or last.get("style") or last.get("abandoned") or not job.get("render")
            or last.get("plan_generation") != job.get("plan_generation")):
        return None
    return sorted(set(last.get("only") or []) & set(job["plan"]["reels"])) or None


def _context(job, step):
    """Snapshot of what a step needs, taken under the lock; the work runs outside it."""
    pol = job["policy"]
    c = {"key": _key(job, step), "inputs_rev": job.get("inputs_rev"), "lang": _lang(job), "policy": dict(pol)}
    if step in ("prepare", "render"):
        _require_storyboard_approval(job)
        c["storyboard"] = copy.deepcopy(job["storyboard"])
        c["storyboard_approval"] = copy.deepcopy(job["storyboard_approval"])
        c["plan_generation"] = job["plan_generation"]
    if step == "intake":
        c["files"] = [[i, f["path"], f.get("role")] for i, f in enumerate(job["files"])
                      if not f.get("asset") and not f.get("error")]
    elif step == "analyze":
        c["samples"] = list(job["inputs"]["samples"])
    elif step == "transcribe":
        c.update(sources=list(job["inputs"]["sources"]), provider=pol["asr_provider"])
    elif step == "storyboard":
        c.update(plan=job["plan"]["path"], plan_generation=job["plan_generation"],
                 root=str(_dir(job["id"]) / "storyboards"), provider=pol["asr_provider"])
    elif step == "prepare":
        c.update(plan=job["plan"]["path"], provider=pol["asr_provider"])
    elif step == "render":
        c.update(plan_path=str(_plan_path(job)), snapshot=job["plan"]["path"], plan_sha=job["plan"]["sha256"],
                 plan_generation=job["plan_generation"], style=job["style_version"]["path"], only=_render_only(job))
    elif step == "deliver":
        m = _manifest(job) or {}
        c["files"] = [[d.get("id"), d.get("file")] for d in m.get("deliverables") or []
                      if isinstance(d, dict) and isinstance(d.get("id"), str)]
    return c


def _w_intake(ctx):
    out = []
    for i, path, role in ctx["files"]:
        r = {"i": i, "path": path}
        try:
            if not role:
                role = r["suggested_role"] = classify(path, ctx["policy"]["sample_max_seconds"])
            a = store.ingest(path, role)
            r.update(role=role, asset=a.id, sha256=a.meta.get("sha256"))
        except _BAD_INPUT as e:
            r["error"] = f"{type(e).__name__}: {e}"
        out.append(r)
    return out


def _c_intake(job, ctx, res):
    for r in res:
        i = r["i"]
        if i >= len(job["files"]) or job["files"][i]["path"] != r["path"]:
            continue
        f = job["files"][i]
        if r.get("suggested_role") and not f.get("role"):
            f.update(role=r["suggested_role"], suggested_role=r["suggested_role"], role_source="heuristic")
            if {"path": f["path"], "role": f["role"]} not in job["request"]["assets"]:
                job["request"]["assets"].append({"path": f["path"], "role": f["role"]})
                job["request_hash"] = fsutil.sha256_json(job["request"])
        if r.get("asset"):
            f.update(asset=r["asset"], sha256=r.get("sha256"))
            f.pop("error", None)
            bucket = job["inputs"][ROLES[f["role"]]]
            if r["asset"] not in bucket:
                bucket.append(r["asset"])
        else:
            f["error"] = r["error"]
            _log(job, f"intake: {Path(f['path']).name}: {r['error']}", "input_error", path=f["path"], error=r["error"])
    job["steps"]["intake"] = {"done": True, "at": _now(), "files": len(job["files"])}
    _log(job, "intake: " + (", ".join(f"{len(v)} {k}" for k, v in job["inputs"].items() if v) or "no usable input"))


def _w_analyze(ctx):
    from . import sample
    warnings = {}
    for sid in ctx["samples"]:
        sample.analyze(sid, ctx["lang"])            # cached per sample
        st = _rj(store.get(sid).cache("style.json"))
        if not isinstance(st, dict):
            raise RuntimeError(f"sample {sid}: analysis left no readable style.json")
        v = contracts.validate_style(st)
        warnings[sid] = [str(x) for x in list(v.get("errors") or []) + list(v.get("warnings") or [])][:20]
    return warnings


def _c_analyze(job, ctx, res):
    job["steps"]["analyze"] = {"done": True, "at": _now(), "key": ctx["key"], "warnings": res}
    for sid, w in res.items():
        if w:
            _log(job, f"sample {sid}: measured style has {len(w)} problem(s) for the analyst: {w[0][:200]}", "warning")


def _w_transcribe(ctx):
    from . import asr, source
    out = {}
    for aid in ctx["sources"]:
        a = store.get(aid)
        tr = source.transcript(a, ctx["lang"], provider=ctx["provider"])
        fsutil.atomic_write_text(a.cache(f"transcript.{ctx['lang']}.txt"), asr.compact(tr))
        out[aid] = len(tr.get("sentences") or [])
    return out


def _c_transcribe(job, ctx, res):
    job["steps"]["transcribe"] = {"done": True, "at": _now(), "key": ctx["key"], "sentences": res}
    empty = [aid for aid, n in res.items() if not n]
    _log(job, f"transcribed {len(res)} source(s)" + (f"; no speech found in {empty}" if empty else ""))


def _w_prepare(ctx):
    _check_approved_context(ctx, ctx["plan"])
    from . import source
    plan = _rj(ctx["plan"])
    if not isinstance(plan, dict):
        raise RuntimeError(f"accepted plan {ctx['plan']} is unreadable")
    out = {}
    trs = _transcripts(plan, ctx["lang"], ctx["provider"])
    for aid, rngs in _planned_ranges(plan, trs).items():
        source.prepare(store.get(aid), rngs if ctx["policy"]["prepare"] == "planned_ranges" else None)
        out[aid] = len(rngs)
    return out


def _c_prepare(job, ctx, res):
    job["steps"]["prepare"] = {"done": True, "at": _now(), "key": ctx["key"], "ranges": res}


def _prefetch_fonts(style_path):
    from . import captions
    st = _rj(style_path) or {}
    styles = [(st.get(k) or {}).get("style") if isinstance(st.get(k), dict) else None for k in ("captions", "titles")]
    try:
        got = captions.prefetch_style_fonts(styles)
        ff.log(f"fonts ready: {', '.join(sorted(got or {})) or 'none needed'}")
    except Exception as e:
        ff.log(f"font prefetch failed ({type(e).__name__}: {e}); rendering falls back where it must")


def _w_render(ctx):
    _check_approved_context(ctx, ctx["snapshot"])
    from . import batch, readiness
    pol = ctx["policy"]
    enc = readiness.resolve_encoder(requested=pol["encoder"], require_gpu=pol["require_gpu"],
                                    allow_cpu_fallback=pol["allow_cpu_fallback"])
    ff.log(f"encoder: requested {enc.get('requested')} → {enc.get('actual')} ({enc.get('reason')})")
    _prefetch_fonts(ctx["style"])
    plan_p = Path(ctx["plan_path"])
    if not plan_p.exists() or fsutil.sha256_file(plan_p) != ctx["plan_sha"]:
        ff.log("plan.json differs from the accepted generation: restoring it before rendering")
        fsutil.atomic_write_text(plan_p, Path(ctx["snapshot"]).read_text(encoding="utf-8"))
    kw = dict(jobs=pol["jobs"], draft=pol["render_quality"] == "draft", encoder=enc["actual"],
              require_gpu=pol["require_gpu"], allow_cpu_fallback=pol["allow_cpu_fallback"],
              reel_timeout_s=pol["reel_timeout_s"], lang=ctx["lang"])
    m = batch.run(str(plan_p), only=ctx["only"], **kw)
    if ctx["only"] and any(isinstance(e, dict) and e.get("status") == "pending" for e in m.get("reels") or []):
        m = batch.run(str(plan_p), only=None, **kw)
    return {"manifest": m, "encoder": enc}


def _c_render(job, ctx, res):
    m = json.loads(json.dumps(res["manifest"], default=str))
    seq = job["render_seq"] = int(job.get("render_seq") or 0) + 1
    # the job keeps the exact manifest this render returned: review and delivery read that one,
    # whatever later happens to the batch directory
    p = _dir(job["id"]) / "manifests" / f"render.{seq}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    fsutil.atomic_write_json(p, m)
    job["render"] = {"seq": seq, "manifest": str(p), "batch_manifest": str(_manifest_path(m, job["id"])),
                     "generation": m.get("generation"), "plan_generation": ctx["plan_generation"],
                     "plan_sha256": ctx["plan_sha"], "counts": m.get("counts"), "summary": m.get("summary"),
                     "encoder": m.get("encoder") or res["encoder"], "at": _now()}
    job["encoder"] = res["encoder"]
    job["steps"]["render"] = {"done": True, "at": _now(), "key": ctx["key"], "seq": job["render_seq"]}
    _log(job, f"render generation {m.get('generation')}: {m.get('counts') or m.get('summary')} "
              f"(encoder {res['encoder'].get('actual')})", "rendered", generation=m.get("generation"),
         counts=m.get("counts"), encoder=res["encoder"])


def _w_deliver(ctx):
    out = {}
    for rid, f in ctx["files"]:
        ok = bool(f) and os.path.isfile(f)
        out[rid] = {"file": f, "exists": ok, "sha256": fsutil.sha256_file(f) if ok else None}
    return out


def _c_deliver(job, ctx, hashes):
    """Final verdict from the manifest's deliverables (hash-checked) and the review ledger."""
    if ctx["inputs_rev"] != job.get("inputs_rev"):
        _log(job, "inputs changed while delivering: re-evaluating", "warning")
        return
    manifest = _manifest(job)
    if not isinstance(manifest, dict):
        job["steps"].pop("render", None)
        _log(job, "batch manifest unreadable at delivery: rendering again", "warning")
        return
    pol, ledger, rq = job["policy"], job["review"]["ledger"], _rq(job)
    entries = _entries(job, manifest)
    spot = _spot_check(list(entries.values()), pol["spot_check"])
    deliverables = {d.get("id"): d for d in manifest.get("deliverables") or [] if isinstance(d, dict)}
    files, accepted_by, blocked, review_files = [], {}, {}, []
    for rid, e in entries.items():
        L = ledger.get(rid) or {}
        reviewed = _required(e, L, rid in spot)
        reason = None
        if not _entry_ok(e):
            reason = f"technical QC {_qc(e)}" + (f": {e.get('error')}" if e.get("error") else "")
        elif reviewed and not (L.get("decision") == "accept" and _decided(e, L)):
            reason = {"unreviewed": "no review result after every retry", "fix_required": "blocking review findings",
                      "repairing": "repair not completed"}.get(L.get("decision"), "not reviewed")
        d, h = deliverables.get(rid), hashes.get(rid) or {}
        if reason is None:
            if not d:
                reason = "not among the manifest's deliverables"
            elif not h.get("exists"):
                reason = "deliverable file missing"
            elif d.get("sha256") and h.get("sha256") != d["sha256"]:
                reason = "deliverable file differs from the manifest's hash"
            elif e.get("render_sha256") and h.get("sha256") != e["render_sha256"]:
                reason = "deliverable file differs from the rendered (and reviewed) reel"
        if reason:
            blocked[rid] = reason
            if e.get("file"):
                review_files.append(e["file"])
            continue
        files.append(d["file"])
        accepted_by[rid] = "review" if reviewed else "qc"
    unresolved = []
    for f in job["findings"]:
        if f.get("status") != "open":
            continue
        if f.get("reel_id") not in entries:
            f["status"] = "retired"
        elif f.get("reel_id") not in accepted_by:
            unresolved.append(_compact(f))
    for rid, why in blocked.items():
        if not any(u["reel_id"] == rid for u in unresolved):
            unresolved.append({"id": None, "reel_id": rid, "code": "BLOCKED", "severity": "blocking", "what": why,
                               "how": None, "status": "open"})
    requested = rq.get("count") if isinstance(rq.get("count"), int) and rq.get("count") > 0 else None
    shortfall = max(0, requested - len(files)) if requested else 0
    mc = manifest.get("counts") or {}
    counts = {"requested": requested, "planned": len(entries), "rendered": mc.get("rendered"),
              "qc_accepted": mc.get("accepted"), "accepted": len(files), "blocked": len(blocked),
              "failed": mc.get("failed"), "retired": len(manifest.get("retired") or []), "delivered": len(files)}
    if not files:
        st, why = "needs_attention", "no reel was accepted"
    elif not blocked and not unresolved and not shortfall:
        st, why = "succeeded", None
    else:
        st = "partial" if pol.get("allow_partial") else "needs_attention"
        why = "; ".join(x for x in (f"{len(blocked)} reel(s) blocked" if blocked else "",
                                    f"{shortfall} short of the {requested} requested" if shortfall else "") if x)
    sheet = Path(manifest["out_dir"]) / "contact_sheet.jpg" if manifest.get("out_dir") else None
    job["deliver"] = {"files": files, "review": review_files,
                      "contact_sheet": str(sheet) if sheet and sheet.exists() else None,
                      "manifest": job["render"]["manifest"], "generation": manifest.get("generation"),
                      "plan_generation": job["plan_generation"], "counts": counts, "unresolved": unresolved,
                      "shortfall": shortfall, "blocked": blocked, "accepted_by": accepted_by,
                      "encoder": job.get("encoder"), "summary": manifest.get("summary")}
    fsutil.atomic_write_json(_dir(job["id"]) / "delivery.json", job["deliver"])
    job["steps"]["deliver"] = {"done": True, "at": _now()}
    job.update(status=st, terminal=True, needs=None, attention=why if st == "needs_attention" else None)
    _log(job, f"{st}: {len(files)} of {len(entries)} planned reel(s) delivered" + (f" ({why})" if why else ""),
         st, counts=counts, shortfall=shortfall, unresolved=len(unresolved))


def _check_approved_context(ctx, plan_path):
    from . import storyboard
    b = ctx.get("storyboard") or {}
    a = ctx.get("storyboard_approval") or {}
    if (a.get("by") != "user" or a.get("sha256") != b.get("sha256")
            or a.get("plan_generation") != ctx.get("plan_generation")
            or not storyboard.intact(b) or b.get("key") != storyboard.fingerprint(
            plan_path, ctx["policy"]["storyboard"], ctx["policy"]["asr_provider"])):
        raise JobError("storyboard approval is missing or stale; regenerate and review the storyboard", code="approval_required")


def _w_storyboard(ctx):
    from . import storyboard
    return storyboard.build(ctx["plan"], ctx["root"], ctx["policy"]["storyboard"], ctx["provider"])


def _c_storyboard(job, ctx, board):
    if ctx["key"] != _key(job, "storyboard") or board["key"] != _storyboard_fingerprint(job):
        _log(job, "inputs changed during storyboarding; preview will be regenerated", "warning")
        return
    job["storyboard"] = board
    job["storyboard_approval"] = None
    job["steps"].pop("approve_storyboard", None)
    job["steps"]["storyboard"] = {"done": True, "key": ctx["key"], "at": _now()}
    _log(job, f"storyboard: {board['frames']} stills, {board['cache_hits']} reel previews reused", "storyboard_built")


_WORK = {"intake": _w_intake, "analyze": _w_analyze, "transcribe": _w_transcribe, "storyboard": _w_storyboard, "prepare": _w_prepare,
         "render": _w_render, "deliver": _w_deliver}
_COMMIT = {"intake": _c_intake, "analyze": _c_analyze, "transcribe": _c_transcribe, "storyboard": _c_storyboard, "prepare": _c_prepare,
           "render": _c_render, "deliver": _c_deliver}


def _preflight(jid, att):
    """After storyboard approval, check a required GPU before full preparation/rendering."""
    job = _load(jid)
    pol = job["policy"]
    if not _valid(job, "approve_storyboard"):
        return
    if job["terminal"] or not (pol.get("require_gpu") or pol.get("encoder") == "nvenc"):
        return
    from . import readiness
    try:
        enc = readiness.resolve_encoder(requested=pol["encoder"], require_gpu=pol["require_gpu"],
                                        allow_cpu_fallback=pol["allow_cpu_fallback"])
        ff.log(f"job {jid}: encoder {enc.get('actual')} ({enc.get('reason')})")
    except readiness.NotReady as e:
        with _txn(jid, att) as job:
            _fail(job, {"stage": "readiness", "cause": f"encoder not ready: {e}", "transient": False, "at": _now(),
                        "type": "NotReady"})


def _loop(jid, att, hb):
    """Advance until the job needs an agent, user approval or input, or ends. Transitions are short
    fenced transactions; the media work between them runs without the lock."""
    _preflight(jid, att)
    while True:
        if _cancel_flag(jid).exists():
            raise _Interrupted("cancel")
        with _txn(jid, att) as job:
            n = job.get("needs") or {}
            if n.get("step") in ("approve_storyboard", "review") and not _valid(job, "storyboard"):
                job["needs"] = None
                job["status"] = "waiting:worker"
            if job["terminal"] or job.get("needs"):
                job["_nosave"] = True
                return job
            step = _next_step(job)
            if step is None:
                _set_waiting(job)
                return job
            if step in JUDGMENT:
                if _open_need(job, step):
                    return job
                continue
            if step == "approve_storyboard":
                _need_storyboard_approval(job)
                return job
            ctx = _context(job, step)
            _mark_running(job, step)
        hb.beat(step)
        t0 = time.monotonic()
        try:
            res = _WORK[step](ctx)
        except Exception as e:
            err = _error_record(e, step)
            with _txn(jid, att) as job:
                _step_failed(job, step, err)
            continue
        with _txn(jid, att) as job:
            _COMMIT[step](job, ctx, res)
            _step_ok(job, step, time.monotonic() - t0)


# ------------------------------------------------------------------ worker processes

def _proc_stat(pid):
    """state, ppid, pgrp, start time (clock ticks) of a process from /proc; None if gone."""
    try:
        s = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    rest = s[s.rfind(")") + 2:].split()
    try:
        return {"state": rest[0], "ppid": int(rest[1]), "pgrp": int(rest[2]), "start": int(rest[19])}
    except (IndexError, ValueError):
        return None


def _pid_alive(pid, start=None):
    if not pid:
        return False
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(pid, os.WNOHANG)                 # reap our own exited child
    st = _proc_stat(pid)
    if st is None:
        if Path("/proc/self/stat").exists():
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        return True
    return st["state"] not in ("Z", "X") and (start is None or st["start"] == start)


def _procs():
    out = {}
    with contextlib.suppress(OSError):
        for d in os.listdir("/proc"):
            if d.isdigit():
                st = _proc_stat(d)
                if st and st["state"] not in ("Z", "X"):
                    out[int(d)] = st
    return out


def _attempt_live(rec):
    if not rec or rec.get("ended"):
        return False
    if rec.get("pid"):
        return _pid_alive(rec["pid"], rec.get("starttime"))
    return time.time() - float(rec.get("started_ts") or 0) < ACK_TIMEOUT + 15     # still starting


def _kill_attempt(rec, sig):
    """Signal an attempt's worker and everything it started. A detached worker leads its own
    process group (media subprocesses stay in it); a foreground one is signalled with its
    descendants. A dead leader's group is only reached through members that still exist, so a
    recycled pid is never hit."""
    pid, pgid = rec.get("pid"), rec.get("pgid")
    if not pid:
        return
    leader = _pid_alive(pid, rec.get("starttime"))
    procs = _procs()
    targets = set()
    if pgid and not rec.get("foreground"):
        if leader:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, sig)
        targets |= {p for p, st in procs.items() if st["pgrp"] == pgid and p != pgid}
    if leader:
        kids, todo = set(), [pid]
        while todo:
            cur = todo.pop()
            for p, st in procs.items():
                if st["ppid"] == cur and p not in kids:
                    kids.add(p)
                    todo.append(p)
        targets |= kids | {pid}
    for p in targets:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(p, sig)


def _read_attempt(jid, att):
    try:
        return _rj(_att_path(jid, att))
    except JobError:
        return None


def _end_attempt(jid, att, code, err):
    """Terminal record for an attempt that did not write its own (never overwrites one)."""
    rec = _read_attempt(jid, att) or {"id": att, "job": jid}
    if rec.get("ended"):
        return rec
    rec.update(ended=_now(), ended_ts=round(time.time(), 3), exit=code, error_record=err)
    _att_path(jid, att).parent.mkdir(parents=True, exist_ok=True)
    fsutil.atomic_write_json(_att_path(jid, att), rec)
    return rec


def _close_dead(jid, job, rec):
    """Close an attempt whose process is gone without a terminal record (inside a transaction)."""
    cur = _read_attempt(jid, rec["id"]) or rec
    if cur.get("ended") or _attempt_live(cur):
        return None
    err = {"stage": cur.get("stage") or "start", "cause": "worker_died", "transient": True, "at": _now(),
           "detail": f"attempt {cur['id']} (pid {cur.get('pid')}) stopped without a terminal record",
           "log_tail": _log_tail(cur)}
    cur = _end_attempt(jid, cur["id"], None, err)
    cur["died"] = True
    fsutil.atomic_write_json(_att_path(jid, cur["id"]), cur)
    if job.get("attempt") == cur["id"] and not job["terminal"]:
        job["error"] = err
        job["errors"] = (job.get("errors") or [])[-29:] + [err]
        if job["status"].startswith("running:") or job["status"] == "intake":
            job["status"] = "waiting:worker"
        _log(job, f"worker {cur['id']} died during {err['stage']}", "worker_died", error=err)
    return cur


def _reap(jid, att):
    dead = None
    with _txn(jid) as job:
        rec = _read_attempt(jid, att)
        dead = _close_dead(jid, job, rec) if rec else None
        if dead is None:
            job["_nosave"] = True
    if dead:
        _kill_attempt(dead, signal.SIGKILL)       # orphaned media processes
    return dead


def _claim(jid, foreground):
    """("new", attempt, record) owning the job — or ("live", …) when a live attempt already has it,
    or ("terminal", None, None). A previous attempt that died without a record is closed and its
    leftovers killed (they would hold the run lock)."""
    dead = None
    with _txn(jid) as job:
        if job["terminal"]:
            job["_nosave"] = True
            return "terminal", None, None
        cur = job.get("attempt")
        rec = _read_attempt(jid, cur) if cur else None
        if rec and not rec.get("ended"):
            if _attempt_live(rec):
                job["_nosave"] = True
                return "live", cur, rec
            dead = _close_dead(jid, job, rec)
        seq = int(job.get("attempt_seq") or 0) + 1
        att = f"{'f' if foreground else 'a'}{seq:03d}-{secrets.token_hex(2)}"
        path = _att_path(jid, att)
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"id": att, "job": jid, "seq": seq, "foreground": foreground, "started": _now(),
               "started_ts": round(time.time(), 3), "pid": None, "pgid": None, "starttime": None,
               "acknowledged": False, "heartbeat": None, "stage": None, "ended": None, "exit": None,
               "error_record": None, "log": None if foreground else str(path.with_suffix(".log"))}
        fsutil.atomic_write_json(path, rec)
        job.update(attempt=att, attempt_seq=seq, error=None)
        _log(job, f"attempt {att} claimed ({'foreground' if foreground else 'detached'})", "attempt_claimed")
    if dead:
        _kill_attempt(dead, signal.SIGKILL)
    return "new", att, rec


def _run_lock(jid, wait_s=15.0):
    """Exclusive, non-inheritable lock held by the worker for its whole run; None if busy."""
    p = _dir(jid) / "run.lock"
    deadline = time.monotonic() + wait_s
    while True:
        fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            os.close(fd)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.2)


class _Heartbeat(threading.Thread):
    """Owns the attempt record while the worker runs: acknowledgement, heartbeat every
    HEARTBEAT_S, current stage, terminal record."""

    def __init__(self, jid, att, foreground):
        super().__init__(name="reel-heartbeat", daemon=True)
        self.path = _att_path(jid, att)
        self.rec = _rj(self.path) or {"id": att, "job": jid}
        st = _proc_stat(os.getpid())
        self.rec.update(pid=os.getpid(), pgid=None if foreground else os.getpgrp(), starttime=st and st["start"],
                        foreground=foreground, acknowledged=True, acknowledged_at=_now(), host=os.uname().nodename)
        self.lock = threading.Lock()
        self.halt = threading.Event()
        self.stage = "starting"

    def _write(self, **kw):
        with self.lock:
            self.rec.update(kw, stage=self.stage, heartbeat=_now(), heartbeat_ts=round(time.time(), 3))
            fsutil.atomic_write_json(self.path, self.rec)

    def begin(self):
        self._write()
        self.start()

    def beat(self, stage):
        self.stage = stage
        self._write()

    def run(self):
        while not self.halt.wait(HEARTBEAT_S):
            with contextlib.suppress(Exception):      # a missed beat is not fatal: liveness is the pid
                self._write()

    def finish(self, code, err):
        self.halt.set()
        if self.is_alive():
            self.join(timeout=5)
        self._write(ended=_now(), ended_ts=round(time.time(), 3), exit=code, error_record=err)


_MAIN_PID = None
_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def _on_signal(signum, frame):
    if os.getpid() != _MAIN_PID:      # a forked pool worker inherited this handler: die the default way
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        return
    raise _Interrupted(signal.Signals(signum).name)


def _interrupted(jid, att, why, hb):
    stage = hb.stage if hb else "start"
    err = {"stage": stage, "cause": f"worker interrupted ({why})", "transient": True, "at": _now()}
    try:
        if _cancel_flag(jid).exists():
            with _txn(jid) as job:
                if job["terminal"]:
                    job["_nosave"] = True
                else:
                    _cancelled(job)
            return 130, {"stage": stage, "cause": "cancelled", "transient": False, "at": _now()}
        with _txn(jid, att) as job:
            if job["terminal"]:
                job["_nosave"] = True
            else:
                if job["status"].startswith("running:"):
                    job["status"] = "waiting:worker"
                job["error"] = err
                _log(job, f"worker interrupted ({why}) during {stage}: `reel job start` resumes", "worker_interrupted",
                     error=err)
    except BaseException as e:
        ff.log(f"job {jid}: interruption not recorded ({type(e).__name__}: {e})")
    return (130 if why == "SIGINT" else 143), err


def work(jid, att, foreground=False):
    """Run the job as attempt `att`: the detached worker's body (`reel job _work`), or in-process
    for `advance`. Takes the run lock, acknowledges, heartbeats, advances, and always leaves a
    terminal attempt record. Returns its exit code (0 = the job settled; 1 = it failed;
    3 = another worker holds the job; 4 = superseded; 130/143 = cancelled/interrupted)."""
    global _MAIN_PID
    _check_id(att, "attempt id")
    fd = _run_lock(jid)
    if fd is None:
        _end_attempt(jid, att, 3, {"stage": "start", "cause": "another worker holds the job's run lock",
                                   "transient": True, "at": _now()})
        return 3
    _MAIN_PID = os.getpid()
    old = {s: signal.signal(s, _on_signal) for s in _SIGNALS}
    hb, code, err = None, 1, None
    try:
        try:
            with _txn(jid, att, save=False):
                pass
            hb = _Heartbeat(jid, att, foreground)
            hb.begin()
            _emit(jid, "worker_started", att, pid=os.getpid(), foreground=foreground)
            job = _loop(jid, att, hb)
            failed = job["status"] == "failed"
            code, err = (1, job.get("error")) if failed else (0, None)
        except (_Interrupted, KeyboardInterrupt) as e:
            code, err = _interrupted(jid, att, getattr(e, "why", "SIGINT"), hb)
        except _Superseded as e:
            code, err = 4, {"stage": hb.stage if hb else "start", "cause": f"superseded: {e}", "transient": False,
                            "at": _now()}
        except Exception as e:
            code, err = 1, _error_record(e, hb.stage if hb else "start")
            try:
                with _txn(jid, att) as job:
                    _fail(job, err)
            except BaseException as e2:
                ff.log(f"job {jid}: failure not recorded ({type(e2).__name__}: {e2})")
    finally:
        for s in _SIGNALS:                  # finish the records undisturbed
            signal.signal(s, signal.SIG_IGN)
        try:
            if hb:
                hb.finish(code, err)
            else:
                _end_attempt(jid, att, code, err)
            _emit(jid, "worker_ended", att, stage=hb.stage if hb else None, exit=code, error=err)
        except BaseException as e:
            ff.log(f"job {jid}: attempt record not finished ({type(e).__name__}: {e})")
        finally:
            for s, h in old.items():
                signal.signal(s, h)
            os.close(fd)
    return code


def _spawn(jid, att):
    log = _att_path(jid, att).with_suffix(".log")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        with open(log, "ab") as f:
            return subprocess.Popen([sys.executable, "-m", "reelkit", "job", "_work", jid, "--attempt", att],
                                    cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True, env=env)
    except OSError as e:
        _end_attempt(jid, att, None, {"stage": "start", "cause": f"could not spawn the worker: {e}", "transient": True,
                                      "at": _now()})
        raise JobError(f"could not spawn a worker: {e}", code="spawn_failed") from None


def _await_ack(jid, att, proc):
    deadline = time.monotonic() + ACK_TIMEOUT
    while True:
        rec = _read_attempt(jid, att) or {}
        if rec.get("acknowledged"):
            return rec
        if rec.get("ended"):
            err = rec.get("error_record") or {}
            raise JobError(f"worker {att} stopped before taking the job: {err.get('cause')}", [err],
                           code="worker_failed")
        exited = proc is not None and proc.poll() is not None
        if exited or time.monotonic() >= deadline or (proc is None and not _attempt_live(rec)):
            if exited:
                rec = _read_attempt(jid, att) or {}     # it may have acknowledged and finished meanwhile
                if rec.get("acknowledged"):
                    return rec
            why = (f"exited with code {proc.returncode} before acknowledging" if exited
                   else f"did not acknowledge within {ACK_TIMEOUT}s")
            if proc is not None and not exited:
                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    proc.wait(5)
            err = {"stage": "start", "cause": f"worker {why}", "transient": True, "at": _now(),
                   "log_tail": _log_tail(rec)}
            _end_attempt(jid, att, None, err)
            raise JobError(f"worker {att} {why} (log: {rec.get('log')})", [err], code="worker_failed")
        time.sleep(0.1)


def start(jid):
    """Spawn a detached worker (its own session, outliving this process) and return once it has
    acknowledged ownership. Idempotent: a live worker is reported, not duplicated."""
    kind, att, _ = _claim(jid, foreground=False)
    if kind == "terminal":
        return {"attempt": None, "pid": None, "acknowledged": False, "terminal": True,
                "status": _load(jid)["status"], "note": "the job is terminal: nothing to run"}
    proc = _spawn(jid, att) if kind == "new" else None
    rec = _await_ack(jid, att, proc)
    return {"attempt": att, "pid": rec.get("pid"), "pgid": rec.get("pgid"), "acknowledged": True,
            "existing": kind == "live", "log": rec.get("log"), "status": _load(jid)["status"]}


def advance(jid):
    """Run the job in this process until it needs an agent or input, or ends — same run lock,
    attempt record and fencing as a detached worker. Returns the job (job["busy"] when another
    worker owns it; job["worker_exit"] = work()'s exit code when this call ran it)."""
    kind, att, _ = _claim(jid, foreground=True)
    if kind == "live":
        job = _load(jid)
        job["busy"] = True
        return job
    code = work(jid, att, foreground=True) if kind == "new" else None
    job = _load(jid)
    job["worker_exit"] = code
    return job


def wait(jid, timeout=None, poll=2.0):
    """Block until the job's worker stops; returns (exit code, status). 0 = it settled (an agent
    or input is needed, or the job is terminal), 2 = the worker failed or died (status.error
    says how), 124 = still running at the timeout."""
    deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
    while True:
        job = _load(jid)
        att = job.get("attempt")
        rec = _read_attempt(jid, att) if att else None
        if not _attempt_live(rec):
            if rec and not rec.get("ended"):
                _reap(jid, att)
                return 2, status(jid)
            if job["terminal"]:
                return (2 if job["status"] == "failed" else 0), status(jid, job)
            st = status(jid, job)
            if rec is None:
                if job.get("needs") or job["status"] in ("waiting:sample", "waiting:source"):
                    return 0, st
                return 2, {**st, "error": st.get("error") or {"cause": "no worker has run this job: `reel job start` it"}}
            return (0 if rec.get("exit") == 0 else 2), st
        if deadline is not None and time.monotonic() >= deadline:
            return 124, status(jid, job)
        time.sleep(poll)


def cancel(jid):
    """Raise the cancel flag (the worker stops between steps), SIGTERM the worker's process group,
    SIGKILL whatever is left after CANCEL_GRACE_S, and mark the job cancelled."""
    job = _load(jid)
    if job["terminal"]:
        return status(jid, job)
    att = job.get("attempt")
    fsutil.atomic_write_text(_cancel_flag(jid), _now() + "\n")
    _emit(jid, "cancel_requested", att)
    rec = _read_attempt(jid, att) if att else None
    if rec and not rec.get("ended") and rec.get("pid"):
        if _attempt_live(rec):
            _kill_attempt(rec, signal.SIGTERM)
            end = time.monotonic() + CANCEL_GRACE_S
            while time.monotonic() < end and _attempt_live(rec):
                time.sleep(0.2)
        _kill_attempt(rec, signal.SIGKILL)
    with _txn(jid) as job:
        if job["terminal"]:
            job["_nosave"] = True
        else:
            _cancelled(job)
    if rec:
        _end_attempt(jid, att, None, {"stage": rec.get("stage"), "cause": "cancelled", "transient": False, "at": _now()})
    return status(jid)


def retry(jid):
    """Reopen a failed or cancelled job; `start` then continues from the last valid step."""
    with _txn(jid) as job:
        if job["status"] not in ("failed", "cancelled"):
            raise JobError(f"job {jid} is {job['status']}: only failed or cancelled jobs can be retried",
                           code="not_retryable")
        with contextlib.suppress(FileNotFoundError):
            _cancel_flag(jid).unlink()
        job.update(status="waiting:worker", terminal=False, error=None, attention=None, needs=None)
        for sa in (job.get("stage_attempts") or {}).values():
            sa["failures"] = 0
        _log(job, "reopened: `reel job start` continues from the last valid step", "retried")
    return status(jid)


# ------------------------------------------------------------------ status

def _attempt_info(job):
    att = job.get("attempt")
    rec = _read_attempt(job["id"], att) if att else None
    if not rec:
        return None
    running = _attempt_live(rec)
    hb = rec.get("heartbeat_ts")
    return {"id": att, "pid": rec.get("pid"), "foreground": rec.get("foreground"),
            "acknowledged": rec.get("acknowledged"), "running": running,
            "state": "running" if running else ("ended" if rec.get("ended") else "died"),
            "stage": rec.get("stage"), "started": rec.get("started"), "heartbeat": rec.get("heartbeat"),
            "heartbeat_age_s": round(time.time() - hb, 1) if hb else None, "ended": rec.get("ended"),
            "exit": rec.get("exit"), "error_record": rec.get("error_record"), "log": rec.get("log")}


def _next_hint(job, att):
    if job["terminal"]:
        return "none"
    if att and att["running"]:
        return "wait"
    if job.get("needs"):
        if job["needs"]["step"] == "approve_storyboard":
            return "user:approve_storyboard"
        return f"agent:{job['needs']['agent']}"
    if job["status"] in ("waiting:sample", "waiting:source"):
        return "input"
    return "start"


def status(jid, job=None):
    job = job or _load(jid)
    att = _attempt_info(job)
    d = job.get("deliver") or {}
    return {"id": job["id"], "status": job["status"], "terminal": bool(job["terminal"]),
            "revision": job.get("revision"), "next": _next_hint(job, att), "needs": job.get("needs"),
            "attempt": att, "busy": bool(att and att["running"]),
            "done": [s for s in ORDER if _valid(job, s)], "inputs": job["inputs"],
            "files": [{k: f.get(k) for k in ("path", "role", "role_source", "suggested_role", "asset", "error")}
                      for f in job["files"]],
            "waiting": job.get("waiting") or [], "deliver": job.get("deliver"),
            "counts": d.get("counts") or (job.get("render") or {}).get("counts"),
            "findings_open": [_compact(f) for f in job.get("findings") or [] if f.get("status") == "open"][:100],
            "plan_generation": job.get("plan_generation"), "style_version": job.get("style_version"),
            "storyboard": job.get("storyboard"), "storyboard_approval": job.get("storyboard_approval"),
            "encoder": job.get("encoder"), "error": job.get("error"), "attention": job.get("attention"),
            "request_hash": job.get("request_hash"), "policy_version": (job.get("policy") or {}).get("version"),
            "log": (job.get("log") or [])[-8:]}
