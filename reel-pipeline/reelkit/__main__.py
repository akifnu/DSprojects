"""`reel` command line. Machine-readable results go to stdout (JSON for `job`, `check --json`
and `batch`), logs and progress to stderr.

  reel check [--json] [--encoder auto|cpu|nvenc] [--require-gpu] [--asr-provider cpu|cuda] [--deep]
                                             readiness: tools, models, encoder, ASR provider (exit 1 = not ready)
  reel ingest FILE… [--kind sample|source|music|sfx|image]
  reel import-url URL [--kind sample|source|music|sfx]   best available streams via yt-dlp
  reel assets [--kind K]
  reel sample REF [--lang en] [--force]      measured style spec + review sheet
  reel source REF [--lang en] [--prepare] [--asr-provider cpu|cuda]   transcript (+ editing copy)
  reel transcript REF [--lang en] [--asr-provider cpu|cuda]           compact sentence index for planning
  reel plan NAME --style REF --source S=REF [--lang en]  scaffold data/batches/NAME/plan.json
  reel batch PLAN [-j N] [--draft] [--only r01,r02] [--encoder auto|cpu|nvenc] [--require-gpu] [--no-cpu-fallback]
  reel render TIMELINE OUT [--draft] [--still T] [--encoder auto|cpu|nvenc]
  reel qc MP4 [--timeline T] [--style S]
  reel loop CLIP SECONDS OUT                 seamless endless loop

  reel job new [FILE…] [--sample F]… [--source F]… [--music F]… [--sfx F]… [--image F]…
               [--request request.json] [--count N] [--duration S] [--lang xx] [--notes T | --notes-file F]
               [--encoder auto|cpu|nvenc] [--require-gpu] [--no-cpu-fallback] [--asr-provider cpu|cuda] [--name N]
                                             bare FILEs get a guessed role (recorded as a suggestion)
  reel job add ID [FILE…] [--sample F]… [--source F]… [--music F]… [--sfx F]… [--image F]…
  reel job start ID                          detached worker; returns once it acknowledged the job
  reel job wait ID [--timeout S]             exit 0 settled (agent / input needed, or terminal),
                                             2 worker failed or died, 124 still running
  reel job advance [ID]                      run in this process (same lock and records as start)
  reel job cancel ID | reel job retry ID
  reel job done ID STEP --need NEED_ID [--revision N] [--note T | --note-file F] [--result-file F]
                                             an agent answered verify_style | plan | review
  reel job status [ID|latest] | reel job list

ID may be "latest" (the most recently created job), resolved once here. Job errors print
{"error", "code", "errors"} with exit 2 for a rejected answer, 1 otherwise.
"""
import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

ROLES = ("sample", "source", "music", "sfx", "image")
ENCODERS = ("auto", "cpu", "nvenc")


def _out(obj):
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str), flush=True)


@contextlib.contextmanager
def _stdout_to_stderr():
    """While media work runs in this process, anything written to fd 1 (Python or native
    libraries, child processes) goes to stderr: stdout carries only the JSON result."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _not_ready(e):
    """NotReady from readiness (encoder / GPU requirement) as a JSON error; anything else re-raised."""
    if type(e).__name__ != "NotReady":
        raise e
    _out({"error": str(e), "code": "not_ready", "errors": [str(e)]})
    return 1


def _asset(ref, kind):
    from . import store
    try:
        return store.get(ref)
    except KeyError:
        if Path(ref).exists():
            return store.ingest(ref, kind)
        raise SystemExit(f"unknown asset or file: {ref}")


def _provider(a):
    return {"provider": a.asr_provider} if getattr(a, "asr_provider", None) else {}


# ------------------------------------------------------------------ media commands

def cmd_ingest(a):
    from . import store
    for f in a.files:
        x = store.ingest(f, a.kind)
        print(x.id, x.kind, x.meta["name"])


def cmd_import_url(a):
    from .download import ingest_url
    try:
        with _stdout_to_stderr():
            x = ingest_url(a.url, a.kind)
    except (ValueError, RuntimeError) as e:
        _out({"error": str(e), "code": "import_failed"})
        return 1
    _out({"id": x.id, "kind": x.kind, "file": str(x.path), "sha256": x.meta["sha256"],
          "provenance": str(x.cache("import.json"))})


def cmd_assets(a):
    from . import store
    for x in store.all_assets(a.kind):
        i = x.info
        print(f"{x.id}  {x.kind:7s} {i.get('duration', 0):8.1f}s  {i.get('width', '-')}x{i.get('height', '-')}  "
              f"{x.meta['name']}")


def cmd_sample(a):
    from . import sample, store
    st = sample.analyze(_asset(a.ref, "sample").id, a.lang, a.force)
    x = store.get(st["sample"]["id"])
    print(f"style:  {x.cache('style.json')}\nreview: {x.cache('review.jpg')}")


def cmd_source(a):
    from . import asr, fsutil, source
    x = _asset(a.ref, "source")
    tr = source.transcript(x, a.lang, **_provider(a))
    p = x.cache(f"transcript.{a.lang}.txt")
    fsutil.atomic_write_text(p, asr.compact(tr))
    print(f"{x.id}: {len(tr['sentences'])} sentences, {tr['duration']:.0f}s → {p}")
    if a.prepare:
        source.prepare(x)


def cmd_transcript(a):
    from . import asr, source
    tr = source.transcript(_asset(a.ref, "source"), a.lang, **_provider(a))
    print(asr.compact(tr))


def cmd_plan(a):
    from .paths import DATA
    d = DATA / "batches" / a.name
    d.mkdir(parents=True, exist_ok=True)
    srcs = dict(s.split("=", 1) for s in a.source)
    plan = {"name": a.name, "style": _asset(a.style, "sample").id,
            "output": {"width": 720, "height": 1280, "fps": 30, "quality": "final", "lang": a.lang},
            "sources": {k: _asset(v, "source").id for k, v in srcs.items()},
            "music": None, "sfx": None, "reels": []}
    p = d / "plan.json"
    if p.exists():
        raise SystemExit(f"{p} exists")
    p.write_text(json.dumps(plan, indent=1))
    print(p)


def cmd_batch(a):
    from . import batch
    from .paths import DATA
    plan = Path(a.plan)
    if not plan.exists():
        plan = DATA / "batches" / a.plan / "plan.json"
    kw = {}
    if a.require_gpu:
        kw["require_gpu"] = True
    if a.allow_cpu_fallback is False:
        kw["allow_cpu_fallback"] = False
    try:
        with _stdout_to_stderr():
            m = batch.run(str(plan), jobs=a.jobs, draft=a.draft, only=a.only.split(",") if a.only else None,
                          encoder=a.encoder, **kw)
    except Exception as e:
        return _not_ready(e)
    _out({k: m.get(k) for k in ("name", "generation", "encoder", "counts", "summary", "deliverables", "retired",
                                "out_dir")})


def cmd_render(a):
    from . import render
    if a.still is not None:
        print(render.still(a.timeline, a.still, a.out))
        return
    try:
        print(render.render(a.timeline, a.out, draft=a.draft, encoder=a.encoder))
    except Exception as e:
        return _not_ready(e)


def cmd_qc(a):
    from . import qc
    tl = json.loads(Path(a.timeline).read_text()) if a.timeline else None
    st = json.loads(Path(a.style).read_text()) if a.style else None
    res = qc.check(a.mp4, timeline=tl, style=st)
    print(json.dumps(res, indent=1)[:4000])
    return 0 if res.get("verdict") != "fail" else 2


def cmd_loop(a):
    from . import loop
    loop.main([a.clip, str(a.seconds), a.out])


# ------------------------------------------------------------------ jobs

def _read_text(orch, path, what):
    try:
        return Path(path).expanduser().read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise orch.JobError(f"cannot read the {what} file {path}", [f"{path}: {e}"], code="invalid_request") from None


def _read_request(orch, path):
    """request.json as a dict; relative asset paths are taken relative to the request file
    (or to the working directory when only that resolves)."""
    p = Path(path).expanduser()
    try:
        req = json.loads(_read_text(orch, p, "request"))
    except ValueError as e:
        raise orch.JobError(f"{p} is not valid JSON", [f"{p}: {e}"], code="invalid_request") from None
    if isinstance(req, dict) and isinstance(req.get("assets"), list):
        base = p.resolve().parent
        for x in req["assets"]:
            if isinstance(x, dict) and isinstance(x.get("path"), str) and x["path"]:
                q = Path(x["path"]).expanduser()
                if not q.is_absolute():
                    x["path"] = str(q.resolve() if not (base / q).exists() and q.exists() else base / q)
    return req


def _roles(a):
    return {r: list(getattr(a, r)) for r in ROLES if getattr(a, r, None)}


def _job_new(orch, a):
    notes = _read_text(orch, a.notes_file, "notes") if a.notes_file else a.notes
    execution = {"encoder": a.encoder, "require_gpu": a.require_gpu, "allow_cpu_fallback": a.allow_cpu_fallback,
                 "asr_provider": a.asr_provider}
    job = orch.create(a.files, count=a.count, lang=a.lang, duration=a.duration, notes=notes, name=a.name,
                      roles=_roles(a), request=_read_request(orch, a.request) if a.request else None,
                      execution=execution)
    _out({**orch.status(job["id"]), "request": job.get("request"), "warnings": job.get("request_warnings") or []})


def _job_add(orch, a):
    _out(orch.add(orch.resolve(a.id), a.files, roles=_roles(a)))


def _job_start(orch, a):
    _out(orch.start(orch.resolve(a.id)))


def _job_wait(orch, a):
    code, st = orch.wait(orch.resolve(a.id), timeout=a.timeout)
    _out({**st, "wait_exit": code})
    return code


def _job_advance(orch, a):
    jid = orch.resolve(a.id)
    with _stdout_to_stderr():
        job = orch.advance(jid)
    st = orch.status(jid)
    _out({**st, "busy": bool(job.get("busy") or st.get("busy")), "worker_exit": job.get("worker_exit")})
    code = job.get("worker_exit")
    if code in (130, 143):
        return code
    if code not in (None, 0) or st["status"] == "failed":
        return 2
    return 0


def _job_cancel(orch, a):
    _out(orch.cancel(orch.resolve(a.id)))


def _job_retry(orch, a):
    _out(orch.retry(orch.resolve(a.id)))


def _job_done(orch, a):
    jid = orch.resolve(a.id)
    note = _read_text(orch, a.note_file, "note") if a.note_file else a.note
    fixes = None
    if a.fixes is not None:
        try:
            fixes = json.loads(a.fixes)
        except ValueError as e:
            raise orch.Rejected("--fixes is not valid JSON", [str(e)], code="bad_json") from None
    ok = [x.strip() for x in a.ok.split(",") if x.strip()] if a.ok is not None else None
    _out(orch.complete(jid, a.step, need_id=a.need, revision=a.revision, note=note or "",
                       result_file=a.result_file, fixes=fixes, ok=ok))


def _job_status(orch, a):
    _out(orch.status(orch.resolve(a.id)))


def _job_list(orch, a):
    _out(orch.list_jobs())


def _job_work(orch, a):
    """The detached worker's body (spawned by `start`); stdout / stderr are the attempt log."""
    jid = orch.resolve(a.id)
    code = orch.work(jid, a.attempt)
    _out({"id": jid, "attempt": a.attempt, "exit": code})
    return code


_JOB = {"new": _job_new, "add": _job_add, "start": _job_start, "wait": _job_wait, "advance": _job_advance,
        "cancel": _job_cancel, "retry": _job_retry, "done": _job_done, "status": _job_status, "list": _job_list,
        "_work": _job_work}


def cmd_job(a):
    from . import orchestrator as orch
    try:
        return _JOB[a.action](orch, a) or 0
    except orch.Rejected as e:
        _out({"error": str(e), "code": e.code, "errors": e.errors})
        return 2
    except orch.JobError as e:
        _out({"error": str(e), "code": e.code, "errors": e.errors})
        return 1
    except TimeoutError as e:
        _out({"error": f"the job's state lock stayed busy: {e}", "code": "lock_timeout", "errors": []})
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        _out({"error": f"{type(e).__name__}: {e}", "code": "internal", "errors": []})
        return 1


# ------------------------------------------------------------------ parser

def _role_flags(p):
    for r in ROLES:
        p.add_argument(f"--{r}", action="append", metavar="FILE", help=f"a {r} file (repeatable)")


def _job_parser(sub):
    p = sub.add_parser("job", help="agentic job orchestration (JSON on stdout)")
    js = p.add_subparsers(dest="action", required=True, metavar="ACTION")
    q = js.add_parser("new", help="register inputs and a request")
    q.add_argument("files", nargs="*", help="files without a declared role (role guessed, recorded as a suggestion)")
    _role_flags(q)
    q.add_argument("--request", help="request.json {schema_version, assets, request, execution}")
    q.add_argument("--count", type=int)
    q.add_argument("--duration", type=float, help="target seconds per reel")
    q.add_argument("--lang", help="transcript / caption language (default en)")
    g = q.add_mutually_exclusive_group()
    g.add_argument("--notes")
    g.add_argument("--notes-file")
    q.add_argument("--encoder", choices=ENCODERS)
    q.add_argument("--require-gpu", action="store_const", const=True, default=None)
    q.add_argument("--no-cpu-fallback", dest="allow_cpu_fallback", action="store_const", const=False, default=None)
    q.add_argument("--asr-provider", choices=("cpu", "cuda"))
    q.add_argument("--name", help="explicit job id (must not exist)")
    q = js.add_parser("add", help="more inputs for an open job")
    q.add_argument("id")
    q.add_argument("files", nargs="*")
    _role_flags(q)
    for name, hlp in (("start", "spawn a detached worker"), ("cancel", "stop the worker, mark cancelled"),
                      ("retry", "reopen a failed or cancelled job")):
        js.add_parser(name, help=hlp).add_argument("id")
    q = js.add_parser("wait", help="block until the worker stops")
    q.add_argument("id")
    q.add_argument("--timeout", type=float)
    for name, hlp in (("advance", "run the job in this process"), ("status", "job status")):
        js.add_parser(name, help=hlp).add_argument("id", nargs="?", default="latest")
    q = js.add_parser("done", help="record an agent's answer to the open need")
    q.add_argument("id")
    q.add_argument("step", help="verify_style | plan | review")
    q.add_argument("--need", help="needs.need_id being answered")
    q.add_argument("--revision", type=int, help="needs.revision being answered")
    g = q.add_mutually_exclusive_group()
    g.add_argument("--note")
    g.add_argument("--note-file")
    q.add_argument("--result-file", help="the answer (default: the need's own file)")
    q.add_argument("--fixes", help='legacy review: JSON [{"reel": "r03", "what": "...", "how": "..."}]')
    q.add_argument("--ok", help="legacy review: comma-separated accepted reel ids")
    js.add_parser("list", help="all jobs")
    q = js.add_parser("_work")           # the detached worker (spawned by start; not listed)
    q.add_argument("id")
    q.add_argument("--attempt", required=True)
    p.set_defaults(fn=cmd_job)


def _parser():
    ap = argparse.ArgumentParser(prog="reel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", add_help=False, help="readiness (see reel check --help)")
    p = sub.add_parser("ingest")
    p.add_argument("files", nargs="+")
    p.add_argument("--kind", default="source", choices=("sample", "source", "music", "sfx", "image", "other"))
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("import-url", help="download the best available media streams with yt-dlp")
    p.add_argument("url")
    p.add_argument("--kind", default="source", choices=("sample", "source", "music", "sfx"))
    p.set_defaults(fn=cmd_import_url)
    p = sub.add_parser("assets")
    p.add_argument("--kind")
    p.set_defaults(fn=cmd_assets)
    p = sub.add_parser("sample")
    p.add_argument("ref")
    p.add_argument("--lang", default="en")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_sample)
    for name, fn in (("source", cmd_source), ("transcript", cmd_transcript)):
        p = sub.add_parser(name)
        p.add_argument("ref")
        p.add_argument("--lang", default="en")
        p.add_argument("--asr-provider", choices=("cpu", "cuda"))
        if name == "source":
            p.add_argument("--prepare", action="store_true")
        p.set_defaults(fn=fn)
    p = sub.add_parser("plan")
    p.add_argument("name")
    p.add_argument("--style", required=True)
    p.add_argument("--source", action="append", required=True, help="KEY=asset-or-file")
    p.add_argument("--lang", default="en")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("batch")
    p.add_argument("plan")
    p.add_argument("-j", "--jobs", type=int)
    p.add_argument("--draft", action="store_true")
    p.add_argument("--only")
    p.add_argument("--encoder", choices=ENCODERS, default="auto")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--no-cpu-fallback", dest="allow_cpu_fallback", action="store_const", const=False, default=None)
    p.set_defaults(fn=cmd_batch)
    p = sub.add_parser("render")
    p.add_argument("timeline")
    p.add_argument("out")
    p.add_argument("--draft", action="store_true")
    p.add_argument("--still", type=float)
    p.add_argument("--encoder", choices=ENCODERS, default="auto")
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("qc")
    p.add_argument("mp4")
    p.add_argument("--timeline")
    p.add_argument("--style")
    p.set_defaults(fn=cmd_qc)
    p = sub.add_parser("loop")
    p.add_argument("clip")
    p.add_argument("seconds", type=float)
    p.add_argument("out")
    p.set_defaults(fn=cmd_loop)
    _job_parser(sub)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "check":         # readiness owns its flags and output
        from . import readiness
        return readiness.main(argv[1:])
    ap = _parser()
    a, extra = ap.parse_known_args(argv)
    if extra:
        # files given after options (`job new --count 3 a.mp4`) land here
        if getattr(a, "action", None) in ("new", "add") and not any(x.startswith("-") for x in extra):
            a.files = list(a.files or []) + extra
        else:
            ap.error("unrecognized arguments: " + " ".join(extra))
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())

