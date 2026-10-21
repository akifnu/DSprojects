"""Batch production: plan.json → N finished, QC'd reels, unattended.

    run(plan_path, jobs=None, draft=False, only=None, encoder="auto", asr=True, compiler=None,
        auto_calibrate=True, require_gpu=False, allow_cpu_fallback=True, reel_timeout_s=None,
        lang=None, asr_provider=None) -> manifest

1. Validate the plan: the batch's own rules plus contracts.validate_plan (again with the
   transcripts once they are loaded) — fail fast, every problem listed at once.
2. Resolve the encoder once (cpu | nvenc; auto / nvenc through readiness.resolve_encoder).
   Requested, actual and the reason are logged and go into the manifest and the render keys.
3. Timelines, in this process, in plan order: a ready timeline (a reel carrying "timeline":
   path.json; its media made absolute), the cached compile (timeline.json + deps.json) when the
   plan entry, style and source inputs are unchanged and every dependency still matches, or
   compile.compile_reel through one shared compile.Context (transcripts, envelopes, editing
   copies for the union of all reels' runs, source looks — each made once). Every timeline is
   checked with contracts.validate_timeline; an invalid one fails its reel without a render.
4. Keys. render_key = hash(timeline + deps.json + actual encoder + draft + renderer code, ffmpeg
   and library versions), paths under the data / code roots written relative so a moved store
   still matches. qc_key = hash(style + ASR setting and fingerprint + language + QC code). A reel
   is reused when its record (reels/<id>/render.json once rendered, cache.json once QC'd too) has
   the current plan entry hash and render key and its MP4 still has the recorded sha-256; a
   changed qc_key (or a QC that never finished) re-runs QC, never the encode.
5. Render + QC run in spawned worker processes (pool size from calibrate(); `jobs=` overrides)
   while the remaining reels compile. A reel's render and, separately, its QC have a deadline
   (reel_timeout_s, else one from its length): past it the worker is killed with its ffmpeg
   children; a crashed worker is replaced. Either way the slot is freed and the reel gets a
   structured error. A transient failure (a render / ffmpeg error record marked transient, a
   crash, a timeout) is retried once; a QC fail is deterministic and never re-rendered from
   the same timeline.
6. `only=` restricts work, never the manifest: a reel outside it is reused when still valid,
   redone when a shared input (style, LUT, fonts, encoder …) changed its render key, and
   otherwise listed as pending.
7. Export a fresh generation, generations/<g>/: out/NN_<slug>.mp4 for accepted reels (QC pass
   or warn), out/_review/ for rendered but failing ones, out/contact_sheet.jpg, reels/<id>/
   snapshots (timeline, deps, qc, sheet) and manifest.json + .csv. Then the `out` symlink is
   swapped to it in one rename and manifest.json / .csv are written at the top. Ids gone from
   the plan are listed as retired; deliverables come only from accepted entries.

Layout: data/batches/<name>/{plan.json, batch.log, manifest.json, manifest.csv,
out -> generations/<g>/out, generations/<g>/…, reels/<id>/{timeline.json, deps.json,
compile.json, look.cube, <id>.mp4, qc.json, sheet.jpg, render.json | cache.json}}. One run of a
batch at a time (batch.lock); generations older than the newest REELKIT_KEEP_GENERATIONS (10)
are removed.
"""
import contextlib
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.connection import wait as wait_any
from pathlib import Path

from . import __version__, contracts, ff, fsutil, paths
from .paths import DATA, FFMPEG, ROOT

RENDER_CODE = ["render.py", "captions.py", "grade.py", "ff.py"]
QC_CODE = ["qc.py", "asr.py"]
COMPILE_CODE = ["compile.py", "plan.py", "reframe.py", "source.py", "grade.py", "sfx.py"]
RECORD_VERSION = 2          # reels/<id>/cache.json layout (older records are not trusted)
COMPILE_VERSION = 1         # reels/<id>/compile.json key layout
ATTEMPTS = 2                # a transient failure is retried once
KILL_GRACE = 60.0           # s past a reel's own deadline before its worker is killed
STALL_S = 120               # s without an encoded frame before a render gives up
KEEP_GENERATIONS = 10


class PlanError(ValueError):
    pass


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


# ------------------------------------------------------------------ plan

def _style_path(ref, plan_dir):
    ref = str(ref)
    p = Path(ref)
    if not p.is_absolute():
        p = Path(plan_dir) / p
    if p.suffix == ".json" and p.exists():
        return p
    try:
        from . import store
        a = store.get(ref)
        s = a.dir / "style.json"
        return s if s.exists() else None
    except Exception:
        return None


def validate(plan, plan_dir="."):
    """Every problem in a plan, as a list of strings (empty = valid)."""
    errs = []
    if not isinstance(plan, dict):
        return ["plan is not a JSON object"]
    name = plan.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        errs.append(f"name {name!r}: letters, digits, '-', '_', '.' only")
    reels = plan.get("reels")
    if not isinstance(reels, list) or not reels:
        errs.append("reels: a non-empty list is required")
        reels = []
    all_prebuilt = bool(reels) and all(isinstance(r, dict) and r.get("timeline") for r in reels)
    if not plan.get("style"):
        errs.append("style: a sample asset id or a style.json path is required")
    elif _style_path(plan["style"], plan_dir) is None:
        errs.append(f"style {plan['style']!r}: no style.json found (analyse the sample first)")
    out = plan.get("output") or {}
    if not isinstance(out, dict):
        errs.append("output: an object is required")
        out = {}
    for k in ("width", "height"):
        v = out.get(k, 720 if k == "width" else 1280)
        if not isinstance(v, int) or v <= 0 or v % 2:
            errs.append(f"output.{k} {v!r}: a positive even integer is required")
    if not isinstance(out.get("fps", 30), (int, float)) or out.get("fps", 30) <= 0:
        errs.append(f"output.fps {out.get('fps')!r} must be positive")
    sources = plan.get("sources") or {}
    if not isinstance(sources, dict):
        errs.append("sources: an object {key: asset id} is required")
        sources = {}
    if not sources and not all_prebuilt:
        errs.append("sources: at least one source asset is required")
    for key, ref in sources.items():
        try:
            from . import store
            store.get(ref)
        except Exception:
            errs.append(f"sources.{key}: asset {ref!r} is not ingested")
    seen = set()
    for i, r in enumerate(reels):
        where = f"reels[{i}]"
        if not isinstance(r, dict):
            errs.append(f"{where}: must be an object")
            continue
        rid = r.get("id")
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
            errs.append(f"{where}.id {rid!r}: letters, digits, '-', '_' only")
        elif rid in seen:
            errs.append(f"{where}.id {rid!r} is used twice")
        seen.add(rid)
        where = f"reel {rid or i}"
        if r.get("timeline"):
            tp = Path(r["timeline"])
            if not tp.is_absolute():
                tp = Path(plan_dir) / tp
            if not tp.exists():
                errs.append(f"{where}: timeline {r['timeline']} not found")
            continue
        segs = r.get("segments")
        if not isinstance(segs, list) or not segs:
            errs.append(f"{where}: segments must be a non-empty list")
            segs = []
        for j, sg in enumerate(segs + ([r["hook"]] if r.get("hook") else [])):
            label = f"{where} segment {j}" if j < len(segs) else f"{where} hook"
            if not isinstance(sg, dict):
                errs.append(f"{label}: must be an object")
                continue
            if sg.get("src") not in sources:
                errs.append(f"{label}: src {sg.get('src')!r} is not in sources")
            if "sentences" in sg:
                ss = sg["sentences"]
                if not (isinstance(ss, list) and 1 <= len(ss) <= 2 and all(isinstance(x, int) and x >= 0 for x in ss)
                        and ss[0] <= ss[-1]):
                    errs.append(f"{label}: sentences must be [first, last] ids, got {ss!r}")
            elif "s" in sg and "e" in sg:
                if not (isinstance(sg["s"], (int, float)) and isinstance(sg["e"], (int, float)) and 0 <= sg["s"] < sg["e"]):
                    errs.append(f"{label}: needs 0 ≤ s < e, got {sg['s']!r}..{sg['e']!r}")
            else:
                errs.append(f"{label}: give sentences [a, b] or s/e times")
    return errs


def _contract_errors(plan, transcripts=None, log=None):
    """contracts.validate_plan on the reels that are compiled (a ready timeline is checked as a
    timeline instead); its warnings go to `log`."""
    reels = [r for r in plan.get("reels") or [] if isinstance(r, dict) and not r.get("timeline")]
    if not reels:
        return []
    try:
        v = contracts.validate_plan(dict(plan, reels=reels), transcripts)
    except Exception as e:
        return [f"plan validation failed: {ff.describe(e)}"]
    if log:
        for w in list(v.get("warnings") or [])[:20]:
            log(f"plan: {w}")
    return [str(x) for x in v.get("errors") or []]


def _plan_error(plan_path, errs):
    return PlanError(f"{plan_path} has {len(errs)} problem(s):\n  - " + "\n  - ".join(errs))


def load_plan(plan_path):
    plan_path = Path(plan_path)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PlanError(f"{plan_path}: {e}") from None
    errs = validate(plan, plan_path.parent)
    if not errs:
        errs = _contract_errors(plan)
    if errs:
        raise _plan_error(plan_path, errs)
    return plan


def slug(text, n=50):
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s[:n].rstrip("-") or "reel"


def _check_timeline(tl):
    """contracts.validate_timeline, never raising."""
    try:
        v = contracts.validate_timeline(tl)
        return {k: [str(x) for x in v.get(k) or []] for k in ("errors", "warnings")}
    except Exception as e:
        return {"errors": [f"timeline validation failed: {ff.describe(e)}"], "warnings": []}


# ------------------------------------------------------------------ encoder

def resolve_encoder(requested="auto", require_gpu=False, allow_cpu_fallback=True):
    """{"requested", "actual": "cpu" | "nvenc", "reason", "gpu"}, resolved once per batch. A plain
    CPU request never touches the GPU probe; anything else goes through readiness, which raises
    readiness.NotReady when a required GPU encoder is unusable."""
    requested = requested or "auto"
    if requested not in ("auto", "cpu", "nvenc"):
        raise ValueError(f"encoder {requested!r}: auto, cpu or nvenc")
    if requested == "cpu" and not require_gpu:
        return {"requested": "cpu", "actual": "cpu", "reason": "cpu requested", "gpu": None}
    try:
        from . import readiness
        r = readiness.resolve_encoder(requested=requested, require_gpu=require_gpu,
                                      allow_cpu_fallback=allow_cpu_fallback)
    except Exception as e:
        if requested == "auto" and not require_gpu and allow_cpu_fallback and type(e).__name__ != "NotReady":
            ff.log(f"encoder probe failed ({ff.describe(e)}): using the CPU encoder")
            return {"requested": "auto", "actual": "cpu", "reason": f"encoder probe failed: {e}", "gpu": None}
        raise
    if r.get("actual") not in ("cpu", "nvenc"):
        raise RuntimeError(f"encoder resolution returned {r.get('actual')!r}")
    return {"requested": r.get("requested") or requested, "actual": r["actual"], "reason": r.get("reason"),
            "gpu": r.get("gpu")}


# ------------------------------------------------------------------ cache keys

_FF_VERSION = None
_LIBS = None


def ffmpeg_version():
    global _FF_VERSION
    if _FF_VERSION is None:
        try:
            out = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True, timeout=30,
                                 stdin=subprocess.DEVNULL).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        if not out:
            return "unknown"
        _FF_VERSION = out.splitlines()[0]
    return _FF_VERSION


def _libs():
    global _LIBS
    if _LIBS is None:
        out = {}
        for name in ("cv2", "numpy", "PIL"):
            try:
                out[name] = getattr(__import__(name), "__version__", None)
            except Exception:
                out[name] = None
        _LIBS = out
    return _LIBS


def _code_hash(files):
    h = hashlib.sha256()
    for f in files:
        p = ROOT / "reelkit" / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _roots():
    """(placeholder, absolute prefix), longest first: paths under them key the same after a move."""
    pairs = set()
    for name, p in (("$DATA", DATA), ("$DATA", paths.DATA), ("$ROOT", ROOT)):
        for q in (Path(p), Path(p).resolve()):
            s = str(q).rstrip(os.sep)
            if len(s) > 1:
                pairs.add((name, s))
    return sorted(pairs, key=lambda x: -len(x[1]))


def _portable(obj, roots):
    if isinstance(obj, str):
        for name, root in roots:
            if obj.startswith(root + os.sep):
                return name + obj[len(root):]
        return obj
    if isinstance(obj, dict):
        return {_portable(k, roots): _portable(v, roots) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_portable(v, roots) for v in obj]
    return obj


def renderer_identity():
    """What decides a reel's pixels and audio besides its timeline and dependencies."""
    return {"reelkit": __version__, "code": _code_hash(RENDER_CODE), "ffmpeg": ffmpeg_version(), "libs": _libs()}


def render_key(tl, deps, encoder, draft, renderer=None):
    """Identity of a reel's MP4: timeline + deps.json + actual encoder + draft + renderer."""
    return fsutil.sha256_json(_portable({"timeline": tl, "deps": deps, "encoder": encoder, "draft": bool(draft),
                                         "renderer": renderer or renderer_identity()}, _roots()))


def qc_key(style_sha, asr, lang):
    """Identity of a reel's QC beyond its MP4: style, ASR on/off and fingerprint (the workers
    re-transcribe with the default provider), language, QC code."""
    from .compile import asr_fingerprint
    return fsutil.sha256_json({"style": style_sha, "asr": bool(asr), "lang": lang,
                               "asr_fingerprint": asr_fingerprint(lang) if asr else None,
                               "code": _code_hash(QC_CODE), "ffmpeg": ffmpeg_version()})


def _compile_models():
    from . import reframe
    try:
        return {"yunet": os.path.getsize(reframe.YUNET)}
    except OSError:
        return {"yunet": None}


# ------------------------------------------------------------------ worker

def _worker_init(cv_threads, asr_threads):
    import cv2
    if cv_threads:
        cv2.setNumThreads(int(cv_threads))
    if asr_threads:
        from . import asr
        asr._threads = lambda: int(asr_threads)


def _reel_job(job, progress=None):
    """Render (unless qc_only) and QC one reel, in a worker process. Never raises: a failure comes
    back as {"stage", "error", "error_record"} (the ff.FFError / render.RenderError record when
    there is one; its "transient" decides a retry)."""
    from . import qc, render
    out = Path(job["out"])
    rdir = out.parent
    res = {"id": job["id"], "rendered": False, "render_sha256": job.get("render_sha256"),
           "render_s": job.get("render_s"), "qc_verdict": None, "issues": [], "qc_sha256": None, "qc_s": None,
           "stage": None, "error": None, "error_record": None}
    sheet_tmp = rdir / f".sheet.{os.getpid()}.tmp.jpg"
    stage = "render"
    try:
        if not job.get("qc_only"):
            t0 = time.monotonic()
            render.render(job["timeline"], out, draft=job["draft"], encoder=job["encoder"], timeout_s=job.get("limit"),
                          stall_s=job.get("stall_s", STALL_S))
            res.update(rendered=True, render_s=round(time.monotonic() - t0, 2), render_sha256=fsutil.sha256_file(out))
            if progress:
                progress({"render_sha256": res["render_sha256"], "render_s": res["render_s"]})
        elif not out.is_file():
            raise FileNotFoundError(f"{out} is missing")
        stage = "qc"
        t0 = time.monotonic()
        rep = qc.check(out, job["timeline"], job.get("style"), asr=job["asr"], lang=job.get("lang"),
                       sheet_path=str(sheet_tmp))
        if sheet_tmp.exists():                 # a new file each time: generations hard-link the old one
            os.replace(sheet_tmp, rdir / "sheet.jpg")
        fsutil.atomic_write_json(rdir / "qc.json", rep)
        res.update(qc_verdict=rep["verdict"], issues=rep.get("issues") or [],
                   qc_sha256=fsutil.sha256_file(rdir / "qc.json"), qc_s=round(time.monotonic() - t0, 2))
    except (Exception, SystemExit) as ex:
        rec = getattr(ex, "record", None)
        rec = dict(rec) if isinstance(rec, dict) else ff.error_record(stage, cause=ex)
        res.update(stage=stage, error=f"{stage}: {ff.describe(ex)}", error_record=rec)
    finally:
        with contextlib.suppress(OSError):
            sheet_tmp.unlink()
    return res


def _worker_main(conn, cv_threads, asr_threads):
    """A pool worker (spawned process): runs reels from `conn` until it reads None."""
    with contextlib.suppress(OSError):
        os.dup2(2, 1)                          # stdout is the CLI's JSON channel: nothing of ours goes there
    _worker_init(cv_threads, asr_threads)

    def progress(info):
        with contextlib.suppress(OSError, ValueError):
            conn.send(("rendered", info))
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        res = _reel_job(job, progress)
        try:
            conn.send(("done", res))
        except (OSError, ValueError):
            return


def _descendants(pid):
    """Live descendants of pid, from /proc (Linux; [] elsewhere)."""
    children = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return []
    for d in names:
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", "rb") as f:
                st = f.read().decode("utf-8", "replace")
            ppid = int(st[st.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    out, todo, seen = [], [pid], {pid}
    while todo:
        for c in children.get(todo.pop(), ()):
            if c not in seen:
                seen.add(c)
                out.append(c)
                todo.append(c)
    return out


def _kill_tree(pid):
    """SIGKILL a worker and everything under it. Its ffmpeg children share the batch's process
    group (a job cancel must reach them), so they are found through /proc instead."""
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGSTOP)           # frozen: it cannot start another child meanwhile
    for p in _descendants(pid) + [pid]:
        with contextlib.suppress(OSError):
            os.kill(p, signal.SIGKILL)


def _clean_after(job, pid):
    """Temp files a killed worker left beside its reel."""
    out = Path(job["out"])
    for pat in (f".{out.stem}.{pid}-*", f".sheet.{pid}.tmp*"):
        for p in out.parent.glob(pat):
            with contextlib.suppress(OSError):
                p.unlink()


def _error_result(job, stage, cause, transient, returncode=None):
    return {"id": job["id"], "rendered": False, "render_sha256": job.get("render_sha256"), "qc_verdict": None,
            "issues": [], "stage": stage, "error": f"{stage}: {cause}",
            "error_record": ff.error_record(stage, None, returncode, "", cause, transient)}


class _Pool:
    """Worker processes running one reel each at a time — spawned, not forked: the caller may
    have threads (a job worker's heartbeat). Workers start as work arrives, up to n. A reel past
    its deadline is killed with its ffmpeg children and a dead worker is replaced; either way the
    slot is freed and the reel comes back as a transient, structured error."""

    def __init__(self, n, initargs, log):
        self.n, self.initargs, self.log = max(1, int(n)), tuple(initargs), log
        self.mp = multiprocessing.get_context("spawn")
        self.slots = []
        self.crashes = 0                       # worker deaths since the last finished reel
        self.max_crashes = max(3, 2 * self.n)

    def busy(self):
        return any(s["job"] is not None for s in self.slots)

    def _start(self):
        parent, child = self.mp.Pipe()
        proc = self.mp.Process(target=_worker_main, args=(child,) + self.initargs, name="reel-worker", daemon=True)
        try:
            proc.start()
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        return {"proc": proc, "conn": parent, "job": None, "deadline": None}

    def _drop(self, s):
        if s in self.slots:
            self.slots.remove(s)
        if s["proc"].is_alive():
            _kill_tree(s["proc"].pid)
        s["proc"].join(5)
        with contextlib.suppress(OSError):
            s["conn"].close()

    def _dispatch(self, queue, events):
        while queue:
            if self.crashes >= self.max_crashes:
                self.log(f"{self.crashes} worker processes died in a row: failing the {len(queue)} queued reel(s)")
                while queue:
                    job = queue.popleft()
                    events.append((job, "done", _error_result(job, "worker", "worker processes keep dying", False)))
                return
            slot = next((s for s in self.slots if s["job"] is None), None)
            if slot is None:
                if len(self.slots) >= self.n:
                    return
                try:
                    slot = self._start()
                except Exception as e:
                    self.crashes += 1
                    self.log(f"could not start a worker process: {ff.describe(e)}")
                    if self.slots:
                        return                 # the running ones free up later
                    continue
                self.slots.append(slot)
            job = queue.popleft()
            try:
                slot["conn"].send(job)
            except Exception as e:
                queue.appendleft(job)
                self.crashes += 1
                self.log(f"a worker process was gone before taking {job['id']} ({e})")
                self._drop(slot)
                continue
            slot["job"] = job
            slot["deadline"] = time.monotonic() + job["limit"] + KILL_GRACE

    def _collect(self, s, events):
        job, dead = s["job"], False
        try:
            while s["job"] is not None and s["conn"].poll():
                kind, payload = s["conn"].recv()
                if kind == "rendered":         # QC gets a deadline of its own
                    s["deadline"] = time.monotonic() + job["limit"] + KILL_GRACE
                else:
                    s["job"] = None
                    self.crashes = 0
                events.append((job, kind, payload))
        except (EOFError, OSError):
            dead = True
        except Exception as e:                 # an unreadable message: the worker cannot be trusted
            self.log(f"{job['id']}: unreadable message from its worker ({ff.describe(e)})")
            dead = True
        if not dead and s["proc"].is_alive():
            return
        if s["job"] is None:
            self._drop(s)
            return
        s["proc"].join(2)
        code, pid = s["proc"].exitcode, s["proc"].pid
        self._drop(s)
        _clean_after(job, pid)
        self.crashes += 1
        events.append((job, "done", _error_result(job, "worker", f"the worker process died (exit code {code})", True,
                                                  code)))

    def _expire(self, s, events):
        job, pid = s["job"], s["proc"].pid
        self._drop(s)
        _clean_after(job, pid)
        events.append((job, "done", _error_result(
            job, "timeout", f"no result within {job['limit'] + KILL_GRACE:g}s: the worker and its ffmpeg children "
                            f"were killed", True)))

    def step(self, queue, timeout):
        """Hand queued reels to idle workers, wait up to `timeout` s for news (less when a deadline
        is nearer) and return [(job, "rendered" | "done", payload)]."""
        events = []
        self._dispatch(queue, events)
        busy = [s for s in self.slots if s["job"] is not None]
        if not busy:
            return events
        wait_s = max(0.0, min([timeout] + [s["deadline"] - time.monotonic() for s in busy]))
        ready = set(wait_any([s["conn"] for s in busy] + [s["proc"].sentinel for s in busy], wait_s))
        for s in busy:
            if s["conn"] in ready or s["proc"].sentinel in ready:
                self._collect(s, events)
            if s in self.slots and s["job"] is not None and time.monotonic() > s["deadline"]:
                self._expire(s, events)
        return events

    def close(self):
        """Stop every worker: idle ones are asked to exit, busy ones (an interrupted batch) killed."""
        for s in self.slots:
            if s["job"] is None:
                with contextlib.suppress(Exception):
                    s["conn"].send(None)
        deadline = time.monotonic() + 5.0
        for s in list(self.slots):
            job, pid = s["job"], s["proc"].pid
            if job is None:
                s["proc"].join(max(0.0, deadline - time.monotonic()))
            self._drop(s)
            if job is not None:
                _clean_after(job, pid)


# ------------------------------------------------------------------ calibration

def _cpus():
    """CPUs this process may use: its affinity mask, capped by a cgroup v2 CPU quota."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 4
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return max(1, n)


def _machine(encoder="cpu", gpu=None):
    return f"{_cpus()}cpu|{encoder}{'@' + str(gpu) if gpu else ''}|{ffmpeg_version()}|reelkit {__version__}"


def _calib_path():
    return DATA / "batches" / "calibration.json"


def _calibrations():
    p = _calib_path()
    d = fsutil.read_json(p) if p.exists() else None
    return d if isinstance(d, dict) else {}


def _calib_timeline(dir_, seconds):
    """A representative 720p reel: 1080p source, cuts, a keyframed push, punch-ins,
    captions with pop + highlight, voice + ducked music."""
    dir_.mkdir(parents=True, exist_ok=True)
    src, tone, mus = dir_ / "src.mp4", dir_ / "voice.wav", dir_ / "music.wav"
    made = ((src, ["-f", "lavfi", "-i", f"testsrc2=s=1080x1920:r=30:d={seconds + 4}", "-c:v", "libx264",
                   "-preset", "faster", "-crf", "12", "-g", "30", "-pix_fmt", "yuv420p"]),
            (tone, ["-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=48000:duration={seconds + 4}",
                    "-af", "volume=0.4"]),
            (mus, ["-f", "lavfi", "-i", f"sine=frequency=880:sample_rate=48000:duration={seconds + 4}",
                   "-af", "volume=0.2"]))
    for path, args in made:
        if not path.exists():
            tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
            try:
                ff.ffmpeg(*args, tmp)
                os.replace(tmp, path)
            finally:
                with contextlib.suppress(OSError):
                    tmp.unlink()
    n = max(2, int(seconds // 2.5))
    seg = seconds / n
    video, audio, caps = [], [], []
    for i in range(n):
        c = {"media": "v", "start": round(i * seg, 3), "in": round(i * 0.7, 3), "dur": round(seg, 3), "fit": "cover"}
        if i % 3 == 0:
            c["scale"] = [[0, 1.0], [seg, 1.12, "inout"]]
        elif i % 3 == 1:
            c.update(scale=1.3, x=40, y=-60)
        video.append(c)
        audio.append({"media": "a", "start": c["start"], "in": c["in"], "dur": c["dur"], "role": "voice"})
    t = 0.0
    while t < seconds - 0.4:
        ws = [{"w": f"word{k}", "s": round(t + k * 0.3, 3), "e": round(t + k * 0.3 + 0.3, 3)} for k in range(3)]
        caps.append({"s": round(t, 3), "e": round(min(seconds, t + 0.9), 3), "text": " ".join(w["w"] for w in ws),
                     "words": ws})
        t += 0.9
    tl = {"width": 720, "height": 1280, "fps": 30, "duration": seconds, "background": "#000000",
          "media": {"v": str(src), "a": str(tone), "m": str(mus)}, "video": video, "captions": caps,
          "styles": {"caption": {"font": "Arimo", "weight": 800, "size": 46, "case": "upper",
                                 "stroke": {"color": "#000000", "width": 4}, "highlight": {"fill": "#ffe600"},
                                 "y": 120, "anim": {"in": "pop", "in_dur": 0.15}}},
          "audio": {"clips": audio + [{"media": "m", "start": 0, "in": 0, "dur": seconds, "role": "music",
                                       "gain_db": -10}], "duck_db": 12, "loudness": -14},
          "encode": {"preset": "medium", "crf": 17}}
    p = dir_ / "timeline.json"
    fsutil.atomic_write_json(p, tl)
    return p


def _calib_render(args):
    tl, out, cv_threads, encoder = args
    _worker_init(cv_threads, None)
    from . import render
    t0 = time.monotonic()
    render.render(tl, out, encoder=encoder)
    return time.monotonic() - t0


def calibrate(seconds=8.0, concurrency=(1, 2, 3, 4), save=True, encoder="cpu", gpu=None):
    """Measure reels/hour for 30 s 720p reels at each pool size × OpenCV threads with this
    encoder; store and return the best configuration for this machine."""
    cores = _cpus()
    d = DATA / "batches" / "_calibration"
    tl = _calib_timeline(d, seconds)
    results = []
    ctx = multiprocessing.get_context("spawn")
    for c in concurrency:
        for cvt in sorted({0, max(1, cores // c)}):
            t0 = time.monotonic()
            r0 = os.times()
            with ProcessPoolExecutor(c, mp_context=ctx) as pool:
                list(pool.map(_calib_render, [(str(tl), str(d / f"out{k}.mp4"), cvt, encoder) for k in range(c)]))
            wall = time.monotonic() - t0
            r1 = os.times()
            cpu = (r1.children_user - r0.children_user) + (r1.children_system - r0.children_system)
            rph = c * 3600.0 / wall * (seconds / 30.0)
            results.append({"jobs": c, "cv_threads": cvt, "wall_s": round(wall, 2), "cpu_s": round(cpu, 1),
                            "reels_per_hour_30s": round(rph, 1)})
            ff.log(f"calibrate: {c} concurrent × cv threads {cvt or 'auto'}: {wall:.1f}s wall, "
                   f"{cpu:.0f} CPU-s → {rph:.0f} reels/h (30 s, 720p, {encoder})")
    best = max(results, key=lambda r: r["reels_per_hour_30s"])
    out = {"machine": _machine(encoder, gpu), "encoder": encoder, "seconds": seconds, "results": results,
           "best": best, "measured": _now()}
    if save:
        p = _calib_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with fsutil.file_lock(p):
            allc = _calibrations()
            allc[_machine(encoder, gpu)] = out
            fsutil.atomic_write_json(p, allc)
    return out


def _best(encoder, gpu):
    b = (_calibrations().get(_machine(encoder, gpu)) or {}).get("best")
    if (isinstance(b, dict) and isinstance(b.get("jobs"), int) and b["jobs"] >= 1
            and isinstance(b.get("cv_threads"), int) and b["cv_threads"] >= 0):
        return b
    return None


def _concurrency(jobs, auto_calibrate=True, encoder="cpu", gpu=None):
    """(jobs, cv_threads, how): explicit, calibrated (reused or measured now), or a safe default."""
    cores = _cpus()
    if jobs:
        n = max(1, int(jobs))
        return n, max(1, cores // n), "explicit"
    best = _best(encoder, gpu)
    if best:
        return best["jobs"], best["cv_threads"], "calibration reused"
    if auto_calibrate:
        try:
            with fsutil.file_lock(DATA / "batches" / "calibrate"):       # one measurement at a time
                best = _best(encoder, gpu)
                if best:
                    return best["jobs"], best["cv_threads"], "calibration reused"
                ff.log(f"no calibration for this machine ({encoder}) yet: measuring (once, ~2 min)")
                best = calibrate(encoder=encoder, gpu=gpu)["best"]
            return best["jobs"], best["cv_threads"], "calibrated now"
        except Exception as e:
            ff.log(f"calibration failed ({ff.describe(e)}): using a safe default")
    return max(1, cores // 2), 2, "default"


def concurrency(jobs=None, auto_calibrate=True, encoder="cpu", gpu=None):
    """(jobs, cv_threads) for this machine: explicit, calibrated, or a safe default."""
    return _concurrency(jobs, auto_calibrate, encoder, gpu)[:2]


# ------------------------------------------------------------------ records and files

RECORDS = ("render.json", "cache.json")      # rendered, QC to do | rendered and QC'd


def _record(rdir):
    """A reel's latest record: render.json (its MP4 is newer than its last QC) before cache.json;
    None when missing, corrupt or from an older layout."""
    for name in RECORDS:
        p = rdir / name
        rec = fsutil.read_json(p) if p.exists() else None
        if isinstance(rec, dict) and rec.get("version") == RECORD_VERSION:
            return rec
    return None


def _render_current(rec, key, peh, out):
    """(sha256, None, False) when the recorded render is current and intact, else (None, reason,
    damaged) — damaged: current by its keys, but its file is gone or changed."""
    if not rec or not rec.get("render_sha256"):
        return None, "not rendered yet", False
    if rec.get("plan_entry_hash") != peh:
        return None, "its plan entry changed", False
    if rec.get("render_key") != key:
        return None, "its timeline or a dependency changed", False
    try:
        sha = fsutil.sha256_file(out)
    except OSError:
        return None, "its rendered file is missing", True
    if sha != rec["render_sha256"]:
        return None, "its rendered file differs from the record", True
    return sha, None, False


def _qc_current(rec, qk, rdir):
    """(QC report, None) when the recorded QC is current and qc.json intact, else (None, reason)."""
    if not rec.get("qc_key"):
        return None, "not QC'd yet"
    if rec["qc_key"] != qk:
        return None, "the QC settings changed"
    p = rdir / "qc.json"
    try:
        if fsutil.sha256_file(p) != rec.get("qc_sha256"):
            return None, "qc.json differs from the record"
    except OSError:
        return None, "qc.json is missing"
    rep = fsutil.read_json(p)
    if not isinstance(rep, dict) or rep.get("verdict") not in ("pass", "warn", "fail"):
        return None, "qc.json is unreadable"
    return rep, None


def _limit(duration, reel_timeout_s=None):
    """Seconds a reel's render (and, separately, its QC) may take."""
    if reel_timeout_s:
        return float(reel_timeout_s)
    try:
        d = float(duration)
    except (TypeError, ValueError):
        d = 60.0
    return 300.0 + 40.0 * max(0.0, d if math.isfinite(d) else 60.0)


def _absolute(tl, base):
    """Make a timeline's media and LUT paths absolute (it is about to move to another directory)."""
    from .render import _resolve
    tl["media"] = {k: _resolve(v, base) for k, v in (tl.get("media") or {}).items()}
    if isinstance(tl.get("grade"), dict) and tl["grade"].get("lut"):
        tl["grade"]["lut"] = _resolve(tl["grade"]["lut"], base)


def _clean_temp(rdir):
    """Leftovers of interrupted writes in a reel directory (the batch lock is held)."""
    for pat in (".*.tmp*", "*.part", "*.part.*"):
        for p in rdir.glob(pat):
            with contextlib.suppress(OSError):
                if p.is_file():
                    p.unlink()


def _link(src, dest, sha=None):
    """dest as a hard link to src, or a copy (verified against `sha`) where linking is impossible."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return
    except FileExistsError:
        raise
    except OSError:
        pass
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(src, tmp)
        if sha and fsutil.sha256_file(tmp) != sha:
            raise ValueError(f"the copy of {src} differs from the rendered reel")
        os.replace(tmp, dest)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _new_generation(bdir, prior):
    root = bdir / "generations"
    root.mkdir(parents=True, exist_ok=True)
    try:
        last = int(prior.get("generation") or 0)
    except (TypeError, ValueError):
        last = 0
    g = max([last] + [int(p.name) for p in root.iterdir() if p.name.isdigit()]) + 1
    while True:
        try:
            (root / str(g)).mkdir()
            return g, root / str(g)
        except FileExistsError:
            g += 1


def _swap_out(bdir, gdir):
    """Point bdir/out at the generation's out/ in one rename (a real out/ directory left by an
    earlier version is moved aside first)."""
    link = bdir / "out"
    if link.is_dir() and not link.is_symlink():
        os.rename(link, bdir / f"out.legacy-{time.strftime('%Y%m%d-%H%M%S')}")
    tmp = bdir / f".out.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    os.symlink(os.path.relpath(gdir / "out", bdir), tmp)
    try:
        os.replace(tmp, link)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _keep_generations():
    try:
        return max(1, int(os.environ.get("REELKIT_KEEP_GENERATIONS") or KEEP_GENERATIONS))
    except ValueError:
        return KEEP_GENERATIONS


def _prune(bdir, current, keep, log):
    """Remove generations older than the newest `keep`, and unfinished ones (no manifest)."""
    root = bdir / "generations"
    for p in list(root.iterdir()):
        if not p.name.isdigit() or int(p.name) >= current:
            continue
        if int(p.name) <= current - keep or not (p / "manifest.json").exists():
            shutil.rmtree(p, ignore_errors=True)
            log(f"removed generation {p.name}")
    for p in bdir.glob(".out.*.tmp"):
        with contextlib.suppress(OSError):
            p.unlink()


def _csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["n", "id", "title", "duration", "status", "verdict", "issues", "file", "sha256", "cached"])
    for e in rows:
        w.writerow([e["n"], e["id"], e["title"], e.get("duration"), e.get("status"), e.get("verdict"),
                    " | ".join(f"[{i.get('severity')}] {i.get('check')}: {i.get('msg')}" for i in e.get("issues") or [])
                    or e.get("error") or "", e.get("file") or "", e.get("render_sha256") or "", e.get("cached")])
    return buf.getvalue()


@contextlib.contextmanager
def _batch_lock(bdir, log):
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(fsutil.file_lock(bdir / "batch", timeout=1))
        except TimeoutError:
            log("another run of this batch is in progress: waiting for it")
            stack.enter_context(fsutil.file_lock(bdir / "batch"))
        yield


# ------------------------------------------------------------------ run

class _Batch:
    def __init__(self, plan, plan_path, bdir, log, jobs, draft, only, encoder, asr, compiler, auto_calibrate,
                 require_gpu, allow_cpu_fallback, reel_timeout_s, lang, asr_provider):
        self.plan, self.plan_path, self.bdir, self.log = plan, plan_path, bdir, log
        self.jobs, self.draft, self.only, self.encoder, self.asr = jobs, bool(draft), only, encoder, bool(asr)
        self.compiler, self.auto_calibrate = compiler, auto_calibrate
        self.require_gpu, self.allow_cpu_fallback = require_gpu, allow_cpu_fallback
        self.reel_timeout_s, self.lang, self.asr_provider = reel_timeout_s, lang, asr_provider
        self.only_set = set(only) if only else None
        self.entries, self.queue, self.pool, self.workers = {}, deque(), None, None
        self.ctx = None
        self.done = self.total = 0
        self.started = time.monotonic()

    def run(self):
        plan, bdir, log = self.plan, self.bdir, self.log
        if self.plan_path != (bdir / "plan.json").resolve():
            fsutil.atomic_write_text(bdir / "plan.json", self.plan_path.read_text(encoding="utf-8"))
        self.load_style()
        self.enc = resolve_encoder(self.encoder, self.require_gpu, self.allow_cpu_fallback)
        log(f"batch {plan['name']}: {len(plan['reels'])} reels"
            f"{f' (work limited to {len(self.only)})' if self.only else ''}, {'draft' if self.draft else 'final'} "
            f"quality, language {self.lang}, encoder {self.enc['actual']} (requested {self.enc['requested']}: "
            f"{self.enc.get('reason')})")
        self.context()
        self.renderer = renderer_identity()
        self.qk = qc_key(self.style_sha, self.asr, self.lang)
        prior = fsutil.read_json(bdir / "manifest.json") if (bdir / "manifest.json").exists() else None
        prior = prior if isinstance(prior, dict) else {}
        try:
            for i, r in enumerate(plan["reels"]):
                try:
                    e, job = self.prepare(i, r)
                except Exception as ex:          # one reel's bookkeeping never stops the others
                    self.log(f"{r['id']}: {ff.describe(ex)}")
                    e, job = self.fail(self.blank(i, r), "prepare", ex), None
                self.entries[r["id"]] = e
                if job:
                    self.submit(job)
                self.pump(0)
            while self.queue or (self.pool and self.pool.busy()):
                self.pump(5.0)
        finally:
            if self.pool:
                self.pool.close()
        return self.export(prior)

    # ---- inputs

    def load_style(self):
        style_p = _style_path(self.plan["style"], self.plan_path.parent)
        style = fsutil.read_json(style_p) if style_p else None
        if not isinstance(style, dict):
            raise PlanError(f"style {self.plan['style']!r}: {style_p or 'no style.json'} is not a readable style")
        self.style_p, self.style, self.style_sha = style_p, style, fsutil.sha256_json(style)
        self.lang = self.lang or (self.plan.get("output") or {}).get("lang") or _dig(style, "audio", "speech", "lang") \
            or "en"

    def context(self):
        """The shared compile context when any reel is compiled; the plan is then checked again
        against its transcripts."""
        self.cplan = dict(self.plan, output=dict(self.plan.get("output") or {}, lang=self.lang),
                          reels=[r for r in self.plan["reels"] if not r.get("timeline")])
        if not self.cplan["reels"]:
            return
        from .compile import Context
        self.ctx = ctx = Context(self.cplan, self.style, provider=self.asr_provider)
        errs = _contract_errors(self.cplan, None if ctx.errors else ctx.transcripts, self.log)
        if errs:
            raise _plan_error(self.plan_path, errs)
        self.asset_sha = {k: a.meta.get("sha256") or a.id for k, a in ctx.assets.items()}
        self.tr_sha = {k: fsutil.sha256_json(t) for k, t in ctx.transcripts.items()}
        self.ckey_base = fsutil.sha256_json({
            "v": COMPILE_VERSION, "output": self.cplan["output"], "music": self.plan.get("music"),
            "sfx": self.plan.get("sfx"), "style": self.style_sha, "lang": self.lang, "provider": self.asr_provider,
            "code": _code_hash(COMPILE_CODE), "models": _compile_models()})

    def compile_key(self, r, peh):
        from .compile import _reel_sources
        srcs = _reel_sources(r)
        return fsutil.sha256_json({"base": self.ckey_base, "entry": peh,
                                   "sources": {str(k): self.asset_sha.get(k) for k in srcs},
                                   "transcripts": {str(k): self.tr_sha.get(k) for k in srcs}})

    # ---- one reel

    def blank(self, i, r):
        return {"id": r["id"], "n": i + 1, "title": r.get("title") or r["id"],
                "plan_entry_hash": contracts.plan_entry_hash(r), "timeline": None, "render_key": None,
                "render_sha256": None, "verdict": None, "qc_verdict": None, "issues": [], "file": None, "sheet": None,
                "qc": None, "status": None, "error": None, "error_record": None, "cached": False, "duration": None,
                "attempts": 0, "work": None}

    def prepare(self, i, r):
        """The reel's manifest entry and, when there is work to do, its worker job."""
        rid = r["id"]
        rdir = self.bdir / "reels" / rid
        rdir.mkdir(parents=True, exist_ok=True)
        _clean_temp(rdir)
        e = self.blank(i, r)
        peh = e["plan_entry_hash"]
        rec = _record(rdir)
        in_scope = self.only_set is None or rid in self.only_set
        if not in_scope and not (rec and rec.get("plan_entry_hash") == peh):
            return self.pending(e, "outside this run's --only, and never rendered from its current plan entry"), None
        try:
            tl_path, tl, deps, how = self.timeline(r, rdir, peh)
        except Exception as ex:
            self.log(f"{rid}: compile failed — {ff.describe(ex)}")
            return self.fail(e, "compile", ex), None
        e.update(timeline=str(tl_path), duration=tl.get("duration"), compiled=how)
        v = _check_timeline(tl)
        for w in v["warnings"][:5]:
            self.log(f"{rid}: timeline warning: {w}")
        if v["errors"]:
            msg = "timeline: invalid — " + "; ".join(v["errors"][:8])
            self.log(f"{rid}: {msg}")
            return self.fail(e, "timeline", msg), None
        key = e["render_key"] = render_key(tl, deps, self.enc["actual"], self.draft, self.renderer)
        style_p, qk = self.style_p, self.qk
        if r.get("editing"):
            from .compile import reel_style
            effective = reel_style(self.style, r)
            style_p = rdir / "style.effective.json"
            fsutil.atomic_write_json(style_p, effective)
            qk = qc_key(fsutil.sha256_json(effective), self.asr, self.lang)
        out = rdir / f"{rid}.mp4"
        job = {"id": rid, "timeline": str(tl_path), "out": str(out), "draft": self.draft, "encoder": self.enc["actual"],
               "style": str(style_p), "qc_key": qk, "asr": self.asr, "lang": self.lang,
               "limit": _limit(tl.get("duration"), self.reel_timeout_s), "stall_s": STALL_S, "attempt": 1,
               "qc_only": False, "plan_entry_hash": peh, "render_key": key, "render_sha256": None, "render_s": None}
        sha, why, damaged = _render_current(rec, key, peh, out)
        if sha:
            e["render_sha256"] = job["render_sha256"] = sha
            job.update(render_s=rec.get("render_s"), qc_only=True)
            rep, why = _qc_current(rec, qk, rdir)
            if rep is not None:
                self.apply_qc(e, rep, rdir)
                e.update(cached=True, render_s=rec.get("render_s"), qc_s=rec.get("qc_s"))
                self.log(f"{rid}: cached ({rep['verdict']})")
                return e, None
        elif not in_scope and damaged:
            return self.pending(e, f"outside this run's --only, and {why}"), None
        e["work"] = "qc" if job["qc_only"] else "render"
        if in_scope:
            self.log(f"{rid}: {'QC only' if job['qc_only'] else 'render'} — {why}")
        else:
            self.log(f"{rid}: outside --only, but {why}: redoing it (a shared input changed)")
        return e, job

    def timeline(self, r, rdir, peh):
        """(path, timeline, deps, how): how is ready (the reel's own timeline file), cached (the
        last compile still matches every input) or compiled."""
        from . import compile as C
        dest = rdir / "timeline.json"
        if r.get("timeline"):
            src = Path(r["timeline"])
            src = src if src.is_absolute() else self.plan_path.parent / src
            tl = json.loads(src.read_text(encoding="utf-8"))
            if not isinstance(tl, dict):
                raise ValueError(f"{src}: not a timeline object")
            _absolute(tl, str(src.resolve().parent))
            self.check_output(tl)
            tl = json.loads(json.dumps(tl, ensure_ascii=False))
            deps = C.dependencies(tl, str(rdir), style=self.style)
            fsutil.atomic_write_json(dest, tl)
            fsutil.atomic_write_json(rdir / "deps.json", deps)
            return dest, tl, deps, "ready"
        self.ctx.check(r)
        key = self.compile_key(r, peh)
        hit, why = self.cached_compile(rdir, key)
        if hit:
            return dest, hit[0], hit[1], "cached"
        self.log(f"{r['id']}: compiling ({why})")
        for attempt in (1, 2):
            try:
                tl, deps = self.compile(r, rdir)
                break
            except Exception as ex:
                if attempt == 2 or not (getattr(ex, "record", None) or {}).get("transient"):
                    raise
                self.log(f"{r['id']}: transient compile failure ({ff.describe(ex)}): once more")
        fsutil.atomic_write_json(rdir / "compile.json", {"key": key, "timeline_sha256": fsutil.sha256_file(dest),
                                                         "deps_sha256": fsutil.sha256_json(deps), "at": _now()})
        return dest, tl, deps, "compiled"

    def deps(self, tl, rdir):
        from . import compile as C
        return C.dependencies(tl, str(rdir), self.cplan, self.style, self.lang, self.asr_provider,
                              self.ctx.transcripts)

    def compile(self, r, rdir):
        from . import compile as C
        dest = rdir / "timeline.json"
        deps = None
        if self.compiler is None:
            C.compile_reel(self.cplan, r, self.style, str(rdir), **self.ctx.shared())
            tl = fsutil.read_json(dest)
            deps = fsutil.read_json(rdir / "deps.json")
            if not isinstance(tl, dict):
                raise RuntimeError(f"the compiler left no readable {dest}")
            self.check_output(tl)
        else:
            res = self.compiler(self.cplan, r, self.style, str(rdir))
            if isinstance(res, dict):
                tl = res
            elif res is not None and Path(res).exists():
                tl = json.loads(Path(res).read_text(encoding="utf-8"))
                if Path(res).resolve() != dest.resolve():
                    _absolute(tl, str(Path(res).resolve().parent))
            else:
                tl = json.loads(dest.read_text(encoding="utf-8"))
            self.check_output(tl)
            tl = json.loads(json.dumps(tl, ensure_ascii=False))
            fsutil.atomic_write_json(dest, tl)
        if not isinstance(deps, dict):
            deps = self.deps(tl, rdir)
            fsutil.atomic_write_json(rdir / "deps.json", deps)
        return tl, deps

    def cached_compile(self, rdir, key):
        """((timeline, deps), None) when the last compile of this reel is still valid, else (None, reason)."""
        p = rdir / "compile.json"
        rec = fsutil.read_json(p) if p.exists() else None
        if not isinstance(rec, dict):
            return None, "no earlier compile"
        if rec.get("key") != key:
            return None, "its plan entry, the style or a source input changed"
        tl_p, deps_p = rdir / "timeline.json", rdir / "deps.json"
        try:
            if fsutil.sha256_file(tl_p) != rec.get("timeline_sha256"):
                return None, "timeline.json differs from its record"
        except OSError:
            return None, "timeline.json is missing"
        tl = fsutil.read_json(tl_p)
        deps = fsutil.read_json(deps_p) if deps_p.exists() else None
        if not isinstance(tl, dict) or not isinstance(deps, dict) or fsutil.sha256_json(deps) != rec.get("deps_sha256"):
            return None, "deps.json is missing or differs from its record"
        try:
            now = self.deps(tl, rdir)
        except Exception as e:
            return None, f"its dependencies cannot be read ({ff.describe(e)})"
        if fsutil.sha256_json(now) != fsutil.sha256_json(deps):
            changed = sorted(k for k in set(now) | set(deps) if now.get(k) != deps.get(k))
            return None, f"{', '.join(changed) or 'a dependency'} changed"
        try:
            self.check_output(tl)
        except ValueError as e:
            return None, str(e)
        return (tl, deps), None

    def check_output(self, tl):
        out = self.plan.get("output") or {}
        for k in ("width", "height", "fps"):
            if k in out and tl.get(k) != out[k]:
                raise ValueError(f"timeline {k} {tl.get(k)} differs from output.{k} {out[k]}")

    # ---- outcomes

    def pending(self, e, why):
        e.update(status="pending", verdict="pending", qc_verdict=None, error=why, error_record=None)
        self.log(f"{e['id']}: pending — {why}")
        return e

    def fail(self, e, stage, err, record=None):
        if isinstance(err, BaseException):
            msg = f"{stage}: {ff.describe(err)}"
            record = (record or getattr(err, "record", None) or getattr(err.__cause__, "record", None)
                      or ff.error_record(stage, cause=err))
        else:
            msg = str(err) if str(err).startswith(stage) else f"{stage}: {err}"
            record = record or ff.error_record(stage, cause=str(err))
        e.update(status="failed", verdict="error", qc_verdict="error", error=msg, error_record=dict(record),
                 cached=False)
        return e

    def apply_qc(self, e, rep, rdir):
        v, issues = rep.get("verdict"), rep.get("issues") or []
        sheet = rdir / "sheet.jpg"
        e.update(qc_verdict=v, verdict=v, issues=issues, qc=str(rdir / "qc.json"),
                 sheet=str(sheet) if sheet.is_file() else None, _mp4=str(rdir / f"{e['id']}.mp4"))
        if v in ("pass", "warn"):
            e.update(status="accepted", error=None, error_record=None)
        else:
            fails = [f"{i.get('check')}: {i.get('msg')}" for i in issues if i.get("severity") == "fail"]
            e.update(status="failed", error="QC fail: " + ("; ".join(fails[:3]) or "see qc.json"), error_record=None)

    def write_record(self, job, res=None):
        rec = {"version": RECORD_VERSION, "id": job["id"], "plan_entry_hash": job["plan_entry_hash"],
               "render_key": job["render_key"], "render_sha256": job.get("render_sha256"),
               "render_s": job.get("render_s"), "encoder": job["encoder"], "draft": job["draft"], "qc_key": None,
               "qc_sha256": None, "qc_verdict": None, "qc_s": None, "finished": _now()}
        qcd = bool(res and res.get("qc_verdict"))
        if qcd:
            rec.update(qc_key=job.get("qc_key", self.qk), qc_sha256=res.get("qc_sha256"), qc_verdict=res["qc_verdict"],
                       qc_s=res.get("qc_s"))
        rdir = Path(job["out"]).parent
        name, stale = ("cache.json", "render.json") if qcd else ("render.json", "cache.json")
        try:
            fsutil.atomic_write_json(rdir / name, rec)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(rdir / stale)        # it describes an older MP4 (or the same one, without QC)
        except OSError as ex:
            self.log(f"{job['id']}: could not write its record ({ex}): it is redone next time")

    # ---- work

    def submit(self, job):
        if self.pool is None:
            n, cvt, how = _concurrency(self.jobs, self.auto_calibrate, self.enc["actual"], self.enc.get("gpu"))
            self.workers = {"jobs": n, "cv_threads": cvt, "source": how}
            self.log(f"render pool: up to {n} worker(s), cv threads {cvt or 'auto'} ({how})")
            self.pool = _Pool(n, (cvt, max(1, _cpus() // n)), self.log)
        self.queue.append(job)
        self.total += 1

    def pump(self, timeout):
        if self.pool is None:
            return
        for job, kind, payload in self.pool.step(self.queue, timeout):
            try:
                self.handle(job, kind, payload)
            except Exception as ex:            # bookkeeping must not lose the other reels
                self.log(f"{job['id']}: could not record its result ({ff.describe(ex)})")
                self.fail(self.entries[job["id"]], "batch", ex)

    def handle(self, job, kind, res):
        rid = job["id"]
        if kind == "rendered":                 # the MP4 is complete: a QC failure need not re-render it
            job.update(render_sha256=res.get("render_sha256"), render_s=res.get("render_s"))
            self.write_record(job)
            return
        e = self.entries[rid]
        e["attempts"] = job["attempt"]
        if res.get("rendered") and res.get("render_sha256") and res["render_sha256"] != job.get("render_sha256"):
            job.update(render_sha256=res["render_sha256"], render_s=res.get("render_s"))
            self.write_record(job)
        if res.get("error"):
            rec = res.get("error_record") or {}
            if rec.get("transient") and job["attempt"] < ATTEMPTS:
                again = dict(job, attempt=job["attempt"] + 1, qc_only=bool(job.get("render_sha256")))
                self.queue.append(again)
                self.log(f"{rid}: {res['error']} — transient, trying again{' (QC only)' if again['qc_only'] else ''}")
                return
            self.fail(e, res.get("stage") or "render", res["error"], rec)
            if job.get("render_sha256"):       # rendered, but without a QC verdict: kept for review
                e.update(render_sha256=job["render_sha256"], _mp4=job["out"])
        else:
            self.write_record(job, res)
            e["render_sha256"] = job["render_sha256"]
            self.apply_qc(e, {"verdict": res.get("qc_verdict"), "issues": res.get("issues") or []},
                          Path(job["out"]).parent)
            e.update(render_s=job.get("render_s"), qc_s=res.get("qc_s"))
        self.done += 1
        elapsed = time.monotonic() - self.started
        eta = elapsed / self.done * max(0, self.total - self.done)
        self.log(f"{self.done}/{self.total} {rid}: {e['verdict']}{' — ' + e['error'] if e.get('error') else ''}  "
                 f"(elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min)")

    # ---- export

    def publish(self, e, gdir, width):
        """Hard-link the reel's current artifacts into the generation. They stay as they are: every
        later write to the reel directory replaces a file instead of changing it."""
        rdir = self.bdir / "reels" / e["id"]
        snap = gdir / "reels" / e["id"]
        for field, name in (("timeline", "timeline.json"), ("qc", "qc.json"), ("sheet", "sheet.jpg")):
            if e.get(field) and os.path.isfile(e[field]):
                try:
                    _link(e[field], snap / name)
                    e[field] = str(snap / name)
                except (OSError, ValueError) as ex:
                    self.log(f"{e['id']}: could not keep {name} in the generation ({ex})")
        if e.get("timeline") and (rdir / "deps.json").is_file():
            with contextlib.suppress(OSError, ValueError):
                _link(rdir / "deps.json", snap / "deps.json")
        mp4 = e.get("_mp4")
        if not mp4 or e["status"] not in ("accepted", "failed"):
            return
        name = f"{e['n']:0{width}d}_{slug(e['title'])}.mp4"
        dest = gdir / "out" / ("" if e["status"] == "accepted" else "_review") / name
        try:
            _link(mp4, dest, e.get("render_sha256"))
            e["file"] = str(dest)
        except (OSError, ValueError) as ex:
            self.log(f"{e['id']}: could not publish {name} ({ex})")
            if e["status"] == "accepted":
                e.update(status="failed", verdict="error", error=f"publish: {ex}",
                         error_record=ff.error_record("publish", cause=ex))

    def retired(self, prior, ids):
        cur = set(ids)
        old = {x for x in prior.get("retired") or [] if isinstance(x, str)}
        old |= {e.get("id") for e in prior.get("reels") or [] if isinstance(e, dict) and isinstance(e.get("id"), str)}
        with contextlib.suppress(OSError):
            old |= {p.name for p in (self.bdir / "reels").iterdir() if any((p / n).is_file() for n in RECORDS)}
        return sorted(old - cur)

    def export(self, prior):
        bdir, plan = self.bdir, self.plan
        ids = [r["id"] for r in plan["reels"]]
        rows = [self.entries[i] for i in ids]
        g, gdir = _new_generation(bdir, prior)
        (gdir / "out" / "_review").mkdir(parents=True, exist_ok=True)
        width = max(2, len(str(len(ids))))
        for e in rows:
            self.publish(e, gdir, width)
            for k in [k for k in e if k.startswith("_")]:
                del e[k]
        retired = self.retired(prior, ids)
        deliverables = [{"id": e["id"], "file": e["file"], "sha256": e["render_sha256"], "verdict": e["qc_verdict"]}
                        for e in rows if e["status"] == "accepted" and e.get("file")]
        status = Counter(e["status"] for e in rows)
        counts = {"planned": len(rows), "rendered": sum(1 for e in rows if e.get("file")),
                  "accepted": len(deliverables), "failed": status["failed"], "pending": status["pending"],
                  "retired": len(retired), "delivered": len(deliverables),
                  "cached": sum(1 for e in rows if e.get("cached")), "worked": self.done}
        manifest = {"name": plan["name"], "style": plan["style"], "style_sha256": self.style_sha,
                    "plan_hash": fsutil.sha256_json(plan), "generation": g,
                    "quality": "draft" if self.draft else "final", "lang": self.lang,
                    "encoder": {k: self.enc.get(k) for k in ("requested", "actual", "reason", "gpu")},
                    "jobs": (self.workers or {}).get("jobs", 0), "workers": self.workers, "only": self.only,
                    "elapsed_s": round(time.monotonic() - self.started, 1), "finished": _now(), "counts": counts,
                    "summary": dict(Counter(e["verdict"] for e in rows)), "reels": rows, "retired": retired,
                    "deliverables": deliverables, "out_dir": str(gdir / "out")}
        try:
            contact_sheet([e for e in rows if e.get("file")], gdir / "out" / "contact_sheet.jpg")
        except Exception as ex:
            self.log(f"contact sheet failed: {ff.describe(ex)}")
        table = _csv(rows)
        fsutil.atomic_write_json(gdir / "manifest.json", manifest)
        fsutil.atomic_write_text(gdir / "manifest.csv", table)
        try:
            _swap_out(bdir, gdir)
        except OSError as ex:
            self.log(f"could not point {bdir / 'out'} at generation {g} ({ex}); the manifest names it directly")
        fsutil.atomic_write_json(bdir / "manifest.json", manifest)
        fsutil.atomic_write_text(bdir / "manifest.csv", table)
        _prune(bdir, g, _keep_generations(), self.log)
        self.log(f"generation {g}: {counts}  → {gdir / 'out'}")
        return manifest


def run(plan_path, jobs=None, draft=False, only=None, encoder="auto", asr=True, compiler=None, auto_calibrate=True,
        require_gpu=False, allow_cpu_fallback=True, reel_timeout_s=None, lang=None, asr_provider=None):
    """Produce every reel of a plan; returns the manifest (see the module docstring). encoder:
    auto | cpu | nvenc, resolved once; asr_provider: the provider transcripts are made with
    (None = asr's default). Raises PlanError for an invalid plan, readiness.NotReady when a
    required GPU encoder is unusable."""
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path)
    only = [str(x) for x in only] if only else None
    if only:
        missing = sorted(set(only) - {r["id"] for r in plan["reels"]})
        if missing:
            raise PlanError(f"--only: no reels {missing} in the plan")
    if reel_timeout_s is not None:
        reel_timeout_s = float(reel_timeout_s)
        if not math.isfinite(reel_timeout_s) or reel_timeout_s <= 0:
            raise ValueError(f"reel_timeout_s {reel_timeout_s!r}: a positive number of seconds is required")
    bdir = DATA / "batches" / plan["name"]
    bdir.mkdir(parents=True, exist_ok=True)
    with open(bdir / "batch.log", "a", encoding="utf-8") as logf:
        def log(msg):
            ff.log(msg)
            with contextlib.suppress(OSError, ValueError):
                logf.write(time.strftime("%H:%M:%S ") + msg + "\n")
                logf.flush()
        with _batch_lock(bdir, log):
            return _Batch(plan, plan_path, bdir, log, jobs=jobs, draft=draft, only=only, encoder=encoder, asr=asr,
                          compiler=compiler, auto_calibrate=auto_calibrate, require_gpu=require_gpu,
                          allow_cpu_fallback=allow_cpu_fallback, reel_timeout_s=reel_timeout_s, lang=lang,
                          asr_provider=asr_provider).run()


def contact_sheet(rows, path, tw=144):
    """One frame (at 15 % of each reel) per reel, labelled, border = verdict."""
    import cv2
    from PIL import Image, ImageDraw
    from .qc import VERDICT_COLOR, _font
    items = [e for e in rows if e.get("file")]
    if not items:
        return None
    info = ff.probe(items[0]["file"])
    th = int(tw * info["height"] / info["width"])
    cols = min(8, len(items))
    nrows = math.ceil(len(items) / cols)
    img = Image.new("RGB", (cols * (tw + 12) + 12, nrows * (th + 40) + 12), (18, 18, 22))
    d = ImageDraw.Draw(img)
    f = _font(12)
    for i, e in enumerate(items):
        x, y = 12 + (i % cols) * (tw + 12), 12 + (i // cols) * (th + 40)
        dur = float(e.get("duration") or 1.0)
        try:
            frame = next(ff.frames(e["file"], tw, th, start=max(0.0, dur * 0.15), duration=0.5), None)
        except ff.FFError:
            frame = None
        col = VERDICT_COLOR.get(e.get("verdict"), (90, 90, 90))
        d.rectangle([x - 3, y - 3, x + tw + 2, y + th + 2], fill=col)
        if frame is not None:
            img.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (x, y))
        d.text((x, y + th + 5), f"{e['n']:02d} {e['title']}"[:22], font=f, fill=(230, 230, 230))
        d.text((x, y + th + 20), f"{e.get('verdict')}  {dur:.1f}s", font=f, fill=col)
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.jpg")
    try:
        img.save(tmp, quality=88)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return path


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Render every reel of a plan")
    ap.add_argument("plan")
    ap.add_argument("-j", "--jobs", type=int)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--encoder", choices=("auto", "cpu", "nvenc"), default="auto")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--no-cpu-fallback", action="store_true")
    ap.add_argument("--reel-timeout", type=float, help="seconds a reel's render (and its QC) may take")
    ap.add_argument("--lang")
    ap.add_argument("--calibrate", action="store_true", help="measure the best pool size and exit")
    a = ap.parse_args()
    if a.calibrate:
        enc = resolve_encoder(a.encoder, a.require_gpu, not a.no_cpu_fallback)
        print(json.dumps(calibrate(encoder=enc["actual"], gpu=enc.get("gpu")), indent=1))
        return
    try:
        m = run(a.plan, a.jobs, a.draft, a.only, encoder=a.encoder, asr=not a.no_asr, require_gpu=a.require_gpu,
                allow_cpu_fallback=not a.no_cpu_fallback, reel_timeout_s=a.reel_timeout, lang=a.lang)
    except PlanError as e:
        sys.exit(str(e))
    print(json.dumps({"generation": m["generation"], "counts": m["counts"], "summary": m["summary"],
                      "out_dir": m["out_dir"]}))


if __name__ == "__main__":
    _main()
