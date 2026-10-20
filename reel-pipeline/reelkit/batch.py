"""Batch production: plan.json → N finished, QC'd reels, unattended.

    run(plan_path, jobs=None, draft=False, only=None)

1. Validate the plan (fail fast, every problem listed at once).
2. Compile each reel to a timeline (compile.compile_reel, in this process, in plan
   order) — or take a ready timeline when a reel carries "timeline": "path.json".
3. Render + QC each reel in a process pool. Pool size comes from calibrate(), which
   measures throughput of 1-4 concurrent renders × OpenCV threads on this machine
   (cached per machine); `jobs=` overrides.
4. Retry a reel once if its render raises or QC fails.
5. Cache: a reel is skipped when hash(timeline + media files + code + tool versions)
   matches its last finished render — interrupted batches resume where they stopped,
   and changing one reel's plan re-renders only that reel.
6. Write manifest.json / manifest.csv, deliverables out/NN_<slug>.mp4 (QC failures
   go to out/_review/), and out/contact_sheet.jpg.

Layout: data/batches/<name>/{plan.json, batch.log, manifest.*, reels/<id>/…, out/}.
"""
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import __version__, ff
from .paths import DATA, FFMPEG, ROOT

RENDER_CODE = ["render.py", "captions.py", "grade.py", "ff.py"]


class PlanError(ValueError):
    pass


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
    if not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(name)):
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
        if not rid or not re.fullmatch(r"[A-Za-z0-9_-]+", str(rid)):
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


def load_plan(plan_path):
    plan_path = Path(plan_path)
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise PlanError(f"{plan_path}: {e}") from None
    errs = validate(plan, plan_path.parent)
    if errs:
        raise PlanError(f"{plan_path} has {len(errs)} problem(s):\n  - " + "\n  - ".join(errs))
    return plan


def slug(text, n=50):
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s[:n].rstrip("-") or "reel"


# ------------------------------------------------------------------ cache keys

_FF_VERSION = None


def ffmpeg_version():
    global _FF_VERSION
    if _FF_VERSION is None:
        out = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True).stdout
        _FF_VERSION = out.splitlines()[0] if out else "unknown"
    return _FF_VERSION


def _code_hash(files):
    h = hashlib.sha256()
    for f in files:
        p = ROOT / "reelkit" / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def render_key(tl, base_dir, draft, encoder):
    from .render import _resolve
    media = {}
    for k, v in sorted(tl.get("media", {}).items()):
        p = _resolve(v, base_dir)
        try:
            st = os.stat(p)
            media[k] = [p, st.st_size, st.st_mtime_ns]
        except OSError:
            media[k] = [p, None, None]
    blob = json.dumps({"tl": tl, "media": media, "draft": draft, "encoder": encoder, "v": __version__,
                       "ffmpeg": ffmpeg_version(), "code": _code_hash(RENDER_CODE)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def qc_key(style, asr):
    blob = json.dumps({"style": style, "asr": asr, "code": _code_hash(["qc.py", "asr.py"])}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ worker

def _worker_init(cv_threads, asr_threads):
    import cv2
    if cv_threads:
        cv2.setNumThreads(int(cv_threads))
    if asr_threads:
        from . import asr
        asr._threads = lambda: int(asr_threads)


def _render_job(job):
    """Render + QC one reel (in a worker process). Retries once on error or QC fail."""
    from . import qc, render
    out = Path(job["out"])
    rdir = out.parent
    result = {"id": job["id"], "attempts": 0, "error": None, "render_s": 0.0, "qc_s": 0.0}
    tl_path = job["timeline"]
    style = json.loads(Path(job["style"]).read_text()) if job.get("style") else None
    for attempt in (1, 2):
        result["attempts"] = attempt
        try:
            if not job.get("qc_only"):
                t0 = time.monotonic()
                tmp = rdir / (out.stem + ".part.mp4")
                render.render(tl_path, tmp, draft=job["draft"], encoder=job["encoder"])
                os.replace(tmp, out)
                result["render_s"] = round(time.monotonic() - t0, 2)
            t0 = time.monotonic()
            rep = qc.check(out, tl_path, style, asr=job["asr"], sheet_path=str(rdir / "sheet.jpg"))
            result["qc_s"] = round(time.monotonic() - t0, 2)
            (rdir / "qc.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
            result["verdict"] = rep["verdict"]
            result["issues"] = rep["issues"]
            result["error"] = None
            if rep["verdict"] != "fail" or attempt == 2 or job.get("qc_only"):
                break
            job = dict(job, qc_only=False)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            result["verdict"] = "error"
            result["issues"] = []
            if attempt == 2:
                break
    return result


# ------------------------------------------------------------------ calibration

def _machine():
    return f"{os.cpu_count()}cpu|{ffmpeg_version()}|reelkit {__version__}"


def _calib_path():
    return DATA / "batches" / "calibration.json"


def _calib_timeline(dir_, seconds):
    """A representative 720p reel: 1080p source, cuts, a keyframed push, punch-ins,
    captions with pop + highlight, voice + ducked music."""
    dir_.mkdir(parents=True, exist_ok=True)
    src, tone, mus = dir_ / "src.mp4", dir_ / "voice.wav", dir_ / "music.wav"
    if not src.exists():
        ff.ffmpeg("-f", "lavfi", "-i", f"testsrc2=s=1080x1920:r=30:d={seconds + 4}", "-c:v", "libx264",
                  "-preset", "faster", "-crf", "12", "-g", "30", "-pix_fmt", "yuv420p", src)
        ff.ffmpeg("-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=48000:duration={seconds + 4}",
                  "-af", "volume=0.4", tone)
        ff.ffmpeg("-f", "lavfi", "-i", f"sine=frequency=880:sample_rate=48000:duration={seconds + 4}",
                  "-af", "volume=0.2", mus)
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
    p.write_text(json.dumps(tl))
    return p


def _calib_render(args):
    tl, out, cv_threads = args
    _worker_init(cv_threads, None)
    from . import render
    t0 = time.monotonic()
    render.render(tl, out)
    return time.monotonic() - t0


def calibrate(seconds=8.0, concurrency=(1, 2, 3, 4), save=True):
    """Measure reels/hour for 30 s 720p reels at each pool size × OpenCV threads;
    store and return the best configuration for this machine."""
    cores = os.cpu_count() or 4
    d = DATA / "batches" / "_calibration"
    tl = _calib_timeline(d, seconds)
    results = []
    for c in concurrency:
        for cvt in sorted({0, max(1, cores // c)}):
            t0 = time.monotonic()
            r0 = os.times()
            with ProcessPoolExecutor(c) as pool:
                list(pool.map(_calib_render, [(str(tl), str(d / f"out{k}.mp4"), cvt) for k in range(c)]))
            wall = time.monotonic() - t0
            r1 = os.times()
            cpu = (r1.children_user - r0.children_user) + (r1.children_system - r0.children_system)
            rph = c * 3600.0 / wall * (seconds / 30.0)
            results.append({"jobs": c, "cv_threads": cvt, "wall_s": round(wall, 2), "cpu_s": round(cpu, 1),
                            "reels_per_hour_30s": round(rph, 1)})
            ff.log(f"calibrate: {c} concurrent × cv threads {cvt or 'auto'}: {wall:.1f}s wall, "
                   f"{cpu:.0f} CPU-s → {rph:.0f} reels/h (30 s, 720p)")
    best = max(results, key=lambda r: r["reels_per_hour_30s"])
    out = {"machine": _machine(), "seconds": seconds, "results": results, "best": best,
           "measured": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if save:
        p = _calib_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        allc = json.loads(p.read_text()) if p.exists() else {}
        allc[_machine()] = out
        p.write_text(json.dumps(allc, indent=1))
    return out


def concurrency(jobs=None, auto_calibrate=True):
    """(jobs, cv_threads) for this machine: explicit, calibrated, or a safe default."""
    cores = os.cpu_count() or 4
    if jobs:
        return int(jobs), max(1, cores // int(jobs))
    p = _calib_path()
    if p.exists():
        best = json.loads(p.read_text()).get(_machine(), {}).get("best")
        if best:
            return best["jobs"], best["cv_threads"]
    if auto_calibrate:
        ff.log("no calibration for this machine yet: measuring (once, ~2 min)")
        best = calibrate()["best"]
        return best["jobs"], best["cv_threads"]
    return max(1, cores // 2), 2


# ------------------------------------------------------------------ run

def _compile(plan, reel, style, rdir, plan_dir, compiler):
    """timeline.json for one reel: a ready timeline (media made absolute) or compile_reel()."""
    from .render import _resolve
    rdir.mkdir(parents=True, exist_ok=True)
    dest = rdir / "timeline.json"
    if reel.get("timeline"):
        src = Path(reel["timeline"])
        src = src if src.is_absolute() else Path(plan_dir) / src
        tl = json.loads(src.read_text())
        tl["media"] = {k: _resolve(v, str(src.resolve().parent)) for k, v in tl.get("media", {}).items()}
    else:
        if compiler is None:
            from .compile import compile_reel as compiler
        res = compiler(plan, reel, style, str(rdir))
        if isinstance(res, dict):
            tl = res
        elif res is not None and Path(res).exists():
            tl = json.loads(Path(res).read_text())
            if Path(res).resolve() != dest.resolve():
                tl["media"] = {k: _resolve(v, str(Path(res).resolve().parent)) for k, v in tl.get("media", {}).items()}
        else:
            tl = json.loads(dest.read_text())
    out = plan.get("output") or {}
    for k in ("width", "height", "fps"):
        if k in out and tl.get(k) != out[k]:
            raise ValueError(f"timeline {k} {tl.get(k)} differs from output.{k} {out[k]}")
    tmp = dest.with_suffix(".part")
    tmp.write_text(json.dumps(tl, indent=1, ensure_ascii=False))
    os.replace(tmp, dest)
    return dest, tl


def run(plan_path, jobs=None, draft=False, only=None, encoder="cpu", asr=True, compiler=None, auto_calibrate=True):
    """Produce every reel of a plan. Returns the manifest dict."""
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path)
    plan_dir = plan_path.parent
    bdir = DATA / "batches" / plan["name"]
    bdir.mkdir(parents=True, exist_ok=True)
    if plan_path != (bdir / "plan.json").resolve():
        shutil.copy2(plan_path, bdir / "plan.json")
    style_p = _style_path(plan["style"], plan_dir)
    style = json.loads(style_p.read_text())
    reels = [r for r in plan["reels"] if not only or r["id"] in set(only)]
    if only and len(reels) != len(set(only)):
        missing = set(only) - {r["id"] for r in reels}
        raise PlanError(f"--only: no reels {sorted(missing)} in the plan")
    order = {r["id"]: i for i, r in enumerate(plan["reels"])}
    width = max(2, len(str(len(plan["reels"]))))
    n_jobs, cvt = concurrency(jobs, auto_calibrate)
    logf = open(bdir / "batch.log", "a")

    def log(msg):
        ff.log(msg)
        logf.write(time.strftime("%H:%M:%S ") + msg + "\n")
        logf.flush()

    log(f"batch {plan['name']}: {len(reels)} reels, {n_jobs} workers (cv threads {cvt}), "
        f"{'draft' if draft else 'final'} quality")
    qk = qc_key(style, asr)
    entries, started = {}, time.monotonic()
    done_times = []
    with ProcessPoolExecutor(n_jobs, initializer=_worker_init, initargs=(cvt, max(1, (os.cpu_count() or 4) // n_jobs))) as pool:
        futs = {}
        for r in reels:
            rid = r["id"]
            rdir = bdir / "reels" / rid
            e = {"id": rid, "n": order[rid] + 1, "title": r.get("title", rid), "cached": False}
            entries[rid] = e
            try:
                tl_path, tl = _compile(plan, r, style, rdir, plan_dir, compiler)
            except Exception as ex:
                e.update(verdict="error", error=f"compile: {type(ex).__name__}: {ex}", issues=[])
                log(f"{rid}: compile failed — {ex}")
                continue
            e["duration"] = tl["duration"]
            e["timeline"] = str(tl_path)
            key = render_key(tl, str(tl_path.parent), draft, encoder)
            out = rdir / f"{rid}.mp4"
            cache = rdir / "cache.json"
            prev = json.loads(cache.read_text()) if cache.exists() else {}
            job = {"id": rid, "timeline": str(tl_path), "out": str(out), "draft": draft, "encoder": encoder,
                   "style": str(style_p), "asr": asr, "key": key, "qc_key": qk}
            if prev.get("key") == key and out.exists() and (rdir / "qc.json").exists():
                if prev.get("qc_key") == qk:
                    rep = json.loads((rdir / "qc.json").read_text())
                    e.update(verdict=rep["verdict"], issues=rep["issues"], cached=True,
                             render_s=prev.get("render_s", 0.0))
                    log(f"{rid}: cached ({rep['verdict']})")
                    continue
                job["qc_only"] = True
            futs[pool.submit(_render_job, job)] = job
        total = len(futs)
        for k, fut in enumerate(as_completed(futs), 1):
            job = futs[fut]
            e = entries[job["id"]]
            try:
                res = fut.result()
            except Exception as ex:                       # worker crashed
                res = {"verdict": "error", "error": f"{type(ex).__name__}: {ex}", "issues": [], "attempts": 1}
            e.update(verdict=res.get("verdict"), issues=res.get("issues", []), error=res.get("error"),
                     attempts=res.get("attempts"), render_s=res.get("render_s", 0.0), qc_s=res.get("qc_s", 0.0))
            if res.get("verdict") in ("pass", "warn", "fail"):
                cache = Path(job["out"]).parent / "cache.json"
                cache.write_text(json.dumps({"key": job["key"], "qc_key": job["qc_key"], "verdict": res["verdict"],
                                             "render_s": res.get("render_s", 0.0),
                                             "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}))
            done_times.append(res.get("render_s", 0.0) + res.get("qc_s", 0.0))
            elapsed = time.monotonic() - started
            per = elapsed / k
            eta = per * (total - k)
            log(f"{k}/{total} {job['id']}: {res.get('verdict')}"
                f"{' — ' + res['error'] if res.get('error') else ''}  "
                f"(elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min)")
    manifest = _deliver(plan, bdir, entries, width, time.monotonic() - started, n_jobs, draft)
    log(f"done: {manifest['summary']}  → {bdir / 'out'}")
    logf.close()
    return manifest


def _deliver(plan, bdir, entries, width, elapsed, n_jobs, draft):
    out = bdir / "out"
    review = out / "_review"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in plan["reels"]:
        e = entries.get(r["id"])
        if e is None:
            continue
        src = bdir / "reels" / r["id"] / f"{r['id']}.mp4"
        name = f"{e['n']:0{width}d}_{slug(e['title'])}.mp4"
        for old in list(out.glob(f"{e['n']:0{width}d}_*.mp4")) + list(review.glob(f"{e['n']:0{width}d}_*.mp4")):
            old.unlink()
        dest = None
        if src.exists() and e.get("verdict") in ("pass", "warn", "fail"):
            dest = (review if e["verdict"] == "fail" else out) / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
        e["file"] = str(dest) if dest else None
        e["qc"] = str(src.parent / "qc.json") if (src.parent / "qc.json").exists() else None
        e["sheet"] = str(src.parent / "sheet.jpg") if (src.parent / "sheet.jpg").exists() else None
        rows.append(e)
    counts = {}
    for e in rows:
        counts[e.get("verdict")] = counts.get(e.get("verdict"), 0) + 1
    manifest = {"name": plan["name"], "style": plan["style"], "quality": "draft" if draft else "final",
                "jobs": n_jobs, "elapsed_s": round(elapsed, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "summary": counts, "reels": rows}
    (bdir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    with open(bdir / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "id", "title", "duration", "verdict", "issues", "file", "cached"])
        for e in rows:
            w.writerow([e["n"], e["id"], e["title"], e.get("duration"), e.get("verdict"),
                        " | ".join(f"[{i['severity']}] {i['check']}: {i['msg']}" for i in e.get("issues") or [])
                        or e.get("error") or "", e.get("file") or "", e.get("cached")])
    contact_sheet(rows, out / "contact_sheet.jpg")
    return manifest


def contact_sheet(rows, path, tw=144):
    """One frame (at 15 % of each reel) per reel, labelled, border = verdict."""
    import cv2
    import numpy as np
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
        t = max(0.0, min(e.get("duration") or 1.0, (e.get("duration") or 1.0) * 0.15))
        frame = next(ff.frames(e["file"], tw, th, start=t, duration=0.5), None)
        col = VERDICT_COLOR.get(e.get("verdict"), (90, 90, 90))
        d.rectangle([x - 3, y - 3, x + tw + 2, y + th + 2], fill=col)
        if frame is not None:
            img.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (x, y))
        d.text((x, y + th + 5), f"{e['n']:02d} {e['title']}"[:22], font=f, fill=(230, 230, 230))
        d.text((x, y + th + 20), f"{e.get('verdict')}  {e.get('duration', 0):.1f}s", font=f, fill=col)
    img.save(path, quality=88)
    return path


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Render every reel of a plan")
    ap.add_argument("plan")
    ap.add_argument("-j", "--jobs", type=int)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="measure the best pool size and exit")
    a = ap.parse_args()
    if a.calibrate:
        print(json.dumps(calibrate(), indent=1))
        return
    try:
        m = run(a.plan, a.jobs, a.draft, a.only, asr=not a.no_asr)
    except PlanError as e:
        sys.exit(str(e))
    print(json.dumps(m["summary"]))


if __name__ == "__main__":
    _main()
