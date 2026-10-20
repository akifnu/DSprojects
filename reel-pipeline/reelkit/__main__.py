"""`reel` command line.

  reel check                                 tools, models, versions
  reel ingest FILE… [--kind sample|source|music|sfx|image]
  reel assets [--kind K]
  reel sample REF [--lang en] [--force]      measured style spec + review sheet
  reel source REF [--lang en] [--prepare]    transcript (+ editing copy) of long-form footage
  reel transcript REF [--lang en]            compact sentence index for planning
  reel plan NAME --style REF --source S=REF  scaffold data/batches/NAME/plan.json
  reel batch PLAN [-j N] [--draft] [--only r01,r02]
  reel render TIMELINE OUT [--draft] [--still T]
  reel qc MP4 [--timeline T] [--style S]
  reel loop CLIP SECONDS OUT                 seamless endless loop

  reel job new FILE… [--count N] [--duration S] [--lang en] [--notes TEXT]
  reel job advance ID|latest                 run mechanical steps until an agent is needed
  reel job done ID STEP [--note T] [--fixes JSON]   an agent finished verify_style|plan|review
  reel job status ID|latest
  reel job list
"""
import argparse
import json
import sys
from pathlib import Path

from . import ff, store
from .paths import DATA, FFMPEG, MODELS


def _asset(ref, kind):
    try:
        return store.get(ref)
    except KeyError:
        if Path(ref).exists():
            return store.ingest(ref, kind)
        raise SystemExit(f"unknown asset or file: {ref}")


def cmd_check(a):
    import subprocess
    ok = True
    v = subprocess.run([FFMPEG, "-hide_banner", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"ffmpeg   {v}")
    filters = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    for f in ("libvmaf", "lut3d", "loudnorm", "zscale"):
        has = f" {f} " in filters
        ok &= has
        print(f"  filter {f:10s} {'ok' if has else 'MISSING'}")
    for m in ("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "sherpa-onnx-whisper-turbo", "silero_vad.onnx",
              "yunet/yunet.onnx"):
        has = (MODELS / m).exists()
        ok &= has
        print(f"model    {m:45s} {'ok' if has else 'MISSING'}")
    for mod in ("numpy", "cv2", "PIL", "sherpa_onnx", "scipy"):
        try:
            m = __import__(mod)
            print(f"python   {mod:12s} {getattr(m, '__version__', '?')}")
        except ImportError:
            ok = False
            print(f"python   {mod:12s} MISSING")
    print(f"data     {DATA}")
    print("READY" if ok else "NOT READY: run reel-pipeline/setup.sh")
    return 0 if ok else 1


def cmd_ingest(a):
    for f in a.files:
        x = store.ingest(f, a.kind)
        print(x.id, x.kind, x.meta["name"])


def cmd_assets(a):
    for x in store.all_assets(a.kind):
        i = x.info
        print(f"{x.id}  {x.kind:7s} {i.get('duration', 0):8.1f}s  {i.get('width', '-')}x{i.get('height', '-')}  "
              f"{x.meta['name']}")


def cmd_sample(a):
    from . import sample
    st = sample.analyze(_asset(a.ref, "sample").id, a.lang, a.force)
    x = store.get(st["sample"]["id"])
    print(f"style:  {x.cache('style.json')}\nreview: {x.cache('review.jpg')}")


def cmd_source(a):
    from . import asr
    x = _asset(a.ref, "source")
    tr = asr.transcribe(str(x.path), a.lang, str(x.cache(f"transcript.{a.lang}.json")))
    p = x.cache(f"transcript.{a.lang}.txt")
    p.write_text(asr.compact(tr))
    print(f"{x.id}: {len(tr['sentences'])} sentences, {tr['duration']:.0f}s → {p}")
    if a.prepare:
        from . import source
        source.prepare(x)


def cmd_transcript(a):
    from . import asr
    x = _asset(a.ref, "source")
    tr = asr.transcribe(str(x.path), a.lang, str(x.cache(f"transcript.{a.lang}.json")))
    print(asr.compact(tr))


def cmd_plan(a):
    d = DATA / "batches" / a.name
    d.mkdir(parents=True, exist_ok=True)
    srcs = dict(s.split("=", 1) for s in a.source)
    plan = {"name": a.name, "style": _asset(a.style, "sample").id,
            "output": {"width": 720, "height": 1280, "fps": 30, "quality": "final", "loudness": -14},
            "sources": {k: _asset(v, "source").id for k, v in srcs.items()},
            "music": None, "sfx": None, "reels": []}
    p = d / "plan.json"
    if p.exists():
        raise SystemExit(f"{p} exists")
    p.write_text(json.dumps(plan, indent=1))
    print(p)


def cmd_batch(a):
    from . import batch
    plan = Path(a.plan)
    if not plan.exists():
        plan = DATA / "batches" / a.plan / "plan.json"
    batch.run(str(plan), jobs=a.jobs, draft=a.draft, only=a.only.split(",") if a.only else None)


def cmd_render(a):
    from . import render
    if a.still is not None:
        print(render.still(a.timeline, a.still, a.out))
    else:
        print(render.render(a.timeline, a.out, draft=a.draft))


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


def cmd_job(a):
    from . import orchestrator as orch
    if a.action == "new":
        job = orch.create(a.args, count=a.count, lang=a.lang, duration=a.duration, notes=a.notes or "")
        print(job["id"])
    elif a.action == "advance":
        job = orch.advance(a.args[0] if a.args else "latest")
        print(json.dumps(orch.status(job["id"]), indent=1, ensure_ascii=False))
    elif a.action == "done":
        fixes = json.loads(a.fixes) if a.fixes else None
        orch.complete(a.args[0], a.args[1], note=a.note or "", **({"fixes": fixes} if fixes is not None else {}))
        print(json.dumps(orch.status(a.args[0]), indent=1, ensure_ascii=False))
    elif a.action == "status":
        print(json.dumps(orch.status(a.args[0] if a.args else "latest"), indent=1, ensure_ascii=False))
    elif a.action == "list":
        for p in sorted((DATA / "jobs").glob("*/job.json")):
            j = json.loads(p.read_text())
            print(f"{j['id']}  {j['status']:28s} rounds {j['rounds']}  {j['updated']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="reel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    p = sub.add_parser("ingest")
    p.add_argument("files", nargs="+")
    p.add_argument("--kind", default="source", choices=store.KINDS)
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("assets")
    p.add_argument("--kind")
    p.set_defaults(fn=cmd_assets)
    p = sub.add_parser("sample")
    p.add_argument("ref")
    p.add_argument("--lang", default="en")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_sample)
    p = sub.add_parser("source")
    p.add_argument("ref")
    p.add_argument("--lang", default="en")
    p.add_argument("--prepare", action="store_true")
    p.set_defaults(fn=cmd_source)
    p = sub.add_parser("transcript")
    p.add_argument("ref")
    p.add_argument("--lang", default="en")
    p.set_defaults(fn=cmd_transcript)
    p = sub.add_parser("plan")
    p.add_argument("name")
    p.add_argument("--style", required=True)
    p.add_argument("--source", action="append", required=True, help="KEY=asset-or-file")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("batch")
    p.add_argument("plan")
    p.add_argument("-j", "--jobs", type=int)
    p.add_argument("--draft", action="store_true")
    p.add_argument("--only")
    p.set_defaults(fn=cmd_batch)
    p = sub.add_parser("render")
    p.add_argument("timeline")
    p.add_argument("out")
    p.add_argument("--draft", action="store_true")
    p.add_argument("--still", type=float)
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
    p = sub.add_parser("job", help="agentic job orchestration")
    p.add_argument("action", choices=["new", "advance", "done", "status", "list"])
    p.add_argument("args", nargs="*")
    p.add_argument("--count", type=int)
    p.add_argument("--duration", type=float)
    p.add_argument("--lang", default="en")
    p.add_argument("--notes")
    p.add_argument("--note")
    p.add_argument("--fixes", help='JSON list: [{"reel": "r03", "what": "...", "how": "..."}]')
    p.set_defaults(fn=cmd_job)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
