"""Readiness and encoder resolution: can this host run the production line as asked?

resolve_encoder(requested, require_gpu, allow_cpu_fallback) turns auto | cpu | nvenc into the
encoder a render actually uses: {"requested", "actual": "cpu" | "nvenc", "reason", "gpu"}.
"cpu" never touches the GPU. Otherwise NVENC counts only after a real 30-frame h264_nvenc
encode with the renderer's settings (an encoder listed by `ffmpeg -encoders` can still lack a
driver, a device or an encode session); the result is memoised in-process and in
DATA/readiness/encoder.json per ffmpeg build, GPU, driver and visible devices. When NVENC is
unusable, require_gpu (or nvenc / auto without CPU fallback) raises NotReady; otherwise the CPU
encoder is used and the reason says why.

check(...) / `reel check`: required Python imports, pinned package versions, ffmpeg ≥ 9 and
ffprobe with the filters and encoders the pipeline uses, every model file present and
non-empty, the data directory writable, free disk, caption fonts, GPU identity (nvidia-smi),
the encoder resolution and the ASR provider ("cuda" only with a CUDA build of sherpa-onnx and a
visible GPU). --deep also verifies the models against $REELKIT_MODELS/CHECKSUMS and loads and
runs each model once, each in its own process. JSON report on stdout, a summary on stderr;
exit 0 ready, 1 not ready.

Standard library only at import (it has to report what is missing); paths.py resolves ffmpeg
lazily, so a missing ffmpeg is reported here, not raised.
"""
import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path

from . import fsutil, paths

ENCODERS = ("auto", "cpu", "nvenc")
IMPORTS = ("numpy", "cv2", "PIL", "scipy", "sherpa_onnx", "onnxruntime", "rapidocr_onnxruntime")
FILTERS = ("lut3d", "loudnorm", "ebur128", "colorspace", "libvmaf", "zscale")
CODECS = ("libx264", "aac", "flac")             # CPU video, delivery audio, lossless source audio
MIN_FFMPEG = 9
# the renderer's NVENC settings (render._encoder_args) on a synthetic 720p source, no output file
SMOKE = ["-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=30",
         "-frames:v", "30", "-pix_fmt", "yuv420p", "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19",
         "-b:v", "0", "-bf", "2", "-f", "null", "-"]
TOOL_TIMEOUT, SMOKE_TIMEOUT, LOAD_TIMEOUT = 30, 90, 900
OK_TTL, NO_GPU_TTL, FAIL_TTL = 7 * 86400, 86400, 300    # s a probe result stands (see _nvenc)
DEEP_LOADS = ("yunet", "vad", "ocr", "parakeet", "whisper")
ALTERNATIVES = {"onnxruntime": ("onnxruntime", "onnxruntime-gpu")}     # the GPU profile's wheel satisfies the pin


class NotReady(RuntimeError):
    """The requested execution profile cannot run on this host (e.g. a required GPU encoder)."""


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _run(cmd, timeout, **kw):
    """(returncode | None, stdout, stderr); None when it could not run or timed out."""
    try:
        p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
                           timeout=timeout, **kw)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout:g} s"
    except OSError as e:
        return None, "", f"{type(e).__name__}: {e.strerror or e}"


def _tail(text, n=3, chars=400):
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    return " | ".join(lines[-n:])[-chars:]


def _names(listing):
    """Filter / encoder names from `ffmpeg -filters` / `-encoders` (the token after the flags)."""
    out = set()
    for line in listing.splitlines():
        p = line.split()
        if len(p) >= 2 and re.fullmatch(r"[A-Z.|]{2,8}", p[0]):
            out.add(p[1])
    return out


def _version_line(exe):
    rc, out, err = _run([exe, "-hide_banner", "-version"], TOOL_TIMEOUT)
    if rc != 0 or not out.strip():
        return None, _tail(err) or f"exit {rc}"
    return out.splitlines()[0].strip(), None


def _major(line):
    """Major version from 'ffmpeg version n9.0.2-…'; "git" for a development (N-…) build."""
    m = re.search(r"version (\S+)", line or "")
    v = m.group(1) if m else ""
    if v.startswith("N-"):
        return "git"
    m = re.match(r"n?(\d+)\.", v)
    return int(m.group(1)) if m else None


def _dist(name):
    for d in ALTERNATIVES.get(name, (name,)):
        try:
            return d, metadata.version(d)
        except metadata.PackageNotFoundError:
            continue
    return None, None


# ------------------------------------------------------------------ GPU and encoder

def gpu_info():
    """nvidia-smi's view: {"present", "gpus": [{index, name, driver, memory_mib, uuid}], "label", "detail"}."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"present": False, "gpus": [], "label": None, "detail": "nvidia-smi not found: no NVIDIA driver visible"}
    rc, out, err = _run([smi, "--query-gpu=index,name,driver_version,memory.total,uuid",
                         "--format=csv,noheader,nounits"], 20)
    gpus = []
    for line in (out or "").strip().splitlines() if rc == 0 else []:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            gpus.append({"index": p[0], "name": p[1], "driver": p[2], "memory_mib": p[3], "uuid": p[4]})
    if not gpus:
        return {"present": False, "gpus": [], "label": None,
                "detail": f"nvidia-smi lists no usable GPU ({_tail(err) or _tail(out) or f'exit {rc}'})"}
    label = f"{gpus[0]['name']} (driver {gpus[0]['driver']})"
    return {"present": True, "gpus": gpus, "label": label,
            "detail": label + (f" + {len(gpus) - 1} more" if len(gpus) > 1 else "") + f", {gpus[0]['memory_mib']} MiB"}


def _nvenc_probe(ffmpeg):
    """(ok, reason) of one real h264_nvenc encode."""
    rc, out, err = _run([ffmpeg, "-hide_banner", "-encoders"], TOOL_TIMEOUT)
    if rc != 0:
        return False, f"ffmpeg -encoders failed: {_tail(err) or rc}"
    if "h264_nvenc" not in _names(out):
        return False, "this ffmpeg build has no h264_nvenc encoder"
    t0 = time.monotonic()
    rc, out, err = _run([ffmpeg, *SMOKE], SMOKE_TIMEOUT)
    if rc == 0:
        return True, f"30-frame h264_nvenc encode passed in {time.monotonic() - t0:.1f} s"
    return False, f"h264_nvenc smoke encode failed ({'timed out' if rc is None else f'exit {rc}'}): " \
                  f"{_tail(err) or 'no error output'}"


_memo, _memo_lock = {}, threading.Lock()


def _nvenc():
    """(ok, reason, gpu label) of NVENC on this host. Memoised per process (a failure for FAIL_TTL)
    and in DATA/readiness/encoder.json per ffmpeg build, GPUs, driver and visible devices: a
    success for OK_TTL, a failure without a visible GPU for NO_GPU_TTL, a failure with a GPU
    never (the device may just have been busy)."""
    ffmpeg = paths.tool("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg not found (run reel-pipeline/setup.sh)", None
    with _memo_lock:
        hit = _memo.get(ffmpeg)
        if hit and (hit[1][0] or time.monotonic() - hit[0] < FAIL_TTL):
            return hit[1]
        res = _nvenc_cached(ffmpeg)
        _memo[ffmpeg] = (time.monotonic(), res)
        return res


def _nvenc_cached(ffmpeg):
    gpu = gpu_info()
    label = gpu["label"]
    version, why = _version_line(ffmpeg)
    if not version:
        return False, f"ffmpeg does not run: {why}", label
    key = fsutil.sha256_json({"ffmpeg": ffmpeg, "version": version, "gpus": gpu["gpus"],
                              "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                                                                     "NVIDIA_DRIVER_CAPABILITIES")}})[:24]
    path = paths.DATA / "readiness" / "encoder.json"

    def entries():
        c = fsutil.read_json(path) if path.exists() else None
        return dict(c["entries"]) if isinstance(c, dict) and isinstance(c.get("entries"), dict) else {}

    def cached():
        e = entries().get(key)
        if not isinstance(e, dict) or not isinstance(e.get("ts"), (int, float)):
            return None
        ttl = OK_TTL if e.get("ok") else (0 if gpu["present"] else NO_GPU_TTL)
        if not 0 <= time.time() - e["ts"] <= ttl:
            return None
        return bool(e.get("ok")), f"{e.get('reason')} (probed {e.get('at')})", label
    hit = cached()
    if hit:
        return hit
    with contextlib.ExitStack() as stack:
        try:                                    # one probe at a time per data directory
            stack.enter_context(fsutil.file_lock(path, timeout=SMOKE_TIMEOUT + 30))
        except (OSError, TimeoutError):
            pass
        hit = cached()
        if hit:
            return hit
        ok, reason = _nvenc_probe(ffmpeg)
        if ok or not gpu["present"]:
            es = {k: v for k, v in entries().items() if isinstance(v, dict) and isinstance(v.get("ts"), (int, float))}
            es[key] = {"ok": ok, "reason": reason, "gpu": label, "ffmpeg": version, "at": _now(), "ts": time.time()}
            try:
                fsutil.atomic_write_json(path, {"version": 1, "entries": dict(sorted(
                    es.items(), key=lambda kv: -kv[1]["ts"])[:20])})
            except OSError as e:
                print(f"[reel] encoder probe result not cached ({e})", file=sys.stderr, flush=True)
        return ok, reason, label


def resolve_encoder(requested="auto", require_gpu=False, allow_cpu_fallback=True):
    """{"requested", "actual": "cpu" | "nvenc", "reason", "gpu"} for a render. Raises NotReady
    when NVENC is required (require_gpu, or nvenc / auto with allow_cpu_fallback false) and a
    real encode fails; never raises anything else."""
    requested = requested or "auto"
    if requested not in ENCODERS:
        raise NotReady(f"encoder {requested!r}: one of {', '.join(ENCODERS)}")
    if requested == "cpu":
        if require_gpu:
            raise NotReady("encoder cpu was requested together with require_gpu")
        return {"requested": "cpu", "actual": "cpu", "reason": "cpu requested", "gpu": None}
    try:
        ok, reason, gpu = _nvenc()
    except Exception as e:                      # a probe must not take the caller down
        ok, reason, gpu = False, f"encoder probe failed: {type(e).__name__}: {e}", None
    if ok:
        return {"requested": requested, "actual": "nvenc", "reason": reason, "gpu": gpu}
    if require_gpu:
        raise NotReady(f"a GPU encoder is required but NVENC is unusable: {reason}")
    if not allow_cpu_fallback:
        raise NotReady(f"NVENC is unusable and CPU fallback is not allowed: {reason}")
    return {"requested": requested, "actual": "cpu", "reason": f"NVENC unusable ({reason}): CPU encoder (libx264)",
            "gpu": gpu}


# ------------------------------------------------------------------ ASR provider and models

def _sherpa():
    """(versions, cuda build?, provider libraries) of the installed sherpa-onnx."""
    vers = {}
    for d in ("sherpa-onnx", "sherpa-onnx-core"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            vers[d] = metadata.version(d)
    libs = []
    try:
        spec = importlib.util.find_spec("sherpa_onnx")
    except (ImportError, ValueError):
        spec = None
    for d in (spec.submodule_search_locations or []) if spec else []:
        libs += sorted(p.name for p in Path(d).rglob("*onnxruntime_providers_cuda*"))
    return vers, bool(libs) or any("cuda" in v for v in vers.values()), libs


def _asr_provider(requested, gpu):
    try:
        from . import asr
        provider, known = asr.resolve_provider(requested), asr.PROVIDERS
    except ValueError as e:
        return {"requested": requested, "provider": None, "available": [], "ok": False, "reason": str(e)}
    except (Exception, SystemExit):             # asr needs numpy and ffmpeg: decide without it
        provider = str(requested or os.environ.get("REELKIT_ASR_PROVIDER") or "cpu").strip().lower()
        known = ("cpu", "cuda")
    vers, cuda, libs = _sherpa()
    ort = []
    with contextlib.suppress(Exception):
        import onnxruntime
        ort = list(onnxruntime.get_available_providers())
    available = (["cpu"] if vers else []) + (["cuda"] if cuda and gpu["present"] else [])
    ver = vers.get("sherpa-onnx") or vers.get("sherpa-onnx-core")
    if provider not in known:
        reason = f"unknown provider {provider!r} (one of {', '.join(known)})"
    elif not vers:
        reason = "sherpa-onnx is not installed"
    elif provider in available:
        reason = f"sherpa-onnx {ver}, {provider} provider" + (f" on {gpu['label']}" if provider == "cuda" else "")
    elif not cuda:
        reason = f"sherpa-onnx {ver} is a CPU build (no CUDA provider library): install the GPU profile (setup.sh --gpu)"
    else:
        reason = f"sherpa-onnx {ver} has the CUDA provider but no GPU is visible ({gpu['detail']})"
    return {"requested": requested, "provider": provider, "available": available, "ok": provider in available,
            "reason": reason, "sherpa": vers, "sherpa_cuda_libs": libs, "onnxruntime_providers": ort}


def _model_sets():
    """[(name, [files])] of every model the pipeline loads, from the modules that load them."""
    from . import asr                           # numpy + ffmpeg; the caller reports its failure
    enc = sorted(asr.WHISPER.glob("*-encoder.int8.onnx")) if asr.WHISPER.is_dir() else []
    prefix = enc[0].name.split("-encoder")[0] if enc else "<model>"
    try:
        from .reframe import YUNET
    except (Exception, SystemExit):
        YUNET = paths.MODELS / "yunet" / "yunet.onnx"
    sets = [("parakeet (English ASR)", [asr.PARAKEET / f for f in ("encoder.int8.onnx", "decoder.int8.onnx",
                                                                     "joiner.int8.onnx", "tokens.txt")]),
            ("whisper (other languages)", [asr.WHISPER / f"{prefix}-{f}" for f in ("encoder.int8.onnx",
                                                                                   "decoder.int8.onnx", "tokens.txt")]),
            ("silero vad", [asr.VAD]), ("yunet faces", [Path(YUNET)])]
    ocr = _ocr_models()
    if ocr is not None:
        sets.append(("ocr (rapidocr det + rec)", ocr))
    return sets


def _ocr_models():
    """rapidocr's detector and recogniser model files (its config.yaml), or None if not installed."""
    try:
        spec = importlib.util.find_spec("rapidocr_onnxruntime")
    except (ImportError, ValueError):
        return None
    if not spec or not spec.origin:
        return None
    root = Path(spec.origin).parent
    try:
        text = (root / "config.yaml").read_text(encoding="utf-8")
    except OSError:
        return [root / "config.yaml"]
    out, section = [], None
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s*$", line)
        if m:
            section = m.group(1)
            continue
        m = re.match(r"^\s+model_path:\s*(\S+)", line)
        if m and section in ("Det", "Rec"):
            out.append(root / m.group(1))
    return out or [root / "models"]


def _files(files):
    missing = [str(f) for f in files if not Path(f).is_file()]
    empty = [str(f) for f in files if Path(f).is_file() and Path(f).stat().st_size == 0]
    if missing or empty:
        return False, "; ".join([f"missing {f}" for f in missing] + [f"empty {f}" for f in empty])
    size = sum(Path(f).stat().st_size for f in files)
    return True, f"{len(files)} file(s), {size / 1e6:.0f} MB in {Path(files[0]).parent}"


def _checksums():
    """(ok | None, detail): model files against $REELKIT_MODELS/CHECKSUMS (sha256sum format)."""
    f = paths.MODELS / "CHECKSUMS"
    if not f.is_file():
        return None, f"no {f} (setup.sh records one when it installs the models)"
    bad, n = [], 0
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line)
        if not m:
            bad.append(f"unreadable line {line[:60]!r}")
            continue
        p = paths.MODELS / m.group(2)
        try:
            if fsutil.sha256_file(p) != m.group(1).lower():
                bad.append(f"{m.group(2)} differs from its recorded sha-256")
            n += 1
        except OSError as e:
            bad.append(f"{m.group(2)}: {e.strerror or e}")
    return not bad, (f"{n} file(s) match {f}" if not bad else "; ".join(bad[:6]))


def _load_child(what, provider):
    """Body of one `--deep` model check, in its own process: load the model and run it once on
    silence or a blank image. Prints {"ok", "detail"} as the last stdout line."""
    t0 = time.monotonic()
    try:
        import numpy as np
        if what == "yunet":
            import cv2
            from .reframe import YUNET
            cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320)).detect(np.zeros((320, 320, 3), np.uint8))
        elif what == "vad":
            from . import asr
            asr.vad_segments(np.zeros(asr.SR, np.float32))
        elif what == "ocr":
            from . import sample_text
            det, _ = sample_text._engine()
            det(np.full((96, 320, 3), 255, np.uint8))
        elif what in ("parakeet", "whisper"):
            from . import asr
            rec = asr._parakeet(provider) if what == "parakeet" else asr._whisper("en", provider)
            s = rec.create_stream()
            s.accept_waveform(asr.SR, np.zeros(asr.SR, np.float32))
            rec.decode_stream(s)
        else:
            raise ValueError(f"unknown model {what!r}")
        res = {"ok": True, "detail": f"loaded and ran in {time.monotonic() - t0:.1f} s"}
    except (Exception, SystemExit) as e:
        res = {"ok": False, "detail": f"{type(e).__name__}: {e}"[:600]}
    print(json.dumps(res), flush=True)
    return 0 if res["ok"] else 1


def _deep_load(what, provider):
    code = "import sys; from reelkit import readiness; sys.exit(readiness._load_child(sys.argv[1], sys.argv[2]))"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(x for x in (str(paths.ROOT), os.environ.get("PYTHONPATH")) if x))
    rc, out, err = _run([sys.executable, "-c", code, what, provider], LOAD_TIMEOUT, env=env, cwd=str(paths.ROOT))
    for line in reversed((out or "").splitlines()):
        if line.startswith("{"):
            with contextlib.suppress(ValueError):
                res = json.loads(line)
                return bool(res.get("ok")), str(res.get("detail"))
    why = "timed out" if rc is None else f"exit {rc}" + (f" (signal {-rc})" if rc < 0 else "")
    return False, f"the load crashed ({why}): {_tail(err) or 'no output'}"


# ------------------------------------------------------------------ the check

def _imports():
    out = []
    for name in IMPORTS:
        try:
            m = importlib.import_module(name)
        except KeyboardInterrupt:
            raise
        except BaseException as e:              # ImportError, OSError of a missing native library, SystemExit
            out.append((name, False, f"{type(e).__name__}: {e}"[:300]))
            continue
        out.append((name, True, str(getattr(m, "__version__", None) or "imported")))
    return out


def _pins():
    req = paths.ROOT / "requirements.txt"
    try:
        lines = req.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        return False, f"{req}: {e.strerror or e}"
    bad, n = [], 0
    for line in lines:
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)", line.split("#", 1)[0].strip())
        if not m:
            continue
        n += 1
        name, want = m.groups()
        dist, have = _dist(name)
        if have is None:
            bad.append(f"{name} missing")
        elif have.split("+")[0] != want:
            bad.append(f"{dist} {have} (pinned {want})")
    return not bad, f"{n} pinned package(s) as in requirements.txt" if not bad else "; ".join(bad)


def _ffmpeg(add):
    """ffmpeg / ffprobe components; the ffmpeg path or None."""
    exe = paths.tool("ffmpeg")
    if not exe:
        add("ffmpeg", False, "not found: run reel-pipeline/setup.sh (or set REELKIT_FFMPEG)")
        add("ffmpeg filters", False, "ffmpeg missing")
        add("ffmpeg encoders", False, "ffmpeg missing")
    else:
        line, why = _version_line(exe)
        major = _major(line)
        ok = major == "git" or (isinstance(major, int) and major >= MIN_FFMPEG)
        add("ffmpeg", ok, f"{line} ({exe})" + ("" if ok else f": version ≥ {MIN_FFMPEG} required") if line
            else f"{exe} does not run: {why}")
        for what, flag, need in (("filters", "-filters", FILTERS), ("encoders", "-encoders", CODECS)):
            rc, out, err = _run([exe, "-hide_banner", flag], TOOL_TIMEOUT)
            have = _names(out) if rc == 0 else set()
            miss = [x for x in need if x not in have]
            add(f"ffmpeg {what}", rc == 0 and not miss,
                (f"missing {', '.join(miss)}" if miss else ", ".join(need)) if rc == 0 else f"failed: {_tail(err)}")
    probe = paths.tool("ffprobe")
    if not probe:
        add("ffprobe", False, "not found: run reel-pipeline/setup.sh (or set REELKIT_FFPROBE)")
    else:
        line, why = _version_line(probe)
        add("ffprobe", bool(line), f"{line} ({probe})" if line else f"{probe} does not run: {why}")
    return exe


def _data(add):
    d = paths.DATA
    try:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "readiness" / f".write-test.{os.getpid()}"
        fsutil.atomic_write_text(p, "ok\n")
        p.unlink()
        add("data directory", True, f"{d} writable")
    except OSError as e:
        add("data directory", False, f"{d}: {e.strerror or e}")
    try:
        need = float(os.environ.get("REELKIT_MIN_FREE_GB") or 5)
    except ValueError:
        need = 5.0
    at = d
    while not at.exists() and at != at.parent:
        at = at.parent
    try:
        free = shutil.disk_usage(at).free / 2 ** 30
        add("free disk", free >= need, f"{free:.1f} GiB free at {at} (need ≥ {need:g}, REELKIT_MIN_FREE_GB)")
    except OSError as e:
        add("free disk", False, f"{at}: {e.strerror or e}")


def _fonts(add):
    try:
        from . import captions
    except (Exception, SystemExit) as e:
        add("fonts", False, f"not checked ({type(e).__name__}: {e})", required=False)
        return
    fb = [f for f in captions.FALLBACKS if os.path.exists(f)]
    lib = paths.LIBRARY / "fonts"
    n_lib = len(list(lib.glob("*.ttf"))) if lib.is_dir() else 0
    n_cache = len(list(paths.FONTS.glob("*.ttf"))) if paths.FONTS.is_dir() else 0
    add("fonts", bool(fb), f"{n_lib} in {lib}, {n_cache} cached in {paths.FONTS}; fallback "
                           f"{fb[0] if fb else 'MISSING (' + ', '.join(captions.FALLBACKS) + ')'}; "
                           + ("offline: no downloads (REELKIT_OFFLINE)" if captions.offline()
                              else "others download on first use"))


def check(encoder="auto", require_gpu=False, asr_provider="cpu", deep=False):
    """{"ready", "components": [{"name", "ok", "detail", "required"}], "encoder", "asr_provider", "gpu", …};
    ready = every required component ok. encoder / require_gpu / asr_provider: the profile to
    check (an nvenc or require_gpu profile needs a passing NVENC encode; auto passes with the CPU
    fallback); asr_provider None = REELKIT_ASR_PROVIDER or cpu."""
    comps = []

    def add(name, ok, detail, required=True):
        comps.append({"name": name, "ok": bool(ok), "detail": str(detail), "required": bool(required)})
    add("python", True, f"{sys.version.split()[0]} ({sys.executable})", required=False)
    for name, ok, detail in _imports():
        add(f"import {name}", ok, detail)
    ok, detail = _pins()
    add("python packages", ok, detail, required=False)
    _ffmpeg(add)
    try:
        sets = _model_sets()
    except (Exception, SystemExit) as e:
        sets = None
        add("models", False, f"not checked: the asr module does not load ({type(e).__name__}: {e})")
    for name, files in sets or []:
        ok, detail = _files(files)
        add(f"model {name}", ok, detail)
    _data(add)
    _fonts(add)
    gpu = gpu_info()
    need_nvenc = bool(require_gpu) or encoder == "nvenc"
    add("gpu", gpu["present"], gpu["detail"], required=need_nvenc or asr_provider == "cuda")
    try:
        enc = resolve_encoder(encoder, require_gpu=require_gpu, allow_cpu_fallback=not need_nvenc)
        add("encoder", True, f"{enc['actual']}: {enc['reason']}", required=need_nvenc)
    except NotReady as e:
        enc = {"requested": encoder, "actual": None, "reason": str(e), "gpu": gpu["label"]}
        add("encoder", False, str(e))
    asr = _asr_provider(asr_provider, gpu)
    add("asr provider", asr["ok"], asr["reason"])
    if deep:
        ok, detail = _checksums()
        add("model checksums", bool(ok), detail, required=ok is not None)
        provider = asr["provider"] if asr["ok"] else "cpu"
        for what in DEEP_LOADS:
            ok, detail = _deep_load(what, provider)
            add(f"load {what}", ok, detail)
    return {"ready": all(c["ok"] for c in comps if c["required"]), "components": comps, "encoder": enc,
            "asr_provider": asr, "gpu": gpu, "deep": bool(deep), "checked": _now(),
            "paths": {"root": str(paths.ROOT), "data": str(paths.DATA), "models": str(paths.MODELS),
                      "ffmpeg": paths.tool("ffmpeg"), "ffprobe": paths.tool("ffprobe")}}


def _summary(rep):
    enc, asr = rep.get("encoder") or {}, rep.get("asr_provider") or {}
    head = "READY" if rep.get("ready") else "NOT READY"
    lines = [f"reel check: {head} - encoder {enc.get('actual') or 'unusable'}, asr {asr.get('provider') or '?'}"
             + ("" if asr.get("ok") else " (unavailable)")]
    for c in rep.get("components") or []:
        mark = "ok  " if c["ok"] else ("FAIL" if c.get("required", True) else "warn")
        lines.append(f"  {mark} {c['name']:<28} {c['detail']}")
    if not rep.get("ready"):
        lines.append("not ready: fix the FAIL lines (reel-pipeline/setup.sh installs tools, packages and models)")
    return "\n".join(lines)


@contextlib.contextmanager
def _stdout_to_stderr():
    """While the checks import native libraries, anything they write to fd 1 goes to stderr:
    stdout carries only the JSON report."""
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="reel check", description="Readiness of this host for the reel production "
                                 "line. JSON report on stdout, summary on stderr; exit 0 ready, 1 not ready.")
    ap.add_argument("--json", action="store_true", help="(accepted for scripts; the JSON report is always printed)")
    ap.add_argument("--encoder", choices=ENCODERS, default="auto", help="encoder profile to check (default auto)")
    ap.add_argument("--require-gpu", action="store_true", help="the profile requires NVENC (no CPU fallback)")
    ap.add_argument("--asr-provider", choices=("cpu", "cuda"), help="default: REELKIT_ASR_PROVIDER or cpu")
    ap.add_argument("--deep", action="store_true", help="also verify model checksums and load every model once")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        with _stdout_to_stderr():
            rep = check(a.encoder, a.require_gpu, a.asr_provider, a.deep)
    except Exception as e:
        rep = {"ready": False, "components": [{"name": "readiness", "ok": False, "required": True,
                                               "detail": f"the check itself failed: {type(e).__name__}: {e}"}],
               "encoder": None, "asr_provider": None, "gpu": None, "deep": bool(a.deep), "checked": _now()}
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str), flush=True)
    print(_summary(rep), file=sys.stderr, flush=True)
    return 0 if rep["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
