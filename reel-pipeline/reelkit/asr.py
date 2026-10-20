"""Speech → words with times → sentences with ids.

English:   NVIDIA Parakeet TDT 0.6B v2 (sherpa-onnx, int8): word timestamps, punctuation,
           ~0.15× real time on 4 cores (1 h of audio ≈ 9 min). timing = "word".
Other:     Whisper large-v3-turbo (sherpa-onnx). Audio is cut into ~6 s chunks at quiet
           points; Silero VAD skips music-only chunks and bounds each chunk's speech; the
           words are spread over that span by length, so only the phrase bounds are measured
           (timing = "phrase"; every word carries its phrase index "p"). sherpa decodes Whisper's byte-level tokens one by
           one and loses multi-byte characters (Devanagari, Arabic…), so a token table
           that spells every token's bytes as hex is used and the UTF-8 text rebuilt.

Long audio is split at the quietest point near every 20 s so no word is cut; chunks are
decoded BATCH streams at a time, so memory stays bounded however long the source.

provider: the sherpa-onnx execution provider ("cpu" | "cuda"; cuda needs a CUDA build of
sherpa-onnx), from the argument, else REELKIT_ASR_PROVIDER, else cpu. Every transcript
carries fingerprint(lang, provider) — model files, language, provider, timing mode,
decoding and sentence configuration, sherpa-onnx version — and a cached transcript
whose fingerprint is not current, or that is unreadable, is transcribed again.
"""
import base64
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import ff, fsutil
from .paths import DATA, MODELS

SR = 16000
PARAKEET = MODELS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
WHISPER = MODELS / "sherpa-onnx-whisper-turbo"
VAD = MODELS / "silero_vad.onnx"
PROVIDERS = ("cpu", "cuda")
TRANSCRIPT_VERSION = 2      # bump when word assembly or the transcript layout changes
BATCH = 16                  # chunk streams decoded at once
PARAKEET_CHUNK = (20.0, 4.0)                 # split_points(target, search)
WHISPER_CHUNK = (6.0, 1.5)
WHISPER_MIN_SPEECH = 0.3                     # s of VAD speech for a chunk to be decoded
WHISPER_PAD = (12.0, 29.0)                   # pad chunks to ≥ 12 s (and 2× their length), cap at 29 s
VAD_CFG = {"threshold": 0.2, "min_silence": 0.2, "min_speech": 0.15, "max_speech": 60}
SENTENCES = {"pause": 0.7, "max_words": 28}
DECODE = {"parakeet": {"method": "greedy_search", "chunk": PARAKEET_CHUNK, "batch": BATCH},
          "whisper": {"method": "greedy_search", "chunk": WHISPER_CHUNK, "batch": BATCH, "tokens": "hex",
                      "min_speech": WHISPER_MIN_SPEECH, "pad": WHISPER_PAD, "vad": VAD_CFG}}
PHRASE_NOTE = "word times are spread over each phrase's speech span by word length; only phrase bounds are measured"
WHISPER_LANGS = frozenset(
    "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi "
    "ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd "
    "gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue".split())
_cache = {}


def _threads():
    return max(1, os.cpu_count() or 4)


def check_lang(lang):
    """lang itself if transcription supports it ('en', 'auto' or a Whisper code); ValueError otherwise."""
    if lang != "auto" and lang not in WHISPER_LANGS:
        raise ValueError(f"unsupported transcription language {lang!r}: 'en', 'auto' or a Whisper language code")
    return lang


def resolve_provider(provider=None):
    p = str(provider or os.environ.get("REELKIT_ASR_PROVIDER") or "cpu").strip().lower()
    if p not in PROVIDERS:
        raise ValueError(f"ASR provider {p!r}: one of {', '.join(PROVIDERS)}")
    return p


# ------------------------------------------------------------------ fingerprint

def _whisper_prefix():
    f = next(WHISPER.glob("*-encoder.int8.onnx"), None) if WHISPER.is_dir() else None
    return f.name.split("-encoder")[0] if f else None


def _model_files(lang):
    """(model dir name, model files) behind a language."""
    if lang == "en":
        return PARAKEET.name, [PARAKEET / f for f in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx",
                                                      "tokens.txt")]
    prefix = _whisper_prefix() or "?"
    return WHISPER.name, [WHISPER / f"{prefix}-{f}" for f in ("encoder.int8.onnx", "decoder.int8.onnx",
                                                              "tokens.txt")] + [VAD]


@lru_cache(maxsize=16)
def _content_id(path, size, mtime_ns):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _file_id(f):
    """Name and size of a model file (token tables, small, by content too)."""
    try:
        st = os.stat(f)
    except OSError:
        return [f.name, None]
    return [f.name, st.st_size] + ([_content_id(str(f), st.st_size, st.st_mtime_ns)] if f.suffix == ".txt" else [])


@lru_cache(maxsize=1)
def _sherpa_version():
    from importlib import metadata
    for name in ("sherpa-onnx", "sherpa_onnx"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def fingerprint(lang="en", provider=None):
    """What a transcript in `lang` made by `provider` depends on, as
    '<model dir>/<lang>/<provider>/<timing>/<hash>'."""
    lang, provider = check_lang(lang), resolve_provider(provider)
    engine, files = _model_files(lang)
    timing = "word" if lang == "en" else "phrase"
    h = fsutil.sha256_json({"v": TRANSCRIPT_VERSION, "lang": lang, "provider": provider, "timing": timing,
                            "engine": engine, "models": [_file_id(f) for f in files], "sherpa": _sherpa_version(),
                            "decode": DECODE["parakeet" if lang == "en" else "whisper"], "sentences": SENTENCES})
    return f"{engine}/{lang}/{provider}/{timing}/{h[:16]}"


def current(fp, lang="en", provider=None):
    """Whether a transcript fingerprint is up to date. An explicit provider must be the one that
    made it; with none asked for, a transcript any supported provider made with the current
    models and configuration is reused (a CPU-side caller does not redo a GPU transcript)."""
    if not isinstance(fp, str) or fp.count("/") != 4:
        return False
    made_by = fp.split("/")[2]
    if provider is None:
        return made_by in PROVIDERS and fp == fingerprint(lang, made_by)
    return fp == fingerprint(lang, provider)


# ------------------------------------------------------------------ models

def _need(*files):
    for f in files:
        if not Path(f).is_file():
            raise FileNotFoundError(f"ASR model file missing: {f} (run reel-pipeline/setup.sh)")


def split_points(a, target=20.0, search=4.0):
    """Chunk boundaries (samples) at the quietest 50 ms near every `target` seconds."""
    hop = SR // 20
    n = len(a) // hop
    if n == 0:
        return [0, len(a)]
    rms = np.sqrt(np.mean(a[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    cuts, t = [0], target
    while t < len(a) / SR - target / 3:
        lo, hi = int((t - search) * 20), min(n, int((t + search) * 20))
        k = lo + int(np.argmin(rms[lo:hi]))
        cuts.append(k * hop)
        t = k / 20 + target
    return cuts + [len(a)]


def _parakeet(provider):
    key = ("parakeet", provider)
    if key not in _cache:
        import sherpa_onnx
        files = [PARAKEET / f for f in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")]
        _need(*files)
        _cache[key] = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(files[0]), decoder=str(files[1]), joiner=str(files[2]), tokens=str(files[3]),
            model_type="nemo_transducer", num_threads=_threads(), decoding_method=DECODE["parakeet"]["method"],
            provider=provider)
    return _cache[key]


def _parakeet_words(a, provider):
    rec = _parakeet(provider)
    cuts = split_points(a, *PARAKEET_CHUNK)
    spans = list(zip(cuts, cuts[1:]))
    words = []
    for i in range(0, len(spans), BATCH):            # a bounded number of streams alive at once
        part = spans[i:i + BATCH]
        streams = []
        for lo, hi in part:
            s = rec.create_stream()
            s.accept_waveform(SR, a[lo:hi])
            streams.append(s)
        rec.decode_streams(streams)
        for (lo, _), s in zip(part, streams):
            r, t0 = s.result, lo / SR
            durs = list(getattr(r, "durations", []) or [])
            for k, (tok, ts) in enumerate(zip(r.tokens, r.timestamps)):
                end = t0 + ts + (durs[k] if k < len(durs) and durs[k] > 0 else 0.08)
                if tok.startswith(("▁", " ")) or not words or words[-1].get("_closed"):
                    words.append({"w": tok.lstrip("▁ "), "s": round(t0 + ts, 3), "e": round(end, 3)})
                else:
                    words[-1]["w"] += tok
                    words[-1]["e"] = round(end, 3)
            if words:
                words[-1]["_closed"] = True  # never glue across chunks
        del streams
    words = [{k: v for k, v in w.items() if k != "_closed"} for w in words if w["w"]]
    for a_, b in zip(words, words[1:]):   # no overlaps; a word ends by the time the next starts
        a_["e"] = round(min(a_["e"], b["s"]), 3)
    return words


def _hex_tokens(src):
    """The hex-spelled token table, beside the model or, when the model dir is read-only, in
    data/asr; rebuilt whenever its content is not exactly what `src` gives."""
    lines = []
    for line in src.read_text(encoding="utf8").splitlines():
        tok, idx = line.rsplit(" ", 1)
        try:
            raw = base64.b64decode(tok, validate=True)
        except Exception:
            raw = tok.encode()
        lines.append(f"{base64.b64encode(raw.hex().encode()).decode()} {idx}")
    text = "\n".join(lines) + "\n"
    tag = hashlib.sha256(text.encode()).hexdigest()[:12]
    for out in (src.with_name(src.stem + "-hex.txt"), DATA / "asr" / f"{src.stem}-hex.{tag}.txt"):
        try:
            if out.is_file() and out.read_text(encoding="utf8") == text:
                return out
            out.parent.mkdir(parents=True, exist_ok=True)
            fsutil.atomic_write_text(out, text)
            return out
        except OSError:
            continue
    raise OSError(f"cannot write the hex token table for {src}")


def _whisper(lang, provider):
    key = ("whisper", lang, provider)
    if key not in _cache:
        import sherpa_onnx
        prefix = _whisper_prefix()
        if not prefix:
            raise FileNotFoundError(f"Whisper model missing in {WHISPER} (run reel-pipeline/setup.sh)")
        enc, dec, tok = (WHISPER / f"{prefix}-{f}" for f in ("encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"))
        _need(enc, dec, tok, VAD)
        _cache[key] = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(enc), decoder=str(dec), tokens=str(_hex_tokens(tok)), language=lang if lang != "auto" else "",
            task="transcribe", num_threads=_threads(), decoding_method=DECODE["whisper"]["method"], provider=provider)
    return _cache[key]


def _whisper_text(result):
    raw = "".join(t.strip() for t in result.tokens)
    try:
        return bytes.fromhex(raw).decode("utf8", "replace").strip()
    except ValueError:
        return result.text.strip()


def vad_segments(a, threshold=VAD_CFG["threshold"]):
    """Silero VAD speech segments (s, e) in seconds. A low threshold keeps speech
    that sits under a music bed."""
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(VAD)
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = VAD_CFG["min_silence"]
    cfg.silero_vad.min_speech_duration = VAD_CFG["min_speech"]
    cfg.silero_vad.max_speech_duration = VAD_CFG["max_speech"]
    cfg.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=len(a) / SR + 5)
    w = cfg.silero_vad.window_size
    for i in range(0, len(a) - w + 1, w):
        vad.accept_waveform(a[i:i + w])
    vad.flush()
    segs = []
    while not vad.empty():
        seg = vad.front
        segs.append((seg.start / SR, (seg.start + len(seg.samples)) / SR))
        vad.pop()
    return segs


def _whisper_words(a, lang, provider):
    """Chunks of ~6 s cut at quiet points; chunks the VAD finds no speech in are
    skipped (Whisper invents words over pure music); each chunk's words are spread
    over the span the VAD marks as speech. Chunks are padded with silence to twice
    their length (≥ 12 s) because sherpa caps Whisper's output at 6 tokens per
    second of input, too few for scripts whose words take many byte tokens."""
    rec = _whisper(lang, provider)
    speech = vad_segments(a)
    cuts = split_points(a, *WHISPER_CHUNK)
    spans = []
    for lo, hi in zip(cuts, cuts[1:]):
        s, e = lo / SR, hi / SR
        inside = [(max(s, p), min(e, q)) for p, q in speech if q > s and p < e]
        if sum(q - p for p, q in inside) >= WHISPER_MIN_SPEECH:
            spans.append((lo, hi, inside[0][0], inside[-1][1]))
    words, detected = [], None
    for i in range(0, len(spans), BATCH):            # a bounded number of streams alive at once
        part = spans[i:i + BATCH]
        streams = []
        for lo, hi, _, _ in part:
            pad = np.zeros(max(int(WHISPER_PAD[0] * SR), 2 * (hi - lo)) - (hi - lo), np.float32)
            st = rec.create_stream()
            st.accept_waveform(SR, np.concatenate([a[lo:hi], pad])[: int(WHISPER_PAD[1] * SR)])
            streams.append(st)
        rec.decode_streams(streams)
        for n, ((_, _, s, e), st) in enumerate(zip(part, streams), start=i):
            text = _whisper_text(st.result)
            detected = detected or getattr(st.result, "lang", None)
            toks = text.split()
            if not toks:
                continue
            weights = np.array([len(t) + 1 for t in toks], float)
            edges = s + (e - s) * np.concatenate([[0], np.cumsum(weights) / weights.sum()])
            words += [{"w": t, "s": round(edges[k], 3), "e": round(edges[k + 1], 3), "p": n}
                      for k, t in enumerate(toks)]
        del streams
    return words, detected


END = re.compile(r"[.?!।؟]$")


def sentences(words, pause=0.7, max_words=28):
    """Sentence index: break on end punctuation, a pause, a phrase boundary (Whisper
    chunks), or overlong runs (at a comma)."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        nxt = words[i + 1] if i + 1 < len(words) else None
        brk = nxt is None or END.search(w["w"]) or nxt["s"] - w["e"] > pause or nxt.get("p") != w.get("p")
        if not brk and len(cur) >= max_words and w["w"].endswith(","):
            brk = True
        if brk:
            out.append(cur)
            cur = []
    return [{"id": n, "s": words[ix[0]]["s"], "e": words[ix[-1]]["e"], "w0": ix[0], "w1": ix[-1],
             "text": " ".join(words[k]["w"] for k in ix)} for n, ix in enumerate(out)]


# ------------------------------------------------------------------ transcripts

def _transcript(lang, provider, words, duration, detected):
    timing = "word" if lang == "en" else "phrase"
    out = {"lang": lang if lang != "auto" else detected, "timing": timing, "duration": round(duration, 3),
           "words": words, "sentences": sentences(words, **SENTENCES), "fingerprint": fingerprint(lang, provider),
           "provider": provider}
    if timing == "phrase":
        out["timing_note"] = PHRASE_NOTE
    return out


def empty(lang="en", duration=None, provider=None):
    """The transcript of a file without an audio stream."""
    lang, provider = check_lang(lang), resolve_provider(provider)
    return _transcript(lang, provider, [], float(duration or 0), "en" if lang == "en" else None)


def _run(path, lang, provider):
    a = ff.audio(path, SR)
    if len(a) < SR // 10:
        ff.log(f"transcribing: {os.path.basename(str(path))} has no audio to speak of")
        return _transcript(lang, provider, [], len(a) / SR, "en" if lang == "en" else None)
    ff.log(f"transcribing {len(a) / SR:.0f}s of audio ({'Parakeet' if lang == 'en' else 'Whisper turbo'}, {provider})")
    if lang == "en":
        words, detected = _parakeet_words(a, provider), "en"
    else:
        words, detected = _whisper_words(a, lang, provider)
    out = _transcript(lang, provider, words, len(a) / SR, detected)
    ff.log(f"transcript: {len(words)} words, {len(out['sentences'])} sentences")
    return out


def _cached(cache, lang, provider, quiet=False):
    """The cached transcript if it is readable and current, else None (with the reason logged)."""
    if not os.path.exists(cache):
        return None
    tr = fsutil.read_json(cache)
    name = os.path.basename(str(cache))
    if not (isinstance(tr, dict) and isinstance(tr.get("words"), list) and isinstance(tr.get("sentences"), list)):
        if not quiet:
            ff.log(f"{name}: cached transcript unreadable; transcribing again")
        return None
    if not current(tr.get("fingerprint"), lang, provider):
        if not quiet:
            why = (f"made as {tr['fingerprint']}, now {fingerprint(lang, provider)}" if tr.get("fingerprint")
                   else "predates ASR fingerprints")
            ff.log(f"{name}: cached transcript is stale ({why}); transcribing again")
        return None
    return tr


def transcribe(path, lang="en", cache=None, provider=None):
    """Words + sentences for an audio/video file. lang: 'en' (Parakeet) or a Whisper
    language code ('hi', 'ur', 'es', …, or 'auto'). provider: "cpu" | "cuda" (default
    REELKIT_ASR_PROVIDER, else cpu). With `cache` (a JSON path) the result is kept there
    and reused while current (see current()); one process transcribes, others wait for it."""
    lang = check_lang(lang)
    use = resolve_provider(provider)
    if not cache:
        return _run(path, lang, use)
    tr = _cached(cache, lang, provider)
    if tr is not None:
        return tr
    with fsutil.file_lock(cache):
        tr = _cached(cache, lang, provider, quiet=True)
        if tr is None:
            tr = _run(path, lang, use)
            fsutil.atomic_write_text(cache, json.dumps(tr, ensure_ascii=False))
    return tr


def compact(tr):
    """One line per sentence, for planning: 'S0012 01:23.4-01:27.9 text'."""
    def ts(t):
        return f"{int(t // 60):02d}:{t % 60:04.1f}"
    return "\n".join(f"S{s['id']:04d} {ts(s['s'])}-{ts(s['e'])} {s['text']}" for s in tr["sentences"])
