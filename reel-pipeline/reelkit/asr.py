"""Speech → words with times → sentences with ids.

English:   NVIDIA Parakeet TDT 0.6B v2 (sherpa-onnx, int8): word timestamps, punctuation,
           ~0.15× real time on 4 cores (1 h of audio ≈ 9 min).
Other:     Whisper large-v3-turbo (sherpa-onnx). Audio is cut into ~6 s chunks at quiet
           points; Silero VAD skips music-only chunks and bounds each chunk's speech; the
           words are spread over that span by length (timing = "phrase"). sherpa decodes Whisper's byte-level tokens one by
           one and loses multi-byte characters (Devanagari, Arabic…), so a token table
           that spells every token's bytes as hex is used and the UTF-8 text rebuilt.

Long audio is split at the quietest point near every 20 s so no word is cut.
"""
import base64
import json
import os
import re

import numpy as np

from . import ff
from .paths import MODELS

SR = 16000
PARAKEET = MODELS / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
WHISPER = MODELS / "sherpa-onnx-whisper-turbo"
VAD = MODELS / "silero_vad.onnx"
_cache = {}


def _threads():
    return max(1, os.cpu_count() or 4)


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


def _parakeet():
    if "parakeet" not in _cache:
        import sherpa_onnx
        _cache["parakeet"] = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(PARAKEET / "encoder.int8.onnx"), decoder=str(PARAKEET / "decoder.int8.onnx"),
            joiner=str(PARAKEET / "joiner.int8.onnx"), tokens=str(PARAKEET / "tokens.txt"),
            model_type="nemo_transducer", num_threads=_threads())
    return _cache["parakeet"]


def _parakeet_words(a):
    rec = _parakeet()
    cuts = split_points(a)
    streams = []
    for lo, hi in zip(cuts, cuts[1:]):
        s = rec.create_stream()
        s.accept_waveform(SR, a[lo:hi])
        streams.append(s)
    rec.decode_streams(streams)
    words = []
    for lo, s in zip(cuts, streams):
        r, t0 = s.result, lo / SR
        durs = list(getattr(r, "durations", []) or [])
        for i, (tok, ts) in enumerate(zip(r.tokens, r.timestamps)):
            end = t0 + ts + (durs[i] if i < len(durs) and durs[i] > 0 else 0.08)
            if tok.startswith(("▁", " ")) or not words or words[-1].get("_closed"):
                words.append({"w": tok.lstrip("▁ "), "s": round(t0 + ts, 3), "e": round(end, 3)})
            else:
                words[-1]["w"] += tok
                words[-1]["e"] = round(end, 3)
        if words:
            words[-1]["_closed"] = True  # never glue across chunks
    words = [{k: v for k, v in w.items() if k != "_closed"} for w in words if w["w"]]
    for a_, b in zip(words, words[1:]):   # no overlaps; a word ends by the time the next starts
        a_["e"] = round(min(a_["e"], b["s"]), 3)
    return words


def _hex_tokens(src):
    out = src.with_name(src.stem + "-hex.txt")
    if not out.exists():
        lines = []
        for line in src.read_text(encoding="utf8").splitlines():
            tok, idx = line.rsplit(" ", 1)
            try:
                raw = base64.b64decode(tok, validate=True)
            except Exception:
                raw = tok.encode()
            lines.append(f"{base64.b64encode(raw.hex().encode()).decode()} {idx}")
        out.write_text("\n".join(lines) + "\n")
    return out


def _whisper(lang):
    key = ("whisper", lang)
    if key not in _cache:
        import sherpa_onnx
        prefix = next(WHISPER.glob("*-encoder.int8.onnx")).name.split("-encoder")[0]
        _cache[key] = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(WHISPER / f"{prefix}-encoder.int8.onnx"), decoder=str(WHISPER / f"{prefix}-decoder.int8.onnx"),
            tokens=str(_hex_tokens(WHISPER / f"{prefix}-tokens.txt")), language=lang if lang != "auto" else "",
            task="transcribe", num_threads=_threads())
    return _cache[key]


def _whisper_text(result):
    raw = "".join(t.strip() for t in result.tokens)
    try:
        return bytes.fromhex(raw).decode("utf8", "replace").strip()
    except ValueError:
        return result.text.strip()


def vad_segments(a, threshold=0.2):
    """Silero VAD speech segments (s, e) in seconds. A low threshold keeps speech
    that sits under a music bed."""
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(VAD)
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = 0.2
    cfg.silero_vad.min_speech_duration = 0.15
    cfg.silero_vad.max_speech_duration = 60
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


def _whisper_words(a, lang):
    """Chunks of ~6 s cut at quiet points; chunks the VAD finds no speech in are
    skipped (Whisper invents words over pure music); each chunk's words are spread
    over the span the VAD marks as speech. Chunks are padded with silence to twice
    their length (≥ 12 s) because sherpa caps Whisper's output at 6 tokens per
    second of input, too few for scripts whose words take many byte tokens."""
    rec = _whisper(lang)
    speech = vad_segments(a)
    cuts = split_points(a, target=6.0, search=1.5)
    jobs = []
    for lo, hi in zip(cuts, cuts[1:]):
        s, e = lo / SR, hi / SR
        inside = [(max(s, p), min(e, q)) for p, q in speech if q > s and p < e]
        if sum(q - p for p, q in inside) < 0.3:
            continue
        pad = np.zeros(max(int(12 * SR), 2 * (hi - lo)) - (hi - lo), np.float32)
        st = rec.create_stream()
        st.accept_waveform(SR, np.concatenate([a[lo:hi], pad])[: 29 * SR])
        jobs.append((inside[0][0], inside[-1][1], st))
    rec.decode_streams([j[2] for j in jobs])
    words, detected = [], None
    for n, (s, e, st) in enumerate(jobs):
        text = _whisper_text(st.result)
        detected = detected or getattr(st.result, "lang", None)
        toks = text.split()
        if not toks:
            continue
        weights = np.array([len(t) + 1 for t in toks], float)
        edges = s + (e - s) * np.concatenate([[0], np.cumsum(weights) / weights.sum()])
        words += [{"w": t, "s": round(edges[i], 3), "e": round(edges[i + 1], 3), "p": n} for i, t in enumerate(toks)]
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


def transcribe(path, lang="en", cache=None):
    """Words + sentences for an audio/video file. lang: 'en' (Parakeet) or a Whisper
    language code ('hi', 'ur', 'es', …, or 'auto')."""
    if cache and os.path.exists(cache):
        return json.loads(open(cache).read())
    a = ff.audio(path, SR)
    ff.log(f"transcribing {len(a) / SR:.0f}s of audio ({'Parakeet' if lang == 'en' else 'Whisper turbo'})")
    if lang == "en":
        words, timing, detected = _parakeet_words(a), "word", "en"
    else:
        words, detected = _whisper_words(a, lang)
        timing = "phrase"
    out = {"lang": lang if lang != "auto" else detected, "timing": timing, "duration": round(len(a) / SR, 3),
           "words": words, "sentences": sentences(words)}
    if cache:
        with open(cache, "w") as f:
            json.dump(out, f, ensure_ascii=False)
    ff.log(f"transcript: {len(words)} words, {len(out['sentences'])} sentences")
    return out


def compact(tr):
    """One line per sentence, for planning: 'S0012 01:23.4-01:27.9 text'."""
    def ts(t):
        return f"{int(t // 60):02d}:{t % 60:04.1f}"
    return "\n".join(f"S{s['id']:04d} {ts(s['s'])}-{ts(s['e'])} {s['text']}" for s in tr["sentences"])
