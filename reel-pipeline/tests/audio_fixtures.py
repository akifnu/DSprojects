"""Synthetic audio for the sample-audio tests: speech-like voice with a matching
transcript, a drum/bass/pad music bed at a known BPM (optionally ducked under the
voice), and library SFX placed at known times."""
import wave

import numpy as np
from scipy import signal

from reelkit import sfx

SR = 48000


def _formant_vowel(dur, f0a, f0b, formants, rng):
    t = np.arange(int(dur * SR)) / SR
    f0 = f0a + (f0b - f0a) * t / dur
    phase = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros_like(t)
    for k in range(1, int(4200 / max(f0a, f0b))):
        fk = k * (f0a + f0b) / 2
        amp = sum(a * np.exp(-0.5 * ((fk - F) / bw) ** 2) for F, bw, a in formants) + 0.02
        out += amp / k ** 0.3 * np.sin(k * phase + rng.uniform(0, 6.28))
    env = np.clip(t / 0.015, 0, 1) * np.clip((dur - t) / 0.03, 0, 1)
    return out * env


VOWELS = [[(700, 90, 1.0), (1200, 110, 0.6), (2600, 150, 0.25)],
          [(300, 70, 1.0), (2300, 120, 0.5), (3000, 150, 0.3)],
          [(500, 80, 1.0), (900, 100, 0.7), (2500, 140, 0.2)],
          [(400, 80, 1.0), (1900, 110, 0.6), (2600, 150, 0.3)]]


def speech(duration, rng, start=0.4):
    """(signal, transcript). Sentences of words separated by short gaps; some words
    begin with an /s/ fricative."""
    y = np.zeros(int(duration * SR))
    words, sents, t = [], [], start
    while t < duration - 1.5:
        w0 = len(words)
        for _ in range(rng.integers(5, 10)):
            if t > duration - 0.8:
                break
            wd = rng.uniform(0.18, 0.45)
            a = int(t * SR)
            seg = np.zeros(int(wd * SR))
            pos = 0
            if rng.random() < 0.3:                     # fricative /s/
                fd = int(rng.uniform(0.06, 0.1) * SR)
                sos = signal.butter(4, [4000, 8500], "bp", fs=SR, output="sos")
                fr = signal.sosfilt(sos, rng.standard_normal(fd)) * np.hanning(fd) * 0.25
                seg[:fd] += fr
                pos = int(fd * 0.8)
            n_syl = rng.integers(1, 3)
            syl_len = (len(seg) - pos) // n_syl
            for s in range(n_syl):
                f0 = rng.uniform(105, 165)
                v = _formant_vowel(syl_len / SR, f0, f0 * rng.uniform(0.85, 1.1), VOWELS[rng.integers(0, 4)], rng)
                seg[pos + s * syl_len: pos + s * syl_len + len(v)] += v
            y[a:a + len(seg)] += seg[:len(y) - a]
            words.append({"w": f"w{len(words)}", "s": round(t, 3), "e": round(t + wd, 3)})
            t += wd + rng.uniform(0.03, 0.14)
        if len(words) > w0:
            words[-1]["w"] += "."
            sents.append({"id": len(sents), "s": words[w0]["s"], "e": words[-1]["e"], "w0": w0, "w1": len(words) - 1,
                          "text": " ".join(w["w"] for w in words[w0:])})
        t += rng.uniform(0.45, 0.9)
    return y, {"lang": "en", "timing": "word", "duration": duration, "words": words, "sentences": sents}


def music(duration, bpm, rng):
    """Kick on beats, hat on eighths, snare on 2 & 4, bass and a pad. Returns (signal, beat_times)."""
    n = int(duration * SR)
    y = np.zeros(n)
    p = 60 / bpm
    beats = np.arange(0.25, duration - 0.3, p)
    tk = np.arange(int(0.3 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(50 + 100 * np.exp(-tk / 0.03)) / SR) * np.exp(-tk / 0.08)
    th = np.arange(int(0.03 * SR)) / SR
    sos = signal.butter(4, 7000, "hp", fs=SR, output="sos")
    hat = signal.sosfilt(sos, rng.standard_normal(len(th))) * np.exp(-th / 0.008) * 0.35
    ts = np.arange(int(0.18 * SR)) / SR
    snare = (signal.sosfilt(signal.butter(2, [1500, 6000], "bp", fs=SR, output="sos"), rng.standard_normal(len(ts)))
             * 0.5 + 0.4 * np.sin(2 * np.pi * 190 * ts)) * np.exp(-ts / 0.05)
    for i, b in enumerate(beats):
        a = int(b * SR)
        y[a:a + len(kick)] += kick[:n - a]
        h = int((b + p / 2) * SR)
        for c in (a, h):
            y[c:c + len(hat)] += hat[:max(0, n - c)]
        if i % 2 == 1:
            y[a:a + len(snare)] += snare[:n - a] * 0.6
        tb = np.arange(int(p * SR * 0.9)) / SR
        note = 55 * 2 ** (rng.integers(0, 5) / 12)
        bass = np.sin(2 * np.pi * note * tb) * np.clip(tb / 0.02, 0, 1) * np.exp(-tb / (p * 0.8)) * 0.4
        y[a:a + len(bass)] += bass[:n - a]
    t = np.arange(n) / SR
    pad = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in (220, 277.2, 329.6, 440)) * 0.04
    return y + pad, beats


def place_sfx(n, kinds, rng, duration, min_gap=1.4, start=1.0, end_margin=2.0):
    """Library SFX at random, non-overlapping times. Returns (signal, truth list)."""
    y = np.zeros(n)
    truth = []
    t = start
    for kind in kinds:
        v = int(rng.integers(0, sfx.VARIANTS[kind]))
        st, _ = sfx.read_wav(sfx.get(kind, v))
        mono = st.mean(0)
        t += rng.uniform(0.2, 0.8)
        if t + len(mono) / SR > duration - end_margin:
            break
        a = int(t * SR)
        y[a:a + len(mono)] += mono
        anchor = t + sfx.peak_offset(kind, v) if kind in ("whoosh", "swish") else (
            t + len(mono) / SR if kind == "riser" else t)
        truth.append({"kind": kind, "class": sfx.CLASS_OF[kind], "t": round(t, 3), "anchor": round(anchor, 3),
                      "dur": len(mono) / SR})
        t += len(mono) / SR + min_gap
    return y, truth


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) + 1e-12


def mix(path, duration, rng, bpm=None, music_db=-14.0, duck_db=0.0, kinds=(), sfx_db=(-6, 0), with_speech=True):
    """Write a 48 kHz stereo WAV; return truth dict."""
    n = int(duration * SR)
    sp, tr = speech(duration, rng) if with_speech else (np.zeros(n), {"words": [], "sentences": [], "lang": None})
    sp_mask = np.zeros(n, bool)
    for w in tr["words"]:
        sp_mask[int(w["s"] * SR):int(w["e"] * SR)] = True
    if with_speech:
        sp *= 10 ** (-20 / 20) / rms(sp[sp_mask])
    y = sp.copy()
    beats = []
    mu = np.zeros(n)
    if bpm:
        mu, beats = music(duration, bpm, rng)
        ref = rms(sp[sp_mask]) if with_speech else 10 ** (-20 / 20)
        mu *= ref * 10 ** (music_db / 20) / rms(mu)
        if duck_db:
            env = np.zeros(n)
            for w in tr["words"]:
                env[int((w["s"] - 0.05) * SR):int((w["e"] + 0.05) * SR)] = 1
            k = int(0.08 * SR)
            env = np.convolve(env, np.ones(k) / k, "same")
            mu *= 10 ** (-duck_db * np.clip(env, 0, 1) / 20)
        y += mu
    fx, truth = place_sfx(n, list(kinds), rng, duration)
    if len(truth):
        # each effect scaled to a level relative to the voice
        out = np.zeros(n)
        ref = rms(sp[sp_mask]) if with_speech else 10 ** (-20 / 20)
        for tt in truth:
            a, b = int(tt["t"] * SR), int((tt["t"] + tt["dur"]) * SR)
            seg = fx[a:b]
            loud = seg[np.abs(seg) > 0.05 * np.abs(seg).max()]
            g = ref * 10 ** (rng.uniform(*sfx_db) / 20) / rms(loud)
            out[a:b] = seg * g
        y += out
    peak = np.abs(y).max()
    if peak > 0.9:
        y *= 0.9 / peak
        # keep the truth's relative levels: everything scaled together
    st = np.stack([y, y])
    pcm = np.clip(np.round(st.T * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    gap_level = None
    if bpm and with_speech:
        gaps = ~sp_mask
        ext = np.zeros(n, bool)
        for w in tr["words"]:
            ext[int((w["s"] - 0.05) * SR):int((w["e"] + 0.05) * SR)] = True
        gm = ~ext
        gap_level = 20 * np.log10(rms(mu[gm]) / rms((sp + mu)[ext]))
    return {"transcript": tr, "beats": beats, "sfx": truth, "gap_level_db": gap_level, "bpm": bpm}
