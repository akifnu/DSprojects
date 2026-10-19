"""Default sound-effect library, synthesised in numpy (no samples, no licences).

Kinds (several variants each): whoosh, swish, pop, click, impact, riser, ding, glitch.
Every file is 48 kHz stereo 16-bit, peak-normalised to -1 dBTP (4x oversampled true
peak), DC-free, with raised-cosine edges so it never clicks in or out.

    get("whoosh")                 → path of library/sfx/whoosh_0.wav
    get("riser", duration=1.6)    → a riser of exactly 1.6 s (cached in data/sfx/)
    generate()                    → (re)writes library/sfx/*.wav + manifest.json

The sample-audio analyser reports SFX in classes whoosh / pop / impact / riser / ding /
glitch (swish → whoosh, click → pop); `CLASS_OF` maps library kinds to those classes.
"""
import json
import wave
from pathlib import Path

import numpy as np
from scipy import signal

from .paths import DATA, LIBRARY

SR = 48000
DIR = LIBRARY / "sfx"
CACHE = DATA / "sfx"
CLASS_OF = {"whoosh": "whoosh", "swish": "whoosh", "pop": "pop", "click": "pop", "impact": "impact",
            "riser": "riser", "ding": "ding", "glitch": "glitch"}
VARIANTS = {"whoosh": 3, "swish": 3, "pop": 3, "click": 3, "impact": 3, "riser": 2, "ding": 3, "glitch": 2}
DEFAULT_DUR = {"whoosh": (0.55, 0.7, 0.45), "swish": (0.22, 0.28, 0.18), "pop": (0.14, 0.12, 0.16),
               "click": (0.05, 0.04, 0.06), "impact": (0.85, 0.7, 0.8), "riser": (1.3, 1.0),
               "ding": (1.0, 0.9, 0.8), "glitch": (0.32, 0.26)}
PEAK_DBTP = -1.0


# ------------------------------------------------------------------ building blocks

def _t(dur):
    return np.arange(int(round(dur * SR))) / SR


def _edges(x, fade_in=0.002, fade_out=0.008):
    """Raised-cosine fades at both ends (no click at start/end)."""
    n = x.shape[-1]
    a, b = max(2, int(fade_in * SR)), max(2, int(fade_out * SR))
    w = np.ones(n)
    w[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    w[n - b:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, b))
    return x * w


def _dc(x):
    sos = signal.butter(2, 20, "hp", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def _sweep_filter(noise, centers, q=1.4):
    """Time-varying band-pass of noise: Gaussian (in log-frequency) bands following
    `centers` (Hz, one value per output sample), applied by STFT masking."""
    nper = 1024
    f, t, Z = signal.stft(noise, SR, nperseg=nper, noverlap=nper * 3 // 4, boundary="even")
    idx = np.clip((t * SR).astype(int), 0, len(centers) - 1)
    fc = centers[idx]
    lf = np.log2(np.maximum(f, 1.0))[:, None]
    mask = np.exp(-0.5 * ((lf - np.log2(fc)[None, :]) / (1.0 / q)) ** 2)
    _, y = signal.istft(Z * mask, SR, nperseg=nper, noverlap=nper * 3 // 4, boundary=True)
    return y[:len(noise)]


def _pan(mono, pos):
    """Constant-power stereo placement; pos in -1..1 (scalar or per-sample)."""
    th = (np.asarray(pos) + 1) * np.pi / 4
    return np.stack([mono * np.cos(th), mono * np.sin(th)])


def _env(n, attack, release, peak_at=None, curve=2.0):
    """Asymmetric envelope rising over `attack` s and falling over `release` s."""
    t = np.arange(n) / SR
    pk = peak_at if peak_at is not None else attack
    up = np.clip(t / max(pk, 1e-4), 0, 1) ** curve
    down = np.exp(-np.maximum(t - pk, 0) / max(release, 1e-4) * 3.0)
    return up * down


# ------------------------------------------------------------------ the sounds

def whoosh(dur=0.55, variant=0, rng=None):
    rng = rng or np.random.default_rng(100 + variant)
    n = int(dur * SR)
    t = _t(dur)
    u = t / dur
    lo, hi = [(380, 2600), (300, 3400), (500, 2200)][variant % 3]
    peak = [0.55, 0.6, 0.5][variant % 3]
    # centre frequency rises to the pass-by then falls (doppler-like)
    centers = np.where(u < peak, lo * (hi / lo) ** (u / peak), hi * (lo * 0.7 / hi) ** ((u - peak) / (1 - peak)))
    noise = rng.standard_normal(n)
    body = _sweep_filter(noise, centers, q=1.3)
    air = _sweep_filter(rng.standard_normal(n), centers * 2.2, q=0.9) * 0.35
    env = np.exp(-0.5 * ((u - peak) / (0.22 if variant != 2 else 0.18)) ** 2)
    mono = (body + air) * env
    pos = np.tanh((u - peak) * 4) * (0.7 if variant % 2 == 0 else -0.7)        # moves across the stereo field
    return _pan(mono, pos)


def swish(dur=0.22, variant=0, rng=None):
    rng = rng or np.random.default_rng(200 + variant)
    n = int(dur * SR)
    u = _t(dur) / dur
    lo, hi = [(1500, 7000), (1200, 6000), (2000, 8000)][variant % 3]
    peak = 0.4
    centers = np.where(u < peak, lo * (hi / lo) ** (u / peak), hi * (lo / hi) ** ((u - peak) / (1 - peak)))
    mono = _sweep_filter(rng.standard_normal(n), centers, q=1.1)
    mono *= np.exp(-0.5 * ((u - peak) / 0.16) ** 2)
    return _pan(mono, np.linspace(-0.5, 0.5, n) * (1 if variant % 2 == 0 else -1))


def pop(dur=0.14, variant=0, rng=None):
    rng = rng or np.random.default_rng(300 + variant)
    t = _t(dur)
    f0, f1 = [(900, 220), (1300, 320), (700, 160)][variant % 3]
    tau = 0.018
    freq = f1 + (f0 - f1) * np.exp(-t / tau)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    tone = np.sin(phase) * np.exp(-t / 0.035)
    click = rng.standard_normal(len(t)) * np.exp(-t / 0.0015) * 0.35
    sos = signal.butter(2, 6000, "lp", fs=SR, output="sos")
    mono = tone + signal.sosfilt(sos, click)
    return _pan(mono, 0.0)


def click(dur=0.05, variant=0, rng=None):
    rng = rng or np.random.default_rng(400 + variant)
    t = _t(dur)
    sos = signal.butter(2, [1500, 9000][0] if variant == 1 else 2500, "hp", fs=SR, output="sos")
    burst = signal.sosfilt(sos, rng.standard_normal(len(t))) * np.exp(-t / [0.0025, 0.0035, 0.002][variant % 3])
    ring = np.sin(2 * np.pi * [3200, 2400, 4100][variant % 3] * t) * np.exp(-t / 0.004) * 0.4
    return _pan(burst + ring, [0.0, 0.15, -0.15][variant % 3])


def impact(dur=0.85, variant=0, rng=None):
    rng = rng or np.random.default_rng(500 + variant)
    t = _t(dur)
    f0, f1 = [(120, 42), (95, 38), (140, 50)][variant % 3]
    freq = f1 + (f0 - f1) * np.exp(-t / 0.06)
    sub = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / [0.28, 0.22, 0.25][variant % 3])
    sos = signal.butter(2, 1800, "lp", fs=SR, output="sos")
    crack = signal.sosfilt(sos, rng.standard_normal(len(t))) * np.exp(-t / 0.03) * 0.8
    tail = signal.sosfilt(signal.butter(2, 400, "lp", fs=SR, output="sos"),
                          rng.standard_normal(len(t))) * np.exp(-t / 0.18) * 0.5
    mono = np.tanh(1.6 * (sub + crack + tail)) / np.tanh(1.6)           # gentle saturation, glues the layers
    st = _pan(mono, 0.0)
    # slight stereo width on the tail only
    st[1] = 0.85 * st[1] + 0.15 * np.roll(st[1], int(0.007 * SR))
    return st


def riser(dur=1.3, variant=0, rng=None):
    rng = rng or np.random.default_rng(600 + variant)
    n = int(dur * SR)
    t = _t(dur)
    u = t / dur
    f0, f1 = [(180, 1800), (120, 1400)][variant % 2]
    freq = f0 * (f1 / f0) ** (u ** 1.5)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    tone = sum(np.sin(k * phase) / k for k in range(1, 6)) * 0.5                 # band-limited saw-ish
    tone += 0.5 * np.sin(phase * 1.003)                                        # detune: width, movement
    noise = _sweep_filter(rng.standard_normal(n), 800 * (8 ** u), q=0.8) * 0.6
    env = 10 ** ((-30 + 30 * u ** 1.6) / 20)                                   # -30 dB → 0 dB crescendo
    mono = (tone + noise) * env
    return _pan(mono, np.sin(2 * np.pi * 1.5 * t * u) * 0.3)


def ding(dur=1.0, variant=0, rng=None):
    t = _t(dur)
    f = [1320, 1760, 1047][variant % 3]
    ratios = [1.0, 2.0, 2.76, 5.40, 8.93]
    amps = [1.0, 0.55, 0.4, 0.22, 0.12]
    decays = [0.55, 0.35, 0.25, 0.12, 0.07]
    mono = sum(a * np.sin(2 * np.pi * f * r * t + 0.3 * k) * np.exp(-t / (d * dur))
               for k, (r, a, d) in enumerate(zip(ratios, amps, decays)))
    mono *= np.clip(t / 0.002, 0, 1)
    return _pan(mono, [0.0, 0.2, -0.2][variant % 3])


def glitch(dur=0.32, variant=0, rng=None):
    rng = rng or np.random.default_rng(800 + variant)
    n = int(dur * SR)
    out = np.zeros(n)
    pos = 0
    while pos < n:
        seg = int(rng.uniform(0.012, 0.045) * SR)
        kind = rng.integers(0, 3)
        tt = np.arange(seg) / SR
        if kind == 0:
            chunk = np.sign(np.sin(2 * np.pi * rng.uniform(300, 2500) * tt)) * 0.6       # square burst
        elif kind == 1:
            chunk = rng.standard_normal(seg) * 0.5                                       # static
            chunk = np.round(chunk * 6) / 6                                              # bit-crush
        else:
            chunk = np.zeros(seg)                                                        # gap
        out[pos:pos + seg] = chunk[:n - pos]
        pos += seg + int(rng.uniform(0.0, 0.01) * SR)
    sos = signal.butter(2, [150, 9000], "bp", fs=SR, output="sos")
    out = signal.sosfilt(sos, out)
    return _pan(out, np.where(rng.random(n) > 0.5, 0.3, -0.3) * 0 + 0.0)


SYNTH = {"whoosh": whoosh, "swish": swish, "pop": pop, "click": click, "impact": impact, "riser": riser,
         "ding": ding, "glitch": glitch}


# ------------------------------------------------------------------ mastering + io

def true_peak_db(x):
    """4x-oversampled true peak (dBTP) of a (channels, n) array."""
    up = signal.resample_poly(np.atleast_2d(x), 4, 1, axis=-1)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


def master(st, peak_db=PEAK_DBTP):
    st = _dc(np.atleast_2d(st))
    st = _edges(st)
    gain = 10 ** ((peak_db - true_peak_db(st)) / 20)
    st = st * gain
    # 16-bit quantisation can nudge the peak; leave 0.05 dB margin
    return st * 10 ** (-0.05 / 20)


def synth(kind, variant=0, duration=None):
    dur = duration or DEFAULT_DUR[kind][variant % len(DEFAULT_DUR[kind])]
    return master(SYNTH[kind](dur, variant))


def write_wav(path, st):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(st.T * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(st.shape[0])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        a = np.frombuffer(w.readframes(n), "<i2").astype(np.float64) / 32768
    return a.reshape(-1, ch).T, sr


def measure(st):
    """Peak (dBTP), DC, edge level, spectral centroid (Hz), duration (s) of a sound."""
    mono = st.mean(0)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1 / SR)
    edge = int(0.001 * SR)
    return {"peak_dbtp": round(float(true_peak_db(st)), 2), "dc": float(np.abs(st.mean(1)).max()),
            "edge": float(max(np.abs(st[:, :edge]).max(), np.abs(st[:, -edge:]).max())),
            "centroid": round(float((spec * freqs).sum() / (spec.sum() + 1e-12)), 1),
            "dur": round(st.shape[1] / SR, 3)}


def generate():
    """Write every variant to library/sfx/ and a manifest describing them."""
    DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"sample_rate": SR, "channels": 2, "peak_dbtp": PEAK_DBTP, "kinds": {}}
    for kind, nv in VARIANTS.items():
        manifest["kinds"][kind] = []
        for v in range(nv):
            st = synth(kind, v)
            name = f"{kind}_{v}.wav"
            write_wav(DIR / name, st)
            back, _ = read_wav(DIR / name)
            manifest["kinds"][kind].append({"file": name, "variant": v, "class": CLASS_OF[kind], **measure(back)})
    (DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def get(kind, variant=0, duration=None):
    """Path of a library sound; with `duration`, a fresh render of that length (cached)."""
    if kind not in SYNTH:
        raise KeyError(f"unknown sfx kind {kind!r}; one of {sorted(SYNTH)}")
    variant %= VARIANTS[kind]
    if duration is None:
        p = DIR / f"{kind}_{variant}.wav"
        if not p.exists():
            generate()
        return p
    p = CACHE / f"{kind}_{variant}_{int(round(duration * 1000))}ms.wav"
    if not p.exists():
        write_wav(p, synth(kind, variant, duration))
    return p
