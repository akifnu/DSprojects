"""Sound of a sample reel: loudness, speech, music bed (tempo, beats, ducking), sound
effects (what, when, how loud, tied to cuts / captions), speech pacing.

    analyze_audio(path, info, transcript, cuts=(), caption_starts=()) -> dict

returns SPEC.md's "audio" section plus "pacing" {speech_wpm, max_pause, cut_on} and
"structure" {first_word_at} extras.

Everything runs on a 22.05 kHz mono mixdown with a 5.8 ms hop.
  Beats:  band-weighted spectral-flux onset envelope → tempo by autocorrelation with a
          log-normal prior around 120 BPM and harmonic support → dynamic-programming beat
          tracker (Ellis 2007) → beats refined to onset peaks → BPM from a fit over beats.
  SFX:    one detector per class, each keyed to what that class does and speech/music do
          not (a single falling partial for pops, a band-moving noise bump for whooshes, a
          slow-decaying sub onset for impacts, a steady crescendo that stops dead for risers,
          a struck inharmonic partial for dings, chopped broadband for glitches, a <25 ms
          transient for clicks). Candidates that recur on the beat grid are the music's drums
          and are dropped; noise bumps that sit inside a word without moving are fricatives.
Event times: "t" is the start; "anchor" is the moment editors line up with a cut — the
peak for whooshes, the end for risers, the start for everything else.
"""
import math

import numpy as np
from scipy import ndimage, signal

from . import ff

SR = 22050
NFFT = 1024
HOP = 128
FPS = SR / HOP
CLASSES = ("whoosh", "pop", "impact", "riser", "ding", "glitch")


# ---------------------------------------------------------------- spectral front end

def _stft(x):
    win = np.hanning(NFFT)
    pad = np.pad(x, (NFFT // 2, NFFT // 2))
    n = 1 + (len(pad) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = pad[idx] * win
    return np.abs(np.fft.rfft(frames, axis=1)).T          # (bins, frames); frame k centred at k*HOP


FREQS = np.fft.rfftfreq(NFFT, 1 / SR)


def _band(P, lo, hi):
    m = (FREQS >= lo) & (FREQS < hi)
    return P[m].sum(0) + 1e-10


def _db(p):
    return 10 * np.log10(np.maximum(p, 1e-10))


def _smooth(v, frames):
    if frames <= 1:
        return v
    k = np.hanning(frames + 2)[1:-1]
    return np.convolve(v, k / k.sum(), "same")


# ---------------------------------------------------------------- beats / tempo

def onset_envelope(S):
    """Spectral flux of log-magnitude, weighted towards the bands drums live in."""
    L = np.log1p(1000 * S / (S.max() + 1e-9))
    flux = np.maximum(0, np.diff(L, axis=1, prepend=L[:, :1]))
    w = np.ones_like(FREQS)
    w[FREQS < 180] = 3.0
    w[FREQS > 5000] = 2.0
    o = (flux * w[:, None]).sum(0)
    o = o - ndimage.uniform_filter1d(o, int(FPS * 0.5))          # remove slow trend
    return np.maximum(o, 0)


def tempo(o, lo=60, hi=200):
    """(bpm, clarity) from the onset envelope's autocorrelation with a log-normal
    prior around 120 BPM and harmonic support (beat-level beats beat sub-divisions)."""
    o = o - o.mean()
    n = len(o)
    if n < FPS * 4:
        return None, 0.0
    f = np.fft.rfft(o, 2 * n)
    ac = np.fft.irfft(np.abs(f) ** 2)[:n]
    ac /= ac[0] + 1e-12
    lags = np.arange(int(FPS * 60 / hi), int(FPS * 60 / lo) + 1)

    def at(l):
        return np.interp(l, np.arange(len(ac)), ac)
    bpms = 60 * FPS / lags
    prior = np.exp(-0.5 * (np.log2(bpms / 120) / 1.0) ** 2)
    score = np.array([at(l) + 0.5 * at(2 * l) + 0.25 * at(4 * l) for l in lags]) * prior
    k = int(np.argmax(score))
    if 0 < k < len(score) - 1:      # parabolic refinement of the peak lag
        a, b, c = score[k - 1], score[k], score[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    else:
        d = 0.0
    lag = lags[k] + d
    return 60 * FPS / lag, float(max(0.0, at(lag)))


def track_beats(o, bpm, tightness=100.0):
    """Ellis dynamic-programming beat tracker. Returns beat frame positions (float)."""
    p = FPS * 60 / bpm
    n = len(o)
    o = o / (o.std() + 1e-9)
    score = o.copy()
    back = -np.ones(n, int)
    lo, hi = int(round(p / 2)), int(round(2 * p))
    taus = np.arange(lo, hi + 1)
    pen = -tightness * np.log(taus / p) ** 2
    for t in range(hi, n):
        cand = score[t - taus] + pen
        j = int(np.argmax(cand))
        score[t] = o[t] + cand[j]
        back[t] = t - taus[j]
    t = n - 1 - int(np.argmax(score[::-1][:int(p) + 1]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t] if back[t] >= 0 else -1
    beats = np.array(beats[::-1], float)
    # refine to the local onset peak (±2 frames) with parabolic interpolation
    raw = o
    out = []
    for b in beats:
        b = int(b)
        lo_, hi_ = max(1, b - 2), min(n - 2, b + 2)
        k = lo_ + int(np.argmax(raw[lo_:hi_ + 1]))
        a, c, e = raw[k - 1], raw[k], raw[k + 1]
        den = a - 2 * c + e
        out.append(k + (0.5 * (a - e) / den if den != 0 else 0.0))
    return np.array(out)


ONSET_LAG = 0.0      # seconds; flux peaks land on the attack with the centred frames used here


def beats_of(x, S=None):
    S = _stft(x) if S is None else S
    o = onset_envelope(S)
    bpm, clarity = tempo(o)
    if bpm is None:
        return {"bpm": None, "beats": [], "clarity": 0.0}
    fr = track_beats(o, bpm)
    times = fr / FPS - ONSET_LAG
    # trim weak beats at the ends (fade in/out, silence)
    strength = np.interp(fr, np.arange(len(o)), o)
    keep = strength > 0.15 * np.median(strength[strength > 0]) if (strength > 0).any() else np.ones(len(fr), bool)
    times = times[keep]
    if len(times) >= 4:
        idx = np.arange(len(times))
        slope = np.polyfit(idx, times, 1)[0]
        bpm = 60.0 / slope
    return {"bpm": round(float(bpm), 2), "beats": [round(float(t), 3) for t in times], "clarity": round(clarity, 3)}


# ---------------------------------------------------------------- speech helpers

def speech_spans(words, pad=0.05, merge=0.3):
    spans = []
    for w in words:
        a, b = w["s"] - pad, w["e"] + pad
        if spans and a - spans[-1][1] < merge:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return spans


def _mask(spans, n_frames, fps=FPS):
    m = np.zeros(n_frames, bool)
    for a, b in spans:
        m[max(0, int(a * fps)):max(0, int(math.ceil(b * fps)))] = True
    return m


def pacing(transcript):
    words = transcript.get("words") or []
    sents = transcript.get("sentences") or []
    if not words:
        return {"speech_wpm": None, "max_pause": None}, None
    span = max(1e-3, words[-1]["e"] - words[0]["s"])
    gaps = []
    for s in sents:
        for i in range(s["w0"], s["w1"]):
            gaps.append(max(0.0, words[i + 1]["s"] - words[i]["e"]))
    max_pause = float(np.percentile(gaps, 95)) if gaps else 0.0
    return {"speech_wpm": round(len(words) / span * 60, 1), "max_pause": round(max_pause, 3)}, words[0]["s"]


# ---------------------------------------------------------------- SFX detectors

def _peaks(v, height, distance):
    pk, _ = signal.find_peaks(v, height=height, distance=max(1, int(distance)))
    return pk


def _floored_db(p):
    """dB with a floor 35 dB under the band's median, so silent stretches don't read as -100 dB."""
    floor = max(1e-10, float(np.median(p)) * 10 ** (-3.5), float(p.max()) * 1e-9)
    return 10 * np.log10(p + floor)


class _Front:
    """Per-frame features shared by the detectors (22.05 kHz, 5.8 ms hop)."""

    def __init__(self, x):
        self.x = x
        self.S = _stft(x)
        P = self.S ** 2
        self.P = P
        self.n = P.shape[1]
        self.sub = _floored_db(_band(P, 30, 75))
        self.low = _floored_db(_band(P, 95, 700))
        self.mid = _floored_db(_band(P, 160, 1600))
        self.hf = _floored_db(_band(P, 1000, 10500))
        self.full = _floored_db(_band(P, 300, 10500))
        m = (FREQS >= 300) & (FREQS < 10500)
        Pm = P[m] + 1e-12
        self.centroid = (Pm * FREQS[m, None]).sum(0) / Pm.sum(0)
        # noisiness: peak-to-mean power within ±0.75 octave of the centroid (noise ≈ 6 dB, tones ≫ 15 dB)
        lf = np.log2(FREQS[m])[:, None]
        near = np.abs(lf - np.log2(self.centroid)[None, :]) < 0.75
        Pn = np.where(near, Pm, np.nan)
        self.peaky = 10 * np.log10(np.nanmax(Pn, 0) / (np.nanmean(Pn, 0) + 1e-12))
        # voicing: cepstral peak prominence in the 80-320 Hz pitch range (speech, sung notes)
        ceps = np.fft.irfft(np.log(P[:int(4000 / (SR / NFFT))] + 1e-10), n=NFFT, axis=0)
        q = ceps[int(SR / 320):int(SR / 80)]
        self.voiced = q.max(0) - np.median(q, axis=0)

    def bg(self, v, sec=2.0, q=20):
        size = max(3, int(sec * FPS))
        return ndimage.percentile_filter(v, q, size=size, mode="nearest")


def _onset_spec(P, k, back=4):
    """Energy that is new at frame k (vs `back` frames earlier): what just started sounding."""
    return np.maximum(P[:, k] - P[:, max(0, k - back)], 0)[:, None]


def _peak_partial(P, k, lo, hi):
    """Strongest spectral line in [lo, hi) at frame k; its frequency, its level above the
    median of ±1 octave around it, and the level of its harmonic neighbours (f/2, 2f, 3f/2)."""
    m = (FREQS >= lo) & (FREQS < hi)
    col = P[m, k]
    j = int(np.argmax(col))
    fq = FREQS[m][j]
    pk = col[max(0, j - 1):j + 2].sum()
    oct_ = (FREQS > fq / 2) & (FREQS < fq * 2)
    prom = 10 * np.log10(pk / (np.median(P[oct_, k]) * 3 + 1e-12))

    def at(f):
        if f < 30 or f > SR / 2 - 50:
            return 0.0
        i = int(round(f / (SR / NFFT)))
        return P[max(0, i - 2):i + 3, k].max()
    harm = max(at(fq / 2), at(fq * 2), at(fq * 1.5), at(fq / 3 * 2 if fq > 300 else 0))
    return fq, prom, 10 * np.log10((harm + 1e-12) / (P[max(0, int(round(fq / (SR / NFFT))) - 1):
                                                       int(round(fq / (SR / NFFT))) + 2, k].max() + 1e-12))


def _onset_rise(v, k, pre=0.12, post=0.035):
    a, b = max(0, k - int(pre * FPS)), max(1, k - 1)
    return float(v[k:k + int(post * FPS) + 1].max() - v[a:b].min()) if b > a else 0.0


def _transients(x):
    """Clicks: the >2 kHz envelope jumps ≥ 12 dB over its last 20 ms (≥ 10 dB within 3 ms —
    no vowel or fricative rises that fast) and is back down within 25 ms."""
    sos = signal.butter(4, 2000, "hp", fs=SR, output="sos")
    y = signal.sosfilt(sos, x)
    hop = int(0.001 * SR)
    p = ndimage.maximum_filter1d(y ** 2, hop)[::hop]
    e = _floored_db(p)
    out = []
    for i in range(20, len(e) - 30):
        pre = np.median(e[i - 20:i - 2])
        if e[i] - pre > 12 and e[i] >= e[i - 1] and e[i] >= e[i + 1] and e[i] - e[i - 3] > 10:
            after = e[i + 1:i + 26]
            if (after < e[i] - 12).any() and (after[-5:] < e[i] - 10).all():
                if not out or i - out[-1] > 40:
                    out.append(i - 3 + int(np.argmax(np.diff(e[i - 3:i + 1]) > 6)))
    return [{"t": o / 1000.0, "dur": 0.03, "sub": "click", "type": "pop"} for o in out]


def _pops(f):
    """A hard onset in 160-1600 Hz that becomes one partial (≥ 18 dB over its octave, harmonic
    neighbours ≥ 15 dB down) whose pitch falls ≥ 15 % between 10 and 70 ms, stays above
    150 Hz, and dies (-15 dB) within 150 ms. A fast chirp smears across bins in its first
    frames, so its shape is judged 45 ms in."""
    out = []
    rise = f.mid - np.concatenate([[f.mid[0]] * 4, f.mid[:-4]])
    d = lambda sec: int(round(sec * FPS))
    for k in _peaks(rise, 12, FPS * 0.06):
        pk = k + int(np.argmax(f.mid[k:k + 4]))
        if pk + d(0.15) >= f.n:
            continue
        new = lambda sec: _onset_spec(f.P, pk + d(sec), back=d(sec) + 4)
        _, prom, harm = _peak_partial(new(0.045), 0, 120, 1600)
        if prom < 18 or harm > -15:
            continue
        fa = _peak_partial(new(0.01), 0, 120, 1600)[0]
        fb = _peak_partial(new(0.07), 0, 120, 1600)[0]
        if not (fb < 0.85 * fa and fb >= 150):
            continue
        tail = f.mid[pk:pk + d(0.15)]
        if tail.max() - tail[-3:].mean() < 15:
            continue
        on = k - 3 + int(np.argmax(rise[max(0, k - 3):k + 1] > 6)) if k >= 3 else k
        out.append({"t": on / FPS, "dur": 0.15, "sub": "pop", "type": "pop"})
    return out


def _impacts(f):
    """Sub-75 Hz onset (≥ 15 dB) that rings: within 20 dB of its peak 0.3 s later."""
    out = []
    v = f.sub
    rise = v - ndimage.minimum_filter1d(np.concatenate([[v[0]] * 30, v[:-30]]), 30)
    for k in _peaks(rise, 15, FPS * 0.3):
        pk = k + int(np.argmax(v[k:k + int(0.06 * FPS) + 1]))
        later = v[pk + int(0.28 * FPS):pk + int(0.32 * FPS)]
        body = slice(pk + int(0.08 * FPS), pk + int(0.14 * FPS))
        sub_dom = f.sub[body].mean() > f.low[body].mean() - 10      # a voice's sub leakage is ≥ 20 dB under it
        if len(later) and later.mean() > v[pk] - 20 and v[pk] > np.median(v) + 15 and sub_dom:
            on = k - 5 + int(np.argmax(np.diff(v[max(0, k - 5):k + 1]) > 3)) if k >= 5 else k
            out.append({"t": on / FPS, "dur": 0.7, "sub": "impact", "type": "impact"})
    return out


def _whooshes(f, words):
    """A smooth noise bump over the bed (≥ 10 dB, 0.08-1.6 s, ≥ 30 ms rise), noise-like
    (peak-to-mean ≤ 13 dB), centroid travelling ≥ 0.6 octave; still bumps inside words are
    fricatives."""
    v = _smooth(f.hf, int(0.03 * FPS))
    rel = v - f.bg(v, 2.5, 20)
    out = []
    for k in _peaks(rel, 10, FPS * 0.15):
        top = v[k]
        a = k
        while a > 0 and v[a - 1] > top - 15 and k - a < 2 * FPS:
            a -= 1
        b = k
        while b < f.n - 1 and v[b + 1] > top - 15 and b - k < 2 * FPS:
            b += 1
        dur = (b - a) / FPS
        if not (0.08 <= dur <= 1.6) or (k - a) / FPS < 0.03:
            continue
        loud = np.arange(a, b + 1)[v[a:b + 1] > top - 10]
        if np.median(f.peaky[loud]) > 13 or np.mean(f.voiced[loud] > 0.25) > 0.3:
            continue
        cen = _smooth(f.centroid, 5)[loud]
        sweep = float(np.log2(cen.max() / max(1.0, cen.min())))
        t0, t1 = a / FPS, b / FPS
        in_word = any(w["s"] - 0.03 <= t0 and t1 <= w["e"] + 0.05 for w in words)
        if sweep < 0.6 and (in_word or dur < 0.25):
            continue
        out.append({"t": t0, "dur": round(dur, 3), "peak": k / FPS, "sub": "swish" if dur < 0.3 else "whoosh",
                    "type": "whoosh"})
    return out


def _running_fit(y, L):
    """Slope (per frame) and r² of a least-squares line over the L frames ending at each frame."""
    n = len(y)
    x = np.arange(n, dtype=float)
    c = lambda v: np.concatenate([[0.0], np.cumsum(v)])
    Sx, Sy, Sxx, Sxy, Syy = c(x), c(y), c(x * x), c(x * y), c(y * y)
    end = np.arange(L, n + 1)
    sx, sy = Sx[end] - Sx[end - L], Sy[end] - Sy[end - L]
    sxx, sxy, syy = Sxx[end] - Sxx[end - L], Sxy[end] - Sxy[end - L], Syy[end] - Syy[end - L]
    vx, vy, cxy = sxx - sx * sx / L, syy - sy * sy / L, sxy - sx * sy / L
    slope = cxy / np.maximum(vx, 1e-9)
    r2 = cxy ** 2 / np.maximum(vx * vy, 1e-9)
    out_s, out_r = np.zeros(n), np.zeros(n)
    out_s[end - 1], out_r[end - 1] = slope, r2
    return out_s, out_r


RIDGE_HOP = 4                 # frames (23 ms)
RIDGE_BPO = 24                # log-frequency bins per octave


def _rising_ridges(f, lo=150, hi=4000, cost=6.0, max_up=3, stall=0.15, leak=0.97):
    """Dynamic programme over the log-frequency spectrogram's excess over its background
    (per-band 25th percentile over 3 s), 23 ms steps: the best path that stays or climbs
    0-3 bins (of 1/24 octave) a step and climbs at least every `stall` seconds. Returns per
    analysis frame the strongest path ending there that has climbed ≥ 1 octave over ≥ 0.4 s,
    as (score, start frame). Pads (flat) and speech harmonics (stall, wobble) cannot."""
    nb = int(np.log2(hi / lo) * RIDGE_BPO)
    edges = lo * 2 ** (np.arange(nb + 1) / RIDGE_BPO)
    M = np.zeros((nb, len(FREQS)))
    for i in range(nb):
        sel = (FREQS >= edges[i]) & (FREQS < edges[i + 1])
        if not sel.any():
            sel = np.abs(FREQS - (edges[i] * edges[i + 1]) ** 0.5) == np.abs(FREQS - (edges[i] * edges[i + 1]) ** 0.5).min()
        M[i, sel] = 1.0
    T = f.n // RIDGE_HOP
    Pd = f.P[:, :T * RIDGE_HOP].reshape(len(FREQS), T, RIDGE_HOP).mean(2)
    L = 10 * np.log10(M @ Pd + 1e-10)
    bgL = ndimage.percentile_filter(L, 25, size=(1, int(3 * FPS / RIDGE_HOP)), mode="nearest")
    E = np.clip(L - bgL, 0, 40) - cost
    R = np.zeros(nb)
    sb = np.arange(nb)
    st = np.zeros(nb, int)
    up = np.zeros(nb, int)
    stall_n = max(1, int(round(stall * FPS / RIDGE_HOP)))
    min_len = int(0.4 * FPS / RIDGE_HOP)
    score = np.zeros(f.n)
    start = -np.ones(f.n, int)
    idx = np.arange(nb)
    for t in range(T):
        cand = np.full((max_up + 1, nb), -np.inf)
        cand[0] = R
        for u in range(1, max_up + 1):
            cand[u, u:] = R[:nb - u]
        j = np.argmax(cand, axis=0)
        best = cand[j, idx]
        src = np.clip(idx - j, 0, nb - 1)
        last_up = np.where(j > 0, t, up[src])
        cont = (best > 0) & (t - last_up <= stall_n)
        R = E[:, t] + np.where(cont, leak * best, 0)
        sb = np.where(cont, sb[src], idx)
        st = np.where(cont, st[src], t)
        up = np.where(cont, last_up, t)
        ok = (idx - sb >= RIDGE_BPO) & (t - st >= min_len) & (R > 60)
        if ok.any():
            k = int(np.argmax(np.where(ok, R, -np.inf)))
            fr = t * RIDGE_HOP
            score[fr:fr + RIDGE_HOP] = R[k]
            start[fr:fr + RIDGE_HOP] = st[k] * RIDGE_HOP
    return score, start


def _risers(f):
    """A steady crescendo over ≥ 0.4-0.6 s, by either sign: a pitched riser's rising ridge
    (see _rising_ridges), or — for noise risers — the level rising ≥ 12 dB/s (r² ≥ 0.8)
    with a rising centroid. Anchored at its loudest point (where it lands, the end)."""
    v = _smooth(f.full, int(0.03 * FPS))
    lc = _smooth(np.log2(np.maximum(f.centroid, 50)), int(0.05 * FPS))
    rscore, rstart = _rising_ridges(f)
    cands = []
    k = 0
    while k < f.n:                       # each run of ridge frames is one riser; it ends where the run ends
        if rscore[k] > 0:
            j = k
            while j + 1 < f.n and rscore[j + 1] > 0:
                j += 1
            cands.append((int(rstart[k:j + 1].min()), j))
            k = j + 1
        else:
            k += 1
    for L in (0.6, 0.9, 1.2, 1.6):
        n = int(L * FPS)
        s_, r2 = _running_fit(v, n)
        cs, _ = _running_fit(lc, n)
        ok = (s_ * FPS >= 12) & (r2 >= 0.8) & (cs * FPS > 0.3)
        for k in np.where(ok & ~np.concatenate([[False], ok[:-1]]))[0]:
            j = k
            while j + 1 < f.n and ok[j + 1]:
                j += 1
            cands.append((k - n, j))
    out = []
    for a, j in sorted(cands, key=lambda c: c[1]):
        end = j + int(np.argmax(v[j:j + int(0.1 * FPS)]))
        if any(abs(end / FPS - o["end"]) < 0.3 for o in out):
            continue
        start = max(0, a)
        out.append({"t": start / FPS, "end": (end + 1) / FPS, "dur": round((end + 1 - start) / FPS, 3),
                    "sub": "riser", "type": "riser"})
    return out


def _dings(f):
    """A struck line above 700 Hz standing ≥ 25 dB over its octave, no harmonic comb below it,
    hard onset without a matching low-band onset, steady pitch, still within 20 dB at 0.25 s."""
    out = []
    hm = (FREQS >= 700) & (FREQS < 7000)
    v = _floored_db(f.P[hm].max(0))
    rise = v - np.concatenate([[v[0]] * 4, v[:-4]])
    for k in _peaks(rise, 15, FPS * 0.15):
        pk = k + int(np.argmax(v[k:k + 4]))
        fq, prom, harm = _peak_partial(_onset_spec(f.P, pk, 6), 0, 700, 7000)
        if prom < 22:
            continue
        if _onset_rise(f.low, k) > _onset_rise(v, k) - 10:
            continue                                  # the low band starts with it: a voice / instrument
        fqs = [_peak_partial(f.P, pk + d, 700, 7000)[0] for d in (5, 15, 30)]
        if max(abs(x - fq) for x in fqs) > 0.03 * fq:
            continue
        later = v[pk + int(0.22 * FPS):pk + int(0.28 * FPS)]
        if not len(later) or later.mean() < v[pk] - 20:
            continue
        on = k - 3 + int(np.argmax(np.diff(v[max(0, k - 3):k + 1]) > 6)) if k >= 3 else k
        out.append({"t": on / FPS, "dur": 0.8, "sub": "ding", "type": "ding"})
    return out


def _glitches(f):
    """Chopped broadband: 0.15-0.7 s standing ≥ 12 dB over the bed and, on a 3 ms envelope,
    switching hard (≥ 12 dB between neighbouring 3 ms windows) ≥ 6 times at ≥ 15 per second
    (voices never switch that hard; a drum groove switches once or twice per hit run)."""
    v = f.hf
    rel = v - f.bg(v, 2.5, 20)
    on = rel > 12
    sos = signal.butter(2, [150, 9000], "bp", fs=SR, output="sos")
    y = signal.sosfilt(sos, f.x)
    w = int(0.003 * SR)
    m = len(y) // w
    env = _floored_db((y[:m * w].reshape(m, w) ** 2).mean(1))
    out, k = [], 0
    while k < f.n:
        if on[k]:
            j = k
            while j < f.n - 1 and on[j + 1:j + 1 + int(0.03 * FPS)].any():
                j += 1
            dur = (j - k) / FPS
            e = env[int(k / FPS / 0.003):int(j / FPS / 0.003) + 1]
            steps = int((np.abs(np.diff(e)) >= 12).sum())
            if 0.15 <= dur <= 0.7 and steps >= 6 and steps / max(dur, 1e-3) >= 15:
                out.append({"t": k / FPS, "dur": round(dur, 3), "sub": "glitch", "type": "glitch"})
            k = j + 1
        else:
            k += 1
    return out


def _profile(f, t):
    k = int(t * FPS)
    return np.array([f.sub[k:k + 4].max(), f.low[k:k + 4].max(), f.mid[k:k + 4].max(), f.hf[k:k + 4].max()])


def _drop_drums(f, cands, beats, period):
    """With a music bed: candidates of a kind that keeps landing on the (half-)beat grid
    with the same band profile are the drums, not effects."""
    if period is None or not len(beats) or len(cands) < 4:
        return cands
    grid = np.sort(np.concatenate([beats, beats + period / 2]))
    on = [c for c in cands if np.abs(grid - c["t"]).min() < 0.025]
    if len(on) < 4:
        return cands
    prof = np.median([_profile(f, c["t"]) for c in on], axis=0)
    return [c for c in cands if not (np.abs(grid - c["t"]).min() < 0.025 and
                                     np.abs(_profile(f, c["t"]) - prof).max() < 6)]


def detect_sfx(x, words, beats_info, f=None):
    f = f or _Front(x)
    music = beats_info.get("music")
    period = 60 / beats_info["bpm"] if beats_info.get("bpm") and music else None
    beats = np.array(beats_info.get("beats") or []) if music else np.array([])
    found = {"glitch": _glitches(f), "riser": _risers(f), "whoosh": _whooshes(f, words), "impact": _impacts(f),
             "ding": _dings(f), "pop": _pops(f), "click": _transients(x)}
    for kind in ("impact", "pop", "click", "ding"):
        found[kind] = _drop_drums(f, found[kind], beats, period)
    events, spans = [], []

    def inside(e, pad=0.03):
        a, b = e["t"] - pad, e.get("end", e["t"] + e["dur"])
        return any(a < sb and sa < b for sa, sb in spans)
    # long, distinctive sounds claim their span first; transients inside them are their parts
    for kind in ("riser", "glitch", "impact", "ding", "whoosh", "pop", "click"):
        for e in sorted(found[kind], key=lambda e: e["t"]):
            if kind in ("pop", "click", "ding", "impact", "whoosh") and inside(e):
                continue
            events.append(e)
            spans.append((e["t"], e.get("end", e["t"] + e["dur"])))
    for e in events:
        e["anchor"] = round(float(e.get("peak", e.get("end", e["t"])) if e["type"] in ("whoosh", "riser") else e["t"]), 3)
        e["t"] = round(float(e["t"]), 3)
        if "peak" in e:
            e["peak"] = round(float(e["peak"]), 3)
        if "end" in e:
            e["end"] = round(float(e["end"]), 3)
    return sorted(events, key=lambda e: e["t"]), f


def _level_db(x, e, ref_rms):
    a = int(e["t"] * SR)
    b = int((e.get("end") or (e["t"] + max(0.03, e["dur"]))) * SR)
    seg = x[a:max(a + 1, b)]
    if e["type"] == "whoosh" and "peak" in e:
        c = int(e["peak"] * SR)
        seg = x[max(0, c - int(0.05 * SR)):c + int(0.05 * SR)]
    if e["type"] in ("pop", "impact", "ding"):
        seg = x[a:a + int(0.05 * SR)]
    rms = math.sqrt(float(np.mean(seg ** 2))) + 1e-9
    return round(20 * math.log10(rms / ref_rms), 1)


# ---------------------------------------------------------------- music bed

def music_bed(x, f, words, beats_info, sfx):
    spans = speech_spans(words)
    sp = _mask(spans, f.n)
    busy = np.zeros(f.n, bool)
    for e in sfx:
        busy[int(e["t"] * FPS):int((e.get("end") or e["t"] + e["dur"]) * FPS) + 1] = True
    gap = ~sp & ~busy
    # drop the first/last 0.25 s of each gap (word edges, breaths)
    gap = ndimage.binary_erosion(gap, iterations=int(0.12 * FPS)) if gap.any() else gap
    P = f.P
    def rms_db(mask):
        if mask.sum() < 3:
            return None
        return float(_db(P[:, mask].sum(0).mean()))
    speech_db, gap_db = rms_db(sp & ~busy), rms_db(gap)
    gap_level = (gap_db - speech_db) if (gap_db is not None and speech_db is not None) else None
    clarity = beats_info.get("clarity", 0.0)
    # a bed is periodic and not far below the voice; room tone is neither
    present = bool(beats_info.get("bpm")) and clarity > 0.12 and (gap_level is None or gap_level > -32)
    duck = None
    if present and gap.sum() > 3 and (sp & ~busy).sum() > 3:
        lowm = (FREQS >= 30) & (FREQS < 90)                     # below the voice: music only
        eg = P[lowm][:, gap].sum(0).mean()
        es = P[lowm][:, sp & ~busy].sum(0).mean()
        duck = round(float(np.clip(10 * np.log10((eg + 1e-12) / (es + 1e-12)), 0, 30)), 1)
    return {"present": present, "gap_level_db": round(gap_level, 1) if gap_level is not None else None,
            "bpm": beats_info["bpm"] if present else None, "beats": beats_info["beats"] if present else [],
            "clarity": clarity, "duck_db": duck if duck is not None else 12}


# ---------------------------------------------------------------- top level

def _near(t, events, before=0.15, after=0.15):
    for e in events:
        lo = e["t"] - before
        hi = max(e["anchor"], e["t"]) + after
        if e["type"] == "riser":
            lo, hi = e["end"] - before, e["end"] + after
        if lo <= t <= hi:
            return True
    return False


def analyze_audio(path, info, transcript, cuts=(), caption_starts=()):
    x = ff.audio(path, SR)
    dur = len(x) / SR
    loud = ff.loudness(path)
    words = (transcript or {}).get("words") or []
    sents = (transcript or {}).get("sentences") or []
    f = _Front(x)
    beats_info = beats_of(x, f.S)
    beats_info["music"] = beats_info.get("clarity", 0) > 0.12
    sfx, _ = detect_sfx(x, words, beats_info, f)
    music = music_bed(x, f, words, beats_info, sfx)
    if not music["present"] and beats_info["music"]:
        beats_info["music"] = False      # no bed after all: re-run without drum rejection
        sfx, _ = detect_sfx(x, words, beats_info, f)
    spans = speech_spans(words)
    sp_mask = np.zeros(len(x), bool)
    for a, b in spans:
        sp_mask[max(0, int(a * SR)):int(b * SR)] = True
    ref = math.sqrt(float(np.mean(x[sp_mask] ** 2))) if sp_mask.any() else math.sqrt(float(np.mean(x ** 2))) + 1e-9
    for e in sfx:
        e["level_db"] = _level_db(x, e, ref + 1e-9)
    types = {}
    for e in sfx:
        types[e["type"]] = types.get(e["type"], 0) + 1
    on_cut = [_near(c, sfx) for c in cuts]
    on_cap = [_near(c, sfx) for c in caption_starts]
    coverage = sum(min(b, dur) - max(a, 0) for a, b in spans) / dur if dur else 0.0
    pace, first_word = pacing(transcript or {})
    bounds = []
    for a, b in zip(sents, sents[1:]):
        bounds.append((a["e"], b["s"]))
    on_sentence = [any(lo - 0.25 <= c <= hi + 0.25 for lo, hi in bounds) for c in cuts]
    beat_t = np.array(music["beats"]) if music["present"] else np.array([])
    on_beat = [bool(len(beat_t)) and float(np.abs(beat_t - c).min()) <= 0.06 for c in cuts]
    return {
        "loudness": loud,
        "speech": {"present": bool(words), "coverage": round(coverage, 3), "lang": (transcript or {}).get("lang")},
        "music": music,
        "sfx": {"events": sfx, "on_cut_rate": round(float(np.mean(on_cut)), 3) if cuts else 0.0,
                "on_caption_rate": round(float(np.mean(on_cap)), 3) if caption_starts else 0.0, "types": types},
        "pacing": {**pace, "cut_on": {"sentence": round(float(np.mean(on_sentence)), 3) if cuts else 0.0,
                                      "beat": round(float(np.mean(on_beat)), 3) if cuts else 0.0}},
        "structure": {"first_word_at": round(first_word, 3) if first_word is not None else None},
    }
