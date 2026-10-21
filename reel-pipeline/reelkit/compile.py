"""plan + style + sources → one timeline.json per reel (see render.py for the format).

compile_reel(plan, reel, style, out_dir)  one reel: out_dir/timeline.json + deps.json
compile_plan(plan, out_root)              every reel, through one Context
Context(plan, style)                      what every reel shares, each made once: assets,
                                          transcripts, audio envelopes, editing copies for the
                                          union of all reels' runs, each source's colour look
dependencies(tl, base_dir, …)             deps.json: what can change a timeline's rendering
                                          without changing its JSON — asset hashes, editing-copy
                                          identities, transcript fingerprints, style / LUT / font /
                                          SFX / music contents, renderer version

What the style decides (all px values are scaled by output_height / sample.height):
  pacing      shot length: long runs are split at word boundaries into shots near the
              sample's median shot length (a punch / re-frame, audio continuous)
  motion      punch-in framing levels and how often they change; slow pushes (in or out,
              as the sample); quick zoom bursts inside shots, timed on a word onset
  transitions the special transition (zoom, flash, whip, dissolve …) at its rate, on
              jump cuts between runs
  captions    words per caption, chars per line, lines, case, lead, typography;
              a caption never ends on a weak word ("the", "of", "and" …); keyword
              highlights mark the plan's emphasis words, else one strong word per caption
  titles      hook title (if the plan gives title_text) and CTA styling
  grade       3D LUT from the source footage's look to the sample's look
  audio       loudness target, music ducking depth; SFX on cuts / caption appearances
              with the sample's kind, rate, timing against the moment and level vs the voice
The framing follows the speaker (reframe.py) and is clamped so the picture always
covers the canvas.
"""
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from . import ff, fsutil, grade, plan as planlib, reframe, render, source, store

WEAK = {"the", "a", "an", "of", "to", "and", "on", "in", "at", "with", "for", "that", "this", "is", "are", "was",
        "were", "we", "you", "i", "it", "as", "or", "but", "these", "those", "all", "have", "has", "be", "by",
        "from", "our", "your", "my", "his", "her", "their", "so", "if", "then", "than", "into", "about", "just",
        "very", "really", "will", "would", "can", "could", "do", "does", "did", "not", "no", "what", "which", "who",
        "when", "where", "how", "there", "here", "they", "he", "she", "me", "us", "them", "its", "it's", "you're",
        "we're", "i'm", "don't"}
END = (".", "?", "!", "…")

QUALITY = {"final": {"preset": "medium", "crf": 17}, "draft": {"preset": "veryfast", "crf": 20}}
ZOOMS = ("zoom_in", "zoom_out")
VOICE_CHAIN_DB = 3.0      # approx. level change of render.py's voice chain (compressor + make-up) on speech


# ------------------------------------------------------------------ style helpers

def load_style(ref):
    if isinstance(ref, dict):
        return ref
    p = Path(str(ref))
    if not p.exists():
        p = store.get(ref).cache("style.json")
    return json.loads(Path(p).read_text())


PX_KEYS = ("size", "x", "y", "letter_spacing")


def scale_style(st, k):
    """A captions.py style with every px value multiplied by k."""
    st = json.loads(json.dumps(st or {}))
    for key in PX_KEYS:
        if isinstance(st.get(key), (int, float)):
            st[key] = st[key] * k
    if isinstance(st.get("max_width"), (int, float)) and st["max_width"] > 1:
        st["max_width"] *= k
    for sub, keys in (("stroke", ("width",)), ("shadow", ("blur", "dx", "dy")),
                      ("box", ("pad_x", "pad_y", "radius")), ("anim", ("slide_px",))):
        if isinstance(st.get(sub), dict):
            for key in keys:
                if isinstance(st[sub].get(key), (int, float)):
                    st[sub][key] *= k
    hb = (st.get("highlight") or {}).get("box")
    if isinstance(hb, dict):
        for key in ("pad_x", "pad_y", "radius"):
            if isinstance(hb.get(key), (int, float)):
                hb[key] *= k
    return st


def _g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or d.get(p) is None:
            return default
        d = d[p]
    return d


# ------------------------------------------------------------------ captions

def _bare(w):
    return planlib.bare(w)


def chunk_captions(words, cfg, emphasis=()):
    """Timeline-timed words → caption events [{s, e, text, words}]."""
    max_words = int(_g(cfg, "words_per_caption", "max", default=0) or
                    (_g(cfg, "words_per_caption", "median", default=3) + 1))
    median = int(_g(cfg, "words_per_caption", "median", default=max(1, max_words - 1)))
    cpl = int(cfg.get("chars_per_line") or 22)
    lines = int(cfg.get("max_lines") or 2)
    lead = float(cfg.get("lead") or 0.0)
    emph = {_bare(w) for w in emphasis}
    words = [dict(w, w=w["w"].strip()) for w in words if w["w"].strip()]

    def chars(ws):
        return sum(len(w["w"]) for w in ws) + max(0, len(ws) - 1)

    # hard breaks: sentence ends and pauses; inside each stretch, break points chosen by
    # dynamic programming: chunk sizes near the sample's median, never ending on a weak
    # word, no orphans, prefer breaking after commas
    stretches, cur = [], []
    for w in words:
        if cur and (cur[-1]["w"].endswith(END) or w["s"] - cur[-1]["e"] > 0.5):
            stretches.append(cur)
            cur = []
        cur.append(w)
    if cur:
        stretches.append(cur)

    def cost(ch, last):
        n = len(ch)
        c = (n - median) ** 2
        if _bare(ch[-1]["w"]) in WEAK and not last and not ch[-1]["w"].endswith(END):
            c += 40
        if n == 1 and len(ch[0]["w"]) < 6 and not last:
            c += 12
        if ch[-1]["w"].endswith((",", ";", ":")):
            c -= 3
        c += 4 * sum(1 for w in ch[:-1] if w["w"].endswith((",", ";", ":")))   # a comma inside a caption
        return c
    chunks = []
    for st in stretches:
        n = len(st)
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for j in range(1, n + 1):
            for i in range(max(0, j - max_words), j):
                ch = st[i:j]
                if chars(ch) > cpl * lines and j - i > 1:
                    continue
                c = best[i] + cost(ch, j == n)
                if c < best[j]:
                    best[j], back[j] = c, i
        cuts, j = [], n
        while j > 0:
            cuts.append((back[j], j))
            j = back[j]
        chunks += [st[i:j] for i, j in reversed(cuts)]
    events = []
    for n, ch in enumerate(chunks):
        text = " ".join(w["w"] for w in ch).strip()
        text = text.rstrip(".,;:") if not text.endswith(("...", "…")) else text
        if len(text) > cpl and lines > 1:                      # balanced two-line split
            mid, best = len(text) / 2, None
            for j, c in enumerate(text):
                if c == " " and (best is None or abs(j - mid) < abs(best - mid)):
                    best = j
            if best is not None:
                text = text[:best] + "\n" + text[best + 1:]
        s = ch[0]["s"] + lead
        e = ch[-1]["e"] + 0.25
        if n + 1 < len(chunks) and chunks[n + 1][0]["s"] - ch[-1]["e"] < 0.6:
            e = chunks[n + 1][0]["s"] + lead
        events.append({"s": s, "e": e, "text": text,
                       "words": [dict({"w": w["w"], "s": round(w["s"], 3), "e": round(w["e"], 3)},
                                      **({"emph": True} if _bare(w["w"]) in emph else {})) for w in ch]})
    for a, b in zip(events, events[1:]):
        a["e"] = min(a["e"], b["s"])
    out = []
    for ev in events:
        ev["s"], ev["e"] = round(max(0.0, ev["s"]), 3), round(ev["e"], 3)
        if out:
            ev["s"] = max(ev["s"], out[-1]["e"])
        if ev["e"] - ev["s"] >= 0.12:
            out.append(ev)
    return out


# ------------------------------------------------------------------ shots

def split_shots(runs, target, emphasis=()):
    """Runs → shots [(run_index, src_s, src_e)], long runs cut at word boundaries."""
    shots = []
    emph = {_bare(w) for w in emphasis}
    for ri, r in enumerate(runs):
        d = r["e"] - r["s"]
        if d <= 1.6 * target or len(r["words"]) < 4:
            shots.append((ri, r["s"], r["e"]))
            continue
        n = max(1, int(round(d / target)))
        cuts, ws = [], r["words"]
        for j in range(1, n):
            want = r["s"] + d * j / n
            # Stay near the reference pacing, but prefer an important word or a
            # phrase boundary over an arbitrary word at exactly the target time.
            def cost(q):
                gap = max(0.0, ws[q]["s"] - ws[q - 1]["e"])
                cue = (0.45 if _bare(ws[q]["w"]) in emph else 0.0) + \
                      (0.25 if ws[q - 1]["w"].endswith(END + (",", ";", ":")) else 0.0) + min(0.2, gap)
                return abs(ws[q]["s"] - want) - cue
            best = min(range(1, len(ws)), key=cost)
            t = round((ws[best - 1]["e"] + ws[best]["s"]) / 2, 4)
            if (not cuts or t - cuts[-1] > 0.6) and t - r["s"] > 0.6 and r["e"] - t > 0.6:
                cuts.append(t)
        bounds = [r["s"]] + cuts + [r["e"]]
        shots += [(ri, a, b) for a, b in zip(bounds, bounds[1:])]
    return shots


def _every(rate):
    return 0 if not rate or rate <= 0 else max(1, int(round(1 / min(1.0, rate))))


def _moment_scores(events, words, emphasis):
    """Editorial cues supplied by the planner plus sentence and pause boundaries.
    These are timing hints, not a claim to infer emotion from the transcript."""
    emph = {_bare(w) for w in emphasis}
    scores = []
    for ev in events:
        s, e = ev["start"], ev["start"] + ev["dur"]
        inside = [i for i, w in enumerate(words) if s <= w["s"] < e]
        score = 8.0 if any(_bare(words[i]["w"]) in emph for i in inside) else 0.0
        if inside:
            i = inside[0]
            # A sentence cue belongs to its boundary, not every later shot in it.
            if words[i]["s"] - s <= 0.3:
                if i == 0 or words[i - 1]["w"].endswith(END):
                    score += 3.0
                if i and words[i]["s"] - words[i - 1]["e"] >= 0.18:
                    score += 2.0
        scores.append(score)
    return scores


def _select_moments(events, rate, scores, eligible=None, min_spacing=0.8):
    """Use the sample rate as an effect budget, rank by content cues, and spread
    equally strong candidates apart. Never place effects on every Nth shot."""
    candidates = list(range(len(events))) if eligible is None else sorted(eligible)
    if not candidates or not rate or rate <= 0:
        return set()
    count = max(1, int(round(min(1.0, rate) * len(candidates))))
    selected = set()
    while candidates and len(selected) < count:
        def distance(i):
            return min((abs(events[i]["start"] - events[j]["start"]) for j in selected), default=0.0)
        i = max(candidates, key=lambda k: (scores[k], distance(k), -k))
        selected.add(i)
        candidates = [k for k in candidates if k != i and
                      abs(events[k]["start"] - events[i]["start"]) >= min_spacing]
    return selected


def _auto_keywords(events, rate):
    """Keyword highlights mark about one word per caption (at the sample's rate): where
    the plan's emphasis put none in a caption, mark its most substantial word."""
    every = _every(rate)
    for i, ev in enumerate(events):
        ws = ev.get("words") or []
        if not every or i % every or len(ws) < 2 or any(w.get("emph") for w in ws):
            continue
        cands = [w for w in ws if _bare(w["w"]) not in WEAK and len(_bare(w["w"])) > 2]
        if cands:
            max(cands, key=lambda w: len(_bare(w["w"])))["emph"] = True


# ------------------------------------------------------------------ motion

def _burst(z, mv, start, dur, words, emph):
    """Scale keyframes of a quick zoom inside a shot (one of the sample's per-shot
    `moves`): it starts on a word onset near the sample's timing, an emphasised word if
    the shot has one, and holds to the cut. None if the shot is too short for it."""
    length = max(0.1, float(mv["t1"]) - float(mv["t0"]))
    lo, hi = 0.1, dur - length - 0.05
    if hi < lo:
        return None
    inside = [w for w in words if start + lo <= w["s"] <= start + hi]
    pref = [w for w in inside if _bare(w["w"]) in emph] or inside
    want = start + min(hi, max(lo, float(mv["t0"])))
    t0 = (min(pref, key=lambda w: abs(w["s"] - want))["s"] if pref else want) - start
    sc = float(mv["scale"])
    a, b = (z, z * sc) if sc >= 1 else (z / sc, z)        # a zoom-out starts tighter, never under-covers
    return [[0, round(a, 4)], [round(t0, 3), round(a, 4)],
            [round(t0 + length, 3), round(b, 4), mv.get("ease") or "out"]]


# ------------------------------------------------------------------ sound effects

def _sfx_profile(style):
    """How the sample uses effects, per occasion (cut / caption appearance): the kind it
    uses there, how often, where the sound's anchor sits relative to the moment (a
    whoosh's or riser's loudest point, an impact's onset) and its level vs the voice."""
    sx = _g(style, "audio", "sfx") or {}
    evs = [e for e in sx.get("events") or [] if e.get("type")]
    occasions = (("cut", [s_["s"] for s_ in (style.get("shots") or [])[1:]], sx.get("on_cut_rate")),
                 ("caption", [e["s"] for e in (_g(style, "captions", "events") or [])], sx.get("on_caption_rate")))
    prof = {}
    for occ, times, rate in occasions:
        if not rate or not times:
            continue
        hits = []
        for e in evs:
            a = e.get("end", e["t"]) if e["type"] == "riser" else e.get("anchor", e["t"])
            c = min(times, key=lambda c: abs(a - c))
            if abs(a - c) <= 0.2:
                hits.append((e, a - c))
        if hits:
            kind = Counter(e["type"] for e, _ in hits).most_common(1)[0][0]
            same = [(e, d) for e, d in hits if e["type"] == kind]
            levels = [e["level_db"] for e, _ in same if e.get("level_db") is not None]
            prof[occ] = {"kind": kind, "rate": rate, "offset": statistics.median([d for _, d in same]),
                         "level_db": statistics.median(levels) if levels else None}
        else:
            types = sx.get("types") or {}
            prof[occ] = {"kind": max(types, key=types.get) if types else ("whoosh" if occ == "cut" else "pop"),
                         "rate": rate, "offset": 0.0, "level_db": None}
    return prof


_SOUNDS = {}


def _sound(path, kind):
    """(duration, anchor offset, level dBFS) of an effect file, measured the way the
    sample analyser measures effects: loudest point / onset, and RMS over the same window."""
    key = (path, kind)
    if key not in _SOUNDS:
        sr = 48000
        x = ff.audio(path, sr).astype(np.float64)
        n = len(x)
        if not n:
            _SOUNDS[key] = (0.0, 0.0, -120.0)
            return _SOUNDS[key]
        w = int(0.02 * sr)
        pk = int(np.argmax(np.convolve(x ** 2, np.ones(w) / w, "same")))
        if kind in ("whoosh", "swish", "riser"):
            anchor = pk / sr
            seg = x[max(0, pk - int(0.05 * sr)):pk + int(0.05 * sr)] if kind != "riser" else x
        else:
            on = int(np.argmax(np.abs(x) > 0.1 * np.abs(x).max()))
            anchor = on / sr
            seg = x[on:on + int(0.05 * sr)] if kind in ("pop", "impact", "ding") else x[on:]
        rms = math.sqrt(float(np.mean(seg ** 2))) if len(seg) else 0.0
        _SOUNDS[key] = (n / sr, anchor, 20 * math.log10(rms + 1e-9))
    return _SOUNDS[key]


def _voice_db(runs, envelopes):
    """RMS level (dBFS) of the reel's speech, from the sources' 5 ms envelopes."""
    parts = []
    for r in runs:
        env = (envelopes or {}).get(r["src"])
        if env is None:
            continue
        for w in r["words"]:
            a, b = int(w["s"] / planlib.HOP), int(math.ceil(w["e"] / planlib.HOP))
            parts.append(np.asarray(env[max(0, a):max(0, b)], np.float64))
    x = np.concatenate(parts) if parts else np.zeros(0)
    return 20 * math.log10(math.sqrt(float(np.mean(x ** 2))) + 1e-9) if len(x) else None


# ------------------------------------------------------------------ inputs

def _sfx_path(kind, plan):
    ref = (plan.get("sfx") or {}).get(kind)
    if ref:
        return str(store.get(ref).path)
    try:
        from . import sfx
        p = sfx.get(kind)
        return str(p) if p else None
    except Exception:
        return None


def _lang(plan, style):
    return (plan.get("output") or {}).get("lang") or _g(style, "audio", "speech", "lang", default="en") or "en"


def _transcript(asset, lang, provider=None):
    return source.transcript(asset, lang, provider=provider) if provider else source.transcript(asset, lang)


def _reel_sources(reel):
    segs = ([reel["hook"]] if reel.get("hook") else []) + list(reel.get("segments") or [])
    return sorted({seg.get("src") for seg in segs if isinstance(seg, dict)}, key=str)


# ------------------------------------------------------------------ dependencies

_SHA = {}
_FP_WARNED = set()


def file_sha(path):
    """Full sha-256 of a file (None when missing), remembered per (path, size, mtime, inode)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(path), st.st_size, st.st_mtime_ns, st.st_ino)
    if key not in _SHA:
        _SHA[key] = fsutil.sha256_file(path)
    return _SHA[key]


def _stat(path):
    try:
        st = os.stat(path)
        return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    except OSError:
        return {"path": str(path), "size": None, "mtime_ns": None}


def asr_fingerprint(lang, provider=None):
    """asr.fingerprint (language, model, provider, timing precision); a stand-in naming the
    language and provider when the ASR module cannot give one."""
    from . import asr
    try:
        return str(asr.fingerprint(lang, provider=provider))
    except Exception as e:
        if (lang, provider) not in _FP_WARNED:
            _FP_WARNED.add((lang, provider))
            ff.log(f"ASR fingerprint unavailable ({ff.describe(e)}): keyed by language and provider only")
        return f"unknown:{lang}:{provider or 'default'}"


def _font_deps(tl):
    """sha-256 of every font file the timeline's text events are drawn with."""
    from . import captions
    styles = tl.get("styles") or {}
    out = {}
    for ev in list(tl.get("captions") or []) + list(tl.get("text") or []):
        st = ev.get("style", "caption")
        st = st if isinstance(st, dict) else styles.get(st) or {}
        s = dict(captions.DEFAULT, **st)
        key = f"{s.get('font')}-{s.get('weight')}"
        if key in out:
            continue
        try:
            out[key] = file_sha(captions.font_file(s["font"], s["weight"]))
        except (Exception, SystemExit) as e:          # no font and no fallback: the render reports it
            out[key] = None
            ff.log(f"font {key}: {e}")
    return dict(sorted(out.items()))


def dependencies(tl, base_dir, plan=None, style=None, lang=None, provider=None, transcripts=None):
    """deps.json of a timeline: everything that can change its rendering without changing its JSON.
    {"assets": {source key: sha256}, "copies": [{path, size, mtime_ns}] (editing copies, soundtracks
    and any other media), "transcripts": {source key: ASR fingerprint — the one the transcript in
    use carries, else asr's for lang / provider}, "style_sha256", "lut_sha256",
    "fonts": {"Family-weight": sha256}, "sfx": {path: sha256}, "music": sha256 | None, "renderer"}."""
    deps = {"assets": {}, "copies": [], "transcripts": {},
            "style_sha256": fsutil.sha256_json(style) if style is not None else None, "lut_sha256": None,
            "fonts": {}, "sfx": {}, "music": None, "renderer": render.RENDERER_VERSION}
    music = None
    if plan:
        used = {r.get("src") for r in _g(tl, "meta", "runs", default=[]) or [] if isinstance(r, dict)}
        for key, ref in sorted((plan.get("sources") or {}).items()):
            if used and key not in used:          # only the sources this reel plays
                continue
            a = store.get(ref)
            deps["assets"][key] = a.meta.get("sha256") or a.id
            fp = ((transcripts or {}).get(key) or {}).get("fingerprint")
            deps["transcripts"][key] = fp if isinstance(fp, str) else asr_fingerprint(lang or "en", provider)
        if isinstance(plan.get("music"), dict) and plan["music"].get("asset"):
            m = store.get(plan["music"]["asset"])
            music = m.meta.get("sha256") or file_sha(m.path)
    for key, ref in sorted((tl.get("media") or {}).items()):
        path = render._resolve(ref, base_dir)
        if key.startswith("sfx_"):
            deps["sfx"][path] = file_sha(path)
        elif key == "music" and music is not None:
            deps["music"] = music
        else:
            deps["copies"].append(_stat(path))
    lut = (tl.get("grade") or {}).get("lut")
    if lut:
        deps["lut_sha256"] = file_sha(render._resolve(lut, base_dir))
    deps["fonts"] = _font_deps(tl)
    return deps


# ------------------------------------------------------------------ shared context

class Context:
    """What every reel of a plan shares, each made once: source assets, transcripts, audio
    envelopes, editing copies for the union of all reels' runs (made on first use) and each
    source's colour look. A source whose transcript or envelope fails is kept in .errors;
    only the reels that use it fail (check())."""

    def __init__(self, plan, style=None, provider=None, reels=None):
        self.plan = planlib.load(plan)
        self.style = load_style(style if style is not None else self.plan["style"])
        o = self.plan.get("output") or {}
        self.lang = _lang(self.plan, self.style)
        self.provider = provider
        self.fps = int(o.get("fps", 30))
        self.max_pause = float(_g(self.style, "pacing", "max_pause", default=0.25) or 0.25)
        self.reels = [r for r in self.plan["reels"] if reels is None or r["id"] in reels]
        self.assets, self.transcripts, self.envelopes, self.errors = {}, {}, {}, {}
        for key, ref in self.plan["sources"].items():
            try:
                a = self.assets[key] = store.get(ref)
                self.transcripts[key] = _transcript(a, self.lang, provider)
                if a.info.get("has_audio"):
                    self.envelopes[key] = planlib.envelope(source.audio(a), cache=str(a.cache("envelope.npy")))
            except Exception as e:
                self.errors[key] = e
                ff.log(f"source {key} ({ref}): {ff.describe(e)}")
        self._copies = None
        self.looks = {}

    def check(self, reel):
        """Raise the failure of a source this reel uses, if one failed."""
        for key in _reel_sources(reel):
            if key in self.errors:
                raise RuntimeError(f"source {key}: {ff.describe(self.errors[key])}") from self.errors[key]

    def copies(self):
        if self._copies is None:
            ranges = {}
            for r in self.reels:
                try:
                    runs = planlib.resolve_reel(r, self.transcripts, self.envelopes, max_pause=self.max_pause)
                except Exception:
                    continue                     # that reel fails in its own compile
                for run in runs:
                    ranges.setdefault(run["src"], []).append((run["s"], run["e"]))
            self._copies = {}
            for key, rs in ranges.items():
                try:
                    self._copies[key] = source.prepare(self.assets[key], rs, fps=self.fps)
                except Exception as e:           # its reels retry the preparation themselves
                    ff.log(f"source {key}: editing copies failed: {ff.describe(e)}")
        return self._copies

    def shared(self):
        """compile_reel keyword arguments for every reel of this context."""
        return {"transcripts": self.transcripts, "copies": self.copies(), "envelopes": self.envelopes,
                "looks": self.looks, "provider": self.provider}


# ------------------------------------------------------------------ compile

def compile_reel(plan, reel, style, out_dir, transcripts=None, copies=None, envelopes=None, reframing=True,
                 looks=None, provider=None, preview=False):
    """Write out_dir/timeline.json and out_dir/deps.json for one reel; returns the timeline's
    path. transcripts / copies / envelopes / looks ({source key: look}, filled in when given)
    are shared inputs (see Context); what is not given is made here."""
    plan = planlib.load(plan)
    style = load_style(style)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    o = plan.get("output", {})
    W, H, fps = int(o.get("width", 720)), int(o.get("height", 1280)), int(o.get("fps", 30))
    k = H / float(_g(style, "sample", "height", default=H))
    lang = _lang(plan, style)
    assets = {key: store.get(ref) for key, ref in plan["sources"].items()}
    if transcripts is None:
        transcripts = {key: _transcript(a, lang, provider) for key, a in assets.items()}
    if envelopes is None:
        envelopes = {key: planlib.envelope(source.audio(a), cache=str(a.cache("envelope.npy")))
                     for key, a in assets.items() if a.info.get("has_audio")}
    max_pause = float(_g(style, "pacing", "max_pause", default=0.25) or 0.25)
    runs = planlib.resolve_reel(reel, transcripts, envelopes, max_pause=max_pause)
    if not runs:
        raise ValueError(f"reel {reel['id']}: nothing left to play")
    copies = dict(copies or {})              # never add one reel's own copies to a shared dict
    for key in {r["src"] for r in runs}:
        if key not in copies:
            if preview:
                raise ValueError(f"preview needs an original-source mapping for {key}; editing copies are disabled")
            copies[key] = source.prepare(assets[key], [(r["s"], r["e"]) for r in runs if r["src"] == key], fps=fps)

    # ---- timeline positions of runs and words
    t, starts = 0.0, []
    for r in runs:
        starts.append(t)
        t += r["e"] - r["s"]
    total = round(t, 4)
    words = []
    for r, t0 in zip(runs, starts):
        for w in r["words"]:
            words.append({**w, "s": t0 + max(0.0, w["s"] - r["s"]), "e": t0 + min(r["e"], w["e"]) - r["s"]})

    # ---- shots, framing, transitions
    target = float(_g(style, "pacing", "median_shot", default=0) or _g(style, "pacing", "avg_shot", default=2.5) or 2.5)
    target = min(6.0, max(0.9, target))
    shots = split_shots(runs, target, reel.get("emphasis", ()))
    mot = style.get("motion") or {}
    levels = [float(x) for x in (mot.get("punch_scales") or [1.0])] or [1.0]
    base = min(levels)
    levels = [max(1.0, x / base) for x in levels]           # relative to the widest framing
    st_shots = style.get("shots") or []
    pushes = [s_["move"]["type"] for s_ in st_shots if (s_.get("move") or {}).get("type") in ZOOMS]
    push_out = sum(t_ == "zoom_out" for t_ in pushes) > len(pushes) / 2
    bursts = [m_ for s_ in st_shots for m_ in (s_.get("moves") or []) if m_.get("type") in ZOOMS and m_.get("scale")]
    burst_rate = sum(1 for s_ in st_shots if any(m_.get("type") in ZOOMS for m_ in s_.get("moves") or [])) \
                 / max(1, len(st_shots)) if bursts else 0
    emph = {_bare(w) for w in reel.get("emphasis", ())}
    n_burst = 0
    special = _g(style, "transitions", "special") or {}
    events = [{"start": starts[ri] + s - runs[ri]["s"], "dur": e - s} for ri, s, e in shots]
    scores = _moment_scores(events, words, emph)
    punch_at = _select_moments(events, mot.get("punch_rate"), scores, range(1, len(shots)))
    burst_at = _select_moments(events, burst_rate, scores,
                               [i for i, ev in enumerate(events) if ev["dur"] > 0.8])
    push_at = _select_moments(events, mot.get("push_rate"), scores,
                              [i for i, ev in enumerate(events) if ev["dur"] > 0.8 and i not in burst_at])
    sentence_cuts = [n for n, (ri, _, _) in enumerate(shots) if n and shots[n - 1][0] != ri
                     and runs[ri - 1]["words"] and runs[ri - 1]["words"][-1]["w"].endswith(END)]
    trans_at = _select_moments(events, special.get("rate"), scores, sentence_cuts) \
               if special.get("type") not in (None, "cut") else set()
    media, video, probes = {}, [], {}
    level = 0
    for n, (ri, s, e) in enumerate(shots):
        r = runs[ri]
        m, off = source.locate(copies[r["src"]], s, e)
        mkey = media.setdefault(m["path"], f"v{len(media)}")
        start = round(starts[ri] + (s - r["s"]), 4)
        dur = round(e - s, 4)
        if n in punch_at:
            level = (level + 1) % len(levels)
        z = levels[level]
        clip = {"media": mkey, "start": start, "in": off, "dur": dur, "fit": "cover"}
        keys = None
        if n in burst_at:
            keys = _burst(z, bursts[n_burst % len(bursts)], start, dur, words, emph)
            n_burst += 1 if keys else 0
        if keys:
            clip["scale"] = keys
        elif n in push_at:
            z1 = z * (1 + float(mot.get("push_scale_per_s") or 0.03) * dur)
            a_, b_ = (z1, z) if push_out else (z, z1)
            clip["scale"] = [[0, round(a_, 4)], [dur, round(b_, 4), mot.get("push_ease") or "inout"]]
        else:
            clip["scale"] = round(z, 4)
        if n in trans_at:         # special transitions only between sentences, never inside one
            clip["transition_in"] = {"type": special["type"], "dur": float(special.get("dur", 0.3))}
        scale_at = (lambda sc: (lambda tt: render.kf(sc, tt, 1.0)))(clip["scale"])
        if reframing:
            info = probes.get(m["path"])
            if info is None:                # setdefault would probe every time: its default is evaluated first
                info = probes[m["path"]] = ff.probe(m["path"])
            rk = reframe.clip_keyframes(m["path"], off, dur, (W, H), scale_at, info=info, fps=fps)
            clip["x"], clip["y"] = rk["x"], rk["y"]
            clip["subject"] = rk["subject"]
        video.append(clip)
    timeline_media = {v: p for p, v in media.items()}

    # ---- text: captions, hook, CTA
    cap_cfg = style.get("captions") or {}
    styles, cap_events, text_events = {}, [], []
    if cap_cfg.get("present", True):
        styles["caption"] = scale_style(cap_cfg.get("style") or {}, k)
        cap_events = chunk_captions(words, cap_cfg, reel.get("emphasis", ()))
        hl = styles["caption"].get("highlight") or {}
        if hl.get("mode") == "keyword":
            _auto_keywords(cap_events, hl.get("rate", 1.0))
    tit = style.get("titles") or {}
    styles["title"] = scale_style(tit.get("style") or cap_cfg.get("style") or {}, k)
    hook_dur = float(_g(style, "structure", "hook", "dur", default=2.0) or 2.0)
    hook_type = _g(style, "structure", "hook", "type", default="title" if tit.get("present") else "caption")
    if reel.get("title_text"):
        hook_dur = min(hook_dur, total)
        if hook_type == "caption":
            cap_events = [c for c in cap_events if c["s"] >= hook_dur - 0.05]
            if cap_events:
                cap_events[0]["s"] = max(cap_events[0]["s"], hook_dur)
            cap_events.insert(0, {"s": 0.0, "e": round(hook_dur, 3), "text": reel["title_text"], "role": "title"})
        else:
            text_events.append({"s": 0.0, "e": round(hook_dur, 3), "text": reel["title_text"], "style": "title", "role": "title"})
    if reel.get("cta"):
        cta_s = _g(style, "structure", "cta", "s")
        sdur = _g(style, "structure", "duration")
        cta_dur = min(3.5, max(1.5, (sdur - cta_s) if (cta_s is not None and sdur) else 2.5))
        cs = round(max(0.0, total - cta_dur), 3)
        kept = []
        for c in cap_events:
            if c["s"] >= cs - 0.1:
                continue
            c["e"] = min(c["e"], cs)
            kept.append(c)
        cap_events = kept + [{"s": cs, "e": round(total, 3), "text": reel["cta"], "role": "title"}]
    cap_events = [c for c in cap_events if c["e"] - c["s"] >= 0.1]

    # ---- audio
    clips = []
    for r, t0 in ([] if preview else zip(runs, starts)):
        akey = f"a_{r['src']}"
        if akey not in timeline_media:
            timeline_media[akey] = source.audio(assets[r["src"]])
        clips.append({"media": akey, "start": round(t0, 4), "in": r["s"], "dur": round(r["e"] - r["s"], 4),
                      "role": "voice", "mono": True})
    music = None if preview else plan.get("music")
    if music:
        mpath = str(store.get(music["asset"]).path)
        timeline_media["music"] = mpath
        mdur = ff.probe(mpath)["duration"]
        t0 = 0.0
        while t0 < total - 0.05 and mdur > 0.5:
            d = min(mdur, total - t0)
            clips.append({"media": "music", "start": round(t0, 3), "in": 0, "dur": round(d, 3), "role": "music",
                          "gain_db": music.get("gain_db", -16), "fade_in": 0.02 if t0 else 0, "fade_out": 0.3})
            t0 += d
    sfx_cfg = _g(style, "audio", "sfx") or {}
    prof = {} if preview else _sfx_profile(style)
    voice_db = None if preview else _voice_db(runs, envelopes)
    moments = []
    if "cut" in prof:
        chosen = _select_moments(events, prof["cut"]["rate"], scores, range(1, len(video)))
        moments += [(events[i]["start"], prof["cut"]) for i in sorted(chosen)]
    if "caption" in prof:
        taken = [t_ for t_, _ in moments]
        cevents = [{"start": c["s"], "dur": c["e"] - c["s"]} for c in cap_events]
        chosen = _select_moments(cevents, prof["caption"]["rate"], _moment_scores(cevents, words, emph),
                                  [i for i, c in enumerate(cap_events) if all(abs(c["s"] - t_) > 0.15 for t_ in taken)])
        moments += [(cap_events[i]["s"], prof["caption"]) for i in sorted(chosen)]
    for t_, p in sorted(moments, key=lambda m_: m_[0]):
        kd = p["kind"]
        path_ = _sfx_path(kd, plan)
        if not path_:
            continue
        d, anchor, level = _sound(path_, kd)
        st = t_ + p["offset"] - anchor
        if d <= 0 or st < 0 or st > total - 0.05:
            continue
        if p["level_db"] is not None and voice_db is not None:
            gain = p["level_db"] + voice_db + VOICE_CHAIN_DB - level
        else:
            gain = float(sfx_cfg.get("gain_db", -10))
        key = f"sfx_{kd}"
        timeline_media.setdefault(key, path_)
        clips.append({"media": key, "start": round(st, 3), "in": 0, "dur": round(min(d, total - st), 3),
                      "role": "sfx", "gain_db": round(min(12.0, max(-40.0, gain)), 1)})

    # ---- grade
    g = style.get("grade") or {}
    tl_grade = {key: g[key] for key in ("sharpen", "grain", "vignette") if g.get(key)}
    if g.get("look") and not preview:
        main = max({r["src"] for r in runs}, key=lambda s_: sum(r["e"] - r["s"] for r in runs if r["src"] == s_))
        if looks is not None and main in looks:
            src_look = looks[main]
        else:
            src_look = source.look(assets[main], copies[main])
            if looks is not None:
                looks[main] = src_look
        lut = grade.build_lut(src_look, g["look"], strength=float(g.get("strength", 1.0)))
        cube = out_dir / "look.cube"
        tmp = out_dir / f".look.{os.getpid()}.tmp.cube"
        try:                                 # a new file, never rewritten in place: renders hard-link it
            grade.write_cube(lut, tmp)
            os.replace(tmp, cube)
        finally:
            if tmp.exists():
                tmp.unlink()
        tl_grade["lut"] = str(cube.resolve())

    loud = o.get("loudness")
    if loud is None:          # the sample's own loudness, within broadcast-sane bounds
        loud = min(-7.0, max(-18.0, float(_g(style, "audio", "loudness", "I", default=-14.0) or -14.0)))
    music_cfg = _g(style, "audio", "music") or {}
    tl = {"width": W, "height": H, "fps": fps, "duration": total, "background": "#000000",
          "media": timeline_media, "video": video, "captions": cap_events, "text": text_events, "styles": styles,
          "grade": tl_grade,
          "audio": {"clips": clips, "voice_chain": True, "duck_db": music_cfg.get("duck_db", 12),
                    "loudness": float(loud), "true_peak": -1.0},
          "encode": dict(QUALITY.get(o.get("quality", "final"), QUALITY["final"])),
          "meta": {"reel": reel["id"], "title": reel.get("title"), "runs": [{"src": r["src"], "s": r["s"], "e": r["e"]}
                                                                         for r in runs],
                   "words": [w["w"] for w in words]}}
    # JSON round trip first: the dependencies (and the batch's render key) see what is on disk
    tl = json.loads(json.dumps(tl, ensure_ascii=False))
    deps = dependencies(tl, str(out_dir), plan, style, lang, provider, transcripts)
    path = out_dir / "timeline.json"
    fsutil.atomic_write_json(path, tl)
    fsutil.atomic_write_json(out_dir / "deps.json", deps)
    return str(path)


def compile_plan(plan, out_root, reels=None, reframing=True, provider=None):
    """Compile every reel (or the listed ids) of a plan through one Context (editing copies
    made once per source for the union of those reels' ranges); returns {reel_id: timeline path}."""
    ctx = Context(plan, provider=provider, reels=reels)
    out = {}
    for r in ctx.reels:
        ctx.check(r)
        out[r["id"]] = compile_reel(ctx.plan, r, ctx.style, Path(out_root) / "reels" / r["id"], reframing=reframing,
                                    **ctx.shared())
    return out

