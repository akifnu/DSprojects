"""Executable contracts: one canonical check for each rule on what crosses a module or agent
boundary (request, style, plan, timeline), the review normaliser and the plan-entry hash.

validate_request / validate_style / validate_plan / validate_timeline return
{"errors": [...], "warnings": [...]}, every message naming the offending field, and never
raise on bad input. Errors block (an agent's answer is refused, a plan or timeline never
reaches rendering); warnings are recorded. Unknown keys are errors where they could ask for
an effect nothing implements (plan reels and segments, timeline clips, text events, text
styles, grade, audio and encode settings) and warnings elsewhere; "meta" (an object) is the
free extension field of the plan, its reels and segments, timeline clips and events and the
timeline itself. Measured style sections are checked only where the compiler consumes them.

normalize_review(raw, assigned, plan_generation) checks a reviewer's envelope (or the legacy
{"fixes", "ok"} form) against the assigned reels: {"results": [{reel_id, decision, findings,
render_sha256?}], "errors": [...], "unreviewed": [ids]}; a reel without exactly one valid
decision stays unreviewed. plan_entry_hash(reel) is fsutil.sha256_json of a plan entry.

Light on purpose: no media libraries; the language list comes from asr on first use.
"""
import bisect
import math
import numbers
import re

from . import fsutil

STYLE_VERSION = 2
PLAN_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1

ROLES = ("sample", "source", "music", "sfx", "image")
ENCODERS = ("auto", "cpu", "nvenc")
ASR_PROVIDERS = ("cpu", "cuda")
DECISIONS = ("accept", "fix_required")
SEVERITIES = ("blocking", "minor")
QC_FAILED = ("fail", "error")
# what render.py implements (Compositor._transition, render.EASES + linear / hold, fit modes)
TRANSITIONS = ("cut", "dissolve", "dip_black", "dip_white", "flash", "zoom", "whip_left", "whip_right", "whip_up",
               "whip_down")
EASES = ("linear", "hold", "in", "out", "inout", "in_cubic", "out_cubic", "inout_cubic", "out_back", "in_quart",
         "out_quart")
FITS = ("cover", "contain", "stretch", "none")
AUDIO_ROLES = ("voice", "music", "sfx")
X264_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow",
                "placebo")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")          # reel ids: also file and directory names
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")        # plan / batch names
COLOR_RE = re.compile(r"#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})")
MAX_FINDINGS = 100          # per reel and review: more is a broken reviewer, not more evidence
TEXT_CHARS = 2000           # finding what / how
EDGE = 0.05                 # s of slack at segment and transcript bounds (plan.segment_words uses the same)


def __getattr__(name):
    if name == "SUPPORTED_LANGS":           # "en" + asr's Whisper languages; imported on first use
        from .asr import WHISPER_LANGS
        langs = globals()["SUPPORTED_LANGS"] = frozenset(WHISPER_LANGS) | {"en"}
        return langs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def lang_problem(lang):
    """None when transcription supports `lang` (asr.check_lang: "en", "auto" or a Whisper code)."""
    if lang == "auto":
        return None
    try:
        langs = globals().get("SUPPORTED_LANGS") or __getattr__("SUPPORTED_LANGS")
    except (Exception, SystemExit) as e:        # asr needs numpy and ffmpeg
        return f"cannot be checked ({type(e).__name__}: {e})"
    if isinstance(lang, str) and lang in langs:
        return None
    return f"{lang!r} is not a supported language ('en', 'auto' or a Whisper language code)"


# ------------------------------------------------------------------ small checkers

def _num(v):
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def _int(v):
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _show(v, n=60):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "…"


def number(lo=None, hi=None, gt=None, integer=False):
    def check(v):
        if not (_int(v) if integer else _num(v)):
            return "must be an integer" if integer else "must be a finite number"
        if gt is not None and not v > gt:
            return f"must be > {gt}"
        if lo is not None and v < lo:
            return f"must be ≥ {lo}"
        if hi is not None and v > hi:
            return f"must be ≤ {hi}"
        return None
    return check


NUM, POS, NONNEG, UNIT = number(), number(gt=0), number(lo=0), number(lo=0, hi=1)
WEIGHT = number(lo=1, hi=1000)


def _string(v):
    return None if isinstance(v, str) else "must be a string"


def _nonempty(v):
    return None if isinstance(v, str) and v.strip() else "must be a non-empty string"


def _boolean(v):
    return None if isinstance(v, bool) else "must be true or false"


def _color(v):
    return None if isinstance(v, str) and COLOR_RE.fullmatch(v) else "must be a hex colour (#rgb, #rrggbb, #rrggbbaa)"


def _object(v):
    return None if isinstance(v, dict) else "must be an object"


def _anything(v):
    return None


def _ease(v):
    if v is None or v in EASES:
        return None
    if isinstance(v, list) and len(v) == 4 and all(_num(x) for x in v):
        return None
    return f"must be one of {', '.join(EASES)} or a cubic-bezier [x1, y1, x2, y2]"


def _language(v):
    return lang_problem(v)


def _opt(check):
    return lambda v: None if v is None else check(v)


def _among(*allowed):
    return lambda v: None if isinstance(v, str) and v in allowed else f"must be one of {', '.join(allowed)}"


class _Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def err(self, where, msg):
        self.errors.append(f"{where}: {msg}" if where else msg)

    def warn(self, where, msg):
        self.warnings.append(f"{where}: {msg}" if where else msg)

    def out(self):
        return {"errors": self.errors, "warnings": self.warnings}


def _check(r, where, v, spec, strict=True):
    """Report what is wrong with v. spec: a checker (v → message or None), a tuple of allowed
    values, or a dict spec for an object (None allowed) whose "*" entry checks every value of a
    free-keyed map; unknown keys are errors when strict, warnings otherwise."""
    if isinstance(spec, dict):
        if v is None:
            return
        if not isinstance(v, dict):
            r.err(where, f"must be an object (got {_show(v)})")
            return
        for k, x in v.items():
            if k in spec:
                _check(r, f"{where}.{k}", x, spec[k], strict)
            elif "*" in spec:
                _check(r, f"{where}.{k}", x, spec["*"], strict)
            else:
                (r.err if strict else r.warn)(where, f"unknown key {k!r} (known: {', '.join(sorted(spec))})")
        return
    if isinstance(spec, tuple):
        if not isinstance(v, (str, int, float, type(None))) or v not in spec:
            r.err(where, f"{_show(v)} is not one of {', '.join(map(str, spec))}")
        return
    msg = spec(v)
    if msg:
        r.err(where, f"{msg} (got {_show(v)})")


def _unknown(r, where, d, known, strict=True, what=""):
    for k in d:
        if k not in known and not (isinstance(k, str) and k.startswith("_")):
            (r.err if strict else r.warn)(where, f"unknown key {k!r}{what} (known: {', '.join(sorted(known))})")


# ------------------------------------------------------------------ text styles (captions.py)

_BOX = {"color": _color, "opacity": UNIT, "radius": NONNEG, "pad_x": NONNEG, "pad_y": NONNEG}
TEXT_STYLE = {
    "font": _nonempty, "weight": WEIGHT, "size": POS, "fill": _color,
    "case": ("upper", "lower", "title", "sentence", "none"), "align": ("left", "center", "right"),
    "line_height": POS, "max_width": POS, "x": NUM, "y": NUM, "letter_spacing": NUM,
    "stroke": {"color": _color, "width": NONNEG},
    "shadow": {"color": _color, "opacity": UNIT, "blur": NONNEG, "dx": NUM, "dy": NUM},
    "box": dict(_BOX, mode=("line", "block")),
    "highlight": {"fill": _color, "box": _BOX, "scale": POS, "mode": ("active", "keyword"), "rate": UNIT,
                  "font": _nonempty, "weight": WEIGHT},
    "anim": {"in": ("none", "pop", "fade", "slide_up", "slide_down", "zoom", "word_reveal"), "in_dur": NONNEG,
             "overshoot": NUM, "from": POS, "out": ("none", "fade", "pop"), "out_dur": NONNEG, "slide_px": NUM},
}


def _text_style(r, where, st):
    if not isinstance(st, dict):
        r.err(where, f"a captions.py style object is required (got {_show(st)})")
        return
    _check(r, where, st, TEXT_STYLE, strict=True)


# ------------------------------------------------------------------ request

REQUEST_FIELDS = {"count": _opt(number(gt=0, integer=True)), "duration_seconds": _opt(POS), "lang": _language,
                  "notes": _string}
EXECUTION = {"encoder": ENCODERS, "require_gpu": _boolean, "allow_cpu_fallback": _boolean, "asr_provider": ASR_PROVIDERS}


def validate_request(req):
    """{"schema_version": 1, "assets": [{"path", "role"}], "request": {"count", "duration_seconds",
    "lang", "notes"}, "execution": {"encoder", "require_gpu", "allow_cpu_fallback", "asr_provider"}}.
    File existence is the orchestrator's (it resolves the paths); this checks shapes and values."""
    r = _Report()
    if not isinstance(req, dict):
        r.err("request", "a JSON object is required")
        return r.out()
    _unknown(r, "request.json", req, ("schema_version", "assets", "request", "execution", "meta"), strict=False,
             what=" (ignored)")
    sv = req.get("schema_version", REQUEST_SCHEMA_VERSION)
    if sv != REQUEST_SCHEMA_VERSION or isinstance(sv, bool):
        r.err("schema_version", f"{_show(sv)}: this version reads {REQUEST_SCHEMA_VERSION}")
    assets = req.get("assets", [])
    if not isinstance(assets, list):
        r.err("assets", "a list of {path, role} is required")
        assets = []
    seen = {}
    for i, a in enumerate(assets):
        w = f"assets[{i}]"
        if not isinstance(a, dict):
            r.err(w, "an object {path, role} is required")
            continue
        _unknown(r, w, a, ("path", "role", "meta"), strict=False, what=" (ignored)")
        if _nonempty(a.get("path")):
            r.err(f"{w}.path", "a file path is required")
            continue
        if a.get("role") not in ROLES:
            r.err(f"{w}.role", f"{_show(a.get('role'))} is not one of {', '.join(ROLES)}")
            continue
        roles = seen.setdefault(a["path"], [])
        if a["role"] in roles:
            r.warn(w, f"{a['path']} is listed twice as {a['role']}")
        elif roles:
            r.warn(w, f"{a['path']} is used as {' and '.join(roles + [a['role']])}")
        roles.append(a["role"])
    rq = req.get("request", {})
    if not isinstance(rq, dict):
        r.err("request", "an object {count, duration_seconds, lang, notes} is required")
        rq = {}
    _unknown(r, "request", rq, REQUEST_FIELDS, strict=False, what=" (ignored)")
    for k, check in REQUEST_FIELDS.items():
        if k in rq and not (k == "notes" and rq[k] is None):
            _check(r, f"request.{k}", rq[k], check)
    if _int(rq.get("count")) and rq["count"] > 500:
        r.warn("request.count", f"{rq['count']} reels is a very large batch")
    d = rq.get("duration_seconds")
    if _num(d) and d > 0 and not 3 <= d <= 180:
        r.warn("request.duration_seconds", f"{d:g} s is outside the usual 3–180 s of a reel")
    if isinstance(rq.get("notes"), str) and len(rq["notes"]) > 20000:
        r.warn("request.notes", f"{len(rq['notes'])} characters: very long instructions")
    ex = req.get("execution", {})
    if not isinstance(ex, dict):
        r.err("execution", "an object {encoder, require_gpu, allow_cpu_fallback, asr_provider} is required")
        ex = {}
    _unknown(r, "execution", ex, EXECUTION, strict=False, what=" (ignored)")
    for k, check in EXECUTION.items():
        if k in ex:
            _check(r, f"execution.{k}", ex[k], check)
    if ex.get("require_gpu") is True and ex.get("allow_cpu_fallback") is True:
        r.warn("execution", "allow_cpu_fallback has no effect together with require_gpu")
    return r.out()


# ------------------------------------------------------------------ style

def _style_shots(r, shots):
    for i, s in enumerate(shots):
        w = f"shots[{i}]"
        if not isinstance(s, dict):
            r.err(w, "must be an object")
            continue
        for k in ("s", "e"):
            if (k == "s" or k in s) and not _num(s.get(k)):      # s times the effects on cuts
                r.err(f"{w}.{k}", f"must be a finite number (got {_show(s.get(k))})")
        if _num(s.get("s")) and _num(s.get("e")) and s["e"] < s["s"]:
            r.warn(w, f"ends ({s['e']}) before it starts ({s['s']})")
        tin = s.get("transition_in")
        if isinstance(tin, dict):
            if tin.get("type") not in TRANSITIONS:
                r.warn(f"{w}.transition_in", f"{_show(tin.get('type'))} is not a transition the renderer draws")
        elif tin not in (None, "cut"):
            r.warn(f"{w}.transition_in", f"{_show(tin)}: \"cut\" or {{type, dur}} expected")
        mv = s.get("move")
        if mv is not None:
            if not isinstance(mv, dict):
                r.err(f"{w}.move", "must be an object")
            elif mv.get("type") not in (None, "static", "zoom_in", "zoom_out", "pan"):
                r.warn(f"{w}.move.type", f"{_show(mv.get('type'))} is not static | zoom_in | zoom_out | pan")
        moves = s.get("moves")
        if moves is None:
            continue
        if not isinstance(moves, list):
            r.err(f"{w}.moves", "must be a list")
            continue
        for j, m in enumerate(moves):
            mw = f"{w}.moves[{j}]"
            if not isinstance(m, dict):
                r.err(mw, "must be an object")
                continue
            if m.get("type") in ("zoom_in", "zoom_out") and m.get("scale"):    # compiled into zoom keyframes
                for k in ("t0", "t1"):
                    _check(r, f"{mw}.{k}", m.get(k), NUM)
                _check(r, f"{mw}.scale", m.get("scale"), POS)
                _check(r, f"{mw}.ease", m.get("ease"), _ease)


STYLE_KEYS = ("version", "sample", "pacing", "shots", "motion", "transitions", "captions", "titles", "overlays", "grade",
              "audio", "structure", "transcript", "review", "analysis", "analysis_seconds", "meta")


def validate_style(style):
    """Structure of a style spec (SPEC.md "Style spec"), checked where the compiler and the
    renderer consume it; the caption and title styles strictly (they are drawn as given)."""
    r = _Report()
    if not isinstance(style, dict):
        r.err("style", "a JSON object is required")
        return r.out()
    _unknown(r, "style", style, STYLE_KEYS, strict=False, what=" (the compiler ignores it)")
    v = style.get("version")
    if not (_int(v) and 1 <= v <= STYLE_VERSION):
        r.err("version", f"{_show(v)}: an integer 1–{STYLE_VERSION} is required")
    sample = style.get("sample")
    if not isinstance(sample, dict):
        r.err("sample", "an object {id, name, duration, width, height, fps} is required")
    else:
        for k, check in (("width", number(gt=0, integer=True)), ("height", number(gt=0, integer=True)),
                         ("duration", POS)):
            _check(r, f"sample.{k}", sample.get(k), check)
        if "fps" in sample:
            _check(r, "sample.fps", sample["fps"], _opt(POS))
    _check(r, "pacing", style.get("pacing"), {"*": _opt(_anything), "avg_shot": _opt(NONNEG),
                                              "median_shot": _opt(NONNEG), "max_pause": _opt(NONNEG),
                                              "min_shot": _opt(NONNEG), "max_shot": _opt(NONNEG)})
    shots = style.get("shots")
    if not isinstance(shots, list):
        r.err("shots", f"a list is required (got {_show(shots)})")
    else:
        _style_shots(r, shots)
    _check(r, "motion", style.get("motion"), {
        "*": _anything, "punch_rate": _opt(NONNEG), "push_rate": _opt(NONNEG), "push_scale_per_s": _opt(NONNEG),
        "push_ease": _ease, "shake": _opt(NONNEG),
        "punch_scales": _opt(lambda x: None if isinstance(x, list) and all(_num(k) and k > 0 for k in x)
                             else "must be a list of positive numbers")})
    tr = style.get("transitions")
    _check(r, "transitions", tr, {"*": _anything})
    sp = tr.get("special") if isinstance(tr, dict) else None
    if sp is not None:          # compiled into clip transition_in
        if not isinstance(sp, dict):
            r.err("transitions.special", "must be null or {type, dur, rate}")
        else:
            _check(r, "transitions.special.type", sp.get("type"), TRANSITIONS)
            if "dur" in sp:
                _check(r, "transitions.special.dur", sp["dur"], NONNEG)
            if "rate" in sp:
                _check(r, "transitions.special.rate", sp["rate"], _opt(UNIT))
    if isinstance(tr, dict) and tr.get("default") is not None and tr["default"] not in TRANSITIONS:
        r.warn("transitions.default", f"{_show(tr['default'])} is not a transition the renderer draws")
    cap = style.get("captions")
    if cap is None:
        r.warn("captions", "missing: reels get captions in the default style")
    elif not isinstance(cap, dict):
        r.err("captions", "must be an object")
    else:
        _check(r, "captions.present", cap.get("present", True), _boolean)
        if cap.get("present", True):
            _text_style(r, "captions.style", cap.get("style"))
        _check(r, "captions", cap, {
            "*": _anything, "style": _anything, "present": _anything,
            "words_per_caption": {"*": _anything, "median": _opt(POS), "max": _opt(POS)},
            "chars_per_line": _opt(NONNEG), "max_lines": _opt(NONNEG), "lead": _opt(NUM),    # 0: the default
            "case": _opt(_among("upper", "lower", "title", "sentence", "none")),
            "events": _opt(lambda x: None if isinstance(x, list) and all(isinstance(e, dict) and _num(e.get("s"))
                                                                         for e in x)
                           else "must be a list of {s, e, text} with numeric s")}, strict=False)
    tit = style.get("titles")
    if tit is not None:
        if not isinstance(tit, dict):
            r.err("titles", "must be an object")
        else:
            _check(r, "titles.present", tit.get("present", False), _boolean)
            if tit.get("style") is not None:
                _text_style(r, "titles.style", tit["style"])
    _check(r, "grade", style.get("grade"), {"*": _anything, "look": _opt(_object), "sharpen": _opt(NONNEG),
                                            "grain": _opt(NONNEG), "vignette": _opt(NONNEG), "strength": _opt(NONNEG)})
    au = style.get("audio")
    _check(r, "audio", au, {
        "*": _anything, "loudness": {"*": _anything, "I": _opt(NUM), "TP": _opt(NUM), "LRA": _opt(NUM)},
        "music": {"*": _anything, "duck_db": _opt(NONNEG), "gain_db": _opt(NUM)},
        "sfx": {"*": _anything, "on_cut_rate": _opt(UNIT), "on_caption_rate": _opt(UNIT), "gain_db": _opt(NUM),
                "events": _opt(lambda x: None if isinstance(x, list) and all(isinstance(e, dict) for e in x)
                               else "must be a list of objects")},
        "speech": {"*": _anything, "lang": _opt(_language)}})
    _check(r, "structure", style.get("structure"), {
        "*": _anything, "duration": _opt(POS),
        "hook": {"*": _anything, "dur": _opt(NONNEG), "type": _opt(_among("caption", "title", "spoken")),
                 "text": _opt(_string)},
        "cta": {"*": _anything, "s": _opt(NUM), "text": _opt(_string)}})
    rv = style.get("review")
    if rv is not None:
        if not isinstance(rv, dict):
            r.err("review", "must be an object {verified, notes}")
        else:
            if "verified" in rv:
                _check(r, "review.verified", rv["verified"], _boolean)
            if rv.get("notes") is not None and not isinstance(rv["notes"], list):
                r.warn("review.notes", "a list of strings is expected")
    an = style.get("analysis")
    if an is None:
        r.warn("analysis", "no analysis fingerprint (a style measured by an earlier version)")
    elif not isinstance(an, dict) or not isinstance(an.get("fingerprint"), str):
        r.err("analysis", "must be an object with a fingerprint string")
    return r.out()


# ------------------------------------------------------------------ transcripts (for plans)

class _Transcript:
    """Sentence / word lookups over one transcript, built once per validation."""

    def __init__(self, tr):
        self.sents, self.words = tr["sentences"], tr["words"]
        self.starts = [w["s"] for w in self.words]
        self.ordered = all(a <= b for a, b in zip(self.starts, self.starts[1:]))
        self.word_sent = {}
        for k, s in enumerate(self.sents):
            for i in range(s["w0"], s["w1"] + 1):
                self.word_sent[i] = k
        d = tr.get("duration")
        ends = [w["e"] for w in self.words] + [s["e"] for s in self.sents]
        self.duration = d if _num(d) and d > 0 else (max(ends) if ends else None)

    def time_words(self, s, e):
        """Word indices plan.segment_words selects for explicit times."""
        if not self.ordered:
            return [i for i, w in enumerate(self.words) if w["s"] >= s - EDGE and w["e"] <= e + EDGE]
        i = bisect.bisect_left(self.starts, s - EDGE)
        out = []
        while i < len(self.words) and self.starts[i] <= e + EDGE:
            if self.words[i]["e"] <= e + EDGE:
                out.append(i)
            i += 1
        return out


def _transcript(tr):
    """A _Transcript, or the reason the transcript cannot be used."""
    if not isinstance(tr, dict):
        return "not an object"
    sents, words = tr.get("sentences"), tr.get("words")
    if not isinstance(sents, list) or not isinstance(words, list):
        return "sentences / words lists are missing"
    for k, w in enumerate(words):
        if not (isinstance(w, dict) and _num(w.get("s")) and _num(w.get("e"))):
            return f"word {k} has no numeric s / e"
    for k, s in enumerate(sents):
        if not (isinstance(s, dict) and _num(s.get("s")) and _num(s.get("e")) and _int(s.get("w0"))
                and _int(s.get("w1")) and 0 <= s["w0"] <= s["w1"] < len(words)):
            return f"sentence {k} lacks numeric s / e or valid word indices w0 ≤ w1"
    return _Transcript(tr)


# ------------------------------------------------------------------ plan

PLAN_KEYS = ("schema_version", "name", "style", "output", "sources", "music", "sfx", "reels", "meta")
REEL_KEYS = ("id", "title", "timeline", "segments", "hook", "title_text", "caption_text", "emphasis", "cta",
             "drop_words", "meta")
SEGMENT_KEYS = ("src", "sentences", "s", "e", "meta")
_EVEN = (lambda v: number(gt=0, integer=True)(v) or (f"{v} is odd: H.264 4:2:0 needs even dimensions" if v % 2
                                                    else None))
OUTPUT = {"width": _EVEN, "height": _EVEN, "fps": number(gt=0, hi=240), "quality": _opt(_among("final", "draft")),
          "lang": _opt(_language), "loudness": _opt(number(lo=-40, hi=0))}       # null = the default


def _req(request, k):
    """A request field from either shape: the normalised request's "request" object or the top level."""
    v = request.get(k)
    if v is None and isinstance(request.get("request"), dict):
        v = request["request"].get(k)
    return v


def _segment(r, w, seg, sources, trs):
    """Check one segment; returns (checked against a transcript, word count, seconds, {sentence ids})."""
    if not isinstance(seg, dict):
        r.err(w, f"must be an object (got {_show(seg)})")
        return False, 0, 0.0, set()
    _unknown(r, w, seg, SEGMENT_KEYS, what=" (segments take src + sentences or s / e)")
    if "meta" in seg:
        _check(r, f"{w}.meta", seg["meta"], _object)
    src = seg.get("src")
    if not isinstance(src, str) or src not in sources:
        r.err(f"{w}.src", f"{_show(src)} is not a key of sources ({', '.join(map(str, sources)) or 'none'})")
        return False, 0, 0.0, set()
    tr = trs.get(src)
    if "sentences" in seg:
        ss = seg["sentences"]
        if not (isinstance(ss, list) and 1 <= len(ss) <= 2 and all(_int(x) and x >= 0 for x in ss)):
            r.err(f"{w}.sentences", f"[first, last] sentence ids (non-negative integers) expected, got {_show(ss)}")
            return False, 0, 0.0, set()
        a, b = ss[0], ss[-1]
        if a > b:
            r.err(f"{w}.sentences", f"[{a}, {b}] must be ascending")
            return False, 0, 0.0, set()
        if "s" in seg or "e" in seg:
            r.warn(w, "s / e are ignored when sentences are given")
        if tr is None:
            return False, 0, 0.0, set()
        n = len(tr.sents)
        if b >= n:
            r.err(f"{w}.sentences", f"sentence {b} is out of range: the transcript of {src} has {n} sentence(s)"
                  + (f" (0–{n - 1})" if n else ""))
            return False, 0, 0.0, set()
        words = tr.sents[b]["w1"] - tr.sents[a]["w0"] + 1
        return True, words, max(0.0, tr.sents[b]["e"] - tr.sents[a]["s"]), set(range(a, b + 1))
    if "s" not in seg or "e" not in seg:
        r.err(w, "give sentences [first, last] or s / e times")
        return False, 0, 0.0, set()
    s, e = seg["s"], seg["e"]
    if not (_num(s) and _num(e) and 0 <= s < e):
        r.err(w, f"needs 0 ≤ s < e in seconds, got {_show(s)}..{_show(e)}")
        return False, 0, 0.0, set()
    if tr is None:
        return False, 0, 0.0, set()
    if tr.duration is not None and s >= tr.duration:
        r.err(w, f"starts at {s:g} s, after the end of {src} ({tr.duration:g} s)")
        return False, 0, 0.0, set()
    if tr.duration is not None and e > tr.duration + 0.5:
        r.err(w, f"ends at {e:g} s, after the end of {src} ({tr.duration:g} s)")
        return False, 0, 0.0, set()
    idx = tr.time_words(s, e)
    if not idx:
        r.warn(w, f"no transcript word lies within {s:g}–{e:g} s: it plays nothing")
    return True, len(idx), e - s, {tr.word_sent[i] for i in idx if i in tr.word_sent}


def _reel(r, i, reel, sources, trs, seen, used, durations):
    w = f"reels[{i}]"
    if not isinstance(reel, dict):
        r.err(w, f"must be an object (got {_show(reel)})")
        return
    rid = reel.get("id")
    if not (isinstance(rid, str) and ID_RE.fullmatch(rid)):
        r.err(f"{w}.id", f"{_show(rid)}: 1–64 letters, digits, '_' or '-', starting with a letter or digit")
    elif rid in seen:
        r.err(f"{w}.id", f"{rid!r} is used twice")
    else:
        seen.add(rid)
        w = f"reel {rid}"
    _unknown(r, w, reel, REEL_KEYS, what=" (an effect the compiler does not implement? use meta for notes)")
    if "meta" in reel:
        _check(r, f"{w}.meta", reel["meta"], _object)
    for k in ("title", "title_text", "cta"):
        if reel.get(k) is not None:
            _check(r, f"{w}.{k}", reel[k], _string)
    if isinstance(reel.get("title_text"), str) and len(reel["title_text"]) > 120:
        r.warn(f"{w}.title_text", f"{len(reel['title_text'])} characters: long for an on-screen title")
    if reel.get("timeline") is not None:
        _check(r, f"{w}.timeline", reel["timeline"], _nonempty)
        if reel.get("segments") or reel.get("hook"):
            r.warn(w, "segments / hook are ignored: the ready timeline is rendered as it is")
        return
    segs = reel.get("segments")
    if not isinstance(segs, list) or not segs:
        r.err(f"{w}.segments", "a non-empty list is required")
        segs = []
    parts = [(f"{w} segment {j}", s) for j, s in enumerate(segs)]
    if reel.get("hook") is not None:
        parts.append((f"{w} hook", reel["hook"]))
    all_checked, words, secs, sents = bool(parts), 0, 0.0, {}
    for label, seg in parts:
        ok, n, d, ids = _segment(r, label, seg, sources, trs)
        all_checked &= ok
        words += n
        secs += d
        if ok:
            sents.setdefault(seg["src"], set()).update(ids)
    if all_checked:
        durations[rid if isinstance(rid, str) else str(i)] = secs
        if not words:
            r.err(w, "no transcript words in any segment: nothing would play")
    if isinstance(rid, str):
        for src, ids in sents.items():
            for other, oids in used.get(src, {}).items():
                common = ids & oids
                if common:
                    r.warn(w, f"reuses {len(common)} sentence(s) of {src} that reel {other} also uses "
                              f"(e.g. {min(common)})")
            used.setdefault(src, {})[rid] = ids
    every = set().union(*sents.values()) if sents else set()
    ct = reel.get("caption_text")
    if ct is not None:
        if not isinstance(ct, dict):
            r.err(f"{w}.caption_text", "an object {sentence id: corrected text} is required")
        else:
            for k, text in ct.items():
                if not (_int(k) and k >= 0) and not (isinstance(k, str) and k.strip().isdigit()):
                    r.err(f"{w}.caption_text", f"key {_show(k)} is not a sentence id")
                    continue
                _check(r, f"{w}.caption_text.{k}", text, _nonempty)
                if all_checked and int(k) not in every:
                    r.warn(f"{w}.caption_text.{k}", "this sentence is not in the reel: the override is ignored")
    em = reel.get("emphasis")
    if em is not None and not (isinstance(em, list) and all(isinstance(x, str) and x.strip() for x in em)):
        r.err(f"{w}.emphasis", f"a list of words is required (got {_show(em)})")
    dw = reel.get("drop_words")
    if dw is not None:
        if not isinstance(dw, list):
            r.err(f"{w}.drop_words", "a list of [sentence id, word index] pairs is required")
        else:
            for j, p in enumerate(dw):
                if not (isinstance(p, list) and len(p) == 2 and all(_int(x) and x >= 0 for x in p)):
                    r.err(f"{w}.drop_words[{j}]", f"[sentence id, word index] expected, got {_show(p)}")
                    continue
                if not all_checked:
                    continue
                fits = [src for src, ids in sents.items() if p[0] in ids
                        and p[1] <= trs[src].sents[p[0]]["w1"] - trs[src].sents[p[0]]["w0"]]
                if not fits:
                    r.warn(f"{w}.drop_words[{j}]", f"word {p[1]} of sentence {p[0]} is not in the reel: ignored")


def validate_plan(plan, transcripts=None, request=None):
    """plan.json (SPEC.md "Plan"). transcripts: {source key: transcript} to check sentence ids,
    times and overrides against; request: the job's request (either shape) — count, duration
    and language mismatches are warnings."""
    r = _Report()
    if not isinstance(plan, dict):
        r.err("plan", "a JSON object is required")
        return r.out()
    _unknown(r, "plan", plan, PLAN_KEYS, strict=False, what=" (ignored)")
    sv = plan.get("schema_version")
    if sv is not None and not (_int(sv) and 1 <= sv <= PLAN_SCHEMA_VERSION):
        r.err("schema_version", f"{_show(sv)}: this version reads 1–{PLAN_SCHEMA_VERSION}")
    name = plan.get("name")
    if name is not None and not (isinstance(name, str) and NAME_RE.fullmatch(name)):
        r.err("name", f"{_show(name)}: 1–64 letters, digits, '.', '_' or '-'")
    _check(r, "style", plan.get("style"), lambda v: _nonempty(v) and "a sample asset id or a style.json path is required")
    if "meta" in plan:
        _check(r, "meta", plan["meta"], _object)
    out = plan.get("output")
    if out is not None:
        _check(r, "output", out, OUTPUT, strict=True)
    sources = plan.get("sources")
    if not isinstance(sources, dict):
        r.err("sources", "an object {key: source asset id} is required")
        sources = {}
    for k, v in sources.items():
        if _nonempty(k) or _nonempty(v):
            r.err(f"sources.{k}", f"keys and asset ids must be non-empty strings (got {_show(v)})")
    if plan.get("music") is not None:
        mu = plan["music"]
        _check(r, "music", mu, {"asset": _nonempty, "gain_db": number(lo=-60, hi=12)})
        if isinstance(mu, dict) and "asset" not in mu:
            r.err("music", "asset is required")
    if plan.get("sfx") is not None:
        _check(r, "sfx", plan["sfx"], {"*": _nonempty})
    reels = plan.get("reels")
    if not isinstance(reels, list) or not reels:
        r.err("reels", "a non-empty list is required")
        reels = []
    trs = {}
    for k, tr in (transcripts or {}).items():
        t = _transcript(tr)
        if isinstance(t, str):
            r.err(f"transcript of {k}", f"unusable ({t}): sentence ids cannot be checked")
        else:
            trs[k] = t
    referenced = {seg.get("src") for x in reels if isinstance(x, dict) and not x.get("timeline")
                  for seg in (x["segments"] if isinstance(x.get("segments"), list) else []) + [x.get("hook")]
                  if isinstance(seg, dict) and isinstance(seg.get("src"), str)}
    if transcripts is not None:
        missing = sorted(k for k in referenced if k in sources and k not in trs and k not in transcripts)
        if missing:
            r.warn("sources", f"no transcript given for {', '.join(missing)}: their sentence ids are not checked")
    seen, used, durations = set(), {}, {}
    for i, reel in enumerate(reels):
        _reel(r, i, reel, sources, trs, seen, used, durations)
    if isinstance(request, dict):
        count, target, lang = _req(request, "count"), _req(request, "duration_seconds"), _req(request, "lang")
        if _int(count) and count > 0 and len(reels) != count:
            r.warn("reels", f"{len(reels)} reel(s) planned; the request asks for {count}")
        if _num(target) and target > 0:
            for rid, secs in durations.items():
                if not 0.75 * target <= secs <= 1.25 * target:
                    r.warn(f"reel {rid}", f"about {secs:.0f} s of source speech; the request asks for {target:g} s")
        olang = out.get("lang") if isinstance(out, dict) else None
        if isinstance(lang, str) and isinstance(olang, str) and lang != olang:
            r.warn("output.lang", f"{olang!r} differs from the request's language {lang!r}")
    return r.out()


def plan_entry_hash(reel):
    """sha-256 of one plan entry's canonical JSON: an entry that hashes the same is unchanged."""
    return fsutil.sha256_json(reel)


# ------------------------------------------------------------------ timeline (render.py)

TIMELINE_KEYS = ("width", "height", "fps", "duration", "background", "media", "video", "captions", "text", "styles",
                 "grade", "audio", "encode", "meta")
CLIP_KEYS = ("media", "start", "in", "dur", "fit", "scale", "x", "y", "rotation", "opacity", "speed", "track",
             "subject", "transition_in", "hold_last", "kind", "meta")
EVENT_KEYS = ("s", "e", "text", "words", "style", "meta")
AUDIO_CLIP_KEYS = ("media", "start", "in", "dur", "role", "mono", "gain_db", "fade_in", "fade_out", "meta")
GRADE = {"lut": _nonempty, "sharpen": NONNEG, "grain": NONNEG, "vignette": NONNEG}
ENCODE = {"preset": X264_PRESETS, "crf": number(lo=0, hi=51), "cq": number(lo=0, hi=51), "lossless": _boolean}
AUDIO = {"clips": _anything, "voice_chain": _boolean, "duck_db": _opt(number(lo=0, hi=60)),     # null: no ducking
         "loudness": number(lo=-40, hi=0), "true_peak": number(lo=-12, hi=0)}


def _keyframes(r, w, v, check):
    """A number, or [t, v] / [t, v, ease] / {t, v, ease} keyframes (t from the clip's start)."""
    if _num(v):
        msg = check(v)
        if msg:
            r.err(w, f"{msg} (got {_show(v)})")
        return
    if not isinstance(v, list) or not v:
        r.err(w, f"a number or a non-empty keyframe list [[t, v, ease?], …] is required (got {_show(v)})")
        return
    last = None
    for j, k in enumerate(v):
        kw = f"{w}[{j}]"
        if isinstance(k, list) and len(k) in (2, 3):
            t, val, e = k[0], k[1], k[2] if len(k) == 3 else None
        elif isinstance(k, dict) and "t" in k and "v" in k and set(k) <= {"t", "v", "ease"}:
            t, val, e = k["t"], k["v"], k.get("ease")
        else:
            r.err(kw, f"a keyframe is [t, v], [t, v, ease] or {{t, v, ease}} (got {_show(k)})")
            continue
        if not _num(t):
            r.err(kw, f"time must be a finite number (got {_show(t)})")
        elif last is not None and t < last:
            r.err(kw, f"time {t:g} comes before the previous keyframe's {last:g}")
        else:
            last = t
        msg = check(val)
        if msg:
            r.err(kw, f"value {msg} (got {_show(val)})")
        msg = _ease(e)
        if msg:
            r.err(kw, f"ease {msg}")


def _video(r, video, media, dur, frame):
    base = []
    for i, c in enumerate(video):
        w = f"video[{i}]"
        if not isinstance(c, dict):
            r.err(w, f"must be an object (got {_show(c)})")
            continue
        _unknown(r, w, c, CLIP_KEYS, what=" (an effect the renderer does not implement? use meta for notes)")
        m = c.get("media")
        if not isinstance(m, str) or m not in media:
            r.err(f"{w}.media", f"{_show(m)} is not a key of the timeline's media")
        ok = True
        for k, check, need in (("start", NONNEG, True), ("dur", POS, True), ("in", NONNEG, False)):
            if k in c or need:
                msg = check(c.get(k))
                if msg:
                    r.err(f"{w}.{k}", f"{msg} (got {_show(c.get(k))})")
                    ok = False
        for k, check in (("fit", FITS), ("speed", number(gt=0, hi=16)), ("track", number(lo=1, integer=True)),
                         ("hold_last", _boolean), ("kind", ("video", "image")), ("meta", _object)):
            if k in c:
                _check(r, f"{w}.{k}", c[k], check)
        for k, check in (("scale", POS), ("x", NUM), ("y", NUM), ("rotation", NUM), ("opacity", UNIT)):
            if k in c:
                _keyframes(r, f"{w}.{k}", c[k], check)
        tin = c.get("transition_in")
        track = c.get("track", 1)
        if tin is not None:
            if not isinstance(tin, dict):
                r.err(f"{w}.transition_in", f"null or {{type, dur}} is required (got {_show(tin)})")
            else:
                _check(r, f"{w}.transition_in", tin, {"type": TRANSITIONS, "dur": NONNEG})
                if "type" not in tin:
                    r.err(f"{w}.transition_in", "type is required")
                elif tin.get("type") != "cut" and not (_num(tin.get("dur")) and tin["dur"] > 0):
                    r.warn(f"{w}.transition_in", f"{tin.get('type')} without a positive dur draws as a plain cut")
                if track != 1:
                    r.warn(f"{w}.transition_in", "only track-1 clips transition; ignored on an overlay")
        if ok:
            end = c["start"] + c["dur"]
            if dur is not None and end > dur + frame:
                r.warn(w, f"ends at {end:.3f} s, after the timeline ({dur:g} s)")
            if track == 1:
                base.append((c["start"], end, i))
    base.sort()
    t = 0.0
    for s, e, i in base:
        if s > t + frame:
            r.warn(f"video[{i}]", f"starts {s - t:.3f} s after the previous track-1 clip ends: background shows")
        elif s < t - frame:
            r.warn(f"video[{i}]", f"overlaps the previous track-1 clip by {t - s:.3f} s")
        t = max(t, e)
    if base and dur is not None and t < dur - frame:
        r.warn("video", f"track 1 ends at {t:.3f} s, before the timeline ({dur:g} s): background shows")
    if not base:
        r.warn("video", "no track-1 clip: every frame is the background")


def _events(r, name, evs, styles, dur):
    if not isinstance(evs, list):
        r.err(name, f"a list of {{s, e, text, words?, style?}} is required (got {_show(evs)})")
        return
    for i, ev in enumerate(evs):
        w = f"{name}[{i}]"
        if not isinstance(ev, dict):
            r.err(w, f"must be an object (got {_show(ev)})")
            continue
        _unknown(r, w, ev, EVENT_KEYS, what=" (styling belongs in the event's style)")
        s, e = ev.get("s"), ev.get("e")
        if not (_num(s) and _num(e) and 0 <= s < e):
            r.err(w, f"needs 0 ≤ s < e in seconds, got {_show(s)}..{_show(e)}")
            s = e = None
        elif dur is not None and s >= dur:
            r.warn(w, f"starts at {s:g} s, after the timeline ends ({dur:g} s)")
        _check(r, f"{w}.text", ev.get("text"), _string)
        if isinstance(ev.get("text"), str) and not ev["text"].strip():
            r.warn(f"{w}.text", "empty: nothing is drawn")
        st = ev.get("style")
        if isinstance(st, dict):
            _text_style(r, f"{w}.style", st)
        elif isinstance(st, str):
            if st not in styles:
                r.err(f"{w}.style", f"{st!r} is not a key of styles ({', '.join(styles) or 'none'})")
        elif st is not None:
            r.err(f"{w}.style", f"a style name or a style object is required (got {_show(st)})")
        if "meta" in ev:
            _check(r, f"{w}.meta", ev["meta"], _object)
        words = ev.get("words")
        if words is None:
            continue
        if not isinstance(words, list):
            r.err(f"{w}.words", "a list of {w, s, e, emph?} is required")
            continue
        last = None
        for j, x in enumerate(words):
            ww = f"{w}.words[{j}]"
            if not (isinstance(x, dict) and isinstance(x.get("w"), str) and _num(x.get("s")) and _num(x.get("e"))):
                r.err(ww, f"{{w: text, s, e}} with numeric times is required (got {_show(x)})")
                continue
            _unknown(r, ww, x, ("w", "s", "e", "emph"), strict=False, what=" (ignored)")
            if "emph" in x:
                _check(r, f"{ww}.emph", x["emph"], _boolean)
            if x["e"] < x["s"]:
                r.err(ww, f"ends ({x['e']:g}) before it starts ({x['s']:g})")
            if last is not None and x["s"] < last - 1e-3:
                r.err(ww, f"starts at {x['s']:g} s, before the previous word ({last:g} s): words must be in order")
            last = x["s"] if last is None else max(last, x["s"])
            if s is not None and (x["s"] < s - 1.0 or x["s"] > e + 1.0):
                r.warn(ww, f"spoken at {x['s']:g} s, outside its caption ({s:g}–{e:g} s)")


def _audio(r, au, media, dur):
    if not isinstance(au, dict):
        r.err("audio", f"must be an object (got {_show(au)})")
        return
    _check(r, "audio", au, AUDIO, strict=True)
    clips = au.get("clips", [])
    if not isinstance(clips, list):
        r.err("audio.clips", "a list is required")
        return
    for i, c in enumerate(clips):
        w = f"audio.clips[{i}]"
        if not isinstance(c, dict):
            r.err(w, f"must be an object (got {_show(c)})")
            continue
        _unknown(r, w, c, AUDIO_CLIP_KEYS, what=" (an audio effect the renderer does not implement?)")
        m = c.get("media")
        if not isinstance(m, str) or m not in media:
            r.err(f"{w}.media", f"{_show(m)} is not a key of the timeline's media")
        ok = True
        for k, check, need in (("start", NONNEG, True), ("dur", POS, True), ("in", NONNEG, False)):
            if k in c or need:
                msg = check(c.get(k))
                if msg:
                    r.err(f"{w}.{k}", f"{msg} (got {_show(c.get(k))})")
                    ok = False
        for k, check in (("role", AUDIO_ROLES), ("mono", _boolean), ("gain_db", number(hi=24)),
                         ("fade_in", NONNEG), ("fade_out", NONNEG), ("meta", _object)):
            if k in c:
                _check(r, f"{w}.{k}", c[k], check)
        if ok:
            fades = sum(c[k] for k in ("fade_in", "fade_out") if _num(c.get(k)) and c[k] > 0)
            if fades > c["dur"] + 1e-6:
                r.warn(w, f"fades ({fades:g} s) are longer than the clip ({c['dur']:g} s)")
            if dur is not None and c["start"] >= dur:
                r.warn(w, f"starts at {c['start']:g} s, after the timeline ends ({dur:g} s)")


def validate_timeline(tl):
    """A timeline (render.py's module docstring): geometry, media references, clip / event /
    audio timings, keyframes, transitions, styles, grade and encode settings."""
    r = _Report()
    if not isinstance(tl, dict):
        r.err("timeline", "a JSON object is required")
        return r.out()
    _unknown(r, "timeline", tl, TIMELINE_KEYS, strict=False, what=" (ignored by the renderer)")
    for k in ("width", "height"):
        _check(r, k, tl.get(k), _EVEN)
    fps = tl.get("fps")
    _check(r, "fps", fps, number(gt=0, hi=240))
    frame = 1.0 / fps if _num(fps) and fps > 0 else 1 / 30
    dur = tl.get("duration")
    if not (_num(dur) and dur > 0):
        r.err("duration", f"a positive number of seconds is required (got {_show(dur)})")
        dur = None
    elif dur > 900:
        r.warn("duration", f"{dur:g} s is long for a reel")
    if "background" in tl:
        _check(r, "background", tl["background"], _color)
    media = tl.get("media", {})
    if not isinstance(media, dict):
        r.err("media", "an object {key: path or asset:<id>} is required")
        media = {}
    for k, v in media.items():
        if _nonempty(v):
            r.err(f"media.{k}", f"a path or asset:<id> is required (got {_show(v)})")
    video = tl.get("video", [])
    if not isinstance(video, list):
        r.err("video", "a list of clips is required")
    else:
        _video(r, video, media, dur, frame)
    styles = tl.get("styles", {})
    if not isinstance(styles, dict):
        r.err("styles", "an object {name: captions.py style} is required")
        styles = {}
    for k, st in styles.items():
        _text_style(r, f"styles.{k}", st)
    for k in ("captions", "text"):
        if k in tl:
            _events(r, k, tl[k], styles, dur)
    if tl.get("grade") is not None:
        _check(r, "grade", tl["grade"], GRADE, strict=True)
    if tl.get("audio") is not None:
        _audio(r, tl["audio"], media, dur)
    if tl.get("encode") is not None:
        _check(r, "encode", tl["encode"], ENCODE, strict=True)
    if "meta" in tl and not isinstance(tl["meta"], dict):
        r.warn("meta", "an object is expected")
    return r.out()


# ------------------------------------------------------------------ review

def _text(v):
    return "" if v is None else str(v)[:TEXT_CHARS]


def _finding(f, decision, where, errors):
    if not isinstance(f, dict):
        errors.append(f"{where}: a finding must be an object (got {_show(f)})")
        return None
    sev = f.get("severity")
    norm = sev.strip().lower() if isinstance(sev, str) else None
    if norm not in SEVERITIES:
        norm = "blocking" if decision == "fix_required" else "minor"
        if sev is not None:
            errors.append(f"{where}: severity {_show(sev)} is not blocking | minor; recorded as {norm}")
    code = f.get("code")
    out = {"code": str(code).strip()[:64] if code not in (None, "") else "UNSPECIFIED", "severity": norm,
           "what": _text(f.get("what")), "how": _text(f.get("how"))}
    if f.get("scope") is not None:
        out["scope"] = f["scope"]
    return out


def _result(x, where, want, plan_generation, errors):
    """One checked {reel_id, decision, findings, render_sha256?}, or None (reason in errors)."""
    if not isinstance(x, dict):
        errors.append(f"{where}: must be an object (got {_show(x)})")
        return None
    rid = x.get("reel_id", x.get("id"))
    if not isinstance(rid, str) or not rid:
        errors.append(f"{where}: reel_id is required")
        return None
    where = f"reel {rid}"
    if rid not in want:
        errors.append(f"{where}: not assigned to this review (unknown id)")
        return None
    pg = x.get("plan_generation")
    if pg is not None and plan_generation is not None and pg != plan_generation:
        errors.append(f"{where}: plan_generation {_show(pg)} is not the current {plan_generation}: stale result")
        return None
    a = want[rid]
    dec = x.get("decision")
    if dec not in DECISIONS:
        errors.append(f"{where}: decision {_show(dec)} is not accept | fix_required")
        return None
    sha = x.get("render_sha256")
    if sha in ("", None):
        sha = None
    elif not isinstance(sha, str) or sha.strip().lower() != str(a.get("render_sha256") or "").lower():
        errors.append(f"{where}: render_sha256 {_show(sha, 20)} is not the assigned render "
                      f"{_show(a.get('render_sha256'), 20)}: a stale review of another render")
        return None
    else:
        sha = sha.strip().lower()
    found = x.get("findings")
    if found is None:
        found = []
    if not isinstance(found, list):
        errors.append(f"{where}: findings must be a list")
        return None
    if len(found) > MAX_FINDINGS:
        errors.append(f"{where}: {len(found)} findings; the first {MAX_FINDINGS} are kept")
    findings = [f for f in (_finding(f, dec, f"{where} finding {j}", errors)
                            for j, f in enumerate(found[:MAX_FINDINGS])) if f]
    blocking = [f["code"] for f in findings if f["severity"] == "blocking"]
    qc = str(a.get("qc_verdict") or "").lower()
    if dec == "accept":
        if qc in QC_FAILED:
            errors.append(f"{where}: accept of a reel whose technical QC is {qc}: a technical failure cannot be "
                          "waived (fix_required with the failing check)")
            return None
        if blocking:
            errors.append(f"{where}: accept with blocking finding(s) {', '.join(blocking)}: decide fix_required or "
                          "mark them minor")
            return None
    elif not blocking:
        errors.append(f"{where}: fix_required without a blocking finding (recorded as fix_required)")
    out = {"reel_id": rid, "decision": dec, "findings": findings}
    if sha:
        out["render_sha256"] = sha
    return out


def _legacy(raw, errors):
    """{"fixes": [{reel, what, how}], "ok": [ids]} as results: each fix a blocking finding."""
    items = {}
    fixes = raw.get("fixes")
    if fixes is None:
        fixes = []
    if not isinstance(fixes, list):
        errors.append("fixes must be a list of {reel, what, how}")
        fixes = []
    for i, f in enumerate(fixes):
        if not isinstance(f, dict):
            errors.append(f"fixes[{i}]: must be an object")
            continue
        rid = f.get("reel", f.get("reel_id", f.get("id")))
        if not isinstance(rid, str) or not rid:
            errors.append(f"fixes[{i}]: reel is required")
            continue
        items.setdefault(rid, {"reel_id": rid, "decision": "fix_required", "findings": []})["findings"].append(
            {"code": f.get("code") or "LEGACY_FIX", "severity": f.get("severity") or "blocking",
             "what": f.get("what"), "how": f.get("how"), **({"scope": f["scope"]} if f.get("scope") else {})})
    ok = raw.get("ok")
    if isinstance(ok, str):
        ok = [x.strip() for x in ok.split(",") if x.strip()]
    if ok is None:
        ok = []
    if not isinstance(ok, list):
        errors.append("ok must be a list of reel ids")
        ok = []
    return list(items.values()) + [{"reel_id": rid, "decision": "accept", "findings": []} for rid in ok]


def normalize_review(raw, assigned, plan_generation):
    """One checked decision per assigned reel. raw: {"results": [{reel_id, render_sha256?, decision,
    findings: [{code, severity, what, how, scope?}]}], "notes", schema_version, job_id, need_id,
    expected_revision, plan_generation} (other top-level keys ignored) or the legacy {fixes, ok};
    assigned: [{id, render_sha256, qc_verdict}]. Unknown ids, stale render hashes or generations,
    malformed results, conflicting decisions and accepting a technically failed reel are errors and
    leave the reel unreviewed; results with the same decision for one reel are merged."""
    want, order = {}, []
    for a in assigned or []:
        if isinstance(a, dict) and isinstance(a.get("id"), str) and a["id"] not in want:
            want[a["id"]] = a
            order.append(a["id"])
    errors, results = [], []

    def done():
        got = {x["reel_id"] for x in results}
        return {"results": results, "errors": errors, "unreviewed": [i for i in order if i not in got]}
    if not isinstance(raw, dict):
        errors.append("the review result must be a JSON object")
        return done()
    sv = raw.get("schema_version")
    if sv is not None and (isinstance(sv, bool) or sv != REVIEW_SCHEMA_VERSION):
        errors.append(f"schema_version {_show(sv)}: this version reads {REVIEW_SCHEMA_VERSION}")
    pg = raw.get("plan_generation")
    if pg is not None and plan_generation is not None and pg != plan_generation:
        errors.append(f"plan_generation {_show(pg)} is not the current {plan_generation}: stale result")
        return done()
    if "results" in raw:
        items = raw["results"]
        if not isinstance(items, list):
            errors.append("results must be a list")
            return done()
        if "fixes" in raw or "ok" in raw:
            errors.append("legacy fixes / ok ignored: results are given")
    elif "fixes" in raw or "ok" in raw:
        items = _legacy(raw, errors)
    else:
        errors.append('no results: {"results": [{reel_id, decision, findings, render_sha256}]} is required')
        return done()
    by = {}
    for i, x in enumerate(items):
        res = _result(x, f"results[{i}]", want, plan_generation, errors)
        if res:
            by.setdefault(res["reel_id"], []).append(res)
    for rid in order:
        group = by.get(rid)
        if not group:
            continue
        if len(group) > 1:
            decisions = sorted({g["decision"] for g in group})
            if len(decisions) > 1:
                errors.append(f"reel {rid}: conflicting decisions {' and '.join(decisions)}: neither is recorded")
                continue
            errors.append(f"reel {rid}: {len(group)} {decisions[0]} results merged into one")
            merged = dict(group[0], findings=[])
            for g in group:
                merged["findings"] += [f for f in g["findings"] if f not in merged["findings"]]
                if g.get("render_sha256"):
                    merged["render_sha256"] = g["render_sha256"]
            group = [merged]
        results.append(group[0])
    return done()
