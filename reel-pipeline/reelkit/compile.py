"""plan + style + sources → one timeline.json per reel (see render.py for the format).

compile_reel(plan, reel, style, out_dir)  one reel
compile_plan(plan, out_root)              every reel; editing copies are made once per
                                          source for the union of all reels' ranges

What the style decides (all px values are scaled by output_height / sample.height):
  pacing      shot length: long runs are split at word boundaries into shots near the
              sample's median shot length (a punch / re-frame, audio continuous)
  motion      punch-in framing levels and how often they change; slow pushes
  transitions the special transition (zoom, flash, whip, dissolve …) at its rate, on
              jump cuts between runs
  captions    words per caption, chars per line, lines, case, lead, typography;
              a caption never ends on a weak word ("the", "of", "and" …)
  titles      hook title (if the plan gives title_text) and CTA styling
  grade       3D LUT from the source footage's look to the sample's look
  audio       loudness target, music ducking depth, SFX on cuts / caption appearances
The framing follows the speaker (reframe.py) and is clamped so the picture always
covers the canvas.
"""
import json
import math
import os
from pathlib import Path

from . import ff, grade, plan as planlib, reframe, render, source, store

WEAK = {"the", "a", "an", "of", "to", "and", "on", "in", "at", "with", "for", "that", "this", "is", "are", "was",
        "were", "we", "you", "i", "it", "as", "or", "but", "these", "those", "all", "have", "has", "be", "by",
        "from", "our", "your", "my", "his", "her", "their", "so", "if", "then", "than", "into", "about", "just",
        "very", "really", "will", "would", "can", "could", "do", "does", "did", "not", "no", "what", "which", "who",
        "when", "where", "how", "there", "here", "they", "he", "she", "me", "us", "them", "its", "it's", "you're",
        "we're", "i'm", "don't"}
END = (".", "?", "!", "…")

QUALITY = {"final": {"preset": "medium", "crf": 17}, "draft": {"preset": "veryfast", "crf": 20}}


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

def split_shots(runs, target):
    """Runs → shots [(run_index, src_s, src_e)], long runs cut at word boundaries."""
    shots = []
    for ri, r in enumerate(runs):
        d = r["e"] - r["s"]
        if d <= 1.6 * target or len(r["words"]) < 4:
            shots.append((ri, r["s"], r["e"]))
            continue
        n = max(1, int(round(d / target)))
        cuts, ws = [], r["words"]
        for j in range(1, n):
            want = r["s"] + d * j / n
            best = min(range(1, len(ws)), key=lambda q: abs(ws[q]["s"] - want))
            t = round((ws[best - 1]["e"] + ws[best]["s"]) / 2, 4)
            if (not cuts or t - cuts[-1] > 0.6) and t - r["s"] > 0.6 and r["e"] - t > 0.6:
                cuts.append(t)
        bounds = [r["s"]] + cuts + [r["e"]]
        shots += [(ri, a, b) for a, b in zip(bounds, bounds[1:])]
    return shots


def _every(rate):
    return 0 if not rate or rate <= 0 else max(1, int(round(1 / min(1.0, rate))))


# ------------------------------------------------------------------ compile

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


def compile_reel(plan, reel, style, out_dir, transcripts=None, copies=None, envelopes=None, reframing=True):
    """Write out_dir/timeline.json for one reel and return its path."""
    plan = planlib.load(plan)
    style = load_style(style)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    o = plan.get("output", {})
    W, H, fps = int(o.get("width", 720)), int(o.get("height", 1280)), int(o.get("fps", 30))
    k = H / float(_g(style, "sample", "height", default=H))
    lang = o.get("lang") or _g(style, "audio", "speech", "lang", default="en") or "en"
    assets = {key: store.get(ref) for key, ref in plan["sources"].items()}
    transcripts = transcripts or {key: source.transcript(a, lang) for key, a in assets.items()}
    envelopes = envelopes or {key: planlib.envelope(source.audio(a), cache=str(a.cache("envelope.npy")))
                              for key, a in assets.items() if a.info.get("has_audio")}
    max_pause = float(_g(style, "pacing", "max_pause", default=0.25) or 0.25)
    runs = planlib.resolve_reel(reel, transcripts, envelopes, max_pause=max_pause)
    if not runs:
        raise ValueError(f"reel {reel['id']}: nothing left to play")
    copies = copies or {}
    for key in {r["src"] for r in runs}:
        if key not in copies:
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
    shots = split_shots(runs, target)
    mot = style.get("motion") or {}
    levels = [float(x) for x in (mot.get("punch_scales") or [1.0])] or [1.0]
    base = min(levels)
    levels = [max(1.0, x / base) for x in levels]           # relative to the widest framing
    punch_every = _every(mot.get("punch_rate"))
    push_every = _every(mot.get("push_rate"))
    special = _g(style, "transitions", "special") or {}
    trans_every = _every(special.get("rate")) if special.get("type") not in (None, "cut") else 0
    media, video, run_cut_count, probes = {}, [], 0, {}
    level = 0
    for n, (ri, s, e) in enumerate(shots):
        r = runs[ri]
        m, off = source.locate(copies[r["src"]], s, e)
        mkey = media.setdefault(m["path"], f"v{len(media)}")
        start = round(starts[ri] + (s - r["s"]), 4)
        dur = round(e - s, 4)
        if n and punch_every and n % punch_every == 0:
            level = (level + 1) % len(levels)
        z = levels[level]
        clip = {"media": mkey, "start": start, "in": off, "dur": dur, "fit": "cover"}
        if push_every and n % push_every == 0 and dur > 0.8:
            z1 = z * (1 + float(mot.get("push_scale_per_s", 0.03)) * dur)
            clip["scale"] = [[0, round(z, 4)], [dur, round(z1, 4), mot.get("push_ease", "inout")]]
        else:
            clip["scale"] = round(z, 4)
        new_run = n > 0 and shots[n - 1][0] != ri
        sentence_end = new_run and runs[ri - 1]["words"] and runs[ri - 1]["words"][-1]["w"].endswith(END)
        if sentence_end:          # special transitions only between sentences, never inside one
            run_cut_count += 1
            if trans_every and run_cut_count % trans_every == 0:
                clip["transition_in"] = {"type": special["type"], "dur": float(special.get("dur", 0.3))}
        scale_at = (lambda sc: (lambda tt: render.kf(sc, tt, 1.0)))(clip["scale"])
        if reframing:
            info = probes.setdefault(m["path"], ff.probe(m["path"]))
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
            cap_events.insert(0, {"s": 0.0, "e": round(hook_dur, 3), "text": reel["title_text"]})
        else:
            text_events.append({"s": 0.0, "e": round(hook_dur, 3), "text": reel["title_text"], "style": "title"})
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
        cap_events = kept + [{"s": cs, "e": round(total, 3), "text": reel["cta"]}]
    cap_events = [c for c in cap_events if c["e"] - c["s"] >= 0.1]

    # ---- audio
    clips = []
    for r, t0 in zip(runs, starts):
        akey = f"a_{r['src']}"
        if akey not in timeline_media:
            timeline_media[akey] = source.audio(assets[r["src"]])
        clips.append({"media": akey, "start": round(t0, 4), "in": r["s"], "dur": round(r["e"] - r["s"], 4),
                      "role": "voice", "mono": True})
    music = plan.get("music")
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
    types = sfx_cfg.get("types") or {}
    kind = max(types, key=types.get) if types else "whoosh"
    sfx_gain = float(sfx_cfg.get("gain_db", -10))
    cut_times = [c["start"] for c in video[1:]]
    sfx_hits = []
    ce = _every(sfx_cfg.get("on_cut_rate"))
    if ce:
        sfx_hits += [(t_, kind) for i, t_ in enumerate(cut_times) if (i + 1) % ce == 0]
    pe = _every(sfx_cfg.get("on_caption_rate"))
    if pe:
        sfx_hits += [(c["s"], "pop") for i, c in enumerate(cap_events) if i % pe == 0]
    for t_, kd in sfx_hits:
        p = _sfx_path(kd, plan)
        if not p:
            continue
        key = f"sfx_{kd}"
        timeline_media.setdefault(key, p)
        d = ff.probe(p)["duration"]
        st = t_ - (d / 2 if kd == "whoosh" else 0.0)
        if st < 0 or st + d > total + 0.5:
            continue
        clips.append({"media": key, "start": round(st, 3), "in": 0, "dur": round(min(d, total - st), 3),
                      "role": "sfx", "gain_db": sfx_gain})

    # ---- grade
    g = style.get("grade") or {}
    tl_grade = {key: g[key] for key in ("sharpen", "grain", "vignette") if g.get(key)}
    if g.get("look"):
        main = max({r["src"] for r in runs}, key=lambda s_: sum(r["e"] - r["s"] for r in runs if r["src"] == s_))
        src_look = source.look(assets[main], copies[main])
        lut = grade.build_lut(src_look, g["look"], strength=float(g.get("strength", 1.0)))
        grade.write_cube(lut, out_dir / "look.cube")
        tl_grade["lut"] = str((out_dir / "look.cube").resolve())

    loud = o.get("loudness") or _g(style, "audio", "loudness", "I", default=-14.0) or -14.0
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
    path = out_dir / "timeline.json"
    path.write_text(json.dumps(tl, indent=1, ensure_ascii=False))
    return str(path)


def compile_plan(plan, out_root, reels=None, reframing=True):
    """Compile every reel (or the listed ids) of a plan; returns {reel_id: timeline path}."""
    plan = planlib.load(plan)
    style = load_style(plan["style"])
    o = plan.get("output", {})
    lang = o.get("lang") or _g(style, "audio", "speech", "lang", default="en") or "en"
    assets = {key: store.get(ref) for key, ref in plan["sources"].items()}
    transcripts = {key: source.transcript(a, lang) for key, a in assets.items()}
    envelopes = {key: planlib.envelope(source.audio(a), cache=str(a.cache("envelope.npy")))
                 for key, a in assets.items() if a.info.get("has_audio")}
    max_pause = float(_g(style, "pacing", "max_pause", default=0.25) or 0.25)
    wanted = [r for r in plan["reels"] if reels is None or r["id"] in reels]
    ranges = {}
    for r in wanted:
        for run in planlib.resolve_reel(r, transcripts, envelopes, max_pause=max_pause):
            ranges.setdefault(run["src"], []).append((run["s"], run["e"]))
    copies = {key: source.prepare(assets[key], rs, fps=int(o.get("fps", 30))) for key, rs in ranges.items()}
    out = {}
    for r in wanted:
        out[r["id"]] = compile_reel(plan, r, style, Path(out_root) / "reels" / r["id"], transcripts=transcripts,
                                    copies=copies, envelopes=envelopes, reframing=reframing)
    return out
