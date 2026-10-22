"""The composition model: revision data, the AI rough cut, and compile() — the one mapping
from source time to composition time that both the browser preview and the final render use.

Edits are stored against stable word ids (source time), never composition time, so a cut
earlier in the reel cannot silently move captions, emphasis, overlays or sounds.
"""
import re

FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "uhm", "umm"}

DEFAULT_CAPTION = {"font": "Inter", "size": 74, "color": "#FFFFFF", "hl": "#FFD400", "outline": "#000000",
                   "outline_w": 5, "box": False, "box_color": "#000000", "y": 0.70, "upper": True,
                   "words": 3, "emph_italic": True, "weight": 800}
DEFAULT_LOOK = {"grade": "none", "zoom": 1.0}
DEFAULT_SETTINGS = {"w": 1080, "h": 1920, "fps": 30, "pad": 0.10, "max_gap": 0.40, "narration_db": 0.0}
GRADES = {"none", "warm", "cool", "punchy", "bw"}
ANIMS = {"none", "fade", "slide-left", "slide-up", "pop", "spin"}


def default_revision():
    return {"removed": [], "reasons": {}, "decisions": [], "text": {}, "emphasis": [],
            "caption": dict(DEFAULT_CAPTION), "caption_preset": None, "captions_on": True,
            "look": dict(DEFAULT_LOOK), "overlays": [], "sounds": [], "settings": dict(DEFAULT_SETTINGS)}


def _num(v, lo, hi, d):
    try:
        return min(hi, max(lo, float(v)))
    except (TypeError, ValueError):
        return d


def normalize(data, word_ids):
    """Validate client-supplied revision data into a well-formed snapshot."""
    base = default_revision()
    ids = set(word_ids)
    out = {
        "removed": [i for i in dict.fromkeys(data.get("removed", [])) if i in ids],
        "reasons": {k: str(v)[:80] for k, v in (data.get("reasons") or {}).items() if k in ids},
        "decisions": list(data.get("decisions", []))[-500:],
        "text": {k: str(v)[:60] for k, v in (data.get("text") or {}).items() if k in ids and str(v).strip()},
        "emphasis": [i for i in dict.fromkeys(data.get("emphasis", [])) if i in ids],
        "captions_on": bool(data.get("captions_on", True)),
        "caption_preset": data.get("caption_preset"),
    }
    cap = dict(base["caption"], **(data.get("caption") or {}))
    cap["size"] = _num(cap["size"], 20, 200, 74)
    cap["outline_w"] = _num(cap["outline_w"], 0, 20, 5)
    cap["y"] = _num(cap["y"], 0.08, 0.92, 0.7)
    cap["words"] = int(_num(cap["words"], 1, 12, 3))
    cap["weight"] = int(_num(cap.get("weight", 800), 100, 900, 800))
    for k in ("color", "hl", "outline", "box_color"):
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(cap[k])):
            cap[k] = base["caption"][k]
    cap["font"] = re.sub(r"[^\w \-]", "", str(cap["font"]))[:40] or "Inter"
    out["caption"] = cap
    look = dict(base["look"], **(data.get("look") or {}))
    look["grade"] = look["grade"] if look["grade"] in GRADES else "none"
    look["zoom"] = _num(look["zoom"], 1.0, 1.6, 1.0)
    out["look"] = look
    st = dict(base["settings"], **(data.get("settings") or {}))
    st.update(w=1080, h=1920, fps=int(_num(st["fps"], 24, 60, 30)), pad=_num(st["pad"], 0, 0.5, 0.1),
              max_gap=_num(st["max_gap"], 0.1, 3, 0.4), narration_db=_num(st["narration_db"], -20, 10, 0))
    out["settings"] = st
    out["overlays"] = []
    for o in data.get("overlays", [])[:50]:
        out["overlays"].append({
            "id": str(o.get("id"))[:20], "asset": str(o.get("asset")), "w0": o.get("w0"), "w1": o.get("w1"),
            "x": _num(o.get("x"), 0, 1, 0.5), "y": _num(o.get("y"), 0, 1, 0.3), "scale": _num(o.get("scale"), 0.05, 1, 0.45),
            "anim_in": o.get("anim_in") if o.get("anim_in") in ANIMS else "none",
            "anim_out": o.get("anim_out") if o.get("anim_out") in ANIMS else "none",
            "mode": "broll" if o.get("mode") == "broll" else "overlay",
            "src_in": _num(o.get("src_in"), 0, 1e6, 0), "audio": bool(o.get("audio", False)),
            "note": str(o.get("note", ""))[:200]})
    out["sounds"] = []
    for s in data.get("sounds", [])[:50]:
        out["sounds"].append({
            "id": str(s.get("id"))[:20], "asset": str(s.get("asset")), "w0": s.get("w0"),
            "gain_db": _num(s.get("gain_db"), -40, 12, 0), "trim_in": _num(s.get("trim_in"), 0, 1e6, 0),
            "length": None if s.get("length") in (None, "") else _num(s.get("length"), 0.05, 3600, 1),
            "loop": bool(s.get("loop")), "fade_in": _num(s.get("fade_in"), 0, 10, 0),
            "fade_out": _num(s.get("fade_out"), 0, 10, 0), "duck": bool(s.get("duck", True))})
    return out


# ------------------------------------------------------------------ AI rough cut

def _norm(text):
    return re.sub(r"[^\w']", "", text.lower())


def rough_cut(transcript, rev):
    """Heuristic proposal (the default provider): drop filler words and earlier takes of a
    line that is repeated within the next few sentences; pauses are tightened by compile().
    Returns new revision data; chronology is preserved and nothing is deleted."""
    words, sents = transcript["words"], transcript["sentences"]
    out = dict(rev, removed=list(rev.get("removed", [])), reasons=dict(rev.get("reasons", {})),
               decisions=list(rev.get("decisions", [])))
    removed = set(out["removed"])

    def drop(ids, reason):
        ids = [i for i in ids if i not in removed]
        if not ids:
            return
        removed.update(ids)
        for i in ids:
            out["reasons"][i] = reason
        out["decisions"].append({"op": "remove", "ids": ids, "actor": "ai", "reason": reason})

    toks = [[t for k in range(s["w0"], s["w1"] + 1) if (t := _norm(words[k]["w"])) not in FILLERS]
            for s in sents]
    for i, s in enumerate(sents):
        a = [t for t in toks[i] if t]
        if len(a) < 3:
            continue
        for j in range(i + 1, min(i + 4, len(sents))):
            b = [t for t in toks[j] if t]
            if len(b) < 3:
                continue
            same_open = a[:3] == b[:3]
            common = len(set(a) & set(b))
            # same opening, broadly the same words, or the earlier line is a fragment of the later one
            if same_open or common / len(set(a) | set(b)) >= 0.6 or common / len(set(a)) >= 0.8:
                drop([words[k]["id"] for k in range(s["w0"], s["w1"] + 1)], "repeated take")
                break
    drop([w["id"] for w in words if _norm(w["w"]) in FILLERS], "filler")
    out["removed"] = [w["id"] for w in words if w["id"] in removed]
    return out


# ------------------------------------------------------------------ compile

def compile(transcript, rev, duration, assets):
    """Source → composition mapping plus everything placed on it. Pure and deterministic."""
    words = transcript["words"] if transcript else []
    st = rev["settings"]
    pad, max_gap = st["pad"], st["max_gap"]
    removed = set(rev["removed"])
    idx = {w["id"]: i for i, w in enumerate(words)}
    issues = []

    # 1. kept source ranges
    segs = []
    if not words:
        segs = [[0.0, duration]]
    else:
        last = None  # index of the previous kept word
        for i, w in enumerate(words):
            if w["id"] in removed:
                continue
            if last == i - 1 and w["s"] - words[last]["e"] <= max_gap:
                segs[-1][1] = w["e"]
            else:
                segs.append([w["s"], w["e"]])
            last = i
        out = []
        for a, b in segs:
            lo = max(0.0, a - pad)
            hi = min(duration, b + pad)
            if out and lo < out[-1][1]:
                lo = out[-1][1]
            if hi - lo > 0.04:
                out.append([round(lo, 3), round(hi, 3)])
        segs = out
    t, segments = 0.0, []
    for a, b in segs:
        segments.append({"s0": a, "s1": b, "t0": round(t, 3)})
        t += b - a
    total = round(t, 3)

    def to_comp(src):
        for sg in segments:
            if sg["s0"] - 1e-6 <= src <= sg["s1"] + 1e-6:
                return round(sg["t0"] + src - sg["s0"], 3)
        return None

    # 2. kept words in composition time
    wt = {}
    for w in words:
        if w["id"] in removed:
            continue
        t0, t1 = to_comp(w["s"]), to_comp(w["e"])
        if t0 is not None and t1 is not None:
            wt[w["id"]] = [t0, t1]

    # 3. caption chunks (break on word count, sentence end, or a cut boundary)
    cap = rev["caption"]
    chunks, cur = [], []
    emph = set(rev["emphasis"])
    kept = [w for w in words if w["id"] in wt]
    for n, w in enumerate(kept):
        text = rev["text"].get(w["id"], w["w"])
        cur.append({"id": w["id"], "text": text.upper() if cap["upper"] else text, "emph": w["id"] in emph,
                    "t0": wt[w["id"]][0], "t1": wt[w["id"]][1]})
        nxt = kept[n + 1] if n + 1 < len(kept) else None
        if (nxt is None or len(cur) >= cap["words"] or re.search(r"[.!?]$", text)
                or wt[nxt["id"]][0] - wt[w["id"]][1] > 0.6):
            chunks.append(cur)
            cur = []
    captions = []
    for n, c in enumerate(chunks):
        t0 = c[0]["t0"]
        t1 = c[-1]["t1"] + 0.25
        if n + 1 < len(chunks):
            t1 = min(t1, chunks[n + 1][0]["t0"])
        captions.append({"t0": round(t0, 3), "t1": round(max(t1, t0 + 0.2), 3), "words": c})
    if not rev["captions_on"]:
        captions = []

    # 4. overlays and sounds anchored to words
    amap = {a["id"]: a for a in assets}

    def anchor(wid, edge):
        if wid is None:
            return 0.0 if edge == 0 else total
        if wid not in idx:
            return None
        return wt.get(wid, [None, None])[edge]

    overlays = []
    for o in rev["overlays"]:
        a = amap.get(o["asset"])
        t0, t1 = anchor(o["w0"], 0), anchor(o["w1"] if o["w1"] else o["w0"], 1)
        issue = None
        if a is None:
            issue = "missing media"
        elif t0 is None or t1 is None:
            issue = "anchor word was cut — re-anchor or remove"
        elif t1 <= t0:
            issue = "end word is before start word"
        if issue:
            issues.append({"kind": "overlay", "id": o["id"], "msg": issue})
        if a and a["type"] == "video" and t0 is not None and t1 is not None:
            room = max(0.0, (a.get("duration") or 0) - o["src_in"])
            if t1 - t0 > room:
                t1 = t0 + room
        overlays.append(dict(o, t0=t0, t1=t1, issue=issue, type=a and a["type"]))
    sounds = []
    for s in rev["sounds"]:
        a = amap.get(s["asset"])
        t0 = anchor(s["w0"], 0)
        issue = "missing sound" if a is None else ("anchor word was cut — re-anchor or remove" if t0 is None else None)
        length = None
        if a is not None and t0 is not None:
            avail = max(0.0, a["duration"] - s["trim_in"])
            length = s["length"] or (total - t0 if s["loop"] else avail)
            length = min(length, total - t0)
            if not s["loop"] and length > avail + 1e-3:
                issue = issue or "sound is shorter than its placement — trim, loop or replace"
                length = avail
        if issue:
            soft = issue.startswith("sound is shorter")
            issues.append({"kind": "sound-warn" if soft else "sound", "id": s["id"], "msg": issue})
        sounds.append(dict(s, t0=t0, t1=None if t0 is None or length is None else round(t0 + length, 3), issue=issue))

    # 5. removed groups for review in the transcript
    groups, g = [], None
    for w in words:
        if w["id"] in removed:
            if g is None:
                g = {"ids": [], "s": w["s"], "reason": rev["reasons"].get(w["id"], "cut")}
                groups.append(g)
            g["ids"].append(w["id"])
            g["e"] = w["e"]
        else:
            g = None
    for w in words:
        if w.get("uncertain") and (w["id"] in emph or any(o["w0"] == w["id"] for o in rev["overlays"])):
            issues.append({"kind": "word", "id": w["id"], "msg": f"“{w['w']}” has uncertain timing; check the anchor"})
    if total < 0.5:
        issues.append({"kind": "composition", "id": None, "msg": "The composition is empty — restore some content"})
    return {"segments": segments, "duration": total, "words": wt, "captions": captions, "overlays": overlays,
            "sounds": sounds, "groups": groups, "issues": issues}
