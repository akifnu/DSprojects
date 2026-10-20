"""plan.json → the exact spans of source audio/video each reel uses.

A reel is a list of segments (sentence ranges or explicit times) plus an optional
cold-open hook. Resolving it:
  1. words of each segment (global word indices into the source transcript)
  2. fillers (um, uh, erm …), immediate repeated-word stumbles and `drop_words` removed
  3. kept words grouped into runs: a run ends where a word was removed or the pause
     between two words is longer than the style's `pacing.max_pause` (a jump cut)
  4. every run boundary is snapped to the quietest 5 ms of audio inside the gap next
     to it (never past the neighbouring word), so no syllable is clipped even when
     transcript timestamps are off by a few tens of ms
  5. caption text overrides (`caption_text`) replace the words shown, not the audio

resolve_reel() → [{"src", "s", "e", "words": [{"w", "s", "e", "si", "wi"}]}] in source
seconds, in playing order.
"""
import json
import os
import re

import numpy as np

from . import ff

FILLERS = {"um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "ahh", "hmm", "mm", "mhm"}
HOP = 0.005                 # envelope resolution (s)
HEAD, TAIL = 0.05, 0.09     # nominal lead-in / tail-out kept around speech (s)
SNAP = 0.06                 # extra search either side of the nominal boundary (s)
MIN_GAP_FOR_CUT = 0.03


def load(ref):
    plan = json.loads(open(ref).read()) if isinstance(ref, (str, os.PathLike)) else ref
    for key in ("style", "sources", "reels"):
        if key not in plan:
            raise ValueError(f"plan: missing '{key}'")
    ids = [r.get("id") for r in plan["reels"]]
    if any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("plan: every reel needs a unique id")
    for r in plan["reels"]:
        segs = r.get("segments") or []
        if not segs:
            raise ValueError(f"plan: reel {r['id']} has no segments")
        for seg in segs + ([r["hook"]] if r.get("hook") else []):
            if seg.get("src") not in plan["sources"]:
                raise ValueError(f"plan: reel {r['id']}: unknown source {seg.get('src')!r}")
            if "sentences" not in seg and not ("s" in seg and "e" in seg):
                raise ValueError(f"plan: reel {r['id']}: a segment needs 'sentences' or 's'/'e'")
    plan.setdefault("output", {})
    return plan


def bare(w):
    return re.sub(r"[^\w']+", "", w.lower())


def envelope(audio_path, cache=None, sr=8000):
    """RMS envelope (5 ms hop) of a soundtrack, cached as .npy."""
    if cache and os.path.exists(cache):
        return np.load(cache)
    a = ff.audio(audio_path, sr)
    hop = int(sr * HOP)
    n = len(a) // hop
    env = np.sqrt(np.mean(a[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12).astype(np.float32)
    if cache:
        np.save(cache, env)
    return env


def _snap(env, lo, hi):
    """Time of the quietest envelope sample in [lo, hi]."""
    if env is None or hi <= lo:
        return (lo + hi) / 2
    a, b = int(lo / HOP), max(int(lo / HOP) + 1, int(np.ceil(hi / HOP)))
    seg = env[max(0, a):min(len(env), b)]
    if not len(seg):
        return (lo + hi) / 2
    # among near-minimum samples prefer the one closest to the middle of the window
    m = seg.min()
    cand = np.where(seg <= m * 1.15 + 1e-6)[0]
    mid = (b - a) / 2
    k = cand[np.argmin(np.abs(cand - mid))]
    return round((max(0, a) + k) * HOP, 4)


def segment_words(tr, seg):
    words, sents = tr["words"], tr["sentences"]
    if "sentences" in seg:
        a, b = seg["sentences"][0], seg["sentences"][-1]
        return list(range(sents[a]["w0"], sents[b]["w1"] + 1))
    s, e = seg["s"], seg["e"]
    return [i for i, w in enumerate(words) if w["s"] >= s - 0.05 and w["e"] <= e + 0.05]


def _sentence_of(tr):
    out = {}
    for s in tr["sentences"]:
        for i in range(s["w0"], s["w1"] + 1):
            out[i] = (s["id"], i - s["w0"])
    return out


def clean(tr, idxs, drop=()):
    """Kept word indices and {index: reason} for removed ones."""
    words, sof = tr["words"], _sentence_of(tr)
    drop = {tuple(d) for d in drop}
    removed = {}
    for k, i in enumerate(idxs):
        b = bare(words[i]["w"])
        if b in FILLERS:
            removed[i] = "filler"
        elif sof.get(i) in drop:
            removed[i] = "dropped"
        elif k + 1 < len(idxs):
            j = idxs[k + 1]
            nb = bare(words[j]["w"])
            same = b == nb or (len(b) >= 3 and b.rstrip("s") == nb.rstrip("s"))
            if b and same and words[j]["s"] - words[i]["e"] < 0.8:
                removed[i] = "stumble"          # keep the second, usually cleaner, take
    return [i for i in idxs if i not in removed], removed


def runs(tr, kept, removed, env, max_pause, src):
    """Group kept words into continuous runs and snap their boundaries into silence."""
    words = tr["words"]
    out, cur = [], []
    for i in kept:
        if cur:
            p = cur[-1]
            gap = words[i]["s"] - words[p]["e"]
            if any(j in removed for j in range(p + 1, i)) or i != p + 1 and gap > MIN_GAP_FOR_CUT \
                    or gap > max_pause:
                out.append(cur)
                cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    res = []
    for r in out:
        first, last = words[r[0]], words[r[-1]]
        prev_e = words[r[0] - 1]["e"] if r[0] > 0 else 0.0
        next_s = words[r[-1] + 1]["s"] if r[-1] + 1 < len(words) else last["e"] + 1.0
        lo = max(prev_e, first["s"] - HEAD - SNAP)
        hi = max(lo, first["s"] - max(0.0, HEAD - SNAP))
        s = _snap(env, lo, min(hi, first["s"]))
        lo2 = min(last["e"] + max(0.0, TAIL - SNAP), next_s)
        hi2 = min(next_s, last["e"] + TAIL + SNAP)
        e = _snap(env, max(last["e"], lo2), max(lo2, hi2))
        res.append({"src": src, "s": round(max(0.0, s), 4), "e": round(max(e, last["e"]), 4), "idx": r})
    return res


def _override(tr, idxs, overrides):
    """Caption words per kept index, with `caption_text` sentence overrides applied."""
    words, sof = tr["words"], _sentence_of(tr)
    shown = {i: words[i]["w"] for i in idxs}
    by_sent = {}
    for i in idxs:
        by_sent.setdefault(sof[i][0], []).append(i)
    for sid, text in (overrides or {}).items():
        sid = int(sid)
        ids = by_sent.get(sid)
        if not ids:
            continue
        toks = text.split()
        if len(toks) == len(ids):
            for i, t in zip(ids, toks):
                shown[i] = t
        else:   # re-time the corrected words over the sentence's kept span, by length
            s, e = words[ids[0]]["s"], words[ids[-1]]["e"]
            w = np.array([len(t) + 1 for t in toks], float)
            edges = s + (e - s) * np.concatenate([[0], np.cumsum(w) / w.sum()])
            for i in ids:
                shown[i] = None
            shown[ids[0]] = [(t, round(edges[k], 3), round(edges[k + 1], 3)) for k, t in enumerate(toks)]
    return shown


def resolve_reel(reel, transcripts, envelopes=None, max_pause=0.25):
    """Runs for one reel (hook first). transcripts / envelopes: {src_key: ...}."""
    envelopes = envelopes or {}
    out = []
    segs = ([reel["hook"]] if reel.get("hook") else []) + reel["segments"]
    for n, seg in enumerate(segs):
        src = seg["src"]
        tr = transcripts[src]
        idxs = segment_words(tr, seg)
        kept, removed = clean(tr, idxs, reel.get("drop_words", ()))
        if not kept:
            continue
        shown = _override(tr, kept, reel.get("caption_text"))
        sof = _sentence_of(tr)
        for r in runs(tr, kept, removed, envelopes.get(src), max_pause, src):
            ws = []
            for i in r.pop("idx"):
                v = shown.get(i)
                w = tr["words"][i]
                if isinstance(v, list):
                    ws += [{"w": t, "s": s, "e": e, "si": sof[i][0], "wi": -1} for t, s, e in v]
                elif v is not None:
                    ws.append({"w": v, "s": w["s"], "e": w["e"], "si": sof[i][0], "wi": sof[i][1]})
            if seg.get("s") is not None and "sentences" not in seg:      # explicit times bound the run
                r["s"], r["e"] = max(r["s"], seg["s"] - 0.1), min(r["e"], seg["e"] + 0.1)
            r["words"] = ws
            r["hook"] = bool(reel.get("hook")) and n == 0
            out.append(r)
    return out
