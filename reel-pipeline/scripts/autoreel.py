"""Raw clip → finished reel in one command, in a job's sample style.

    python3 scripts/autoreel.py <raw-video> <style.json> <out.mp4>

1. In parallel: a 720p/30 edit copy of the raw (prep.sh) and a word-timed
   transcript (transcribe.py).
2. Cuts: one segment per sentence, pauses trimmed to `pad`; a sentence still
   running when the file ends (cut off mid-word) is dropped.
3. Captions in the style's look: short phrases from the spoken words
   (≤ max_words, ≤ max_seconds); the style's `hook` replaces captions over the
   first hook_seconds, its `cta` over the last cta_seconds.
4. Jump zooms: sentences alternate between the style's `zooms`.
5. Renders with fast_render.py (720p, −14 LUFS) and writes the timeline next
   to the output (<out>.project.json) so any reel can be tweaked and re-rendered.

style.json (all keys optional):
  {"hook": "...", "cta": "...", "caption": {FableCut text props}, "zooms": [1.0, 1.25],
   "pad": 0.08, "max_words": 4, "max_seconds": 1.8, "hook_seconds": 2.0, "cta_seconds": 2.5}
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(HERE, "..", "workspace")

CAPTION = {"font": "Arimo", "weight": 700, "fontSize": 56, "color": "#ffffff", "bgColor": "#004B06",
           "bgOpacity": 1, "textShadow": 0, "strokeWidth": 0, "lineHeight": 1.25, "align": "center", "y": 80}


def log(msg, t0=[time.monotonic()]):
    print(f"[autoreel {time.monotonic() - t0[0]:5.1f}s] {msg}", file=sys.stderr, flush=True)


def sentences(words):
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None or w["word"][-1:] in ".?!" or nxt["start"] - w["end"] > 0.6:
            out.append(cur)
            cur = []
    return out


WEAK = {"the", "a", "an", "of", "to", "and", "on", "in", "at", "with", "for", "that", "this", "is", "are",
        "we", "you", "you'll", "it", "as", "or", "but", "these", "all", "have", "be", "by", "from", "our", "your"}


def bare(w):
    return w["word"].lower().strip(".,!?").rstrip("s")


def phrases(words, max_words, max_seconds):
    """Caption chunks that end on a content word, never on 'the' / 'of' / 'and'."""
    words = [w for i, w in enumerate(words) if i + 1 >= len(words) or bare(w) != bare(words[i + 1])]  # stumbles
    out, cur = [], []
    for w in words:
        full = len(cur) >= max_words or (cur and w["end"] - cur[0]["start"] > max_seconds)
        if full and cur[-1]["word"].lower().strip(".,!?") not in WEAK:
            out.append(cur)
            cur = []
        elif full and len(cur) > 1:  # hand the trailing weak words to the next chunk
            k = len(cur)
            while k > 1 and cur[k - 1]["word"].lower().strip(".,!?") in WEAK:
                k -= 1
            out.append(cur[:k])
            cur = cur[k:]
        cur.append(w)
        if w["word"][-1:] in ".?!,":
            out.append(cur)
            cur = []
    out += [cur] if cur else []
    merged = []  # a chunk of nothing but filler words joins the next one
    for ch in out:
        if merged and all(w["word"].lower().strip(".,!?") in WEAK for w in merged[-1]):
            merged[-1] = merged[-1] + ch
        else:
            merged.append(ch)
    return merged


def caption_text(ws):
    text = " ".join(w["word"] for w in ws).strip().rstrip(".,")
    return text[:1].upper() + text[1:]


def main():
    raw, style_path, out = sys.argv[1:4]
    style = json.load(open(style_path))
    name = os.path.splitext(os.path.basename(out))[0]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", raw],
                               capture_output=True, text=True, check=True).stdout)
    work = os.path.join(os.path.dirname(os.path.abspath(out)), f"{name}.work")
    os.makedirs(work, exist_ok=True)
    transcript = os.path.join(work, "transcript.json")

    log("prep 720p copy ∥ transcribe")
    with ThreadPoolExecutor(2) as pool:
        prep = pool.submit(subprocess.run, [os.path.join(HERE, "prep.sh"), raw, name],
                           capture_output=True, text=True, check=True)
        if not os.path.exists(transcript):
            subprocess.run([sys.executable, os.path.join(HERE, "transcribe.py"), raw, transcript],
                           check=True, capture_output=True)
        log("transcript ready")
        src = prep.result().stdout.strip().splitlines()[-1]
    log(f"edit copy ready: {src}")

    words = json.load(open(transcript))["words"]
    sents = sentences(words)
    if sents and sents[-1][-1]["end"] > dur - 0.5:
        sents = sents[:-1]  # cut off mid-sentence by the end of the recording
    pad, zooms = style.get("pad", 0.08), style.get("zooms", [1.0, 1.25])
    cap = dict(CAPTION, **style.get("caption", {}))

    clips, caps, t = [], [], 0.0
    for i, ws in enumerate(sents):
        a = max(0.0, ws[0]["start"] - pad)
        b = min(dur, ws[-1]["end"] + pad)
        if i + 1 < len(sents):
            b = min(b, sents[i + 1][0]["start"] - 0.02)
        d = round(b - a, 3)
        z = zooms[i % len(zooms)]
        clips.append({"id": f"v{i}", "mediaId": "m", "kind": "video", "track": "V1", "start": round(t, 3), "in": a,
                      "duration": d, "props": {"fit": "cover", "scale": z, "volume": 0}})
        clips.append({"id": f"a{i}", "mediaId": "m", "kind": "audio", "track": "A1", "start": round(t, 3), "in": a,
                      "duration": d, "props": {"volume": 1, "channelMode": "mono"}})
        for ph in phrases(ws, style.get("max_words", 4), style.get("max_seconds", 1.8)):
            caps.append([t + ph[0]["start"] - a, t + ph[-1]["end"] - a, caption_text(ph)])
        t += d
    total = t
    # captions run until the next one starts (no flicker between phrases)
    for c, nxt in zip(caps, caps[1:]):
        c[1] = max(c[1], nxt[0] - 0.001) if nxt[0] - c[1] < 0.4 else c[1]
    hook_s, cta_s = style.get("hook_seconds", 2.0), style.get("cta_seconds", 2.5)
    if style.get("hook"):
        caps = [c for c in caps if c[0] >= hook_s]
        caps.insert(0, [0, hook_s, style["hook"]])
    if style.get("cta"):
        caps = [c for c in caps if c[1] <= total - cta_s] + [[total - cta_s, total, style["cta"]]]
    for i, (s, e, text) in enumerate(caps):
        s = max(s, caps[i - 1][1] if i else 0)
        if e - s > 0.05:
            clips.append({"id": f"t{i}", "mediaId": None, "kind": "text", "track": "V3", "start": round(s, 3),
                          "duration": round(e - s, 3), "props": dict(cap, text=text)})

    project = {"name": name, "width": 720, "height": 1280, "fps": 30, "revision": 1, "panSchema": 1,
               "background": "#000000", "media": [{"id": "m", "name": os.path.basename(src), "kind": "video", "src": src}],
               "clips": clips}
    pj = os.path.splitext(out)[0] + ".project.json"
    json.dump(project, open(pj, "w"), indent=1)
    log(f"timeline: {len(sents)} sentences, {sum(c['kind'] == 'text' for c in clips)} captions, {total:.1f}s")
    subprocess.run([sys.executable, os.path.join(HERE, "fast_render.py"), pj, out, "--cpu-preset", "veryfast"],
                   check=True, capture_output=True)
    log(f"rendered {out}")


if __name__ == "__main__":
    main()
