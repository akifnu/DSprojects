"""Word-timestamped transcript of a video/audio file (English).

    python3 transcribe.py <media> <out.json>

Uses NVIDIA Parakeet TDT 0.6B v2 through sherpa-onnx (model fetched by
setup.sh from the sherpa-onnx GitHub release, since huggingface.co is often
not reachable from the cloud container). Long files are split on ~25 s
windows. Writes {"segments":[{start,end,text}], "words":[{word,start,end}]};
`words` is the input format of FableCut's examples/auto-captions/auto_captions.py.
"""
import json
import os
import subprocess
import sys

import numpy as np
import sherpa_onnx

MODEL = os.path.join(os.path.dirname(__file__), "..", ".models", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
SR = 16000
WINDOW = 25.0

media, out = sys.argv[1], sys.argv[2]
pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", media, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                     capture_output=True, check=True).stdout
audio = np.frombuffer(pcm, dtype=np.float32)

rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{MODEL}/encoder.int8.onnx", decoder=f"{MODEL}/decoder.int8.onnx",
    joiner=f"{MODEL}/joiner.int8.onnx", tokens=f"{MODEL}/tokens.txt",
    model_type="nemo_transducer", num_threads=os.cpu_count() or 4)

words = []
for off in range(0, len(audio), int(WINDOW * SR)):
    chunk = audio[off:off + int(WINDOW * SR)]
    s = rec.create_stream()
    s.accept_waveform(SR, chunk)
    rec.decode_stream(s)
    r = s.result
    t0 = off / SR
    for tok, ts in zip(r.tokens, r.timestamps):
        if tok.startswith("▁") or tok.startswith(" ") or not words:
            words.append({"word": tok.lstrip("▁ "), "start": round(t0 + ts, 2), "end": round(t0 + ts, 2)})
        else:
            words[-1]["word"] += tok
        words[-1]["end"] = round(t0 + ts + 0.08, 2)
words = [w for w in words if w["word"]]
for a, b in zip(words, words[1:]):  # a word lasts until the next one starts (capped)
    a["end"] = round(min(b["start"], a["start"] + 1.0), 2)

# segments: break after sentence punctuation or a pause > 0.6 s
segments, cur = [], []
for i, w in enumerate(words):
    cur.append(w)
    nxt = words[i + 1] if i + 1 < len(words) else None
    if nxt is None or w["word"][-1:] in ".?!" or nxt["start"] - w["end"] > 0.6:
        segments.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": " ".join(x["word"] for x in cur)})
        cur = []
for s in segments:
    print(f"[{s['start']:7.2f} → {s['end']:7.2f}] {s['text']}")
json.dump({"segments": segments, "words": words}, open(out, "w"), indent=1)
print(f"{len(segments)} segments, {len(words)} words → {out}")
