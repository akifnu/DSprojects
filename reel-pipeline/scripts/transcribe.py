"""Word-timestamped transcript of a video/audio file (faster-whisper).

    python3 transcribe.py <media> <out.json> [--model small] [--lang en]

Writes {"language", "segments":[{start,end,text}], "words":[{word,start,end}]}.
The `words` array is the input format of FableCut's
examples/auto-captions/auto_captions.py.
"""
import argparse
import json

from faster_whisper import WhisperModel

ap = argparse.ArgumentParser()
ap.add_argument("media")
ap.add_argument("out")
ap.add_argument("--model", default="small")
ap.add_argument("--lang", default=None)
a = ap.parse_args()

model = WhisperModel(a.model, device="cpu", compute_type="int8")
segs, info = model.transcribe(a.media, language=a.lang, word_timestamps=True, vad_filter=True)
segments, words = [], []
for s in segs:
    segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
    for w in s.words or []:
        words.append({"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)})
    print(f"[{s.start:7.2f} → {s.end:7.2f}] {s.text.strip()}")
json.dump({"language": info.language, "segments": segments, "words": words}, open(a.out, "w"), indent=1)
print(f"{len(segments)} segments, {len(words)} words → {a.out}")
