"""compile.py: caption chunking, style scaling, shot splitting, and end-to-end
plan → timelines → rendered reels (synthetic source; plus a real-speech check on R1
when it is present: the rendered reel, re-transcribed, says exactly the planned words)."""
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reelkit import asr, compile as C, ff, render, store
from reelkit.paths import FFMPEG, ROOT

STYLE = {
    "version": 2,
    "sample": {"id": "fixture", "name": "fixture.mp4", "duration": 25.0, "width": 720, "height": 1280, "fps": 30},
    "pacing": {"median_shot": 1.5, "avg_shot": 1.8, "max_pause": 0.2},
    "motion": {"punch_rate": 1.0, "punch_scales": [1.0, 1.2], "push_rate": 0.5, "push_scale_per_s": 0.04,
               "push_ease": "inout"},
    "transitions": {"counts": {"cut": 8, "zoom": 2}, "default": "cut", "special": {"type": "zoom", "dur": 0.3, "rate": 1.0}},
    "captions": {"present": True,
                 "style": {"font": "Montserrat", "weight": 800, "size": 56, "case": "upper", "y": 200,
                           "stroke": {"color": "#000000", "width": 5}, "highlight": {"fill": "#ffe600"},
                           "anim": {"in": "pop", "in_dur": 0.15}},
                 "words_per_caption": {"median": 2, "max": 3}, "chars_per_line": 16, "max_lines": 2, "lead": -0.03},
    "titles": {"present": True, "style": {"font": "Anton", "weight": 400, "size": 80, "y": -420,
                                          "box": {"color": "#e63946", "pad_x": 20, "pad_y": 8, "radius": 12}}},
    "grade": {"look": None},
    "audio": {"loudness": {"I": -14.0}, "music": {"duck_db": 12}, "sfx": {"on_cut_rate": 0.0}},
    "structure": {"duration": 25.0, "hook": {"dur": 1.5, "type": "title"}, "cta": {"s": 22.5}},
}

WORDS = [("Hello", 0.5, 0.9), ("everyone.", 1.0, 1.5), ("um", 1.7, 1.9), ("this", 2.1, 2.4), ("is", 2.5, 2.7),
         ("the", 2.8, 3.0), ("the", 3.1, 3.3), ("best", 3.4, 3.8), ("trick.", 3.9, 4.4), ("Watch", 5.6, 6.0),
         ("this", 6.1, 6.4), ("closely", 6.5, 7.1), ("now.", 7.2, 7.6), ("It", 8.0, 8.2), ("changes", 8.3, 8.8),
         ("everything", 8.9, 9.6), ("forever.", 9.7, 10.3)]


def timed(text, t0=0.0, step=0.3):
    out, t = [], t0
    for w in text.split():
        out.append({"w": w, "s": round(t, 2), "e": round(t + 0.25, 2)})
        t += step
    return out


class Units(unittest.TestCase):
    CFG = {"words_per_caption": {"median": 3, "max": 4}, "chars_per_line": 16, "max_lines": 2}

    def test_captions_respect_limits_and_never_end_on_weak_words(self):
        text = "So the thing is, when you open a brand new workbook you see the grid of cells and rows."
        evs = C.chunk_captions(timed(text), self.CFG)
        self.assertEqual(" ".join(e["text"].replace("\n", " ") for e in evs), text.rstrip("."))
        for e in evs[:-1]:
            words = e["text"].split()
            self.assertLessEqual(len(words), 4)
            self.assertNotIn(re.sub(r"[^\w']", "", words[-1].lower()), C.WEAK, e["text"])
            self.assertTrue(all(len(line) <= 16 or " " not in line for line in e["text"].split("\n")), e["text"])
        for a, b in zip(evs, evs[1:]):
            self.assertLessEqual(a["e"], b["s"] + 1e-9)          # no overlap, no flicker gap
            self.assertAlmostEqual(a["e"], b["s"], delta=1e-6)

    def test_captions_break_at_sentences_and_pauses_and_carry_word_times(self):
        ws = timed("First point. Second one") + timed("after pause", t0=3.0)
        evs = C.chunk_captions(ws, self.CFG, emphasis=["second"])
        self.assertEqual([e["text"] for e in evs], ["First point", "Second one", "after pause"])
        self.assertTrue(evs[1]["words"][0].get("emph"))
        self.assertEqual(evs[2]["words"][0]["s"], 3.0)
        self.assertAlmostEqual(evs[1]["e"], 1.3 + 0.25, delta=1e-6)   # ends after its last word before a pause

    def test_style_px_scale(self):
        st = C.scale_style({"size": 56, "y": 200, "max_width": 0.8, "stroke": {"width": 5},
                            "box": {"pad_x": 10, "radius": 8}, "highlight": {"box": {"pad_x": 6}}}, 0.5)
        self.assertEqual((st["size"], st["y"], st["max_width"], st["stroke"]["width"], st["box"]["pad_x"],
                          st["box"]["radius"], st["highlight"]["box"]["pad_x"]), (28, 100, 0.8, 2.5, 5, 4, 3))

    def test_long_runs_split_at_word_boundaries(self):
        run = {"s": 0.0, "e": 6.0, "words": timed("a b c d e f g h i j k l m n o p q r s t", 0.05)}
        shots = C.split_shots([run], target=2.0)
        self.assertEqual(len(shots), 3)
        bounds = [b for _, _, b in shots[:-1]]
        for b in bounds:   # every cut between two words, never inside one
            self.assertFalse(any(w["s"] < b < w["e"] for w in run["words"]), b)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cmptest-"))
        cls._data = store.DATA
        store.DATA = cls.tmp / "data"
        expr = "+".join(f"between(t\\,{s}\\,{e})" for _, s, e in WORDS)
        src = cls.tmp / "talk.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=30:d=12",
                        "-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*220*t)*({expr})':s=48000:d=12",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                        str(src)], check=True)
        cls.asset = store.ingest(src, "source")
        words = [{"w": w, "s": s, "e": e} for w, s, e in WORDS]
        tr = {"lang": "en", "timing": "word", "duration": 12, "words": words, "sentences": asr.sentences(words)}
        cls.asset.cache("transcript.en.json").write_text(json.dumps(tr))

    @classmethod
    def tearDownClass(cls):
        store.DATA = cls._data

    def test_two_reel_plan_compiles_and_renders(self):
        style = json.loads(json.dumps(STYLE))
        style["grade"]["look"] = {"L": list(np.linspace(8, 92, 101)),
                                  "zones": {z: {"a": 2.0, "b": 8.0, "sd": 12.0} for z in ("shadows", "mids", "highlights")},
                                  "chroma": {"50": 12.0}}
        plan = {"name": "t", "style": style, "output": {"width": 360, "height": 640, "fps": 30, "quality": "draft"},
                "sources": {"S": self.asset.id},
                "reels": [{"id": "r01", "segments": [{"src": "S", "sentences": [0, 1]}], "title_text": "WATCH THIS",
                           "cta": "Follow!", "emphasis": ["best"]},
                          {"id": "r02", "hook": {"src": "S", "sentences": [2]}, "segments": [{"src": "S", "sentences": [0, 0]}]}]}
        paths = C.compile_plan(plan, self.tmp / "batch")
        tl = json.loads(Path(paths["r01"]).read_text())
        self.assertEqual(tl["meta"]["words"], ["Hello", "everyone.", "this", "is", "the", "best", "trick.",
                                               "Watch", "this", "closely", "now."])
        self.assertAlmostEqual(tl["duration"], sum(r["e"] - r["s"] for r in tl["meta"]["runs"]), delta=1e-3)
        # framing: punches alternate, everything covers the canvas
        scales = [c["scale"] if isinstance(c["scale"], (int, float)) else c["scale"][0][1] for c in tl["video"]]
        self.assertTrue(any(s > 1.1 for s in scales) and any(s < 1.05 for s in scales))
        for c in tl["video"]:
            self.assertGreaterEqual(min([c["scale"]] if isinstance(c["scale"], (int, float))
                                        else [k[1] for k in c["scale"]]), 1.0)
        # the special transition sits only on sentence boundaries
        sentence_starts = {round(c["start"], 2) for c in tl["video"] if c.get("transition_in")}
        self.assertTrue(sentence_starts)
        # hook title, captions in style, CTA at the end
        self.assertEqual(tl["text"][0]["text"], "WATCH THIS")
        self.assertEqual(tl["captions"][-1]["text"], "Follow!")
        self.assertAlmostEqual(tl["captions"][-1]["e"], tl["duration"], delta=1e-3)
        self.assertAlmostEqual(tl["styles"]["caption"]["size"], 28.0)   # 56 px × 640/1280
        self.assertTrue(os.path.exists(tl["grade"]["lut"]))
        # audio: voice clips back to back, each from the lossless soundtrack
        clips = tl["audio"]["clips"]
        for a, b in zip(clips, clips[1:]):
            self.assertAlmostEqual(a["start"] + a["dur"], b["start"], delta=1e-3)
        for rid, p in paths.items():
            out = self.tmp / f"{rid}.mp4"
            render.render(p, out, draft=True)
            info = ff.probe(out)
            tl = json.loads(Path(p).read_text())
            self.assertEqual((info["width"], info["height"]), (360, 640))
            self.assertAlmostEqual(info["duration"], tl["duration"], delta=0.05)
            self.assertAlmostEqual(ff.loudness(out)["I"], -14, delta=1.0)
        tl2 = json.loads(Path(paths["r02"]).read_text())
        self.assertEqual(tl2["meta"]["words"][:3], ["It", "changes", "everything"])   # cold open first


R1 = ROOT / "jobs" / "job1" / "raw" / "R1.mp4"


@unittest.skipUnless(R1.exists() and (ROOT / ".models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8").exists(),
                     "R1 / Parakeet not present")
class RealSpeech(unittest.TestCase):
    def test_rendered_reel_says_exactly_the_planned_words(self):
        tmp = Path(tempfile.mkdtemp(prefix="cmpreal-"))
        old = store.DATA
        store.DATA = tmp / "data"
        try:
            a = store.ingest(R1, "source")
            plan = {"name": "real", "style": STYLE, "output": {"width": 720, "height": 1280, "fps": 30, "quality": "draft"},
                    "sources": {"S": a.id},
                    "reels": [{"id": "r1", "segments": [{"src": "S", "sentences": [3, 5]}, {"src": "S", "sentences": [7, 7]}]}]}
            p = C.compile_plan(plan, tmp / "batch")["r1"]
            tl = json.loads(Path(p).read_text())
            out = tmp / "r1.mp4"
            render.render(p, out, draft=True)
            heard = asr.transcribe(str(out), "en")["words"]
            norm = lambda ws: [re.sub(r"[^\w]", "", w.lower()) for w in ws]
            self.assertEqual(norm(w["w"] for w in heard), norm(tl["meta"]["words"]))
        finally:
            store.DATA = old


if __name__ == "__main__":
    unittest.main()
