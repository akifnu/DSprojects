"""QC gate: a clean render passes; every injected defect is caught.
Run: python3 -m unittest discover -s reel-pipeline/tests -v"""
import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

from reelkit import qc, render
from reelkit.paths import FFMPEG

D = Path(tempfile.mkdtemp(prefix="qctest-"))
W, H, FPS, DUR = 360, 640, 30, 4.0


def ff(*args):
    subprocess.run([FFMPEG, "-v", "error", "-y", *map(str, args)], check=True)


ff("-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:r={FPS}:d=8", "-c:v", "libx264", "-preset", "ultrafast",
   "-qp", "10", "-pix_fmt", "yuv420p", D / "src.mp4")
ff("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8", "-af", "volume=0.4", D / "tone.wav")

WORDS = [{"w": w, "s": round(i * 0.4, 3), "e": round(i * 0.4 + 0.4, 3)}
         for i, w in enumerate("this is a clean test reel with captions that match the speech".split())]
WORDS = [w for w in WORDS if w["e"] <= DUR]


def timeline(**over):
    caps = []
    for k in range(0, len(WORDS), 3):
        ch = WORDS[k:k + 3]
        caps.append({"s": ch[0]["s"], "e": WORDS[k + 3]["s"] if k + 3 < len(WORDS) else DUR,
                     "text": " ".join(w["w"] for w in ch), "words": ch})
    tl = {"width": W, "height": H, "fps": FPS, "duration": DUR, "background": "#000000",
          "media": {"v": str(D / "src.mp4"), "a": str(D / "tone.wav")},
          "video": [{"media": "v", "start": 0, "in": 0, "dur": 2.0, "fit": "cover"},
                    {"media": "v", "start": 2.0, "in": 3.0, "dur": 2.0, "fit": "cover"}],
          "captions": caps,
          "styles": {"caption": {"font": "Arimo", "weight": 700, "size": 26, "fill": "#ffffff", "y": 40,
                                 "box": {"color": "#000000", "opacity": 1.0, "mode": "line"}}},
          "audio": {"clips": [{"media": "a", "start": 0, "in": 0, "dur": 2.0, "role": "voice"},
                              {"media": "a", "start": 2.0, "in": 3.0011, "dur": 2.0, "role": "voice"}],
                    "voice_chain": False, "loudness": -14},
          "encode": {"preset": "ultrafast"}}
    for k, v in over.items():
        tl[k] = v
    return tl


def make(name, tl):
    out = D / f"{name}.mp4"
    render.render(tl, out, draft=True)
    return out


def checks(rep, sev=None):
    return {i["check"] for i in rep["issues"] if sev is None or i["severity"] == sev}


class QC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tl = timeline()
        cls.clean = make("clean", cls.tl)

    def run_qc(self, path, tl=None, words=WORDS):
        return qc.check(path, tl or self.tl, transcript_words=words)

    def test_clean_render_passes(self):
        rep = self.run_qc(self.clean)
        self.assertEqual(rep["verdict"], "pass", rep["issues"])
        out = qc.sheet(self.clean, rep, self.tl, str(D / "clean.sheet.jpg"))
        self.assertTrue(Path(out).stat().st_size > 10000)

    def test_black_frames(self):
        bad = D / "black.mp4"
        ff("-i", self.clean, "-vf", "drawbox=c=black:t=fill:enable='between(t,1.0,1.4)'",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "copy", bad)
        rep = self.run_qc(bad)
        self.assertIn("black", checks(rep, "fail"))

    def test_frozen_run(self):
        bad = D / "frozen.mp4"
        ff("-i", self.clean, "-i", self.clean, "-filter_complex", "[0:v][1:v]freezeframes=first=40:last=75:replace=40",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-map", "0:a", "-c:a", "copy", bad)
        rep = self.run_qc(bad)
        self.assertIn("frozen", checks(rep, "fail"))

    def test_click_at_edit(self):
        tl = timeline()
        for c in tl["audio"]["clips"]:
            c["fade_in"] = c["fade_out"] = 0.0
        tl["audio"]["clips"][1]["in"] = 3.0 + 0.25 / 440       # tone at 0 meets tone at its peak: a step
        rep = self.run_qc(make("click", tl), tl)
        self.assertIn("click", checks(rep, "fail"))

    def test_wrong_loudness(self):
        bad = D / "quiet.mp4"
        ff("-i", self.clean, "-c:v", "copy", "-af", "volume=-6dB", "-c:a", "aac", "-b:a", "192k", bad)
        rep = self.run_qc(bad)
        self.assertIn("loudness", checks(rep, "fail"))

    def test_uncovered_border(self):
        tl = timeline()
        tl["video"][1]["scale"] = 0.85
        rep = self.run_qc(make("border", tl), tl)
        self.assertIn("border", checks(rep, "fail"))
        self.assertTrue(any(i["check"] == "border" and "leaves the canvas" in i["msg"] for i in rep["issues"]))
        self.assertTrue(any(i["check"] == "border" and "uncovered canvas edge" in i["msg"] for i in rep["issues"]))

    def test_captions_do_not_match_speech(self):
        spoken = [dict(w, w=x) for w, x in zip(WORDS, "the quick brown fox jumps over a lazy dog again and".split())]
        rep = self.run_qc(self.clean, words=spoken)
        self.assertIn("captions", checks(rep, "fail"))

    def test_caption_outside_safe_zone(self):
        tl = timeline()
        tl["styles"] = copy.deepcopy(tl["styles"])
        tl["styles"]["caption"]["y"] = 270                     # bottom band, under the platform UI
        rep = self.run_qc(make("unsafe", tl), tl)
        self.assertIn("safe_zone", checks(rep, "fail"))

    def test_source_too_short(self):
        tl = timeline()
        tl["video"][1]["in"] = 7.0                             # needs 2 s from a source that has 1 s left
        rep = qc.check(self.clean, tl, transcript_words=WORDS)
        self.assertIn("source", checks(rep, "fail"))


if __name__ == "__main__":
    unittest.main()
