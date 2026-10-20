"""Batch runner: a 4-reel plan with 2 workers is interrupted, resumed (finished reels
come from the cache), and delivers manifest + files; a changed reel alone re-renders;
a bad plan fails before any work.
Run: python3 -m unittest discover -s reel-pipeline/tests -v"""
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from reelkit import batch
from reelkit.paths import FFMPEG, ROOT

D = Path(tempfile.mkdtemp(prefix="batchtest-"))
DATA = D / "data"


def ff(*args):
    subprocess.run([FFMPEG, "-v", "error", "-y", *map(str, args)], check=True)


ff("-f", "lavfi", "-i", "testsrc2=s=360x640:r=30:d=12", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "12",
   "-pix_fmt", "yuv420p", D / "src.mp4")
ff("-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=12", "-af", "volume=0.4", D / "tone.wav")


def timeline(i, text="hello there"):
    dur = 3.0
    return {"width": 360, "height": 640, "fps": 30, "duration": dur, "background": "#000000",
            "media": {"v": "src.mp4", "a": "tone.wav"},
            "video": [{"media": "v", "start": 0, "in": i * 1.0, "dur": 1.5, "fit": "cover"},
                      {"media": "v", "start": 1.5, "in": i * 1.0 + 4, "dur": 1.5, "fit": "cover",
                       "scale": [[0, 1.0], [1.5, 1.1, "inout"]]}],
            "captions": [{"s": 0.2, "e": 2.8, "text": f"{text} {i}"}],
            "styles": {"caption": {"font": "Arimo", "weight": 700, "size": 26, "y": 40,
                                   "box": {"color": "#000000", "opacity": 1.0}}},
            "audio": {"clips": [{"media": "a", "start": 0, "in": 0, "dur": 3.0, "role": "voice"}],
                      "voice_chain": False, "loudness": -14}}


def write_plan(n=4):
    for i in range(n):
        (D / f"tl{i}.json").write_text(json.dumps(timeline(i)))
    (D / "style.json").write_text(json.dumps({"version": 2, "sample": {"height": 640},
                                              "pacing": {"median_shot": 1.5, "max_pause": 0.5}}))
    plan = {"name": "testbatch", "style": "style.json", "output": {"width": 360, "height": 640, "fps": 30},
            "sources": {}, "reels": [{"id": f"r{i}", "title": f"Reel number {i}!", "timeline": f"tl{i}.json"}
                                     for i in range(n)]}
    p = D / "plan.json"
    p.write_text(json.dumps(plan))
    return p


class Batch(unittest.TestCase):
    def setUp(self):
        self._data = batch.DATA
        batch.DATA = DATA

    def tearDown(self):
        batch.DATA = self._data

    def test_bad_plan_fails_fast(self):
        bad = D / "bad.json"
        bad.write_text(json.dumps({"name": "bad name", "style": "nope.json",
                                   "reels": [{"id": "a", "segments": [{"src": "X", "sentences": [5, 2]}]},
                                             {"id": "a"}]}))
        with self.assertRaises(batch.PlanError) as cm:
            batch.run(bad, jobs=1)
        msg = str(cm.exception)
        for part in ("name", "style", "sources", "used twice", "sentences must be", "segments must be"):
            self.assertIn(part, msg)

    def test_interrupt_resume_manifest_and_invalidation(self):
        plan = write_plan()
        bdir = DATA / "batches" / "testbatch"
        env = dict(os.environ, REELKIT_DATA=str(DATA), PYTHONPATH=str(ROOT))
        code = f"from reelkit import batch; batch.run({str(plan)!r}, jobs=2, draft=True, asr=False)"
        proc = subprocess.Popen([sys.executable, "-c", code], env=env, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 120
        while time.time() < deadline and not list(bdir.glob("reels/*/cache.json")):
            time.sleep(0.05)
        os.killpg(proc.pid, signal.SIGKILL)                     # the whole batch dies mid-run
        proc.wait()
        finished = {p.parent.name for p in bdir.glob("reels/*/cache.json")}
        self.assertTrue(1 <= len(finished) < 4, f"interrupt landed after {len(finished)} reels")
        mtimes = {r: (bdir / "reels" / r / f"{r}.mp4").stat().st_mtime_ns for r in finished}

        m = batch.run(plan, jobs=2, draft=True, asr=False)        # resume
        self.assertEqual(len(m["reels"]), 4)
        for e in m["reels"]:
            self.assertIn(e["verdict"], ("pass", "warn"), e)
            self.assertEqual(e["cached"], e["id"] in finished)
            self.assertTrue(Path(e["file"]).exists())
        for r, t in mtimes.items():
            self.assertEqual((bdir / "reels" / r / f"{r}.mp4").stat().st_mtime_ns, t, "cached reel was re-rendered")
        names = sorted(p.name for p in (bdir / "out").glob("*.mp4"))
        self.assertEqual(names, [f"{i + 1:02d}_reel-number-{i}.mp4" for i in range(4)])
        with open(bdir / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["id"] for r in rows], ["r0", "r1", "r2", "r3"])
        self.assertTrue((bdir / "out" / "contact_sheet.jpg").exists())
        self.assertTrue((bdir / "reels" / "r0" / "sheet.jpg").exists())

        # change one reel: only it re-renders
        (D / "tl2.json").write_text(json.dumps(timeline(2, text="changed words")))
        m2 = batch.run(plan, jobs=2, draft=True, asr=False)
        cached = {e["id"]: e["cached"] for e in m2["reels"]}
        self.assertEqual(cached, {"r0": True, "r1": True, "r2": False, "r3": True})


if __name__ == "__main__":
    unittest.main()
