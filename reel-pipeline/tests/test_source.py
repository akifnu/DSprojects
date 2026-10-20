"""source.py: editing copies (merging, handles, caching, whole-file rule, frame-exact
mapping), lossless audio, transcript caching, footage look."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reelkit import ff, source, store
from reelkit.paths import FFMPEG


def source_index(img, means, near):
    """Which source frame (within ±6 of `near`) a decoded frame shows, by brightness."""
    lo = max(0, near - 6)
    return lo + int(np.argmin(np.abs(means[lo:near + 7] - float(img.mean()))))


class Source(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="srctest-"))
        cls._data = store.DATA
        store.DATA = cls.tmp / "data"
        # 60 fps, frame n has luma 24 + 4·(n mod 40) (mid-range, distinct within ±6 frames): the frame shown can be read back
        cls.src = cls.tmp / "counter.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=black:s=640x360:r=60:d=12,format=gray,geq=lum='24+4*mod(N\\,40)'",
                        "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100:duration=12",
                        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-g", "60",
                        "-c:a", "aac", "-shortest", str(cls.src)], check=True)
        cls.asset = store.ingest(cls.src, "source")
        cls.means = np.array([float(f.mean()) for f in ff.frames(cls.src, 64, 36)])   # every 60 fps frame

    @classmethod
    def tearDownClass(cls):
        store.DATA = cls._data

    def test_merge_ranges(self):
        self.assertEqual(source.merge_ranges([(5, 6), (2, 3), (3.5, 4)], 20), [(1.0, 7.0)])
        self.assertEqual(source.merge_ranges([(1, 2), (12, 13)], 20), [(0.0, 3.0), (11.0, 14.0)])
        self.assertEqual(source.merge_ranges([(19, 21)], 20), [(18.0, 20.0)])

    def test_copies_map_source_time_to_the_right_frame(self):
        copies = source.prepare(self.asset, [(2.0, 2.6), (8.0, 8.5)], fps=30, short_side=360)
        for s, e in ((2.0, 2.6), (8.0, 8.5)):       # covered with ≥ 1 s handles (or by a cached whole-file copy)
            self.assertTrue(any(m["s"] <= s - 1 + 1e-6 and m["e"] >= e + 1 - 1e-6 for m in copies), (s, e, copies))
        for m in copies:
            info = ff.probe(m["path"])
            self.assertEqual((info["width"], info["height"], info["fps"]), (640, 360, 30.0))
            self.assertFalse(info["has_audio"])
        for t in (2.0, 2.3, 8.2, 8.4):          # on the 30 fps grid of the copies
            m, off = source.locate(copies, t)
            f = next(ff.frames(m["path"], 64, 36, start=off, duration=0.2))
            self.assertEqual(source_index(f, self.means, round(t * 60)), round(t * 60), f"t={t}")
        with self.assertRaises(KeyError):
            source.locate([{"s": 1.0, "e": 3.6}, {"s": 7.0, "e": 9.5}], 5.0)
        self.assertEqual(source.locate([{"s": 1.0, "e": 3.6}], 2.5)[1], 1.5)

    def test_cache_reuse_and_whole_file(self):
        a = source.prepare(self.asset, [(4.0, 4.5)], fps=30, short_side=360)
        mtime = os.path.getmtime(a[0]["path"])
        b = source.prepare(self.asset, [(4.1, 4.4)], fps=30, short_side=360)   # covered by the cached copy
        self.assertEqual(a[0]["path"], b[0]["path"])
        self.assertEqual(os.path.getmtime(b[0]["path"]), mtime)
        whole = source.prepare(self.asset, [(0, 5), (6, 11)], fps=30, short_side=360)
        self.assertEqual([(m["s"], m["e"]) for m in whole], [(0.0, 12.0)])
        index = json.loads(self.asset.cache("mezz/index.json").read_text())
        self.assertTrue(all(os.path.exists(m["path"]) for m in index))

    def test_audio_is_lossless_48k_and_transcript_is_cached(self):
        p = source.audio(self.asset)
        info = ff.probe(p)
        self.assertEqual((info["acodec"], info["sample_rate"], info["channels"]), ("flac", 48000, 2))
        self.assertAlmostEqual(info["duration"], 12.0, delta=0.1)
        fake = {"lang": "en", "timing": "word", "words": [], "sentences": []}
        self.asset.cache("transcript.en.json").write_text(json.dumps(fake))
        self.assertEqual(source.transcript(self.asset, "en"), fake)

    def test_look_stats(self):
        copies = source.prepare(self.asset, [(2.0, 3.0)], fps=30, short_side=360)
        lk = source.look(self.asset, copies)
        self.assertEqual(len(lk["L"]), 101)
        self.assertTrue(self.asset.cache("look.json").exists())


if __name__ == "__main__":
    unittest.main()
