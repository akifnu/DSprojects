"""Renderer tests on synthetic media: frame accuracy, sub-pixel zoom smoothness,
transitions, captions, audio edits, ducking, loudness.
Run: python3 -m unittest discover -s reel-pipeline/tests -v"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from reelkit import captions, ff, render
from reelkit.paths import FFMPEG

FPS = 30


def make(path, *args):
    subprocess.run([FFMPEG, "-v", "error", "-y", *args, str(path)], check=True)


class Media:
    """Fixtures generated once per test run."""
    dir = Path(tempfile.mkdtemp(prefix="reeltest-"))
    # frame k has luma 16 + 4k (k < 50): a frame's index can be read back after encoding
    counter = dir / "counter.mp4"
    make(counter, "-f", "lavfi", "-i", "color=black:s=320x568:r=30:d=1.67,format=gray,geq=lum='16+4*N'",
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-g", "1")
    dots = dir / "dots.png"
    img = np.zeros((568, 320, 3), np.uint8)
    for x, y in ((110, 284), (210, 284)):
        cv2.circle(img, (x, y), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(dots), img)
    dotsv = dir / "dots.mp4"
    make(dotsv, "-loop", "1", "-i", str(dots), "-t", "3", "-r", "30", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p")
    tone = dir / "tone.wav"
    make(tone, "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4", "-af", "volume=0.5")
    music = dir / "music.wav"
    make(music, "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=4", "-af", "volume=0.3")


def frame_index(img):
    return int(round((float(img.mean()) - 16) / 4))


class Keyframes(unittest.TestCase):
    def test_ease_and_keys(self):
        self.assertAlmostEqual(render.ease("linear", 0.3), 0.3)
        self.assertAlmostEqual(render.ease("inout", 0.5), 0.5, places=3)
        self.assertLess(render.ease("in", 0.25), 0.25)
        self.assertGreater(render.ease("out", 0.25), 0.25)
        keys = [[0, 1.0], [2, 2.0, "linear"], [3, 2.0]]
        self.assertAlmostEqual(render.kf(keys, -1), 1.0)
        self.assertAlmostEqual(render.kf(keys, 1.0), 1.5)
        self.assertAlmostEqual(render.kf(keys, 2.5), 2.0)
        self.assertAlmostEqual(render.kf(keys, 9), 2.0)
        self.assertAlmostEqual(render.kf(1.25, 5), 1.25)


class Video(unittest.TestCase):
    def comp(self, tl):
        return render.Compositor(tl, str(Media.dir))

    def test_cuts_are_frame_accurate(self):
        tl = {"width": 320, "height": 568, "fps": FPS, "duration": 1.0, "media": {"c": str(Media.counter)},
              "video": [{"media": "c", "start": 0, "in": 0.2, "dur": 0.5},
                        {"media": "c", "start": 0.5, "in": 1.0, "dur": 0.5}]}
        c = self.comp(tl)
        got = [frame_index(c.frame(f)) for f in range(30)]
        c.close()
        self.assertEqual(got, list(range(6, 21)) + list(range(30, 45)))

    def test_keyframed_zoom_is_smooth_subpixel(self):
        tl = {"width": 320, "height": 568, "fps": FPS, "duration": 2.0, "media": {"d": str(Media.dotsv)},
              "video": [{"media": "d", "start": 0, "in": 0, "dur": 2.0, "fit": "cover",
                         "scale": [[0, 1.0], [2.0, 1.5, "inout"]]}]}
        c = self.comp(tl)
        gaps = []
        for f in range(60):
            g = c.frame(f)[..., 1].astype(np.float64)
            row = g[270:300].sum(axis=0)
            xs = np.arange(len(row))
            left, right = row[:160], row[160:]
            gaps.append((right * xs[160:]).sum() / right.sum() - (left * xs[:160]).sum() / left.sum())
        c.close()
        want = [100 * render.kf([[0, 1.0], [2.0, 1.5, "inout"]], f / FPS) for f in range(60)]
        err = np.abs(np.array(gaps) - np.array(want))
        self.assertLess(err.max(), 0.35, f"dot spacing off the eased curve by {err.max():.2f}px")
        steps = np.diff(gaps)
        self.assertTrue((steps > -0.05).all(), "zoom must not jitter backwards")

    def test_dissolve_midpoint_blends_both_clips(self):
        tl = {"width": 320, "height": 568, "fps": FPS, "duration": 1.0, "media": {"c": str(Media.counter)},
              "video": [{"media": "c", "start": 0, "in": 0.0, "dur": 0.5},
                        {"media": "c", "start": 0.5, "in": 1.0, "dur": 0.5,
                         "transition_in": {"type": "dissolve", "dur": 0.2}}]}
        c = self.comp(tl)
        mid = c.frame(15).mean()       # u = 0.5: half outgoing (frame 15 → 76) half incoming (frame 30 → 136)
        c.close()
        self.assertAlmostEqual(mid, (76 + 136) / 2, delta=3)

    def test_caption_lands_where_styled_and_highlights_active_word(self):
        style = {"font": "Arimo", "weight": 700, "size": 30, "fill": "#ffffff", "y": 100,
                 "highlight": {"fill": "#ff0000"}}
        tl = {"width": 320, "height": 568, "fps": FPS, "duration": 1.0, "background": "#000000",
              "captions": [{"s": 0.0, "e": 1.0, "text": "one two",
                            "words": [{"w": "one", "s": 0.0, "e": 0.5}, {"w": "two", "s": 0.5, "e": 1.0}]}],
              "styles": {"caption": style}}
        c = self.comp(tl)
        a, b = c.frame(5), c.frame(20)
        c.close()
        ys = np.where(a.max(axis=(1, 2)) > 128)[0]
        self.assertAlmostEqual((ys.min() + ys.max()) / 2, 284 + 100, delta=6)
        red_a = (a[..., 2] > 200) & (a[..., 1] < 60)
        red_b = (b[..., 2] > 200) & (b[..., 1] < 60)
        self.assertLess(np.where(red_a)[1].mean(), 160)   # "one" (left word) active first
        self.assertGreater(np.where(red_b)[1].mean(), 160)

    def test_still_matches_rendered_frame(self):
        tl = {"width": 320, "height": 568, "fps": FPS, "duration": 1.0, "media": {"c": str(Media.counter)},
              "video": [{"media": "c", "start": 0, "in": 0.3, "dur": 1.0}]}
        out = Media.dir / "still.png"
        render.still(tl, 0.5, out)
        self.assertEqual(frame_index(cv2.imread(str(out))), 24)


class Audio(unittest.TestCase):
    def test_edit_fades_loudness_ducking_and_mux(self):
        tl = {"width": 160, "height": 284, "fps": FPS, "duration": 3.0,
              "media": {"v": str(Media.counter), "tone": str(Media.tone), "m": str(Media.music)},
              "video": [{"media": "v", "start": 0, "in": 0, "dur": 1.6}],
              "audio": {"clips": [{"media": "tone", "start": 0.0, "in": 0.0, "dur": 1.0, "role": "voice"},
                                  {"media": "tone", "start": 1.0, "in": 1.7311, "dur": 0.8, "role": "voice"},
                                  {"media": "m", "start": 0, "in": 0, "dur": 3.0, "role": "music"}],
                        "voice_chain": False, "duck_db": 15, "loudness": -16},
              "encode": {"preset": "ultrafast"}}
        out = Media.dir / "a.mp4"
        render.render(tl, out)
        info = ff.probe(out)
        self.assertAlmostEqual(info["duration"], 3.0, delta=0.05)
        self.assertAlmostEqual(ff.loudness(out)["I"], -16, delta=1.0)
        a = ff.audio(out, 48000)
        # no click at the jump cut: around 1.0 s the waveform stays continuous
        cut = a[int(0.99 * 48000):int(1.01 * 48000)]
        self.assertLess(np.abs(np.diff(cut)).max(), 0.2)

        def band(sig, f):
            spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
            freqs = np.fft.rfftfreq(len(sig), 1 / 48000)
            return spec[np.abs(freqs - f) < 20].max()
        under_voice = band(a[int(0.3 * 48000):int(0.7 * 48000)], 1000)
        alone = band(a[int(2.4 * 48000):int(2.8 * 48000)], 1000)
        self.assertAlmostEqual(20 * np.log10(alone / under_voice), 15, delta=2.0)


if __name__ == "__main__":
    unittest.main()
