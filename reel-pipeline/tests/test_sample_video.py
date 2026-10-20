"""Round-trip tests of the picture analyzer: timelines with known edits are rendered by
reelkit.render, analysed by reelkit.sample_video, and every edit, transition, move and
scale is checked against the truth.
Run: python3 -m unittest discover -s reel-pipeline/tests -v"""
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from reelkit import captions, ff, render
from reelkit import sample_video as sv
from reelkit.paths import FFMPEG

FPS = 30
W, H = 360, 640
DIR = Path(tempfile.mkdtemp(prefix="reeltest-video-"))
rng = np.random.default_rng(7)


def encode(frames_or_png, out, seconds=None):
    if isinstance(frames_or_png, (str, Path)):
        cmd = [FFMPEG, "-v", "error", "-y", "-loop", "1", "-i", str(frames_or_png), "-t", str(seconds), "-r", "30"]
        subprocess.run(cmd + ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out)], check=True)
        return out
    h, w = frames_or_png[0].shape[:2]
    p = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out)],
                         stdin=subprocess.PIPE)
    for f in frames_or_png:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    return out


def scene(seed):
    """A textured still: smooth colour blobs + hard-edged shapes (features to track)."""
    r = np.random.default_rng(seed)
    base = cv2.resize(r.random((24, 14, 3)).astype(np.float32), (720, 1280), interpolation=cv2.INTER_CUBIC)
    img = (np.clip(base, 0, 1) * 200 + 30).astype(np.uint8)
    for _ in range(120):
        x, y = int(r.integers(0, 720)), int(r.integers(0, 1280))
        c = tuple(int(v) for v in r.integers(0, 255, 3))
        if r.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(r.integers(10, 60)), y + int(r.integers(10, 60))), c, -1)
        else:
            cv2.circle(img, (x, y), int(r.integers(5, 30)), c, -1, lineType=cv2.LINE_AA)
    p = DIR / f"scene{seed}.png"
    cv2.imwrite(str(p), img)
    return encode(p, DIR / f"scene{seed}.mp4", 6)


def mover():
    """Static textured background with a large disc moving smoothly (a 'talking head')."""
    bg = cv2.imread(str(DIR / "scene1.png")) if (DIR / "scene1.png").exists() else None
    if bg is None:
        scene(1)
        bg = cv2.imread(str(DIR / "scene1.png"))
    frames = []
    for k in range(180):
        f = bg.copy()
        cx = int(360 + 160 * math.sin(k / 30 * 1.6))
        cy = int(640 + 120 * math.cos(k / 30 * 1.1))
        cv2.circle(f, (cx, cy), 170, (40, 60, 200), -1, lineType=cv2.LINE_AA)
        cv2.circle(f, (cx - 50, cy - 40), 25, (250, 250, 250), -1, lineType=cv2.LINE_AA)
        frames.append(f)
    return encode(frames, DIR / "mover.mp4")


def ui():
    """Synthetic screen recording: white sheet, grid, ribbon, typing, a menu, a cursor."""
    frames = []
    text = "Quarterly revenue by region"
    for k in range(270):
        img = Image.new("RGB", (720, 1280), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 720, 120], fill=(243, 243, 243))
        for i, lab in enumerate(["File", "Home", "Insert", "Draw", "Layout", "Data", "View"]):
            d.text((14 + i * 98, 20), lab, fill=(40, 40, 40))
        for i in range(10):
            d.rectangle([14 + i * 70, 60, 70 + i * 70, 100], outline=(150, 150, 150))
        for y in range(160, 1280, 36):
            d.line([(0, y), (720, y)], fill=(215, 215, 215))
            d.text((6, y + 10), str((y - 160) // 36 + 1), fill=(90, 90, 90))
        for x in range(50, 720, 110):
            d.line([(x, 130), (x, 1280)], fill=(215, 215, 215))
            d.text((x + 45, 136), "ABCDEFG"[(x - 50) // 110], fill=(90, 90, 90))
        n = min(len(text), k // 3)
        d.text((60, 205), text[:n], fill=(0, 0, 0))
        d.rectangle([160, 196, 380, 232], outline=(30, 120, 60), width=2)
        if 45 <= k < 110:   # a context menu opens
            d.rectangle([300, 520, 560, 820], fill=(250, 250, 250), outline=(120, 120, 120))
            for i, item in enumerate(["Cut", "Copy", "Paste", "Insert", "Delete", "Format cells", "Hide"]):
                d.text((320, 540 + i * 38), item, fill=(30, 30, 30))
        cx, cy = 200 + 2 * k, 700 - k
        d.polygon([(cx, cy), (cx + 14, cy + 20), (cx + 4, cy + 18)], fill=(0, 0, 0))
        frames.append(np.array(img)[..., ::-1].copy())
    return encode(frames, DIR / "ui.mp4")


def textured_testsrc():
    out = DIR / "testsrc.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=720x1280:r=30:d=6",
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


CAPTION = {"font": "Arimo", "weight": 700, "size": 26, "fill": "#ffffff", "box": {"color": "#004B06", "radius": 6},
           "y": 120, "anim": {"in": "pop", "in_dur": 0.12}}


def caption_events(duration, every=0.6):
    words = "this is how to make every column fit the data instantly and never stretch again".split()
    out, t, i = [], 0.2, 0
    while t + 0.3 < duration:
        out.append({"s": round(t, 3), "e": round(min(duration, t + every), 3), "text": " ".join(words[i:i + 2])})
        t += every
        i = (i + 2) % (len(words) - 1)
    return out


def caption_boxes(tl):
    """Ground-truth boxes of the rendered captions (alpha bbox at full size)."""
    boxes = []
    for ev in tl["captions"]:
        te = captions.TextEvent(ev, CAPTION, (W, H))
        rgb, a, x0, y0 = te.patch((ev["s"] + ev["e"]) / 2)
        ys, xs = np.where(a > 0.02)
        boxes.append({"s": ev["s"], "e": ev["e"],
                      "box": [x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1]})
    return boxes


def analyse(tl, name, text_boxes=()):
    out = DIR / f"{name}.mp4"
    tl = dict(tl, encode={"preset": "veryfast", "crf": 14})
    render.render(tl, out)
    info = ff.probe(out)
    return sv.analyze_video(str(out), info, text_boxes=text_boxes), info


class Natural(unittest.TestCase):
    """Moving textured footage: hard cuts, every transition, zooms, pans, punch, jump cut."""

    @classmethod
    def setUpClass(cls):
        s1, s2, s3, mv, ts = scene(1), scene(2), scene(3), mover(), textured_testsrc()
        V = [
            {"media": "s1", "start": 0.0, "in": 0, "dur": 2.0, "scale": [[0, 1.0], [2.0, 1.25, "inout"]]},
            {"media": "s2", "start": 2.0, "in": 0, "dur": 1.6, "scale": 1.3, "x": [[0, -40], [1.6, 40]]},
            {"media": "s3", "start": 3.6, "in": 0, "dur": 1.6, "transition_in": {"type": "dissolve", "dur": 0.4}},
            {"media": "ts", "start": 5.2, "in": 1, "dur": 1.6, "transition_in": {"type": "flash", "dur": 0.3}},
            {"media": "s1", "start": 6.8, "in": 0, "dur": 1.4, "scale": 1.1,
             "transition_in": {"type": "whip_left", "dur": 0.3}},
            {"media": "s2", "start": 8.2, "in": 0, "dur": 1.4, "transition_in": {"type": "zoom", "dur": 0.3}},
            {"media": "s3", "start": 9.6, "in": 0, "dur": 1.2, "transition_in": {"type": "dip_black", "dur": 0.4}},
            {"media": "mv", "start": 10.8, "in": 0.5, "dur": 1.2},
            {"media": "mv", "start": 12.0, "in": 1.7, "dur": 1.2, "scale": 1.4},             # punch-in 1.4
            {"media": "mv", "start": 13.2, "in": 3.9, "dur": 1.3, "scale": 1.4},             # jump cut
        ]
        cls.truth = [(2.0, "cut"), (3.6, "dissolve", 0.4), (5.2, "flash", 0.3), (6.8, "whip_left", 0.3),
                     (8.2, "zoom", 0.3), (9.6, "dip_black", 0.4), (10.8, "cut"), (12.0, "cut"), (13.2, "cut")]
        tl = {"width": W, "height": H, "fps": FPS, "duration": 14.5,
              "media": {"s1": str(s1), "s2": str(s2), "s3": str(s3), "mv": str(mv), "ts": str(ts)}, "video": V}
        cls.r, cls.info = analyse(tl, "natural")
        cls.shots = cls.r["shots"]

    def test_every_edit_found_once_on_the_right_frame(self):
        got = [(round(s["s"] * FPS), s["transition_in"]) for s in self.shots[1:]]
        self.assertEqual(len(got), len(self.truth), f"edits found: {got}")
        for (f, tin), tr in zip(got, self.truth):
            want = round(tr[0] * FPS)
            kind = tin if tin == "cut" else tin["type"]
            self.assertEqual(kind, tr[1], f"at frame {want}: {tin}")
            if kind == "cut":
                self.assertEqual(f, want, f"cut at {f}, expected {want}")
            else:
                self.assertLessEqual(abs(f - want), 1, f"{kind} centre at {f}, expected {want}")
                self.assertLessEqual(abs(tin["dur"] - tr[2]) * FPS, 2.01, f"{kind} lasts {tin['dur']}s, expected {tr[2]}s")

    def test_zoom_move_amount_and_ease(self):
        mv = self.shots[0]["move"]
        self.assertEqual(mv["type"], "zoom_in")
        self.assertAlmostEqual(mv["scale"], 1.25, delta=0.025)
        self.assertIn(mv["ease"], ("inout", "inout_cubic"))

    def test_pan_direction_and_amount(self):
        mv = self.shots[1]["move"]
        self.assertEqual(mv["type"], "pan")
        self.assertAlmostEqual(mv["dx"], 80 / W, delta=0.02)
        self.assertAlmostEqual(mv["dy"], 0, delta=0.01)
        self.assertEqual(mv["ease"], "linear")

    def test_punch_jump_and_new_scene(self):
        self.assertTrue(self.shots[1].get("new_scene"))
        self.assertTrue(self.shots[7].get("new_scene"))
        self.assertAlmostEqual(self.shots[8].get("punch", 0), 1.4, delta=0.042)
        self.assertTrue(self.shots[9].get("jump_cut"), self.shots[9])

    def test_pacing_and_transition_summary(self):
        p = self.r["pacing"]
        self.assertEqual(p["shot_count"], 10)
        self.assertAlmostEqual(sum(p["shot_durations"]), 14.5, delta=0.05)
        c = self.r["transitions"]["counts"]
        self.assertEqual(c.get("cut"), 4)
        self.assertEqual(sum(c.values()), 9)


class ScreenRecording(unittest.TestCase):
    """Typing, a menu, a cursor and captions must not be cuts; punches and a fast
    zoom move must be measured."""

    @classmethod
    def setUpClass(cls):
        u = ui()
        V = [{"media": "u", "start": 0.0, "in": 0.0, "dur": 2.5},
             {"media": "u", "start": 2.5, "in": 2.5, "dur": 2.0, "scale": 1.6, "x": 60, "y": 150},
             {"media": "u", "start": 4.5, "in": 4.5, "dur": 4.0,
              "scale": [[0, 1.0], [1.5, 1.0], [1.9, 1.3, "out"], [4.0, 1.3]]}]
        tl = {"width": W, "height": H, "fps": FPS, "duration": 8.5, "media": {"u": str(u)}, "video": V,
              "captions": caption_events(8.5), "styles": {"caption": CAPTION}}
        cls.r, cls.info = analyse(tl, "screen", caption_boxes(tl))
        cls.shots = cls.r["shots"]

    def test_only_the_two_punches_are_edits(self):
        got = [(round(s["s"] * FPS), s["transition_in"]) for s in self.shots[1:]]
        self.assertEqual([g[0] for g in got], [75, 135], f"edits: {got}")
        self.assertTrue(all(g[1] == "cut" for g in got))

    def test_punch_scales(self):
        self.assertAlmostEqual(self.shots[1]["punch"], 1.6, delta=0.048)
        self.assertAlmostEqual(self.shots[2]["punch"], 1 / 1.6, delta=0.019)

    def test_fast_zoom_move_inside_a_shot(self):
        mv = self.shots[2].get("moves")
        self.assertTrue(mv, self.shots[2])
        z = mv[0]
        self.assertEqual(z["type"], "zoom_in")
        self.assertAlmostEqual(z["scale"], 1.3, delta=0.026)
        self.assertAlmostEqual(z["t0"], 1.5, delta=0.1)
        self.assertAlmostEqual(z["t1"], 1.9, delta=0.1)
        self.assertIn(z["ease"], ("out", "out_cubic"))
        self.assertEqual(self.shots[0]["move"]["type"], "static")

    def test_clean_screen_has_no_grain_or_vignette(self):
        g = self.r["grade"]
        self.assertEqual(g["grain"], 0.0)
        self.assertEqual(g["vignette"], 0.0)
        self.assertGreater(g["look"]["L"][50], 85)


class Look(unittest.TestCase):
    def test_grain_and_vignette_estimates(self):
        s = scene(4)
        out = DIR / "graded.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(s), "-t", "2", "-vf",
                        f"scale={W}:{H},vignette=angle={math.pi / 5 * 1.0:.4f},noise=c0s=12:c0f=t+u",
                        "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(out)], check=True)
        g = sv._look(str(out), ff.probe(out), ())
        self.assertAlmostEqual(g["vignette"], 1.0, delta=0.35)
        self.assertAlmostEqual(g["grain"], 12, delta=5)


if __name__ == "__main__":
    unittest.main()
