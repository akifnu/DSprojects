"""Round-trip tests for the on-screen text analyzer: render known caption styles with
reelkit.render over moving backgrounds, analyze, compare.
Run: python3 -m unittest tests.test_sample_text -v   (from reel-pipeline/)"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from reelkit import captions as C
from reelkit import ff, render
from reelkit import sample_text as st
from reelkit.paths import FFMPEG

W, H, FPS, DUR = 540, 960, 30, 3.0
TMP = Path(tempfile.mkdtemp(prefix="texttest-"))


def _bg_moving():
    p = TMP / "bg.mp4"
    if not p.exists():
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"gradients=s={W}x{H}:r={FPS}:d={DUR}:speed=0.03:c0=#20305a:c1=#7a3b2e:c2=#2e6a4a:n=3",
                        "-c:v", "libx264", "-qp", "10", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def _bg_ui():
    """A screen-recording-like frame: toolbar, grid with small labels, a menu; slow pan."""
    p = TMP / "ui.mp4"
    if p.exists():
        return p
    img = Image.new("RGB", (W + 60, H), (250, 250, 250))
    d = ImageDraw.Draw(img)
    f = C._font("Inter", 400, 13)
    d.rectangle([0, 0, W + 60, 70], fill=(236, 236, 236))
    for i, t in enumerate(["File", "Home", "Insert", "Draw", "Page Layout", "Formulas", "Data", "Review"]):
        d.text((8 + i * 68, 12), t, font=f, fill=(40, 40, 40))
    for i, t in enumerate(["Cut", "Copy", "Paste", "Format Painter", "Wrap Text", "Merge & Center"]):
        d.text((8 + i * 92, 42), t, font=f, fill=(60, 60, 60))
    for r in range(40):
        y = 90 + r * 22
        d.line([0, y, W + 60, y], fill=(215, 215, 215))
        d.text((6, y + 4), str(r + 1), font=f, fill=(90, 90, 90))
        if r % 3 == 0:
            d.text((60, y + 4), f"Client {r} Product Sold", font=f, fill=(20, 20, 20))
    d.rectangle([300, 400, 470, 560], fill=(255, 255, 255), outline=(150, 150, 150))
    for i, t in enumerate(["Insert...", "Delete", "Rename", "View Code", "Hide"]):
        d.text((310, 408 + i * 30), t, font=f, fill=(30, 30, 30))
    src = TMP / "ui.png"
    img.save(src)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-loop", "1", "-i", str(src), "-vf",
                    f"crop={W}:{H}:'min(60,t*20)':0,fps={FPS}", "-t", str(DUR), "-c:v", "libx264", "-qp", "10",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def _events(texts, t0=0.3, dur=0.8, gap=0.0):
    evs, t = [], t0
    for text in texts:
        words = text.replace("\n", " ").split()
        ws = [{"w": w, "s": round(t + i * dur / len(words), 3), "e": round(t + (i + 1) * dur / len(words), 3)}
              for i, w in enumerate(words)]
        evs.append({"s": round(t, 3), "e": round(t + dur, 3), "text": text, "words": ws})
        t += dur + gap
    return evs


def _make(name, style, texts, bg, extra_text=(), title_style=None, t0=0.3, dur=0.8):
    out = TMP / f"{name}.mp4"
    caps = _events(texts, t0, dur)
    tl = {"width": W, "height": H, "fps": FPS, "duration": DUR, "media": {"bg": str(bg)},
          "video": [{"media": "bg", "start": 0, "in": 0, "dur": DUR, "fit": "cover"}],
          "captions": caps, "text": list(extra_text),
          "styles": {"caption": style, **({"title": title_style} if title_style else {})},
          "encode": {"preset": "veryfast", "crf": 14}}
    render.render(tl, out)
    return out, caps


def _de(hex1, hex2):
    a = np.array(C.rgba(hex1)[:3][::-1], float)
    b = np.array(C.rgba(hex2)[:3][::-1], float)
    return st._de(a, b)


CASES = {
    "pill": {"style": {"font": "Arimo", "weight": 700, "size": 40, "fill": "#ffffff", "y": 120,
                       "box": {"color": "#004B06", "radius": 8, "pad_x": 12, "pad_y": 3, "mode": "line"}},
             "texts": ["Stop stretching columns", "Instead...", "Try this\nsecret trick"], "anim": "none"},
    "stroke_pop": {"style": {"font": "Montserrat", "weight": 900, "size": 46, "case": "upper", "fill": "#ffffff",
                             "stroke": {"color": "#000000", "width": 4}, "highlight": {"fill": "#ffe600"},
                             "y": 150, "anim": {"in": "pop", "in_dur": 0.2}},
                   "texts": ["this changes everything", "watch this now", "seriously look"], "anim": "pop"},
    "block_wordbox": {"style": {"font": "Poppins", "weight": 700, "size": 38, "fill": "#111111", "y": 140,
                                "box": {"color": "#ffffff", "radius": 14, "pad_x": 16, "pad_y": 8, "mode": "block"},
                                "highlight": {"fill": "#ffffff", "box": {"color": "#7b2ff7", "radius": 8}}},
                      "texts": ["stop scrolling and watch", "this is the trick", "nobody tells you"],
                      "anim": "none"},
    "shadow_fade": {"style": {"font": "Roboto", "weight": 700, "size": 42, "fill": "#ffffff", "y": 160,
                              "shadow": {"color": "#000000", "opacity": 0.7, "blur": 3, "dx": 0, "dy": 3},
                              "anim": {"in": "fade", "in_dur": 0.25}},
                    "texts": ["the quiet part", "said out loud", "every single time"], "anim": "fade"},
    "reveal": {"style": {"font": "Bebas Neue", "weight": 400, "size": 64, "fill": "#ffffff", "y": 120,
                         "stroke": {"color": "#000000", "width": 3}, "anim": {"in": "word_reveal"}},
               "texts": ["ONE WORD AT A TIME", "BUILDING UP SLOWLY", "UNTIL IT IS DONE"], "anim": "word_reveal"},
}


class RoundTrip(unittest.TestCase):
    results = {}

    @classmethod
    def setUpClass(cls):
        bg = _bg_moving()
        cls.out = {}
        for name, c in CASES.items():
            path, caps = _make(name, c["style"], c["texts"], bg)
            cls.out[name] = (st.analyze_text(path, ff.probe(path)), caps)

    def check(self, name):
        c = CASES[name]
        res, caps = self.out[name]
        cap = res["captions"]
        self.assertTrue(cap["present"], f"{name}: no caption track")
        self.assertEqual(len(cap["events"]), len(caps), f"{name}: {cap['events']}")
        self.assertEqual(res["titles"]["events"], [], f"{name}: false titles")
        for got, want in zip(cap["events"], caps):
            self.assertLessEqual(abs(got["s"] - want["s"]), 1.01 / FPS, f"{name} start {got} vs {want}")
            self.assertLessEqual(abs(got["e"] - want["e"]), 1.01 / FPS, f"{name} end {got} vs {want}")
        s, want = cap["style"], c["style"]
        self.assertLess(abs(s["size"] / want["size"] - 1), 0.06, f"{name} size {s.get('size')}")
        self.assertLess(_de(s["fill"], want["fill"]), 6, f"{name} fill {s['fill']}")
        for key in ("box", "stroke", "shadow"):
            self.assertEqual(key in s, key in want, f"{name}: {key} presence {s.get(key)}")
        if "box" in want:
            self.assertLess(_de(s["box"]["color"], want["box"]["color"]), 6, f"{name} box {s['box']}")
        if "stroke" in want:
            self.assertLess(_de(s["stroke"]["color"], want["stroke"]["color"]), 6, f"{name} stroke {s['stroke']}")
        self.assertEqual("highlight" in s, "highlight" in want, f"{name} highlight {s.get('highlight')}")
        if "highlight" in want and want["highlight"].get("fill") and want["highlight"]["fill"] != want["fill"]:
            self.assertLess(_de(s["highlight"]["fill"], want["highlight"]["fill"]), 6, f"{name} {s['highlight']}")
        if "highlight" in want and want["highlight"].get("box"):
            self.assertLess(_de(s["highlight"]["box"]["color"], want["highlight"]["box"]["color"]), 6)
        self.assertEqual(s.get("anim", {}).get("in", "none"), c["anim"], f"{name} anim {s.get('anim')}")
        RoundTrip.results[name] = s.get("font") == want["font"]
        return s

    def test_pill(self):
        self.check("pill")

    def test_stroke_pop(self):
        self.check("stroke_pop")

    def test_block_wordbox(self):
        self.check("block_wordbox")

    def test_shadow_fade(self):
        self.check("shadow_fade")

    def test_reveal(self):
        self.check("reveal")

    def test_zz_fonts(self):
        fams = {n: (self.out[n][0]["captions"]["style"].get("font"), CASES[n]["style"]["font"]) for n in CASES}
        right = sum(a == b for a, b in fams.values())
        self.assertGreaterEqual(right, len(CASES) - 1, f"fonts {fams}")


class TitleAndHandle(unittest.TestCase):
    def test_slide_title_and_persistent_handle(self):
        title_style = {"font": "Anton", "weight": 400, "size": 62, "fill": "#ffd60a", "y": -300,
                       "stroke": {"color": "#000000", "width": 3}, "anim": {"in": "slide_up", "in_dur": 0.3,
                                                                           "slide_px": 50}}
        handle = {"s": 0.0, "e": DUR, "text": "@reelkit.studio",
                  "style": {"font": "Inter", "weight": 600, "size": 24, "fill": "#ffffff", "x": 150, "y": 400,
                            "shadow": {"opacity": 0.6, "blur": 2, "dy": 2}}}
        title = {"s": 0.4, "e": 2.6, "text": "STOP DOING THIS", "style": "title"}
        path, _ = _make("title", {"size": 40}, [], _bg_moving(), extra_text=[title, handle],
                        title_style=title_style)
        res = st.analyze_text(path, ff.probe(path))
        self.assertFalse(res["captions"]["present"], res["captions"]["events"])
        self.assertEqual(len(res["titles"]["events"]), 1, res["titles"]["events"])
        t = res["titles"]["events"][0]
        self.assertLessEqual(abs(t["s"] - 0.4), 1.01 / FPS)
        self.assertEqual(res["titles"]["style"]["anim"]["in"], "slide_up")
        self.assertLess(_de(res["titles"]["style"]["fill"], "#ffd60a"), 6)
        self.assertEqual(len(res["overlays"]["persistent"]), 1, res["overlays"])
        self.assertIn("reelkit", res["overlays"]["persistent"][0]["text"])


class ScreenRecording(unittest.TestCase):
    def test_ui_text_ignored_captions_found(self):
        style = CASES["pill"]["style"]
        path, caps = _make("ui", style, ["Right click the tab", "Select View Code", "That's it!"], _bg_ui())
        res = st.analyze_text(path, ff.probe(path))
        self.assertEqual([e["text"].replace("\n", " ") for e in res["captions"]["events"]],
                         [c["text"] for c in caps])
        self.assertEqual(res["titles"]["events"], [])
        self.assertEqual(res["overlays"]["persistent"], [])
        self.assertEqual(len(res["boxes"]), 3)


if __name__ == "__main__":
    unittest.main()
