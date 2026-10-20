"""Caption and title typography: Google Fonts, per-line or block boxes with rounded
corners, outline, soft shadow, active-word highlight, and in/out animation.

A text event is rendered to RGBA patches (cached per animation step and highlighted
word) that the compositor alpha-blends; nothing is drawn per frame twice.

Style keys (all optional; sizes in px at the timeline's canvas size):
  font, weight, size, fill, case ("upper"|"lower"|"title"|"none"), align, line_height,
  max_width, x, y (centre offset from canvas centre), letter_spacing,
  stroke {color, width}, shadow {color, opacity, blur, dx, dy},
  box {color, opacity, radius, pad_x, pad_y, mode: "line"|"block"},
  highlight {fill, box {color, opacity, radius, pad_x, pad_y}, scale},   active word
  anim {in: none|pop|fade|slide_up|slide_down|zoom|word_reveal, in_dur, overshoot,
        out: none|fade|pop, out_dur, slide_px}
"""
import io
import math
import os
import re
import urllib.request
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import ff
from .paths import FONTS, LIBRARY

FALLBACKS = ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


@lru_cache(maxsize=256)
def font_file(family, weight=700):
    """TTF for a Google Font family/weight, cached in data/fonts (nearest available weight)."""
    FONTS.mkdir(parents=True, exist_ok=True)
    local = LIBRARY / "fonts" / f"{family}-{weight}.ttf"
    if local.exists():
        return str(local)
    path = FONTS / f"{family.replace(' ', '_')}-{weight}.ttf"
    if path.exists():
        return str(path)
    for w in [weight] + sorted({400, 500, 600, 700, 800, 900} - {weight}, key=lambda x: abs(x - weight)):
        try:
            url = f"https://fonts.googleapis.com/css2?family={family.replace(' ', '+')}:wght@{w}"
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/4.0"}), timeout=8) as r:
                css = r.read().decode()
            ttf = re.search(r"url\((https://[^)]+\.ttf)\)", css).group(1)
            with urllib.request.urlopen(ttf, timeout=15) as r:
                data = r.read()
            ImageFont.truetype(io.BytesIO(data), 16)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            if w != weight:
                ff.log(f"font {family} has no weight {weight}; using {w}")
            return str(path)
        except Exception:
            continue
    for fb in FALLBACKS:
        if os.path.exists(fb):
            ff.log(f"font {family} unavailable; using {fb}")
            return fb
    raise SystemExit(f"font {family} unavailable and no fallback installed")


@lru_cache(maxsize=512)
def _font(family, weight, size):
    return ImageFont.truetype(font_file(family, weight), max(1, int(round(size))))


def rgba(color, opacity=1.0):
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    a = int(c[6:8], 16) / 255 if len(c) == 8 else 1.0
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) + (int(round(255 * a * opacity)),)


def apply_case(text, case):
    if case == "upper":
        return text.upper()
    if case == "lower":
        return text.lower()
    if case == "title":
        return " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
    return text


DEFAULT = {"font": "Montserrat", "weight": 800, "size": 56, "fill": "#ffffff", "case": "none", "align": "center",
           "line_height": 1.2, "max_width": 0.86, "x": 0, "y": 0, "letter_spacing": 0}


def resolve(style, canvas_w):
    s = dict(DEFAULT, **(style or {}))
    if s["max_width"] <= 1.0:
        s["max_width"] = s["max_width"] * canvas_w
    return s


def _words_width(font, words, spacing):
    return sum(font.getlength(w) for w in words) + spacing * (len(words) - 1)


def layout(text, s, scale=1.0):
    """Lines of words fitting max_width (explicit \\n respected)."""
    font = _font(s["font"], s["weight"], s["size"] * scale)
    space = font.getlength(" ") + s.get("letter_spacing", 0)
    lines = []
    for para in apply_case(text, s["case"]).split("\n"):
        cur = []
        for w in para.split():
            if cur and _words_width(font, cur + [w], space) > s["max_width"] * scale:
                lines.append(cur)
                cur = []
            cur.append(w)
        lines.append(cur)
    return font, space, [l for l in lines if l]


def render(text, style, canvas_w, highlight=None, scale=1.0, reveal=None):
    """RGBA patch (PIL Image) of a text event and the offset of its centre.
    highlight: index of the active word; reveal: number of words visible."""
    s = resolve(style, canvas_w)
    font, space, lines = layout(text, s, scale)
    asc, desc = font.getmetrics()
    lh = s["size"] * scale * s["line_height"]
    stroke = s.get("stroke") or {}
    sw = int(round(stroke.get("width", 0) * scale))
    box = s.get("box")
    hl = s.get("highlight") or {}
    pad_x = (box or {}).get("pad_x", 0.35 * s["size"]) * scale if box else 0
    pad_y = (box or {}).get("pad_y", 0.12 * s["size"]) * scale if box else 0
    shadow = s.get("shadow")
    margin = int(sw + pad_x + 4 + (shadow.get("blur", 0) * 3 + abs(shadow.get("dx", 0)) + abs(shadow.get("dy", 0))
                                    if shadow else 0) + (hl.get("scale", 1) - 1) * s["size"] * scale + 8)
    hl_font = _font(s["font"], s["weight"], s["size"] * scale * hl.get("scale", 1)) if hl.get("scale", 1) != 1 else font
    # word widths with the active word at its highlight size, so it pushes its
    # neighbours instead of overlapping them
    wlist, k = [], 0
    for line in lines:
        wlist.append([(hl_font if (highlight is not None and k + i == highlight) else font).getlength(w)
                      for i, w in enumerate(line)])
        k += len(line)
    widths = [sum(ws) + space * (len(ws) - 1) for ws in wlist]
    block_w = max(widths) if widths else 0
    W = int(math.ceil(block_w + 2 * margin))
    H = int(math.ceil(lh * len(lines) + 2 * margin))
    text_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    box_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td, bd = ImageDraw.Draw(text_layer), ImageDraw.Draw(box_layer)
    fill = rgba(s["fill"])
    k = 0
    words_xy = []
    for li, (line, lw, ws) in enumerate(zip(lines, widths, wlist)):
        x0 = margin + {"left": 0, "right": block_w - lw}.get(s["align"], (block_w - lw) / 2)
        y0 = margin + li * lh
        cy = y0 + lh / 2
        if box and box.get("mode", "line") == "line" and line:
            nvis = len(line) if reveal is None else max(0, min(len(line), reveal - k))
            if nvis:
                vw = sum(ws[:nvis]) + space * (nvis - 1)
                bd.rounded_rectangle([x0 - pad_x, cy - lh / 2 - pad_y, x0 + vw + pad_x, cy + lh / 2 + pad_y],
                                     radius=box.get("radius", 0.25 * s["size"]) * scale,
                                     fill=rgba(box.get("color", "#000000"), box.get("opacity", 1)))
        x = x0
        for w, ww in zip(line, ws):
            if reveal is None or k < reveal:
                words_xy.append((k, w, x, cy, ww))
            x += ww + space
            k += 1
    if box and box.get("mode") == "block" and words_xy:
        bd.rounded_rectangle([margin + min(0, 0) - pad_x, margin - pad_y, margin + block_w + pad_x,
                              margin + lh * len(lines) + pad_y], radius=box.get("radius", 0.25 * s["size"]) * scale,
                             fill=rgba(box.get("color", "#000000"), box.get("opacity", 1)))
    for k_, w, x, cy, ww in words_xy:
        active = highlight is not None and k_ == highlight
        if active and hl.get("box"):
            hb = hl["box"]
            px, py = hb.get("pad_x", 0.12 * s["size"]) * scale, hb.get("pad_y", 0.08 * s["size"]) * scale
            bd.rounded_rectangle([x - px, cy - lh / 2 - py + lh * 0.08, x + ww + px, cy + lh / 2 + py - lh * 0.08],
                                 radius=hb.get("radius", 0.2 * s["size"]) * scale,
                                 fill=rgba(hb.get("color", "#ffd60a"), hb.get("opacity", 1)))
        f = hl_font if active else font
        col = rgba(hl["fill"]) if active and hl.get("fill") else fill
        td.text((x, cy), w, font=f, fill=col, anchor="lm", stroke_width=sw,
                stroke_fill=rgba(stroke.get("color", "#000000")) if sw else None)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if shadow:
        a = Image.new("RGBA", (W, H), rgba(shadow.get("color", "#000000"), 1))
        mask = Image.alpha_composite(box_layer, text_layer).getchannel("A")
        mask = mask.point(lambda v: int(v * shadow.get("opacity", 0.6)))
        if shadow.get("blur", 0):
            mask = mask.filter(ImageFilter.GaussianBlur(shadow["blur"] * scale))
        a.putalpha(mask)
        out.alpha_composite(a, (int(round(shadow.get("dx", 0) * scale)), int(round(shadow.get("dy", 0) * scale))))
    out = Image.alpha_composite(out, box_layer)
    out = Image.alpha_composite(out, text_layer)
    return out, (W / 2, H / 2)


def _ease_out_back(u, overshoot):
    c = overshoot * 10
    return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2


class TextEvent:
    """One caption/title on screen from s to e; words carry timings for highlight/reveal."""

    def __init__(self, ev, style, canvas):
        self.s, self.e = ev["s"], ev["e"]
        self.text = ev["text"]
        self.words = ev.get("words") or []
        self.style = resolve(style, canvas[0])
        self.canvas = canvas
        self.cache = {}

    def _state(self, t):
        a = self.style.get("anim") or {}
        hl = None
        if self.style.get("highlight") and self.words:
            for i, w in enumerate(self.words):
                if w["s"] <= t < (self.words[i + 1]["s"] if i + 1 < len(self.words) else self.e):
                    hl = i
        reveal = None
        if a.get("in") == "word_reveal" and self.words:
            reveal = sum(1 for w in self.words if w["s"] <= t + 1e-6) or 1
        scale, alpha, dy = 1.0, 1.0, 0.0
        ind, outd = a.get("in_dur", 0.15), a.get("out_dur", 0.12)
        u = (t - self.s) / ind if ind else 1.0
        v = (self.e - t) / outd if outd else 1.0
        if a.get("in") in ("pop", "zoom") and u < 1:
            steps = max(1, round(ind * 30))
            u = math.floor(u * steps) / steps
            base = a.get("from", 0.6 if a.get("in") == "pop" else 1.15)
            scale = base + (1 - base) * (_ease_out_back(u, a.get("overshoot", 0.12)) if a.get("in") == "pop" else u)
        elif a.get("in") == "fade" and u < 1:
            alpha = u
        elif a.get("in") in ("slide_up", "slide_down") and u < 1:
            dy = (1 - (1 - (1 - u) ** 3)) * a.get("slide_px", 40) * (1 if a["in"] == "slide_up" else -1)
            alpha = u
        if a.get("out") == "fade" and v < 1:
            alpha *= max(0.0, v)
        elif a.get("out") == "pop" and v < 1:
            scale *= 0.7 + 0.3 * max(0.0, v)
            alpha *= max(0.0, v)
        return hl, reveal, round(scale, 3), round(alpha, 3), round(dy, 1)

    def patch(self, t):
        """(rgb float32 premultiplied, alpha float32, x0, y0) for time t, or None."""
        if not (self.s <= t < self.e):
            return None
        hl, reveal, scale, alpha, dy = self._state(t)
        key = (hl, reveal, scale)
        if key not in self.cache:
            img, (cx, cy) = render(self.text, self.style, self.canvas[0], highlight=hl, scale=scale, reveal=reveal)
            arr = np.asarray(img, np.float32) / 255.0
            self.cache[key] = (arr[..., 2::-1] * arr[..., 3:4], arr[..., 3], cx, cy)
        rgb, a, cx, cy = self.cache[key]
        x0 = int(round(self.canvas[0] / 2 + self.style["x"] - cx))
        y0 = int(round(self.canvas[1] / 2 + self.style["y"] - cy + dy))
        return rgb, a * alpha if alpha < 1 else a, x0, y0


def blend(canvas, rgb, a, x0, y0):
    """Alpha-blend a premultiplied patch onto a uint8 BGR canvas in place."""
    H, W = canvas.shape[:2]
    h, w = a.shape
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    pr, pa = rgb[ya - y0:yb - y0, xa - x0:xb - x0], a[ya - y0:yb - y0, xa - x0:xb - x0, None]
    region = canvas[ya:yb, xa:xb].astype(np.float32)
    canvas[ya:yb, xa:xb] = np.clip(region * (1 - pa) + pr * 255.0, 0, 255).astype(np.uint8)


def prefetch_fonts(families, weights=(400, 500, 600, 700, 800, 900), workers=8):
    """Download (cache) Google Fonts files ahead of use — setup time, not analysis time."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(f, w) for f in families for w in weights]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda fw: font_file(*fw), jobs))
