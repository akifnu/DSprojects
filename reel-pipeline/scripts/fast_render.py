"""Renders a FableCut project.json straight with ffmpeg — no browser.

    python3 fast_render.py <project.json> <out.mp4> [--lufs -14]
    python3 fast_render.py <project.json> <still.jpg> --frame <seconds>   (preview one frame)

The browser export draws every frame in software here (~1 frame/s); this
builds one ffmpeg filtergraph instead and runs at many times realtime. It
covers what talking-head / screen-recording reels use:

  video/image clips  fit (cover/contain/stretch), static scale / x / y / opacity
  audio              audio clips, and video clips' own sound (volume, channelMode mono)
  text clips         font, weight, size, colour, per-line pill (bgColor/bgOpacity),
                     stroke, align, uppercase, x / y, lineHeight — no textAnim
  delivery           1080×1920-style H.264 CRF 19 / AAC 256k, two-pass loudnorm

Anything else (keyframes, grades, filters, transitions, masks, text animation,
speed, SVG, comps…) is listed and the script exits 2 — build the effect another
way (or add it here) rather than reaching for the editor's browser export.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(HERE, "..", "workspace")
FONT_DIR = os.path.join(WORKSPACE, "fonts")
LIB_FONTS = os.path.join(HERE, "..", ".fablecut", "library", "fonts")

# FableCut prop defaults (the editor writes every prop into every clip on load)
DEFAULTS = {"x": 0, "y": 0, "scale": 1, "rotation": 0, "opacity": 1, "volume": 1, "pan": 0, "anchorX": 0,
            "anchorY": 0, "trimStart": 0, "trimEnd": 100, "trimOffset": 0, "autoOrient": False, "pathOffset": 0,
            "pathAlign": "start", "motionBlur": 0, "motionBlurSamples": 8, "gain": 0, "channelMode": "stereo",
            "duck": 0, "speed": 1, "brightness": 100, "contrast": 100, "saturation": 100, "hue": 0, "blur": 0,
            "grayscale": 0, "sepia": 0, "invert": 0, "temperature": 0, "tint": 0, "vignette": 0,
            "filterPreset": "none", "fit": "contain", "cropL": 0, "cropT": 0, "cropR": 0, "cropB": 0,
            "flipH": False, "flipV": False, "cornerRadius": 0, "blend": "normal", "chromaKey": "",
            "chromaTolerance": 26, "chromaSoftness": 12, "bgRemove": False, "shake": 0, "shakeSpeed": 8,
            "rgbSplit": 0, "grain": 0, "text": "Title", "fontSize": 72, "color": "#ffffff", "color2": "",
            "font": "Segoe UI", "bold": True, "italic": False, "weight": 0, "align": "center", "letterSpacing": 0,
            "lineHeight": 1.2, "uppercase": False, "textShadow": 12, "glow": 0, "glowColor": "", "textAnim": "none",
            "wordRate": 0.15, "direction": "auto", "strokeWidth": 0, "strokeColor": "#000000", "bgColor": "#000000",
            "bgOpacity": 0, "boxW": 0, "boxH": 0, "boxFit": False, "vAlign": "middle"}
AUDIO = {"volume", "gain", "pan", "channelMode", "duck", "speed"}
TEXT = {"text", "fontSize", "color", "color2", "font", "bold", "italic", "weight", "align", "letterSpacing",
        "lineHeight", "uppercase", "textShadow", "glow", "glowColor", "textAnim", "wordRate", "direction",
        "strokeWidth", "strokeColor", "bgColor", "bgOpacity", "boxW", "boxH", "boxFit", "vAlign"}
VISUAL = set(DEFAULTS) - AUDIO - TEXT
# which props matter for a kind, and which of those this renderer draws
RELEVANT = {"video": VISUAL | AUDIO, "image": VISUAL, "audio": AUDIO, "text": VISUAL | TEXT}
SUPPORTED = {"fit", "scale", "x", "y", "opacity", "volume", "gain", "channelMode",
             "text", "fontSize", "color", "font", "bold", "weight", "align", "lineHeight", "uppercase",
             "strokeWidth", "strokeColor", "bgColor", "bgOpacity", "wordRate", "direction", "textShadow"}


def media_path(src):
    if src.startswith("/media/"):
        return os.path.join(WORKSPACE, "media", src[len("/media/"):])
    if src.startswith("/library/"):
        return os.path.join(HERE, "..", ".fablecut", "library", src[len("/library/"):])
    raise SystemExit(f"unsupported media src {src}")


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    streams = json.loads(out)["streams"]
    v = next((s for s in streams if s["codec_type"] == "video"), None)
    return {"w": v and v["width"], "h": v and v["height"], "audio": any(s["codec_type"] == "audio" for s in streams)}


def font_file(family, weight):
    """A TTF for a Google Font family at a weight, cached in workspace/fonts."""
    os.makedirs(FONT_DIR, exist_ok=True)
    path = os.path.join(FONT_DIR, f"{family.replace(' ', '_')}-{weight}.ttf")
    if os.path.exists(path):
        return path
    try:
        url = f"https://fonts.googleapis.com/css2?family={family.replace(' ', '+')}:wght@{weight}"
        css = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/4.0"})).read().decode()
        ttf = re.search(r"url\((https://[^)]+\.ttf)\)", css).group(1)
        urllib.request.urlretrieve(ttf, path)
        return path
    except Exception:
        lib = os.path.join(LIB_FONTS, f"{family}.woff2")
        if os.path.exists(lib):
            return lib
        print(f"warning: font {family} unavailable, using Liberation Sans", file=sys.stderr)
        return "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def hex_rgba(h, a=1.0):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (round(255 * a),)


def render_text(p, W, H, out):
    """FableCut-style text: centred on (W/2+x, H/2+y), one pill per line."""
    text = p.get("text", "Title")
    if p.get("uppercase"):
        text = text.upper()
    size = p.get("fontSize", 72)
    weight = p.get("weight") or (700 if p.get("bold", True) else 400)
    font = ImageFont.truetype(font_file(p.get("font", "Inter"), weight), size)
    lines = text.split("\n")
    pitch = size * p.get("lineHeight", 1.2)
    pad_x, radius = round(size * 0.4), round(size * 0.2)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = W / 2 + p.get("x", 0), H / 2 + p.get("y", 0)
    top = cy - pitch * len(lines) / 2
    widths = [d.textlength(l, font=font) for l in lines]
    box = max(widths)
    for i, (line, w) in enumerate(zip(lines, widths)):
        align = p.get("align", "center")
        x0 = {"left": cx - box / 2, "right": cx + box / 2 - w}.get(align, cx - w / 2)
        y0 = top + i * pitch
        if p.get("bgOpacity", 0) > 0:
            d.rounded_rectangle([x0 - pad_x, y0, x0 + w + pad_x, y0 + pitch], radius,
                                fill=hex_rgba(p.get("bgColor", "#000000"), p["bgOpacity"]))
        d.text((x0, y0 + pitch / 2), line, font=font, anchor="lm", fill=hex_rgba(p.get("color", "#ffffff")),
               stroke_width=round(p.get("strokeWidth", 0)), stroke_fill=hex_rgba(p.get("strokeColor", "#000000")))
    if p.get("opacity", 1) < 1:
        img.putalpha(img.getchannel("A").point(lambda a: round(a * p["opacity"])))
    img.save(out)


def check(clip):
    props = clip.get("props") or {}
    bad = [f"{k}={v!r}" for k, v in props.items()
           if k in RELEVANT[clip["kind"]] and k not in SUPPORTED and v != DEFAULTS.get(k)]
    if clip["kind"] == "text" and props.get("textShadow", DEFAULTS["textShadow"]) != 0:
        bad.append("textShadow (default 12; set 0 for the fast path)")
    if props.get("fit", "contain") not in ("cover", "contain", "stretch"):
        bad.append("fit")
    if clip.get("keyframes"):
        bad.append("keyframes")
    for k in ("transitionIn", "transitionOut", "masks", "fx", "grade", "parent", "fadeIn", "fadeOut", "matte"):
        if clip.get(k) or props.get(k):
            bad.append(k)
    return [f"{clip['id']}: {b}" for b in bad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    ap.add_argument("--lufs", type=float, default=-14)
    ap.add_argument("--frame", type=float, help="render only the picture at this time, as a still")
    a = ap.parse_args()
    proj = json.load(open(a.project))
    W, H, fps = proj["width"], proj["height"], proj.get("fps", 30)
    media = {m["id"]: m for m in proj.get("media", [])}
    off = set(proj.get("disabledTracks") or [])
    clips = [c for c in proj["clips"] if not c.get("disabled") and c["track"] not in off]
    total = max(c["start"] + c["duration"] for c in clips)

    problems = []
    for c in clips:
        problems += check(c) if c["kind"] in RELEVANT else [f"{c['id']}: kind {c['kind']}"]
    for k in ("comps", "buses", "master"):
        if proj.get(k):
            problems.append(f"project.{k}")
    if any(t.get("gain") or t.get("fx") for t in proj.get("tracks") or []):
        problems.append("track gain / fx")
    if problems:
        print("fast_render can't render these props / features yet:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(2)
    if a.frame is not None:  # a still needs only the pictures on screen then
        clips = [c for c in clips if c["kind"] != "audio" and c["start"] <= a.frame < c["start"] + c["duration"]]

    tmp = tempfile.mkdtemp(prefix="fastrender-")
    probes = {}
    inputs, vf, af, alabels = [], [], [], []

    def add_input(args):
        inputs.extend(args)
        return sum(1 for x in inputs if x == "-i") - 1

    # picture: lower tracks first, V3 over V2 over V1
    def track_no(c):
        return int(re.sub(r"\D", "", c["track"]) or 0)

    vf.append(f"color=c={proj.get('background', '#000000')}:s={W}x{H}:r={fps}:d={total:.3f},format=yuv420p[base]")
    last = "base"
    visual = sorted([c for c in clips if c["kind"] in ("video", "image", "text")], key=lambda c: (track_no(c), c["start"]))
    for n, c in enumerate(visual):
        p, s, e = c.get("props") or {}, c["start"], c["start"] + c["duration"]
        enable = f"between(t\\,{s:.3f}\\,{e - 0.0005:.3f})"
        if c["kind"] == "text":
            png = os.path.join(tmp, f"t{n}.png")
            render_text(p, W, H, png)
            k = add_input(["-i", png])
            vf.append(f"[{last}][{k}:v]overlay=0:0:enable='{enable}'[o{n}]")
        else:
            path = media_path(media[c["mediaId"]]["src"])
            info = probes.setdefault(path, probe(path))
            if c["kind"] == "image":
                k = add_input(["-loop", "1", "-t", f"{c['duration']:.3f}", "-i", path])
            else:
                k = add_input(["-ss", f"{c.get('in', 0):.3f}", "-t", f"{c['duration']:.3f}", "-i", path])
            fit, sc = p.get("fit", "contain"), p.get("scale", 1)
            if fit == "stretch":
                ow, oh = W * sc, H * sc
            else:
                f = (max if fit == "cover" else min)(W / info["w"], H / info["h"])
                ow, oh = info["w"] * f * sc, info["h"] * f * sc
            ow, oh = 2 * round(ow / 2), 2 * round(oh / 2)
            x, y = round(W / 2 + p.get("x", 0) - ow / 2), round(H / 2 + p.get("y", 0) - oh / 2)
            alpha = f",format=rgba,colorchannelmixer=aa={p['opacity']}" if p.get("opacity", 1) < 1 else ""
            vf.append(f"[{k}:v]fps={fps},scale={ow}:{oh}:flags=lanczos,setsar=1{alpha},"
                      f"setpts=PTS-STARTPTS+{s:.3f}/TB[c{n}]")
            vf.append(f"[{last}][c{n}]overlay={x}:{y}:eof_action=pass:enable='{enable}'[o{n}]")
        last = f"o{n}"
    vf.append(f"[{last}]format=yuv420p[vout]")

    # sound: audio clips + video clips' own sound
    for n, c in enumerate(c for c in clips if c["kind"] in ("audio", "video") and a.frame is None):
        p = c.get("props") or {}
        vol = p.get("volume", 1) * 10 ** (p.get("gain", 0) / 20)
        path = media_path(media[c["mediaId"]]["src"])
        info = probes.setdefault(path, probe(path))
        if vol == 0 or not info["audio"]:
            continue
        k = add_input(["-ss", f"{c.get('in', 0):.3f}", "-t", f"{c['duration']:.3f}", "-i", path])
        mono = "aformat=channel_layouts=mono," if p.get("channelMode") == "mono" else ""
        ms = round(c["start"] * 1000)
        af.append(f"[{k}:a]{mono}aformat=sample_rates=48000:channel_layouts=stereo,volume={vol},"
                  f"adelay={ms}|{ms},apad[a{n}]")
        alabels.append(f"[a{n}]")
    if af:
        af.append(f"{''.join(alabels)}amix=inputs={len(alabels)}:normalize=0:duration=longest,atrim=0:{total:.3f}[aout]")

    # pass 1: the mix to WAV, measured
    wav = os.path.join(tmp, "mix.wav")
    if af:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(af),
                        "-map", "[aout]", "-t", f"{total:.3f}", wav], check=True)
        m = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav, "-af",
                            f"loudnorm=I={a.lufs}:TP=-1:LRA=11:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
        j = json.loads(m[m.rindex("{"):])
        norm = (f"loudnorm=I={a.lufs}:TP=-1:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
                f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:"
                f"linear=true,aresample=48000")

    if a.frame is not None:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(vf),
                        "-map", "[vout]", "-ss", f"{a.frame:.3f}", "-frames:v", "1", "-q:v", "3", a.out], check=True)
        print(a.out)
        return

    # pass 2: picture + normalized mix
    cmd = ["ffmpeg", "-loglevel", "error", "-y", *inputs]
    if af:
        cmd += ["-i", wav]
        wi = sum(1 for x in cmd if x == "-i") - 1
        vf.append(f"[{wi}:a]{norm}[anorm]")
    cmd += ["-filter_complex", ";".join(vf), "-map", "[vout]"]
    if af:
        cmd += ["-map", "[anorm]", "-c:a", "aac", "-b:a", "256k", "-ar", "48000"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "19", "-maxrate", "25M", "-bufsize", "50M",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-g", str(round(fps)), "-movflags", "+faststart",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-t", f"{total:.3f}", a.out]
    subprocess.run(cmd, check=True)
    print(a.out)


if __name__ == "__main__":
    main()
