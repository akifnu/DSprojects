"""Final render: one deterministic ffmpeg graph built from a frozen revision + compile().

The browser preview and this renderer read the same compiled timeline (segments, caption
chunks, overlay/sound times), so what the creator approved is what gets exported.
"""
from pathlib import Path

from . import media, store

GRADE_FILTERS = {
    "none": "",
    "warm": "colorbalance=rs=0.06:bs=-0.06,eq=saturation=1.08",
    "cool": "colorbalance=rs=-0.05:bs=0.06,eq=saturation=0.97",
    "punchy": "eq=contrast=1.12:saturation=1.25",
    "bw": "hue=s=0,eq=contrast=1.1",
}
ANIM_D = 0.35


def ass_color(hexstr, alpha=0):
    h = hexstr.lstrip("#")
    return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def ass_time(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def ass_escape(s):
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(comp, cap, w, h):
    """One event per active-word span: the active word takes the highlight colour, emphasised
    words are italic + enlarged + highlighted (same rules as the preview)."""
    bold = -1 if cap["weight"] >= 600 else 0
    border = 3 if cap["box"] else 1
    outline_col = ass_color(cap["box_color"] if cap["box"] else cap["outline"])
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Cap,{cap['font']},{int(cap['size'])},{ass_color(cap['color'])},{ass_color(cap['hl'])},{outline_col},"
        f"{outline_col},{bold},0,0,0,100,100,0,0,{border},{cap['outline_w']:.1f},0,5,60,60,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    pos = f"{{\\an5\\pos({w // 2},{int(cap['y'] * h)})}}"
    hl = ass_color(cap["hl"])
    for c in comp["captions"]:
        ws = c["words"]
        # spans: [chunk start, first word end], then each next word start → its end/next start
        bounds = [c["t0"]] + [x["t0"] for x in ws[1:]] + [c["t1"]]
        for k in range(len(ws)):
            a, b = max(bounds[k], c["t0"]), bounds[k + 1]
            if b - a < 0.01:
                continue
            parts = []
            for j, x in enumerate(ws):
                t = ass_escape(x["text"])
                tags = ""
                if x["emph"]:
                    tags += f"\\c{hl}\\fscx115\\fscy115" + ("\\i1" if cap["emph_italic"] else "")
                elif j == k:
                    tags += f"\\c{hl}"
                parts.append(f"{{{tags}}}{t}{{\\r}}" if tags else t)
            lines.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{pos}{' '.join(parts)}")
    return "\n".join(lines) + "\n"


def _anim_xy(o, x, y, w_out, h_out):
    """Overlay position expressions for slide in/out (t is composition time)."""
    t0, t1 = o["t0"], o["t1"]
    xe, ye = str(x), str(y)
    if o["anim_in"] == "slide-left":
        xe = f"if(lt(t,{t0 + ANIM_D}),{x}+(W-{x})*(1-(t-{t0})/{ANIM_D}),{xe})"
    if o["anim_in"] == "slide-up":
        ye = f"if(lt(t,{t0 + ANIM_D}),{y}+(H-{y})*(1-(t-{t0})/{ANIM_D}),{ye})"
    if o["anim_out"] == "slide-left":
        xe = f"if(gt(t,{t1 - ANIM_D}),{x}-({x}+{w_out})*(t-{t1 - ANIM_D})/{ANIM_D},{xe})"
    if o["anim_out"] == "slide-up":
        ye = f"if(gt(t,{t1 - ANIM_D}),{y}-({y}+{h_out})*(t-{t1 - ANIM_D})/{ANIM_D},{ye})"
    return xe, ye


def build_command(src, src_info, rev, comp, assets, ass_path, out):
    st, look = rev["settings"], rev["look"]
    W, H, fps = st["w"], st["h"], st["fps"]
    amap = {a["id"]: a for a in assets}
    total = comp["duration"]
    inputs = ["-i", str(src)]
    fg = []
    n_in = 1

    # base: kept segments → concat
    segs = comp["segments"]
    has_audio = src_info["has_audio"]
    for k, sg in enumerate(segs):
        fg.append(f"[0:v]trim=start={sg['s0']}:end={sg['s1']},setpts=PTS-STARTPTS[v{k}]")
        if has_audio:
            fg.append(f"[0:a]atrim=start={sg['s0']}:end={sg['s1']},asetpts=PTS-STARTPTS[a{k}]")
        else:
            fg.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={sg['s1'] - sg['s0']}[a{k}]")
    fg.append("".join(f"[v{k}][a{k}]" for k in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=1[bv][ba]")
    z = look["zoom"]
    grade = GRADE_FILTERS[look["grade"]]
    fg.append(f"[bv]scale={int(W * z)}:{int(H * z)}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"fps={fps},setsar=1{',' + grade if grade else ''},format=yuv420p[base]")
    vlast = "base"

    extra_audio = []
    for n, o in enumerate(comp["overlays"]):
        a = amap.get(o["asset"])
        if o["issue"] or a is None:
            continue
        dur = round(o["t1"] - o["t0"], 3)
        f = str(store.asset_file(a))
        if a["type"] == "image":
            inputs += ["-loop", "1", "-t", str(dur), "-i", f]
        elif a["type"] == "gif":
            inputs += ["-ignore_loop", "0", "-t", str(dur), "-i", f]
        else:
            inputs += ["-ss", str(o["src_in"]), "-t", str(dur), "-i", f]
        i = n_in
        n_in += 1
        chain = f"[{i}:v]"
        if o["mode"] == "broll":
            chain += f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={fps},setsar=1,format=rgba"
            ow, oh, x, y = W, H, 0, 0
        else:
            ow = int(W * o["scale"]) // 2 * 2
            chain += f"scale={ow}:-2,fps={fps},format=rgba"
            if a.get("width") and a.get("height"):
                oh = int(ow * a["height"] / a["width"])
            else:
                oh = ow
            x, y = int(o["x"] * W - ow / 2), int(o["y"] * H - oh / 2)
        if o["anim_in"] in ("spin", "pop") or o["anim_out"] in ("spin",):
            spin = f"2*PI*min(t/{ANIM_D},1)" if o["anim_in"] == "spin" else "0"
            if o["anim_out"] == "spin":
                spin += f"+2*PI*max(0,(t-{dur - ANIM_D})/{ANIM_D})"
            chain += f",rotate=a='{spin}':c=none:ow='hypot(iw,ih)':oh=ow"
            d = int((ow * ow + oh * oh) ** 0.5)
            x, y, ow, oh = x - (d - ow) // 2, y - (d - oh) // 2, d, d
        if o["anim_in"] in ("fade", "pop"):
            chain += f",fade=t=in:st=0:d={ANIM_D}:alpha=1"
        if o["anim_out"] in ("fade", "pop"):
            chain += f",fade=t=out:st={max(0, dur - ANIM_D)}:d={ANIM_D}:alpha=1"
        chain += f",setpts=PTS-STARTPTS+{o['t0']}/TB[ov{n}]"
        fg.append(chain)
        xe, ye = _anim_xy(o, x, y, ow, oh)
        fg.append(f"[{vlast}][ov{n}]overlay=x='{xe}':y='{ye}':eof_action=pass:"
                  f"enable='between(t,{o['t0']},{o['t1']})'[vo{n}]")
        vlast = f"vo{n}"
        if o["mode"] == "broll" and o["audio"] and a["type"] == "video" and a.get("has_audio"):
            fg.append(f"[{i}:a]asetpts=PTS-STARTPTS,adelay={int(o['t0'] * 1000)}:all=1[ova{n}]")
            extra_audio.append(f"ova{n}")

    if comp["captions"]:
        p = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        fg.append(f"[{vlast}]ass='{p}'[vcap]")
        vlast = "vcap"
    fg.append(f"[{vlast}]format=yuv420p[vout]")

    # audio: narration + sounds (ducked under narration when requested)
    fg.append(f"[ba]volume={st['narration_db']}dB,asplit=2[nar][side]")
    duck, plain = [], []
    for n, s in enumerate(comp["sounds"]):
        a = amap.get(s["asset"])
        if a is None or s["t0"] is None or s["t1"] is None:
            continue
        length = round(s["t1"] - s["t0"], 3)
        if length <= 0:
            continue
        if s["loop"]:
            inputs += ["-stream_loop", "-1"]
        inputs += ["-i", str(store.asset_file(a))]
        i = n_in
        n_in += 1
        ch = (f"[{i}:a]atrim=start={s['trim_in']}:duration={length + s['trim_in'] if s['loop'] else length},"
              f"asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=stereo")
        if s["loop"]:
            ch += f",atrim=duration={length}"
        if s["fade_in"]:
            ch += f",afade=t=in:d={s['fade_in']}"
        if s["fade_out"]:
            ch += f",afade=t=out:st={max(0, length - s['fade_out'])}:d={s['fade_out']}"
        ch += f",volume={s['gain_db']}dB,adelay={int(s['t0'] * 1000)}:all=1[snd{n}]"
        fg.append(ch)
        (duck if s["duck"] else plain).append(f"snd{n}")
    mix = ["nar"]
    if duck:
        fg.append("".join(f"[{x}]" for x in duck) + f"amix=inputs={len(duck)}:normalize=0:duration=longest[dbus]")
        fg.append("[dbus][side]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=300[ducked]")
        mix.append("ducked")
    else:
        fg.append("[side]anullsink")
    mix += plain + extra_audio
    if len(mix) > 1:
        fg.append("".join(f"[{x}]" for x in mix) + f"amix=inputs={len(mix)}:normalize=0:duration=first,"
                  "alimiter=limit=0.95[aout]")
    else:
        fg.append("[nar]alimiter=limit=0.95[aout]")

    script = Path(out).with_suffix(".filtergraph.txt")
    script.write_text(";\n".join(fg))
    return ["ffmpeg", "-y", "-v", "error", "-stats_period", "1", "-progress", "pipe:1", *inputs,
            "-/filter_complex", str(script), "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", str(total), "-movflags", "+faststart", str(out)]


def preflight(rev, comp, assets):
    """Blocking problems and warnings before a final render."""
    blocking = [i["msg"] + f" ({i['kind']})" for i in comp["issues"]
                if i["kind"] in ("overlay", "sound", "composition")]
    warnings = [i["msg"] for i in comp["issues"] if i["kind"] in ("word", "sound-warn")]
    if comp["captions"] and not media.font_available(rev["caption"]["font"]):
        warnings.append(f"Font “{rev['caption']['font']}” is not installed on the renderer; a fallback font will be used.")
    return blocking, warnings


def validate_output(path, comp, st):
    info = media.probe(path)
    problems = []
    if not info["has_video"] or not info["has_audio"]:
        problems.append("missing audio or video stream")
    if (info.get("width"), info.get("height")) != (st["w"], st["h"]):
        problems.append(f"dimensions {info.get('width')}×{info.get('height')}")
    if abs(info["duration"] - comp["duration"]) > max(0.25, 0.02 * comp["duration"]):
        problems.append(f"duration {info['duration']:.2f}s, expected {comp['duration']:.2f}s")
    return info, problems
