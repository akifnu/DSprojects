"""Reproducible render benchmark.

    python3 bench.py                      # standard 30 s 720×1280 reel, final quality
    python3 bench.py --vmaf               # + VMAF against a lossless render of the same timeline
    python3 bench.py --source clip.mp4    # use real footage (looped) instead of the synthetic source
    python3 bench.py --sweep              # x264 preset/CRF sweep: speed vs VMAF
    python3 bench.py --batch              # reels/hour for concurrent renders × workers

The standard reel exercises everything a production reel uses: 14 cuts with punch-ins,
a keyframed push-in, dissolve / whip / zoom transitions, word captions with outline,
shadow, active-word highlight and pop-in, a sticker, voice + music with ducking, a
3D-LUT grade with sharpening. Fixtures are generated once into data/bench/.

Reported: wall seconds, CPU seconds (this process + children), frames/s, × real time,
and VMAF (harmonic mean, min) when asked.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reelkit import ff, grade, render  # noqa: E402
from reelkit.paths import DATA, FFMPEG  # noqa: E402

BENCH = DATA / "bench"
W, H, FPS, DUR = 720, 1280, 30, 30.0
WORDS = ("this one simple trick changes how you edit every single reel forever so watch closely "
         "because most creators never learn what actually keeps viewers hooked until the very end").split()


def cpu_seconds():
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def _ff(*args):
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def fixtures(source=None):
    BENCH.mkdir(parents=True, exist_ok=True)
    src = BENCH / ("src_synthetic.mp4" if not source else f"src_{Path(source).stem}.mp4")
    if not src.exists():
        if source:
            _ff("-stream_loop", "-1", "-i", source, "-t", 40, "-an", "-vf",
                "fps=30,scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                "-c:v", "libx264", "-preset", "fast", "-crf", 14, "-g", 30, "-pix_fmt", "yuv420p", src)
        else:
            #             # moving colour fields (rendered small, upscaled: they are smooth anyway), a moving
            # test pattern and light camera-like grain — a typical 1080p mezzanine bitrate
            _ff("-f", "lavfi", "-i", "gradients=s=270x480:r=30:d=40:speed=0.012:nb_colors=4:"
                "c0=0x24476b:c1=0xd99a5b:c2=0x3f8f5f:c3=0xe8e2d0",
                "-f", "lavfi", "-i", "testsrc2=s=600x600:r=30:d=40",
                "-filter_complex", "[0]scale=1080:1920:flags=bicubic[bg];"
                "[bg][1]overlay=x='240+200*sin(t*0.7)':y='660+320*cos(t*0.45)',"
                "noise=alls=3:allf=t,format=yuv420p",
                "-c:v", "libx264", "-preset", "fast", "-crf", 16, "-g", 30, src)
    voice = BENCH / "voice.wav"
    if not voice.exists():
        _ff("-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.4:d=40:r=48000",
            "-af", "bandpass=f=900:width_type=h:w=1800,volume='0.25+0.75*abs(sin(2*PI*2.7*t))':eval=frame", voice)
    music = BENCH / "music.wav"
    if not music.exists():
        _ff("-f", "lavfi", "-i", "aevalsrc='0.2*sin(2*PI*220*t)+0.15*sin(2*PI*277.2*t)+0.12*sin(2*PI*329.6*t)"
            "+0.05*sin(2*PI*55*t)*lt(mod(t,0.5),0.08)':d=40:s=48000", music)
    sticker = BENCH / "sticker.png"
    if not sticker.exists():
        import cv2
        img = np.zeros((240, 240, 4), np.uint8)
        cv2.circle(img, (120, 120), 110, (40, 70, 245, 240), -1, cv2.LINE_AA)
        cv2.putText(img, "NEW", (52, 140), cv2.FONT_HERSHEY_DUPLEX, 1.9, (255, 255, 255, 255), 4, cv2.LINE_AA)
        cv2.imwrite(str(sticker), img)
    lut = BENCH / "look.cube"
    if not lut.exists():
        g = np.linspace(0, 1, 33)
        R, G, B = np.meshgrid(g, g, g, indexing="ij")
        rgb = np.stack([R, G, B], -1)
        s = 1 / (1 + np.exp(-(rgb - 0.5) * 6))                  # contrast S-curve
        s = (s - s.min()) / (s.max() - s.min())
        warm = s * np.array([1.04, 1.0, 0.93]) + np.array([0.0, 0.0, 0.03]) * (1 - s)
        grade.write_cube(np.clip(warm, 0, 1), lut)
    return src, voice, music, sticker, lut


def timeline(src, voice, music, sticker, lut, encode):
    video, audio, t, i = [], [], 0.0, 0
    durs = [2.2, 1.9, 2.4, 3.1, 1.8, 2.0, 2.6, 1.7, 2.2, 2.0, 2.4, 1.9, 2.1, 1.7]
    durs[-1] = round(DUR - sum(durs[:-1]), 3)
    for i, d in enumerate(durs):
        c = {"media": "src", "start": round(t, 3), "in": round(1.0 + i * 2.6, 3), "dur": d, "fit": "cover",
             "scale": 1.25 if i % 2 else 1.0, "x": 40 if i % 2 else 0, "y": -30 if i % 2 else 0}
        if i == 3:
            c["scale"] = [[0, 1.0], [d, 1.12, "inout"]]
        if i == 5:
            c["transition_in"] = {"type": "dissolve", "dur": 0.3}
        if i == 9:
            c["transition_in"] = {"type": "whip_left", "dur": 0.25}
        if i == 12:
            c["transition_in"] = {"type": "zoom", "dur": 0.3}
        video.append(c)
        audio.append({"media": "voice", "start": round(t, 3), "in": round(1.0 + i * 2.6, 3), "dur": d,
                      "role": "voice", "mono": True})
        t += d
    video.append({"media": "sticker", "start": 4.0, "dur": 2.5, "track": 2, "x": 200, "y": -420,
                  "scale": [[0, 0.2], [0.35, 1.0, "out_back"]], "rotation": [[0, -20], [0.35, 8, "out"]]})
    caps, k, ts = [], 0, 0.15
    while ts < DUR - 0.6:
        ws = []
        for j in range(3):
            w = WORDS[(k + j) % len(WORDS)]
            ws.append({"w": w, "s": round(ts + 0.32 * j, 3), "e": round(ts + 0.32 * (j + 1), 3)})
        caps.append({"s": ws[0]["s"], "e": ws[-1]["e"], "text": " ".join(x["w"] for x in ws), "words": ws})
        k += 3
        ts += 0.96
    return {"width": W, "height": H, "fps": FPS, "duration": DUR, "background": "#000000",
            "media": {"src": str(src), "voice": str(voice), "music": str(music), "sticker": str(sticker)},
            "video": video, "captions": caps,
            "styles": {"caption": {"font": "Montserrat", "weight": 900, "size": 46, "case": "upper",
                                   "stroke": {"color": "#000000", "width": 4},
                                   "shadow": {"opacity": 0.45, "blur": 5, "dy": 3},
                                   "highlight": {"fill": "#ffe600", "scale": 1.06}, "y": 170,
                                   "anim": {"in": "pop", "in_dur": 0.16}}},
            "grade": {"lut": str(lut), "sharpen": 0.25},
            "audio": {"clips": audio + [{"media": "music", "start": 0, "in": 0, "dur": DUR, "role": "music",
                                         "gain_db": -8}], "duck_db": 12, "loudness": -14},
            "encode": encode}


def run(tl, out, **kw):
    c0, w0 = cpu_seconds(), time.monotonic()
    render.render(tl, out, **kw)
    wall, cpu = time.monotonic() - w0, cpu_seconds() - c0
    n = int(subprocess.run([ff.FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                            "stream=nb_read_packets", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True).stdout.strip() or 0)
    return {"wall": round(wall, 2), "cpu": round(cpu, 1), "frames": n, "fps": round(n / wall, 1),
            "x_realtime": round(DUR / wall, 2)}


def vmaf(test, ref):
    log = BENCH / "vmaf.json"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(test), "-i", str(ref), "-lavfi",
                    f"[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]libvmaf=n_threads=4:"
                    f"log_fmt=json:log_path={log}", "-f", "null", "-"], check=True)
    p = json.loads(log.read_text())["pooled_metrics"]["vmaf"]
    return {"vmaf": round(p["harmonic_mean"], 2), "vmaf_min": round(p["min"], 2)}


def reference(tl, tag):
    ref = BENCH / f"ref_{tag}.mp4"
    if not ref.exists():
        rtl = dict(tl, encode={"lossless": True})
        render.render(rtl, ref)
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--crf", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None, help="parallel compositing workers (render default if unset)")
    ap.add_argument("--vmaf", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--single", help=argparse.SUPPRESS)
    a = ap.parse_args()
    fx = fixtures(a.source)
    tag = "synthetic" if not a.source else Path(a.source).stem
    enc = {k: v for k, v in (("preset", a.preset), ("crf", a.crf)) if v is not None}
    tl = timeline(*fx, encode=enc)
    kw = {"workers": a.workers} if a.workers is not None else {}
    if a.single:                       # one render inside a --batch run
        print(json.dumps(run(tl, a.single, **kw)))
        return
    if a.sweep:
        ref = reference(tl, tag)
        print(f"{'preset':10s} {'crf':>4s} {'wall':>6s} {'cpu':>6s} {'fps':>6s} {'xRT':>5s} {'vmaf':>6s} {'min':>6s} {'MB':>5s}")
        for preset in ("veryfast", "faster", "fast", "medium", "slow"):
            for crf in (16, 17, 18, 19, 20):
                t = dict(tl, encode={"preset": preset, "crf": crf})
                out = BENCH / f"sweep_{preset}_{crf}.mp4"
                r = run(t, out, **kw)
                v = vmaf(out, ref)
                print(f"{preset:10s} {crf:4d} {r['wall']:6.1f} {r['cpu']:6.1f} {r['fps']:6.1f} {r['x_realtime']:5.2f} "
                      f"{v['vmaf']:6.2f} {v['vmaf_min']:6.2f} {out.stat().st_size / 1e6:5.1f}", flush=True)
        return
    if a.batch:
        print(f"{'concurrent':>10s} {'workers':>7s} {'wall':>6s} {'reels/h':>8s}")
        for conc, workers in ((1, 4), (2, 2), (2, 1), (4, 1), (3, 1), (1, 3), (1, 2)):
            w0 = time.monotonic()
            procs = [subprocess.Popen([sys.executable, __file__, "--single", str(BENCH / f"batch_{i}.mp4"),
                                       "--workers", str(workers)] + (["--source", a.source] if a.source else []),
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) for i in range(conc)]
            for p in procs:
                p.communicate()
            wall = time.monotonic() - w0
            print(f"{conc:10d} {workers:7d} {wall:6.1f} {conc * 3600 / wall:8.0f}", flush=True)
        return
    out = BENCH / f"bench_{tag}.mp4"
    r = run(tl, out, **kw)
    if a.vmaf:
        r.update(vmaf(out, reference(tl, tag)))
    r["size_mb"] = round(out.stat().st_size / 1e6, 2)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
