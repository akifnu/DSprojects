"""Run with: python3 -m unittest discover -s reel-pipeline/tests -v."""
import array
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fast_render", ROOT / "scripts/fast_render.py")
render = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render)
RUN = subprocess.run


def ffmpeg(*args):
    return RUN(["ffmpeg", "-v", "error", "-y", "-filter_complex_threads", "2", *args],
               check=True, capture_output=True)


class RenderTests(unittest.TestCase):
    def test_missing_gpu_falls_back_only_in_auto_mode(self):
        error = subprocess.CalledProcessError(1, "ffmpeg", stderr="no NVIDIA device")
        with patch.object(render.subprocess, "run", side_effect=error):
            self.assertEqual(render.select_encoder("auto"), "cpu")
            with self.assertRaisesRegex(SystemExit, "NVIDIA encoding unavailable"):
                render.select_encoder("nvenc")
        with patch.object(render.subprocess, "run") as run:
            self.assertEqual(render.select_encoder("cpu"), "cpu")
            run.assert_not_called()

    def test_successful_gpu_check_selects_nvenc(self):
        with patch.object(render.subprocess, "run", return_value=SimpleNamespace()):
            self.assertEqual(render.select_encoder("auto"), "nvenc")

    def test_gpu_check_timeout_falls_back(self):
        with patch.object(render.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffmpeg", 15)):
            self.assertEqual(render.select_encoder("auto"), "cpu")

    def test_video_only_and_preview_cli_clean_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            # Resolve fixture media locally; exercise main() and its cleanup.
            source = directory / "still.png"
            render.Image.new("RGB", (180, 320), "red").save(source)
            project = dict(width=90, height=160, fps=30, media=[dict(id="m", src="/media/still.png")],
                           clips=[dict(id="i", kind="image", track="V1", mediaId="m", start=0,
                                       duration=0.8, props=dict(fit="contain"))])
            json_file = directory / "project.json"
            json_file.write_text(json.dumps(project))
            real_temporary_directory = tempfile.TemporaryDirectory
            def temp_directory(**kwargs):
                return real_temporary_directory(dir=directory, **kwargs)
            for name, options in [("out.mp4", ["--encoder", "cpu"]),
                                  ("preview.jpg", ["--frame", "0.4", "--encoder", "nvenc"])]:
                with self.subTest(output=name), patch.object(render, "media_path", return_value=str(source)), \
                        patch.object(sys, "argv", ["fast_render.py", str(json_file), str(directory/name),
                                                  "--threads", "2", *options]), \
                        patch.object(render.tempfile, "TemporaryDirectory", side_effect=temp_directory):
                    render.main()
                    self.assertTrue((directory / name).is_file())
                    self.assertFalse(any(p.name.startswith("fastrender-") for p in directory.iterdir()))

    def test_failed_font_lookup_is_bounded_and_cached(self):
        render.font_file.cache_clear()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(render, "FONT_DIR", directory), \
                patch.object(render.urllib.request, "urlopen", side_effect=TimeoutError) as fetch:
            first = render.font_file("UnavailableFont", 700)
            self.assertEqual(render.font_file("UnavailableFont", 700), first)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(fetch.call_args.kwargs["timeout"], 5)
        render.font_file.cache_clear()

    def test_job1_probes_once_and_opens_only_stage_inputs(self):
        project = json.loads((ROOT / "jobs/job1/project-1.json").read_text())
        args = SimpleNamespace(frame=None, encoder="cpu", threads=2, lufs=-14,
                               cpu_preset="fast", out="unused.mp4")
        commands = []
        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            if cmd[0] == "ffprobe":
                return SimpleNamespace(stdout=json.dumps(dict(streams=[
                    dict(codec_type="video", width=1080, height=1920), dict(codec_type="audio")])))
            return SimpleNamespace(stderr=json.dumps(dict(input_i=-25, input_tp=-5, input_lra=2,
                                                          input_thresh=-35, target_offset=0)))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(render, "render_text"), patch.object(render.subprocess, "run", side_effect=fake_run):
            render.render_project(project, args, directory)
        self.assertEqual(sum(c[0] == "ffprobe" for c in commands), 1)
        stages = [c for c in commands if c[0] == "ffmpeg"]
        # audio mix 8, loudness 1, video 9 (captions mocked), normalization 1, mux 2;
        # video and audio run concurrently, so compare without order
        self.assertEqual(sorted(c.count("-i") for c in stages), [1, 1, 2, 8, 9])

    def test_scale_project_to_720p_scales_positions_and_text(self):
        project = dict(width=1080, height=1920, clips=[
            dict(id="t", props=dict(x=30, y=80, fontSize=56, strokeWidth=6, scale=2.0, text="Hi"))])
        scaled = render.scale_project(project, 720)
        self.assertEqual((scaled["width"], scaled["height"]), (720, 1280))
        p = scaled["clips"][0]["props"]
        for key, want in dict(x=20, y=80 * 2 / 3, fontSize=56 * 2 / 3, strokeWidth=4).items():
            self.assertAlmostEqual(p[key], want)
        self.assertEqual(p["scale"], 2.0)          # fit-relative, unchanged
        self.assertEqual(project["width"], 1080)   # input untouched
        self.assertIs(render.scale_project(project, 1080), project)

    def test_crop_preserves_framing_and_lanczos_pixels(self):
        # Compare decoded pixels to the original scale-then-overlay implementation.
        for iw, ih, ow, oh, x, y in [(180, 320, 360, 640, -120, -202),
                                      (180, 320, 324, 576, -60, -121),
                                      (180, 320, 180, 320, 10, 30),
                                      (180, 320, 90, 160, 0, 0),
                                      (180, 320, 360, 640, 20, -201)]:
            with self.subTest(geometry=(ow, oh, x, y)):
                transform, ox, oy = render.visible_scale(iw, ih, ow, oh, x, y, 90, 160)
                def picture(scale, px, py):
                    graph = (f"color=c=white:s=90x160:r=30:d=0.04[base];"
                             f"[0:v]{scale},setsar=1[v];[base][v]overlay={px}:{py},format=yuv420p[out]")
                    return ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={iw}x{ih}:r=30:d=0.04",
                                  "-filter_complex", graph, "-map", "[out]", "-frames:v", "1",
                                  "-threads", "2", "-f", "rawvideo", "-").stdout
                reference = picture(f"scale={ow}:{oh}:flags=lanczos", x, y)
                optimized = picture(transform, ox, oy)
                self.assertEqual(len(optimized), len(reference))
                errors = [abs(a-b) for a, b in zip(reference, optimized)]
                self.assertLess(sum(errors) / len(errors), 0.5)
                # The visible pattern stays registered, including odd overlay offsets.
                self.assertLess(sum(e > 8 for e in errors) / len(errors), 0.01)

    def test_real_render_with_audio_gap_captions_and_shared_media(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "media").mkdir()
            source = directory / "media/source.mp4"
            ffmpeg("-f", "lavfi", "-i", "testsrc2=s=180x320:r=30:d=2.4", "-f", "lavfi", "-i",
                   "sine=frequency=440:sample_rate=48000:duration=2.4", "-c:v", "libx264", "-preset",
                   "ultrafast", "-threads", "2", "-c:a", "aac", "-shortest", str(source))
            def clip(ident, kind, start, seek, duration, props):
                return dict(id=ident, kind=kind, mediaId="m", track="A1" if kind == "audio" else "V1",
                            start=start, duration=duration, **{"in": seek}, props=props)
            project = dict(width=90, height=160, fps=30, background="#ffffff",
                           media=[dict(id="m", src="/media/source.mp4")], clips=[
                clip("v0", "video", 0, 0.1, 0.6, dict(volume=0, fit="cover", scale=2, x=4, y=-12)),
                clip("v1", "video", 0.9, 1.1, 0.6, dict(volume=0, fit="contain")),
                clip("a0", "audio", 0, 0.1, 0.6, dict(volume=1, channelMode="mono")),
                clip("a1", "audio", 0.9, 1.1, 0.6, dict(volume=0)),
                clip("t0", "text", 0, 0, 0.6, dict(text="Caption", textShadow=0, fontSize=10,
                                                   bgColor="#004b06", bgOpacity=1))])
            args = SimpleNamespace(frame=None, encoder="cpu", threads=2, lufs=-14,
                                   cpu_preset="fast", out=str(directory / "out.mp4"))
            commands = []
            def trace(cmd, **kwargs):
                commands.append(cmd)
                return RUN(cmd, **kwargs)
            font = next(str(p) for p in Path("/usr/share/fonts/truetype").rglob("*Sans*Bold.ttf"))
            with patch.object(render, "WORKSPACE", str(directory)), \
                    patch.object(render, "font_file", return_value=font), \
                    patch.object(render.subprocess, "run", side_effect=trace):
                render.render_project(project, args, str(directory))
            self.assertEqual(sum(c[0] == "ffprobe" for c in commands), 1)
            stages = [c for c in commands if c[0] == "ffmpeg"]
            # audio mix 1, loudness 1, video 3, normalization 1, mux 2 (unordered: concurrent)
            self.assertEqual(sorted(c.count("-i") for c in stages), [1, 1, 1, 2, 3])
            metadata = json.loads(RUN(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                       "-of", "json", args.out], check=True, capture_output=True).stdout)
            self.assertEqual([s["codec_name"] for s in metadata["streams"]], ["h264", "aac"])
            self.assertEqual((metadata["streams"][0]["width"], metadata["streams"][0]["height"]), (90, 160))
            self.assertAlmostEqual(float(metadata["format"]["duration"]), 1.5, delta=0.05)
            samples = array.array("f", ffmpeg("-i", args.out, "-vn", "-ac", "1", "-ar", "48000",
                                             "-f", "f32le", "-").stdout)
            def power(start, end):
                values = samples[int(start*48000):int(end*48000)]
                return sum(v*v for v in values) / len(values)
            self.assertGreater(power(0.1, 0.4), 0.001)
            self.assertLess(power(1.0, 1.3), 0.000001)
            frame = ffmpeg("-ss", "0.75", "-i", args.out, "-frames:v", "1", "-threads", "2",
                           "-pix_fmt", "rgb24", "-f", "rawvideo", "-").stdout
            self.assertGreater(sum(frame) / len(frame), 245)  # white timeline gap


if __name__ == "__main__":
    unittest.main()
