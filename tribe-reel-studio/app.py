"""TRIBE v2 Reel Studio: upload a reel, see the predicted brain response.

Run:
    python app.py              # real TRIBE v2 model (GPU strongly recommended)
    python app.py --share      # also create a public gradio.live link (Colab)
    python app.py --demo       # synthetic predictions, for trying the UI without a GPU
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import brain_viz as bv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reel_studio")

CACHE = Path(os.environ.get("TRIBE_CACHE", "./cache")).resolve()
OUT_ROOT = CACHE / "reel_outputs"
LANGUAGES = ["english", "french", "spanish", "dutch", "chinese"]
DEMO = False
_MODEL = None


# ----------------------------------------------------------------------
# Video helpers
# ----------------------------------------------------------------------


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def has_audio_stream(path: str) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", path], capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_to_video(path: str, seconds: float) -> str:
    """Turn a still photo into a silent clip (TRIBE only takes video/audio/text)."""
    out = CACHE / "uploads" / f"{Path(path).stem}_{int(seconds)}s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", path, "-t", str(seconds),
                    "-vf", "scale='min(1080,iw)':-2,format=yuv420p", "-r", "25", "-c:v", "libx264",
                    str(out)], check=True)
    return str(out)


def stage_upload(path: str) -> Path:
    """Copy the upload to a stable, content-addressed .mp4 so features get cached
    and re-analysing the same reel is fast. Non-mp4 inputs are transcoded."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    dst = CACHE / "uploads" / f"reel_{h.hexdigest()[:16]}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        if Path(path).suffix.lower() == ".mp4":
            shutil.copy(path, dst)
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", str(dst)], check=True)
    return dst


def extract_frames(video: Path, times: list[float], out_dir: Path, height=360) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in times:
        p = out_dir / f"frame_{t:07.2f}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video),
                        "-frames:v", "1", "-vf", f"scale=-2:{height}", str(p)], check=False)
        paths.append(p)
    return paths


# ----------------------------------------------------------------------
# TRIBE inference
# ----------------------------------------------------------------------


WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
WHISPER_LANG = dict(english="en", french="fr", spanish="es", dutch="nl", chinese="zh")
_WHISPER = None


def transcribe_words(wav_filename, language: str) -> pd.DataFrame:
    """Drop-in for TRIBE's WhisperX step: faster-whisper in-process, with the same output
    columns. TRIBE runs WhisperX large-v3 through `uvx`, which rebuilds a separate
    environment on every container start; this skips that and uses the faster turbo model."""
    global _WHISPER
    import torch
    from faster_whisper import WhisperModel

    if language not in WHISPER_LANG:
        raise ValueError(f"Language {language} not supported")
    if _WHISPER is None:
        cuda = torch.cuda.is_available()
        _WHISPER = WhisperModel(WHISPER_MODEL, device="cuda" if cuda else "cpu",
                                compute_type="float16" if cuda else "int8")
    segments, _ = _WHISPER.transcribe(str(wav_filename), language=WHISPER_LANG[language],
                                      word_timestamps=True, vad_filter=True)
    rows = []
    for i, seg in enumerate(segments):
        sentence = seg.text.strip().replace('"', "")
        for w in seg.words or []:
            rows.append(dict(text=w.word.strip().replace('"', ""), start=w.start,
                             duration=w.end - w.start, sequence_id=i, sentence=sentence))
    return pd.DataFrame(rows, columns=["text", "start", "duration", "sequence_id", "sentence"])


def warm_up():
    """Fetch LLaMA 3.2 (gated, so it can't be preloaded at build time) and load the models
    in the background while the UI is already up, so the first upload doesn't wait for it."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("meta-llama/Llama-3.2-3B", allow_patterns=["*.json", "*.safetensors"])
        get_model()
        logger.info("Warm-up done: models ready.")
    except Exception:
        logger.exception("Background warm-up failed; models will load on first analysis")


def get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from tribev2.demo_utils import TribeModel

        from tribev2.eventstransforms import ExtractWordsFromAudio

        if torch.cuda.is_available():
            # TF32 tensor cores for float32 matmuls/convs: large speedup on Ampere+ GPUs
            # (L40S, A100), negligible accuracy cost.
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            logger.warning("No CUDA GPU detected: inference will be very slow on CPU.")
        ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(transcribe_words)
        _MODEL = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(CACHE))
    return _MODEL


def build_events(video: Path, language: str) -> tuple[pd.DataFrame, str]:
    """Same pipeline as TribeModel.get_events_dataframe, with a language choice and
    a fallback for reels without intelligible speech (music-only, etc.)."""
    from neuralset.events.transforms import (AddContextToWords, AddSentenceToWords, AddText,
                                             ChunkEvents, ExtractAudioFromVideo, RemoveMissing)
    from neuralset.events.utils import standardize_events
    from tribev2.eventstransforms import ExtractWordsFromAudio

    base = [
        ExtractAudioFromVideo(),
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30),
    ]
    text = [
        ExtractWordsFromAudio(language=language),
        AddText(),
        AddSentenceToWords(max_unmatched_ratio=0.05),
        AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
        RemoveMissing(),
    ]
    event = {"type": "Video", "filepath": str(video), "start": 0,
             "timeline": "default", "subject": "default"}

    def run(transforms):
        ev = standardize_events(pd.DataFrame([event]))
        for t in transforms:
            ev = t(ev)
        return standardize_events(ev)

    try:
        events = run(base + text)
        n_words = int((events.type == "Word").sum())
        if n_words:
            return events, f"Transcribed {n_words} words ({language})."
        note = "No speech detected: using video + audio only."
    except Exception as e:  # e.g. music-only reel, transcription failure
        logger.exception("Transcription stage failed")
        note = f"Speech transcription failed ({type(e).__name__}); using video + audio only."
    return run(base), note


def predict_tribe(video: Path, language: str, progress) -> tuple[np.ndarray, list[float], float, str]:
    progress(0.05, desc="Loading TRIBE v2 (first run downloads weights)…")
    model = get_model()
    progress(0.15, desc="Extracting audio + transcribing speech…")
    events, note = build_events(video, language)
    progress(0.3, desc="Extracting V-JEPA2 / Wav2Vec-BERT / LLaMA features + predicting…")
    preds, segments = model.predict(events=events, verbose=True)
    starts = [float(s.start) for s in segments]
    return preds.astype(np.float32), starts, float(model.data.TR), note


def predict_demo(video: Path, duration: float) -> tuple[np.ndarray, list[float], float, str]:
    """Synthetic but structured predictions for UI testing. NOT brain data."""
    rng = np.random.default_rng(0)
    tr = 1.0
    n = max(int(duration // tr), 1)
    masks = bv.load_network_masks()
    if masks is None:  # offline fallback: coarse spatial blobs from mesh geometry
        mesh = bv.load_mesh()
        coords = np.r_[mesh["left"]["coords"], mesh["right"]["coords"]]
        centers = coords[rng.choice(len(coords), len(bv.NETWORKS), replace=False)]
        lab = np.argmin(((coords[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        masks = {net: lab == k for k, net in enumerate(bv.NETWORKS)}
    preds = 0.05 * rng.standard_normal((n, 2 * bv.N_VERTS_HEMI)).astype(np.float32)
    t = np.arange(n)
    for k, net in enumerate(bv.NETWORKS):
        tc = np.convolve(rng.standard_normal(n + 6), np.hanning(7), "same")[3:n + 3]
        tc = 0.25 * tc + 0.15 * np.sin(2 * np.pi * t / (6 + k))
        preds[:, masks[net]] += tc[:, None].astype(np.float32)
    return preds, list(t * tr), tr, "DEMO MODE: synthetic data, not TRIBE predictions."


# ----------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------


def make_brain_movie(video: Path, dense, ts, tr, thr, vmax, diverging, out_dir: Path, progress) -> Path:
    renderer = bv.BrainRenderer()
    anim = bv.PanelAnimator(renderer, ts, tr, thr, vmax, diverging)
    frames_dir = out_dir / "panels"
    frames_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for i in range(len(dense)):
        progress(0.6 + 0.3 * i / max(len(dense), 1), desc=f"Rendering brain movie {i + 1}/{len(dense)}")
        Image.fromarray(anim.frame(dense[i], i)).save(frames_dir / f"p_{i:05d}.png")
    anim.close()

    out = out_dir / "reel_with_brain.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
           "-framerate", f"{1 / tr}", "-start_number", "0", "-i", str(frames_dir / "p_%05d.png"),
           "-filter_complex",
           "[0:v]scale=-2:720,setsar=1,fps=25[a];[1:v]scale=-2:720,setsar=1,fps=25[b];"
           "[a][b]hstack=inputs=2,format=yuv420p[v]",
           "-map", "[v]"]
    if has_audio_stream(str(video)):
        cmd += ["-map", "0:a", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-shortest",
            "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out


def make_moments(video: Path, dense, ts, peaks, tr, thr, vmax, diverging, out_dir: Path):
    import matplotlib.pyplot as plt
    from PIL import Image

    renderer = bv.BrainRenderer(views=("Left lateral", "Right lateral"))
    frames = extract_frames(video, [i * tr + tr / 2 for i in peaks], out_dir / "frames")
    gallery = []
    for rank, (i, fp) in enumerate(zip(peaks, frames), 1):
        fig = plt.figure(figsize=(9, 3.6), dpi=100)
        gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1, 1], wspace=0.02)
        ax = fig.add_subplot(gs[0])
        if fp.exists():
            ax.imshow(Image.open(fp))
        ax.axis("off")
        for k, view in enumerate(renderer.views):
            renderer.draw(fig.add_subplot(gs[k + 1]), view, dense[i], thr, vmax, diverging)
        nets = {n: s[i] for n, s in ts.items() if n != "Overall"}
        top = ", ".join(sorted(nets, key=nets.get, reverse=True)[:3])
        caption = f"#{rank} · t = {i * tr:.0f}s" + (f" · strongest: {top}" if top else "")
        fig.suptitle(caption, fontsize=11, fontweight="bold")
        p = out_dir / f"moment_{rank}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        gallery.append((str(p), caption))
    return gallery


def analyze(video_path, image_path, image_seconds, language, colormap, threshold_pct, make_movie,
            progress=gr.Progress()):
    if not video_path and image_path:
        video_path = image_to_video(image_path, image_seconds)
    if not video_path:
        raise gr.Error("Upload a reel or a photo first.")
    t0 = time.time()
    progress(0.01, desc="Preparing video…")
    video = stage_upload(video_path)
    duration = probe_duration(str(video))
    if duration > 300:
        raise gr.Error(f"Video is {duration:.0f}s long; please keep it under 5 minutes.")

    if DEMO:
        preds, starts, tr, note = predict_demo(video, duration)
    else:
        preds, starts, tr, note = predict_tribe(video, language, progress)

    progress(0.55, desc="Summarizing…")
    diverging = colormap.startswith("Activation & suppression")
    n_steps = int(np.ceil(duration / tr))
    dense = bv.densify(preds, starts, tr, n_steps=n_steps)
    masks = bv.load_network_masks()
    ts = bv.network_timeseries(dense, masks)
    peaks = bv.find_peaks(ts["Overall"], k=3)
    thr, vmax = bv.vertex_limits(dense, threshold_pct=threshold_pct)
    if diverging:
        thr = vmax * threshold_pct / 200  # show more of the signed range
    mean_map = np.nanmean(dense, axis=0)

    out_dir = OUT_ROOT / f"{video.stem}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    avg_png = bv.render_static(bv.BrainRenderer(), bv.rescale_like(mean_map, vmax), thr, vmax, diverging,
                               "Average predicted activation over the whole reel",
                               str(out_dir / "average_activation.png"))
    gallery = make_moments(video, dense, ts, peaks, tr, thr, vmax, diverging, out_dir)
    fig3d = bv.brain_3d_figure(dense, tr, thr, vmax, diverging, mean_map=mean_map)
    html_3d = out_dir / "brain_3d_interactive.html"
    fig3d.write_html(str(html_3d), include_plotlyjs=True, auto_play=False,
                     default_height="100%", default_width="100%")
    viewer = (f'<iframe src="/gradio_api/file={html_3d}" style="width:100%;height:680px;border:0" '
              f'title="Interactive 3D brain"></iframe>')
    fig_int = bv.intensity_figure(ts, tr, peaks)

    movie = None
    if make_movie:
        movie = make_brain_movie(video, dense, ts, tr, thr, vmax, diverging, out_dir, progress)

    table = pd.DataFrame(bv.summary_table(ts, tr),
                         columns=["Region / network", "Mean", "Peak", "Peak at (s)", "Top network % of time"])
    np.save(out_dir / "predictions_fsaverage5.npy", dense)
    ts_df = pd.DataFrame({"time_s": np.arange(len(dense)) * tr, **ts})
    ts_df.to_csv(out_dir / "intensity_timeseries.csv", index=False)
    table.to_csv(out_dir / "network_summary.csv", index=False)
    files = [str(out_dir / f) for f in ("intensity_timeseries.csv", "network_summary.csv",
                                        "predictions_fsaverage5.npy", "brain_3d_interactive.html")]
    if movie:
        files.insert(0, str(movie))

    status = (f"**Done in {time.time() - t0:.0f}s.** {note}  \n"
              f"{len(preds)} timesteps predicted over a {duration:.1f}s reel (1 step = {tr:g}s; "
              f"5s hemodynamic lag already compensated).")
    if masks is None:
        status += "  \n⚠️ Destrieux atlas unavailable (no internet?), so network breakdown is disabled."
    if DEMO:
        status = "### ⚠️ DEMO MODE: synthetic data, NOT real TRIBE v2 predictions\n" + status
    return (status, str(movie) if movie else None, viewer, fig_int, avg_png, gallery, table, files)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

INTRO = """
# 🧠 TRIBE v2 Reel Studio
Upload a reel (vertical or horizontal video). **TRIBE v2** (Meta FAIR) predicts the fMRI response of an
*average human brain* watching it, second by second, on ~20k cortical vertices (fsaverage5).

<small>Predictions are model estimates for an average viewer, not measurements of real people. Network groupings are
anatomical approximations (Destrieux atlas). TRIBE v2 is licensed CC BY-NC 4.0 (non-commercial use only).</small>
"""

HELP = """
**How to read the results**
- **Brain movie**: your reel next to the predicted brain response. Brighter colors mean a stronger predicted response.
- **3D brain**: rotate the brain, scrub through time with the slider, or pick *Average*.
- **Intensity**: the response over time, for the whole cortex and for each brain system. The dotted lines mark the 3 strongest moments.
- **Key moments**: the frames where the overall response peaked, with the brain map at each one.
- **Visual** = what's on screen · **Auditory** = sound/music · **Language** = speech and meaning ·
  **Faces & objects** = people and things · **Emotion & reward** = OFC/ACC/insula (rough proxy).
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TRIBE v2 Reel Studio") as demo:
        gr.Markdown(INTRO)
        if DEMO:
            gr.Markdown("### ⚠️ Running in DEMO mode: outputs are synthetic, for UI testing only.")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                video_in = gr.Video(label="Your reel (.mp4 / .mov / .webm)", sources=["upload"], height=480)
                with gr.Accordion("…or analyze a photo", open=False):
                    image_in = gr.Image(label="Photo (shown as a still clip)", type="filepath", sources=["upload"])
                    image_seconds = gr.Slider(5, 30, value=10, step=1, label="Show the photo for (s)")
                language = gr.Dropdown(LANGUAGES, value="english", label="Spoken language in the reel")
                with gr.Accordion("Display options", open=False):
                    colormap = gr.Radio(["Activation only (hot)", "Activation & suppression (blue/red)"],
                                        value="Activation only (hot)", label="Color map")
                    threshold = gr.Slider(0, 95, value=55, step=5,
                                          label="Hide weak responses below (% of peak)")
                    make_movie = gr.Checkbox(value=True, label="Render side-by-side brain movie")
                run_btn = gr.Button("🧠 Analyze reel", variant="primary", size="lg")
                status = gr.Markdown()
                gr.Markdown(HELP)
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("🎬 Brain movie"):
                        movie_out = gr.Video(label="Reel + predicted brain response", height=560)
                    with gr.Tab("🧊 3D brain"):
                        gr.Markdown("Drag to rotate, scroll to zoom, use the slider or ▶ Play to move "
                                    "through time, and the buttons at the top for preset views.")
                        plot3d = gr.HTML()
                    with gr.Tab("📈 Intensity"):
                        intensity = gr.Plot(label="Intensity over time")
                        table = gr.Dataframe(label="Per-network summary", interactive=False, wrap=True)
                    with gr.Tab("⭐ Key moments"):
                        avg_img = gr.Image(label="Average activation", type="filepath")
                        gallery = gr.Gallery(label="Peak moments", columns=1, height="auto")
                    with gr.Tab("⬇️ Downloads"):
                        files = gr.File(label="Results", file_count="multiple")
        run_btn.click(
            analyze,
            inputs=[video_in, image_in, image_seconds, language, colormap, threshold, make_movie],
            outputs=[status, movie_out, plot3d, intensity, avg_img, gallery, table, files],
        )
    return demo


def main():
    global DEMO
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic predictions (no GPU/weights)")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--preload", action="store_true", help="load the model before serving")
    ap.add_argument("--no-warmup", action="store_true", help="skip background model download")
    args = ap.parse_args()
    DEMO = args.demo
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.preload and not DEMO:
        get_model()
    elif not DEMO and not args.no_warmup:
        import threading

        threading.Thread(target=warm_up, daemon=True).start()
    build_ui().queue(max_size=8).launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share,
        allowed_paths=[str(CACHE)], show_error=True,
    )


if __name__ == "__main__":
    main()
