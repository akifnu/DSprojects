# reel-pipeline

Sample reel + long-form footage in → finished vertical reels out, cut in the
sample's style: pacing, punch-ins and zooms, transitions, captions (font, colours,
box, highlight, animation), colour grade, sound effects, music ducking and loudness.
Everything renders with ffmpeg 9.0 and a Python/OpenCV compositor. No browser and
no editor UI are involved.

## Setup

```
reel-pipeline/setup.sh          # idempotent; --check to verify only
reel-pipeline/reel check        # READY / what is missing
```

`setup.sh` installs the ffmpeg 9.0 static build (with libx264/x265, libvmaf,
libass, zscale and NVENC) into `.tools/` and makes it the system `ffmpeg` when run
as root. It also installs the pinned Python packages (`requirements.txt`) and the
models, all from GitHub releases: Parakeet TDT 0.6B (English speech, word timings),
Whisper large-v3-turbo (other languages), Silero VAD and YuNet faces. A
SessionStart hook runs it in every cloud session.

## The production line

| Step | Command | What it does |
|---|---|---|
| ingest | `reel ingest FILE… --kind sample\|source\|music\|sfx` | content-addressed store, duplicates detected |
| sample | `reel sample REF` | measured style spec `style.json` + `review.jpg` (real caption vs. re-rendered from the spec) |
| source | `reel source REF [--prepare]` | transcript + sentence index (`transcript.en.txt`), editing copies |
| plan | `reel plan NAME --style REF --source S=REF` | scaffold `data/batches/NAME/plan.json`; Claude fills the reels |
| batch | `reel batch NAME [-j N] [--draft]` | compile → render → QC every reel in parallel, resumable |
| render | `reel render timeline.json out.mp4 [--still T]` | one timeline, or one preview frame |
| qc | `reel qc reel.mp4 --timeline T --style S` | technical, edit and style checks |
| loop | `reel loop clip.mp4 50 out.mp4` | endless background loop with an invisible cut |

Modules (`reelkit/`): `store` · `asr` · `sample` (with `sample_text`, `sample_video`,
`sample_audio`, `grade`) · `source` · `reframe` · `plan` · `compile` · `render` ·
`captions` · `sfx` · `qc` · `batch` · `loop`. The data shapes passed between them are
in [SPEC.md](SPEC.md).

## Measured on this 4-core cloud box (no GPU)

| | |
|---|---|
| English transcription | 0.15× real time (1 h of audio ≈ 9 min) |
| Hindi/Urdu/other transcription | ≈ 1.8× real time (Whisper turbo on CPU) |
| 4K60 HEVC → 1080p editing copy | ≈ 1× real time (decode-bound) |
| sample analysis (25 s reel) | ≈ 30–60 s |
| batch rendering, 30 s 720p reels | ≈ 110 CPU-s each; best at 2 parallel renders ≈ 120–130 reels/h |

Faster: more cores (everything scales), an NVIDIA GPU (NVENC encode; `encoder="nvenc"`),
or long-form delivered as 1080p30 H.264 instead of 4K60 HEVC.

## Tests

```
cd reel-pipeline && python3 -m unittest discover -s tests -v
```
Round-trip tests: content is rendered with known settings, then the analysers
must recover those settings, and the QC gate must catch injected defects.
