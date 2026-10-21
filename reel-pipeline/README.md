# reel-pipeline

Sample reel + long-form footage in → finished vertical reels out, cut in the
sample's style: pacing, punch-ins and zooms, transitions, captions (font, colours,
box, highlight, animation), colour grade, sound effects, music ducking and loudness.
Everything renders with ffmpeg 9 and a Python/OpenCV compositor. No browser and
no editor UI are involved.

## Setup

```
reel-pipeline/setup.sh            # install what is missing, then `reel check`; safe to rerun
reel-pipeline/setup.sh --check    # install nothing, report (REELKIT_PREBUILT=1 does the same)
reel-pipeline/setup.sh --verify   # also verify model checksums and load every model once
reel-pipeline/setup.sh --gpu      # also the GPU profile (requirements-gpu.txt)
reel-pipeline/reel check          # readiness: JSON on stdout, summary on stderr, exit 1 = not ready
```

`setup.sh` installs the ffmpeg 9.0 static build (libx264/x265, libvmaf, zscale, NVENC)
into `.tools/` (and makes it the system `ffmpeg` when run as root), the pinned Python
packages (`requirements.txt`, transitive dependencies included) and the models, all from
GitHub releases: Parakeet TDT 0.6B (English speech, word timings), Whisper large-v3-turbo
(other languages), Silero VAD and YuNet faces. Models go to `$REELKIT_MODELS` (default
`.models/`), the same variable the runtime reads. Every download lands in a temp file,
is verified (complete archive, non-empty model files, sha-256 against `CHECKSUMS` where
one was recorded; the first download records it) and is renamed into place, so an
interrupted install leaves nothing half-installed and simply runs again. BtbN's default
ffmpeg URL is a moving "latest" build: pin `REELKIT_FFMPEG_URL` + `REELKIT_FFMPEG_SHA256`
for a reproducible deployment.

A SessionStart hook runs `setup.sh` in every cloud session: with everything in place it
downloads nothing and finishes in seconds, and as a hook it never fails the session — it
reports what is missing. The render image (`Dockerfile.render`) runs the install at build
time, prewarms the font matcher's fonts, runs as an unprivileged user with `/data` as the
data volume and sets REELKIT_PREBUILT=1. It packages the media CLI, not the Claude Code
controller (the agents and workflow in `.claude/` run in a Claude Code session that calls
`reel`).

## Agentic operation

Upload a sample reel and long-form footage in a Claude Code session (or submit a
request.json headlessly): a job is created and the `reel-job` workflow drives it. Python
(`reelkit/orchestrator.py`, `reel job …`) owns the job state, every mechanical step and
every completion rule; agents (`.claude/agents/`) only supply judgment:

| Role | Does | Must not |
|---|---|---|
| `reel-runner` | starts / waits for the worker, records agents' answers (`start`, `wait`, `done`) | decide success |
| `sample-analyst` | verifies the measured style in the job's working copy | touch shared style files |
| `reel-planner` | chooses the moments, writes plan.json, applies fixes | change reels outside a repair's `only` |
| `qc-reviewer` | one decision per assigned reel with precise findings | waive a technical QC failure |

```
reel job new ─► [intake · analyse sample]─► sample-analyst verifies style ─► [transcribe]
  ─► reel-planner writes plan ─► [editing copies · render · QC] ─► qc-reviewers ─┬─► deliver
                    ▲                                                           │
                    └──────── fixes (budget per reel and per job) ◄─────────────┘
```

## Operations

```
reel job new --sample S.mp4 --source talk.mp4 [--music M] [--count 12] [--duration 30] [--lang en]
             [--notes T | --request request.json] [--encoder auto|cpu|nvenc] [--require-gpu]
             [--no-cpu-fallback] [--asr-provider cpu|cuda] [--name N]
reel job add ID --source more.mp4     # more inputs; unaffected work is kept
reel job start ID                     # detached worker; returns once it acknowledged the job
reel job wait ID [--timeout S]        # 0 settled (agent / input needed, or terminal), 2 worker failed or died,
                                      # 124 still running
reel job status ID|latest             # JSON: status, needs (need_id, revision), counts, open findings
reel job done ID STEP --need NEED_ID --revision N [--note-file F] [--result-file F]
reel job cancel ID | reel job retry ID | reel job list
```

- **Requests.** Roles are explicit (`--sample/--source/--music/--sfx/--image` or a
  request.json, see SPEC.md "Request"); a bare file gets a heuristic role, recorded as a
  suggestion. Invalid counts, durations, languages, roles or missing files fail `job new`
  with every reason listed, before any expensive work. Missing inputs make the job wait
  (`waiting:sample` / `waiting:source`); `job add` resumes it.
- **Worker.** `job start` spawns a detached worker that heartbeats into
  `attempts/<attempt>.json`; `job wait` trusts that record and reports a worker that died
  without one (exit 2). `job cancel` raises the cancel flag and stops the worker's process
  group (SIGTERM, then SIGKILL). Every transition is a line in `events.jsonl`.
- **Answers.** An agent's answer quotes the open `need_id` (and `revision`); stale,
  out-of-order or invalid answers are rejected with the reasons (exit 2) and the need stays
  open. Review results go into `inbox/<need_id>.json` (a file, never JSON in a shell
  argument); every finding is recorded whatever the repair budget.
- **Terminal states.** `succeeded` (every planned reel delivered, reviewed where required,
  no open blocking finding, no shortfall), `partial` (some reels accepted, the rest listed
  with reasons), `needs_attention` (a blocking finding or missing judgment outlived its
  budget, or nothing was accepted), `failed` (infrastructure or validation error, typed
  record in `error`), `cancelled`. `deliver.files` comes only from the batch manifest's
  hash-checked deliverables; failing renders are under `out/_review/`.

### GPU profile

A GPU host does not make the pipeline a GPU pipeline by itself: composition, grading
and source preparation stay on the CPU; NVENC replaces the final x264 encode and the
CUDA build of sherpa-onnx can run ASR.

```
reel-pipeline/setup.sh --gpu                           # requirements-gpu.txt: CUDA sherpa-onnx, onnxruntime-gpu
reel-pipeline/reel check --encoder nvenc --require-gpu --asr-provider cuda --deep
reel-pipeline/reel job new … --encoder nvenc --require-gpu [--asr-provider cuda]
```

`reel check` counts NVENC only after a real 30-frame h264_nvenc encode with the renderer's
settings (an encoder listing proves nothing about the driver or device) and reports the GPU
from nvidia-smi; `cuda` ASR needs a CUDA build of sherpa-onnx and a visible GPU. With
`--require-gpu` a job without a usable NVENC fails before any expensive work; `auto` falls
back to the CPU encoder only while CPU fallback is allowed, and the manifest and job record
the requested and actual encoder with the reason. The exact CUDA wheels depend on the host's
driver, CUDA and cuDNN — validate them there (requirements-gpu.txt says how to pin them). In
containers, pass `--gpus all`; the image sets NVIDIA_DRIVER_CAPABILITIES=compute,utility,video.

## The production line

### Import footage from a URL

```
python3 -m pip install -r reel-pipeline/requirements-url.txt
reel-pipeline/reel import-url 'https://…' --kind source
```

This optional yt-dlp importer selects the best available video plus audio (or the
best combined stream), remuxes split streams into MKV without a lossy transcode,
then ingests the completed file into the verified asset store. Music/SFX imports
select the best audio. The JSON result includes `id`, `file` and an `import.json`
provenance record. Use its local `file` in `reel job new --source FILE`, or use its
asset `id` with `reel source ID`. Playlists are rejected; failed downloads clean
their staging directory. Local-file jobs do not require yt-dlp.

HyperFrames was considered for richer HTML motion graphics. The core compositor
remains in use: the available fixtures do not establish that a browser renderer
improves these talking-head edits. A renderer migration needs matched reference
renders, timing/audio checks and a measured visual comparison first.

### Effect timing and caption verification

The compiler uses the reference effect rates as budgets. Planner `emphasis` words,
sentence boundaries and pauses rank effect opportunities, with minimum spacing;
punches, pushes, transitions and SFX no longer follow an every-N-shot cadence.
These cues improve placement control; they do not measure audience retention or
infer emotion. The planner should mark a few words carrying the hook and payoff.

Caption QC fails when verification is requested but re-transcription errors or
returns no words. Reports record `caption_verification` as `checked`, `unavailable`,
`skipped` (explicit ASR opt-out), or `not_applicable`. Hook/CTA text is marked as
editorial `role: title` and excluded from verbatim speech matching. Technical QC
failure continues to block delivery regardless of an agent's review decision.

| Step | Command | What it does |
|---|---|---|
| ingest | `reel ingest FILE… --kind sample\|source\|music\|sfx\|image` | content-addressed store (private verified copies), duplicates detected |
| sample | `reel sample REF` | measured style spec `style.json` + `review.jpg` (real caption vs. re-rendered from the spec) |
| source | `reel source REF [--prepare] [--asr-provider cpu\|cuda]` | transcript + sentence index (`transcript.en.txt`), editing copies |
| plan | `reel plan NAME --style REF --source S=REF` | scaffold `data/batches/NAME/plan.json`; Claude fills the reels |
| batch | `reel batch NAME [-j N] [--draft] [--only r01,r02] [--encoder auto\|cpu\|nvenc] [--require-gpu]` | compile → render → QC every reel in parallel, resumable, one manifest generation per run |
| render | `reel render timeline.json out.mp4 [--still T] [--encoder auto\|cpu\|nvenc]` | one timeline, or one preview frame |
| qc | `reel qc reel.mp4 --timeline T --style S` | technical, edit and style checks |
| loop | `reel loop clip.mp4 50 out.mp4` | endless background loop with an invisible cut |

Modules (`reelkit/`): `store` · `asr` · `sample` (with `sample_text`, `sample_video`,
`sample_audio`, `grade`) · `source` · `reframe` · `plan` · `compile` · `render` ·
`captions` · `sfx` · `qc` · `batch` · `loop` · `orchestrator` · `contracts` (the executable
schemas) · `readiness` · `fsutil` · `ff` · `paths`. The data shapes passed between them are
in [SPEC.md](SPEC.md).

## Measured on this 4-core cloud box (no GPU)

| | |
|---|---|
| English transcription | 0.15× real time (1 h of audio ≈ 9 min) |
| Hindi/Urdu/other transcription | ≈ 1.8× real time (Whisper turbo on CPU) |
| 4K60 HEVC → 1080p editing copy | ≈ 1× real time (decode-bound) |
| sample analysis (25 s reel) | ≈ 30–60 s |
| batch rendering, 30 s 720p reels | ≈ 110 CPU-s each; best at 2 parallel renders ≈ 120–130 reels/h |

Faster: more cores (everything scales), an NVIDIA GPU (NVENC encode, see the GPU profile),
or long-form delivered as 1080p30 H.264 instead of 4K60 HEVC. No GPU speedup has been
measured here; measure a full job before relying on one.

## Tests

```
cd reel-pipeline && python3 -m unittest discover -s tests -v
```
Round-trip tests: content is rendered with known settings, then the analysers
must recover those settings, and the QC gate must catch injected defects.

`.github/workflows/reel-pipeline-ci.yml` runs on relevant pushes and pull requests.
Its required core suite covers editing/rendering, a complete job with stubbed
analysis and agent answers but real ingest/render/QC/delivery, stale and repeated
answers, review/render fencing, damaged state recovery, cancellation/restart,
bounded retries, worker death, immutable manifest generations, cache tampering,
encoder fallback policy, process timeouts and atomic output failures. URL-import
tests mock the downloader; they do not depend on a live media website.

```
cd reel-pipeline
python3 -m pip install -r requirements.txt -r requirements-url.txt
bash tests/bootstrap.sh
REELKIT_OFFLINE=1 python3 tests/run_suite.py --suite core -v
REELKIT_OFFLINE=1 python3 tests/run_suite.py --suite analysis -v
```

Bootstrap installs the smaller test profile (ffmpeg 9, YuNet, a moving-face image,
OCR package models and the font bank). Speech fixtures carry current ASR
fingerprints; production speech-model inference and the Claude JavaScript
controller are not covered by this profile. The private R1 real-speech test skips
when its source/model are absent.

**Known baseline failures:** the existing `test_sample_text.py` round-trip suite
fails caption-track/title detection and font/highlight/animation recovery checks.
It runs as a visible diagnostic CI step with `continue-on-error`, outside the
required core gate. Full `unittest discover` and `run_suite.py --suite all` still
include it and return failure. A green core CI result is not evidence that sample
caption-style recovery is correct; review the measured style against the sample.

