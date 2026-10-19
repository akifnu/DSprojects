# reel-pipeline

Sample reels + raw long footage in → finished vertical reels out, edited by
Claude with [FableCut](https://github.com/ronak-create/FableCut) (an
agent-drivable, JSON-timeline video editor; pinned to v1.15.0).

## How to use it

1. Start a Claude Code session on this repo and give it a job:
   *"New reel job `cafe-oct`: samples are A.mp4 and B.mp4, raw footage is
   long1.mp4 — make 3 reels like sample A, 20–30 s each."*
2. Get the files to the session (any of):
   - upload them in the chat,
   - commit them to a branch / attach them to a GitHub release (github.com is reachable),
   - share https links — only hosts the environment's network policy allows
     (add e.g. `drive.google.com`, `drive.usercontent.google.com`,
     `dl.dropboxusercontent.com` under *Allowed domains* in the environment settings).
3. Claude follows `.claude/skills/reel-from-sample/SKILL.md`: analyzes each
   sample into a brief (cuts, beats, BPM, drop, caption style, look), surveys
   the raw footage, picks selects, builds the timeline, renders
   1080×1920 MP4s with ffmpeg and QC-checks them against the sample.
4. Finished reels are sent to you in the chat and land in `jobs/<job>/out/`.
   The brief and each reel's `project.json` are committed so any reel can be
   re-opened and tweaked.

## Layout

| path | what |
|---|---|
| `setup.sh` | installs FableCut (pinned) into `.fablecut/`, sherpa-onnx and the Parakeet speech model into `.models/`; idempotent, run by the SessionStart hook |
| `fablecut.sh` | runs a FableCut entry point with `FABLECUT_DATA_DIR=workspace/`, browser features off |
| `scripts/start.sh` | starts the editor server on :7777 |
| `scripts/survey.sh` | scene list + time-stamped contact sheets of long footage |
| `scripts/prep.sh` | optional: trimmed 1080p/30 H.264 copy of heavy footage, for faster renders |
| `scripts/transcribe.py` | word-level English transcript (Parakeet via sherpa-onnx) for speech-led reels and captions |
| `scripts/render.sh` | renders the timeline with ffmpeg (`fast_render.py`) → `jobs/<job>/out/`; `--frame t` previews one frame |
| `scripts/fast_render.py` | project.json → one ffmpeg filtergraph: cuts, zooms, captions, audio mix, −14 LUFS |
| `scripts/qc.sh` | format, duration, loudness and contact sheet of a reel vs. its sample |
| `jobs/<job>/` | `brief.md`, `project-<n>.json` (committed); `samples/ raw/ work/ out/` (ignored) |
| `workspace/` | FableCut's data dir: `project.json`, `media/`, `exports/`, `analysis/` (ignored) |

## Rendering

Everything is rendered by ffmpeg, never by a browser. `scripts/render.sh`
defaults to **720p (720×1280)**, x264 `veryfast`: a 28 s reel renders in about
10 s from a 720p edit copy of the footage (`prep.sh`). `RES=1080` gives
1080×1920; `CPU_PRESET=fast` gives smaller files for a few more seconds. `fast_render.py` covers cuts, static zooms/reframes, captions with
pills/stroke and the audio mix; it lists anything else it can't draw yet
instead of rendering it wrong. FableCut's own export (a browser compositor) is
not used. Without a GPU it ran at ~1 frame/s here.

Exports log stage timings, probe each source once, and crop hidden source regions
before scaling where alignment permits. The main video track is joined end to end
instead of stacked, captions become one image track, and the audio mix and
two-pass loudness run while the video encodes, then the two are muxed.
`VIDEO_ENCODER=auto` checks NVIDIA NVENC and falls back to CPU when unavailable;
`VIDEO_ENCODER=nvenc` requires a working GPU encoder. See
[GPU_RENDERING.md](GPU_RENDERING.md) for cloud-worker setup, asset transfer,
Docker commands, and validation. Filters still run on the CPU.

**Transcription** uses NVIDIA Parakeet (English) from the sherpa-onnx GitHub
release (~480 MB download on first setup), because `huggingface.co` is blocked.

## Run it locally with the editor UI

```
reel-pipeline/setup.sh && reel-pipeline/scripts/start.sh   # → http://localhost:7777
```
