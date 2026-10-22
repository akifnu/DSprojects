# Reels Studio (first draft)

A browser editor for the **R1 – Core editor** scope of the AI Reels Editing Studio PRD:
import a talking-head video, review an AI rough cut in the transcript, restyle captions,
add media and sounds, preview, and export a 1080×1920 MP4. Every edit is reversible and
the source is never modified.

```
python3 reels-studio/server.py        # http://127.0.0.1:8765
```

Needs Python 3.11+, `starlette`, `uvicorn`, `python-multipart`, ffmpeg/ffprobe with libass,
and the `reel-pipeline/` checkout next to it (its `reelkit.asr` does transcription with the
models `reel-pipeline/setup.sh` installs). Data goes to `reels-studio/data/` (`STUDIO_DATA`).

## What's in the draft

| PRD | Status |
| --- | --- |
| FR-01 import + validation | MP4/MOV/M4V/WEBM, ffprobe validation, read-only original, proxy, retryable processing job |
| FR-02 transcript | reelkit ASR word timings, stable word ids, uncertain words flagged, caption-text corrections (audio untouched) |
| FR-03 AI rough cut | heuristic provider: fillers, earlier repeated takes, pause tightening; re-runs create a new revision |
| FR-04/05 review, cut, undo | collapsed removed groups with reason → expand → restore; cut selection; undo/redo; revision history with restore |
| FR-06 synced preview | proxy playback that skips removed ranges; word ↔ playhead sync; timeline lanes |
| FR-07/08 lookbook + captions | cross-project caption/look presets (applied as snapshots); font, weight, size, colours, outline/box, position, case, words per line; missing-font warning |
| FR-09 emphasis | per-word-id emphasis that survives unrelated cuts |
| FR-10 media | upload images/GIFs/video with rights note; overlays anchored to words; missing/cut anchors flagged and block export |
| FR-11 motion notes | simple parser: slide/spin/fade/pop + “exit on ‘word’” |
| FR-14 B-roll (manual) | full-frame video over selected words, source in point, optional clip audio |
| FR-17/19 export | frozen-revision render job, progress/cancel/retry, output validation (streams, size, duration), outdated label |
| FR-18 sounds | bundled + uploaded sounds, audition, swap, gain, trim, length, loop, fades, ducking under narration |

Not yet: generated three-option looks (FR-12/13), AI B-roll choice, face covering (FR-15/16),
CapCut handoff, frame capture, carousel/story, trial reels (R3), auth/multi-user (single-user
local app for now), and an LLM rough-cut provider (`studio/edit.rough_cut` is the seam).

## Layout

- `server.py` – Starlette API + static files
- `studio/edit.py` – revision schema, rough cut, `compile()` (source → composition mapping used by preview *and* render)
- `studio/render.py` – ASS captions + one ffmpeg graph per export
- `studio/jobs.py` – durable background jobs (processing, export)
- `studio/store.py` – file-backed projects, immutable revisions (stale writes → 409), presets, library
- `static/` – the editor (vanilla JS)

## Test link (Hugging Face Space)

1. Create a Hugging Face token with **write** access (huggingface.co → Settings → Access Tokens).
2. In this GitHub repo → Settings → Secrets and variables → Actions, add secrets
   `HF_TOKEN` (that token) and `STUDIO_ACCESS_KEY` (any password you choose).
   Optional variable `HF_SPACE` (default `reels-studio`).
3. Push to `main` or the dev branch (or run the *Deploy Reels Studio to Hugging Face* workflow).
   The first build takes ~10 minutes (ffmpeg + 1.6 GB of speech models).
4. Open `https://<hf-user>-reels-studio.hf.space/?key=<STUDIO_ACCESS_KEY>`.

Manual alternative: `HF_TOKEN=… HF_SPACE=… STUDIO_ACCESS_KEY=… python3 reels-studio/deploy/push_space.py`.
Free Spaces have no persistent disk: projects disappear when the Space restarts or sleeps.
