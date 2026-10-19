---
name: reel-from-sample
description: Turn raw long footage into finished vertical reels that follow a sample reel's structure and style, using the FableCut editor in reel-pipeline/. Use when the user provides reel samples and raw footage, asks to "make reels like this", or drops files into reel-pipeline/jobs/.
---

# Reel from sample

The pipeline lives in `reel-pipeline/` (see its README). FableCut provides the
edit format (a JSON timeline, `reel-pipeline/workspace/project.json`) and the
sample analyzer. **Rendering is ffmpeg-only** (`scripts/render.sh` →
`scripts/fast_render.py`): never use the editor's headless-browser export or
timeline-frame features. Its full manual is
`reel-pipeline/.fablecut/CLAUDE.md` — read sections from it with
`grep -n "^## \|^### "` + `sed -n`, never the whole 140 KB file.

## 0. Setup (each session)

```
reel-pipeline/setup.sh && reel-pipeline/scripts/start.sh
```

## 1. Intake

One job per folder: `reel-pipeline/jobs/<job>/{samples,raw,work,out}`.
Files may arrive as chat uploads, files in the repo, or https links
(`curl -L -o …`; only hosts the environment's network policy allows). Never
commit media — `jobs/*/samples|raw|work|out` are git-ignored. Do commit
`jobs/<job>/brief.md` and `jobs/<job>/project.json` snapshots.

If it is unclear how many reels to make, how long, or which footage goes with
which sample, ask once before cutting.

## 2. Read the sample (the brief is the sample)

For every sample:
- `curl -s -X POST localhost:7777/api/analyze -H 'content-type: application/json' -d '{"src":"/media/<sample>.mp4"}'`
  after copying it into `workspace/media/` → cuts, shots (with energy), BPM, beats, drop,
  and `<sample>-music.m4a` extracted. Lower `threshold` if cuts are missed.
- Look at it: `scripts/qc.sh <sample>` writes a 12-frame sheet; for detail use
  `curl -s "localhost:7777/api/frame?src=/media/<sample>.mp4&frames=24&w=320" -o sheet.jpg`
  (or `&t=<sec>` for one frame) and Read the image.
- If it has speech: `python3 scripts/transcribe.py <file> <out.json>`.

Write `jobs/<job>/brief.md`: duration, aspect, shot count / avg shot length,
BPM, where the drop is, hook (first 1–2 s), caption style (font, size,
position, animation, case, colours, stroke/pill), text overlays, grade/look,
transitions, zooms/speed ramps, audio (music only / voice + music / SFX), CTA
or end card. This is the spec every reel is checked against.

## 3. Mine the raw footage

- `scripts/survey.sh <raw> jobs/<job>/work/survey` → scene list + time-stamped
  contact sheets. Read the sheets; don't watch blindly.
- Speech-led footage: transcribe it and pick the strongest self-contained
  lines (hook first). Picture-led footage: pick moments that match each sample
  shot's energy (calm ≤ 40, action ≥ 70).
- Record selects in the brief as `raw file · in → out · why`.
- Copy raw files into `workspace/media/` and **start `scripts/prep.sh <raw> <name>`
  in the background right away** (720p/30 H.264 edit copy; trim long files to their
  selects with `<in> <out>`). Decoding 4K/HEVC/60 fps originals is the slowest
  thing in the pipeline; doing it once, while transcribing and planning, keeps
  every render ~10 s. Put the `.edit.mp4` on the timeline, not the original.

## 4. Build the timeline

Write `workspace/project.json` with `PUT /api/project` (body's `revision` must
exceed the current one), or `fablecut_set_project` / `fablecut_patch_project`
when the MCP tools are loaded. Skeleton:

```jsonc
{ "name":"<job>-<n>", "width":1080, "height":1920, "fps":30, "revision":N, "panSchema":1,
  "background":"#000000", "markers":[{"t":<beat>}…, {"t":<drop>,"label":"drop","color":"red"}],
  "media":[{"id":"m_s1","name":"s1.mp4","kind":"video","src":"/media/s1.mp4"}, …],
  "clips":[
    {"id":"v1","mediaId":"m_s1","kind":"video","track":"V1","start":0,"in":0.5,"duration":1.8,
     "props":{"fit":"cover","volume":0}},
    {"id":"mus","mediaId":"m_music","kind":"audio","track":"A1","start":0,"in":0,"duration":<len>},
    {"id":"cap1","mediaId":null,"kind":"text","track":"V3","start":0.2,"duration":1.6,
     "props":{"text":"…","font":"Montserrat","fontSize":80,"textShadow":0,"strokeWidth":6,"y":420}}
  ]}
```

- Shot timing: follow the sample's `shots[]`, snapped to `beats`; hero shot on `drop`.
- 16:9 raw on 9:16: `fit:"cover"` crops the centre; shift framing with `x`
  (px from centre) or `scale` > 1. Check every shot's framing with a frame grab.
- Audio: a video clip plays its own sound. To control a segment's sound separately
  (mute a tail, duck), set the video clip's `volume:0` and add an `audio` clip with the
  same `mediaId` / `in` / `duration` / `linkGroup` on A1. Otherwise both play.
- Speech reels: keep the select's own audio (`volume:1`) on V1, music at
  `volume:0.15–0.25` on A1, or duck it with the `fablecut_auto_duck` MCP tool
  (the repo's `.mcp.json` registers the `fablecut_*` tools) — see "Audio mix". Captions: transcript words →
  `.fablecut/examples/auto-captions/auto_captions.py`, then restyle to the brief.
- **What the ffmpeg renderer draws** (see `scripts/fast_render.py` docstring): cuts;
  static `fit` / `scale` / `x` / `y` / `opacity` (punch-ins are hard-cut zooms, as in
  most tutorial reels); audio `volume` / `gain` / mono; text with font, weight, size,
  colour, per-line pill, stroke, `textShadow:0`. It refuses anything else (keyframes,
  grades, transitions, textAnim, speed…) with a list. When a sample needs one of
  those, add it to `fast_render.py` (ffmpeg has zoompan, xfade, eq/curves, setpts)
  and check it against the sample — don't fall back to a browser.
- Look at the composed picture before rendering:
  `scripts/render.sh --frame <t> <still.jpg>` (~5 s each) for every shot.

## 5. Render and QC

```
reel-pipeline/scripts/render.sh reel-pipeline/jobs/<job>/out [name]
reel-pipeline/scripts/qc.sh <out.mp4> <sample.mp4>
```

Output is **720×1280** by default (the user's delivery size; `RES=1080` for
1080×1920): H.264 CRF 19 veryfast, AAC 256k, two-pass loudnorm to −14 LUFS
(`LUFS=` overrides). A 30 s reel renders in about 10 s. Keep the turnaround fast:
run prep, transcription and the sample analysis in parallel, build the
timeline from a script (`jobs/<job>/build_reelN.py`), preview 3–4 key frames,
render once, QC. Transcribe the render (`scripts/transcribe.py`) and read it: jump cuts in
narration must still say what was intended, and muted parts must be silent. Word
timestamps can be off by ~0.3 s, so check every cut inside a sentence in the
transcript. Compare the two sheets and the numbers against the brief (length,
cut rhythm, caption look, loudness). Fix and re-render rather than shipping a mismatch. Then copy
`workspace/project.json` to `jobs/<job>/project-<n>.json`, send the MP4 with
SendUserFile, and commit the brief + project snapshots.

## Be honest

A rebuild matches the sample's structure and styling, not its footage. Say what
was approximated (fonts not identified exactly, effects not reproduced) and
only use samples as references, never their footage, in the finished reels.
