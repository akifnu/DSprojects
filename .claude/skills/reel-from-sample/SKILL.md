---
name: reel-from-sample
description: Turn raw long footage into finished vertical reels that follow a sample reel's structure and style, using the FableCut editor in reel-pipeline/. Use when the user provides reel samples and raw footage, asks to "make reels like this", or drops files into reel-pipeline/jobs/.
---

# Reel from sample

The pipeline lives in `reel-pipeline/` (see its README). FableCut is the editor:
a JSON timeline (`reel-pipeline/workspace/project.json`) rendered by a browser
compositor and encoded by ffmpeg. Its full manual is
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
- Cut each select (with ~0.5 s handles) to a browser-safe file:
  `scripts/prep.sh <raw> <select-name> <in> <out>` → prints its `/media/…` src.
  Music / voice-over audio: `scripts/prep.sh <file> <name>` → FLAC.
  **The container's headless Chromium cannot decode H.264/AAC; every file on the
  timeline must go through prep.sh** (WebM/VP9/Opus or FLAC). Prep selects,
  not whole long files.

## 4. Build the timeline

Write `workspace/project.json` with `PUT /api/project` (body's `revision` must
exceed the current one), or `fablecut_set_project` / `fablecut_patch_project`
when the MCP tools are loaded. Skeleton:

```jsonc
{ "name":"<job>-<n>", "width":1080, "height":1920, "fps":30, "revision":N, "panSchema":1,
  "background":"#000000", "markers":[{"t":<beat>}…, {"t":<drop>,"label":"drop","color":"red"}],
  "media":[{"id":"m_s1","name":"s1.webm","kind":"video","src":"/media/s1.webm"}, …],
  "clips":[
    {"id":"v1","mediaId":"m_s1","kind":"video","track":"V1","start":0,"in":0.5,"duration":1.8,
     "props":{"fit":"cover","volume":0}},
    {"id":"mus","mediaId":"m_music","kind":"audio","track":"A1","start":0,"in":0,"duration":<len>},
    {"id":"cap1","mediaId":null,"kind":"text","track":"V3","start":0.2,"duration":1.6,
     "props":{"text":"…","font":"Montserrat","fontSize":80,"textAnim":"word-pop","strokeWidth":6,"y":420}}
  ]}
```

- Shot timing: follow the sample's `shots[]`, snapped to `beats`; hero shot on `drop`.
- 16:9 raw on 9:16: `fit:"cover"` crops the centre; shift framing with `x`
  (px from centre) or `scale` > 1. Check every shot's framing with a frame grab.
- Speech reels: keep the select's own audio (`volume:1`) on V1, music at
  `volume:0.15–0.25` on A1, or duck it with the `fablecut_auto_duck` MCP tool
  (the repo's `.mcp.json` registers the `fablecut_*` tools) — see "Audio mix". Captions: transcript words →
  `.fablecut/examples/auto-captions/auto_captions.py`, then restyle to the brief.
- Look: `filterPreset`, `props.grade` (manual "Color"), `keyframes.scale` for
  push-ins, `keyframes.speed` for ramps, adjust layers for impact frames
  (manual "Recipes", "Remake a reference video").
- Check the composed picture before exporting: a 1–2 s test export, or frame grabs
  of each select at its `in`.

## 5. Export and QC

```
reel-pipeline/scripts/export.sh reel-pipeline/jobs/<job>/out instagram-reels
reel-pipeline/scripts/qc.sh <out.mp4> <sample.mp4>
```

Export renders at roughly 1 frame/s here (no GPU): ~10 min for a 20 s reel.
Run it with `run_in_background` and keep working on the next reel's plan.
Compare the two sheets and the numbers against the brief (length, cut rhythm,
caption look; export.sh normalizes audio to −14 LUFS, override with `LUFS=`). Fix and re-export rather than shipping a mismatch. Then copy
`workspace/project.json` to `jobs/<job>/project-<n>.json`, send the MP4 with
SendUserFile, and commit the brief + project snapshots.

## Be honest

A rebuild matches the sample's structure and styling, not its footage. Say what
was approximated (fonts not identified exactly, effects not reproduced) and
only use samples as references, never their footage, in the finished reels.
