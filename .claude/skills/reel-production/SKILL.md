---
name: reel-production
description: Cut finished vertical reels from long-form footage in the exact style of a sample reel, singly or in batches of 50+, with the reelkit production line in reel-pipeline/. Use whenever the user uploads sample reels, long-form footage, music or SFX, or asks for reels "like this".
---

# Reel production (reelkit)

Everything goes through `reel-pipeline/reel` (see its README; data shapes in SPEC.md).
Rendering is ffmpeg 9 + the OpenCV compositor only — never a browser or FableCut.

## 1. Setup (each session, ~10 s when cached)
`reel-pipeline/setup.sh && reel-pipeline/reel check` → must print READY.

## 2. Ingest
`reel ingest <uploads…> --kind sample` (reference reels), `--kind source` (long-form),
`--kind music`, `--kind sfx`. Duplicate uploads resolve to the same asset id — tell the
user when that happens.

## 3. Read the sample (once per style)
`reel sample <id>` → `data/assets/<id>/style.json` + `review.jpg`.
Read review.jpg: shots strip, and every real caption next to the same words
re-rendered from the measured style. If something is off (font, colour, size,
position, animation), correct style.json by hand, set `review.verified: true` and
note what you changed. Do not cut anything from an unverified style.

## 4. Prepare the long-form source
`reel source <id> --lang en|hi|… --prepare` (run in the background; 4K60 HEVC decodes
at ~1× real time) → `transcript.<lang>.txt`: one line per sentence, `S0123 mm:ss-mm:ss text`.

## 5. Plan (the only creative step — yours)
`reel plan <name> --style <sample id> --source S=<source id>`, then fill `reels` in
data/batches/<name>/plan.json (SPEC.md "Plan"). Per reel: segments by sentence ids,
optional cold-open hook, title_text, caption_text fixes (e.g. Hindi transcripts →
correct/romanised lines), emphasis words, CTA. Target the sample's duration ±20 %.
Pick self-contained moments with a strong first line; don't reuse a sentence across
reels unless asked. For 50+ reels read the transcript in chunks and write the plan in
one pass.

## 6. Run
`reel batch <name>` (run_in_background; resumable; `--only r03,r07` to redo some,
`--draft` for fast previews). Outputs in data/batches/<name>/: out/NN_<title>.mp4,
manifest.json/.csv, per-reel qc.json + sheet.jpg, out/contact_sheet.jpg.

## 7. Review and deliver
Read manifest: fix every `fail` (adjust plan/style, re-run with --only), look at the
sheet of every `warn`, and spot-check passes. Then send the reels with SendUserFile
and summarise: count, durations, any approximation (fonts not identified exactly,
copyrighted music the user must add in-app, etc.).

## Rules
- Never commit media; commit code and docs only. The user wants changes pushed straight to `main`.
- Speed: one long-form → plan in one pass → one batch run. Don't hand-build timelines.
- Honest QC: report what the checks and your review found, including warnings.
