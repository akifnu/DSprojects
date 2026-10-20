# reelkit contracts

Every stage reads and writes plain JSON. These shapes are the contract between
modules; change them only together with every reader.

## Data store (`data/`, git-ignored)

```
data/assets/<id>/            one per ingested file; id = sha-256[:16]
    meta.json                {id, kind, name, file, sha256, probe, ingested}
    original.<ext>
    transcript.<lang>.json   asr.transcribe output
    style.json               sample analysis (kind = sample)
    review.jpg               sample review sheet
    mezz/<s>_<e>.mp4         editing copies of source ranges (kind = source)
    faces.json               face tracks (kind = source)
data/batches/<name>/
    plan.json                written by Claude (selection + overrides)
    reels/<id>/timeline.json compiled
    reels/<id>/<id>.mp4      rendered
    reels/<id>/qc.json       checks
    reels/<id>/sheet.jpg     review sheet
    manifest.json            all reels: status, files, QC verdicts
    out/NN_<title>.mp4       deliverables
data/fonts/                  Google Fonts cache
```

## Job — `data/jobs/<id>/job.json` (orchestrator.py)

```jsonc
{"id": "j20261020-103000", "status": "needs:plan" | "rendering" | "done" | "blocked: …",
 "request": {"count": 30, "lang": "en", "duration": 30, "notes": ""},
 "files": [{"path": "…", "kind": "sample|source|music|image", "asset": "<id>"}],
 "inputs": {"samples": [], "sources": [], "music": [], "sfx": [], "images": []},
 "steps": {"intake": {"done": true, "at": "10:30:02"}},           // completed steps
 "needs": {"step": "plan", "agent": "reel-planner", "model": "inherit", …},  // when waiting
 "rounds": 0, "fixes": [], "only": null,                          // review → re-plan loop
 "manifest": "data/batches/<id>/manifest.json",
 "deliver": {"files": [], "review": [], "contact_sheet": "…", "summary": {}},
 "log": [{"at": "10:30:02", "msg": "…"}]}
```
Batches of a job live in `data/batches/<job id>/`.

## Transcript (asr.py)

```jsonc
{"lang": "en", "timing": "word" | "phrase", "duration": 30.0,
 "words": [{"w": "Hello", "s": 0.08, "e": 0.24, "p": 0}],      // p only for phrase timing
 "sentences": [{"id": 0, "s": 0.08, "e": 1.36, "w0": 0, "w1": 2, "text": "..."}]}
```

## Style spec — `style.json` (sample analysis → compiler)

Measured from a sample reel; px values are at the **sample's** canvas size and the
compiler scales them by `out_height / sample.height`.

```jsonc
{
 "version": 2,
 "sample": {"id": "d19e…", "name": "S1.mp4", "duration": 25.7, "width": 720, "height": 1280, "fps": 30},
 "pacing": {
   "shot_count": 12, "avg_shot": 2.1, "median_shot": 1.8, "min_shot": 0.4, "max_shot": 6.0,
   "shot_durations": [1.97, 3.77],
   "cuts_per_min": 28,
   "speech_wpm": 180,
   "max_pause": 0.15,               // longest silence kept inside speech (s): longer ones are cut
   "cut_on": {"sentence": 0.7, "beat": 0.3}
 },
 "shots": [{"s": 0, "e": 1.97, "dur": 1.97,
            "transition_in": "cut" | {"type": "dissolve|flash|zoom|whip_left|whip_up|…", "dur": 0.3},
            "move": {"type": "static|zoom_in|zoom_out|pan", "scale": 1.12, "ease": "inout",
                     "dx": 0, "dy": 0, "shake": 0.0},
            "moves": [{"type": "zoom_in", "t0": 0.4, "t1": 0.8, "scale": 1.15, "ease": "out"}],
                                      // quick push-ins / zoom bursts inside the shot (times from shot start)
            "punch": 1.25,            // scale vs previous shot when the cut re-frames the same scene
            "jump_cut": true}],       // (or "new_scene": true)
 "motion": {"punch_rate": 0.5, "punch_scales": [1.0, 1.25],
            "push_rate": 0.3, "push_scale_per_s": 0.03, "push_ease": "inout", "shake": 0.0},
 "transitions": {"counts": {"cut": 10, "zoom": 2}, "default": "cut",
                 "special": {"type": "zoom", "dur": 0.3, "rate": 0.15}},
 "captions": {
   "present": true,
   "style": {},                      // captions.py style dict (font, weight, size, fill, stroke,
                                     //   shadow, box, highlight, anim, case, y, max_width, line_height)
   "words_per_caption": {"median": 3, "max": 5},
   "max_lines": 2, "chars_per_line": 22,
   "duration": {"median": 1.2, "min": 0.4},
   "lead": -0.05,                    // caption start − first spoken word start (s)
   "case": "upper|title|sentence|none",
   "events": [{"s": 0, "e": 1.97, "text": "…", "box": [x0, y0, x1, y1], "role": "caption"}]
 },
 "titles": {"present": true, "style": {}, "events": []},       // hooks / headlines
 "overlays": {"persistent": [{"text": "@handle", "box": []}]},
 "grade": {"look": {}, "sharpen": 0.0, "grain": 0.0, "vignette": 0.0,    // look = grade.Look.stats()
           "noise_std": 0.0, "corner_ratio": 1.0, "acutance": 0.0},       // raw measures behind grain/vignette
 "audio": {
   "loudness": {"I": -14.1, "TP": -0.1, "LRA": 3.9},
   "speech": {"present": true, "coverage": 0.8, "lang": "en"},
   "music": {"present": false, "gap_level_db": null, "bpm": null, "beats": [], "duck_db": 12},
   "sfx": {"events": [{"t": 1.97, "type": "whoosh", "dur": 0.3, "level_db": -18}],
           "on_cut_rate": 0.6, "on_caption_rate": 0.0, "types": {"whoosh": 5}}
 },
 "structure": {"duration": 25.7, "hook": {"dur": 2.0, "text": "…", "type": "caption|title|spoken"},
               "cta": {"text": "…", "s": 21.9}, "loop": false, "first_word_at": 0.1},
 "transcript": {},                   // the sample's own transcript
 "review": {"verified": false, "notes": []}   // set by Claude after checking the review sheet
}
```

## Plan — `plan.json` (Claude → compiler)

```jsonc
{
 "name": "podcast-ep12",
 "style": "<sample asset id>" | "path/to/style.json",
 "output": {"width": 720, "height": 1280, "fps": 30, "quality": "final" | "draft", "loudness": -14},
 "sources": {"S": "<source asset id>"},
 "music": {"asset": "<id>", "gain_db": -18} | null,
 "sfx": {"whoosh": "<asset id>"} | null,            // null = library/sfx defaults
 "reels": [{
   "id": "r01", "title": "Why most diets fail",
   "timeline": "path/to/timeline.json",            // optional: pre-built, skips compiling
   "segments": [{"src": "S", "sentences": [120, 127]}, {"src": "S", "s": 812.4, "e": 830.1}],
   "hook": {"src": "S", "sentences": [128]},       // optional cold open, placed first
   "title_text": "STOP doing this",                 // optional on-screen hook title
   "caption_text": {"121": "corrected sentence"},  // optional per-sentence caption fixes
   "emphasis": ["never", "free"],                  // optional words to highlight
   "cta": "Follow for part 2",                     // optional end text
   "drop_words": [[121, 3]]                        // optional (sentence id, word index) cuts
 }]
}
```

## Timeline — `timeline.json` (compiler → renderer)

Defined in `reelkit/render.py`'s module docstring (canvas, media, video clips with
keyframes and transitions, captions with word timings, styles, grade, audio clips,
encode settings).

## Rules for every module

- ffmpeg/ffprobe only through `reelkit.paths.FFMPEG` / `FFPROBE` (static 9.0 build).
- No browsers, no network at render time (fonts are cached on first use).
- Every module has `tests/test_<module>.py` (unittest), runnable on synthetic media
  in seconds; heavy runs use `nice -n 10` and short clips.
- Speed claims are measured in CPU-seconds and wall-seconds on this 4-core box.
