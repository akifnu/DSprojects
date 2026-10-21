# reelkit contracts

Every stage reads and writes plain JSON. These shapes are the contract between modules and
between the Python side and the agents; change them only together with every reader.

`reelkit/contracts.py` is the executable form and the one implementation of these rules (a
prompt telling an agent what to produce is not the enforcement point): `validate_request`,
`validate_style`, `validate_plan`, `validate_timeline` return `{"errors": [...],
"warnings": [...]}` (errors block, warnings are recorded), `normalize_review` checks review
results, `plan_entry_hash` fingerprints plan entries. Unknown keys are errors where they could
ask for an effect nothing implements (plan reels and segments, timeline clips, text events,
text styles, grade, audio and encode settings) and warnings elsewhere; `meta` (an object) is
the free extension field of the plan, its reels and segments, timeline clips and events and the
timeline itself.

Common rules (`reelkit/fsutil.py`): JSON and caches are written to a unique temp file beside the
target, fsynced and renamed (a reader sees the old file or the new one); per-key advisory locks
are `fsutil.file_lock(path)` → `<path>.lock`; an unreadable or corrupt cache entry is a miss
(logged), never a crash. Hashes are full sha-256 hex; `sha256_json` = sha-256 of canonical
JSON (sorted keys, no whitespace, UTF-8).

## Data store (`data/`, REELKIT_DATA, git-ignored)

```
data/assets/<id>/                id = sha-256[:16] of the stored bytes; one per distinct content
    meta.json                    {id, kind, roles, name, file, sha256, bytes, probe, ingested, origins, store}
                                 kind: the role it was first stored for, never changed; roles: every
                                 role it was ingested for; origins: where it came from (metadata
                                 only); store: 2 = a private copy verified against sha256
    original.<ext>               the stored bytes: copied (never hard-linked to the upload), verified,
                                 published with one rename under a per-id lock
    transcript.<lang>.json       asr.transcribe output; .txt = the compact sentence index for planning
    audio.flac, envelope.npy     the source's soundtrack and its 5 ms RMS envelope (plan.envelope)
    style.json, review.jpg       sample analysis (a sample) and its review sheet
    style.verified.json          the last style a job verified for this measurement; reused by later
                                 jobs only while analysis.fingerprint matches
    mezz/index.json              editing copies of source ranges: [{s, e, file, width, height, fps}],
    mezz/<s>_<e>_<h>p<fps>.mp4   file relative to mezz/ (a moved data volume keeps its copies)
    look.<key>.json              colour statistics of the footage measured (key: the copies' ranges and
                                 geometry, sampling, grade code)
data/assets/.tmp/                ingest temp copies (swept when stale)
data/assets/.locks/              per-id ingest locks
data/assets/.seen/<key>          digest memo per (path, inode, size, mtimes): a hint, never proof
data/assets/.trash/              broken asset directories moved out of the way, then deleted
data/batches/<name>/             see "Batch layout and manifest"
data/jobs/<id>/                  see "Job"
data/fonts/                      Google Fonts cache (library/fonts, prewarmed in the image, is read first)
data/readiness/encoder.json      NVENC probe results per ffmpeg build / GPU / driver (readiness.py)
data/asr/                        Whisper hex token tables when the model directory is read-only
```

Models live in `$REELKIT_MODELS` (default `.models/`, the same variable for setup.sh and the
runtime) with `CHECKSUMS` (sha256sum format, recorded when setup.sh downloads a model); tools
in `.tools/` (`ffmpeg -> ffmpeg-n9.0-<sha>`, `CHECKSUMS`, `ffmpeg.version`, `profile`).

## Request — `data/jobs/<id>/request.json` (contracts.validate_request)

```jsonc
{"schema_version": 1,
 "assets": [{"path": "/inputs/style sample.mp4", "role": "sample"},     // sample | source | music | sfx | image
            {"path": "/inputs/interview.mp4", "role": "source"}],
 "request": {"count": 12,                  // positive integer or null (planner decides)
             "duration_seconds": 30,       // positive number or null (the sample's length)
             "lang": "en",                 // "en", "auto" or a Whisper language code
             "notes": "Keep technical terms and avoid a CTA"},   // the user's instructions, for the planner
 "execution": {"encoder": "auto",          // auto | cpu | nvenc
               "require_gpu": false,       // true: no NVENC → fails before full preparation/rendering
               "allow_cpu_fallback": true, // auto / nvenc may fall back to x264 (recorded with the reason)
               "asr_provider": "cpu"}}     // cpu | cuda (needs the GPU profile)
```
Paths are made absolute and must exist; a file without a role gets a heuristic guess, recorded
as a suggestion (`role_source: "heuristic"`). Invalid count, duration, language, role or a
missing file fail `job new` with every reason listed. The normalised request and its
`request_hash` travel with the plan need and stay in the job's provenance.

## Job — `data/jobs/<id>/job.json` (orchestrator.py)

Ids: `j<YYYYmmdd-HHMMSS>-<6 hex>` (created exclusively) or an explicit name matching
`[A-Za-z0-9][A-Za-z0-9._-]{0,63}` that must not exist; `latest` is resolved once by the CLI.

```jsonc
{"id": "j20261020-103000-4f2a9c", "schema": 2,
 "revision": 17,                           // +1 per saved transaction (under the job's state lock)
 "status": "needs:review", "terminal": false,
 "policy": {"version": 4,
            "storyboard": {"width": 360, "frames_per_reel": 3, "max_frames": 240, "timeout_s": 120,
                           "gif_width": 240, "gif_fps": 6, "gif_seconds": 3,
                           "max_gif_frames": 1440, "max_gif_bytes": 2097152}, …},
 "request": {…}, "request_hash": "<sha-256>", "request_warnings": [],
 "files": [{"path", "role", "role_source", "suggested_role"?, "asset", "sha256", "error"?, "added_rev"}],
 "inputs": {"samples": [], "sources": [], "music": [], "sfx": [], "images": []},   // asset ids by role
 "steps": {"intake": {"done": true, "at": "…", "key": …}, …},   // a step whose key changed is redone
 "needs": {"step": "review", "need_id": "review-4", "revision": 17, "agent": "qc-reviewer", "model": "sonnet",
           …step detail…},                 // open judgment request; null otherwise
 "completions": {"<need_id>": {"step", "sha256", "accepted_sha256", "revision", "at"}},
 "style_version": {"version": 2, "path": ".../styles/style.v2.json", "sha256", "sample", "fingerprint"},
 "plan": {"generation": 3, "path": ".../plans/plan.g3.json", "sha256", "reels": ["r01", …],
          "entry_hashes": {"r01": "<sha-256>"}, "warnings": []},
 "plan_generation": 3,
 "storyboard": {"key", "sha256", "manifest", "page", "sheet", "files": {"<path>": "<sha256>"},
                "frames", "gif_frames", "gifs": [{"reel_id", "file", "frames", "fps", "width", "height",
                                                  "duration", "sampled_ranges": [{"s", "e"}], "bytes"}],
                "cache_hits", "estimated_render_frames", "elapsed_s", "limitations"},
 "storyboard_approval": {"by": "user", "sha256", "plan_generation", "need_id", "at", "notes"},
 "storyboard_feedback": [{"plan_generation", "at", "notes"}],
 "render": {"seq": 4, "manifest": ".../manifests/render.4.json", "generation": 5, "counts": {…}, "encoder": {…}},
 "encoder": {"requested": "auto", "actual": "nvenc", "reason": "…", "gpu": "NVIDIA L4 (driver …)"},
 "review": {"ledger": {"r01": {"decision": "accept" | "fix_required" | "unreviewed" | "repairing",
                               "render_sha256", "qc_verdict", "findings": ["F0003"], "need_id",
                               "plan_generation", "requests", "at"}},
            "attempts_by_reel": {"r01": 1}, "rounds": 1, "notes": [], "errors": []},
 "findings": [{"id": "F0003", "reel_id": "r01", "code", "severity": "blocking" | "minor", "what", "how",
               "scope"?, "status": "open" | "noted" | "resolved" | "superseded" | "retired",
               "need_id", "plan_generation", "render_sha256", "at"}],   // every finding, whatever the budget
 "repair": null, "repairs": [],            // the repair round in progress / history
 "attempt": "a003-9f1c", "attempt_seq": 3,  // the worker attempt that owns the job (fencing token;
                                           //   a = detached worker, f = foreground `job advance`)
 "error": null, "errors": [], "attention": null,
 "deliver": {…},                           // see below, once terminal
 "log": [{"at", "msg"}]}
```

Status while open: `intake` (new) | `waiting:sample` / `waiting:source` (inputs missing; `job add`
resumes, unaffected work is kept) | `waiting:worker` (`job start` continues) | `needs:<step>` (an
agent or user must answer) | `running:<step>`. Terminal: `succeeded` (every planned reel delivered and
reviewed where required, no open blocking finding, no shortfall) | `partial` (allow_partial and ≥ 1
reel accepted; the rest listed) | `needs_attention` (a blocking finding or missing judgment
outlived its budget, or nothing was accepted) | `failed` (infrastructure / validation; typed record
in `error`) | `cancelled`.

Needs, by step (all carry `step, need_id, revision, agent, errors`; agent needs also
carry `model, rejections`):
- `verify_style`: `{style: jobs/<id>/style.work.json (the job-local copy to edit), review: review.jpg,
  sample, sample_id, extra_samples, review_out, fixes}` — answered by setting `review.verified`;
  validated by contracts.validate_style, then published as the immutable `styles/style.v<k>.json`.
- `plan`: `{plan, style (the pinned style version path), transcripts, sources, count,
  target_duration, lang, notes, request (normalised), request_hash, fixes, only, repair_round,
  plan_generation, previous_plan, errors}` — validated by contracts.validate_plan against the
  transcripts and the request; in a repair round every entry outside `only` must keep its
  contracts.plan_entry_hash.
- `approve_storyboard`: `{agent: "user", storyboard_sha256, page, sheet, manifest,
  frames, gifs, gif_frames, estimated_render_frames, limitations, plan_generation, errors}`. The workflow
  returns immediately with `requires_user_approval: true`. The main session presents the
  board and records the user's specific decision with `job approve` or `job revise`;
  both require `--need`, `--revision` and `--storyboard SHA256`. Revision additionally
  needs nonempty feedback (`--notes-file` or `--notes`). `job done` cannot answer it.
  Approval binds the current plan generation, complete input fingerprint and hashes
  of every preview artifact. A changed/missing artifact or input rejects approval;
  `prepare` and `render` recheck it before starting. There is no automatic approval
  timeout. A new accepted plan always requires a new decision; feedback is included
  as `storyboard_feedback` in the next plan need and does not spend QC repair budget.
- `review`: `{plan_generation, manifest, render_generation, assigned: [{id, render_sha256,
  qc_verdict, status, issues, error, file, sheet, qc, previous_findings}], rounds_left_by_reel,
  result_file: jobs/<id>/inbox/<need_id>.json, style, plan, schema_version}`.

Files beside job.json: `job.prev.json` (the previous revision, for recovery), `request.json`,
`style.work.json`, `styles/style.v<k>.json` (read-only), `plans/plan.g<k>.json` (read-only),
`inbox/<need_id>.json` (review results), `manifests/render.<seq>.json` (the manifest each render
returned), `delivery.json`, `attempts/<attempt>.json` `{id, pid, pgid, started, acknowledged,
heartbeat, stage, ended, exit, error_record}` with `attempts/<attempt>.log`, `events.jsonl` (one
JSON object per line: `at, ts, job, stage, attempt, revision, event, …`), `cancel` (flag file),
`state.lock`. Batches of a job live in `data/batches/<job id>/`.

`storyboards/board-<key>-<nonce>/` holds independently copied JPEG panels,
`<reel>-preview.gif`, `contact-sheet.jpg`, `index.html` and `manifest.json`; the manifest
lists reels, planned speech, sampling times, original source ranges per shot, durations,
GIF dimensions/durations/excerpt ranges, limitations, options and frame counts.
`storyboards/cache/<reel-key>/` holds compiled preview timelines and hash-checked
frames and GIFs. Per-reel keys include source identities, transcript contents, style, fonts,
output configuration and preview/compiler/renderer code. Jobs from older policy
versions receive this mandatory gate on migration; no approval is synthesized.
The preview compiler uses original media and cached transcripts with no audio,
editing copies, full tracking or full-source colour measurement. Defaults are
three sparse stills per reel, at most 360×640, plus a silent looping GIF of selected
hook/middle/payoff excerpts (up to three seconds, 240px wide, 6fps, 2MiB).
GIFs compose only these excerpts through the existing compositor, including text,
motion and transitions, without running the full video/audio renderer. Their palette
and frame rate are reduced and they do not verify smooth tracking or the final grade.
Budgets: 240 stills/job, 1,440 GIF frames/job (conservatively reserved before composing),
120 seconds per board build. Hash checks cover GIFs in both cache and board copies;
damaged board copies cannot be approved, and damaged cache copies regenerate only
the affected reel. Budget violations fail before full preparation/rendering.

`deliver` = `{files: [accepted deliverable paths from manifest.deliverables], review: [failing
reels' files], contact_sheet, manifest, generation, plan_generation, counts: {requested, planned,
rendered, qc_accepted, accepted, blocked, failed, retired, delivered}, unresolved: [findings],
shortfall: requested − accepted (or 0), blocked: {reel: reason}, accepted_by: {reel: "review" | "qc"},
encoder, summary}`. Delivery reads only the manifest's deliverables (hash-checked against the
rendered and reviewed reel), never a directory listing.

## Review result — `inbox/<need_id>.json` (contracts.normalize_review)

```jsonc
{"schema_version": 1, "job_id": "j…", "need_id": "review-4", "expected_revision": 17, "plan_generation": 3,
 "results": [{"reel_id": "r01", "render_sha256": "<copied from the assignment>", "decision": "fix_required",
              "findings": [{"code": "CLIPPED_FIRST_WORD", "severity": "blocking",
                            "what": "first word clipped at 0.0 s", "how": "start one sentence earlier (S0411)",
                            "scope": "style"}]},          // scope optional; "style" or a STYLE_* code = style-level
             {"reel_id": "r02", "render_sha256": "…", "decision": "accept", "findings": []}],
 "notes": "anything systemic"}
```
Other top-level keys are ignored; `render_sha256` may be omitted. The legacy `{"fixes": [{reel,
what, how}], "ok": [ids]}` is normalised (each fix → a blocking finding of a fix_required
result). Rules: a stale `job_id` / `need_id` / `expected_revision` / `plan_generation` rejects the
envelope; an unknown reel id, a `render_sha256` other than the assigned render (stale), a
malformed result, conflicting decisions for one reel, `accept` with a blocking finding, or
`accept` of a reel whose QC verdict is fail / error (a technical failure cannot be waived) is an
error and leaves that reel unreviewed; results with the same decision for one reel are merged.
Omitted reels stay unreviewed and are re-requested (policy review_retries); every finding is
recorded in `job.findings` whatever the repair budget.

## Transcript (asr.py)

```jsonc
{"lang": "en", "timing": "word" | "phrase", "duration": 30.0, "provider": "cpu" | "cuda",
 "fingerprint": "<model dir>/<lang>/<provider>/<timing>/<hash>",   // models, language, provider, config
 "timing_note": "…",                                              // phrase timing only: word times are estimated
 "words": [{"w": "Hello", "s": 0.08, "e": 0.24, "p": 0}],         // p only for phrase timing
 "sentences": [{"id": 0, "s": 0.08, "e": 1.36, "w0": 0, "w1": 2, "text": "..."}]}
```
A cached transcript whose fingerprint is not current is transcribed again.

## Style spec — `style.json` (sample analysis → compiler; contracts.validate_style)

Measured from a sample reel; px values are at the **sample's** canvas size and the
compiler scales them by `out_height / sample.height`. Required: `version` (≤ 2), `sample`
{width, height, duration}, `shots` (list); `captions.style` and `titles.style` are checked
strictly (they are drawn as given: only captions.py keys, supported animations, highlight and
box modes, hex colours).

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
                 "special": {"type": "zoom", "dur": 0.3, "rate": 0.15}},   // type: a transition render.py draws
 "captions": {
   "present": true,
   "style": {},                      // captions.py style dict (font, weight, size, fill, stroke,
                                     //   shadow, box, highlight, anim, case, y, max_width, line_height);
                                     //   highlight.mode "active" (karaoke) | "keyword" (one word per
                                     //   caption stays marked), highlight.rate = share of captions marked
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
 "analysis": {"fingerprint": "…",    // what the measurement depends on: analysis code, language,
              "code": "…", "lang": "en",   //   the transcript's ASR fingerprint, the sample's content
              "asr": "<transcript fingerprint>", "sample_sha256": "…", "seconds": 41.2},
 "review": {"verified": false, "notes": []}   // set by the sample-analyst in the job's working copy
}
```
A verified style is immutable by version: the job pins `styles/style.v<k>.json` (review.verified
true, `review.analysis_fingerprint`, `review.version`) and the plan's `style` names that path.

## Plan — `plan.json` (planner → compiler; contracts.validate_plan)

```jsonc
{
 "name": "podcast-ep12",                         // the job id for job batches
 "style": "<sample asset id>" | "path/to/style.json",   // a job pins its verified style version's path
 "output": {"width": 720, "height": 1280, "fps": 30, "quality": "final" | "draft", "lang": "en",
            "loudness": -14},                     // optional; default = the sample's (clamped −18…−7 LUFS)
 "sources": {"S": "<source asset id>"},
 "music": {"asset": "<id>", "gain_db": -18} | null,
 "sfx": {"whoosh": "<asset id>"} | null,            // null = library/sfx defaults
 "meta": {},                                       // free extension, ignored by the compiler
 "reels": [{
   "id": "r01", "title": "Why most diets fail",    // id: [A-Za-z0-9][A-Za-z0-9_-]{0,63}, unique
   "timeline": "path/to/timeline.json",            // optional: pre-built, skips compiling
   "segments": [{"src": "S", "sentences": [120, 127]}, {"src": "S", "s": 812.4, "e": 830.1}],
   "hook": {"src": "S", "sentences": [128]},       // optional cold open, placed first
   "title_text": "STOP doing this",                 // optional on-screen hook title
   "caption_text": {"121": "corrected sentence"},  // optional per-sentence caption fixes
   "emphasis": ["never", "free"],                  // optional keywords (drawn when highlight.mode = keyword)
   "cta": "Follow for part 2",                     // optional end text
   "drop_words": [[121, 3]],                       // optional (sentence id, word index) cuts
   "meta": {}                                      // free extension; any other key is an error
 }]
}
```
Checked against the transcripts: sentence ids `[first, last]` ascending and inside the transcript,
explicit times `0 ≤ s < e` within the source, every reel plays some speech, caption_text keys are
sentence ids (overrides for sentences outside the reel are warnings), drop_words pairs exist.
Warnings: count / duration (±25 % of source speech) / language different from the request, a
sentence used by two reels. `output.lang` is never required. plan_entry_hash(reel) =
sha256_json of the entry: an entry that hashes the same is unchanged (repair rounds, batch cache).

## Timeline — `timeline.json` (compiler → renderer; contracts.validate_timeline)

Defined in `reelkit/render.py`'s module docstring: canvas (`width`, `height` even, `fps`,
`duration`, `background`), `media` {key: path | asset:<id>}, `video` clips {media, start, in, dur,
fit, scale / x / y / rotation / opacity (number or keyframes [t, v] / [t, v, ease] / {t, v, ease}),
speed, track, subject, transition_in {type, dur}, hold_last, kind}, `captions` / `text` events
{s, e, text, words [{w, s, e, emph?}], style (name or object)}, `styles`, `grade` {lut, sharpen,
grain, vignette}, `audio` {clips [{media, start, in, dur, role voice|music|sfx, mono, gain_db,
fade_in, fade_out}], voice_chain, duck_db, loudness, true_peak}, `encode` {preset, crf, cq,
lossless}, `meta` {reel, title, runs, words}. A source that runs out before its clip ends fails
the render unless the clip sets `hold_last: true` (its last frame is held).
Checked: geometry and timings finite, media references resolve, keyframe times in order, eases
and transitions are ones render.py implements (cut, dissolve, dip_black, dip_white, flash, zoom,
whip_left/right/up/down), caption words in order; unknown top-level keys are warnings, unknown
clip / event / audio / grade / encode keys errors. An invalid timeline fails its reel without a render.

## deps.json — beside each compiled timeline (compile.dependencies)

What can change a reel's rendering without changing its timeline JSON:
```jsonc
{"assets": {"S": "<sha-256>"}, "copies": [{"path", "size", "mtime_ns"}],   // editing copies, soundtracks, media
 "transcripts": {"S": "<ASR fingerprint>"}, "style_sha256": "…", "lut_sha256": "…",   // LUT contents
 "fonts": {"Montserrat-800": "<sha-256 of the font file>"}, "sfx": {"<path>": "<sha-256>"},
 "music": "<sha-256>" | null, "renderer": 1}                                 // render.RENDERER_VERSION
```

## Batch layout and manifest (batch.py)

```
data/batches/<name>/
    plan.json  batch.log  batch.lock
    manifest.json  manifest.csv              the current generation's, also in generations/<g>/
    out -> generations/<g>/out               swapped in one rename once a generation is complete
    reels/<id>/  timeline.json  deps.json  compile.json {key, timeline_sha256, deps_sha256, at}
                 look.cube  <id>.mp4  qc.json  sheet.jpg
                 render.json | cache.json    {version, id, plan_entry_hash, render_key, render_sha256,
                                             encoder, draft, qc_key, qc_sha256, qc_verdict, …}:
                                             render.json = rendered, QC to do; cache.json = rendered and QC'd
    generations/<g>/  manifest.json  manifest.csv
                 out/NN_<slug>.mp4           accepted reels (QC pass / warn): the deliverables
                 out/_review/NN_<slug>.mp4   rendered but failing reels: never deliverables
                 out/contact_sheet.jpg
                 reels/<id>/{timeline.json, deps.json, qc.json, sheet.jpg}   snapshots
```
Generations older than the newest REELKIT_KEEP_GENERATIONS (10) are removed.

```jsonc
{"name", "style", "style_sha256", "plan_hash", "generation": 5, "quality", "lang",
 "encoder": {"requested", "actual", "reason", "gpu"}, "jobs", "workers", "only", "elapsed_s", "finished",
 "counts": {"planned", "rendered", "accepted", "failed", "pending", "retired", "delivered", "cached", "worked"},
 "summary": {"pass": 9, "warn": 2, "fail": 1},
 "reels": [{"id", "n", "title", "plan_entry_hash", "timeline", "render_key", "render_sha256", "verdict",
            "qc_verdict", "issues", "file", "sheet", "qc", "status": "accepted" | "failed" | "pending",
            "error", "error_record", "cached", "duration", "attempts"}],
 "retired": ["r09"],                        // ids gone from the plan (their files are not delivered)
 "deliverables": [{"id", "file", "sha256", "verdict"}],   // accepted entries of this generation only
 "out_dir": ".../generations/5/out"}
```
The manifest always covers every reel of the current plan (`only=` restricts work, never the
manifest). A deliverable's file is byte-identical to its entry's `render_sha256`.
`error_record` = `{stage, cmd, returncode, stderr_tail, cause, transient}` (ff.FFError /
render.RenderError).

## Readiness — `reel check` (readiness.py)

```jsonc
{"ready": true,                             // every required component ok
 "components": [{"name": "ffmpeg", "ok": true, "detail": "ffmpeg version n9.0.2-… (…/ffmpeg)", "required": true}],
 "encoder": {"requested": "auto", "actual": "cpu" | "nvenc" | null, "reason": "…", "gpu": "…" | null},
 "asr_provider": {"requested", "provider", "available": ["cpu", "cuda"], "ok", "reason", "sherpa": {…},
                  "sherpa_cuda_libs": [], "onnxruntime_providers": []},
 "gpu": {"present", "gpus": [{"index", "name", "driver", "memory_mib", "uuid"}], "label", "detail"},
 "deep": false, "checked": "…", "paths": {"root", "data", "models", "ffmpeg", "ffprobe"}}
```
Components: imports (numpy, cv2, PIL, scipy, sherpa_onnx, onnxruntime, rapidocr_onnxruntime),
pinned package versions (warning), ffmpeg ≥ 9 + ffprobe, filters (lut3d, loudnorm, ebur128,
colorspace, libvmaf, zscale) and encoders (libx264, aac, flac), every model file present and
non-empty (Parakeet, Whisper, Silero VAD, YuNet, OCR), data directory writable, free disk
(REELKIT_MIN_FREE_GB, default 5), caption fonts, GPU (nvidia-smi), encoder, ASR provider.
NVENC counts only after a real 30-frame h264_nvenc encode with the renderer's settings; `cuda`
ASR only with a CUDA build of sherpa-onnx and a visible GPU. `--deep` adds the model checksums
and one load-and-run of every model, each in its own process.

## Command line results

Every `reel job …` command prints one JSON object on stdout (logs go to stderr):
- `job new` → the job's status (`id`, `status`, …) + `request`, `warnings`;
  `job add` → status + `added`; `job start` → `{attempt, pid, pgid, acknowledged, existing, log, status}`
  once the worker acknowledged ownership.
- `job status` → `{id, status, terminal, revision, next, needs, attempt, busy, done, inputs, files,
  waiting, deliver, counts, findings_open, plan_generation, style_version, encoder, error,
  attention, request_hash, policy_version, log}`.
- `job wait` → status + `wait_exit`; exit 0 = the worker settled (an agent or input is needed, or
  the job is terminal), 2 = the worker failed or died (status.error says how), 124 = still running
  at the timeout.
- `job done` → status (`idempotent: true` for a repeat of an accepted answer); exit 2 when the
  answer is rejected.
- Errors: `{"error", "code", "errors": [...]}`, exit 2 for a rejected answer, 1 otherwise.
- `reel check` → the readiness JSON; exit 0 ready, 1 not ready. `reel batch` → `{name,
  generation, encoder, counts, summary, deliverables, retired, out_dir}`.

## Rules for every module

- ffmpeg/ffprobe only through `reelkit.paths.FFMPEG` / `FFPROBE` (the static 9.x build from
  setup.sh); every child process stays in the caller's process group and has its stderr drained.
- Validation through `reelkit.contracts`; atomic writes, tolerant reads, locks and hashes through
  `reelkit.fsutil`.
- No browsers, and no network at render time: fonts are prefetched before renders fan out
  (captions.prefetch_style_fonts; REELKIT_OFFLINE=1 forbids downloads).
- Tests are unittest (`tests/test_<module>.py`) on synthetic media; heavy runs use `nice -n 10`
  and short clips. Speed claims are measured in CPU-seconds and wall-seconds on a stated machine.
