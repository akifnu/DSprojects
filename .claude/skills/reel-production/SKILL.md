---
name: reel-production
description: Cut finished vertical reels from long-form footage in the exact style of a sample reel, singly or in batches of 50+, with the agentic reelkit production line in reel-pipeline/. Use whenever the user uploads sample reels, long-form footage, music or SFX, or asks for reels "like this".
---

# Reel production (agentic)

Uploads trigger this automatically (UserPromptSubmit hook `.claude/hooks/reel-upload.py`).

## The layers
- **Orchestrator (code)** `reelkit/orchestrator.py`, CLI `reel job …`: a job state machine
  `intake → analyze → VERIFY_STYLE → transcribe → PLAN → prepare → render → REVIEW → deliver`.
  It runs every mechanical step itself and stops only at the CAPS steps, saying which agent is
  needed. Its POLICY sets the output/efficiency balance (cache + reuse verified styles, decode only
  planned ranges, calibrated parallel renders, review = all fail/warn + √n spot checks, ≤ 2 fix
  rounds then escalate, model tier per role).
- **Workflow** `.claude/workflows/reel-job.js` (saved workflow `reel-job`, args `{job}`): loops
  advance → hand-off until done, fanning QC review out to parallel reviewers.
- **Agents** `.claude/agents/`: `reel-runner` (haiku: runs/waits on CLI), `sample-analyst`
  (verifies/corrects the measured style), `reel-planner` (chooses moments, writes plan.json,
  applies fixes), `qc-reviewer` (sonnet: reviews renders, returns precise fixes).

## What you (the main session) do
1. `reel-pipeline/reel check` → READY, else `reel-pipeline/setup.sh` (ffmpeg 9.0 + models).
2. `reel-pipeline/reel job new <files> [--count N] [--duration S] [--lang en|hi|…] [--notes "…"]`
   with whatever the user asked for. Prints the job id.
3. Run the Workflow tool with `name: "reel-job"`, `args: {"job": "<id>"}` (this skill authorises it).
4. When it returns: `reel job status <id>`; send `deliver.files` with SendUserFile; summarise
   counts/verdicts, anything escalated (`deliver.review`), and approximations (fonts not exactly
   identified, music the user must add in-app, etc.).
Missing inputs (no sample, or no footage yet) → the job reports `blocked: …`; tell the user what to
upload; `reel job advance <id>` continues once files are added via a new job.

## Manual control
`reel job status|advance|done|list`, and underneath: `reel sample`, `reel source`, `reel batch`,
`reel render`, `reel qc`, `reel loop`. Shapes in reel-pipeline/SPEC.md.

## Rules
- Never commit media; code/docs only, pushed straight to `main`.
- Never use a browser/FableCut; rendering is ffmpeg 9 + the OpenCV compositor.
- Report QC honestly, including warnings and anything escalated.
