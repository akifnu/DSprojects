#!/usr/bin/env python3
"""UserPromptSubmit hook: when a message carries uploaded video/audio, tell Claude to
start (or extend) a reel job and run the reel-job workflow. Stdlib only, fast."""
import json
import os
import re
import sys

MEDIA = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
prompt = data.get("prompt", "")
paths = []
for m in re.finditer(r'@?"?(/[^"\s]+)"?', prompt):
    p = m.group(1).rstrip('",')
    if p.lower().endswith(MEDIA) and os.path.exists(p) and p not in paths:
        paths.append(p)
if not paths:
    sys.exit(0)
files = " ".join(f'"{p}"' for p in paths)
context = (
    f"[reel production line] {len(paths)} media file(s) uploaded:\n" + "\n".join(f"- {p}" for p in paths) +
    "\nFollow the reel-production skill: (1) `reel-pipeline/setup.sh` if `reel-pipeline/reel check` is not READY; "
    f"(2) `reel-pipeline/reel job new {files} [--count N] [--duration S] [--lang xx] [--notes ...]` using any "
    "count/duration/language/instructions in the user's message (files are auto-classified: short vertical video = "
    "sample, long video = source, audio = music); if the user is adding files to a job that is still open, "
    "say so and create a new job only for a new request; (3) run the saved workflow `reel-job` with "
    "args {\"job\": \"<job id>\"}; (4) when it returns, send the reels in the job's deliver.files with SendUserFile "
    "and summarise verdicts and approximations. If only samples or only footage were uploaded, run steps 1-2, "
    "report what the job is waiting for, and stop."
)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}))
