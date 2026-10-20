#!/usr/bin/env python3
"""UserPromptSubmit hook: when a message names uploaded video, audio or image files, tell Claude
how to open a reel job with explicit roles (or add the files to an open job) and run the reel-job
workflow.

Paths are found in every form a prompt carries them — @"quoted", "quoted", 'quoted', @path,
plain absolute paths with spaces or non-ASCII characters, `\\ `-escaped spaces, file:// URLs — by
checking candidate spans against the disk (the longest existing one wins). Files inside the
pipeline's own data store (renders, assets) are ignored. Stdlib only and fast; it never gets in the
way of the prompt: any error exits 0 without output."""
import json
import os
import re
import shlex
import sys
import unicodedata
import urllib.parse

VIDEO = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")
AUDIO = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")
IMAGE = (".png", ".jpg", ".jpeg", ".webp")
MEDIA = VIDEO + AUDIO + IMAGE
QUOTES = "\"'`"
TRAIL = ".,;:!?)]}>" + QUOTES      # punctuation after a path that is not part of it
LEAD = "([{<,;=" + QUOTES          # characters that may directly precede an unquoted path
MAX_PROMPT = 200_000
MAX_SPAN = 1024
MAX_FILES = 100
TERMINAL = ("succeeded", "partial", "needs_attention", "failed", "cancelled", "done")
REEL_WORDS = re.compile(r"reel|short|tiktok|vertical|caption|clip|video|footage|logo|thumbnail|overlay|job", re.I)


def _word(*words):
    return re.compile(r"(?<![a-z])(?:" + "|".join(words) + r")(?![a-z])", re.I)


SFX_HINT = _word("sfx", "whoosh", "swoosh", "boom", "hit", "riser", "impact", "ding", "transition")
SAMPLE_HINT = _word("sample", "style", "reference", "ref", "example", "inspo", "template", "like")
SOURCE_HINT = _word("podcast", "interview", "episode", r"ep\d+", "full", "raw", "long", "source", "footage",
                    "webinar", "talk", "lecture", "stream", "vlog")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.realpath(os.environ.get("REELKIT_DATA") or os.path.join(ROOT, "reel-pipeline", "data"))


def _resolve(raw, cwd, relative_ok):
    """Absolute path of an existing media file the text names, else None."""
    p = raw.strip()
    if p.lower().startswith("file://"):
        p = urllib.parse.unquote(p[7:])
        if p.startswith("localhost/"):
            p = p[9:]
    p = os.path.expanduser(p.replace("\\ ", " "))
    if not os.path.isabs(p):
        if not relative_ok:
            return None
        p = os.path.join(cwd, p)
    if not p.lower().endswith(MEDIA):
        return None
    for q in dict.fromkeys((p, unicodedata.normalize("NFC", p), unicodedata.normalize("NFD", p))):
        try:
            if os.path.isfile(q):
                return os.path.abspath(q)
        except (OSError, ValueError):       # embedded NUL, overlong names
            continue
    return None


def _quoted(line, cwd):
    """Spans between matching quotes; relative paths only in the @"…" mention form."""
    for i, ch in enumerate(line):
        if ch not in QUOTES:
            continue
        j = line.find(ch, i + 1)
        if j < 0 or j - i > MAX_SPAN:
            continue
        p = _resolve(line[i + 1:j], cwd, relative_ok=i > 0 and line[i - 1] == "@")
        if p:
            yield p


def _bare(line, cwd):
    """Unquoted @path / absolute / ~ / file:// spans, which may contain spaces: from each start,
    try every end at a whitespace boundary, longest first."""
    n = len(line)
    for i, ch in enumerate(line):
        if i and not (line[i - 1].isspace() or line[i - 1] in LEAD):
            continue
        if ch == "@":
            p0, rel = i + 1, True
        elif ch in "/~" or line.startswith("file://", i):
            p0, rel = i, False
        else:
            continue
        if p0 >= n or line[p0] in QUOTES or line[p0].isspace():
            continue
        lim = min(n, p0 + MAX_SPAN)
        ends = [j for j in range(p0 + 1, lim) if line[j].isspace()] + ([n] if n <= lim else [])
        for j in reversed(ends):
            cand = line[p0:j].rstrip(TRAIL)
            if cand.lower().endswith(MEDIA):
                p = _resolve(cand, cwd, rel)
                if p:
                    yield p
                    break


def find_paths(prompt, cwd):
    out = {}
    for line in prompt[:MAX_PROMPT].splitlines():
        if "." not in line:
            continue
        for p in list(_quoted(line, cwd)) + list(_bare(line, cwd)):
            real = os.path.realpath(p)
            if real == DATA or real.startswith(DATA + os.sep):
                continue                    # the pipeline's own renders / assets, not an upload
            out.setdefault(real, p)
            if len(out) >= MAX_FILES:
                return list(out.values())
    return list(out.values())


def _hint(p):
    name = os.path.basename(p)
    ext = os.path.splitext(name)[1].lower()
    if ext in IMAGE:
        return "image", "image"
    if ext in AUDIO:
        return "audio", "sfx (from the name)" if SFX_HINT.search(name) else "music"
    sample, source = bool(SAMPLE_HINT.search(name)), bool(SOURCE_HINT.search(name))
    if sample != source:
        return "video", ("sample" if sample else "source") + " (from the name)"
    return "video", "sample or source: decide from the message (short vertical reel = sample)"


def _size(p):
    try:
        b = os.path.getsize(p)
    except OSError:
        return "?"
    return f"{b / 1e6:.1f} MB" if b >= 1e5 else f"{b / 1e3:.0f} kB"


def open_jobs(limit=8):
    """(id, status) of unfinished jobs, newest first; old and new job.json versions."""
    try:
        dirs = sorted((e for e in os.scandir(os.path.join(DATA, "jobs")) if e.is_dir()),
                      key=lambda e: e.stat().st_mtime, reverse=True)[:30]
    except OSError:
        return []
    out = []
    for e in dirs:
        try:
            with open(os.path.join(e.path, "job.json"), "rb") as f:
                j = json.loads(f.read().decode("utf-8", "replace"))
        except (OSError, ValueError):
            continue
        st = str(j.get("status") or "?") if isinstance(j, dict) else "?"
        if not isinstance(j, dict) or j.get("terminal") or st in TERMINAL:
            continue
        out.append((e.name, st[:60]))
        if len(out) >= limit:
            break
    return out


def context(paths):
    lines = [f"[reel production line] {len(paths)} media file(s) in this message (paths shell-quoted):"]
    for p in paths:
        kind, hint = _hint(p)
        lines.append(f"- {shlex.quote(p)} ({kind}, {_size(p)}; role hint: {hint})")
    jobs = open_jobs()
    lines.append("Open reel jobs: " + (", ".join(f"{i} ({s})" for i, s in jobs) if jobs else "none") +
                 " (`reel-pipeline/reel job list` lists them all).")
    if all(_hint(p)[0] == "image" for p in paths):
        lines.append("Only images: follow the steps below only if they are meant for reels (e.g. a logo or a "
                     "picture to use); otherwise ignore this note.")
    lines += [
        "Follow the reel-production skill:",
        "1. `reel-pipeline/reel check` must report ready (exit 0); otherwise run `reel-pipeline/setup.sh` and check "
        "again.",
        "2. Adding to an open job (the user says so, or it is waiting for exactly this input): "
        "`reel-pipeline/reel job add <id> --source FILE …` with explicit roles. Otherwise a new job: "
        "`reel-pipeline/reel job new --sample FILE --source FILE [--music F] [--sfx F] [--image F] [--count N] "
        "[--duration S] [--lang xx]`. Give every file an explicit role when the user's message or the files make it "
        "clear (a reel to imitate = sample, long-form footage to cut = source, a track = music, effects = sfx, a "
        "logo/picture = image). Pass files bare only when nothing indicates a role: the CLI then guesses (short "
        "vertical video = sample, longer = source, audio = music), records the guess as a suggestion, and you tell "
        "the user what was guessed.",
        "   The user's count, duration, language and instructions go into the request. For instructions (notes), "
        "write a request.json with the Write tool — {\"schema_version\": 1, \"assets\": [{\"path\": …, \"role\": …}], "
        "\"request\": {\"count\": …, \"duration_seconds\": …, \"lang\": …, \"notes\": …}} — and pass "
        "`--request FILE` instead of quoting free text into a shell command. The default execution is CPU-safe "
        "(--encoder auto); add `--encoder nvenc --require-gpu` only when the user asks for a GPU-only run.",
        "3. Run the saved workflow `reel-job` with args {\"job\": \"<job id>\"}.",
        "4. The job's status JSON is the result (`reel-pipeline/reel job status <id>`): deliver.files are the "
        "accepted reels from the batch manifest; report deliver.review, deliver.unresolved findings, counts and "
        "shortfall honestly. Attach the accepted files with SendUserFile only if that tool is available; otherwise "
        "list their paths. If the job waits for a sample or footage, say what to upload and stop. If the workflow "
        "returns blocked (e.g. its step budget ran out), run it again with the same job id.",
    ]
    return "\n".join(lines)


def main():
    raw = sys.stdin.buffer.read()
    data = json.loads(raw.decode("utf-8", "replace")) if raw.strip() else {}
    prompt = data.get("prompt") if isinstance(data, dict) else None
    if not isinstance(prompt, str) or not prompt:
        return
    cwd = data.get("cwd") if isinstance(data.get("cwd"), str) and data.get("cwd") else os.getcwd()
    paths = find_paths(prompt, cwd)
    if not paths:
        return
    if all(_hint(p)[0] == "image" for p in paths) and not REEL_WORDS.search(prompt) and not open_jobs(1):
        return                              # a screenshot for something else
    sys.stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                                        "additionalContext": context(paths)}}) + "\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException:                   # never block or break the user's prompt
        pass
    sys.exit(0)
