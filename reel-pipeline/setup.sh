#!/usr/bin/env bash
# Installs and checks everything the reel production line needs; safe to rerun.
#
#   setup.sh                  install whatever is missing or broken, then `reel check`
#   setup.sh --check          install nothing: report what is missing, then `reel check`
#                             (REELKIT_PREBUILT=1 means the same: an image that has everything)
#   setup.sh --verify         the final check is `reel check --deep`: model files against
#                             $REELKIT_MODELS/CHECKSUMS and every model loaded once
#   setup.sh --gpu            also the GPU profile, requirements-gpu.txt (CUDA build of sherpa-onnx,
#                             onnxruntime-gpu); remembered in .tools/profile for later runs
#   setup.sh --no-check       skip the final `reel check` (an image layer built before the code)
#   setup.sh --session-start  report, never fail: also implied when it runs as a Claude Code hook
#                             (CLAUDE_PROJECT_DIR set, stdin not a terminal)
#
# What it installs:
#   ffmpeg 9.0 static build (BtbN: libx264/x265, libvmaf, zscale, NVENC) → .tools/ffmpeg, also the
#     system ffmpeg when run as root. The default URL is BtbN's moving "latest" 9.0 build; for a
#     reproducible deployment set REELKIT_FFMPEG_URL to a dated autobuild and REELKIT_FFMPEG_SHA256.
#   Python packages pinned in requirements.txt (and requirements-gpu.txt with --gpu)
#   models → $REELKIT_MODELS (default .models; the runtime reads the same variable), GitHub releases:
#     Parakeet TDT 0.6B v2 int8 (English ASR), Whisper large-v3-turbo int8 (other languages),
#     Silero VAD, YuNet faces
#
# Every download goes to a temp file or directory beside its destination and is verified before
# it is renamed into place: archives must extract completely (the xz / bzip2 checks and tar's end
# marker catch a truncated download), the files the runtime loads must be non-empty, and sha-256
# sums must match $REELKIT_MODELS/CHECKSUMS and .tools/CHECKSUMS where one is recorded (a first
# download records it). An interrupted run leaves nothing half-installed, and the next run starts
# that piece again. When everything is in place nothing is downloaded and the run takes seconds:
# the SessionStart hook runs it in every cloud session.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$HERE/.tools"
MODELS="${REELKIT_MODELS:-$HERE/.models}"
case "$MODELS" in /*) ;; *) MODELS="$PWD/$MODELS" ;; esac
export REELKIT_MODELS="$MODELS"
PY=python3                                  # the interpreter `reel` runs
FFMPEG_SERIES="${REELKIT_FFMPEG_SERIES:-9.0}"
FFMPEG_URL="${REELKIT_FFMPEG_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n${FFMPEG_SERIES}-latest-linux64-gpl-${FFMPEG_SERIES}.tar.xz}"
SHERPA="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
YUNET_URL="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
PARAKEET=sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
WHISPER=sherpa-onnx-whisper-turbo

check_only=0; verify=0; gpu=0; final_check=1; session=0
if [ "${REELKIT_PREBUILT:-0}" = 1 ]; then check_only=1; fi
for arg in "$@"; do
  case "$arg" in
    --check) check_only=1 ;;
    --verify) verify=1 ;;
    --gpu) gpu=1 ;;
    --no-check) final_check=0 ;;
    --session-start) session=1 ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
    *) echo "setup.sh: unknown option $arg (see --help)" >&2; exit 2 ;;
  esac
done
if [ -n "${CLAUDE_PROJECT_DIR:-}" ] && [ ! -t 0 ]; then session=1; fi
profile=cpu
if [ "$gpu" = 1 ] || [ "$(cat "$TOOLS/profile" 2>/dev/null || true)" = gpu ]; then profile=gpu; fi

say() { echo "[setup] $*" >&2; }

# ------------------------------------------------------------------ what is in place

ffmpeg_ok() {   # the static build of the wanted series (or the build REELKIT_FFMPEG_URL names) runs
  local out
  [ -x "$TOOLS/ffmpeg/bin/ffprobe" ] || return 1
  out="$("$TOOLS/ffmpeg/bin/ffmpeg" -hide_banner -version 2>/dev/null)" || return 1
  case "${out%%$'\n'*}" in "ffmpeg version n${FFMPEG_SERIES}"*) return 0 ;; esac
  [ -n "${REELKIT_FFMPEG_URL:-}" ]          # a custom build: `reel check` still requires ffmpeg ≥ 9
}

py_ok() {   # every pin of requirements.txt installed (the GPU profile's wheel stands in for onnxruntime)
  "$PY" - "$HERE/requirements.txt" "$1" <<'PY'
import importlib.util, pathlib, re, sys
from importlib import metadata
alt = {"onnxruntime": ("onnxruntime", "onnxruntime-gpu")}
bad = []
for line in open(sys.argv[1], encoding="utf-8"):
    m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)", line.split("#", 1)[0].strip())
    if not m:
        continue
    name, want = m.groups()
    have = None
    for d in alt.get(name, (name,)):
        try:
            have = metadata.version(d)
            break
        except metadata.PackageNotFoundError:
            pass
    if have is None or have.split("+")[0] != want:
        bad.append(f"{name}=={want} (installed: {have or 'none'})")
if sys.argv[2] == "gpu":
    spec = importlib.util.find_spec("sherpa_onnx")
    libs = [p for d in (spec.submodule_search_locations or []) for p in pathlib.Path(d).rglob("*providers_cuda*")] \
        if spec else []
    if not libs:
        bad.append("a CUDA build of sherpa-onnx (no CUDA provider library in sherpa_onnx/)")
    try:
        metadata.version("onnxruntime-gpu")
    except metadata.PackageNotFoundError:
        bad.append("onnxruntime-gpu")
if bad:
    print("Python packages missing or different: " + "; ".join(bad), file=sys.stderr)
    sys.exit(1)
PY
}

model_files() {   # the files the runtime loads (asr.py, reframe.py), relative to $MODELS; no spaces
  case "$1" in
    parakeet) echo "$PARAKEET/encoder.int8.onnx $PARAKEET/decoder.int8.onnx $PARAKEET/joiner.int8.onnx $PARAKEET/tokens.txt" ;;
    whisper) echo "$WHISPER/turbo-encoder.int8.onnx $WHISPER/turbo-decoder.int8.onnx $WHISPER/turbo-tokens.txt" ;;
    vad) echo "silero_vad.onnx" ;;
    yunet) echo "yunet/yunet.onnx" ;;
  esac
}

model_ok() {
  local f
  for f in $(model_files "$1"); do
    [ -s "$MODELS/$f" ] || return 1
  done
}

# ------------------------------------------------------------------ verified downloads

fetch() {   # url dest: the whole file or a failure
  curl -fsSL --retry 5 --retry-delay 3 --retry-connrefused --connect-timeout 30 -o "$2" "$1"
}

record_sum() {   # sums key sha: one sha256sum-format line per key
  local sums="$1" key="$2" sha="$3" tmp
  tmp="$(mktemp "$sums.XXXXXX")"
  if [ -f "$sums" ]; then
    awk -v k="$key" '!($2 == k || $2 == "*" k)' "$sums" >"$tmp"
  fi
  echo "$sha  $key" >>"$tmp"
  chmod 644 "$tmp"
  mv -f "$tmp" "$sums"
}

sum_check() {   # file key sums [expected]: the sha-256 must equal `expected`, else the sum recorded
                # for key (a moving "latest" artifact is re-recorded with a warning); then recorded
  local file="$1" key="$2" sums="$3" want="${4:-}" have rec=""
  have="$(sha256sum "$file" | cut -d' ' -f1)"
  if [ -n "$want" ]; then
    if [ "$have" != "${want,,}" ]; then
      say "$key: sha-256 $have, expected $want"
      return 1
    fi
  elif [ -f "$sums" ]; then
    rec="$(awk -v k="$key" '$2 == k || $2 == "*" k {s = $1} END {print s}' "$sums")"
    if [ -n "$rec" ] && [ "$rec" != "$have" ]; then
      case "$key" in
        *latest*) say "$key: the moving 'latest' build changed (sha-256 $have, recorded $rec)" ;;
        *) say "$key: sha-256 $have does not match the one recorded in $sums ($rec); if upstream" \
               "really changed, delete that line and run setup.sh again"
           return 1 ;;
      esac
    fi
  fi
  record_sum "$sums" "$key" "$have"
}

publish_dir() {   # src dest: dest becomes src by rename (a previous dest is moved aside first)
  local src="$1" dest="$2" old=""
  if [ -e "$dest" ] || [ -L "$dest" ]; then
    old="$(dirname "$dest")/.tmp-old-$(basename "$dest").$$"
    mv -T "$dest" "$old"
  fi
  mv -T "$src" "$dest"
  if [ -n "$old" ]; then rm -rf "$old"; fi
}

install_ffmpeg() {
  local name tmp top="" dir d sha
  name="$(basename "$FFMPEG_URL")"
  tmp="$(mktemp -d "$TOOLS/.tmp-ffmpeg.XXXXXX")"
  say "downloading ffmpeg ($FFMPEG_URL)"
  fetch "$FFMPEG_URL" "$tmp/$name"
  sum_check "$tmp/$name" "$name" "$TOOLS/CHECKSUMS" "${REELKIT_FFMPEG_SHA256:-}"
  mkdir "$tmp/x"
  tar -xJf "$tmp/$name" -C "$tmp/x"         # completes only for an intact archive
  for d in "$tmp/x"/*/; do top="${d%/}"; break; done
  if [ -z "$top" ] || [ ! -x "$top/bin/ffmpeg" ] || [ ! -x "$top/bin/ffprobe" ]; then
    say "$name holds no bin/ffmpeg + bin/ffprobe"
    return 1
  fi
  if ! "$top/bin/ffmpeg" -hide_banner -version >/dev/null 2>&1; then
    say "the downloaded ffmpeg does not run on this host"
    return 1
  fi
  sha="$(sha256sum "$tmp/$name" | cut -c1-12)"
  dir="ffmpeg-n${FFMPEG_SERIES}-$sha"                # one directory per build: renders keep theirs
  rm -rf "${TOOLS:?}/$dir"
  mv -T "$top" "$TOOLS/$dir"
  rm -f "$TOOLS/.ffmpeg.link"
  ln -s "$dir" "$TOOLS/.ffmpeg.link"
  if [ -d "$TOOLS/ffmpeg" ] && [ ! -L "$TOOLS/ffmpeg" ]; then mv -T "$TOOLS/ffmpeg" "$TOOLS/.tmp-old-ffmpeg.$$"; fi
  mv -Tf "$TOOLS/.ffmpeg.link" "$TOOLS/ffmpeg"     # the switch is one rename
  rm -rf "$tmp" "$TOOLS/.tmp-old-ffmpeg.$$"
  for d in "$TOOLS"/ffmpeg-n*; do
    if [ "$d" != "$TOOLS/$dir" ]; then rm -rf "$d"; fi
  done
}

install_archive() {   # name dir: $SHERPA/<dir>.tar.bz2 → $MODELS/<dir>, only the files the runtime loads
  local name="$1" dir="$2" tmp f
  tmp="$(mktemp -d "$MODELS/.tmp-$dir.XXXXXX")"
  say "downloading $dir"
  fetch "$SHERPA/$dir.tar.bz2" "$tmp/archive.tar.bz2"
  tar -xjf "$tmp/archive.tar.bz2" -C "$tmp" --wildcards "$dir/*.int8.onnx" "$dir/*tokens.txt"
  rm -f "$tmp/archive.tar.bz2"
  for f in $(model_files "$name"); do
    if [ ! -s "$tmp/$f" ]; then
      say "$dir.tar.bz2: $f is missing or empty"
      return 1
    fi
    sum_check "$tmp/$f" "$f" "$MODELS/CHECKSUMS"
  done
  chmod -R a+rX "$tmp/$dir"
  publish_dir "$tmp/$dir" "$MODELS/$dir"
  rm -rf "$tmp"
}

install_file() {   # url rel: one model file → $MODELS/<rel>
  local url="$1" rel="$2" tmp
  mkdir -p "$(dirname "$MODELS/$rel")"
  tmp="$(mktemp "$(dirname "$MODELS/$rel")/.tmp-$(basename "$rel").XXXXXX")"
  say "downloading $rel"
  fetch "$url" "$tmp"
  if [ ! -s "$tmp" ]; then
    say "$rel: the download is empty"
    return 1
  fi
  sum_check "$tmp" "$rel" "$MODELS/CHECKSUMS"
  chmod 644 "$tmp"
  mv -f "$tmp" "$MODELS/$rel"
}

pip_flags() {   # pip may install into a system Python marked externally managed (as in the image)
  if "$PY" -c 'import os, sys, sysconfig
sys.exit(0 if sys.prefix == sys.base_prefix and os.path.exists(os.path.join(sysconfig.get_paths()["stdlib"], "EXTERNALLY-MANAGED")) else 1)'; then
    echo --break-system-packages
  fi
  if [ "$(id -u)" = 0 ]; then echo --root-user-action=ignore; fi
}

install_python() {
  local flags hv
  read -r -a flags <<<"$(pip_flags | tr '\n' ' ')"
  say "installing the pinned Python packages (requirements.txt)"
  "$PY" -m pip install -q --disable-pip-version-check ${flags[@]+"${flags[@]}"} -r "$HERE/requirements.txt"
  # rapidocr pulls in the GUI opencv-python; both install cv2, the headless one must be what is on disk
  hv="$(sed -n 's/^opencv-python-headless==\([^ #]*\).*/\1/p' "$HERE/requirements.txt")"
  if [ -n "$hv" ] && "$PY" -c 'import importlib.metadata as m; m.version("opencv-python")' 2>/dev/null; then
    "$PY" -m pip install -q --disable-pip-version-check ${flags[@]+"${flags[@]}"} --no-deps --force-reinstall \
      "opencv-python-headless==$hv"
  fi
}

install_gpu_python() {
  local flags
  read -r -a flags <<<"$(pip_flags | tr '\n' ' ')"
  say "installing the GPU profile (requirements-gpu.txt)"
  if "$PY" -c 'import importlib.metadata as m; m.version("onnxruntime")' 2>/dev/null; then
    "$PY" -m pip uninstall -y -q ${flags[@]+"${flags[@]}"} onnxruntime   # onnxruntime-gpu installs the same module
  fi
  "$PY" -m pip install -q --disable-pip-version-check ${flags[@]+"${flags[@]}"} --no-deps --force-reinstall \
    --only-binary=:all: -r "$HERE/requirements-gpu.txt"
  if ! py_ok gpu; then
    say "the GPU profile is incomplete: requirements-gpu.txt needs the CUDA build of sherpa-onnx for this host"
    return 1
  fi
  echo gpu >"$TOOLS/profile"
}

install_all() {   # runs in a subshell with errexit: the first failing step stops it
  mkdir -p "$TOOLS" "$MODELS"
  exec 9>"$TOOLS/.setup.lock"
  if command -v flock >/dev/null; then flock -w 3600 9; fi        # one setup at a time
  find "$TOOLS" "$MODELS" -maxdepth 2 -name '.tmp-*' -exec rm -rf {} + 2>/dev/null || true
  if ! ffmpeg_ok; then install_ffmpeg; fi
  local v
  v="$("$TOOLS/ffmpeg/bin/ffmpeg" -hide_banner -version)"
  if [ "$(cat "$TOOLS/ffmpeg.version" 2>/dev/null || true)" != "${v%%$'\n'*}" ]; then
    echo "${v%%$'\n'*}" >"$TOOLS/ffmpeg.version"
  fi
  if [ "$(id -u)" = 0 ] && [ -w /usr/local/bin ]; then          # plain `ffmpeg` = this build everywhere
    for t in ffmpeg ffprobe; do
      if [ "$(readlink "/usr/local/bin/$t" 2>/dev/null || true)" != "$TOOLS/ffmpeg/bin/$t" ]; then
        ln -sfn "$TOOLS/ffmpeg/bin/$t" "/usr/local/bin/$t"
      fi
    done
  fi
  local changed=0
  if ! py_ok cpu 2>/dev/null; then install_python; changed=1; fi
  # a CPU install may have put the CPU onnxruntime back over the GPU wheel: re-apply the profile
  if [ "$profile" = gpu ] && { [ "$changed" = 1 ] || ! py_ok gpu 2>/dev/null; }; then install_gpu_python; fi
  if ! model_ok parakeet; then install_archive parakeet "$PARAKEET"; fi
  if ! model_ok whisper; then install_archive whisper "$WHISPER"; fi
  if ! model_ok vad; then install_file "$SHERPA/silero_vad.onnx" silero_vad.onnx; fi
  if ! model_ok yunet; then install_file "$YUNET_URL" yunet/yunet.onnx; fi
}

missing() {   # what an install run would fetch, one per line
  if ! ffmpeg_ok; then echo "ffmpeg $FFMPEG_SERIES (.tools/ffmpeg)"; fi
  local m extra=""
  if [ "$profile" = gpu ]; then extra=" + requirements-gpu.txt"; fi
  if ! py_ok "$profile" 2>/dev/null; then echo "Python packages (requirements.txt$extra)"; fi
  for m in parakeet whisper vad yunet; do
    if ! model_ok "$m"; then echo "model $m ($MODELS)"; fi
  done
}

# ------------------------------------------------------------------ run

installed=1
if [ "$check_only" = 0 ]; then
  set +e
  ( set -eE; trap 'say "failed (line $LINENO): $BASH_COMMAND"' ERR; install_all )
  [ $? = 0 ] || installed=0
  set -e
fi
gaps="$(missing)"
if [ -n "$gaps" ]; then
  while IFS= read -r g; do say "missing: $g"; done <<<"$gaps"
fi

ready=1
report=""
if [ "$final_check" = 1 ]; then
  deep=()
  if [ "$verify" = 1 ]; then deep=(--deep); fi
  if report="$("$HERE/reel" check ${deep[@]+"${deep[@]}"} 2>&1 >/dev/null)"; then ready=1; else ready=0; fi
  echo "$report" >&2
elif [ -n "$gaps" ]; then
  ready=0
fi

if [ "$installed" = 1 ] && [ "$ready" = 1 ]; then
  ver="$(cut -d' ' -f1-3 "$TOOLS/ffmpeg.version" 2>/dev/null || true)"
  echo "reel-pipeline ready${ver:+ ($ver)}"
  exit 0
fi
if [ "$installed" = 0 ]; then
  echo "reel-pipeline setup did not finish (see the [setup] lines); run reel-pipeline/setup.sh again"
fi
if [ -n "$report" ]; then
  printf '%s\n' "$report" | grep -E '^reel check:|^ +FAIL' || true
fi
if [ "$session" = 1 ]; then exit 0; fi       # a session starts anyway; the agent sees the report
exit 1
