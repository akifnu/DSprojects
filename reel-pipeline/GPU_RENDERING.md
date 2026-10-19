# Faster CPU exports and NVIDIA cloud rendering

The editor can stay on its current host. Rendering must run on the GPU machine;
setting an environment variable on a CPU-only Claude Code container does not
attach a remote GPU. Transfer the timeline and its referenced media to a GPU
worker, render there, and copy the MP4 back. A queue/remote submission service is
not included in this change.

## Existing host

`scripts/render.sh` renders 720p with x264 `veryfast` (CRF 19) by default;
`fast_render.py` called directly keeps the project canvas and `fast`. For
smaller files at the cost of a few seconds:

```bash
cd reel-pipeline
CPU_PRESET=fast scripts/render.sh jobs/job1/out
```

Each export logs preparation, audio mixing, loudness analysis, video rendering,
and total elapsed time to stderr. `RENDER_THREADS=4` bounds FFmpeg decoder,
complex-filter, and encoder thread settings; leave it unset for FFmpeg defaults.
The mix/analysis/normalization stages preserve two-pass loudness normalization.

## Choose a GPU worker

Use an NVIDIA GPU that actually has an H.264 NVENC encoder, such as an L4,
A10, or RTX 4090. Check the exact model in NVIDIA's
[encode/decode support matrix](https://developer.nvidia.com/video-encode-decode-support-matrix).
GPU memory size alone does not establish video encoding support.

For example, deploy a Runpod GPU Pod with an official Ubuntu/PyTorch template,
SSH access, and persistent storage mounted at `/workspace`. Use the Pod's
Connect instructions to SSH and transfer files. See Runpod's
[SSH and file-transfer guide](https://www.runpod.io/blog/how-to-connect-to-your-pod-with-ssh-and-move-files-with-scp-and-filezilla).
Keep adequate CPU resources too: this renderer still decodes, crops, scales,
composites captions, and processes audio on the CPU. NVENC accelerates the final
video encoder; the filter graph is not a CUDA graph.

For a container-based Pod, set this environment variable **before launching**:

```text
NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

The `video` capability exposes NVIDIA's video-codec driver libraries. Changing
it inside an already-running container does not mount missing host libraries.
See the [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).
The host needs compatible NVIDIA drivers. A CUDA toolkit or PyTorch installation
alone is insufficient.

## Run on the worker without the editor server

On the existing editor host, export the current timeline:

```bash
curl -fsS http://localhost:7777/api/project -o reel-pipeline/workspace/project.json
```

Clone the branch containing this change on the worker. Copy
`reel-pipeline/workspace/project.json` and `reel-pipeline/workspace/media/` from
the editor host to the same relative locations in the worker checkout. Copy
`workspace/fonts/` to reuse fonts offline. If the JSON uses `/library/` media,
also copy the referenced `.fablecut/library/` assets; otherwise that directory
and FableCut itself are unnecessary. Media names must match the timeline's
`/media/` references. Raw videos and workspace assets are gitignored, so cloning
the repository alone will not transfer them.

In a Debian/Ubuntu worker terminal with root privileges:

```bash
apt-get update
apt-get install -y ffmpeg python3 python3-pil fonts-liberation ca-certificates
nvidia-smi
ffmpeg -hide_banner -encoders | grep h264_nvenc
cd /workspace/DSprojects/reel-pipeline
mkdir -p jobs/job1/out
python3 scripts/fast_render.py workspace/project.json jobs/job1/out/gpu.mp4 \
  --encoder nvenc --threads 4
scripts/qc.sh jobs/job1/out/gpu.mp4
```

Adjust the checkout path to where you cloned it. The renderer performs a real
one-frame NVENC encode before preparing the job. A listing of `h264_nvenc` alone
does not verify driver compatibility. `--encoder nvenc` exits with an actionable
error if the preflight fails. `--encoder auto` instead selects CPU encoding when
that preflight fails; it does not retry arbitrary failures halfway through an
export. `--encoder cpu` skips the GPU check entirely.

NVENC defaults to `--nvenc-preset p4 --nvenc-cq 19`, with the existing 25 Mbps
max rate. NVENC CQ 19 and x264 CRF 19 are different controls and are not promises
of identical quality or size. Compare representative reels before switching.

If the editor and renderer share the same GPU host and filesystem, the existing
wrapper can use it directly:

```bash
VIDEO_ENCODER=auto RENDER_THREADS=4 scripts/render.sh jobs/job1/out
```

## Optional Docker worker

On a GPU host with Docker, NVIDIA drivers, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured, run these commands from the repository root:

```bash
docker build -f reel-pipeline/Dockerfile.render -t reel-render reel-pipeline
mkdir -p reel-pipeline/jobs/job1/out
docker run --rm --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v "$PWD/reel-pipeline/workspace:/app/reel-pipeline/workspace" \
  -v "$PWD/reel-pipeline/jobs:/jobs" \
  reel-render workspace/project.json /jobs/job1/out/gpu.mp4 \
  --encoder nvenc --threads 4
```

If needed, also mount `.fablecut/library` at
`/app/reel-pipeline/.fablecut/library`. The image is a render-only CLI; it does
not run the editor or download the transcription model. Building the image
does not require a GPU, but running NVENC does.

## Validate

```bash
python3 -m unittest discover -s reel-pipeline/tests -v
```

The tests exercise real CPU exports, caption rendering, audio gaps, shared-input
probing, and crop-before-scale pixel comparisons. GPU selection/fallback is
mocked in tests. A real GPU export and QC on your chosen worker are still
required to validate that host's NVIDIA driver and performance.
