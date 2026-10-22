# Hugging Face Docker Space for Reels Studio. Build context = the folder push_space.py assembles:
#   Dockerfile, README.md, reel-pipeline/{setup.sh,requirements*.txt,reel,reelkit,library}, reels-studio/
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip curl ca-certificates xz-utils bzip2 util-linux fontconfig \
      fonts-liberation fonts-dejavu-core fonts-inter \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg 9 (libass), pinned Python packages and the ASR models: one cached layer, independent of code.
WORKDIR /app/reel-pipeline
COPY reel-pipeline/requirements.txt reel-pipeline/requirements-gpu.txt reel-pipeline/setup.sh ./
ENV REELKIT_MODELS=/app/reel-pipeline/.models
RUN ./setup.sh --no-check && chmod -R a+rX /app
RUN python3 -m pip install --break-system-packages -q starlette==1.7.0 uvicorn==0.54.0 python-multipart==0.0.32

COPY reel-pipeline/reel ./
COPY reel-pipeline/reelkit ./reelkit
COPY reel-pipeline/library ./library
COPY reels-studio /app/reels-studio

# Spaces run the container as uid 1000; only the data dir is writable.
RUN useradd -m -u 1000 user 2>/dev/null || true; mkdir -p /data/studio && chown -R 1000 /data
USER 1000
ENV STUDIO_DATA=/data/studio STUDIO_HOST=0.0.0.0 STUDIO_PORT=7860 REELKIT_DATA=/data/reelkit HOME=/data
WORKDIR /app/reels-studio
EXPOSE 7860
CMD ["python3", "server.py"]
