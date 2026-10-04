---
title: TRIBE v2 Reel Studio
emoji: 🧠
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Upload a reel or photo, see the predicted brain response
# Downloaded once at build time instead of on every wake-up. LLaMA 3.2 is gated, which
# build-time preloading doesn't support, so app.py fetches it in the background at startup.
preload_from_hub:
  - facebook/tribev2 config.yaml,best.ckpt
  - facebook/vjepa2-vitg-fpc64-256
  - facebook/w2v-bert-2.0
  - mobiuslabsgmbh/faster-whisper-large-v3-turbo
---

# TRIBE v2 Reel Studio

Upload a reel or a photo. Meta FAIR's TRIBE v2 predicts how an average human brain responds to it, second by second.
The results are a brain movie, an interactive 3D cortex, intensity per brain system, and peak moments.

Deployed automatically from https://github.com/akifnu/DSprojects (branch `tribe-reel-studio`, folder `tribe-reel-studio/`).
Model license: CC BY-NC 4.0 (non-commercial use only).
