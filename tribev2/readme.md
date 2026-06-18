# TRIBE v2 — Kahneman Framing RCT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akifnu/DSprojects/blob/main/tribev2/notebooks/Kahneman_Framing_RCT.ipynb)

## Colab (crash-proof, text-only)

1. Open the notebook link above
2. **Runtime → Change runtime type → GPU** (A100 40 GB recommended)
3. Add Colab secret **`HF_TOKEN`** with your Hugging Face read token (LLaMA 3.2 access), or paste when prompted
4. **Runtime → Run all** — first run installs pinned deps and restarts once

The notebook:

- Uses **direct text → word events** (no audio, no gTTS)
- Pins `numpy`, `torch`, and `transformers` versions known to work on Colab
- **Checkpoints** each prediction to `/content/framing_rct_checkpoint.json` so you can resume after a crash
- Retries transient CUDA/OOM errors up to 3 times per stimulus

Start with `MAX_SCENARIOS = 6`; raise after a clean run.

## Full RCT dataset (optional, local)

The repo also contains a **318-pair / 63,600-trial** text RCT for offline research:

```bash
cd tribev2
pip install -r requirements-ci.txt && pip install -e .
pip install -r requirements-gpu.txt
export HF_TOKEN=<your-read-token>
python scripts/generate_rct_dataset.py
python scripts/run_framing_rct.py --max-scenarios 12 --preload-llama
```

See [`data/framing_rct/`](data/framing_rct/README.md).

## References

- [facebook/tribev2](https://huggingface.co/facebook/tribev2)
- Tversky & Kahneman (1981), *Science*
