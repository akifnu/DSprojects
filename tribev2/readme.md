# TRIBE v2 — Kahneman Framing RCT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akifnu/DSprojects/blob/main/tribev2/notebooks/Kahneman_Framing_RCT.ipynb)

## Colab quick demo (~10–20 min on A100 after first install)

1. Open the notebook link above
2. **Runtime → GPU** (A100 40 GB recommended)
3. Colab secret **`HF_TOKEN`** (Hugging Face read token)
4. **Runtime → Run all** (auto-restarts once on first run)

The quick demo runs **3 classic Kahneman pairs (6 texts)** with:

- **Text-only path** — no audio, no gTTS, no 400 MB spaCy download
- **Text-only model load** — skips loading audio/video encoders (less VRAM, faster)
- **Batched inference** — all pending texts in one forward pass when possible
- **Checkpoint resume** — `/content/framing_rct_checkpoint.json`

## Full RCT dataset (optional, local)

```bash
cd tribev2
pip install -r requirements-ci.txt && pip install -e .
pip install -r requirements-gpu.txt
export HF_TOKEN=<your-read-token>
python scripts/run_framing_rct.py --max-scenarios 12 --preload-llama
```

See [`data/framing_rct/`](data/framing_rct/README.md).

## References

- [facebook/tribev2](https://huggingface.co/facebook/tribev2)
- Tversky & Kahneman (1981), *Science*
