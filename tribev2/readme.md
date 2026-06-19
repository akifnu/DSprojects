# TRIBE v2 — Kahneman Framing RCT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akifnu/DSprojects/blob/main/tribev2/notebooks/Framing_RCT_NoSetup.ipynb)

## Colab quick demo (~10–20 min on A100 after first install)

1. Open [**Framing_RCT_NoSetup.ipynb**](https://colab.research.google.com/github/akifnu/DSprojects/blob/main/tribev2/notebooks/Framing_RCT_NoSetup.ipynb)
2. **Runtime → GPU** (A100 40 GB recommended)
3. Colab secret **`HF_TOKEN`** (Hugging Face read token with LLaMA access)
4. **Runtime → Run all** (auto-restarts once on first install)

The quick demo runs **3 classic Kahneman pairs (6 texts)** with text-only inference (no audio/gTTS).

## Full RCT dataset (local)

| | |
|--|--|
| Scenario pairs | **318** |
| Trial assignments | **63,600** |

```bash
cd tribev2
pip install -r requirements-ci.txt && pip install -e .
pip install -r requirements-gpu.txt
export HF_TOKEN=<your-read-token>
python scripts/run_framing_rct.py --max-scenarios 12 --preload-llama
```

Dataset: [`data/framing_rct/`](data/framing_rct/README.md)

## References

- [facebook/tribev2](https://huggingface.co/facebook/tribev2)
- Tversky & Kahneman (1981), *Science*
