# TRIBE v2 — Kahneman Framing RCT (text only)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akifnu/DSprojects/blob/main/tribev2/notebooks/Framing_RCT_NoSetup.ipynb)

## Massive text RCT

| | |
|--|--|
| Scenario pairs | **318** gain/loss texts |
| Unique texts | **636** |
| Subjects | **200** |
| Trial assignments | **63,600** |
| Input modality | **Text only** (`text_path` → TRIBE v2) |

## Colab

1. Open [**Framing_RCT_NoSetup.ipynb**](https://colab.research.google.com/github/akifnu/DSprojects/blob/main/tribev2/notebooks/Framing_RCT_NoSetup.ipynb)
2. Runtime → **A100 GPU** (text inference needs ~40 GB VRAM for LLaMA 3.2)
3. Accept [LLaMA 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-3B) on Hugging Face
4. **Run all** — paste your Hugging Face read token when prompted (or set Colab secret `HF_TOKEN`)
5. Increase `MAX_SCENARIOS` in the notebook toward **318** for the full study

The notebook downloads `scenarios.json` (all 318 text pairs) from GitHub — no audio, no gTTS.

## Local

```bash
python scripts/generate_rct_dataset.py   # rebuild 318-pair dataset
export HF_TOKEN=<token>
python scripts/run_framing_rct.py --preload-llama --max-scenarios 12
```

Dataset: [`data/framing_rct/`](data/framing_rct/README.md)
