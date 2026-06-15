# TRIBE v2 Capability Testing

Evaluate [facebook/tribev2](https://huggingface.co/facebook/tribev2) with repeatable benchmarks and a **large-scale Kahneman framing RCT** (text only).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akifnu/DSprojects/blob/main/tribev2/notebooks/Kahneman_Framing_RCT.ipynb)

## Kahneman framing RCT (v1)

| | |
|--|--|
| Scenario pairs | **318** gain/loss texts |
| Unique stimuli | **636** |
| Subjects | **200** |
| Trial assignments | **63,600** |
| Modality | **Text only** |

### Run in Google Colab (recommended)

1. Open the notebook: [`notebooks/Kahneman_Framing_RCT.ipynb`](notebooks/Kahneman_Framing_RCT.ipynb)
2. Set runtime to **A100 GPU**
3. Add Colab secret `HF_TOKEN` ([LLaMA 3.2 access](https://huggingface.co/meta-llama/Llama-3.2-3B))
4. Run all cells

The notebook clones this repo, builds the RCT dataset, and runs TRIBE inference. Start with `--max-scenarios 12`; increase for the full study.

### Regenerate dataset locally

```bash
cd tribev2
pip install -r requirements-ci.txt && pip install -e .
python scripts/generate_rct_dataset.py
```

### Run inference locally

```bash
pip install -r requirements.txt
export HF_TOKEN=<token>
python scripts/run_framing_rct.py --preload-llama --max-scenarios 12
```

Dataset files: [`data/framing_rct/`](data/framing_rct/README.md)

## Project layout

```
tribev2/
├── notebooks/Kahneman_Framing_RCT.ipynb   # Colab entry point
├── data/framing_rct/                      # RCT dataset (318 pairs)
├── scripts/
│   ├── generate_rct_dataset.py
│   └── run_framing_rct.py
└── src/tribe_capabilities/
```

## References

- [TRIBE v2 weights](https://huggingface.co/facebook/tribev2)
- [Official repo](https://github.com/facebookresearch/TRIBEv2)
- Tversky & Kahneman (1981), *Science*
