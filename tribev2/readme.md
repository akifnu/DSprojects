# TRIBE v2 — Kahneman Framing RCT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akifnu/DSprojects/blob/main/tribev2/notebooks/Kahneman_Framing_RCT.ipynb)

## Colab (ready to go)

**No Hugging Face token. No API keys. No repo clone.**

1. Open the notebook link above
2. **Runtime → T4 GPU**
3. **Runtime → Run all**

The notebook installs TRIBE v2, synthesizes 12 gain/loss framing texts as speech, runs inference, and prints paired statistics.

## Full RCT dataset (optional, local)

The repo also contains a **318-pair / 63,600-trial** text RCT for offline research:

```bash
cd tribev2
pip install -r requirements-ci.txt && pip install -e .
python scripts/generate_rct_dataset.py
```

See [`data/framing_rct/`](data/framing_rct/README.md).

## References

- [facebook/tribev2](https://huggingface.co/facebook/tribev2)
- Tversky & Kahneman (1981), *Science*
