# TRIBE v2 — Kahneman Framing RCT

## Open in Colab (use this link)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akifnu/DSprojects/blob/main/tribev2/notebooks/Framing_RCT_NoSetup.ipynb)

**No HF_TOKEN. No secrets. No git clone.**

1. Open the link above in a **new browser tab** (not a saved Drive copy)
2. Runtime → **T4 GPU**
3. Runtime → **Run all** (if runtime restarts once after install, click **Run all** again)

### If you see `cannot import name '_center' from numpy`

The notebook auto-fixes this by pinning `numpy==2.2.6`. After the one-time restart, run all cells again.

### If you still see `HF_TOKEN` or `/content/DSprojects`

You are running an **old saved Colab notebook**. Fix:

1. Close that tab
2. In Google Drive, delete any saved copy named `Kahneman_Framing_RCT`
3. Open the **Framing_RCT_NoSetup.ipynb** link above

## Full RCT dataset (optional, local)

318 scenario pairs in `data/framing_rct/` — see [`data/framing_rct/README.md`](data/framing_rct/README.md).
