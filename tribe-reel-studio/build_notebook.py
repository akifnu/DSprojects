"""Generates TRIBE_Reel_Studio_Colab.ipynb with app.py and brain_viz.py embedded."""
import json
from pathlib import Path

here = Path(__file__).parent


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


cells = [
    md("""
# 🧠 TRIBE v2 Reel Studio (Colab launcher)

Upload a reel → see the **predicted brain response** (Meta FAIR's TRIBE v2): a side-by-side brain movie, an interactive 3D cortex,
intensity over time per brain system, and the peak moments.

**One-time setup**
1. **Runtime → Change runtime type → GPU** (a T4 works; an A100/L4 is faster).
2. TRIBE v2 uses **LLaMA 3.2-3B**, which is gated: on Hugging Face, open https://huggingface.co/meta-llama/Llama-3.2-3B and click
   *Agree / request access* (usually approved within minutes).
3. Create a token at https://huggingface.co/settings/tokens (type *Read*) and add it in Colab: 🔑 **Secrets** (left sidebar) →
   name `HF_TOKEN`, paste the token, and enable *Notebook access*.
4. **Runtime → Run all**. The last cell prints a public `https://….gradio.live` link. Open it and upload your reel.

The first analysis downloads ~20 GB of models (TRIBE, V-JEPA2, LLaMA, Wav2Vec-BERT, WhisperX), which takes several minutes.
After that, a 30–60 s reel takes about 2–5 minutes on a T4.

<small>Predictions describe an *average* viewer's cortex, not real measurements. TRIBE v2 is CC BY-NC 4.0 (non-commercial only).</small>
"""),
    code("""
#@title 1 · Check GPU + install (≈3–5 min)
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "⚠️ No GPU: switch the runtime to GPU first!"
!pip -q install uv
!uv pip install --system -q "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git" "transformers>=4.46" "faster-whisper>=1.1" "ctranslate2>=4.5" gradio plotly
"""),
    code("""
#@title 2 · Hugging Face login (needed for gated LLaMA 3.2)
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception as e:
    print("Colab secret HF_TOKEN not found:", e)
if not os.environ.get("HF_TOKEN"):
    from getpass import getpass
    os.environ["HF_TOKEN"] = getpass("Paste your Hugging Face token: ")
from huggingface_hub import login, HfApi
login(token=os.environ["HF_TOKEN"])
try:
    HfApi().model_info("meta-llama/Llama-3.2-3B")
    print("✅ LLaMA 3.2-3B access OK")
except Exception as e:
    print("❌ No access to meta-llama/Llama-3.2-3B yet. Request it on huggingface.co first.\\n", e)
"""),
    code("%%writefile brain_viz.py\n" + (here / "brain_viz.py").read_text()),
    code("%%writefile app.py\n" + (here / "app.py").read_text()),
    code("""
#@title 3 · Launch the app → click the gradio.live link that appears below
!python app.py --share --preload
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
out = here / "TRIBE_Reel_Studio_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
