# TRIBE v2 Capability Testing

A self-contained subproject for evaluating [facebook/tribev2](https://huggingface.co/facebook/tribev2), Meta's trimodal brain encoding model that predicts fMRI cortical responses from video, audio, and text.

This harness focuses on **repeatable capability checks**: environment validation, multimodal inference, and contrast benchmarks inspired by the TRIBE v2 paper's in-silico experiments.

## What TRIBE v2 does

| Property | Detail |
|----------|--------|
| Input | Video (V-JEPA2), audio (Wav2Vec-BERT), text (LLaMA 3.2-3B) |
| Output | `(timesteps, 20484)` predictions on the fsaverage5 cortical surface at 1 Hz |
| Weights | [`facebook/tribev2`](https://huggingface.co/facebook/tribev2) on Hugging Face |
| License | CC-BY-NC 4.0 (research, non-commercial) |

## Hardware requirements

Full trimodal inference loads three large encoders simultaneously and needs **at least 40 GB VRAM** (A100 40 GB or better). Audio-only or video-only paths use less memory but still benefit from a datacenter GPU.

CPU-only machines can still run the harness unit tests and environment checks.

## Quick start

```bash
cd tribev2
bash scripts/setup.sh
export HF_TOKEN=<your_huggingface_read_token>   # required for text inference
python scripts/run_capabilities.py --preload-llama
```

The first run downloads:

- ~1 GB TRIBE encoder weights (`facebook/tribev2`)
- ~6 GB LLaMA 3.2-3B weights (gated; accept the license on Hugging Face first)
- Additional encoder weights on first use per modality

## Project layout

```
tribev2/
├── config/default.yaml          # Model + benchmark configuration
├── stimuli/                     # Sample text inputs for contrast tests
├── src/tribe_capabilities/      # Harness library
├── scripts/
│   ├── setup.sh                 # Virtualenv + dependency install
│   ├── check_environment.py     # GPU / package / HF token checks
│   ├── run_capabilities.py      # Main benchmark runner
│   └── run_framing_rct.py       # Kahneman framing RCT generator + analysis
├── data/framing_rct/            # RCT dataset (assignments, stimuli, protocol)
├── tests/                       # Unit tests (no GPU required)
└── outputs/reports/             # JSON capability reports (generated)
```

## Capability benchmarks

`scripts/run_capabilities.py` runs:

1. **Environment audit** — Python, CUDA, VRAM, NumPy, Hugging Face token
2. **Language vs visual contrast** — Two text stimuli through the model; reports activation contrast statistics
3. **Optional modality tests** — Enable audio/video in `config/default.yaml` after adding media files under `stimuli/`

## Kahneman loss/gain framing RCT

A pre-registered RCT dataset tests whether TRIBE v2 cortical predictions show the same **loss vs gain framing asymmetry** documented by Kahneman and Tversky (1981).

| Property | Value |
|----------|-------|
| Design | Within-subjects crossover (one frame per scenario per subject) |
| Subjects | 60 |
| Scenarios | 8 matched gain/loss pairs (health, financial, economic, …) |
| Trials | 480 randomized assignments |
| Analysis | Paired gain vs loss cortical magnitude per scenario |

```bash
# Generate or refresh the RCT dataset (CPU only)
python scripts/run_framing_rct.py --generate-only

# Run TRIBE inference + paired statistics on GPU
export HF_TOKEN=<token>
python scripts/run_framing_rct.py --preload-llama
```

Dataset files live in [`data/framing_rct/`](data/framing_rct/README.md):

- `assignments.csv` — 480 trial-level RCT assignments
- `unique_stimuli.csv` — 16 gain/loss stimuli for model inference
- `protocol.json` — study metadata and hypotheses
- `stimuli/*.txt` — one file per unique stimulus

The analysis tests whether **loss-framed** wording produces stronger mean absolute cortical activation than objectively equivalent **gain-framed** wording, matching the directional salience predicted by prospect theory.

Results are saved to `outputs/reports/framing_rct_analysis.json`.

## GitHub Actions

| Workflow | Runner | Cost | What it does |
|----------|--------|------|--------------|
| `tribev2-ci.yml` | `ubuntu-latest` (CPU) | **Free** on public repos | Unit tests + RCT dataset validation |
| `tribev2-framing-gpu.yml` | GPU larger runner (T4) | **Paid** (~$0.052/min) | Framing inference smoke test |

GitHub does **not** include GPU runners in the free Actions tier. To run inference in CI:

1. Enable [GPU larger runners](https://docs.github.com/en/actions/reference/runners/larger-runners) on a GitHub Team/Enterprise org
2. Add repository secret `HF_TOKEN`
3. Run **Actions → TRIBE v2 Framing GPU → Run workflow** (start with `max_scenarios: 2`)

See [`.github/workflows/README.md`](../.github/workflows/README.md) for full setup. T4 runners have 16 GB VRAM; full inference needs ~40 GB, so use local A100 if the GPU job OOMs.

Example report fields:

```json
{
  "benchmarks": [
    {
      "name": "language_vs_visual",
      "status": "passed",
      "details": {
        "condition_a": {"shape": [18, 20484], "mean_activation": 0.02},
        "contrast": {"contrast_mean_abs": 0.15}
      }
    }
  ]
}
```

## Commands

```bash
# Environment only (works on CPU)
python scripts/check_environment.py

# Environment JSON for CI
python scripts/check_environment.py --json

# Skip GPU benchmarks (write environment report only)
python scripts/run_capabilities.py --skip-gpu

# Full run + cortical HTML snapshot
python scripts/run_capabilities.py --preload-llama --visualize

# Unit tests (GPU test auto-skips without CUDA)
pytest
```

## Hugging Face authentication

Text inference uses the gated [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) model:

1. Accept Meta's license on the model page
2. Create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Export `HF_TOKEN` or run `huggingface-cli login`

Audio-only inference does not require the LLaMA token.

## References

- [TRIBE v2 Hugging Face weights](https://huggingface.co/facebook/tribev2)
- [Official GitHub repository](https://github.com/facebookresearch/TRIBEv2)
- [Meta AI blog post](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/)
- [Interactive demo](https://aidemos.atmeta.com/tribev2/)
- [Colab demo notebook](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)

## Citation

```bibtex
@article{dascoli2026foundation,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brooks, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
  journal={arXiv preprint arXiv:2605.04326},
  year={2026}
}
```
