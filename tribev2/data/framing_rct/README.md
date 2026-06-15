# Kahneman Framing RCT (text only)

Large-scale randomized crossover dataset for testing loss vs gain framing with TRIBE v2.

## Scale

| Item | Count |
|------|------:|
| Scenario pairs (gain + loss) | 318 |
| Unique text stimuli | 636 |
| Subjects | 200 |
| Trial assignments | 63,600 |

## Design

- **Within-subjects crossover**: each subject sees every scenario once
- **Balanced framing**: 100 subjects gain / 100 subjects loss per scenario
- **Block-randomized** presentation order
- **Seed**: 42

## Files

| File | Description |
|------|-------------|
| `scenarios.json` | Full scenario bank |
| `assignments.csv` | 63,600-row RCT assignment table |
| `unique_stimuli.csv` | 636 inference texts |
| `stimuli/*.txt` | One file per text |
| `protocol.json` | Study metadata |

## Regenerate

```bash
python scripts/generate_rct_dataset.py
```

## Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akifnu/DSprojects/blob/main/tribev2/notebooks/Kahneman_Framing_RCT.ipynb)

## Run locally (GPU)

```bash
export HF_TOKEN=<token>
python scripts/run_framing_rct.py --preload-llama --max-scenarios 12
```

Remove `--max-scenarios` for the full 318-pair study (long run).
