# Kahneman Framing RCT Dataset

Pre-registered style dataset for testing whether TRIBE v2 cortical predictions show the same **loss vs gain framing asymmetry** documented by Kahneman and Tversky.

## Design

| Property | Value |
|----------|-------|
| Study ID | `kahneman_framing_rct_v1` |
| Design | Within-subjects crossover RCT |
| Subjects | 60 (`SUBJ_001` … `SUBJ_060`) |
| Scenarios | 8 matched gain/loss pairs |
| Trials | 480 (60 subjects × 8 scenarios) |
| Random seed | 42 |
| Analysis unit (in-silico) | Scenario pair |

Each subject sees every scenario exactly once. Frame assignment alternates across subjects so each scenario is shown in the **gain frame to 30 subjects** and the **loss frame to 30 subjects**.

Presentation order is block-randomized (gain/loss order swapped within blocks of two trials).

## Files

| File | Description |
|------|-------------|
| `scenarios.json` | Scenario bank with objective-equivalent gain/loss wordings |
| `assignments.csv` | Full RCT assignment table (480 rows) |
| `unique_stimuli.csv` | 16 unique stimuli for TRIBE inference (8 scenarios × 2 frames) |
| `protocol.json` | Study metadata and generation parameters |
| `stimuli/*.txt` | One text file per unique stimulus |

## Behavioral benchmark (Kahneman)

Classic framing effects predict different choices for objectively equivalent options:

- **Gain frame** → risk-averse preferences (choose certain outcomes)
- **Loss frame** → risk-seeking preferences (choose gambles to avoid certain losses)

Neural proxy tested here:

> Loss-framed wording elicits **stronger mean absolute cortical activation** than gain-framed wording for the same underlying outcome.

## Run analysis

```bash
cd tribev2
python scripts/run_framing_rct.py --generate-only   # regenerate dataset
python scripts/run_framing_rct.py --preload-llama   # GPU inference + stats
```

Results are written to `outputs/reports/framing_rct_analysis.json`.

## Scenarios

1. Asian disease problem (health)
2. Surgery success/failure rate (health)
3. Money keep/lose framing (financial)
4. Credit card discount/surcharge (financial)
5. Employment vs unemployment rate (economic)
6. Lean vs fat beef label (consumer)
7. Exam correct vs incorrect rate (education)
8. Vaccine no-side-effect vs side-effect rate (health)

## Citation

```bibtex
@article{tversky1981framing,
  title={The framing of decisions and the psychology of choice},
  author={Tversky, Amos and Kahneman, Daniel},
  journal={Science},
  volume={211},
  number={4481},
  pages={453--458},
  year={1981}
}
```
