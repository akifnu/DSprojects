# TRIBE v2 Reel Studio

Upload a reel, an audio file or a photo and see TRIBE v2's predicted brain response.
Use **Start at** / **Seconds to analyze** to analyze only part of a video or audio file
(2 extra seconds are analyzed past the cut and discarded, so the last second stays reliable).
Keep **Color scale max** the same when comparing reels; charts always show the same lines on a fixed axis.
- **Brain movie**: your reel side by side with 4 cortical views, updated every second, with an intensity timeline cursor (audio kept).
- **3D brain**: an interactive cortex you can rotate, with a time slider, Play button and Left/Right/Top/Front/Back presets.
- **Intensity**: downloadable graph (PNG + interactive HTML) of the response over time for the whole cortex and 9 brain systems (Visual, Faces & objects, Auditory, Language,
  Attention, Motor & touch, Prefrontal, Emotion & reward, Default mode), plus a per-system summary table.
- **Key moments**: the 3 peak seconds (frame + brain map) and the average activation map.
- **Downloads**: brain movie MP4, intensity CSV, network summary CSV, raw predictions (`.npy`, T × 20484 fsaverage5), and the standalone 3D HTML.

## Easiest: Google Colab (free GPU)
Open `TRIBE_Reel_Studio_Colab.ipynb` in Colab and follow the 4 setup steps at the top
(GPU runtime, LLaMA 3.2 access, `HF_TOKEN` secret, Run all). Then open the printed `gradio.live` link.

## Your own GPU machine
```bash
git clone https://github.com/facebookresearch/tribev2 && cd tribev2
pip install uv && uv pip install -e ".[plotting]" "transformers>=4.46" "faster-whisper>=1.1" "ctranslate2>=4.5" gradio plotly
huggingface-cli login            # account with access to meta-llama/Llama-3.2-3B
python reel_app/app.py --preload # then open http://localhost:7860  (add --share for a public link)
```
`python reel_app/app.py --demo` runs the UI with synthetic data (no GPU or weights), for trying the interface only.

## Notes
- Predictions are for an *average* subject's cortex, already shifted by 5 s to compensate for the hemodynamic lag.
  Values are relative (a.u.), so compare reels with each other rather than reading absolute numbers.
- The network groupings are anatomical approximations from the Destrieux atlas, not functional localizers.
  "Emotion & reward" is a rough cortical proxy (OFC/ACC/insula). Subcortical structures like the amygdala are not modeled here.
- Speech is transcribed with WhisperX (English, French, Spanish, Dutch or Chinese). For reels with no speech (music only),
  the app falls back to video + audio features only.
- TRIBE v2 is licensed CC BY-NC 4.0 (non-commercial use only).
