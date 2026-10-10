# Perfume Recommender

This project is a simple way to help people find perfumes based on feelings, memories, or moods—not just by searching for a brand or a name.

## Background

The idea for this tool came from my own visits to perfume shops during my travels. I noticed that many customers didn’t ask for a specific perfume. Instead, they would describe a scene, a season, or a feeling they wanted to capture. Sometimes the staff found it difficult to match these requests, and it could take a long time to find the right scent.

With this tool, I wanted to make that process a bit easier, both for people who enjoy perfume and for shop owners who want to help their customers more quickly.

## What it does

You can describe the type of perfume you’re looking for in your own words. The tool will look through a database of perfumes and try to find options that match what you’ve described. It uses language models and some AI to make sense of your words, then suggests five perfumes that are likely to fit your request. These suggestions are chosen with help from several virtual “agents” that each look at your input in a slightly different way.

## Files in this folder

You’ll find:
- `Perfume_Recommender.ipynb` (the main notebook)
- `fragrantica.xlsx` (the perfume data)
- `faiss_index.bin` and `perfume_embeddings.npy` (for faster searching)
- `readme.md` (this file)

## How the search works

The index and embedding files are already included so your searches will be quick. If you delete these files or change the data, the tool will notice and make new ones by itself. You don’t need to do anything extra.

## The data

The perfume data comes from the [Fragrantica.com Fragrance Dataset on Kaggle](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset?select=fra_perfumes.csv). I cleaned it a bit and saved it as `fragrantica.xlsx` for this project. It has 23,973 perfumes and only the columns the recommender uses: `Perfume`, `Brand`, `Gender`, `Year`, `Rating Value`, `Rating Count`, `Notes` (top, middle and base notes) and `Accord` (the main accords, strongest first).

## Getting started

You can use this on Google Colab, Jupyter, or locally. To get going, just run:

```bash
git clone https://github.com/akifnu/DSprojects.git
cd DSprojects/perfumerecom
```

## Version 2: Jev on OpenRouter

`Perfume_Recommender_Decision1.ipynb` is a new version of the tool. It uses no LLMs: [Jev](https://openrouter.ai/docs/guides/community/jev) (`typesafe/jev-1.13`), a decision model on OpenRouter, does the judging, and a small embedding model helps with search.

How a search works:

1. **Jev reads the request.** In one request it answers who the perfume is for (men, women, unisex or anyone) and a yes/no question for each of about 50 accords: *"Would a perfume that fits this request have a prominent aquatic accord?"* The probabilities form a **scent profile**. "A summer evening by the sea", for example, comes out high on aquatic, citrus and fresh.
2. **The shortlist of 30 is chosen by three signals:**
   - **accord fit:** how well each perfume's main accords match the scent profile (50%)
   - **meaning:** embedding similarity between the request and the perfume's name, brand, notes and accords (35%), using [Microsoft Harrier OSS v1 270M](https://huggingface.co/microsoft/harrier-oss-v1-270m)
   - **popularity:** number and average of ratings, so well-loved perfumes beat obscure ones when the fit is similar (15%)

   Perfumes you name ("like Aventus but softer") are always included. A request for men only searches men's and unisex perfumes, and the same goes for women.
3. **Jev judges the shortlist.** Five experts (Literal, Emotional, Contextual, Resolver, Synthesizer) each compare all 30 perfumes side by side and give each one a probability of being the best fit. The final top 5 blends their consensus (60%) with the shortlist score (40%).

The searchable catalog uses v1's filters except the release year: brands with at least 60 perfumes in `fragrantica.xlsx`, which drops little-known brands, and a rating of 4.0 or higher, from any year. That's 5,041 of the 23,973 perfumes; v1 also required 2000 or later, which left 4,173. The filters and weights are settings at the top of the notebook's cells.

The perfume embeddings are precomputed and stored in this folder, so Colab downloads them in seconds instead of embedding the catalog in every session. The **Perfume embeddings** GitHub Actions workflow (`.github/workflows/perfume-embeddings.yml`) rebuilds them with `build_embeddings.py` whenever the data or the notebook changes on `main`. You can also start it by hand from the Actions tab.

To run it you need an [OpenRouter API key](https://openrouter.ai/keys). On Colab, save it as a secret named `Decision_api` (with "Notebook access" turned on); otherwise the notebook asks for it. The notebook downloads `fragrantica.xlsx` from this repo if it isn't next to the notebook.
