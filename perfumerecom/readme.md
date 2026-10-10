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

`Perfume_Recommender_Decision1.ipynb` is a new version of the tool. **Every decision is made by [Jev](https://openrouter.ai/docs/guides/community/jev)** (`typesafe/jev-1.13`), a decision model on OpenRouter. No LLMs, no embedding model and no similarity search are involved. Jev looks at every perfume in the catalog and picks the best ones in a knockout tournament:

1. **Who is it for?** Jev decides men, women, unisex or anyone. A request for men searches men's and unisex perfumes, a request for women searches women's and unisex perfumes, and "unisex" searches unisex perfumes only.
2. **Heats:** the catalog is shuffled and split into groups of about 80. For each group Jev answers *"which of these perfumes best fits the customer?"*, seeing each perfume's notes, accords, rating and number of ratings, and gives every perfume a probability. The best few from each group go through, about 1 in 20.
3. **Semi-final:** the survivors compete the same way until 30 are left.
4. **Final:** five experts (Literal, Emotional, Contextual, Resolver, Synthesizer) compare the 30 side by side. The top five by average vote are the recommendations.

Perfumes you name in the request ("like Aventus but softer") go straight to the final.

The catalog uses v1's filters except the release year: brands with at least 60 perfumes in `fragrantica.xlsx`, which drops little-known brands, and a rating of 4.0 or higher, from any year. That's 5,041 of the 23,973 perfumes; v1 also required 2000 or later, which left 4,173. The filters, heat size and number of finalists are settings in the notebook.

A search is about 40 Jev requests sent in parallel, a few seconds and roughly a cent on your OpenRouter account. Nothing needs to be downloaded besides three small Python packages, so it runs on any Colab runtime, with no GPU.

To run it you need an [OpenRouter API key](https://openrouter.ai/keys). On Colab, save it as a secret named `Decision_api` (with "Notebook access" turned on); otherwise the notebook asks for it. The notebook downloads `fragrantica.xlsx` from this repo if it isn't next to the notebook.
