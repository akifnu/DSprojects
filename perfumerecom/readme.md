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

`Perfume_Recommender_Decision1.ipynb` is a new version of the tool. Every decision is made by [Jev](https://openrouter.ai/docs/guides/community/jev) (`typesafe/jev-1.13`), a decision model on OpenRouter. There are no LLMs, no embeddings, and no hand-written rules or hints. Jev is asked two plain questions and sees each perfume exactly as the data describes it: name, brand, gender, main accords and notes.

1. **Heats:** the catalog is shuffled into groups of 40, and for each group Jev answers *"Which perfume best matches the customer's request?"* with a probability for every perfume. Each perfume is judged in two different groups and its results are averaged. The best eighth go through, and this repeats until 30 finalists remain.
2. **Final:** Jev rates each finalist on its own: *"How well does this perfume match the customer's request?"*, from "does not match" to "matches perfectly". The top five are the recommendations.

A search over the 5,041-perfume catalog is about 290 Jev requests sent 16 at a time, roughly 10–20 seconds and about 3 cents. The notebook also has:
- **Check Jev:** one small request that prints Jev's raw answer, to confirm it's read correctly.
- **Accuracy Test:** test requests generated from the data (perfume names, pairs of notes, accords with gender), plus any you add to `MY_TESTS`, reported as a percentage of requests where a right perfume is in the top five.

The catalog uses v1's filters except the release year: brands with at least 60 perfumes in `fragrantica.xlsx` and a rating of 4.0 or higher, from any year (5,041 of 23,973 perfumes).

To run it you need an [OpenRouter API key](https://openrouter.ai/keys). On Colab, save it as a secret named `Decision_api` (with "Notebook access" turned on); otherwise the notebook asks for it. The notebook downloads `fragrantica.xlsx` from this repo if it isn't next to the notebook.
