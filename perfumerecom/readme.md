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

The perfume data comes from the [Fragrantica.com Fragrance Dataset on Kaggle](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset?select=fra_perfumes.csv). I cleaned it a bit and saved it as `fragrantica.xlsx` for this project. It has 23,973 perfumes and only the columns the recommender uses: `Perfume`, `Brand`, `Gender`, `Year`, `Rating Value`, `Notes` (top, middle and base notes) and `Accord` (the main accords, strongest first).

## Getting started

You can use this on Google Colab, Jupyter, or locally. To get going, just run:

```bash
git clone https://github.com/akifnu/DSprojects.git
cd DSprojects/perfumerecom
```

## Version 2: Jev on OpenRouter

`Perfume_Recommender_Decision1.ipynb` is a new version of the tool. What changed:

- **Every perfume is searchable:** all 23,973 perfumes in `fragrantica.xlsx`. v1 only searched about 4,000 (big brands, released in 2000 or later, rated 4.0 or higher). Those filters can be switched back on at the top of the data cell.
- **Search by name:** each perfume's name and brand are part of the text it's searched by. If your request names a perfume ("like Aventus but softer"), that perfume is always put on the shortlist.
- **Gender filter:** a request for men searches men's and unisex perfumes, a request for women searches women's and unisex perfumes, and "unisex" searches unisex perfumes only.
- **Faster search model:** [Microsoft Harrier OSS v1 270M](https://huggingface.co/microsoft/harrier-oss-v1-270m) replaces e5-large. It's about half the size and scores higher on the MTEB benchmark. Setting `EMBED_MODEL` back to v1's e5 model is a one-line change.
- **No FAISS:** with about 24,000 perfumes, a single numpy calculation takes a few milliseconds. The first run embeds the catalog once and caches it.
- **New ranking model:** the five virtual experts run on [Jev](https://openrouter.ai/docs/guides/community/jev) (`typesafe/jev-1.13`) through OpenRouter's Decisions API, instead of Llama 4 on Groq.

Jev doesn't write text. It picks from options you define and gives a probability for each one. Each expert (Literal, Emotional, Contextual, Resolver, Synthesizer) looks at all 30 shortlisted perfumes side by side and gives each one a probability of being the best fit. The top 5 are the perfumes with the highest average, and the results table shows every expert's vote.

To run it you need an [OpenRouter API key](https://openrouter.ai/keys). On Colab, save it as a secret named `Decision_api` (with "Notebook access" turned on); otherwise the notebook asks for it. The notebook downloads `fragrantica.xlsx` from this repo if it isn't next to the notebook.
