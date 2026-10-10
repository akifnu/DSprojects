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

The perfume data comes from the [Fragrantica.com Fragrance Dataset on Kaggle](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset?select=fra_perfumes.csv). I cleaned it a bit and saved it as `fragrantica.xlsx` for this project.

## Getting started

You can use this on Google Colab, Jupyter, or locally. To get going, just run:

```bash
git clone https://github.com/akifnu/DSprojects.git
cd DSprojects/perfumerecom
```

## Version 2: Microsoft Decision-1 on OpenRouter

`Perfume_Recommender_Decision1.ipynb` is a new version of the tool. Three things changed:

- **Faster search model:** [Microsoft Harrier OSS v1 270M](https://huggingface.co/microsoft/harrier-oss-v1-270m) replaces e5-large. It's about half the size and scores higher on the MTEB multilingual benchmark (66.6 vs 63.2), especially on retrieval (66.4 vs 57.1).
- **No FAISS:** with a few thousand perfumes, a single numpy calculation is just as fast. The first run embeds the catalog once and saves it as `perfume_embeddings_harrier270m.npy`.
- **New ranking model:** the five virtual experts run on [Microsoft-Decision-1](https://openrouter.ai/microsoft) through OpenRouter's Decisions API instead of Llama 4 on Groq. They score a shortlist of 15 perfumes instead of 25.

Microsoft-Decision-1 doesn't write text. It answers fixed questions with a probability for each possible answer. Each expert (Literal, Emotional, Contextual, Resolver, Synthesizer) scores each of the 15 candidates from 0 to 4. The top 5 are the perfumes with the highest average score, and you can see every expert's score in the results table.

To run it you need an [OpenRouter API key](https://openrouter.ai/keys). On Colab, save it as a secret named `Decision_api` (with "Notebook access" turned on); otherwise the notebook asks for it. Keep `fragrantica.xlsx` next to the notebook, or upload it to `/content` on Colab.
