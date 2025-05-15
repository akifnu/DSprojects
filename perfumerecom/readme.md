# 🟢 Perfume Recommender

## Introduction

**Perfume Recommender** is an AI-powered tool designed to help anyone find their perfect scent—not just by brand or name, but by emotion, memory, or mood. Powered by advanced language models and intelligent ranking, this system bridges the gap between what you *feel* and what you want to smell.

---

## 📁 Files Included

- **Perfume_Recommender.ipynb** – Main notebook for recommendations
- **fragrantica.xlsx** – Cleaned perfume database (required for searching)
- **faiss_index.bin** – Precomputed search index (for fast recommendations)
- **perfume_embeddings.npy** – Precomputed AI embeddings of all perfume descriptions
- **readme.md** – This documentation

No setup required: All necessary embeddings and search indexes are already provided!

---

## 📊 Dataset

This project uses the [Fragrantica.com Fragrance Dataset](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset?select=fra_perfumes.csv), further cleaned and formatted as `fragrantica.xlsx` for best results.

---

## 🚀 Features

- **Natural Language Search**: Describe your dream scent in your own words—no need to know the brand or perfume name.
- **AI Matching**: Advanced language models and semantic search match your description to the best perfumes in the database.
- **Panel of Virtual Experts**: Five AI “agents” rank the top matches from different perspectives (literal, emotional, contextual, etc.) and agree on the best five for you.
- **Instant Recommendations**: Get your results in seconds—no more endless sampling.
- **Ready for Shops or Individuals**: Useful for shoppers, gift-givers, or boutique staff wanting to delight their clients.

---

## 🛠️ Getting Started

**Clone the repository:**

```bash
git clone https://github.com/akifnu/DSprojects.git
cd DSprojects/perfumerecom
