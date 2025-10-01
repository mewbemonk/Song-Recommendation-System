# 🎵 Song Recommendation Engine

A content-based song recommender built with TF-IDF and cosine similarity. Users can select a song and get top 5 lyrically similar recommendations using a clean Streamlit interface.

---

## 🚀 Features

- TF-IDF vectorization of lyrics + artist metadata
- Cosine similarity-based recommendations
- Streamlit UI for interactive selection
- Modular saving/loading with `joblib`
- NLTK-powered lemmatization and tokenization

---

## 🧠 How It Works

1. Lyrics and artist names are combined into a `tag` field.
2. Text is cleaned: lowercased, tokenized, lemmatized, and punctuation removed.
3. TF-IDF vectorizer transforms the corpus.
4. Cosine similarity finds the most similar songs.
5. Streamlit displays recommendations based on user selection.

---

## 📦 Installation

```bash
pip install -r requirements.txt