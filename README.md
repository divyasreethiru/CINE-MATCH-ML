# 🎬 CineMatch — ML Movie Recommendation System

> Content-based movie recommender using TF-IDF + Cosine Similarity
> Built for: college portfolio · internships · hackathons

---

## 📁 Project Structure

```
movie-recommender/
├── data/                         ← Put TMDB CSVs here
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── ml/
│   └── recommender.py            ← Core ML engine
│
├── backend/
│   ├── app.py                    ← Flask REST API
│   └── requirements.txt
│
├── frontend/
│   └── index.html                ← Full UI (zero dependencies)
│
└── models/                       ← Auto-created after training
    ├── similarity.pkl
    ├── tfidf.pkl
    └── movies.pkl
```

---

## 🗃️ Dataset

Download from Kaggle:
👉 https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Download both files:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Place them in the `data/` folder.

---

## ⚙️ How It Works

### Algorithm: Content-Based Filtering

1. **Feature Extraction** — For each movie, extract:
   - Genres, keywords, top-5 cast, director (weighted 2×), overview

2. **TF-IDF Vectorization** — Convert feature text to numerical vectors
   - Vocabulary size: 15,000 terms
   - Captures term importance across the corpus

3. **Cosine Similarity** — Compute pairwise similarity between all movies
   - Result: 4803 × 4803 similarity matrix
   - Score of 1 = identical, 0 = completely different

4. **Recommendation** — For a query movie, return top-N most similar films

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
cd ml
python recommender.py ../data
```
This saves `models/similarity.pkl`, `models/tfidf.pkl`, `models/movies.pkl`

### Step 3 — Start the backend
```bash
cd backend
python app.py
# Server runs at http://localhost:5000
```

### Step 4 — Open the frontend
Open `frontend/index.html` in any browser.
(No build step needed — pure HTML/CSS/JS)

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server + model status |
| GET | `/api/recommend?title=Inception&n=10` | Get recommendations |
| GET | `/api/search?q=dark&limit=7` | Autocomplete search |
| GET | `/api/movies/popular` | Top 20 popular movies |
| GET | `/api/movies/:id` | Movie detail by TMDB ID |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Dataset | TMDB 5000 Movies |
| Algorithm | TF-IDF + Cosine Similarity |
| Vocabulary | 15,000 terms |
| Matrix shape | ~4800 × ~4800 |
| Training time | ~10 seconds |
| Inference time | <50ms |

---

## 🧠 Interview Talking Points

- **Why TF-IDF?** Captures term importance; rare but meaningful words (e.g., director names) get higher weight than common words
- **Why Cosine Similarity?** Direction matters, not magnitude. Two movies with same genres/cast point in same direction regardless of overview length.
- **Limitations:** Cold-start problem (new movies with no data), no user personalization, popularity bias
- **Improvements:** Collaborative filtering, hybrid model (content + user ratings), matrix factorization (SVD), neural embeddings

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, pandas, numpy |
| Backend | Flask, Flask-CORS |
| Frontend | HTML5, CSS3, Vanilla JS |
| Data | TMDB 5000 (Kaggle) |

---

## 📈 Possible Extensions

- [ ] Add collaborative filtering (user-based)
- [ ] Deploy on Render / Railway / Heroku
- [ ] Add TMDB API integration for live posters
- [ ] Build watchlist with localStorage
- [ ] Add genre/year filters
- [ ] Dockerize the full stack

---

*Built with ❤️ as an ML portfolio project*