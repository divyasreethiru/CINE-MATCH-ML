"""
Flask REST API — Movie Recommender Backend
Endpoints:
  GET  /api/recommend?title=<movie>&n=10
  GET  /api/search?q=<query>&limit=10
  GET  /api/movies/popular
  GET  /health
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add ml directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))
from recommender import load_model, get_recommendations, search_movies

# ──────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow frontend on different port

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

# ──────────────────────────────────────────────
# MODEL LOADING (lazy singleton)
# ──────────────────────────────────────────────

_model = {"sim": None, "tfidf": None, "df": None, "loaded": False}

def get_model():
    if not _model["loaded"]:
        try:
            sim, tfidf, df = load_model(MODEL_DIR)
            _model["sim"]    = sim
            _model["tfidf"]  = tfidf
            _model["df"]     = df
            _model["loaded"] = True
            print(f"Model loaded: {len(df)} movies")
        except Exception as e:
            print(f"Model load failed: {e}")
    return _model["sim"], _model["tfidf"], _model["df"]


def enrich_poster(movies: list) -> list:
    """Prepend TMDB base URL to poster paths."""
    for m in movies:
        if m.get("poster_path"):
            m["poster_url"] = TMDB_IMAGE_BASE + m["poster_path"]
        else:
            m["poster_url"] = f"https://placehold.co/300x450/1a1a2e/ffffff?text={m['title'].replace(' ', '+')}"
    return movies


def _row_to_dict(row):
    return {
        "id":           int(row.get("id", 0)),
        "title":        row["title"],
        "vote_average": round(float(row.get("vote_average", 0)), 1),
        "vote_count":   int(row.get("vote_count", 0)),
        "genres":       row["genres_list"][:3],
        "director":     row["director"],
        "cast":         row.get("cast_list", [])[:5],
        "poster_path":  row.get("poster_path", ""),
        "overview":     str(row.get("overview", ""))[:200],
        "release_date": str(row.get("release_date", "")),
        "runtime":      int(row["runtime"]) if pd.notna(row.get("runtime")) else None,
    }

def _rows_to_list(df_slice):
    return [_row_to_dict(row) for _, row in df_slice.iterrows()]



# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route("/health")
def health():
    sim, _, df = get_model()
    return jsonify({
        "status": "ok",
        "model_loaded": sim is not None,
        "movies_count": len(df) if df is not None else 0
    })


@app.route("/api/stats")
def stats():
    _, _, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503

    all_genres = {g for gs in df["genres_list"] for g in gs}
    years      = df["release_date"].astype(str).str[:4]

    return jsonify({
        "total_movies":  len(df),
        "avg_rating":    round(float(df["vote_average"].mean()), 2),
        "total_genres":  len(all_genres),
        "year_range":    [years.min(), years.max()],
        "top_directors": df["director"].value_counts().head(5).to_dict(),
    })


@app.route("/api/movies/random")
def random_movie():
    genre = request.args.get("genre", "").strip()
    _, _, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503

    pool = df
    if genre:
        pool = df[df["genres_list"].apply(lambda g: genre.lower() in [x.lower() for x in g])]

    if pool.empty:
        return jsonify({"error": f"No movies found for genre '{genre}'"}), 404

    row    = pool.sample(1).iloc[0]
    result = enrich_poster([_row_to_dict(row)])[0]
    return jsonify(result)


@app.route("/api/recommend")
def recommend():
    title = request.args.get("title", "").strip()
    n     = min(int(request.args.get("n", 10)), 20)

    if not title:
        return jsonify({"error": "title parameter required"}), 400

    sim, tfidf, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503

    results = get_recommendations(title, df, sim, top_n=n)
    results = enrich_poster(results)

    if not results:
        return jsonify({"error": f"Movie '{title}' not found", "results": []}), 404

    return jsonify({
        "query":   title,
        "count":   len(results),
        "results": results
    })


@app.route("/api/recommend/batch", methods=["POST"])
def recommend_batch():
    body   = request.get_json(silent=True) or {}
    titles = body.get("titles", [])[:5]
    n      = min(body.get("n", 10), 20)

    if not titles:
        return jsonify({"error": "titles array required"}), 400

    sim, tfidf, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503

    seed_set = {t.lower() for t in titles}
    seen, results = set(), []

    for title in titles:
        for movie in get_recommendations(title, df, sim, top_n=n * 2):
            key = movie["title"].lower()
            if key not in seen and key not in seed_set:
                seen.add(key)
                results.append(movie)

    results = results[:n]
    return jsonify({
        "seeds":   titles,
        "count":   len(results),
        "results": enrich_poster(results)
    })


@app.route("/api/recommend/hybrid")
def recommend_hybrid():
    title = request.args.get("title", "").strip()
    n     = min(int(request.args.get("n", 10)), 20)

    if not title:
        return jsonify({"error": "title parameter required"}), 400

    sim, tfidf, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503

    results = get_recommendations(title, df, sim, top_n=n)
    results = enrich_poster(results)

    if not results:
        return jsonify({"error": f"Movie '{title}' not found", "results": []}), 404

    return jsonify({
        "query":   title,
        "count":   len(results),
        "results": results
    })


@app.route("/api/search")
def search():
    q     = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 10)), 20)

    if len(q) < 2:
        return jsonify({"results": []})

    _, _, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503

    titles = search_movies(q, df, limit=limit)
    return jsonify({"results": titles})


@app.route("/api/movies/popular")
def popular():
    _, _, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503

    # Top movies by weighted rating (vote_average * log(vote_count))
    df_copy = df.copy()
    df_copy["score"] = df_copy["vote_average"] * np.log1p(df_copy["vote_count"].fillna(0))
    top = df_copy.nlargest(20, "score")

    results = []
    for _, row in top.iterrows():
        results.append({
            "id":           int(row.get("id", 0)),
            "title":        row["title"],
            "vote_average": round(float(row.get("vote_average", 0)), 1),
            "genres":       row["genres_list"][:3],
            "director":     row["director"],
            "poster_path":  row.get("poster_path", ""),
            "overview":     str(row["overview"])[:200] + "..." if len(str(row.get("overview", ""))) > 200 else str(row.get("overview", "")),
        })

    results = enrich_poster(results)
    return jsonify({"results": results})


@app.route("/api/genres")
def genres():
    _, _, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503
    all_genres = sorted({g for genres in df["genres_list"] for g in genres})
    return jsonify({"genres": all_genres})


@app.route("/api/movies/by-genre")
def by_genre():
    genre      = request.args.get("genre", "").strip()
    sort_by    = request.args.get("sort", "rating")
    limit      = min(int(request.args.get("limit", 20)), 50)

    if not genre:
        return jsonify({"error": "genre parameter required"}), 400

    _, _, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503

    mask     = df["genres_list"].apply(lambda g: genre.lower() in [x.lower() for x in g])
    filtered = df[mask].copy()

    if filtered.empty:
        return jsonify({"error": f"No movies found for genre '{genre}'", "results": []}), 404

    if sort_by == "popularity":
        filtered["score"] = filtered["vote_average"] * np.log1p(filtered["vote_count"].fillna(0))
        filtered = filtered.nlargest(limit, "score")
    else:
        filtered = filtered.nlargest(limit, "vote_average")

    results = enrich_poster(_rows_to_list(filtered))
    return jsonify({"genre": genre, "count": len(results), "results": results})


@app.route("/api/directors")
def directors():
    _, _, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503
    top = df["director"].value_counts().head(50)
    return jsonify({
        "directors": [{"name": k, "movie_count": int(v)} for k, v in top.items()]
    })


@app.route("/api/eval")
def eval_metrics():
    # Import evaluation function
    from recommender import evaluate_model, load_full_model
    try:
        sim_matrix, tfidf, df, collab_sim, popularity_scores, sentiment_scores = load_full_model(MODEL_DIR)
        metrics = evaluate_model(None, df, sim_matrix, collab_sim=collab_sim, popularity_scores=popularity_scores, sentiment_scores=sentiment_scores, k=10, sample_size=50)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/movies/<int:movie_id>")
def movie_detail(movie_id):
    _, _, df = get_model()
    if df is None:
        return jsonify({"error": "Model not loaded"}), 503

    matches = df[df["id"] == movie_id]
    if matches.empty:
        return jsonify({"error": "Movie not found"}), 404

    row = matches.iloc[0]
    result = {
        "id":           int(row.get("id", 0)),
        "title":        row["title"],
        "overview":     str(row.get("overview", "")),
        "vote_average": round(float(row.get("vote_average", 0)), 1),
        "vote_count":   int(row.get("vote_count", 0)),
        "genres":       row["genres_list"],
        "director":     row["director"],
        "cast":         row["cast_list"],
        "keywords":     row["keywords_list"][:10],
        "poster_path":  row.get("poster_path", ""),
        "runtime":      int(row.get("runtime", 0)) if pd.notna(row.get("runtime")) else None,
        "release_date": str(row.get("release_date", "")),
    }
    result = enrich_poster([result])[0]
    return jsonify(result)


# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}...")
    app.run(debug=True, host="0.0.0.0", port=port)