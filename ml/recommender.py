"""
Movie Recommendation Engine
Uses TF-IDF + Cosine Similarity (Content-Based Filtering)
Dataset: TMDB 5000 Movies (https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
"""

import ast
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

def load_data(movies_path: str, credits_path: str) -> pd.DataFrame:
    """Load and merge TMDB movie + credits datasets."""
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    credits.rename(columns={"movie_id": "id"}, inplace=True)
    # Drop duplicate 'title' column to prevent 'title_x' and 'title_y' after merge
    if "title" in credits.columns:
        credits.drop(columns=["title"], inplace=True)
    df = movies.merge(credits, on="id")
    return df


def load_ratings(ratings_path: str) -> pd.DataFrame | None:
    """Load user ratings if available."""
    if not os.path.exists(ratings_path):
        return None
    df = pd.read_csv(ratings_path)
    required = {"user_id", "movie_id", "rating"}
    if not required.issubset(df.columns):
        raise ValueError("Ratings file must contain user_id, movie_id, rating")
    return df


def load_reviews(reviews_path: str) -> pd.DataFrame | None:
    """Load movie reviews if available."""
    if not os.path.exists(reviews_path):
        return None
    df = pd.read_csv(reviews_path)
    required = {"movie_id", "review_text"}
    if not required.issubset(df.columns):
        raise ValueError("Reviews file must contain movie_id and review_text")
    return df


# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def _safe_parse(obj):
    """Parse stringified list/dict safely."""
    try:
        return ast.literal_eval(obj)
    except Exception:
        return []


def extract_genres(row) -> list:
    return [g["name"].replace(" ", "") for g in _safe_parse(row)]


def extract_keywords(row) -> list:
    return [k["name"].replace(" ", "") for k in _safe_parse(row)]


def extract_cast(row, top_n: int = 5) -> list:
    return [c["name"].replace(" ", "") for c in _safe_parse(row)[:top_n]]


def extract_director(row) -> str:
    for person in _safe_parse(row):
        if person.get("job") == "Director":
            return person["name"].replace(" ", "")
    return ""


def build_soup(row) -> str:
    """Combine all features into a single text blob."""
    genres   = " ".join(row["genres_list"])
    keywords = " ".join(row["keywords_list"])
    cast     = " ".join(row["cast_list"])
    director = row["director"] + " " + row["director"]  # weight director 2×
    overview = row["overview"] if isinstance(row["overview"], str) else ""
    return f"{genres} {keywords} {cast} {director} {overview}"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Select useful columns
    cols = ["id", "title", "overview", "genres", "keywords", "cast", "crew",
            "vote_average", "vote_count", "popularity", "release_date",
            "poster_path", "runtime"]
    df = df[[c for c in cols if c in df.columns]]

    # Parse JSON-like columns
    df["genres_list"]   = df["genres"].apply(extract_genres)
    df["keywords_list"] = df["keywords"].apply(extract_keywords)
    df["cast_list"]     = df["cast"].apply(extract_cast)
    df["director"]      = df["crew"].apply(extract_director)

    # Drop rows with missing critical fields
    df.dropna(subset=["title", "overview"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Build feature soup
    df["soup"] = df.apply(build_soup, axis=1)

    return df


def compute_popularity(df: pd.DataFrame, ratings_df: pd.DataFrame | None = None) -> dict:
    if ratings_df is not None and not ratings_df.empty:
        agg = ratings_df.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index()
        agg["score"] = agg["mean"] * np.log1p(agg["count"])
        score_map = agg.set_index("movie_id")["score"].to_dict()
    else:
        scores = df["vote_average"].fillna(0) * np.log1p(df["vote_count"].fillna(0))
        score_map = dict(zip(df["id"].astype(int), scores))
    if not score_map:
        return {}
    min_score = min(score_map.values())
    max_score = max(score_map.values())
    if min_score == max_score:
        return {movie_id: 0.0 for movie_id in score_map}
    return {
        movie_id: float((score - min_score) / (max_score - min_score))
        for movie_id, score in score_map.items()
    }


def score_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    positive = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    score = positive - negative
    return float(np.tanh(score / max(1, len(tokens) / 10)))


def compute_review_sentiment(reviews_df: pd.DataFrame) -> dict:
    if reviews_df is None or reviews_df.empty:
        return {}
    reviews_df = reviews_df.copy()
    reviews_df = reviews_df[reviews_df["review_text"].notna()]
    if reviews_df.empty:
        return {}
    reviews_df["sentiment"] = reviews_df["review_text"].apply(score_sentiment)
    return reviews_df.groupby("movie_id")["sentiment"].mean().to_dict()


def train_collaborative(ratings_df: pd.DataFrame, df: pd.DataFrame) -> np.ndarray | None:
    if ratings_df is None or ratings_df.empty:
        return None
    pivot = ratings_df.pivot_table(
        index="movie_id",
        columns="user_id",
        values="rating",
        fill_value=0,
    )
    pivot = pivot.reindex(df["id"].astype(int).tolist(), fill_value=0)
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return None
    return cosine_similarity(pivot.values)


# ──────────────────────────────────────────────
# 3. MODEL TRAINING
# ──────────────────────────────────────────────

def train(df: pd.DataFrame):
    """
    Train TF-IDF vectorizer and compute cosine similarity matrix.
    Returns: (similarity_matrix, tfidf_vectorizer, processed_df)
    """
    tfidf = TfidfVectorizer(stop_words="english", max_features=15000)
    tfidf_matrix = tfidf.fit_transform(df["soup"])

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Cosine similarity (dense is fine for ≤10k movies)
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    return sim_matrix, tfidf, df


# ──────────────────────────────────────────────
# 4. RECOMMENDATION FUNCTION
# ──────────────────────────────────────────────

def get_recommendations(title: str,
                        df: pd.DataFrame,
                        sim_matrix: np.ndarray,
                        top_n: int = 10) -> list[dict]:
    """
    Return top-N similar movies for a given title.
    Each result: {title, vote_average, genres, director, poster_path, similarity}
    """
    # Normalise input
    title_lower = title.strip().lower()
    matches = df[df["title"].str.lower() == title_lower]

    if matches.empty:
        # Fuzzy fallback: partial match
        matches = df[df["title"].str.lower().str.contains(title_lower, na=False)]

    if matches.empty:
        return []

    idx = matches.index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]

    results = []
    for i, score in scores:
        row = df.iloc[i]
        results.append({
            "id":           int(row.get("id", 0)),
            "title":        row["title"],
            "vote_average": round(float(row.get("vote_average", 0)), 1),
            "genres":       row["genres_list"][:3],
            "director":     row["director"],
            "poster_path":  row.get("poster_path", ""),
            "similarity":   round(float(score), 3),
            "overview":     str(row["overview"])[:200] + "..." if len(str(row["overview"])) > 200 else str(row["overview"]),
        })
    return results


def search_movies(query: str, df: pd.DataFrame, limit: int = 10) -> list[dict]:
    """Search movie titles for autocomplete."""
    q = query.strip().lower()
    mask = df["title"].str.lower().str.contains(q, na=False)
    results = df[mask].head(limit)
    return results["title"].tolist()


def get_collaborative_recommendations(title: str,
                                      df: pd.DataFrame,
                                      collab_sim: np.ndarray,
                                      top_n: int = 10) -> list[dict]:
    if collab_sim is None:
        return []
    title_lower = title.strip().lower()
    matches = df[df["title"].str.lower() == title_lower]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(title_lower, na=False)]
    if matches.empty:
        return []
    idx = matches.index[0]
    scores = list(enumerate(collab_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]
    results = []
    for i, score in scores:
        row = df.iloc[i]
        results.append({
            "id": int(row.get("id", 0)),
            "title": row["title"],
            "vote_average": round(float(row.get("vote_average", 0)), 1),
            "genres": row["genres_list"][:3],
            "director": row["director"],
            "poster_path": row.get("poster_path", ""),
            "collaborative_score": round(float(score), 3),
            "overview": str(row["overview"])[:200] + "..." if len(str(row["overview"])) > 200 else str(row["overview"]),
        })
    return results


def _build_explanation(seed: pd.Series,
                       candidate: pd.Series,
                       content_score: float,
                       collaborative_score: float,
                       popularity_score: float,
                       sentiment_score: float) -> list[str]:
    reasons = []
    shared_genres = set(seed["genres_list"]).intersection(candidate["genres_list"])
    if shared_genres:
        reasons.append(f"Shared genres: {', '.join(sorted(shared_genres))}")
    if seed["director"] and seed["director"] == candidate["director"]:
        reasons.append("Same director")
    shared_cast = set(seed["cast_list"]).intersection(candidate["cast_list"])
    if shared_cast:
        reasons.append(f"Shared cast: {', '.join(sorted(shared_cast))}")
    if collaborative_score > 0.2:
        reasons.append("Popular among users with similar tastes")
    if popularity_score > 0.6:
        reasons.append("High popularity")
    if sentiment_score > 0.3:
        reasons.append("Strong positive viewer sentiment")
    if not reasons:
        reasons.append("Similar to your selected movie")
    return reasons


def get_hybrid_recommendations(title: str,
                               df: pd.DataFrame,
                               content_sim: np.ndarray,
                               collab_sim: np.ndarray | None = None,
                               popularity_scores: dict | None = None,
                               sentiment_scores: dict | None = None,
                               weights: dict | None = None,
                               top_n: int = 10) -> list[dict]:
    title_lower = title.strip().lower()
    matches = df[df["title"].str.lower() == title_lower]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(title_lower, na=False)]
    if matches.empty:
        return []
    idx = matches.index[0]
    n = len(df)
    if weights is None:
        weights = {
            "content": 0.55,
            "collaborative": 0.25,
            "popularity": 0.15,
            "sentiment": 0.05,
        }
    supported = {
        "content": content_sim is not None,
        "collaborative": collab_sim is not None,
        "popularity": bool(popularity_scores),
        "sentiment": bool(sentiment_scores),
    }
    active_weights = {k: v for k, v in weights.items() if supported.get(k, False)}
    if not active_weights:
        return get_recommendations(title, df, content_sim, top_n=top_n)
    total = sum(active_weights.values())
    active_weights = {k: v / total for k, v in active_weights.items()}
    content_scores = content_sim[idx] if content_sim is not None else np.zeros(n)
    collab_scores = collab_sim[idx] if collab_sim is not None else np.zeros(n)
    popularity_array = np.array([popularity_scores.get(int(mid), 0.0) for mid in df["id"].astype(int)]) if popularity_scores else np.zeros(n)
    sentiment_array = np.array([sentiment_scores.get(int(mid), 0.0) for mid in df["id"].astype(int)]) if sentiment_scores else np.zeros(n)
    combined = np.zeros(n)
    combined += active_weights.get("content", 0.0) * content_scores
    combined += active_weights.get("collaborative", 0.0) * collab_scores
    combined += active_weights.get("popularity", 0.0) * popularity_array
    combined += active_weights.get("sentiment", 0.0) * sentiment_array
    scores = list(enumerate(combined))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]
    seed = df.iloc[idx]
    results = []
    for i, score in scores:
        row = df.iloc[i]
        explanation = _build_explanation(
            seed,
            row,
            float(content_scores[i]) if content_scores is not None else 0.0,
            float(collab_scores[i]) if collab_scores is not None else 0.0,
            float(popularity_array[i]) if popularity_scores else 0.0,
            float(sentiment_array[i]) if sentiment_scores else 0.0,
        )
        results.append({
            "id": int(row.get("id", 0)),
            "title": row["title"],
            "vote_average": round(float(row.get("vote_average", 0)), 1),
            "genres": row["genres_list"][:3],
            "director": row["director"],
            "poster_path": row.get("poster_path", ""),
            "similarity": round(float(content_scores[i]), 3),
            "collaborative_score": round(float(collab_scores[i]), 3) if collab_sim is not None else None,
            "popularity_score": round(float(popularity_array[i]), 3) if popularity_scores else None,
            "sentiment_score": round(float(sentiment_array[i]), 3) if sentiment_scores else None,
            "combined_score": round(float(score), 4),
            "explanation": explanation,
            "overview": str(row["overview"])[:200] + "..." if len(str(row["overview"])) > 200 else str(row["overview"]),
        })
    return results


def _binary_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not recommended or k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def _binary_recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def _average_precision(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def _ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_model(ratings_df: pd.DataFrame,
                   df: pd.DataFrame,
                   content_sim: np.ndarray,
                   collab_sim: np.ndarray | None = None,
                   popularity_scores: dict | None = None,
                   sentiment_scores: dict | None = None,
                   k: int = 10,
                   sample_size: int = 100) -> dict:
    if ratings_df is None or ratings_df.empty:
        return {}
    merged = ratings_df.merge(df[["id", "title"]], left_on="movie_id", right_on="id")
    user_groups = merged.groupby("user_id")
    candidates = [user for user, group in user_groups if len(group) > 1]
    if not candidates:
        return {}
    sample_size = min(sample_size, len(candidates))
    selected = np.random.choice(candidates, sample_size, replace=False)
    precisions = []
    recalls = []
    aps = []
    ndcgs = []
    for user_id in selected:
        group = user_groups.get_group(user_id)
        if "timestamp" in group.columns:
            group = group.sort_values(by="timestamp")
        if len(group) < 2:
            continue
        seed = group.iloc[0]
        holdout = group.iloc[-1]
        recommendations = get_hybrid_recommendations(
            seed["title"],
            df,
            content_sim,
            collab_sim=collab_sim,
            popularity_scores=popularity_scores,
            sentiment_scores=sentiment_scores,
            top_n=k * 3,
        )
        recommended_ids = [item["id"] for item in recommendations]
        relevant = {int(holdout["movie_id"])}
        precisions.append(_binary_precision_at_k(recommended_ids, relevant, k))
        recalls.append(_binary_recall_at_k(recommended_ids, relevant, k))
        aps.append(_average_precision(recommended_ids, relevant, k))
        ndcgs.append(_ndcg_at_k(recommended_ids, relevant, k))
    if not precisions:
        return {}
    return {
        "precision_at_k": round(float(np.mean(precisions)), 4),
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "map_at_k": round(float(np.mean(aps)), 4),
        "ndcg_at_k": round(float(np.mean(ndcgs)), 4),
        "evaluator_sample_size": len(precisions),
        "requested_sample_size": sample_size,
    }


# ──────────────────────────────────────────────
# 5. SAVE / LOAD MODEL
# ──────────────────────────────────────────────

def save_model(sim_matrix, tfidf, df, out_dir: str = "models",
               collab_sim: np.ndarray | None = None,
               popularity_scores: dict | None = None,
               sentiment_scores: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/similarity.pkl", "wb") as f:
        pickle.dump(sim_matrix, f)
    with open(f"{out_dir}/tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    df.to_pickle(f"{out_dir}/movies.pkl")
    if collab_sim is not None:
        with open(f"{out_dir}/collaborative.pkl", "wb") as f:
            pickle.dump(collab_sim, f)
    if popularity_scores is not None:
        with open(f"{out_dir}/popularity.pkl", "wb") as f:
            pickle.dump(popularity_scores, f)
    if sentiment_scores is not None:
        with open(f"{out_dir}/sentiment.pkl", "wb") as f:
            pickle.dump(sentiment_scores, f)
    print(f"Model saved to {out_dir}/")


def load_model(model_dir: str = "models"):
    with open(f"{model_dir}/similarity.pkl", "rb") as f:
        sim_matrix = pickle.load(f)
    with open(f"{model_dir}/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    df = pd.read_pickle(f"{model_dir}/movies.pkl")
    return sim_matrix, tfidf, df


def load_full_model(model_dir: str = "models") -> tuple:
    sim_matrix, tfidf, df = load_model(model_dir)
    collab_sim = None
    popularity_scores = None
    sentiment_scores = None
    if os.path.exists(f"{model_dir}/collaborative.pkl"):
        with open(f"{model_dir}/collaborative.pkl", "rb") as f:
            collab_sim = pickle.load(f)
    if os.path.exists(f"{model_dir}/popularity.pkl"):
        with open(f"{model_dir}/popularity.pkl", "rb") as f:
            popularity_scores = pickle.load(f)
    if os.path.exists(f"{model_dir}/sentiment.pkl"):
        with open(f"{model_dir}/sentiment.pkl", "rb") as f:
            sentiment_scores = pickle.load(f)
    return sim_matrix, tfidf, df, collab_sim, popularity_scores, sentiment_scores


# ──────────────────────────────────────────────
# 6. ENTRYPOINT (train & save)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "../data"
    ratings_path = os.path.join(DATA_DIR, "user_ratings.csv")
    reviews_path = os.path.join(DATA_DIR, "movie_reviews.csv")

    print("Loading data...")
    df_raw = load_data(
        os.path.join(DATA_DIR, "tmdb_5000_movies.csv"),
        os.path.join(DATA_DIR, "tmdb_5000_credits.csv"),
    )

    print("Preprocessing...")
    df_processed = preprocess(df_raw)
    print(f"Dataset: {len(df_processed)} movies")

    print("Training content model...")
    sim_matrix, tfidf, df_final = train(df_processed)

    ratings_df = load_ratings(ratings_path)
    review_df = load_reviews(reviews_path)
    collab_sim = None
    popularity_scores = compute_popularity(df_final, ratings_df)
    sentiment_scores = compute_review_sentiment(review_df)
    if ratings_df is not None:
        print("Training collaborative filter...")
        collab_sim = train_collaborative(ratings_df, df_final)

    print("Saving model...")
    save_model(
        sim_matrix,
        tfidf,
        df_final,
        out_dir="../models",
        collab_sim=collab_sim,
        popularity_scores=popularity_scores,
        sentiment_scores=sentiment_scores,
    )

    recs = get_hybrid_recommendations(
        "The Dark Knight",
        df_final,
        sim_matrix,
        collab_sim=collab_sim,
        popularity_scores=popularity_scores,
        sentiment_scores=sentiment_scores,
        top_n=5,
    )
    print("\nTop 5 hybrid recommendations for 'The Dark Knight':")
    for r in recs:
        print(f"  {r['title']} (combined={r['combined_score']})")
