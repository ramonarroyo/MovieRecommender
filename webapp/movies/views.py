from pathlib import Path

from django.shortcuts import render

from src.movie_recommender import MovieRecommender

recommender = MovieRecommender()
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "movies_10.csv"


def search(request):
    """Render the search form and show recommendations."""
    recommendations = None
    error = None
    if request.method == "POST":
        title = request.POST.get("title", "")
        if title:
            try:
                if recommender.df is None:
                    if DATASET_PATH.exists():
                        recommender.load_dataset(DATASET_PATH)
                    else:
                        raise ValueError(
                            "Dataset not found. Please run dataset reducer."
                        )
                recommendations = recommender.recommend(title)
            except Exception as exc:
                error = str(exc)
    return render(
        request,
        "movies/search.html",
        {"recommendations": recommendations, "error": error},
    )
