"""Views for the movie recommendation app."""

from functools import lru_cache
from pathlib import Path

from django.conf import settings
from django.shortcuts import render

from src.movie_recommender import MovieRecommender


@lru_cache(maxsize=1)
def _load_recommender(dataset: str) -> MovieRecommender:
    """Return a recommender with ``dataset`` loaded."""
    recommender = MovieRecommender()
    recommender.load_dataset(dataset)
    return recommender


def get_recommender() -> MovieRecommender:
    """Return a cached :class:`MovieRecommender` instance."""
    dataset_path = Path(settings.RECOMMENDER_DATASET_PATH)
    if not dataset_path.exists():
        raise ValueError("Dataset not found. Please run dataset reducer.")
    return _load_recommender(str(dataset_path))


def search(request):
    """Render the search form and show recommendations."""
    recommendations = None
    error = None
    if request.method == "POST":
        title = request.POST.get("title", "")
        if title:
            try:
                recommender = get_recommender()
                recommendations = recommender.recommend(title)
            except Exception as exc:
                error = str(exc)
    return render(
        request,
        "movies/search.html",
        {"recommendations": recommendations, "error": error},
    )
