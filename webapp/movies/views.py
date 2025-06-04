import os
from django.shortcuts import render
from src.movie_recommender import MovieRecommender

recommender = MovieRecommender()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'movies_10.csv')


def search(request):
    recommendations = None
    error = None
    if request.method == 'POST':
        title = request.POST.get('title', '')
        if title:
            try:
                if recommender.df is None:
                    if os.path.exists(DATASET_PATH):
                        recommender.load_dataset(DATASET_PATH)
                    else:
                        raise ValueError('Dataset not found. Please run dataset reducer.')
                recommendations = recommender.recommend(title)
            except Exception as exc:
                error = str(exc)
    return render(request, 'movies/search.html', {'recommendations': recommendations, 'error': error})
