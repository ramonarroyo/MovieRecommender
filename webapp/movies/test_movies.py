"""Unit tests for the movie recommendation utilities."""

from pathlib import Path
import os
import tempfile
import unittest

import pandas as pd

from src.dataset_reducer import MovieDatasetReducer
from src.movie_recommender import MovieRecommender


class MovieUtilsTest(unittest.TestCase):
    """Tests for :mod:`src.movie_recommender` and helpers."""

    def test_weighted_rating(self) -> None:
        """Weighted rating should combine average rating and vote count."""
        reducer = MovieDatasetReducer()
        row = {"numVotes": 100, "averageRating": 8.0}
        rating = reducer.weighted_rating(row, C=7.0, m=50)
        self.assertAlmostEqual(rating, 7.6667, places=4)

    def test_clean_data_and_create_soup(self) -> None:
        """Verify cleaning helpers and soup creation."""
        rec = MovieRecommender()
        self.assertEqual(rec.clean_data(["Actor X", "Actor Y"]), ["actorx", "actory"])
        self.assertEqual(rec.clean_data("Action Thriller"), "actionthriller")
        self.assertEqual(rec.clean_data(None), "")

        row = {
            "actors": ["actorx", "actory"],
            "director": ["directora", "directora"],
            "genres": ["action", "thriller"],
        }
        soup = rec.create_soup(row)
        self.assertEqual(soup, "actorxactory directoradirectora actionthriller")

    def test_load_and_recommend(self) -> None:
        """Loading a small dataset should enable recommendations."""
        rec = MovieRecommender()
        df = pd.DataFrame(
            {
                "title": ["Movie A", "Movie B", "Movie C"],
                "director": ["Director A", "Director B", "Director C"],
                "genres": ["Action", "Action", "Drama"],
                "score": [9.0, 8.0, 8.5],
                "actors": [
                    "Actor X Actor Y",
                    "Actor Y Actor Z",
                    "Actor X Actor W",
                ],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w+") as tmp:
            df.to_csv(tmp.name, index=False)
        try:
            rec.load_dataset(Path(tmp.name))
            result = rec.recommend("Movie A", top_n=2).tolist()
        finally:
            os.unlink(tmp.name)
        self.assertEqual(result, ["Movie B", "Movie C"])


if __name__ == "__main__":
    unittest.main()
