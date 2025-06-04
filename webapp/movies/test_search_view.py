import os
import sys
import tempfile
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
import django
django.setup()
from django.test.utils import setup_test_environment
setup_test_environment()

import pandas as pd
from django.test import Client, TestCase, override_settings

from movies.views import _load_recommender


@override_settings(ALLOWED_HOSTS=["testserver"])
class SearchViewTest(TestCase):
    """Tests for the ``search`` view using the Django test client."""

    def setUp(self) -> None:
        _load_recommender.cache_clear()
        self.client = Client()

    def create_dataset(self) -> str:
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
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w+")
        df.to_csv(tmp.name, index=False)
        tmp.close()
        return tmp.name

    def test_search_returns_recommendations(self) -> None:
        """Posting a valid title should return movie recommendations."""
        dataset_path = self.create_dataset()
        try:
            with override_settings(RECOMMENDER_DATASET_PATH=dataset_path):
                response = self.client.post("/", {"title": "Movie A"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                list(response.context["recommendations"]),
                ["Movie B", "Movie C"],
            )
            self.assertIsNone(response.context["error"])
        finally:
            os.unlink(dataset_path)

    def test_search_missing_dataset_shows_error(self) -> None:
        """An error message should be shown when the dataset is missing."""
        missing_path = "/tmp/does_not_exist.csv"
        with override_settings(RECOMMENDER_DATASET_PATH=missing_path):
            response = self.client.post("/", {"title": "Movie A"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["recommendations"])
        self.assertIn("Dataset not found", response.context["error"])

