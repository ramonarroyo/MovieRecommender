#! python
"""Command line interface for recommending movies."""

import argparse
from src import MovieRecommender


def main():
    parser = argparse.ArgumentParser(description="Recommend movies from a dataset")
    parser.add_argument('dataset', help='CSV dataset produced by movies.py')
    parser.add_argument('title', nargs='?', help='Movie title to search for')
    args = parser.parse_args()

    recommender = MovieRecommender()
    recommender.load_dataset(args.dataset)

    title = args.title or input("What movie would you like a recommendation for? ")
    try:
        recommendations = recommender.recommend(title)
        print(recommendations)
    except ValueError as exc:
        print(exc)


if __name__ == "__main__":
    main()
