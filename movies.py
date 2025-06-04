#! python
"""Command line interface for reducing the movie dataset."""

import argparse
import logging

from src import MovieDatasetReducer


def main():
    parser = argparse.ArgumentParser(description="Create a reduced movie dataset")
    parser.add_argument(
        "-p",
        "--percentage",
        type=float,
        default=0.90,
        help="Quantile of votes to keep (e.g. 0.90 keeps top 10% of movies)",
    )
    parser.add_argument(
        "-o", "--output", default="movies_10", help="Output filename without extension"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    reducer = MovieDatasetReducer()
    reducer.reduce_dataset(args.percentage, args.output)


if __name__ == "__main__":
    main()
