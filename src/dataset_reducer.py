"""Utilities for reducing the large IMDb dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


class MovieDatasetReducer:
    """Reduce raw IMDb TSV dumps to a smaller CSV dataset."""

    @staticmethod
    def weighted_rating(row: pd.Series, C: float, m: float) -> float:
        """Compute IMDb weighted rating for a single row."""
        v = row["numVotes"]
        R = row["averageRating"]
        return (v / (v + m) * R) + (m / (m + v) * C)

    def reduce_dataset(
        self, percentage: float, output_name: str | Path
    ) -> pd.DataFrame:
        """Reduce the raw IMDb dump.

        Parameters
        ----------
        percentage:
            Quantile of vote counts to retain. ``0.90`` keeps roughly the top
            10%% of movies by vote count.
        output_name:
            Path prefix for the CSV file that will be written.
        """
        logger.info("Getting movie dataset...")
        title = pd.read_csv(
            "title.basics.tsv",
            low_memory=False,
            chunksize=1000000,
            sep="\t",
            encoding="utf-8",
            usecols=["tconst", "titleType", "primaryTitle", "genres"],
        )
        director = pd.read_csv(
            "title.crew.tsv",
            low_memory=False,
            chunksize=1000000,
            sep="\t",
            encoding="utf-8",
            usecols=["tconst", "directors"],
        )
        ratings = pd.read_csv("title.ratings.tsv", sep="\t", encoding="utf-8")

        metadata = pd.concat(title)
        director = pd.concat(director)
        metadata = metadata.loc[metadata["titleType"] == "movie"]

        metadata = metadata.merge(ratings, on="tconst")
        metadata = metadata.merge(director, on="tconst")
        title, ratings, director = None, None, None

        metadata["directors"] = metadata["directors"].apply(lambda x: x[:9])

        logger.info(
            "Reducing data to %s%% of most popular movies.",
            round((1 - percentage) * 100),
        )
        C = metadata["averageRating"].mean()
        m = metadata["numVotes"].quantile(percentage)
        metadata = metadata.loc[metadata["numVotes"] >= m]
        metadata["score"] = metadata.apply(self.weighted_rating, C=C, m=m, axis=1)

        logger.info("Getting movie directors...")
        names = pd.read_csv(
            "name.basics.tsv",
            low_memory=False,
            chunksize=1000000,
            sep="\t",
            encoding="utf-8",
            usecols=["nconst", "primaryName"],
        )
        names = pd.concat(names)
        metadata = metadata.merge(
            names, how="left", left_on="directors", right_on="nconst"
        )

        series = ["directors", "nconst", "averageRating", "numVotes"]
        for column in series:
            del metadata[column]

        metadata.rename(columns={"primaryName": "director"}, inplace=True)

        logger.info("Getting movie cast members...")
        cast = pd.read_csv(
            "title.principals.tsv",
            low_memory=False,
            chunksize=1000000,
            sep="\t",
            encoding="utf-8",
            usecols=["tconst", "nconst", "category"],
        )
        cast = pd.concat(cast)
        cast = cast.loc[cast["category"] == "actor"]
        metadata = metadata.merge(cast, how="left", on="tconst")
        metadata = metadata.merge(names, how="left", on="nconst")
        cast, names = None, None

        metadata.rename(columns={"primaryName": "actors"}, inplace=True)
        metadata.rename(columns={"primaryTitle": "title"}, inplace=True)

        metadata = (
            metadata.astype(str)
            .groupby(["title", "director", "genres", "score"])["actors"]
            .apply(list)
            .reset_index()
        )

        metadata["genres"] = (
            metadata["genres"].astype(str).apply(lambda x: x.split(","))
        )

        metadata = metadata.sort_values("score", ascending=False)

        output_path = Path(f"{output_name}.csv")
        metadata.to_csv(output_path)
        logger.info("Saved dataset to %s", output_path)
        return metadata
