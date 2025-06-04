import pandas as pd

class MovieDatasetReducer:
    """Reduce raw IMDb data to a smaller dataset."""

    @staticmethod
    def weighted_rating(x, C, m):
        v = x['numVotes']
        R = x['averageRating']
        return (v / (v + m) * R) + (m / (m + v) * C)

    def reduce_dataset(self, percentage, output_name):
        print("Getting movie dataset...")
        title = pd.read_csv('title.basics.tsv', low_memory=False, chunksize=1000000,
                            sep='\t', encoding='utf-8',
                            usecols=['tconst', 'titleType', 'primaryTitle', 'genres'])
        director = pd.read_csv('title.crew.tsv', low_memory=False, chunksize=1000000,
                               sep='\t', encoding='utf-8',
                               usecols=['tconst', 'directors'])
        ratings = pd.read_csv('title.ratings.tsv', sep='\t', encoding='utf-8')

        metadata = pd.concat(title)
        director = pd.concat(director)
        metadata = metadata.loc[metadata['titleType'] == 'movie']

        metadata = metadata.merge(ratings, on='tconst')
        metadata = metadata.merge(director, on='tconst')
        title, ratings, director = None, None, None

        metadata['directors'] = metadata['directors'].apply(lambda x: x[:9])

        print(f"Reducing data to {round((1 - percentage) * 100)}% of most popular movies.")
        C = metadata['averageRating'].mean()
        m = metadata['numVotes'].quantile(percentage)
        metadata = metadata.loc[metadata['numVotes'] >= m]
        metadata['score'] = metadata.apply(self.weighted_rating, C=C, m=m, axis=1)

        print("Getting movie directors...")
        names = pd.read_csv('name.basics.tsv', low_memory=False, chunksize=1000000,
                            sep='\t', encoding='utf-8', usecols=['nconst', 'primaryName'])
        names = pd.concat(names)
        metadata = metadata.merge(names, how='left', left_on='directors', right_on='nconst')

        series = ['directors', 'nconst', 'averageRating', 'numVotes']
        for column in series:
            del metadata[column]

        metadata.rename(columns={'primaryName': 'director'}, inplace=True)

        print("Getting movie cast members...")
        cast = pd.read_csv('title.principals.tsv', low_memory=False, chunksize=1000000, sep='\t',
                           encoding='utf-8', usecols=['tconst', 'nconst', 'category'])
        cast = pd.concat(cast)
        cast = cast.loc[cast['category'] == 'actor']
        metadata = metadata.merge(cast, how='left', on='tconst')
        metadata = metadata.merge(names, how='left', on='nconst')
        cast, names = None, None

        metadata.rename(columns={'primaryName': 'actors'}, inplace=True)
        metadata.rename(columns={'primaryTitle': 'title'}, inplace=True)

        metadata = metadata.astype(str).groupby(['title', 'director', 'genres', 'score'])['actors'].apply(list).reset_index()

        metadata['genres'] = metadata['genres'].astype(str).apply(lambda x: x.split(','))

        metadata = metadata.sort_values('score', ascending=False)

        metadata.to_csv(f"{output_name}.csv")
        print("Saved.")
        return metadata
