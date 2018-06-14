#! python
# movies.py creates a reduced dataframe for faster recommendation of movies without having to do this process repeatedly

import pandas as pd

def weighted_rating(x, C, m):
    v = x['numVotes']
    R = x['averageRating']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)

def get_movies(percentage, name):
    print("Getting movie dataset...")
    title = pd.read_csv('title.basics.tsv', low_memory=False, chunksize=1000000, sep='\t',
                        encoding='utf-8', usecols=['tconst', 'titleType', 'primaryTitle', 'genres'])
    director = pd.read_csv('title.crew.tsv', low_memory=False, chunksize=1000000, sep='\t', encoding='utf-8',
                         usecols=['tconst', 'directors'])
    ratings = pd.read_csv('title.ratings.tsv', sep='\t', encoding='utf-8')

    metadata = pd.concat(title)
    director = pd.concat(director)
    metadata = metadata.loc[metadata['titleType'] == 'movie']

    metadata = metadata.merge(ratings, on='tconst')
    metadata = metadata.merge(director, on='tconst')
    title, ratings, director = None, None, None  # Clear variables to save memory

    metadata['directors'] = metadata['directors'].apply(lambda x: x[:9])

    print("Reducing data to {}% of most popular movies.".format(round((1-percentage) * 100)))
    C = metadata['averageRating'].mean()
    m = metadata['numVotes'].quantile(percentage)
    metadata = metadata.loc[metadata['numVotes'] >= m]
    metadata['score'] = metadata.apply(weighted_rating, C=C, m=m, axis=1)

    print("Getting movie directors...")
    names = pd.read_csv('name.basics.tsv', low_memory=False, chunksize=1000000, sep='\t',
                        encoding='utf-8', usecols=['nconst', 'primaryName'])
    names = pd.concat(names)
    metadata = metadata.merge(names, how='left', left_on='directors', right_on='nconst')

    # Remove unnecessary series to save space
    series = ['directors', 'nconst', 'averageRating', 'numVotes']
    for column in series:
        del metadata[column]

    # Change name of column because we will introduce data with the same column name
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

    # Group all the actors per movie into a list in a single row
    metadata = metadata.astype(str).groupby(['title', 'director', 'genres', 'score']
                                            )['actors'].apply(list).reset_index()

    metadata['genres'] = metadata['genres'].astype(str).apply(lambda x: x.split(','))

    # Sort movies from highest score to lowest
    metadata = metadata.sort_values('score', ascending=False)

    metadata.to_csv(name + ".csv")
    print("Saved.")
    return metadata


if __name__ == "__main__":
    get_movies(0.90, "movies_10")
