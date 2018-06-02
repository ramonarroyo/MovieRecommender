#! python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get movie to find recommendations for
movie = input("What movie would you like a recommendation for? ")

# Read movie data from TSV files, use chunks for files that are too big
print("Getting movie dataset...")
title = pd.read_csv('title.basics.tsv', low_memory=False, chunksize=1000000, sep='\t',
                    encoding='utf-8', usecols=['tconst', 'titleType', 'originalTitle', 'genres'])
ratings = pd.read_csv('title.ratings.tsv', sep='\t', encoding='utf-8')
director = pd.read_csv('title.crew.tsv', low_memory=False, chunksize=1000000, sep='\t', encoding='utf-8',
                       usecols=['tconst', 'directors'])

# Concatenate chunks into one dataframe and reduce data to only movies
metadata = pd.concat(title)
director = pd.concat(director)
metadata = metadata.loc[metadata['titleType'] == 'movie']

# Check if movie exists in data before continuing
if movie not in list(metadata['originalTitle']):
    raise KeyError('This movie is not in the dataset.')

# Merge the different dataframes into just one
metadata = metadata.merge(ratings, on='tconst')
metadata = metadata.merge(director, on='tconst')
ratings, director = None, None  # Clear variables to save memory

# Choose only the first director
metadata['directors'] = metadata['directors'].apply(lambda x: x[:9])

# Use IMDB weighted rating equation to grab the best rated movies
print("Reducing data to the top 5% of movies by rating.")
C = metadata['averageRating'].mean()
m = metadata['numVotes'].quantile(0.95)
metadata = metadata.loc[metadata['numVotes'] >= m]

def weighted_rating(x, C=C, m=m):
    v = x['numVotes']
    R = x['averageRating']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)

metadata['score'] = metadata.apply(weighted_rating, axis=1)

# Read movie cast data and merge
print("Getting movie directors...")
names = pd.read_csv('name.basics.tsv', low_memory=False, chunksize=1000000, sep='\t',
                    encoding='utf-8', usecols=['nconst', 'primaryName'])
names = pd.concat(names)
metadata = metadata.merge(names, how='left', left_on='directors', right_on='nconst')

# Remove unnecessary series to save space
series = ['directors', 'nconst', 'titleType', 'averageRating', 'numVotes']
for column in series:
    del metadata[column]

# Change name of column because we will introduce data with the same column name
metadata.rename(columns={'primaryName': 'director'}, inplace=True)

# Read and merge actor data now that the dataframe is smaller
print("Getting movie cast members...")
cast = pd.read_csv('title.principals.tsv', low_memory=False, chunksize=1000000, sep='\t',
                   encoding='utf-8', usecols=['tconst', 'nconst', 'category'])
cast = pd.concat(cast)
cast = cast.loc[cast['category'] == 'actor']
metadata = metadata.merge(cast, how='left', on='tconst')
metadata = metadata.merge(names, how='left', on='nconst')
cast, names = None, None

series = ['tconst', 'nconst', 'category']
for column in series:
    del metadata[column]

metadata.rename(columns={'primaryName': 'actors'}, inplace=True)

# Group all the actors per movie into a list in a single row
metadata = metadata.astype(str).groupby(['originalTitle', 'director', 'genres', 'score']
                                        )['actors'].apply(list).reset_index()

# Group all genres into a list
metadata['genres'] = metadata['genres'].astype(str).apply(lambda x: x.split(','))

# Sort movies from highest score to lowest
metadata = metadata.sort_values('score', ascending=False)

# Give the director twice the weight
metadata['director'] = metadata['director'].apply(lambda x: [x, x])

# Strip strings of spaces and convert lower case for more accurate recommendations
print("Cleaning data...")
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        return ""

series = ['actors', 'genres', 'director']
for column in series:
    metadata[column] = metadata[column].apply(clean_data)

# Create string with all the data to feed to the vectorizer
def create_soup(x):
    return ' '.join(x['actors']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genres'])

metadata['soup'] = metadata.apply(create_soup, axis=1)

# Create a count matrix and calculate the cosine similarities
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['originalTitle'])

def get_recommendations(title, cosine_sim=cosine_sim2):
    idx = indices[title]
    # Get pairwise similarity score of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get scores of the top 10 similar movies
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return metadata['originalTitle'].iloc[movie_indices]


print(get_recommendations(movie))
