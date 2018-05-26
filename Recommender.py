#! python

import pandas as pd
from time import time


print("Opening files.")
title_chunk = pd.read_csv('title.basics.tsv', low_memory=False, chunksize=1000000, sep='\t',
                     encoding='utf-8', usecols=['tconst', 'titleType', 'originalTitle', 'genres'])
ratings = pd.read_csv('title.ratings.tsv', sep='\t', encoding='utf-8')
director = pd.read_csv('title.crew.tsv', low_memory=False, chunksize=1000000, sep='\t', encoding='utf-8',
                   usecols=['tconst', 'directors'])
names = pd.read_csv('name.basics.tsv', low_memory=False, chunksize=1000000, sep='\t',
                     encoding='utf-8', usecols=['nconst', 'primaryName'])


print("Merging data.")
metadata = pd.concat(title_chunk)
director = pd.concat(director)
metadata = metadata.loc[metadata['titleType'] == 'movie']

metadata = metadata.merge(ratings, on='tconst')
metadata = metadata.merge(director, on='tconst')

def one_dir(x):
    return x[:9]

metadata['directors'] = metadata['directors'].apply(one_dir)

print("Getting the top 5% of rated movies.")
# Reduces the dataset to the top 5% of movies by score
C = metadata['averageRating'].mean()
m = metadata['numVotes'].quantile(0.95)
metadata = metadata.loc[metadata['numVotes'] >= m]

def weighted_rating(x, C=C, m=m):
    v = x['numVotes']
    R = x['averageRating']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

director = None

print("Concatenating names...")

names = pd.concat(names)

metadata['score'] = metadata.apply(weighted_rating, axis=1)
metadata = metadata.sort_values('score', ascending=False)

print("Getting director name.")

metadata = metadata.merge(names, how='inner', left_on='directors', right_on='nconst') #not working

print(metadata.shape)
print(metadata.head(5))