#! python3


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from ast import literal_eval

movie = input("What movie would you like a recommendation for? ")

metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
credit = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# Check if movie exists in data before continuing
if movie not in list(metadata['title']):
    raise KeyError('This movie is not in the database.')

# Remove bad data
metadata = metadata.drop([19730, 29503, 35587])

# Converts IDs to int, required for merging
metadata['id'] = metadata['id'].astype('int')
credit['id'] = credit['id'].astype('int')
keywords['id'] = keywords['id'].astype('int')

metadata = metadata.merge(credit, on='id')
metadata = metadata.merge(keywords, on='id')

# Reduces the dataset to the top 10% of movies by score
C = metadata['vote_average'].mean()
m = metadata['vote_count'].quantile(0.90)
metadata= metadata.loc[metadata['vote_count'] >= m]

def weighted_rating(x, C=C, m=m):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

metadata['score'] = metadata.apply(weighted_rating, axis=1)
metadata = metadata.sort_values('score', ascending=False)

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

def get_director(cast):
    for employee in cast:
        if employee['job'] == 'Director':
            return employee['name']
    return np.nan

metadata['director'] = metadata['crew'].apply(get_director)
# Give the director twice the weight
metadata['director'] = metadata['director'].apply(lambda x: [x, x])

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# Convert keywords to their stem to more easily match them
stemmer = SnowballStemmer('english')
metadata['keywords'] = metadata['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])

# Strip strings of spaces and convert lower case
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        return ""

features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast'])\
           + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genres'])

metadata['soup'] = metadata.apply(create_soup, axis=1)

# Create a count matrix and calculate the cosine similarities

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])

def get_recommendations(title, cosine_sim=cosine_sim2):
    idx = indices[title]
    # Get pairwise similarity score of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x : x[1], reverse=True)
    # Get scores of the top 10 similar movies
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]

print(get_recommendations(movie))
