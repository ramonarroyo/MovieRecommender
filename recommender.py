#! python
# Use a reduced dataset created from movies.py to recommend movies.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        return ""


def create_soup(x):
    return ''.join(x['actors']) + ' ' + ''.join(x['director']) + ' ' + ''.join(x['genres'])


def get_recommendations(title, indices, cosine_sim, df):
    idx = indices[title]
    # Get pairwise similarity score of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get scores of the top 10 similar movies
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]


def main(dataset):
    movie = input("What movie would you like a recommendation for? ")
    df = pd.read_csv(dataset, sep=',', encoding='utf-8', usecols=['title', 'director', 'genres', 'score', 'actors'])

    if movie not in list(df['title']):
        print('This movie is not in the dataset.')
        return None
    
    # Give director double weight for more accurate recommendations
    df['director'] = df['director'].apply(lambda x: [x, x])

    series = ['actors', 'genres', 'director']
    for column in series:
        df[column] = df[column].astype('str').apply(clean_data)

    df['soup'] = df.apply(create_soup, axis=1)

    # Create a count matrix and calculate the cosine similarities
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    df = df.reset_index()
    indices = pd.Series(df.index, index=df['title'])
    recommendations = get_recommendations(movie, indices, cosine_sim2, df)
    print(recommendations)


if __name__ == "__main__":
    main("movies_10.csv")
