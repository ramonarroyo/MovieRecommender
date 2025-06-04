import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """Load movie data and generate recommendations."""

    def __init__(self):
        self.df = None
        self.indices = None
        self.cosine_sim = None

    @staticmethod
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        return ""

    @staticmethod
    def create_soup(x):
        return ''.join(x['actors']) + ' ' + ''.join(x['director']) + ' ' + ''.join(x['genres'])

    def load_dataset(self, dataset):
        df = pd.read_csv(dataset, sep=',', encoding='utf-8',
                         usecols=['title', 'director', 'genres', 'score', 'actors'])
        df['director'] = df['director'].apply(lambda x: [x, x])
        for column in ['actors', 'genres', 'director']:
            df[column] = df[column].astype('str').apply(self.clean_data)
        df['soup'] = df.apply(self.create_soup, axis=1)
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(df['soup'])
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        df = df.reset_index(drop=True)
        self.indices = pd.Series(df.index, index=df['title'])
        self.df = df
        return df

    def recommend(self, title, top_n=10):
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
        if title not in self.indices:
            raise ValueError("This movie is not in the dataset.")
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]
        return self.df['title'].iloc[movie_indices]
