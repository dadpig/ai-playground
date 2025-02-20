import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import pickle

movies_path = os.path.join('..', 'dataset', 'movies_metadata.csv')
ratings_path = os.path.join('..', 'dataset', 'ratings.csv')
model_path = 'trained_model.pkl'

def load_dataset_for_training():
    print("Start loading datasets for training.")
    movies = pd.read_csv(movies_path, low_memory=False)
    ratings = pd.read_csv(ratings_path)
    ratings = ratings.dropna()
    movies = movies.dropna()
    ratings['movieId'] = ratings['movieId'].astype(int)
    print("Finished loading datasets for training.")
    return movies, ratings

# Check if the model file exists (already trained):
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    movies = pd.read_csv(movies_path, low_memory=False)
    movies = movies.dropna()
    print("Model loaded from file.")
else:
    movies, ratings = load_dataset_for_training()
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    model = SVD()
    cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    trainset = data.build_full_trainset()
    model.fit(trainset)

    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

def get_movie_metadata(movie_id):
    movie = movies[movies['id'] == str(movie_id)].iloc[0]
    return f"Title: {movie['title']}, Genres: {movie['genres']}"

def recommend_movies(user_id, model, n_recommendations=5):
    predictions = [model.predict(user_id, int(movie_id)) for movie_id in movies['id'].astype(int)]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = predictions[:n_recommendations]
    recommendations = [get_movie_metadata(pred.iid) for pred in top_n]

    return recommendations

user_id = 1
recommendations = recommend_movies(user_id, model)
for rec in recommendations:
    print(rec)
