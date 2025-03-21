import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS

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
#Use SVD algorithm for collaborative filtering
    model = SVD(random_state=42)  # Set the random state for reproducibility
    cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    trainset = data.build_full_trainset()
    model.fit(trainset)

    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)



def get_movie_metadata(movie_id):
    movie = movies[movies['id'] == str(movie_id)].iloc[0]

    return {
        'title': movie['title'],
        'imdb_id': movie['imdb_id'],
        'imdb_url': f"https://www.imdb.com/title/{movie['imdb_id']}",
        'score': movie['vote_average'],
        'votes': movie['vote_count'],
        'genres': movie['genres']
    }


# API:

def get_default_recommendations(n_recommendations=5):
    default_movies = movies.sort_values('vote_average', ascending=False).head(n_recommendations)
    return [get_movie_metadata(movie_id) for movie_id in default_movies['id']]


app = Flask(__name__)
CORS(app)


@app.route('/recommend_movies', methods=['POST'])
def recommend_movies():
    n_recommendations = 5
    data = request.get_json()
    user_id = data.get('user_id')

    if user_id not in model.trainset.all_users():
        return jsonify(get_default_recommendations())

    predictions = [model.predict(user_id, int(movie_id)) for movie_id in movies['id'].astype(int)]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n_recommendations]
    recommendations = [get_movie_metadata(pred.iid) for pred in top_n]
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)