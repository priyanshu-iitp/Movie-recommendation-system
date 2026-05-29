from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movies data
movies = pickle.load(open('movies.pkl', 'rb'))

# Generate similarity matrix dynamically
similarity = cosine_similarity(movies.drop(['title'], axis=1))


def recommend(movie):

    matched_movies = movies[
        movies['title'].str.lower() == movie.lower()
    ]

    if matched_movies.empty:
        return ["Movie not found"]

    movie_index = matched_movies.index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(
            movies.iloc[i[0]].title
        )

    return recommended_movies


@app.route('/', methods=['GET', 'POST'])
def home():

    recommendations = []

    if request.method == 'POST':

        movie_name = request.form['movie']

        recommendations = recommend(movie_name)

    return render_template(
        'index.html',
        movie_list=sorted(movies['title'].values),
        recommendations=recommendations
    )


if __name__ == '__main__':
    app.run(debug=True)
