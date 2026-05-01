# app.py
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    movies = joblib.load("movies.pkl")
    tfidf = joblib.load("tfidf.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    indices = joblib.load("indices.pkl")
    return movies,tfidf,tfidf_matrix, indices

movies, tfidf, tfidf_matrix, indices = load_data()

# ---------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(movie_title, top_k=5):

    matches = movies[
        movies["title"].str.contains(
            movie_title,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:
        return []

    exact_title = matches.iloc[0]["title"]
    idx = indices[exact_title]

    # compute only one row similarity
    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    movie_scores = list(enumerate(sim_scores))

    movie_scores = sorted(
        movie_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:top_k+1]

    movie_indices = [i[0] for i in movie_scores]

    return movies["title"].iloc[movie_indices].tolist()
# ---------------------------------------------------
# GENRE FUNCTION
# ---------------------------------------------------
def top_by_genre(selected_genres, top_n=5):

    if "genres_list" not in movies.columns:
        movies["genres_list"] = movies["genres"].str.split("|")

    filtered = movies[
        movies["genres_list"].apply(
            lambda g: all(x in g for x in selected_genres)
        )
    ]

    return filtered["title"].head(top_n).tolist()


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("🎬 Movie Recommender")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Recommend by Movie",
        "Recommend by Genre",
        "About"
    ]
)

# ---------------------------------------------------
# HOME
# ---------------------------------------------------
if menu == "Home":

    st.title("🎥 Movie Recommendation Platform")
    st.write("Welcome to your MovieLens Recommendation System.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Movies", f"{movies.shape[0]:,}")
    col2.metric("Features", "Similarity + Genres")
    col3.metric("Dataset", "MovieLens")

    st.image(
        "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=1200",
        use_container_width=True
    )

# ---------------------------------------------------
# MOVIE SEARCH
# ---------------------------------------------------
elif menu == "Recommend by Movie":

    st.title("🎯 Recommend by Movie Name")

    movie_name = st.text_input("Enter movie title")

    top_n = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Recommend"):

        if movie_name.strip() == "":
            st.warning("Please enter a movie name.")
        else:
            result = recommend_movies(movie_name, top_n)

            if result:
                st.success("Recommendations Found")
                for i, movie in enumerate(result, 1):
                    st.write(f"{i}. {movie}")
            else:
                st.error("Movie not found.")

# ---------------------------------------------------
# GENRE SEARCH
# ---------------------------------------------------
elif menu == "Recommend by Genre":

    st.title("🎭 Recommend by Genre")

    genre_list = [
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"
    ]

    selected = st.multiselect(
        "Choose genres",
        genre_list
    )

    top_n = st.slider("Top results", 1, 10, 5)

    if st.button("Search Genre"):

        if not selected:
            st.warning("Select at least one genre.")
        else:
            result = top_by_genre(selected, top_n)

            if result:
                st.success("Movies Found")
                for i, movie in enumerate(result, 1):
                    st.write(f"{i}. {movie}")
            else:
                st.error("No results found.")

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------
else:

    st.title("📘 About Project")

    st.write("""
    This Movie Recommendation System was built using:

    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorizer
    - Cosine Similarity
    - Streamlit

    Dataset:
    MovieLens 20M
    """)

    