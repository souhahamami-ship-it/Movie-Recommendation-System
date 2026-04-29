import streamlit as st
import joblib
import pandas as pd

# Load saved model
movies = joblib.load("movies.pkl")
tfidf = joblib.load("tfidf.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
indices = joblib.load("indices.pkl")



from sklearn.metrics.pairwise import cosine_similarity

def recommend(movie_title, top_k=5):
    matches = movies[movies['title'].str.contains(
        movie_title, case=False, na=False, regex=False
    )]

    if matches.empty:
        return []

    exact_title = matches.iloc[0]['title']
    idx = indices[exact_title]

    # Compute similarity only for selected movie
    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Sort scores
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # REMOVE the same movie
    sim_scores = [i for i in sim_scores if i[0] != idx]
    sim_scores = sim_scores[:top_k]


    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices].tolist()

# UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite one!")

movie_name = st.text_input("Enter movie name:")

top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend"):
    results = recommend(movie_name, top_k)

    if results:
        st.success("Recommended Movies:")
        for i, movie in enumerate(results, 1):
            st.write(f"{i}. {movie}")
    else:
        st.error("Movie not found.")