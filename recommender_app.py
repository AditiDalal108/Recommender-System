import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from cmfrec import CMF

import warnings

warnings.filterwarnings('ignore')



# Load the pre-trained model and feature engineered data

import zipfile
import os

zip_path = "models.zip"
extract_path = "models/"
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print("ModelsUnzipped successfully!")

zip_path = "datasets.zip"
extract_path = "datasets/"
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print("Data Unzipped successfully!")


with open('models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('models/mf_model.pkl', 'rb') as f:
    mf_model = pickle.load(f)

with open('datasets/mf_ratings.pkl', 'rb') as f:
    mf_ratings = pickle.load(f)

with open('datasets/movie_info.pkl', 'rb') as f:
    movie_info = pickle.load(f)

movies = pd.read_csv(
    'datasets/movies.dat',
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movie_id", "title", "genres"])

movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
movies["title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
movies['genres'] = movies['genres'].str.split('|')

# Function to get 'similar movies' based on KNN model

def similar_movies(name, n = 10):
    idx = movies[movies['title']==name].index[0]
    distances, indices = knn_model.kneighbors(movie_info.iloc[idx].values.reshape(1,-1), n_neighbors = n+1)
    similar_movies = movies.iloc[indices[0][1:]]['title']

    return similar_movies

print(similar_movies("Toy Story"))

#Funtion to get 'recommendations based on user preferences' using Matrix Factorization model

def recommend_movies(movie, n=7):
    idx = movies[movies['title'] == movie]['movie_id'].values[0]
    movie_embedding = mf_model.B_[idx].reshape(1, -1)

    similarities = cosine_similarity(movie_embedding, mf_model.B_)

    similar_indices = similarities.argsort()[0][-n-1:-1][::-1]
    recommended_movies = movies[movies['movie_id'].isin(similar_indices)]['title'].values

    return recommended_movies



#Building the Streamlit App

img = Image.open("poster.png")

col1, col2 = st.columns(2)



with col1:
    st.title("Movie Recommender")
    st.write("Find your next favorite film")
with col2:
    st.image(img, width=500)

with st.container():
    st.header("Top Recommendations")

movie_list = sorted(movies['title'].unique())


st.sidebar.subheader("Select a movie you like:")
movie = st.sidebar.selectbox("Choose a movie", movie_list)

similar_btn = st.sidebar.button("Find Similar Movies")
recommend_btn = st.sidebar.button("Recommend What I'd Like")
reset_btn = st.sidebar.button("Reset")

if similar_btn:
    st.subheader(f"Movies similar to '{movie}':")
    similar = similar_movies(movie)
    for idx, sim_movie in enumerate(similar, 1):
        st.write(f"{sim_movie}")

if recommend_btn:
    st.subheader(f"Viewers who liked '{movie}' also liked:")
    recommended = recommend_movies(movie)
    for idx, rec_movie in enumerate(recommended, 1):
        st.write(f"{rec_movie}")
