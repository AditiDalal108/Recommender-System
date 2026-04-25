# Recommender-System
A Netflix-style hybrid movie recommender system built using Matrix Factorization and Collaborative Filtering. Users can find 'Similar Movies' or get personalized "What to Watch Next" recommendations through an interactive Streamlit app.


## 🚀 App Live Demo

👉 : https://recommender-system-byaditidalal.streamlit.app/

## 🧠 Features

- 🔍 **Find Similar Movies**
  - Content-based filtering using movie features (genre, decade)

- 🎯 **What Should I Watch Next**
  - Hybrid recommender using:
    - Collaborative Filtering
    - Matrix Factorization (item embeddings)

- ⚡ Fast recommendations using precomputed embeddings

- 🎨 Clean and interactive UI built with Streamlit

## 🏗️ Tech Stack

- Python  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn  
- Matrix Factorization (`cmfrec`)

## 📊 Dataset

This project uses the **MovieLens 1M Dataset**

## 🧩 How It Works

- Takes a movie as input from the user as query point
  
1. 'Find Similar Movies'

- Uses Content-Based Filtering on movie features like genres, time of release etc
- Finds similar movies using KNN Modelling technique

2. 'Recommend What I'd Like'

- Uses a mix of Matrix Factorization and Collaborative Filtering
- Learns latent embeddings for movies based on user ratings via Matrix factorization
- Finds movies that were liked by other users with similar tastes and preferences,
  using Cosine Similarity of the movie embeddings

3. Hybrid Recommendation

- Combines:
  Item similarity (content-based)
  Embedding similarity (collaborative)


## 📸 App screenshot
<img width="1553" height="846" alt="image" src="https://github.com/user-attachments/assets/ac1748f5-afd4-4421-9fc3-18d7fb75e152" />

## 🔮 Future Improvements

If I were to extend this project:

- Add movie posters for better UX
- Introduce user profiles for personalized recommendations
- Deploy with a database backend
- Improve ranking with weighted scoring and bringing more factors into play
- Incorporate deep learning models

