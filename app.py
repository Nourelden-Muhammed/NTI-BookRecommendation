import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

import pandas as pd

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score
from sklearn.manifold import TSNE

from scipy.sparse import csr_matrix

import pickle

# Data Preprocessing

books_df = pd.read_csv("\Dataset\Books.csv", low_memory=False)
ratings_df = pd.read_csv("\Dataset\Ratings.csv")
users_df = pd.read_csv("\Dataset\Users.csv")

 # Books DataFrame Preprocessing
books_df["Book-Author"].fillna("Unknown", inplace=True)
books_df["Publisher"].fillna("Unknown", inplace=True)
books_df["Image-URL-L"].fillna("No Image", inplace=True)
books_df["Year-Of-Publication"] = pd.to_numeric(books_df["Year-Of-Publication"], errors='coerce')
books_df["Year-Of-Publication"].fillna(books_df["Year-Of-Publication"].median(), inplace=True)
books_df["Year-Of-Publication"] = books_df["Year-Of-Publication"].astype(int)

books_df.drop(columns=["Image-URL-S", "Image-URL-M"], inplace=True)

# Users DataFrame Preprocessing
users_df["Age"].fillna(users_df["Age"].median(), inplace=True)
users_df["Age"] = users_df["Age"].astype(int)

# Ratings DataFrame Preprocessing
explicit_ratings_df = ratings_df[ratings_df["Book-Rating"] != 0]

# Merge datasets
merged_df = explicit_ratings_df.merge(books_df, on="ISBN")

# Filter users and books with at least 10 ratings
ratings_count_per_user = merged_df["User-ID"].value_counts()
users_to_keep = ratings_count_per_user[ratings_count_per_user >= 10].index
filtered_users_df = merged_df[merged_df["User-ID"].isin(users_to_keep)]
ratings_count_per_book = filtered_users_df["Book-Title"].value_counts()
books_to_keep = ratings_count_per_book[ratings_count_per_book >= 10].index
final_df = filtered_users_df[filtered_users_df["Book-Title"].isin(books_to_keep)]

# Save preprocessed data
final_df.to_csv("preprocessed_data.csv", index=False)
print("Preprocessed data saved to preprocessed_data.csv")

# Create a pivot table and sparse matrix for the recommendation system.

book_pivot = final_df.pivot_table(columns='User-ID', index='Book-Title', values='Book-Rating').fillna(0)
book_sparse_matrix = csr_matrix(book_pivot)

# Save pivot table and sparse matrix
pd.to_pickle(book_pivot, "book_pivot.pkl")
pd.to_pickle(book_sparse_matrix, "book_sparse_matrix.pkl")
print("Pivot table and sparse matrix saved.")

# Train the KNN model using cosine similarity.

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(book_sparse_matrix)

# Save the trained model
with open('model_knn.pkl', 'wb') as f:
    pickle.dump(model_knn, f)
print("KNN model saved to model_knn.pkl")

# Save book titles
with open('book_titles.pkl', 'wb') as f:
    pickle.dump(book_pivot.index, f)
print("Book titles saved to book_titles.pkl")




# Set page configuration as the FIRST Streamlit command
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide"
)

# Define file paths
BASE_PATH = r"\Dataset"
FILES = {
    "books": os.path.join(BASE_PATH, "Books.csv"),
    "ratings": os.path.join(BASE_PATH, "Ratings.csv"),
    "book_pivot": "book_pivot.pkl",
    "model_knn":  "model_knn.pkl",
    "book_titles": "book_titles.pkl"
}

# Load preprocessed data and models
try:
    book_pivot = pd.read_pickle(FILES["book_pivot"])
    model_knn = pickle.load(open(FILES["model_knn"], "rb"))
    book_titles = pickle.load(open(FILES["book_titles"], "rb"))
    books_df = pd.read_csv(FILES["books"])
    ratings_df = pd.read_csv(FILES["ratings"])
except FileNotFoundError as e:
    st.error(f"Error: {e}. Ensure the following files are in {BASE_PATH}: Books.csv, Ratings.csv, book_pivot.pkl, model_knn.pkl, book_titles.pkl")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading files: {e}")
    st.stop()

# Merge book titles with image URLs and author
books_df = books_df[["ISBN", "Book-Title", "Book-Author", "Image-URL-L"]].drop_duplicates(subset="Book-Title")
book_pivot_reset = book_pivot.reset_index()[["Book-Title"]]
book_info = book_pivot_reset.merge(books_df, on="Book-Title", how="left")

# Function to get top 20 books by number of ratings
@st.cache_data
def get_top_20_books(ratings_df, books_df):
    ratings_df = ratings_df[ratings_df["Book-Rating"] != 0]  # Explicit ratings only
    top_books = ratings_df.merge(books_df, on="ISBN").groupby("Book-Title").agg({
        "Book-Rating": "count",
        "Book-Author": "first",
        "Image-URL-L": "first"
    }).rename(columns={"Book-Rating": "num_ratings"}).reset_index()
    top_books = top_books.sort_values("num_ratings", ascending=False).head(20).reset_index(drop=True)
    return top_books

# Function to recommend books
def recommend_books(book_name, pivot_table, model, num_recommendations=5):
    if book_name not in pivot_table.index:
        return None, []
    book_id = pivot_table.index.get_loc(book_name)
    distances, indices = model.kneighbors(pivot_table.iloc[book_id,:].values.reshape(1, -1), n_neighbors=num_recommendations+1)
    recommendations = []
    for i in range(1, len(distances.flatten())):
        title = pivot_table.index[indices.flatten()[i]]
        info = book_info[book_info["Book-Title"] == title]
        if not info.empty:
            recommendations.append({
                "title": title,
                "author": info["Book-Author"].values[0] if not pd.isna(info["Book-Author"].values[0]) else "Unknown",
                "image_url": info["Image-URL-L"].values[0] if not pd.isna(info["Image-URL-L"].values[0]) else "No Image"
            })
    return f"Recommendations for '{book_name}'", recommendations

# Main function to render the app
def main():
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
      #  selected_mode = st.radio("Select Mode", ["Home", "Recommender"])
        option = st.sidebar.selectbox("Choose an option:",
                              ["Top 20 Books", "Get Recommendations"])

    # Home Page
    if option == "Top 20 Books":
        st.title("📚 Book Recommender System")
        st.markdown("Welcome to our Book Recommender System! Discover top-rated books or find personalized recommendations.")

        st.subheader("Top 20 Most Rated Books")
        top_books = get_top_20_books(ratings_df, books_df)

        # Create a grid layout with 4 columns
        cols = st.columns(4, gap="medium")
        for idx, row in top_books.iterrows():
            col = cols[idx % 4]
            with col:
                with st.container():
                    st.markdown(f'<div class="book-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="book-rank">#{idx + 1}</div>', unsafe_allow_html=True)
                    if row["Image-URL-L"] and row["Image-URL-L"] != "No Image":
                        try:
                            st.image(row["Image-URL-L"], width=120, caption=row["Book-Title"][:25] + "..." if len(row["Book-Title"]) > 25 else row["Book-Title"])
                        except Exception:
                            st.write("Image not available")
                    else:
                        st.write("Image not available")
                    st.markdown(f'<div class="book-title">{row["Book-Title"][:30] + "..." if len(row["Book-Title"]) > 30 else row["Book-Title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="book-author">by {row["Book-Author"][:25] + "..." if len(row["Book-Author"]) > 25 else row["Book-Author"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="book-ratings">Ratings: {row["num_ratings"]}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    # Recommender Page
    elif option == "Get Recommendations":
        st.title("📖 Book Recommender Tool")
        st.markdown("Select a book title to get personalized recommendations.")

        # Book title input with autocomplete
        book_title = st.selectbox("Select or type a book title", options=[""] + list(book_titles), index=0)

        if st.button("Recommend"):
            if book_title:
                try:
                    message, recommendations = recommend_books(book_title, book_pivot, model_knn)
                    if recommendations:
                        st.subheader(message)
                        # Create a grid layout for recommendations
                        cols = st.columns(4, gap="medium")
                        for idx, rec in enumerate(recommendations):
                            col = cols[idx % 4]
                            with col:
                                with st.container():
                                    st.markdown(f'<div class="book-container">', unsafe_allow_html=True)
                                    if rec["image_url"] and rec["image_url"] != "No Image":
                                        try:
                                            st.image(rec["image_url"], width=120, caption=rec["title"][:25] + "..." if len(rec["title"]) > 25 else rec["title"])
                                        except Exception:
                                            st.write("Image not available")
                                    else:
                                        st.write("No image available")
                                    st.markdown(f'<div class="book-title">{rec["title"][:30] + "..." if len(rec["title"]) > 30 else rec["title"]}</div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="book-author">by {rec["author"][:25] + "..." if len(rec["author"]) > 25 else rec["author"]}</div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"Book '{book_title}' not found in the dataset. Please check the spelling.")
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
            else:
                st.warning("Please select a book title.")

if __name__ == "__main__":
    main()