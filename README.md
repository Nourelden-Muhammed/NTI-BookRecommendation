# 📘 Book Recommendation System

A personalized content-based and collaborative filtering book recommender system using *K-Nearest Neighbors (KNN), enriched with an interactive **Streamlit* web interface.

---

## 🎯 Objectives

-  Display the *Top 20 Most Rated Books*
-  Recommend *similar books* based on a user-input title
-  Evaluate results with *Precision@k* and *Recall@k* metrics

---

## 🗂 Datasets Used

| Dataset     | Description                                        |
|-------------|----------------------------------------------------|
| Books     | Metadata including title, author, publisher, year  |
| Ratings   | User ratings of books (0 to 10 scale)              |
| Users     | User demographic info (location, age)              |

---

## ⚙ Project Workflow

### 1. 🔧 Data Preprocessing

- Handled missing values (e.g., author, publisher)
- Filtered invalid or extreme age entries (kept ages between 5 and 100)
- Normalized publication years
- Dropped irrelevant image links

### 2. 📊 Exploratory Data Analysis (EDA)

- Rating distribution
- Most rated and highest-rated books
- Most active users
- Age and location insights

### 3. 🤖 Book Recommendation Model (KNN)

- Created user-item interaction matrix
- Converted to sparse matrix for memory efficiency
- Applied *K-Nearest Neighbors* with *cosine similarity*
- Built recommendation function: recommend_books(title)

### 4. 🧪 Evaluation

- Used ranking-based metrics like:
  - Precision@k
  - Recall@k

---

## 🧠 Why KNN?

- ✔ Simple and intuitive
- ✔ No model training required
- ✔ Effective for *item-item similarity*
- ✔ Works well with sparse datasets

---

## 🌐 Streamlit Web Interface

An interactive web app built with *Streamlit* lets users:

- Search for a book
- Get 5 similar books with similarity scores and cover images
- View the *Top 20 Most Rated Books*
