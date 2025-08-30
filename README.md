# 📘 Book Recommendation System

A personalized **content-based** and **collaborative filtering** book recommender system using *K-Nearest Neighbors (KNN)*, enriched with an interactive **Streamlit** web interface.

---

## 📑 Table of Contents
- [📈 System Flow](#-system-flow)
- [🎯 Objectives](#-objectives)
- [🗂 Datasets Used](#-datasets-used)
- [⚙ Project Workflow](#-project-workflow)
- [🌐 Streamlit Web Interface](#-streamlit-web-interface)

---

## 📈 System Flow

![System Flow](https://github.com/Nourelden-Muhammed/NTI-BookRecommendation/blob/main/Book.Recommendation.Flow.png?raw=true)

---

## 🎯 Objectives

- Display the **Top 20 Most Rated Books**
- Recommend **similar books** based on a user-input title
- Evaluate results with **Precision@k** and **Recall@k** metrics

---

## 🗂 Datasets Used

| Dataset | Description                                      |
|---------|--------------------------------------------------|
| Books   | Metadata including title, author, publisher, year |
| Ratings | User ratings of books (0 to 10 scale)            |
| Users   | User demographic info (location, age)            |

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
- Built recommendation function: `recommend_books(title)`  

### 4. 🧪 Evaluation
- Used ranking-based metrics like:  
  - Precision@k  
  - Recall@k  

---

## 🌐 Streamlit Web Interface

An interactive web app built with **Streamlit** lets users:

- Search for a book  
- Get 5 similar books with similarity scores and cover images  
- View the **Top 20 Most Rated Books**  

<p align="center">
  <a href="https://nti-bookrecommendation.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Open%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open Streamlit App">
  </a>
</p>
