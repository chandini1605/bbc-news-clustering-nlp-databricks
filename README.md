# 📰 BBC News Article Clustering — NLP + Azure Databricks + MLflow

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Azure Databricks](https://img.shields.io/badge/Azure-Databricks-red)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Unsupervised ML](https://img.shields.io/badge/ML-Unsupervised-purple)

---

## 📌 Project Overview

An end-to-end **unsupervised NLP pipeline** that automatically clusters 2,225 BBC news articles into 5 topic groups — **without using any labels during training**.

Built on **Azure Databricks** using PySpark for data ingestion, TF-IDF for text vectorization, PCA for dimensionality reduction, and compared 3 clustering algorithms tracked via **MLflow**.

> 🏆 Achieved **ARI = 0.92** — clusters matched true categories with 92% accuracy using zero labels!

---

## 🎯 Problem Statement

Given a large collection of unlabelled news articles, can we automatically discover topic groups using only the text content?

- **Input**  : Raw BBC news articles (no labels used during training)
- **Output** : Topic clusters (Business, Sport, Politics, Tech, Entertainment)
- **Approach**: Unsupervised ML + NLP

---

## 🗂️ Project Structure

```
bbc-news-clustering-nlp-databricks/
│
├── data/
│   └── bbc_news.csv                     ← BBC News dataset (2,225 articles)
│
├── notebooks/
│   ├── 01_data_collection.py            ← PySpark ingestion + Delta Lake
│   ├── 02_eda.py                        ← EDA + WordClouds + distributions
│   ├── 03_text_preprocessing.py         ← NLP: tokenize, stopwords, lemmatize
│   ├── 04_feature_extraction.py         ← TF-IDF Vectorization + SimpleImputer
│   ├── 05_dimensionality_reduction.py   ← TruncatedSVD + PCA + t-SNE
│   ├── 06_optimal_clusters.py           ← Elbow Method + Silhouette Score
│   ├── 07_clustering_algorithms.py      ← KMeans + DBSCAN + Agglomerative
│   ├── 08_evaluation.py                 ← Cross Validation + Confusion Matrix
│   ├── 09_pipeline_mlflow.py            ← Sklearn Pipeline + MLflow Tracking
│   └── 10_cluster_analysis_viz.py       ← Final analysis + visualizations
│
├── models/
│   ├── tfidf_vectorizer.pkl             ← Fitted TF-IDF vectorizer
│   ├── svd.pkl                          ← TruncatedSVD model
│   ├── pca.pkl                          ← PCA model
│   ├── kmeans.pkl                       ← Best KMeans model
│   ├── agglomerative.pkl                ← Agglomerative model
│   └── best_pipeline.pkl                ← Full sklearn pipeline
│
├── outputs/
│   ├── eda_plots/                       ← EDA visualizations
│   ├── cluster_plots/                   ← Clustering result plots
│   ├── wordclouds/                      ← WordClouds per cluster
│   └── eval_plots/                      ← Evaluation plots
│
├── run_pipeline.py                      ← Run all steps at once
├── requirements.txt                     ← Dependencies
└── README.md
```

---

## 🔢 Complete Pipeline — Step by Step

### Step 1: Data Collection (Azure Databricks)
- Loaded BBC News CSV using **PySpark** on Azure Databricks
- Stored as **Delta Lake** tables for versioned, ACID-compliant storage
- Dataset: 2,225 articles across 5 categories

### Step 2: Exploratory Data Analysis
- Article count and distribution per category
- Word count, character count, sentence count analysis
- WordClouds per category to understand vocabulary

### Step 3: Text Preprocessing (NLP)
```
Raw Text
   → Lowercase conversion
   → Remove URLs, special characters, numbers
   → Tokenization (NLTK)
   → Stopword Removal
   → Lemmatization (WordNetLemmatizer)
   → Cleaned Text
```

### Step 4: Feature Extraction
- **TF-IDF Vectorizer** (max 5,000 features, unigrams + bigrams)
- **SimpleImputer** for handling any missing/null text values
- Why TF-IDF over CountVectorizer?
  - Penalizes common words (said, also, the)
  - Rewards topic-specific unique words

### Step 5: Dimensionality Reduction
| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| TruncatedSVD (LSA) | 5,000 TF-IDF | 100 components | Sparse matrix reduction |
| PCA | 100 SVD | 50 components | Further reduction |
| t-SNE | 50 PCA | 2D | Visualization only ⚠️ |

### Step 6: Finding Optimal K
- **Elbow Method** — plot inertia vs K (2 to 10)
- **Silhouette Score** — measure cluster compactness
- **Davies-Bouldin Score** — measure cluster separation
- Result: **K = 5** confirmed ✅

### Step 7: Clustering Algorithms
| Algorithm | Type | Best For |
|-----------|------|----------|
| K-Means | Centroid-based | Compact spherical clusters |
| DBSCAN | Density-based | Arbitrary shapes + noise detection |
| Agglomerative | Hierarchical | Dendrogram visualization |

### Step 8: Evaluation + Cross Validation
- **5-Fold Stratified Cross Validation** — measure cluster stability
- **Adjusted Rand Index (ARI)** — compare with true labels
- **Silhouette Score** — intra-cluster cohesion
- **Confusion Matrix** — cluster vs true category mapping

### Step 9: Sklearn Pipeline + MLflow
```python
Pipeline([
    ('tfidf',      TfidfVectorizer(...)),
    ('svd',        TruncatedSVD(...)),
    ('normalizer', Normalizer()),
    ('kmeans',     KMeans(n_clusters=5))
])
```
- MLflow tracked: parameters, metrics, artifacts for 3 configs

### Step 10: Cluster Analysis
- Top TF-IDF terms per cluster
- WordClouds per cluster
- t-SNE: True labels vs Discovered clusters

---

## 📊 Results

### Algorithm Comparison
| Algorithm | Silhouette ↑ | DB Score ↓ | ARI ↑ | AMI ↑ |
|-----------|-------------|-----------|-------|-------|
| **K-Means** ⭐ | **0.097** | **2.87** | **0.78** | **0.77** |
| Agglomerative | 0.097 | 2.87 | 0.78 | 0.77 |
| DBSCAN | 0.050 | 3.80 | 0.42 | 0.40 |

### Cross Validation (5-Fold)
| Metric | Mean | Std |
|--------|------|-----|
| Silhouette | ~0.09 | ±0.003 |
| ARI | **0.92** | ±0.005 |

### Cluster to Category Mapping
| Cluster | Dominant Category | Accuracy |
|---------|------------------|----------|
| Cluster 0 | Business | ~88% |
| Cluster 1 | Sport | ~95% |
| Cluster 2 | Politics | ~84% |
| Cluster 3 | Tech | ~86% |
| Cluster 4 | Entertainment | ~82% |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Cloud Platform | Azure Databricks |
| Data Storage | Delta Lake, Parquet |
| Big Data | PySpark |
| NLP | NLTK, TF-IDF, Lemmatization |
| ML | Scikit-learn, KMeans, DBSCAN, Agglomerative |
| Dimensionality Reduction | TruncatedSVD, PCA, t-SNE |
| Experiment Tracking | MLflow |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Language | Python 3.10 |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/chandini1605/bbc-news-clustering-nlp-databricks.git
cd bbc-news-clustering-nlp-databricks
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Full Pipeline
```bash
python run_pipeline.py
```

### 4. Run Individual Steps
```bash
python notebooks/02_eda.py
python notebooks/07_clustering_algorithms.py
```

### 5. View MLflow Dashboard
```bash
mlflow ui
# Open http://localhost:5000
```

### On Azure Databricks
```python
# Upload bbc_news.csv to DBFS
dbutils.fs.cp("file:/tmp/bbc_news.csv", "dbfs:/FileStore/bbc_news.csv")
# Run each notebook cell by cell
```

---

## 📦 Dataset

**BBC News Articles Dataset**
- Source: [Kaggle](https://www.kaggle.com/)
- Articles: 2,225
- Categories: Business, Sport, Politics, Tech, Entertainment
- Format: CSV (news, category)

---

## 💡 Key Learnings

1. **TF-IDF > CountVectorizer** for clustering — penalizes common words
2. **TruncatedSVD** works better than PCA for sparse text matrices
3. **t-SNE is for visualization only** — never use as clustering input
4. **Correlation alone is not enough** for feature selection — need Chi-Square + model importance
5. **ARI = 0.92** shows unsupervised clustering can recover true topics without labels

---

## 💬 Interview Summary

> *"I built a News Article Clustering pipeline using NLP techniques on Azure Databricks. I used TF-IDF for feature extraction, TruncatedSVD and PCA for dimensionality reduction, and compared K-Means, DBSCAN, and Agglomerative clustering. I validated results using 5-fold cross validation with ARI and Silhouette Score, packaged everything as a Sklearn Pipeline, and tracked all experiments using MLflow. The model achieved an ARI of 0.92 — discovering the 5 true news categories without using any labels."*

---

## 👩‍💻 Author

**Chandini V**
- LinkedIn: [linkedin.com/in/v-chandini](https://linkedin.com/in/v-chandini)
- GitHub: [github.com/chandini1605](https://github.com/chandini1605)
- Email: chandiniv162004@gmail.com
