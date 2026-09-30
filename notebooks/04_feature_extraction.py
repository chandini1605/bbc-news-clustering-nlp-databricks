# ============================================================
# STEP 4: FEATURE EXTRACTION - TF-IDF VECTORIZATION
# BBC News Clustering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle, os

os.makedirs("outputs/eda_plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("outputs/cleaned_news.csv")
df['cleaned_text'] = df['cleaned_text'].fillna('')

print("=" * 60)
print("STEP 4: FEATURE EXTRACTION")
print("TF-IDF Vectorization")
print("=" * 60)

# ── 1. Why TF-IDF over CountVectorizer ───────────────────────
print("\n📌 TF-IDF vs CountVectorizer:")
print("   CountVectorizer : Raw word frequency counts")
print("   TF-IDF          : Penalizes common words, rewards unique words")
print("   ✅ TF-IDF is better for clustering because:")
print("      - Common words like 'said','also' get low weight")
print("      - Topic-specific words get high weight")

# ── 2. SimpleImputer for any NaN texts ───────────────────────
# Even though text — convert NaN to empty string safely
print("\n📌 Applying SimpleImputer for missing text values...")
df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)
print(f"   Null texts remaining: {df['cleaned_text'].isnull().sum()}")
print("✅ SimpleImputer equivalent applied for text")

# ── 3. TF-IDF Vectorization ───────────────────────────────────
print("\n📌 Applying TF-IDF Vectorization...")
tfidf = TfidfVectorizer(
    max_features=5000,      # Top 5000 words
    min_df=2,               # Word must appear in at least 2 docs
    max_df=0.95,            # Ignore words in >95% of docs
    ngram_range=(1, 2),     # Unigrams + Bigrams
    sublinear_tf=True       # Apply log normalization
)

X_tfidf = tfidf.fit_transform(df['cleaned_text'])
print(f"✅ TF-IDF Matrix Shape: {X_tfidf.shape}")
print(f"   Documents  : {X_tfidf.shape[0]:,}")
print(f"   Features   : {X_tfidf.shape[1]:,}")
print(f"   Sparsity   : {(1 - X_tfidf.nnz / (X_tfidf.shape[0]*X_tfidf.shape[1]))*100:.2f}%")

# ── 4. Top TF-IDF Terms Per Category ─────────────────────────
print("\n📌 Top TF-IDF Terms Per Category:")
feature_names = tfidf.get_feature_names_out()

for cat in df['category'].unique():
    cat_docs = df[df['category'] == cat].index
    cat_tfidf = X_tfidf[cat_docs].toarray().mean(axis=0)
    top_idx = cat_tfidf.argsort()[-10:][::-1]
    top_terms = [feature_names[i] for i in top_idx]
    print(f"\n   {cat.upper()}: {top_terms}")

# ── 5. Visualize Top Terms Per Category ──────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
colors = ['#2563eb','#16a34a','#dc2626','#d97706','#7c3aed']

for i, cat in enumerate(df['category'].unique()):
    cat_docs = df[df['category'] == cat].index
    cat_tfidf = X_tfidf[cat_docs].toarray().mean(axis=0)
    top_idx   = cat_tfidf.argsort()[-15:][::-1]
    top_terms = [feature_names[j] for j in top_idx]
    top_scores= [cat_tfidf[j] for j in top_idx]

    axes[i].barh(top_terms[::-1], top_scores[::-1], color=colors[i], edgecolor='black')
    axes[i].set_title(f'Top TF-IDF Terms: {cat.upper()}', fontweight='bold')
    axes[i].set_xlabel('TF-IDF Score')

axes[5].axis('off')
plt.suptitle('Top TF-IDF Terms Per News Category', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda_plots/06_tfidf_terms_per_category.png', dpi=150)
plt.close()
print("\n✅ Saved: 06_tfidf_terms_per_category.png")

# ── 6. Save TF-IDF Matrix & Vectorizer ───────────────────────
import scipy.sparse as sp
sp.save_npz('outputs/X_tfidf.npz', X_tfidf)
pickle.dump(tfidf, open('models/tfidf_vectorizer.pkl', 'wb'))
df.to_csv("outputs/cleaned_news.csv", index=False)

print("\n✅ TF-IDF matrix saved to outputs/X_tfidf.npz")
print("✅ Vectorizer saved to models/tfidf_vectorizer.pkl")
print("\n✅ Step 4 Complete!")
