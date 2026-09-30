# ============================================================
# STEP 10: CLUSTER ANALYSIS + FINAL VISUALIZATION
# Top Words Per Cluster + WordClouds + Summary
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import scipy.sparse as sp
import pickle, os, warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs/wordclouds", exist_ok=True)
os.makedirs("outputs/cluster_plots", exist_ok=True)

df        = pd.read_csv("outputs/clustered_news.csv")
km_labels = np.load("outputs/km_labels.npy")
X_tfidf   = sp.load_npz("outputs/X_tfidf.npz")
tfidf     = pickle.load(open('models/tfidf_vectorizer.pkl','rb'))
X_tsne    = np.load("outputs/X_tsne.npy")

df['cluster'] = km_labels
feature_names = tfidf.get_feature_names_out()
colors5 = ['#2563eb','#16a34a','#dc2626','#d97706','#7c3aed']

print("=" * 60)
print("STEP 10: CLUSTER ANALYSIS & VISUALIZATION")
print("=" * 60)

# ── 1. Cluster Composition ────────────────────────────────────
print("\n📌 Cluster Composition:")
for c in range(5):
    mask = km_labels == c
    cat_dist = df[mask]['category'].value_counts()
    dom = cat_dist.index[0]
    print(f"\n   Cluster {c} → Dominant: {dom.upper()} ({cat_dist.iloc[0]}/{mask.sum()})")
    print(f"   {cat_dist.to_dict()}")

# ── 2. WordCloud Per Cluster ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for c in range(5):
    mask = km_labels == c
    text = ' '.join(df[mask]['cleaned_text'].fillna('').astype(str))
    wc = WordCloud(width=600, height=400, background_color='white',
                   colormap='tab10', max_words=80).generate(text)
    axes[c].imshow(wc, interpolation='bilinear')
    dom = df[mask]['category'].value_counts().index[0]
    axes[c].set_title(f'Cluster {c} — Dominant: {dom.upper()}',
                      fontweight='bold', fontsize=12)
    axes[c].axis('off')

axes[5].axis('off')
plt.suptitle('Word Clouds Per Cluster', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/wordclouds/cluster_wordclouds.png', dpi=150)
plt.close()
print("\n✅ Saved: cluster_wordclouds.png")

# ── 3. Top TF-IDF Terms Per Cluster ──────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for c in range(5):
    mask     = km_labels == c
    X_c      = X_tfidf[mask].toarray().mean(axis=0)
    top_idx  = X_c.argsort()[-15:][::-1]
    top_terms= [feature_names[i] for i in top_idx]
    top_scores=[X_c[i] for i in top_idx]
    dom = df[mask]['category'].value_counts().index[0]

    axes[c].barh(top_terms[::-1], top_scores[::-1],
                 color=colors5[c], edgecolor='black', alpha=0.85)
    axes[c].set_title(f'Cluster {c}: {dom.upper()}\nTop TF-IDF Terms',
                      fontweight='bold')
    axes[c].set_xlabel('Mean TF-IDF Score')

axes[5].axis('off')
plt.suptitle('Top TF-IDF Terms Per Cluster', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/cluster_plots/07_top_terms_per_cluster.png', dpi=150)
plt.close()
print("✅ Saved: 07_top_terms_per_cluster.png")

# ── 4. Final t-SNE with Cluster Labels ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# True labels
cat_colors = {'business':'#2563eb','sport':'#16a34a',
              'politics':'#dc2626','tech':'#d97706','entertainment':'#7c3aed'}
for cat, color in cat_colors.items():
    mask = df['category'] == cat
    axes[0].scatter(X_tsne[mask,0], X_tsne[mask,1],
                    c=color, label=cat.upper(), alpha=0.6, s=15)
axes[0].set_title('True Category Labels', fontweight='bold', fontsize=13)
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')
axes[0].legend(markerscale=2)
axes[0].grid(True, alpha=0.2)

# Cluster labels
for c in range(5):
    mask = km_labels == c
    dom  = df[mask]['category'].value_counts().index[0]
    axes[1].scatter(X_tsne[mask,0], X_tsne[mask,1],
                    c=colors5[c], label=f'C{c}:{dom.upper()}', alpha=0.6, s=15)
axes[1].set_title('K-Means Cluster Labels', fontweight='bold', fontsize=13)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].legend(markerscale=2)
axes[1].grid(True, alpha=0.2)

plt.suptitle('t-SNE: True Labels vs Discovered Clusters',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/cluster_plots/08_tsne_true_vs_cluster.png', dpi=150)
plt.close()
print("✅ Saved: 08_tsne_true_vs_cluster.png")

# ── 5. Cluster Size Distribution ─────────────────────────────
cluster_sizes = pd.Series(km_labels).value_counts().sort_index()
cluster_names = [df[km_labels==c]['category'].value_counts().index[0].upper()
                 for c in range(5)]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar([f'C{i}:{n}' for i,n in enumerate(cluster_names)],
              cluster_sizes, color=colors5, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, cluster_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', fontweight='bold')
ax.set_title('Cluster Size Distribution', fontweight='bold', fontsize=13)
ax.set_ylabel('Number of Articles')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('outputs/cluster_plots/09_cluster_sizes.png', dpi=150)
plt.close()
print("✅ Saved: 09_cluster_sizes.png")

df.to_csv("outputs/final_clustered_news.csv", index=False)
print("\n✅ Final data saved to outputs/final_clustered_news.csv")
print("✅ Step 10 Complete!")
