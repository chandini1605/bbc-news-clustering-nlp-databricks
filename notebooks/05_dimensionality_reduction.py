# ============================================================
# STEP 5: DIMENSIONALITY REDUCTION - PCA + t-SNE
# BBC News Clustering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
import scipy.sparse as sp
import pickle, os

os.makedirs("outputs/cluster_plots", exist_ok=True)

df       = pd.read_csv("outputs/cleaned_news.csv")
X_tfidf  = sp.load_npz("outputs/X_tfidf.npz")

print("=" * 60)
print("STEP 5: DIMENSIONALITY REDUCTION")
print("TF-IDF (5000) → SVD (100) → t-SNE (2D)")
print("=" * 60)

print(f"\n📌 Original TF-IDF Shape: {X_tfidf.shape}")
print("   ⚠️  PCA doesn't work on sparse matrices directly")
print("   ✅ Using TruncatedSVD (LSA) for sparse TF-IDF — better for text!")

# ── 1. TruncatedSVD (LSA) ─────────────────────────────────────
# Better than PCA for sparse text matrices
print("\n📌 Step 5a: TruncatedSVD (100 components)...")
svd = TruncatedSVD(n_components=100, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

explained_var = svd.explained_variance_ratio_.sum() * 100
print(f"✅ SVD Shape        : {X_svd.shape}")
print(f"✅ Variance Retained: {explained_var:.2f}%")

# Normalize after SVD
normalizer = Normalizer(copy=False)
X_svd_norm = normalizer.fit_transform(X_svd)
print("✅ Normalized SVD features")

# ── 2. Explained Variance Plot ────────────────────────────────
cumvar = np.cumsum(svd.explained_variance_ratio_) * 100
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, 101), svd.explained_variance_ratio_ * 100,
             color='#2563eb', linewidth=2)
axes[0].set_title('Explained Variance Per Component', fontweight='bold')
axes[0].set_xlabel('Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, 101), cumvar, color='#dc2626', linewidth=2)
axes[1].axhline(y=80, color='green', linestyle='--', label='80% threshold')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('TruncatedSVD Explained Variance (Text Feature Reduction)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/cluster_plots/01_svd_variance.png', dpi=150)
plt.close()
print("\n✅ Saved: 01_svd_variance.png")

# ── 3. PCA on SVD output ──────────────────────────────────────
print("\n📌 Step 5b: PCA on SVD output (50 components)...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_svd_norm)
print(f"✅ PCA Shape: {X_pca.shape}")
print(f"✅ PCA Variance Retained: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# ── 4. t-SNE for 2D Visualization ────────────────────────────
print("\n📌 Step 5c: t-SNE for 2D visualization...")
print("   ⚠️  t-SNE is for VISUALIZATION only, not for clustering input!")
tsne = TSNE(n_components=2, random_state=42, perplexity=40,
            max_iter=1000, learning_rate=200)
X_tsne = tsne.fit_transform(X_pca)
print(f"✅ t-SNE Shape: {X_tsne.shape}")

# ── 5. t-SNE Plot Colored by TRUE Category ───────────────────
colors_map = {
    'business': '#2563eb', 'sport': '#16a34a',
    'politics': '#dc2626', 'tech': '#d97706',
    'entertainment': '#7c3aed'
}
color_list = [colors_map[c] for c in df['category']]

fig, ax = plt.subplots(figsize=(12, 8))
for cat, color in colors_map.items():
    mask = df['category'] == cat
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=color, label=cat.upper(), alpha=0.6, s=20)

ax.set_title('t-SNE Visualization of BBC News Articles\n(Colored by True Category)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.legend(markerscale=2, fontsize=12)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('outputs/cluster_plots/02_tsne_true_labels.png', dpi=150)
plt.close()
print("✅ Saved: 02_tsne_true_labels.png")

# ── 6. Save Reduced Features ──────────────────────────────────
np.save("outputs/X_svd.npy",  X_svd_norm)
np.save("outputs/X_pca.npy",  X_pca)
np.save("outputs/X_tsne.npy", X_tsne)
pickle.dump(svd,        open('models/svd.pkl', 'wb'))
pickle.dump(pca,        open('models/pca.pkl', 'wb'))
pickle.dump(normalizer, open('models/normalizer.pkl', 'wb'))

print("\n✅ Saved: X_svd.npy, X_pca.npy, X_tsne.npy")
print("✅ Step 5 Complete!")
