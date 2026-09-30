# ============================================================
# STEP 8: CLUSTER EVALUATION + CROSS VALIDATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              adjusted_rand_score, confusion_matrix,
                              classification_report)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import os, warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs/eval_plots", exist_ok=True)

df     = pd.read_csv("outputs/cleaned_news.csv")
X_pca  = np.load("outputs/X_pca.npy")
X_tsne = np.load("outputs/X_tsne.npy")
km_labels = np.load("outputs/km_labels.npy")

label_map = {'business':0,'sport':1,'politics':2,'tech':3,'entertainment':4}
y_true    = df['category'].map(label_map).values
cats      = list(label_map.keys())

print("=" * 60)
print("STEP 8: CLUSTER EVALUATION + CROSS VALIDATION")
print("=" * 60)

# ── 1. Map Cluster Labels to Category Names ───────────────────
print("\n📌 Mapping Clusters to Categories:")
cluster_to_cat = {}
for cluster in range(5):
    mask  = km_labels == cluster
    true_labels = df['category'][mask]
    dominant    = true_labels.value_counts().idxmax()
    cluster_to_cat[cluster] = dominant
    print(f"   Cluster {cluster} → {dominant.upper()} ({true_labels.value_counts().iloc[0]} / {mask.sum()} docs)")

df['predicted_category'] = [cluster_to_cat[l] for l in km_labels]

# ── 2. Confusion Matrix ───────────────────────────────────────
cm = confusion_matrix(df['category'], df['predicted_category'], labels=cats)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[c.upper() for c in cats],
            yticklabels=[c.upper() for c in cats])
ax.set_title('Confusion Matrix: True vs Predicted Category\n(K-Means Clustering)',
             fontweight='bold', fontsize=13)
ax.set_ylabel('True Category')
ax.set_xlabel('Predicted Category')
plt.tight_layout()
plt.savefig('outputs/eval_plots/01_confusion_matrix.png', dpi=150)
plt.close()
print("\n✅ Saved: 01_confusion_matrix.png")

# ── 3. Classification Report ──────────────────────────────────
print("\n📌 Classification Report (Cluster vs True Labels):")
print(classification_report(df['category'], df['predicted_category']))

# ── 4. Cross Validation on Clustering ────────────────────────
print("\n📌 Cross Validation (5-Fold Stratified):")
print("   ⚠️  For clustering, CV measures stability of silhouette scores")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sil_scores, ari_scores = [], []

df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)

for fold, (train_idx, val_idx) in enumerate(skf.split(df['cleaned_text'], df['category'])):
    X_fold = X_pca[train_idx]
    y_fold = y_true[train_idx]
    X_val  = X_pca[val_idx]
    y_val  = y_true[val_idx]

    km_fold = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels_fold = km_fold.fit_predict(X_fold)

    sil = silhouette_score(X_fold, labels_fold)
    ari = adjusted_rand_score(y_fold, labels_fold)
    sil_scores.append(sil)
    ari_scores.append(ari)
    print(f"   Fold {fold+1}: Silhouette={sil:.4f}  ARI={ari:.4f}")

print(f"\n   Mean Silhouette : {np.mean(sil_scores):.4f} ± {np.std(sil_scores):.4f}")
print(f"   Mean ARI        : {np.mean(ari_scores):.4f} ± {np.std(ari_scores):.4f}")

# ── 5. CV Score Plot ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1,6), sil_scores, color='#2563eb', edgecolor='black', alpha=0.8)
axes[0].axhline(np.mean(sil_scores), color='red', linestyle='--',
                label=f'Mean={np.mean(sil_scores):.3f}')
axes[0].set_title('5-Fold CV: Silhouette Scores', fontweight='bold')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('Silhouette Score')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(1,6), ari_scores, color='#16a34a', edgecolor='black', alpha=0.8)
axes[1].axhline(np.mean(ari_scores), color='red', linestyle='--',
                label=f'Mean={np.mean(ari_scores):.3f}')
axes[1].set_title('5-Fold CV: Adjusted Rand Index', fontweight='bold')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('ARI Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Cross Validation Stability of K-Means Clustering',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eval_plots/02_cross_validation.png', dpi=150)
plt.close()
print("\n✅ Saved: 02_cross_validation.png")

# ── 6. Silhouette Plot Per Sample ────────────────────────────
from sklearn.metrics import silhouette_samples
sil_vals = silhouette_samples(X_pca, km_labels)
colors5  = ['#2563eb','#16a34a','#dc2626','#d97706','#7c3aed']

fig, ax = plt.subplots(figsize=(12, 7))
y_lower = 10
for i in range(5):
    ith_sil = np.sort(sil_vals[km_labels == i])
    size    = ith_sil.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
                     alpha=0.7, color=colors5[i], label=f'Cluster {i}')
    ax.text(-0.05, y_lower + 0.5*size, str(i))
    y_lower = y_upper + 10

ax.axvline(x=silhouette_score(X_pca, km_labels), color='red',
           linestyle='--', label=f'Mean={silhouette_score(X_pca,km_labels):.3f}')
ax.set_title('Silhouette Plot — K-Means (K=5)', fontweight='bold')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/eval_plots/03_silhouette_plot.png', dpi=150)
plt.close()
print("✅ Saved: 03_silhouette_plot.png")

df.to_csv("outputs/clustered_news.csv", index=False)
print("\n✅ Step 8 Complete!")
